// Copyright 2026 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "draco/mesh/spatial_mesh_stream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "draco/compression/config/compression_shared.h"
#include "draco/compression/decode.h"
#include "draco/compression/expert_encode.h"

namespace draco {
namespace {

constexpr char kMagic[4] = {'D', 'M', 'S', '1'};
constexpr uint16_t kVersion = 1;
constexpr uint32_t kInvalidChunk = std::numeric_limits<uint32_t>::max();

Status StreamError(const std::string &message) {
  return Status(Status::DRACO_ERROR, message);
}

bool IsFinite(float value) { return std::isfinite(value); }

bool IsFinite(const Vector3f &value) {
  return IsFinite(value[0]) && IsFinite(value[1]) && IsFinite(value[2]);
}

class ContainerWriter {
 public:
  void WriteBytes(const void *data, size_t size) {
    const uint8_t *bytes = static_cast<const uint8_t *>(data);
    data_.insert(data_.end(), bytes, bytes + size);
  }

  void WriteU8(uint8_t value) { data_.push_back(value); }

  void WriteU16(uint16_t value) { WriteBytes(&value, sizeof(value)); }

  void WriteU32(uint32_t value) { WriteBytes(&value, sizeof(value)); }

  void WriteU64(uint64_t value) { WriteBytes(&value, sizeof(value)); }

  void WriteI64(int64_t value) { WriteBytes(&value, sizeof(value)); }

  void WriteF32(float value) { WriteBytes(&value, sizeof(value)); }

  void WriteString(const std::string &value) {
    const uint16_t size = static_cast<uint16_t>(
        std::min<size_t>(value.size(), std::numeric_limits<uint16_t>::max()));
    WriteU16(size);
    WriteBytes(value.data(), size);
  }

  const std::vector<uint8_t> &data() const { return data_; }

 private:
  std::vector<uint8_t> data_;
};

class ContainerReader {
 public:
  ContainerReader(const uint8_t *data, size_t size)
      : data_(data), size_(size) {}

  bool ReadBytes(void *out, size_t size) {
    if (size > remaining()) {
      return false;
    }
    std::memcpy(out, data_ + pos_, size);
    pos_ += size;
    return true;
  }

  bool ReadU8(uint8_t *value) { return ReadBytes(value, sizeof(*value)); }

  bool ReadU16(uint16_t *value) { return ReadBytes(value, sizeof(*value)); }

  bool ReadU32(uint32_t *value) { return ReadBytes(value, sizeof(*value)); }

  bool ReadU64(uint64_t *value) { return ReadBytes(value, sizeof(*value)); }

  bool ReadI64(int64_t *value) { return ReadBytes(value, sizeof(*value)); }

  bool ReadF32(float *value) { return ReadBytes(value, sizeof(*value)); }

  bool ReadString(std::string *value) {
    uint16_t size;
    if (!ReadU16(&size) || size > remaining()) {
      return false;
    }
    value->assign(reinterpret_cast<const char *>(data_ + pos_), size);
    pos_ += size;
    return true;
  }

  size_t remaining() const { return size_ - pos_; }
  size_t position() const { return pos_; }

 private:
  const uint8_t *data_;
  size_t size_;
  size_t pos_ = 0;
};

struct AttributeEncoding {
  int quantization_bits = 0;
  bool explicit_quantization = false;
  std::vector<float> quantization_origin;
  float quantization_range = 0.f;
};

int GetQuantizationBits(const SpatialMeshStreamOptions &options,
                        GeometryAttribute::Type type) {
  switch (type) {
    case GeometryAttribute::POSITION:
      return options.position_quantization_bits;
    case GeometryAttribute::TEX_COORD:
      return options.tex_coord_quantization_bits;
    case GeometryAttribute::NORMAL:
      return options.normal_quantization_bits;
    case GeometryAttribute::COLOR:
      return options.color_quantization_bits;
    case GeometryAttribute::GENERIC:
      return options.generic_quantization_bits;
    default:
      return 0;
  }
}

Status BuildAttributeEncodings(
    const Mesh &mesh, const SpatialMeshStreamOptions &options,
    std::vector<AttributeEncoding> *encodings) {
  encodings->resize(mesh.num_attributes());
  for (int ai = 0; ai < mesh.num_attributes(); ++ai) {
    const PointAttribute *const attribute = mesh.attribute(ai);
    AttributeEncoding &encoding = (*encodings)[ai];
    if (attribute->data_type() != DT_FLOAT32) {
      continue;
    }
    encoding.quantization_bits =
        GetQuantizationBits(options, attribute->attribute_type());
    if (encoding.quantization_bits <= 0) {
      continue;
    }
    if (encoding.quantization_bits > 30) {
      return StreamError("Quantization bits must be in the range [0, 30].");
    }

    // Normals use an absolute octahedral transform, so identical normal
    // values already produce identical decoded bytes across chunks.
    if (attribute->attribute_type() == GeometryAttribute::NORMAL &&
        attribute->num_components() == 3) {
      continue;
    }

    const int num_components = attribute->num_components();
    const int64_t entry_size =
        DataTypeLength(attribute->data_type()) * num_components;
    std::vector<float> min_values(num_components,
                                  std::numeric_limits<float>::infinity());
    std::vector<float> max_values(
        num_components, -std::numeric_limits<float>::infinity());
    std::vector<uint8_t> raw(attribute->byte_stride());
    for (AttributeValueIndex avi(0); avi < attribute->size(); ++avi) {
      attribute->GetValue(avi, raw.data());
      std::vector<float> values(num_components);
      std::memcpy(values.data(), raw.data(), entry_size);
      for (int c = 0; c < num_components; ++c) {
        if (!IsFinite(values[c])) {
          return StreamError(
              "Input attributes must contain only finite float values.");
        }
        min_values[c] = std::min(min_values[c], values[c]);
        max_values[c] = std::max(max_values[c], values[c]);
      }
    }
    float range = 0.f;
    for (int c = 0; c < num_components; ++c) {
      range = std::max(range, max_values[c] - min_values[c]);
    }
    if (range == 0.f) {
      range = 1.f;
    }
    if (!IsFinite(range)) {
      return StreamError("Attribute quantization range is not finite.");
    }
    encoding.explicit_quantization = true;
    encoding.quantization_origin = std::move(min_values);
    encoding.quantization_range = range;
  }
  return OkStatus();
}

Status CopyAttributeValue(const PointAttribute &source,
                          PointIndex source_point, PointAttribute *target,
                          AttributeValueIndex target_value,
                          std::vector<uint8_t> *scratch) {
  scratch->resize(std::max<size_t>(scratch->size(),
                                  static_cast<size_t>(source.byte_stride())));
  source.GetValue(source.mapped_index(source_point), scratch->data());
  if (target->byte_stride() > source.byte_stride()) {
    return StreamError("Attribute byte strides are incompatible.");
  }
  target->SetAttributeValue(target_value, scratch->data());
  return OkStatus();
}

Status BuildChunkMesh(const Mesh &mesh,
                      const std::vector<uint32_t> &point_ids,
                      const std::vector<uint32_t> &face_ids, Mesh *chunk) {
  std::unordered_map<uint32_t, uint32_t> source_to_local;
  source_to_local.reserve(point_ids.size() * 2);
  for (uint32_t i = 0; i < point_ids.size(); ++i) {
    source_to_local[point_ids[i]] = i;
  }

  chunk->set_num_points(static_cast<uint32_t>(point_ids.size()));
  std::vector<uint8_t> scratch;
  for (int ai = 0; ai < mesh.num_attributes(); ++ai) {
    const PointAttribute *const source = mesh.attribute(ai);
    std::unique_ptr<PointAttribute> target(new PointAttribute());
    target->Init(source->attribute_type(), source->num_components(),
                 source->data_type(), source->normalized(), point_ids.size());
    target->set_unique_id(source->unique_id());
#ifdef DRACO_TRANSCODER_SUPPORTED
    target->set_name(source->name());
#endif
    for (uint32_t i = 0; i < point_ids.size(); ++i) {
      const Status status =
          CopyAttributeValue(*source, PointIndex(point_ids[i]), target.get(),
                             AttributeValueIndex(i), &scratch);
      if (!status.ok()) {
        return status;
      }
    }
    const int target_id = chunk->AddAttribute(std::move(target));
    if (target_id != ai) {
      return StreamError("Failed to preserve the attribute order.");
    }
    chunk->SetAttributeElementType(ai, mesh.GetAttributeElementType(ai));
  }

  chunk->SetNumFaces(face_ids.size());
  for (uint32_t i = 0; i < face_ids.size(); ++i) {
    const Mesh::Face &source_face = mesh.face(FaceIndex(face_ids[i]));
    Mesh::Face target_face;
    for (int c = 0; c < 3; ++c) {
      const auto it = source_to_local.find(source_face[c].value());
      if (it == source_to_local.end()) {
        return StreamError("A chunk face references a missing point.");
      }
      target_face[c] = PointIndex(it->second);
    }
    chunk->SetFace(FaceIndex(i), target_face);
  }
  return OkStatus();
}

Status EncodeChunk(const Mesh &mesh,
                   const std::vector<uint32_t> &point_ids,
                   const std::vector<uint32_t> &face_ids,
                   const std::vector<AttributeEncoding> &encodings,
                   const SpatialMeshStreamOptions &options,
                   std::vector<uint8_t> *payload) {
  Mesh chunk;
  const Status build_status =
      BuildChunkMesh(mesh, point_ids, face_ids, &chunk);
  if (!build_status.ok()) {
    return build_status;
  }

  ExpertEncoder encoder(chunk);
  encoder.SetSpeedOptions(options.speed, options.speed);
  encoder.SetEncodingMethod(MESH_SEQUENTIAL_ENCODING);
  for (int ai = 0; ai < chunk.num_attributes(); ++ai) {
    const AttributeEncoding &encoding = encodings[ai];
    if (encoding.quantization_bits <= 0 ||
        chunk.attribute(ai)->data_type() != DT_FLOAT32) {
      continue;
    }
    if (encoding.explicit_quantization) {
      encoder.SetAttributeExplicitQuantization(
          ai, encoding.quantization_bits,
          chunk.attribute(ai)->num_components(),
          encoding.quantization_origin.data(), encoding.quantization_range);
    } else {
      encoder.SetAttributeQuantization(ai, encoding.quantization_bits);
    }
  }

  EncoderBuffer buffer;
  const Status status = encoder.EncodeToBuffer(&buffer);
  if (!status.ok()) {
    return status;
  }
  payload->assign(reinterpret_cast<const uint8_t *>(buffer.data()),
                  reinterpret_cast<const uint8_t *>(buffer.data()) +
                      buffer.size());
  return OkStatus();
}

double SquaredDistanceToBounds(const Vector3f &point,
                               const BoundingBox &bounds) {
  double distance = 0.0;
  for (int c = 0; c < 3; ++c) {
    double delta = 0.0;
    if (point[c] < bounds.GetMinPoint()[c]) {
      delta = static_cast<double>(bounds.GetMinPoint()[c]) - point[c];
    } else if (point[c] > bounds.GetMaxPoint()[c]) {
      delta = static_cast<double>(point[c]) - bounds.GetMaxPoint()[c];
    }
    distance += delta * delta;
  }
  return distance;
}

}  // namespace

Status SpatialMeshStreamEncoder::Encode(
    const Mesh &mesh, const SpatialMeshStreamOptions &options,
    EncoderBuffer *out_buffer, SpatialMeshConsistency *consistency) {
  if (out_buffer == nullptr) {
    return StreamError("Output buffer is null.");
  }
  if (!IsFinite(options.chunk_size) || options.chunk_size <= 0.f) {
    return StreamError("chunk_size must be finite and greater than zero.");
  }
  if (options.speed < 0 || options.speed > 10) {
    return StreamError("Encoding speed must be in the range [0, 10].");
  }
  if (mesh.num_faces() == 0 || mesh.num_points() == 0) {
    return StreamError("Spatial mesh streaming requires a non-empty mesh.");
  }
  const int position_id =
      mesh.GetNamedAttributeId(GeometryAttribute::POSITION);
  if (position_id < 0) {
    return StreamError("The mesh has no position attribute.");
  }
  const PointAttribute *const position = mesh.attribute(position_id);
  if (position->data_type() != DT_FLOAT32 ||
      position->num_components() != 3) {
    return StreamError("Positions must be float32 with three components.");
  }

  std::vector<AttributeEncoding> encodings;
  const Status encoding_status =
      BuildAttributeEncodings(mesh, options, &encodings);
  if (!encoding_status.ok()) {
    return encoding_status;
  }

  BoundingBox mesh_bounds;
  std::vector<uint8_t> raw_position(position->byte_stride());
  for (AttributeValueIndex avi(0); avi < position->size(); ++avi) {
    position->GetValue(avi, raw_position.data());
    Vector3f value;
    std::memcpy(&value[0], raw_position.data(), sizeof(float) * 3);
    if (!IsFinite(value)) {
      return StreamError("Positions must contain only finite values.");
    }
    mesh_bounds.Update(value);
  }
  if (!mesh_bounds.IsValid()) {
    return StreamError("Failed to compute mesh bounds.");
  }

  struct ChunkBuild {
    std::array<int64_t, 3> grid_coordinate = {{0, 0, 0}};
    BoundingBox bounds;
    std::vector<uint32_t> face_ids;
  };
  std::map<std::array<int64_t, 3>, ChunkBuild> chunks_by_grid;
  std::vector<uint8_t> raw_vertex(position->byte_stride());
  for (FaceIndex fi(0); fi < mesh.num_faces(); ++fi) {
    const Mesh::Face &face = mesh.face(fi);
    VectorD<double, 3> centroid(0.0, 0.0, 0.0);
    for (int c = 0; c < 3; ++c) {
      position->GetValue(position->mapped_index(face[c]),
                         raw_vertex.data());
      Vector3f vertex;
      std::memcpy(&vertex[0], raw_vertex.data(), sizeof(float) * 3);
      centroid += VectorD<double, 3>(vertex[0], vertex[1], vertex[2]);
    }
    const VectorD<double, 3> face_centroid = centroid / 3.0;

    ChunkBuild *chunk = nullptr;
    std::array<int64_t, 3> grid;
    for (int c = 0; c < 3; ++c) {
      const double relative =
          (face_centroid[c] - mesh_bounds.GetMinPoint()[c]) /
          options.chunk_size;
      if (!std::isfinite(relative)) {
        return StreamError("Failed to map a face to a spatial chunk.");
      }
      const double grid_value = std::floor(relative);
      if (grid_value < std::numeric_limits<int32_t>::min() ||
          grid_value > std::numeric_limits<int32_t>::max()) {
        return StreamError("Spatial chunk coordinate exceeds int32 range.");
      }
      grid[c] = static_cast<int64_t>(grid_value);
    }
    chunk = &chunks_by_grid[grid];
    chunk->grid_coordinate = grid;
    chunk->face_ids.push_back(fi.value());

    for (int c = 0; c < 3; ++c) {
      position->GetValue(position->mapped_index(face[c]),
                         raw_vertex.data());
      Vector3f vertex;
      std::memcpy(&vertex[0], raw_vertex.data(), sizeof(float) * 3);
      chunk->bounds.Update(vertex);
    }
  }

  struct EncodedChunk {
    SpatialMeshChunkInfo info;
    std::vector<uint32_t> point_ids;
    std::vector<uint32_t> face_ids;
    std::vector<uint8_t> payload;
  };
  std::vector<EncodedChunk> encoded_chunks;
  encoded_chunks.reserve(chunks_by_grid.size());
  std::vector<uint32_t> point_chunk_count(mesh.num_points(), 0);
  uint64_t total_compressed_bytes = 0;
  uint32_t chunk_id = 0;
  for (const auto &entry : chunks_by_grid) {
    const ChunkBuild &build = entry.second;
    std::vector<uint32_t> point_ids;
    point_ids.reserve(build.face_ids.size() * 3);
    for (uint32_t face_id : build.face_ids) {
      const Mesh::Face &face = mesh.face(FaceIndex(face_id));
      for (int c = 0; c < 3; ++c) {
        point_ids.push_back(face[c].value());
      }
    }
    std::sort(point_ids.begin(), point_ids.end());
    point_ids.erase(std::unique(point_ids.begin(), point_ids.end()),
                    point_ids.end());
    for (uint32_t point_id : point_ids) {
      if (point_id >= point_chunk_count.size()) {
        return StreamError("A chunk references an invalid source point.");
      }
      ++point_chunk_count[point_id];
    }

    EncodedChunk chunk;
    chunk.info.chunk_id = chunk_id++;
    chunk.info.grid_coordinate = build.grid_coordinate;
    chunk.info.bounds = build.bounds;
    chunk.info.num_points = static_cast<uint32_t>(point_ids.size());
    chunk.info.num_faces = static_cast<uint32_t>(build.face_ids.size());
    chunk.point_ids = point_ids;
    chunk.face_ids = build.face_ids;
    const Status encode_status =
        EncodeChunk(mesh, chunk.point_ids, chunk.face_ids, encodings, options,
                    &chunk.payload);
    if (!encode_status.ok()) {
      return encode_status;
    }
    chunk.info.compressed_bytes = chunk.payload.size();
    total_compressed_bytes += chunk.payload.size();
    encoded_chunks.push_back(std::move(chunk));
  }

  uint32_t boundary_point_count = 0;
  for (uint32_t count : point_chunk_count) {
    if (count > 1) {
      ++boundary_point_count;
    }
  }
  for (EncodedChunk &chunk : encoded_chunks) {
    for (uint32_t point_id : chunk.point_ids) {
      if (point_chunk_count[point_id] > 1) {
        ++chunk.info.boundary_point_count;
      }
    }
  }

  ContainerWriter writer;
  writer.WriteBytes(kMagic, sizeof(kMagic));
  writer.WriteU16(kVersion);
  writer.WriteU16(0);
  writer.WriteU32(static_cast<uint32_t>(encoded_chunks.size()));
  writer.WriteU32(static_cast<uint32_t>(mesh.num_attributes()));
  writer.WriteU32(mesh.num_points());
  writer.WriteU32(mesh.num_faces());
  writer.WriteU32(boundary_point_count);
  writer.WriteU64(total_compressed_bytes);
  writer.WriteF32(options.chunk_size);
  writer.WriteF32(mesh_bounds.GetMinPoint()[0]);
  writer.WriteF32(mesh_bounds.GetMinPoint()[1]);
  writer.WriteF32(mesh_bounds.GetMinPoint()[2]);
  writer.WriteF32(mesh_bounds.GetMaxPoint()[0]);
  writer.WriteF32(mesh_bounds.GetMaxPoint()[1]);
  writer.WriteF32(mesh_bounds.GetMaxPoint()[2]);

  for (int ai = 0; ai < mesh.num_attributes(); ++ai) {
    const PointAttribute *const attribute = mesh.attribute(ai);
    const AttributeEncoding &encoding = encodings[ai];
    writer.WriteU8(static_cast<uint8_t>(attribute->attribute_type()));
    writer.WriteU8(attribute->num_components());
    writer.WriteU8(static_cast<uint8_t>(attribute->data_type()));
    writer.WriteU8(attribute->normalized() ? 1 : 0);
    writer.WriteU32(attribute->unique_id());
#ifdef DRACO_TRANSCODER_SUPPORTED
    writer.WriteString(attribute->name());
#else
    writer.WriteString(std::string());
#endif
    writer.WriteU8(encoding.explicit_quantization ? 1 : 0);
    writer.WriteU32(static_cast<uint32_t>(encoding.quantization_bits));
    for (int c = 0; c < attribute->num_components(); ++c) {
      writer.WriteF32(encoding.quantization_origin.empty()
                          ? 0.f
                          : encoding.quantization_origin[c]);
    }
    writer.WriteF32(encoding.quantization_range);
  }

  for (const EncodedChunk &chunk : encoded_chunks) {
    writer.WriteU32(chunk.info.chunk_id);
    for (int c = 0; c < 3; ++c) {
      writer.WriteI64(chunk.info.grid_coordinate[c]);
    }
    for (int c = 0; c < 3; ++c) {
      writer.WriteF32(chunk.info.bounds.GetMinPoint()[c]);
    }
    for (int c = 0; c < 3; ++c) {
      writer.WriteF32(chunk.info.bounds.GetMaxPoint()[c]);
    }
    writer.WriteU32(chunk.info.num_points);
    writer.WriteU32(chunk.info.num_faces);
    writer.WriteU64(chunk.payload.size());
    writer.WriteU32(chunk.info.boundary_point_count);
    for (uint32_t point_id : chunk.point_ids) {
      writer.WriteU32(point_id);
    }
    for (uint32_t face_id : chunk.face_ids) {
      writer.WriteU32(face_id);
    }
    writer.WriteBytes(chunk.payload.data(), chunk.payload.size());
  }

  SpatialMeshConsistency result;
  result.chunk_count = static_cast<uint32_t>(encoded_chunks.size());
  result.source_point_count = mesh.num_points();
  result.source_face_count = mesh.num_faces();
  result.boundary_point_count = boundary_point_count;
  if (options.verify_boundary_vertices) {
    DecoderBuffer decoder_buffer;
    decoder_buffer.Init(
        reinterpret_cast<const char *>(writer.data().data()),
        writer.data().size());
    StatusOr<std::unique_ptr<SpatialMeshStream>> stream_or =
        SpatialMeshStream::Open(&decoder_buffer);
    if (!stream_or.ok()) {
      return stream_or.status();
    }
    const Status verify_status =
        stream_or.value()->VerifyBoundaryVertices(&result);
    if (!verify_status.ok()) {
      return verify_status;
    }
    if (!result.exact ||
        result.boundary_vertex_mismatches != 0) {
      return StreamError("Boundary vertices are not bit-identical.");
    }
  }

  const Status output_status =
      out_buffer->Encode(writer.data().data(), writer.data().size())
          ? OkStatus()
          : StreamError("Failed to write the output buffer.");
  if (!output_status.ok()) {
    return output_status;
  }
  if (consistency != nullptr) {
    *consistency = result;
  }
  return OkStatus();
}

StatusOr<std::unique_ptr<SpatialMeshStream>> SpatialMeshStream::Open(
    DecoderBuffer *in_buffer) {
  if (in_buffer == nullptr) {
    return StreamError("Input buffer is null.");
  }
  ContainerReader reader(
      reinterpret_cast<const uint8_t *>(in_buffer->data_head()),
      in_buffer->remaining_size());
  char magic[sizeof(kMagic)];
  if (!reader.ReadBytes(magic, sizeof(magic)) ||
      std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    return StreamError("Invalid spatial mesh stream magic.");
  }
  uint16_t version;
  uint16_t reserved;
  uint32_t num_chunks;
  uint32_t num_attributes;
  uint32_t source_point_count;
  uint32_t source_face_count;
  uint32_t boundary_point_count;
  uint64_t total_compressed_bytes;
  float chunk_size;
  float mesh_min[3];
  float mesh_max[3];
  if (!reader.ReadU16(&version) || !reader.ReadU16(&reserved) ||
      !reader.ReadU32(&num_chunks) ||
      !reader.ReadU32(&num_attributes) ||
      !reader.ReadU32(&source_point_count) ||
      !reader.ReadU32(&source_face_count) ||
      !reader.ReadU32(&boundary_point_count) ||
      !reader.ReadU64(&total_compressed_bytes) ||
      !reader.ReadF32(&chunk_size)) {
    return StreamError("Truncated spatial mesh stream header.");
  }
  for (int c = 0; c < 3; ++c) {
    if (!reader.ReadF32(&mesh_min[c])) {
      return StreamError("Truncated spatial mesh stream bounds.");
    }
  }
  for (int c = 0; c < 3; ++c) {
    if (!reader.ReadF32(&mesh_max[c])) {
      return StreamError("Truncated spatial mesh stream bounds.");
    }
  }
  (void)reserved;
  (void)chunk_size;
  (void)mesh_min;
  (void)mesh_max;
  if (version != kVersion) {
    return StreamError("Unsupported spatial mesh stream version.");
  }
  if (num_chunks == 0 || num_chunks > (1u << 24)) {
    return StreamError("Invalid spatial chunk count.");
  }
  if (num_attributes == 0 || num_attributes > 1024) {
    return StreamError("Invalid spatial attribute count.");
  }
  if (source_point_count == 0 || source_face_count == 0) {
    return StreamError("Spatial mesh stream contains no geometry.");
  }

  std::unique_ptr<SpatialMeshStream> stream(new SpatialMeshStream());
  stream->source_point_count_ = source_point_count;
  stream->source_face_count_ = source_face_count;
  stream->compressed_bytes_ = total_compressed_bytes;
  stream->point_chunk_count_.resize(source_point_count, 0);

  stream->attribute_data_.resize(num_attributes);
  stream->attribute_infos_.resize(num_attributes);
  for (uint32_t ai = 0; ai < num_attributes; ++ai) {
    uint8_t attribute_type;
    uint8_t num_components;
    uint8_t data_type;
    uint8_t normalized;
    uint32_t unique_id;
    std::string name;
    uint8_t explicit_quantization;
    uint32_t quantization_bits;
    if (!reader.ReadU8(&attribute_type) ||
        !reader.ReadU8(&num_components) ||
        !reader.ReadU8(&data_type) || !reader.ReadU8(&normalized) ||
        !reader.ReadU32(&unique_id) || !reader.ReadString(&name) ||
        !reader.ReadU8(&explicit_quantization) ||
        !reader.ReadU32(&quantization_bits)) {
      return StreamError("Truncated spatial attribute metadata.");
    }
    if (attribute_type >= GeometryAttribute::NAMED_ATTRIBUTES_COUNT ||
        num_components == 0 || num_components > 16 ||
        data_type == DT_INVALID || data_type >= DT_TYPES_COUNT) {
      return StreamError("Invalid spatial attribute metadata.");
    }
    const int32_t component_size = DataTypeLength(
        static_cast<DataType>(data_type));
    if (component_size <= 0) {
      return StreamError("Unsupported spatial attribute data type.");
    }
    SpatialMeshAttributeInfo info;
    info.attribute_type =
        static_cast<GeometryAttribute::Type>(attribute_type);
    info.num_components = num_components;
    info.data_type = static_cast<DataType>(data_type);
    info.normalized = normalized != 0;
    info.unique_id = unique_id;
    info.name = std::move(name);
    info.explicit_quantization = explicit_quantization != 0;
    info.quantization_bits = static_cast<int>(quantization_bits);
    std::vector<float> origin(num_components);
    for (int c = 0; c < num_components; ++c) {
      if (!reader.ReadF32(&origin[c])) {
        return StreamError("Truncated spatial quantization data.");
      }
    }
    if (!reader.ReadF32(&info.quantization_range)) {
      return StreamError("Truncated spatial quantization range.");
    }
    if (info.quantization_bits < 0 || info.quantization_bits > 30) {
      return StreamError("Invalid spatial quantization bits.");
    }
    if (info.explicit_quantization) {
      info.quantization_origin = std::move(origin);
    }
    stream->attribute_infos_[ai] = std::move(info);
    stream->attribute_data_[ai].info = stream->attribute_infos_[ai];
    stream->attribute_data_[ai].stride =
        static_cast<size_t>(component_size) * num_components;
  }

  std::vector<uint8_t> face_seen(source_face_count, 0);
  uint64_t assigned_faces = 0;
  uint64_t actual_compressed_bytes = 0;
  stream->chunk_data_.resize(num_chunks);
  stream->chunk_infos_.resize(num_chunks);
  for (uint32_t ci = 0; ci < num_chunks; ++ci) {
    ChunkData &chunk = stream->chunk_data_[ci];
    if (!reader.ReadU32(&chunk.info.chunk_id)) {
      return StreamError("Truncated spatial chunk header.");
    }
    if (chunk.info.chunk_id != ci) {
      return StreamError("Spatial chunk ids must be contiguous.");
    }
    for (int c = 0; c < 3; ++c) {
      if (!reader.ReadI64(&chunk.info.grid_coordinate[c])) {
        return StreamError("Truncated spatial chunk grid coordinate.");
      }
    }
    Vector3f min_point;
    Vector3f max_point;
    for (int c = 0; c < 3; ++c) {
      if (!reader.ReadF32(&min_point[c])) {
        return StreamError("Truncated spatial chunk bounds.");
      }
    }
    for (int c = 0; c < 3; ++c) {
      if (!reader.ReadF32(&max_point[c])) {
        return StreamError("Truncated spatial chunk bounds.");
      }
    }
    chunk.info.bounds = BoundingBox(min_point, max_point);
    uint64_t payload_size;
    if (!reader.ReadU32(&chunk.info.num_points) ||
        !reader.ReadU32(&chunk.info.num_faces) ||
        !reader.ReadU64(&payload_size) ||
        !reader.ReadU32(&chunk.info.boundary_point_count)) {
      return StreamError("Truncated spatial chunk metadata.");
    }
    chunk.info.compressed_bytes = payload_size;
    if (chunk.info.num_points == 0 || chunk.info.num_faces == 0 ||
        chunk.info.num_points > source_point_count ||
        chunk.info.num_faces > source_face_count) {
      return StreamError("Invalid spatial chunk sizes.");
    }

    chunk.point_ids.resize(chunk.info.num_points);
    std::unordered_set<uint32_t> point_ids;
    point_ids.reserve(chunk.info.num_points * 2);
    for (uint32_t i = 0; i < chunk.info.num_points; ++i) {
      uint32_t point_id;
      if (!reader.ReadU32(&point_id) || point_id >= source_point_count) {
        return StreamError("Invalid source point id in a spatial chunk.");
      }
      if (!point_ids.insert(point_id).second) {
        return StreamError("Duplicate source point id in a spatial chunk.");
      }
      chunk.point_ids[i] = point_id;
      ++stream->point_chunk_count_[point_id];
    }
    chunk.face_ids.resize(chunk.info.num_faces);
    for (uint32_t i = 0; i < chunk.info.num_faces; ++i) {
      uint32_t face_id;
      if (!reader.ReadU32(&face_id) || face_id >= source_face_count) {
        return StreamError("Invalid source face id in a spatial chunk.");
      }
      if (face_seen[face_id] != 0) {
        return StreamError("A source face is assigned to multiple chunks.");
      }
      face_seen[face_id] = 1;
      chunk.face_ids[i] = face_id;
    }
    assigned_faces += chunk.info.num_faces;
    if (payload_size > reader.remaining()) {
      return StreamError("Truncated spatial chunk payload.");
    }
    chunk.payload.resize(static_cast<size_t>(payload_size));
    if (!reader.ReadBytes(chunk.payload.data(),
                          static_cast<size_t>(payload_size))) {
      return StreamError("Truncated spatial chunk payload.");
    }
    actual_compressed_bytes += payload_size;
  }
  if (assigned_faces != source_face_count) {
    return StreamError("Not all source faces are assigned to chunks.");
  }
  if (actual_compressed_bytes != total_compressed_bytes) {
    return StreamError("Spatial compressed byte count does not match.");
  }
  uint32_t actual_boundary_points = 0;
  for (uint32_t count : stream->point_chunk_count_) {
    if (count > 1) {
      ++actual_boundary_points;
    }
  }
  if (actual_boundary_points != boundary_point_count) {
    return StreamError("Spatial boundary point count does not match.");
  }
  stream->boundary_point_count_ = actual_boundary_points;
  for (uint32_t ci = 0; ci < num_chunks; ++ci) {
    stream->chunk_infos_[ci] = stream->chunk_data_[ci].info;
  }
  return stream;
}

uint32_t SpatialMeshStream::num_chunks() const {
  return static_cast<uint32_t>(chunk_infos_.size());
}

uint32_t SpatialMeshStream::num_attributes() const {
  return static_cast<uint32_t>(attribute_infos_.size());
}

uint32_t SpatialMeshStream::source_point_count() const {
  return source_point_count_;
}

uint32_t SpatialMeshStream::source_face_count() const {
  return source_face_count_;
}

uint32_t SpatialMeshStream::boundary_point_count() const {
  return boundary_point_count_;
}

uint64_t SpatialMeshStream::compressed_bytes() const {
  return compressed_bytes_;
}

const std::vector<SpatialMeshChunkInfo> &SpatialMeshStream::chunks() const {
  return chunk_infos_;
}

const std::vector<SpatialMeshAttributeInfo> &
SpatialMeshStream::attributes() const {
  return attribute_infos_;
}

Status SpatialMeshStream::VerifyBoundaryVertices(
    SpatialMeshConsistency *consistency) const {
  SpatialMeshConsistency result;
  result.chunk_count = num_chunks();
  result.source_point_count = source_point_count_;
  result.source_face_count = source_face_count_;
  result.boundary_point_count = boundary_point_count_;
  std::unordered_map<uint32_t, std::vector<uint8_t>> first_signatures;
  first_signatures.reserve(boundary_point_count_ * 2);

  for (uint32_t ci = 0; ci < num_chunks(); ++ci) {
    const ChunkData &chunk = chunk_data_[ci];
    DecoderBuffer buffer;
    buffer.Init(reinterpret_cast<const char *>(chunk.payload.data()),
                chunk.payload.size());
    Decoder decoder;
    std::unique_ptr<Mesh> mesh(new Mesh());
    const Status status = decoder.DecodeBufferToGeometry(&buffer, mesh.get());
    if (!status.ok()) {
      return status;
    }
    result.decoded_point_count += mesh->num_points();
    result.decoded_face_count += mesh->num_faces();
    if (mesh->num_points() != chunk.point_ids.size() ||
        mesh->num_faces() != chunk.face_ids.size() ||
        static_cast<uint32_t>(mesh->num_attributes()) != num_attributes()) {
      result.exact = false;
      if (consistency != nullptr) {
        *consistency = result;
      }
      return OkStatus();
    }
    for (int ai = 0; ai < mesh->num_attributes(); ++ai) {
      const PointAttribute *const decoded = mesh->attribute(ai);
      const SpatialMeshAttributeInfo &expected = attribute_infos_[ai];
      if (decoded->attribute_type() != expected.attribute_type ||
          decoded->num_components() != expected.num_components ||
          decoded->data_type() != expected.data_type ||
          decoded->normalized() != expected.normalized) {
        result.exact = false;
        if (consistency != nullptr) {
          *consistency = result;
        }
        return OkStatus();
      }
    }

    std::vector<uint8_t> signature;
    for (uint32_t pi = 0; pi < chunk.point_ids.size(); ++pi) {
      const uint32_t source_point_id = chunk.point_ids[pi];
      if (point_chunk_count_[source_point_id] <= 1) {
        continue;
      }
      signature.clear();
      for (int ai = 0; ai < mesh->num_attributes(); ++ai) {
        const PointAttribute *const decoded = mesh->attribute(ai);
        const size_t stride = attribute_data_[ai].stride;
        const size_t offset = signature.size();
        signature.resize(offset + stride);
        decoded->GetValue(decoded->mapped_index(PointIndex(pi)),
                          signature.data() + offset);
      }
      auto inserted =
          first_signatures.emplace(source_point_id, signature);
      if (!inserted.second) {
        ++result.boundary_vertex_comparisons;
        if (inserted.first->second != signature) {
          ++result.boundary_vertex_mismatches;
        }
      }
    }
  }
  result.exact = result.boundary_vertex_mismatches == 0 &&
                 result.decoded_face_count == source_face_count_ &&
                 result.decoded_point_count >= source_point_count_;
  if (consistency != nullptr) {
    *consistency = result;
  }
  return OkStatus();
}

Status SpatialMeshFrame::DecodeIntoSlot(const SpatialMeshStream &stream,
                                        uint32_t chunk_index, Slot *slot) {
  const SpatialMeshStream::ChunkData &chunk = stream.chunk_data_[chunk_index];
  DecoderBuffer buffer;
  buffer.Init(reinterpret_cast<const char *>(chunk.payload.data()),
              chunk.payload.size());
  Decoder decoder;
  Mesh mesh;
  const Status status = decoder.DecodeBufferToGeometry(&buffer, &mesh);
  if (!status.ok()) {
    return status;
  }
  if (mesh.num_points() != chunk.info.num_points ||
      mesh.num_faces() != chunk.info.num_faces ||
      static_cast<uint32_t>(mesh.num_attributes()) !=
          stream.num_attributes()) {
    return StreamError("Decoded spatial chunk size does not match metadata.");
  }

  slot->chunk_id = kInvalidChunk;
  slot->last_used = update_id_;
  slot->point_ids.assign(chunk.point_ids.begin(), chunk.point_ids.end());
  slot->source_face_ids.assign(chunk.face_ids.begin(), chunk.face_ids.end());
  slot->indices.resize(static_cast<size_t>(mesh.num_faces()) * 3);
  for (FaceIndex fi(0); fi < mesh.num_faces(); ++fi) {
    const Mesh::Face &face = mesh.face(fi);
    for (int c = 0; c < 3; ++c) {
      if (face[c].value() >= mesh.num_points()) {
        return StreamError("Decoded spatial chunk has an invalid index.");
      }
      slot->indices[fi.value() * 3 + c] = face[c].value();
    }
  }

  slot->attributes.resize(stream.num_attributes());
  slot->attribute_views.resize(stream.num_attributes());
  slot->position_attribute_index = -1;
  std::vector<uint8_t> scratch;
  for (uint32_t ai = 0; ai < stream.num_attributes(); ++ai) {
    const SpatialMeshStream::AttributeData &expected =
        stream.attribute_data_[ai];
    const PointAttribute *const decoded = mesh.attribute(ai);
    if (decoded->attribute_type() != expected.info.attribute_type ||
        decoded->num_components() != expected.info.num_components ||
        decoded->data_type() != expected.info.data_type ||
        decoded->normalized() != expected.info.normalized) {
      return StreamError("Decoded spatial attribute does not match metadata.");
    }
    AttributeStorage &storage = slot->attributes[ai];
    storage.info = &stream.attribute_infos_[ai];
    storage.stride = expected.stride;
    storage.values.resize(static_cast<size_t>(mesh.num_points()) *
                          storage.stride);
    scratch.resize(
        std::max<size_t>(scratch.size(), decoded->byte_stride()));
    for (uint32_t pi = 0; pi < mesh.num_points(); ++pi) {
      decoded->GetValue(decoded->mapped_index(PointIndex(pi)),
                        scratch.data());
      std::memcpy(storage.values.data() + static_cast<size_t>(pi) *
                                             storage.stride,
                  scratch.data(), storage.stride);
    }
    slot->attribute_views[ai] = {storage.info, storage.values.data(),
                                 storage.stride};
    if (expected.info.attribute_type == GeometryAttribute::POSITION) {
      if (expected.info.data_type != DT_FLOAT32 ||
          expected.info.num_components != 3) {
        return StreamError("Spatial position attribute is not float32x3.");
      }
      slot->position_attribute_index = static_cast<int32_t>(ai);
    }
  }
  if (slot->position_attribute_index < 0) {
    return StreamError("Spatial chunk has no position attribute.");
  }
  slot->chunk_id = chunk_index;
  return OkStatus();
}

uint64_t SpatialMeshFrame::ComputeAllocatedBytes() const {
  uint64_t bytes = visible_chunks_.capacity() *
                   sizeof(SpatialMeshChunkView);
  bytes += selected_scratch_.capacity() * sizeof(uint32_t);
  bytes += selected_mask_scratch_.capacity() * sizeof(uint8_t);
  bytes += chunk_to_slot_.capacity() * sizeof(int32_t);
  bytes += slots_.capacity() * sizeof(std::unique_ptr<Slot>);
  for (const auto &slot_ptr : slots_) {
    if (!slot_ptr) {
      continue;
    }
    const Slot &slot = *slot_ptr;
    bytes += sizeof(Slot);
    bytes += slot.point_ids.capacity() * sizeof(uint32_t);
    bytes += slot.source_face_ids.capacity() * sizeof(uint32_t);
    bytes += slot.indices.capacity() * sizeof(uint32_t);
    bytes += slot.attributes.capacity() * sizeof(AttributeStorage);
    bytes +=
        slot.attribute_views.capacity() * sizeof(SpatialMeshAttributeView);
    for (const AttributeStorage &storage : slot.attributes) {
      bytes += storage.values.capacity();
    }
  }
  return bytes;
}

void SpatialMeshFrame::ResetStats() {
  stats_ = SpatialMeshFrameStats();
  stats_.allocated_bytes = ComputeAllocatedBytes();
  stats_.peak_allocated_bytes = stats_.allocated_bytes;
}

SpatialMeshFrame::SpatialMeshFrame(uint32_t max_resident_chunks)
    : max_resident_chunks_(max_resident_chunks) {}

SpatialMeshFrame::~SpatialMeshFrame() = default;

SpatialMeshFrame::SpatialMeshFrame(SpatialMeshFrame &&) noexcept = default;

SpatialMeshFrame &SpatialMeshFrame::operator=(SpatialMeshFrame &&) noexcept =
    default;

void SpatialMeshFrame::SetMaxResidentChunks(uint32_t max_resident_chunks) {
  max_resident_chunks_ = max_resident_chunks;
}

uint32_t SpatialMeshFrame::max_resident_chunks() const {
  return max_resident_chunks_;
}

const std::vector<SpatialMeshChunkView> &
SpatialMeshFrame::visible_chunks() const {
  return visible_chunks_;
}

const SpatialMeshFrameStats &SpatialMeshFrame::stats() const { return stats_; }

void SpatialMeshFrame::Clear() {
  for (auto &slot : slots_) {
    slot->chunk_id = kInvalidChunk;
  }
  std::fill(chunk_to_slot_.begin(), chunk_to_slot_.end(), -1);
  visible_chunks_.clear();
  selected_scratch_.clear();
  selected_mask_scratch_.clear();
  ResetStats();
}

void SpatialMeshFrame::Release() {
  slots_.clear();
  slots_.shrink_to_fit();
  chunk_to_slot_.clear();
  chunk_to_slot_.shrink_to_fit();
  visible_chunks_.clear();
  visible_chunks_.shrink_to_fit();
  selected_scratch_.clear();
  selected_scratch_.shrink_to_fit();
  selected_mask_scratch_.clear();
  selected_mask_scratch_.shrink_to_fit();
  update_id_ = 0;
  stats_ = SpatialMeshFrameStats();
}

Status SpatialMeshStream::DecodeNear(const Vector3f &camera, float radius,
                                     SpatialMeshFrame *frame) const {
  if (frame == nullptr) {
    return StreamError("Decode frame is null.");
  }
  if (!IsFinite(camera)) {
    return StreamError("Camera position must be finite.");
  }
  if (std::isnan(radius) || radius < 0.f) {
    return StreamError("Decode radius must be non-negative and not NaN.");
  }

  const double radius_squared =
      std::isinf(radius) ? std::numeric_limits<double>::infinity()
                         : static_cast<double>(radius) * radius;
  std::vector<uint32_t> &selected = frame->selected_scratch_;
  selected.clear();
  if (selected.capacity() < chunk_data_.size()) {
    selected.reserve(chunk_data_.size());
  }
  for (uint32_t ci = 0; ci < chunk_data_.size(); ++ci) {
    if (SquaredDistanceToBounds(camera, chunk_data_[ci].info.bounds) <=
        radius_squared) {
      selected.push_back(ci);
    }
  }

  if (frame->chunk_to_slot_.size() != chunk_data_.size()) {
    frame->chunk_to_slot_.assign(chunk_data_.size(), -1);
  }
  const uint32_t wanted_slots =
      std::max<uint32_t>(frame->max_resident_chunks_,
                         static_cast<uint32_t>(selected.size()));
  while (frame->slots_.size() < wanted_slots) {
    frame->slots_.push_back(std::unique_ptr<SpatialMeshFrame::Slot>(
        new SpatialMeshFrame::Slot()));
    frame->slots_.back()->chunk_id = kInvalidChunk;
  }

  ++frame->update_id_;
  std::vector<uint8_t> &selected_mask = frame->selected_mask_scratch_;
  selected_mask.assign(chunk_data_.size(), 0);
  for (uint32_t chunk_index : selected) {
    selected_mask[chunk_index] = 1;
  }

  for (uint32_t chunk_index : selected) {
    int32_t &slot_index = frame->chunk_to_slot_[chunk_index];
    if (slot_index >= 0) {
      SpatialMeshFrame::Slot &slot = *frame->slots_[slot_index];
      if (slot.chunk_id == chunk_index) {
        slot.last_used = frame->update_id_;
        ++frame->stats_.cache_hits;
        continue;
      }
      slot_index = -1;
    }

    int32_t victim_index = -1;
    uint64_t oldest_use = std::numeric_limits<uint64_t>::max();
    for (uint32_t si = 0; si < frame->slots_.size(); ++si) {
      const SpatialMeshFrame::Slot &slot = *frame->slots_[si];
      if (slot.chunk_id == kInvalidChunk) {
        victim_index = static_cast<int32_t>(si);
        break;
      }
      if (!selected_mask[slot.chunk_id] && slot.last_used < oldest_use) {
        oldest_use = slot.last_used;
        victim_index = static_cast<int32_t>(si);
      }
    }
    if (victim_index < 0) {
      frame->slots_.push_back(std::unique_ptr<SpatialMeshFrame::Slot>(
          new SpatialMeshFrame::Slot()));
      victim_index = static_cast<int32_t>(frame->slots_.size() - 1);
      frame->slots_[victim_index]->chunk_id = kInvalidChunk;
    }

    SpatialMeshFrame::Slot &slot = *frame->slots_[victim_index];
    if (slot.chunk_id != kInvalidChunk) {
      frame->chunk_to_slot_[slot.chunk_id] = -1;
      slot.chunk_id = kInvalidChunk;
      ++frame->stats_.evictions;
    }
    const Status status = frame->DecodeIntoSlot(*this, chunk_index, &slot);
    if (!status.ok()) {
      return status;
    }
    frame->chunk_to_slot_[chunk_index] = victim_index;
    ++frame->stats_.decoded_chunks;
  }

  frame->visible_chunks_.clear();
  frame->visible_chunks_.reserve(selected.size());
  for (uint32_t chunk_index : selected) {
    const int32_t slot_index = frame->chunk_to_slot_[chunk_index];
    if (slot_index < 0) {
      return StreamError("Selected spatial chunk is not resident.");
    }
    const SpatialMeshFrame::Slot &slot = *frame->slots_[slot_index];
    if (slot.chunk_id != chunk_index ||
        slot.position_attribute_index < 0) {
      return StreamError("Selected spatial chunk buffer is invalid.");
    }
    SpatialMeshChunkView view;
    view.info = &chunk_data_[chunk_index].info;
    view.positions = reinterpret_cast<const float *>(
        slot.attributes[slot.position_attribute_index].values.data());
    view.point_ids = slot.point_ids.data();
    view.source_face_ids = slot.source_face_ids.data();
    view.indices = slot.indices.data();
    view.attributes = slot.attribute_views.data();
    view.num_attributes = slot.attribute_views.size();
    frame->visible_chunks_.push_back(view);
  }

  frame->stats_.visible_chunks =
      static_cast<uint32_t>(frame->visible_chunks_.size());
  frame->stats_.resident_chunks = 0;
  frame->stats_.resident_bytes = 0;
  for (const auto &slot : frame->slots_) {
    if (slot->chunk_id == kInvalidChunk) {
      continue;
    }
    ++frame->stats_.resident_chunks;
    frame->stats_.resident_bytes +=
        slot->point_ids.size() * sizeof(uint32_t);
    frame->stats_.resident_bytes +=
        slot->source_face_ids.size() * sizeof(uint32_t);
    frame->stats_.resident_bytes += slot->indices.size() * sizeof(uint32_t);
    for (const auto &storage : slot->attributes) {
      frame->stats_.resident_bytes += storage.values.size();
    }
  }
  frame->stats_.allocated_bytes = frame->ComputeAllocatedBytes();
  frame->stats_.peak_allocated_bytes =
      std::max(frame->stats_.peak_allocated_bytes,
               frame->stats_.allocated_bytes);
  return OkStatus();
}

}  // namespace draco

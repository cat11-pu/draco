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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "draco/compression/config/compression_shared.h"
#include "draco/compression/decode.h"
#include "draco/compression/expert_encode.h"
#include "draco/core/draco_test_base.h"
#include "draco/core/options.h"

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace draco {
namespace {

std::unique_ptr<Mesh> MakeGridMesh(int cells, float extent) {
  std::unique_ptr<Mesh> mesh(new Mesh());
  const int side = cells + 1;
  const int num_points = side * side;
  mesh->set_num_points(num_points);

  std::unique_ptr<PointAttribute> position(new PointAttribute());
  position->Init(GeometryAttribute::POSITION, 3, DT_FLOAT32, false,
                 num_points);
  std::unique_ptr<PointAttribute> tex_coord(new PointAttribute());
  tex_coord->Init(GeometryAttribute::TEX_COORD, 2, DT_FLOAT32, false,
                  num_points);
  std::unique_ptr<PointAttribute> normal(new PointAttribute());
  normal->Init(GeometryAttribute::NORMAL, 3, DT_FLOAT32, false, num_points);
  const float step = (2.f * extent) / cells;
  for (int z = 0; z < side; ++z) {
    for (int x = 0; x < side; ++x) {
      const int index = z * side + x;
      const float px = -extent + step * x;
      const float pz = -extent + step * z;
      const float p[3] = {px, 0.125f * std::sin(px) * std::cos(pz),
                          pz};
      const float uv[2] = {static_cast<float>(x) / cells,
                           static_cast<float>(z) / cells};
      const float n[3] = {0.f, 1.f, 0.f};
      position->SetAttributeValue(AttributeValueIndex(index), p);
      tex_coord->SetAttributeValue(AttributeValueIndex(index), uv);
      normal->SetAttributeValue(AttributeValueIndex(index), n);
    }
  }
  mesh->AddAttribute(std::move(position));
  mesh->AddAttribute(std::move(tex_coord));
  mesh->AddAttribute(std::move(normal));

  mesh->SetNumFaces(cells * cells * 2);
  FaceIndex face_id(0);
  for (int z = 0; z < cells; ++z) {
    for (int x = 0; x < cells; ++x) {
      const PointIndex p00(z * side + x);
      const PointIndex p10(z * side + x + 1);
      const PointIndex p01((z + 1) * side + x);
      const PointIndex p11((z + 1) * side + x + 1);
      mesh->SetFace(face_id++, {{p00, p01, p10}});
      mesh->SetFace(face_id++, {{p10, p01, p11}});
    }
  }
  return mesh;
}

std::unique_ptr<Mesh> MakeSingleTriangle() {
  std::unique_ptr<Mesh> mesh(new Mesh());
  mesh->set_num_points(3);
  std::unique_ptr<PointAttribute> position(new PointAttribute());
  position->Init(GeometryAttribute::POSITION, 3, DT_FLOAT32, false, 3);
  const float positions[9] = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
  for (int i = 0; i < 3; ++i) {
    position->SetAttributeValue(AttributeValueIndex(i),
                                positions + i * 3);
  }
  mesh->AddAttribute(std::move(position));
  mesh->SetNumFaces(1);
  mesh->SetFace(FaceIndex(0),
                {{PointIndex(0), PointIndex(1), PointIndex(2)}});
  return mesh;
}

std::vector<uint8_t> EncodeStream(
    const Mesh &mesh, const SpatialMeshStreamOptions &options,
    SpatialMeshConsistency *consistency) {
  EncoderBuffer buffer;
  const Status status =
      SpatialMeshStreamEncoder::Encode(mesh, options, &buffer, consistency);
  if (!status.ok()) {
    std::cerr << "Encode failed: " << status.error_msg() << std::endl;
    return {};
  }
  return std::vector<uint8_t>(
      reinterpret_cast<const uint8_t *>(buffer.data()),
      reinterpret_cast<const uint8_t *>(buffer.data()) + buffer.size());
}

std::unique_ptr<SpatialMeshStream> OpenStream(
    const std::vector<uint8_t> &encoded) {
  if (encoded.empty()) {
    return nullptr;
  }
  DecoderBuffer buffer;
  buffer.Init(reinterpret_cast<const char *>(encoded.data()), encoded.size());
  auto stream_or = SpatialMeshStream::Open(&buffer);
  if (!stream_or.ok()) {
    std::cerr << "Open failed: " << stream_or.status().error_msg() << std::endl;
    return nullptr;
  }
  return std::move(stream_or).value();
}

std::vector<uint8_t> EncodeWholeMeshReference(
    const Mesh &mesh, const SpatialMeshStreamOptions &options) {
  ExpertEncoder encoder(mesh);
  encoder.SetSpeedOptions(options.speed, options.speed);
  encoder.SetEncodingMethod(MESH_SEQUENTIAL_ENCODING);
  for (int ai = 0; ai < mesh.num_attributes(); ++ai) {
    const PointAttribute *const attribute = mesh.attribute(ai);
    if (attribute->data_type() != DT_FLOAT32) {
      continue;
    }
    int bits = 0;
    switch (attribute->attribute_type()) {
      case GeometryAttribute::POSITION:
        bits = options.position_quantization_bits;
        break;
      case GeometryAttribute::TEX_COORD:
        bits = options.tex_coord_quantization_bits;
        break;
      case GeometryAttribute::NORMAL:
        bits = options.normal_quantization_bits;
        break;
      case GeometryAttribute::COLOR:
        bits = options.color_quantization_bits;
        break;
      default:
        bits = options.generic_quantization_bits;
        break;
    }
    if (bits > 0) {
      encoder.SetAttributeQuantization(ai, bits);
    }
  }
  EncoderBuffer buffer;
  if (!encoder.EncodeToBuffer(&buffer).ok()) {
    return {};
  }
  return std::vector<uint8_t>(
      reinterpret_cast<const uint8_t *>(buffer.data()),
      reinterpret_cast<const uint8_t *>(buffer.data()) + buffer.size());
}

std::unique_ptr<Mesh> DecodeWholeMeshReference(
    const std::vector<uint8_t> &encoded) {
  DecoderBuffer decoder_buffer;
  decoder_buffer.Init(reinterpret_cast<const char *>(encoded.data()),
                      encoded.size());
  Decoder decoder;
  auto mesh_or = decoder.DecodeMeshFromBuffer(&decoder_buffer);
  if (!mesh_or.ok()) {
    return nullptr;
  }
  return std::move(mesh_or).value();
}

Vector3f ChunkCenter(const SpatialMeshChunkInfo &chunk) {
  return (chunk.bounds.GetMinPoint() + chunk.bounds.GetMaxPoint()) / 2.f;
}

TEST(SpatialMeshStreamTest, RejectsInvalidInputs) {
  SpatialMeshStreamOptions options;
  EncoderBuffer buffer;

  Mesh empty;
  EXPECT_FALSE(SpatialMeshStreamEncoder::Encode(empty, options, &buffer).ok());

  auto mesh = MakeGridMesh(2, 1.f);
  options.chunk_size = 0.f;
  EXPECT_FALSE(
      SpatialMeshStreamEncoder::Encode(*mesh, options, &buffer).ok());
  options.chunk_size = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(
      SpatialMeshStreamEncoder::Encode(*mesh, options, &buffer).ok());
  options.chunk_size = 1.f;
  options.speed = 11;
  EXPECT_FALSE(
      SpatialMeshStreamEncoder::Encode(*mesh, options, &buffer).ok());
}

TEST(SpatialMeshStreamTest, ExplicitQuantizationOptionsRoundTrip) {
  Options options;
  const float scalar = -0.121492236f;
  const Vector3f vector(-4.f, scalar, 3.14159265f);
  options.SetFloat("scalar", scalar);
  options.SetVector("vector", &vector[0], 3);
  EXPECT_EQ(options.GetFloat("scalar"), scalar);
  Vector3f decoded;
  ASSERT_TRUE(options.GetVector("vector", 3, &decoded[0]));
  EXPECT_EQ(decoded[0], vector[0]);
  EXPECT_EQ(decoded[1], vector[1]);
  EXPECT_EQ(decoded[2], vector[2]);
}

TEST(SpatialMeshStreamTest, BoundaryVerticesAreBitIdentical) {
  auto mesh = MakeGridMesh(8, 4.f);
  SpatialMeshStreamOptions options;
  options.chunk_size = 1.f;
  options.position_quantization_bits = 16;
  SpatialMeshConsistency encoded_consistency;
  const auto encoded =
      EncodeStream(*mesh, options, &encoded_consistency);
  ASSERT_FALSE(encoded.empty());
  EXPECT_EQ(encoded_consistency.boundary_vertex_mismatches, 0u);
  EXPECT_TRUE(encoded_consistency.exact);
  EXPECT_GT(encoded_consistency.chunk_count, 1u);
  EXPECT_GT(encoded_consistency.boundary_point_count, 0u);

  auto stream = OpenStream(encoded);
  ASSERT_NE(stream, nullptr);
  EXPECT_EQ(stream->num_chunks(), encoded_consistency.chunk_count);
  EXPECT_EQ(stream->source_point_count(), mesh->num_points());
  EXPECT_EQ(stream->source_face_count(), mesh->num_faces());
  EXPECT_GT(stream->boundary_point_count(), 0u);

  SpatialMeshConsistency verified;
  ASSERT_TRUE(stream->VerifyBoundaryVertices(&verified).ok());
  EXPECT_EQ(verified.boundary_vertex_mismatches, 0u);
  EXPECT_EQ(verified.decoded_face_count, mesh->num_faces());
  EXPECT_TRUE(verified.exact);

  std::cout << "[stream] chunks=" << stream->num_chunks()
            << " points=" << stream->source_point_count()
            << " faces=" << stream->source_face_count()
            << " boundary_points=" << stream->boundary_point_count()
            << " comparisons=" << verified.boundary_vertex_comparisons
            << " mismatches=" << verified.boundary_vertex_mismatches
            << " bytes=" << stream->compressed_bytes() << std::endl;
}

TEST(SpatialMeshStreamTest, FullAndLocalDecodeMatchWholeMeshReference) {
  auto mesh = MakeGridMesh(24, 4.f);
  SpatialMeshStreamOptions options;
  options.chunk_size = 1.f;
  options.position_quantization_bits = 16;
  SpatialMeshConsistency encoded_consistency;
  const auto encoded =
      EncodeStream(*mesh, options, &encoded_consistency);
  ASSERT_FALSE(encoded.empty());
  auto stream = OpenStream(encoded);
  ASSERT_NE(stream, nullptr);
  const auto whole_encoded = EncodeWholeMeshReference(*mesh, options);
  ASSERT_FALSE(whole_encoded.empty());
  auto whole = DecodeWholeMeshReference(whole_encoded);
  ASSERT_NE(whole, nullptr);
  ASSERT_EQ(whole->num_points(), mesh->num_points());
  ASSERT_EQ(whole->num_faces(), mesh->num_faces());
  ASSERT_EQ(whole->num_attributes(), mesh->num_attributes());

  SpatialMeshFrame full;
  ASSERT_TRUE(stream->DecodeNear(Vector3f(0.f, 0.f, 0.f),
                                 std::numeric_limits<float>::infinity(),
                                 &full)
                  .ok());
  EXPECT_EQ(full.visible_chunks().size(), stream->num_chunks());

  uint64_t point_items = 0;
  uint64_t face_items = 0;
  uint64_t mismatches = 0;
  std::vector<uint64_t> attribute_mismatches(whole->num_attributes(), 0);
  for (const SpatialMeshChunkView &chunk : full.visible_chunks()) {
    for (uint32_t pi = 0; pi < chunk.info->num_points; ++pi) {
      const PointIndex source_point(chunk.point_ids[pi]);
      ASSERT_LT(source_point.value(), whole->num_points());
      for (int ai = 0; ai < whole->num_attributes(); ++ai) {
        const PointAttribute *const whole_att = whole->attribute(ai);
        const SpatialMeshAttributeView &view = chunk.attributes[ai];
        std::vector<uint8_t> expected(view.stride);
        whole_att->GetValue(whole_att->mapped_index(source_point),
                            expected.data());
        const uint8_t *actual =
            view.data + static_cast<size_t>(pi) * view.stride;
        if (std::memcmp(expected.data(), actual, view.stride) != 0) {
          ++mismatches;
          ++attribute_mismatches[ai];
        }
      }
      ++point_items;
    }
    for (uint32_t fi = 0; fi < chunk.info->num_faces; ++fi) {
      const uint32_t source_face_id = chunk.source_face_ids[fi];
      const Mesh::Face &expected = whole->face(FaceIndex(source_face_id));
      for (int c = 0; c < 3; ++c) {
        const uint32_t local_vertex = chunk.indices[fi * 3 + c];
        ASSERT_LT(local_vertex, chunk.info->num_points);
        const uint32_t actual_source_point =
            chunk.point_ids[local_vertex];
        if (expected[c].value() != actual_source_point) {
          ++mismatches;
        }
      }
      ++face_items;
    }
  }
  EXPECT_EQ(mismatches, 0u);
  EXPECT_EQ(point_items,
            static_cast<uint64_t>(std::accumulate(
                stream->chunks().begin(), stream->chunks().end(), 0u,
                [](uint32_t sum, const SpatialMeshChunkInfo &chunk) {
                  return sum + chunk.num_points;
                })));
  EXPECT_EQ(face_items, mesh->num_faces());

  SpatialMeshFrame local;
  ASSERT_TRUE(
      stream->DecodeNear(Vector3f(-1.25f, 0.f, -1.25f), 1.f, &local).ok());
  EXPECT_GT(local.visible_chunks().size(), 0u);
  EXPECT_LT(local.visible_chunks().size(), stream->num_chunks());
  for (const SpatialMeshChunkView &local_chunk : local.visible_chunks()) {
    const SpatialMeshChunkView *full_chunk = nullptr;
    for (const SpatialMeshChunkView &candidate : full.visible_chunks()) {
      if (candidate.info->chunk_id == local_chunk.info->chunk_id) {
        full_chunk = &candidate;
        break;
      }
    }
    ASSERT_NE(full_chunk, nullptr);
    ASSERT_EQ(local_chunk.info->num_points, full_chunk->info->num_points);
    ASSERT_EQ(local_chunk.info->num_faces, full_chunk->info->num_faces);
    EXPECT_EQ(std::memcmp(local_chunk.point_ids, full_chunk->point_ids,
                          local_chunk.info->num_points * sizeof(uint32_t)),
              0);
    EXPECT_EQ(std::memcmp(local_chunk.indices, full_chunk->indices,
                          local_chunk.info->num_faces * 3 * sizeof(uint32_t)),
              0);
    for (int ai = 0; ai < whole->num_attributes(); ++ai) {
      EXPECT_EQ(std::memcmp(local_chunk.attributes[ai].data,
                            full_chunk->attributes[ai].data,
                            local_chunk.info->num_points *
                                local_chunk.attributes[ai].stride),
                0);
    }
  }

  std::cout << "[item-consistency] reference=whole-mesh"
            << " chunk_points=" << point_items
            << " chunk_faces=" << face_items
            << " local_chunks=" << local.visible_chunks().size()
            << " mismatches=" << mismatches;
  for (int ai = 0; ai < whole->num_attributes(); ++ai) {
    std::cout << " attr" << ai << "_mismatches=" << attribute_mismatches[ai];
  }
  std::cout << std::endl;
}

TEST(SpatialMeshStreamTest, QueryBoundaryAndCacheOverflow) {
  auto mesh = MakeGridMesh(8, 4.f);
  SpatialMeshStreamOptions options;
  options.chunk_size = 1.f;
  const auto encoded = EncodeStream(*mesh, options, nullptr);
  ASSERT_FALSE(encoded.empty());
  auto stream = OpenStream(encoded);
  ASSERT_NE(stream, nullptr);
  ASSERT_GT(stream->num_chunks(), 1u);

  const SpatialMeshChunkInfo &chunk = stream->chunks()[0];
  const Vector3f center = ChunkCenter(chunk);
  const float radius = 0.25f;
  const Vector3f touching_camera(
      chunk.bounds.GetMinPoint()[0] - radius, center[1], center[2]);

  SpatialMeshFrame frame;
  ASSERT_TRUE(stream->DecodeNear(touching_camera, radius, &frame).ok());
  bool found = false;
  for (const auto &view : frame.visible_chunks()) {
    found |= view.info->chunk_id == chunk.chunk_id;
  }
  EXPECT_TRUE(found);

  ASSERT_TRUE(
      stream->DecodeNear(touching_camera, radius - 0.01f, &frame).ok());
  found = false;
  for (const auto &view : frame.visible_chunks()) {
    found |= view.info->chunk_id == chunk.chunk_id;
  }
  EXPECT_FALSE(found);

  ASSERT_TRUE(stream->DecodeNear(Vector3f(1000.f, 1000.f, 1000.f), 1.f,
                                 &frame)
                  .ok());
  EXPECT_TRUE(frame.visible_chunks().empty());
  EXPECT_EQ(frame.stats().visible_chunks, 0u);

  SpatialMeshFrame one_slot(1);
  ASSERT_TRUE(stream->DecodeNear(Vector3f(0.f, 0.f, 0.f),
                                 std::numeric_limits<float>::infinity(),
                                 &one_slot)
                  .ok());
  EXPECT_EQ(one_slot.visible_chunks().size(), stream->num_chunks());
  EXPECT_EQ(one_slot.stats().resident_chunks, stream->num_chunks());
}

TEST(SpatialMeshStreamTest, RejectsTruncatedContainer) {
  auto mesh = MakeSingleTriangle();
  SpatialMeshStreamOptions options;
  options.chunk_size = 1.f;
  const auto encoded = EncodeStream(*mesh, options, nullptr);
  ASSERT_FALSE(encoded.empty());
  DecoderBuffer buffer;
  buffer.Init(reinterpret_cast<const char *>(encoded.data()),
              encoded.size() - 1);
  EXPECT_FALSE(SpatialMeshStream::Open(&buffer).ok());
  buffer.Init("not-a-stream", 12);
  EXPECT_FALSE(SpatialMeshStream::Open(&buffer).ok());
}

TEST(SpatialMeshStreamTest, SingleChunkHasNoBoundarySet) {
  auto mesh = MakeSingleTriangle();
  SpatialMeshStreamOptions options;
  options.chunk_size = 10.f;
  SpatialMeshConsistency consistency;
  const auto encoded = EncodeStream(*mesh, options, &consistency);
  ASSERT_FALSE(encoded.empty());
  EXPECT_EQ(consistency.chunk_count, 1u);
  EXPECT_EQ(consistency.boundary_point_count, 0u);
  EXPECT_TRUE(consistency.exact);
}

TEST(SpatialMeshStreamTest, HundredThousandViewSwitchesReuseBuffers) {
  auto mesh = MakeGridMesh(8, 4.f);
  SpatialMeshStreamOptions options;
  options.chunk_size = 1.f;
  const auto encoded = EncodeStream(*mesh, options, nullptr);
  ASSERT_FALSE(encoded.empty());
  auto stream = OpenStream(encoded);
  ASSERT_NE(stream, nullptr);
  ASSERT_GE(stream->num_chunks(), 2u);

  const Vector3f camera_a = ChunkCenter(stream->chunks().front());
  const Vector3f camera_b = ChunkCenter(stream->chunks().back());
  SpatialMeshFrame frame(2);
  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(
        stream->DecodeNear((i & 1) ? camera_a : camera_b, 0.f, &frame).ok());
  }
  ASSERT_EQ(frame.stats().decoded_chunks, 2u);
  const uint64_t decoded_after_warmup = frame.stats().decoded_chunks;
  const uint64_t reused_after_warmup = frame.stats().cache_hits;
  const uint64_t allocated_after_warmup = frame.stats().allocated_bytes;

#if defined(__GLIBC__)
  const size_t heap_before = mallinfo2().uordblks;
#endif
  constexpr int kSwitches = 100000;
  for (int i = 0; i < kSwitches; ++i) {
    ASSERT_TRUE(
        stream->DecodeNear((i & 1) ? camera_a : camera_b, 0.f, &frame).ok());
  }
#if defined(__GLIBC__)
  const size_t heap_after = mallinfo2().uordblks;
#endif

  EXPECT_EQ(frame.stats().decoded_chunks, decoded_after_warmup);
  EXPECT_GE(frame.stats().cache_hits - reused_after_warmup,
            static_cast<uint64_t>(kSwitches));
  EXPECT_EQ(frame.stats().evictions, 0u);
  EXPECT_EQ(frame.stats().allocated_bytes, allocated_after_warmup);
  EXPECT_LE(frame.stats().peak_allocated_bytes, allocated_after_warmup);
#if defined(__GLIBC__)
  EXPECT_LE(heap_after, heap_before + 64 * 1024);
#endif

  std::cout << "[cache-100k] switches=" << kSwitches
            << " decoded_chunks=" << frame.stats().decoded_chunks
            << " cache_hits=" << frame.stats().cache_hits
            << " evictions=" << frame.stats().evictions
            << " allocated_bytes=" << frame.stats().allocated_bytes;
#if defined(__GLIBC__)
  std::cout << " heap_before=" << heap_before
            << " heap_after=" << heap_after
            << " heap_growth=" << (static_cast<int64_t>(heap_after) -
                                  static_cast<int64_t>(heap_before));
#endif
  std::cout << std::endl;

  frame.Release();
  EXPECT_TRUE(frame.visible_chunks().empty());
  EXPECT_EQ(frame.stats().allocated_bytes, 0u);
}

}  // namespace
}  // namespace draco

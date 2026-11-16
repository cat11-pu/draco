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
#ifndef DRACO_MESH_SPATIAL_MESH_STREAM_H_
#define DRACO_MESH_SPATIAL_MESH_STREAM_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "draco/core/bounding_box.h"
#include "draco/core/decoder_buffer.h"
#include "draco/core/encoder_buffer.h"
#include "draco/core/status.h"
#include "draco/core/status_or.h"
#include "draco/core/vector_d.h"
#include "draco/mesh/mesh.h"

namespace draco {

// Options used by SpatialMeshStreamEncoder.
struct SpatialMeshStreamOptions {
  // Faces are assigned to chunks by their centroid. A chunk covers
  // [mesh_min + k * chunk_size, mesh_min + (k + 1) * chunk_size) per axis.
  float chunk_size = 1.0f;

  // Quantization settings. Position and texture coordinates use explicit,
  // mesh-wide quantization ranges so that duplicated boundary vertices decode
  // to bit-identical values in every chunk.
  int position_quantization_bits = 14;
  int tex_coord_quantization_bits = 10;
  int normal_quantization_bits = 8;
  int color_quantization_bits = 8;
  int generic_quantization_bits = 8;

  // Draco encoding/decoding speed trade-off. Chunks always use sequential mesh
  // encoding so that the decoded point order matches the chunk's point table.
  int speed = 5;

  // Decodes every chunk once during Encode() and verifies that all vertices
  // shared by multiple chunks have identical attribute bytes.
  bool verify_boundary_vertices = true;
};

// Machine-readable result of boundary and decode verification.
struct SpatialMeshConsistency {
  uint32_t chunk_count = 0;
  uint32_t source_point_count = 0;
  uint32_t source_face_count = 0;
  uint32_t boundary_point_count = 0;
  uint64_t boundary_vertex_comparisons = 0;
  uint64_t boundary_vertex_mismatches = 0;
  uint64_t decoded_point_count = 0;
  uint64_t decoded_face_count = 0;
  bool exact = false;
};

struct SpatialMeshAttributeInfo {
  GeometryAttribute::Type attribute_type = GeometryAttribute::INVALID;
  uint8_t num_components = 0;
  DataType data_type = DT_INVALID;
  bool normalized = false;
  uint32_t unique_id = 0;
  std::string name;

  // |quantization_origin| is populated only when explicit quantization is used.
  int quantization_bits = 0;
  bool explicit_quantization = false;
  std::vector<float> quantization_origin;
  float quantization_range = 0.f;
};

struct SpatialMeshChunkInfo {
  uint32_t chunk_id = 0;
  std::array<int64_t, 3> grid_coordinate = {{0, 0, 0}};
  BoundingBox bounds;
  uint32_t num_points = 0;
  uint32_t num_faces = 0;
  uint64_t compressed_bytes = 0;
  uint32_t boundary_point_count = 0;
};

// Encodes a mesh into independent spatial chunks. The container stores one
// standard Draco mesh bitstream per chunk plus the source point/face maps
// needed to stitch chunk boundaries without guessing.
class SpatialMeshStreamEncoder {
 public:
  static Status Encode(const Mesh &mesh,
                       const SpatialMeshStreamOptions &options,
                       EncoderBuffer *out_buffer,
                       SpatialMeshConsistency *consistency = nullptr);
};

// Attributes of one decoded chunk. All pointers remain valid until the owning
// SpatialMeshFrame is updated or destroyed.
struct SpatialMeshAttributeView {
  const SpatialMeshAttributeInfo *info = nullptr;
  const uint8_t *data = nullptr;
  size_t stride = 0;
};

// One drawable chunk. |indices| contains 3 * num_faces local vertex indices.
// |point_ids| maps every local vertex to the source mesh point id, which is the
// stable key for stitching and for boundary consistency checks.
struct SpatialMeshChunkView {
  const SpatialMeshChunkInfo *info = nullptr;
  const float *positions = nullptr;
  const uint32_t *point_ids = nullptr;
  const uint32_t *source_face_ids = nullptr;
  const uint32_t *indices = nullptr;
  const SpatialMeshAttributeView *attributes = nullptr;
  size_t num_attributes = 0;
};

struct SpatialMeshFrameStats {
  uint32_t visible_chunks = 0;
  uint32_t resident_chunks = 0;
  uint64_t cache_hits = 0;
  uint64_t decoded_chunks = 0;
  uint64_t evictions = 0;
  uint64_t resident_bytes = 0;
  uint64_t allocated_bytes = 0;
  uint64_t peak_allocated_bytes = 0;
};

class SpatialMeshStream;

// Reusable decode target. Pass the same frame to every DecodeNear() call to
// reuse the previous frame's vertex, index, and attribute buffers. Slots are
// retained in an LRU cache; evicted slots keep their capacity for reuse. The
// stream must outlive the frame because views reference its attribute metadata.
class SpatialMeshFrame {
 public:
  explicit SpatialMeshFrame(uint32_t max_resident_chunks = 64);
  ~SpatialMeshFrame();

  SpatialMeshFrame(const SpatialMeshFrame &) = delete;
  SpatialMeshFrame &operator=(const SpatialMeshFrame &) = delete;
  SpatialMeshFrame(SpatialMeshFrame &&) noexcept;
  SpatialMeshFrame &operator=(SpatialMeshFrame &&) noexcept;

  // Zero means "at least as many slots as the current query needs".
  void SetMaxResidentChunks(uint32_t max_resident_chunks);
  uint32_t max_resident_chunks() const;

  const std::vector<SpatialMeshChunkView> &visible_chunks() const;
  const SpatialMeshFrameStats &stats() const;

  // Invalidates resident chunk ids but keeps allocated capacity for reuse.
  void Clear();

  // Releases all memory owned by the frame.
  void Release();

 private:
  friend class SpatialMeshStream;

  struct AttributeStorage {
    const SpatialMeshAttributeInfo *info = nullptr;
    size_t stride = 0;
    std::vector<uint8_t> values;
  };

  struct Slot {
    uint32_t chunk_id = 0;
    uint64_t last_used = 0;
    std::vector<uint32_t> point_ids;
    std::vector<uint32_t> source_face_ids;
    std::vector<uint32_t> indices;
    std::vector<AttributeStorage> attributes;
    std::vector<SpatialMeshAttributeView> attribute_views;
    int32_t position_attribute_index = -1;
  };

  Status DecodeIntoSlot(const SpatialMeshStream &stream, uint32_t chunk_index,
                        Slot *slot);
  uint64_t ComputeAllocatedBytes() const;
  void ResetStats();

  uint32_t max_resident_chunks_ = 64;
  uint64_t update_id_ = 0;
  std::vector<std::unique_ptr<Slot>> slots_;
  std::vector<int32_t> chunk_to_slot_;
  std::vector<SpatialMeshChunkView> visible_chunks_;
  std::vector<uint32_t> selected_scratch_;
  std::vector<uint8_t> selected_mask_scratch_;
  SpatialMeshFrameStats stats_;
};

// Read-only, random-access container. Open() parses metadata and payloads but
// does not decompress any chunk.
class SpatialMeshStream {
 public:
  static StatusOr<std::unique_ptr<SpatialMeshStream>> Open(
      DecoderBuffer *in_buffer);

  uint32_t num_chunks() const;
  uint32_t num_attributes() const;
  uint32_t source_point_count() const;
  uint32_t source_face_count() const;
  uint32_t boundary_point_count() const;
  uint64_t compressed_bytes() const;

  const std::vector<SpatialMeshChunkInfo> &chunks() const;
  const std::vector<SpatialMeshAttributeInfo> &attributes() const;

  // Decodes only chunks whose AABB intersects the sphere
  // (|camera|, |radius|). |frame| is fully replaced and its buffers are reused.
  Status DecodeNear(const Vector3f &camera, float radius,
                    SpatialMeshFrame *frame) const;

  // Decompresses every chunk and verifies that all source points shared by
  // multiple chunks have bit-identical attribute values.
  Status VerifyBoundaryVertices(
      SpatialMeshConsistency *consistency = nullptr) const;

 private:
  friend class SpatialMeshFrame;
  SpatialMeshStream() = default;

  struct AttributeData {
    SpatialMeshAttributeInfo info;
    size_t stride = 0;
  };

  struct ChunkData {
    SpatialMeshChunkInfo info;
    std::vector<uint32_t> point_ids;
    std::vector<uint32_t> face_ids;
    std::vector<uint8_t> payload;
  };

  std::vector<AttributeData> attribute_data_;
  std::vector<SpatialMeshAttributeInfo> attribute_infos_;
  std::vector<ChunkData> chunk_data_;
  std::vector<SpatialMeshChunkInfo> chunk_infos_;
  std::vector<uint32_t> point_chunk_count_;
  uint32_t source_point_count_ = 0;
  uint32_t source_face_count_ = 0;
  uint32_t boundary_point_count_ = 0;
  uint64_t compressed_bytes_ = 0;
};

}  // namespace draco

#endif  // DRACO_MESH_SPATIAL_MESH_STREAM_H_

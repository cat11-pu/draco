#ifndef DRACO_COMPRESSION_ATTRIBUTES_SPATIAL_BLOCK_ATTRIBUTE_CODEC_H_
#define DRACO_COMPRESSION_ATTRIBUTES_SPATIAL_BLOCK_ATTRIBUTE_CODEC_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "draco/core/draco_types.h"
#include "draco/core/status.h"
#include "draco/point_cloud/point_cloud.h"

namespace draco {

// Causal predictors that can be selected independently for each spatial block.
// Both predictors use only already reconstructed samples. The first sample of
// a block therefore continues from the previous block instead of restarting at
// zero.
enum class SpatialBlockPredictor : uint8_t {
  // Predict the current sample from the immediately preceding sample.
  kPrevious = 0,
  // Extrapolate from the two immediately preceding samples. Falls back to
  // kPrevious when fewer than two samples are available.
  kLinear = 1,
};

const char* GetSpatialBlockPredictorName(SpatialBlockPredictor predictor);

struct SpatialBlockCodecOptions {
  // Maximum number of points in one block. Points are first ordered
  // lexicographically by position (with the input point index as a stable
  // tie-breaker), then split into consecutive blocks of this size. The last
  // block may contain fewer points.
  uint32_t max_points_per_block = 1024;

  // Hard upper bound on per-component reconstruction error. The quantization
  // step is set to this value, so nearest-step quantization keeps the error at
  // or below half of the threshold. A value of zero selects exact coding.
  double max_reconstruction_error = 0.0;

  // Predictors considered for every block. The first predictor wins when two
  // predictors have the same encoded residual cost.
  std::vector<SpatialBlockPredictor> predictors = {
      SpatialBlockPredictor::kPrevious,
      SpatialBlockPredictor::kLinear};
};

// Residual statistics for one block. Residuals are measured against the
// selected causal predictor before quantization. Reconstruction error is
// measured against the original attribute values, so it includes quantization.
struct SpatialBlockResidualStats {
  uint32_t block_index = 0;
  uint32_t first_point_index = 0;
  uint32_t last_point_index = 0;
  uint32_t num_points = 0;
  SpatialBlockPredictor predictor = SpatialBlockPredictor::kPrevious;
  std::array<double, 3> min_position = {{0.0, 0.0, 0.0}};
  std::array<double, 3> max_position = {{0.0, 0.0, 0.0}};
  bool has_previous_context = false;
  uint64_t residual_count = 0;
  double sum_abs_residual = 0.0;
  double max_abs_residual = 0.0;
  double rms_residual = 0.0;
  double max_abs_reconstruction_error = 0.0;
  // Maximum absolute predictor residual at the first point of the block. This
  // is the seam residual, not the difference between adjacent source values.
  double boundary_abs_residual = 0.0;
};

// Result of SpatialBlockAttributeCodec::Encode(). |data| is self-contained and
// can be passed directly to Decode() or DecodeBlock().
struct EncodedSpatialBlockAttribute {
  uint32_t num_points = 0;
  uint8_t num_components = 0;
  DataType data_type = DT_INVALID;
  bool normalized = false;
  double max_reconstruction_error = 0.0;
  double quantization_step = 0.0;
  bool exact = true;
  std::vector<uint8_t> data;
  std::vector<SpatialBlockResidualStats> blocks;
};

// Narrow interface for spatially blocked attribute prediction.
//
// Contract:
//  - Encode() reads one attribute and the POSITION attribute from
//    |point_cloud|.
//  - Points are grouped into spatial blocks and one predictor is selected per
//    block.
//  - Each block stores its own previous-sample seed. Consequently, decoding all
//    blocks in one call and decoding each block independently produce exactly
//    the same values, item by item.
//  - Decode() writes point-major values in the original point order. The size
//    is num_points * num_components.
//  - DecodeBlock() writes point-major values in spatial block order and can
//    optionally return the corresponding original point indices.
class SpatialBlockAttributeCodec {
 public:
  static Status Encode(const PointCloud &point_cloud, int attribute_id,
                       const SpatialBlockCodecOptions &options,
                       EncodedSpatialBlockAttribute* encoded);

  static Status Decode(const EncodedSpatialBlockAttribute &encoded,
                       std::vector<double>* values);
  static Status Decode(const uint8_t* data, size_t data_size,
                       std::vector<double>* values);

  static Status DecodeBlock(const EncodedSpatialBlockAttribute &encoded,
                            uint32_t block_index, std::vector<double>* values,
                            std::vector<uint32_t>* point_indices = nullptr);
  static Status DecodeBlock(const uint8_t* data, size_t data_size,
                            uint32_t block_index, std::vector<double>* values,
                            std::vector<uint32_t>* point_indices = nullptr);
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_SPATIAL_BLOCK_ATTRIBUTE_CODEC_H_

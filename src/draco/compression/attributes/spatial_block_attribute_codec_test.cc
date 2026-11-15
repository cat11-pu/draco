#include "draco/compression/attributes/spatial_block_attribute_codec.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "draco/core/draco_test_base.h"
#include "draco/point_cloud/point_cloud_builder.h"

namespace draco {
namespace {

template <typename T, size_t NumComponents>
std::unique_ptr<PointCloud> MakePointCloud(
    const std::vector<std::array<float, 3>> &positions,
    const std::vector<std::array<T, NumComponents>> &values,
    DataType data_type) {
  PointCloudBuilder builder;
  builder.Start(static_cast<PointIndex::ValueType>(positions.size()));
  const int position_id =
      builder.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT32);
  const int value_id =
      builder.AddAttribute(GeometryAttribute::COLOR, NumComponents, data_type);
  for (PointIndex i(0); i < static_cast<uint32_t>(positions.size()); ++i) {
    builder.SetAttributeValueForPoint(position_id, i,
                                      positions[i.value()].data());
    builder.SetAttributeValueForPoint(value_id, i, values[i.value()].data());
  }
  return builder.Finalize(false);
}

std::vector<std::array<float, 3>> MakeLinearPositions(size_t num_points) {
  std::vector<std::array<float, 3>> positions(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    positions[i] = {{static_cast<float>(i), 0.0f, 0.0f}};
  }
  return positions;
}

template <typename T, size_t NumComponents>
void ExpectValuesNear(
    const std::vector<double> &decoded,
    const std::vector<std::array<T, NumComponents>> &original,
    double tolerance) {
  ASSERT_EQ(decoded.size(), original.size() * NumComponents);
  for (size_t i = 0; i < original.size(); ++i) {
    for (size_t c = 0; c < NumComponents; ++c) {
      EXPECT_NEAR(decoded[i * NumComponents + c],
                  static_cast<double>(original[i][c]), tolerance)
          << "point " << i << " component " << c;
    }
  }
}

bool DecodeBlockwise(const EncodedSpatialBlockAttribute &encoded,
                     std::vector<double>* values) {
  values->assign(static_cast<size_t>(encoded.num_points) *
                     encoded.num_components,
                 0.0);
  std::vector<uint8_t> covered(encoded.num_points, 0);
  for (uint32_t block_index = 0;
       block_index < static_cast<uint32_t>(encoded.blocks.size());
       ++block_index) {
    std::vector<double> block_values;
    std::vector<uint32_t> point_indices;
    const Status status = SpatialBlockAttributeCodec::DecodeBlock(
        encoded, block_index, &block_values, &point_indices);
    if (!status.ok()) {
      return false;
    }
    if (block_values.size() !=
        static_cast<size_t>(point_indices.size()) *
            encoded.num_components) {
      return false;
    }
    for (size_t i = 0; i < point_indices.size(); ++i) {
      if (point_indices[i] >= encoded.num_points || covered[point_indices[i]]) {
        return false;
      }
      covered[point_indices[i]] = 1;
      for (uint8_t c = 0; c < encoded.num_components; ++c) {
        (*values)[static_cast<size_t>(point_indices[i]) *
                      encoded.num_components +
                  c] = block_values[i * encoded.num_components + c];
      }
    }
  }
  for (uint8_t value : covered) {
    if (value == 0) {
      return false;
    }
  }
  return true;
}

TEST(SpatialBlockAttributeCodecTest, ExactRoundTripAndBlockwiseDecode) {
  const std::vector<std::array<float, 3>> positions = MakeLinearPositions(16);
  std::vector<std::array<uint8_t, 3>> colors(16);
  for (size_t i = 0; i < colors.size(); ++i) {
    colors[i] = {{static_cast<uint8_t>(i * 3),
                  static_cast<uint8_t>(255 - i * 7),
                  static_cast<uint8_t>((i * 11) & 0xff)}};
  }
  const std::unique_ptr<PointCloud> cloud =
      MakePointCloud(positions, colors, DT_UINT8);
  ASSERT_NE(cloud, nullptr);

  SpatialBlockCodecOptions options;
  options.max_points_per_block = 4;
  options.max_reconstruction_error = 0.0;
  EncodedSpatialBlockAttribute encoded;
  const Status status =
      SpatialBlockAttributeCodec::Encode(*cloud, 1, options, &encoded);
  ASSERT_TRUE(status.ok()) << status;
  ASSERT_EQ(encoded.blocks.size(), 4u);
  ASSERT_FALSE(encoded.data.empty());

  std::vector<double> whole;
  ASSERT_TRUE(SpatialBlockAttributeCodec::Decode(encoded, &whole).ok());
  ExpectValuesNear(whole, colors, 0.0);

  std::vector<double> blockwise;
  ASSERT_TRUE(DecodeBlockwise(encoded, &blockwise));
  EXPECT_EQ(whole, blockwise);

  for (const SpatialBlockResidualStats &block : encoded.blocks) {
    EXPECT_EQ(block.num_points, 4u);
    EXPECT_EQ(block.residual_count, 4u * 3u);
    EXPECT_EQ(block.max_abs_reconstruction_error, 0.0);
    EXPECT_LE(block.rms_residual, block.max_abs_residual);
    EXPECT_LE(block.min_position[0], block.max_position[0]);
  }
  EXPECT_FALSE(encoded.blocks[0].has_previous_context);
  for (size_t i = 1; i < encoded.blocks.size(); ++i) {
    EXPECT_TRUE(encoded.blocks[i].has_previous_context);
  }
}

TEST(SpatialBlockAttributeCodecTest, ThresholdAndBlockwiseEquality) {
  std::vector<std::array<float, 3>> positions;
  std::vector<std::array<float, 2>> values;
  for (int i = 0; i < 37; ++i) {
    positions.push_back(
        {{static_cast<float>(i % 7) * 0.25f,
          static_cast<float>(i / 7) * 0.5f,
          static_cast<float>((i * 13) % 5) * 0.125f}});
    values.push_back({{0.75f * i + 0.125f,
                       -0.5f * i + 0.25f * ((i * 17) % 3)}});
  }
  const std::unique_ptr<PointCloud> cloud = MakePointCloud(
      positions, values, DT_FLOAT32);
  ASSERT_NE(cloud, nullptr);

  SpatialBlockCodecOptions options;
  options.max_points_per_block = 7;
  options.max_reconstruction_error = 0.25;
  EncodedSpatialBlockAttribute encoded;
  const Status status =
      SpatialBlockAttributeCodec::Encode(*cloud, 1, options, &encoded);
  ASSERT_TRUE(status.ok()) << status;
  ASSERT_EQ(encoded.blocks.size(), 6u);

  std::vector<double> whole;
  ASSERT_TRUE(SpatialBlockAttributeCodec::Decode(encoded, &whole).ok());
  for (size_t i = 0; i < values.size(); ++i) {
    for (int c = 0; c < 2; ++c) {
      EXPECT_LE(std::fabs(whole[i * 2 + c] -
                          static_cast<double>(values[i][c])),
                options.max_reconstruction_error);
    }
  }

  std::vector<double> blockwise;
  ASSERT_TRUE(DecodeBlockwise(encoded, &blockwise));
  EXPECT_EQ(whole, blockwise);

  for (const SpatialBlockResidualStats &block : encoded.blocks) {
    EXPECT_LE(block.max_abs_reconstruction_error,
              options.max_reconstruction_error);
  }

  options.max_reconstruction_error = 0.0;
  ASSERT_TRUE(
      SpatialBlockAttributeCodec::Encode(*cloud, 1, options, &encoded).ok());
  ASSERT_TRUE(SpatialBlockAttributeCodec::Decode(encoded, &whole).ok());
  ExpectValuesNear(whole, values, 0.0);
}

TEST(SpatialBlockAttributeCodecTest, LinearBoundaryHasZeroSeamResidual) {
  const std::vector<std::array<float, 3>> positions = MakeLinearPositions(16);
  std::vector<std::array<float, 1>> values(16);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i][0] = static_cast<float>(10.0 + 2.0 * i);
  }
  const std::unique_ptr<PointCloud> cloud =
      MakePointCloud(positions, values, DT_FLOAT32);
  ASSERT_NE(cloud, nullptr);

  SpatialBlockCodecOptions options;
  options.max_points_per_block = 4;
  options.max_reconstruction_error = 0.0;
  options.predictors = {SpatialBlockPredictor::kLinear};
  EncodedSpatialBlockAttribute encoded;
  const Status status =
      SpatialBlockAttributeCodec::Encode(*cloud, 1, options, &encoded);
  ASSERT_TRUE(status.ok()) << status;
  ASSERT_EQ(encoded.blocks.size(), 4u);

  for (const SpatialBlockResidualStats &block : encoded.blocks) {
    EXPECT_EQ(block.predictor, SpatialBlockPredictor::kLinear);
  }
  EXPECT_DOUBLE_EQ(encoded.blocks[0].sum_abs_residual, 12.0);
  EXPECT_DOUBLE_EQ(encoded.blocks[0].max_abs_residual, 10.0);
  EXPECT_DOUBLE_EQ(encoded.blocks[0].rms_residual, std::sqrt(26.0));
  for (size_t i = 1; i < encoded.blocks.size(); ++i) {
    EXPECT_TRUE(encoded.blocks[i].has_previous_context);
    EXPECT_DOUBLE_EQ(encoded.blocks[i].boundary_abs_residual, 0.0);
  }

  std::vector<double> whole;
  ASSERT_TRUE(SpatialBlockAttributeCodec::Decode(encoded, &whole).ok());
  std::vector<double> blockwise;
  ASSERT_TRUE(DecodeBlockwise(encoded, &blockwise));
  EXPECT_EQ(whole, blockwise);
  ExpectValuesNear(whole, values, 0.0);
}

TEST(SpatialBlockAttributeCodecTest, EdgeBlockShapesAndDuplicatePositions) {
  std::vector<std::array<float, 3>> positions(9, {{0.0f, 0.0f, 0.0f}});
  std::vector<std::array<uint16_t, 2>> values(9);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = {{static_cast<uint16_t>(i + 100),
                  static_cast<uint16_t>(1000 - i)}};
  }
  const std::unique_ptr<PointCloud> cloud =
      MakePointCloud(positions, values, DT_UINT16);
  ASSERT_NE(cloud, nullptr);

  SpatialBlockCodecOptions options;
  options.max_points_per_block = 1;
  options.max_reconstruction_error = 0.0;
  EncodedSpatialBlockAttribute encoded;
  ASSERT_TRUE(
      SpatialBlockAttributeCodec::Encode(*cloud, 1, options, &encoded).ok());
  ASSERT_EQ(encoded.blocks.size(), 9u);
  for (size_t i = 0; i < encoded.blocks.size(); ++i) {
    EXPECT_EQ(encoded.blocks[i].first_point_index, i);
    EXPECT_EQ(encoded.blocks[i].num_points, 1u);
  }
  std::vector<double> decoded;
  ASSERT_TRUE(SpatialBlockAttributeCodec::Decode(encoded, &decoded).ok());
  ExpectValuesNear(decoded, values, 0.0);

  options.max_points_per_block = 4;
  ASSERT_TRUE(
      SpatialBlockAttributeCodec::Encode(*cloud, 1, options, &encoded).ok());
  ASSERT_EQ(encoded.blocks.size(), 3u);
  EXPECT_EQ(encoded.blocks[2].num_points, 1u);
  ASSERT_TRUE(SpatialBlockAttributeCodec::Decode(encoded, &decoded).ok());
  ExpectValuesNear(decoded, values, 0.0);
}

TEST(SpatialBlockAttributeCodecTest, SinglePointAndExactBlockMultiple) {
  std::vector<std::array<float, 3>> positions = MakeLinearPositions(1);
  std::vector<std::array<int32_t, 1>> values(1);
  values[0][0] = -7;
  std::unique_ptr<PointCloud> cloud =
      MakePointCloud(positions, values, DT_INT32);
  ASSERT_NE(cloud, nullptr);

  SpatialBlockCodecOptions options;
  options.max_points_per_block = 32;
  EncodedSpatialBlockAttribute encoded;
  ASSERT_TRUE(
      SpatialBlockAttributeCodec::Encode(*cloud, 1, options, &encoded).ok());
  ASSERT_EQ(encoded.blocks.size(), 1u);
  std::vector<double> decoded;
  ASSERT_TRUE(SpatialBlockAttributeCodec::Decode(encoded, &decoded).ok());
  ExpectValuesNear(decoded, values, 0.0);

  positions = MakeLinearPositions(8);
  std::vector<std::array<int32_t, 1>> eight_values(8);
  for (int i = 0; i < 8; ++i) {
    eight_values[i][0] = i * 5 - 3;
  }
  cloud = MakePointCloud(positions, eight_values, DT_INT32);
  ASSERT_NE(cloud, nullptr);
  options.max_points_per_block = 4;
  ASSERT_TRUE(
      SpatialBlockAttributeCodec::Encode(*cloud, 1, options, &encoded).ok());
  EXPECT_EQ(encoded.blocks.size(), 2u);
  ASSERT_TRUE(SpatialBlockAttributeCodec::Decode(encoded, &decoded).ok());
  ExpectValuesNear(decoded, eight_values, 0.0);
}

TEST(SpatialBlockAttributeCodecTest, RejectsInvalidInputsAndPayloads) {
  const std::vector<std::array<float, 3>> positions = MakeLinearPositions(4);
  std::vector<std::array<uint8_t, 1>> values(4);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i][0] = static_cast<uint8_t>(i);
  }
  const std::unique_ptr<PointCloud> cloud =
      MakePointCloud(positions, values, DT_UINT8);
  ASSERT_NE(cloud, nullptr);

  EncodedSpatialBlockAttribute encoded;
  SpatialBlockCodecOptions options;
  options.max_points_per_block = 0;
  EXPECT_FALSE(
      SpatialBlockAttributeCodec::Encode(*cloud, 1, options, &encoded).ok());
  options.max_points_per_block = 4;
  options.predictors.clear();
  EXPECT_FALSE(
      SpatialBlockAttributeCodec::Encode(*cloud, 1, options, &encoded).ok());
  options.predictors = {SpatialBlockPredictor::kPrevious};
  options.max_reconstruction_error = -1.0;
  EXPECT_FALSE(
      SpatialBlockAttributeCodec::Encode(*cloud, 1, options, &encoded).ok());
  options.max_reconstruction_error = 0.0;
  EXPECT_FALSE(
      SpatialBlockAttributeCodec::Encode(*cloud, -1, options, &encoded).ok());

  PointCloudBuilder builder;
  builder.Start(4);
  const int generic_id =
      builder.AddAttribute(GeometryAttribute::GENERIC, 1, DT_FLOAT32);
  const std::array<float, 1> value = {{1.0f}};
  for (PointIndex i(0); i < 4; ++i) {
    builder.SetAttributeValueForPoint(generic_id, i, value.data());
  }
  const std::unique_ptr<PointCloud> no_position = builder.Finalize(false);
  ASSERT_NE(no_position, nullptr);
  EXPECT_FALSE(
      SpatialBlockAttributeCodec::Encode(*no_position, 0, options, &encoded)
          .ok());

  ASSERT_TRUE(
      SpatialBlockAttributeCodec::Encode(*cloud, 1, options, &encoded).ok());
  std::vector<double> decoded;
  EXPECT_FALSE(
      SpatialBlockAttributeCodec::DecodeBlock(encoded, 99, &decoded).ok());
  EXPECT_FALSE(SpatialBlockAttributeCodec::Decode(nullptr, 0, &decoded).ok());

  std::vector<uint8_t> truncated(encoded.data.begin(), encoded.data.end() - 1);
  EXPECT_FALSE(
      SpatialBlockAttributeCodec::Decode(truncated.data(), truncated.size(),
                                         &decoded)
          .ok());
  std::vector<uint8_t> trailing = encoded.data;
  trailing.push_back(0);
  EXPECT_FALSE(
      SpatialBlockAttributeCodec::Decode(trailing.data(), trailing.size(),
                                         &decoded)
          .ok());
  std::vector<uint8_t> huge_point_count = encoded.data;
  const uint32_t bad_point_count = 0xffffffffu;
  ASSERT_GE(huge_point_count.size(), 16u);
  std::memcpy(huge_point_count.data() + 12, &bad_point_count,
              sizeof(bad_point_count));
  EXPECT_FALSE(
      SpatialBlockAttributeCodec::Decode(huge_point_count.data(),
                                         huge_point_count.size(), &decoded)
          .ok());

  PointCloudBuilder empty_builder;
  empty_builder.Start(0);
  empty_builder.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT32);
  empty_builder.AddAttribute(GeometryAttribute::COLOR, 1, DT_UINT8);
  const std::unique_ptr<PointCloud> empty = empty_builder.Finalize(false);
  ASSERT_NE(empty, nullptr);
  EXPECT_FALSE(
      SpatialBlockAttributeCodec::Encode(*empty, 1, options, &encoded).ok());
}

}  // namespace
}  // namespace draco

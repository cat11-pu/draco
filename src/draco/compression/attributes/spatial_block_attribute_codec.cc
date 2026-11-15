#include "draco/compression/attributes/spatial_block_attribute_codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "draco/core/decoder_buffer.h"
#include "draco/core/encoder_buffer.h"
#include "draco/core/varint_decoding.h"
#include "draco/core/varint_encoding.h"

namespace draco {
namespace {

constexpr char kMagic[8] = {'D', 'R', 'B', 'L', 'K', 'A', 'T', '1'};
constexpr uint32_t kVersion = 1;
constexpr uint8_t kHasPrevious = 1;
constexpr uint8_t kHasPreviousTwo = 2;

struct ParsedHeader {
  uint32_t num_points = 0;
  uint32_t num_blocks = 0;
  uint8_t num_components = 0;
  uint8_t data_type = 0;
  bool exact = false;
  double quantization_step = 0.0;
};

struct DecodeState {
  bool has_prev = false;
  bool has_prev2 = false;
  std::vector<double> prev;
  std::vector<double> prev2;
};

struct BlockData {
  std::vector<uint32_t> point_indices;
  std::vector<double> values;
  bool has_prev = false;
  bool has_prev2 = false;
  std::vector<double> seed_prev;
  std::vector<double> seed_prev2;
  DecodeState end_state;
};

struct CandidateResult {
  SpatialBlockPredictor predictor = SpatialBlockPredictor::kPrevious;
  double cost = 0.0;
  std::vector<int64_t> quantized;
  SpatialBlockResidualStats stats;
  DecodeState end_state;
};

bool IsFinite(double value) { return std::isfinite(value); }

bool IsValidDataType(uint8_t data_type) {
  return data_type > DT_INVALID && data_type < DT_TYPES_COUNT;
}

bool GetValueCount(const ParsedHeader &header, size_t* value_count) {
  if (header.num_components == 0) {
    return false;
  }
  if (static_cast<size_t>(header.num_points) >
      std::numeric_limits<size_t>::max() / header.num_components) {
    return false;
  }
  *value_count =
      static_cast<size_t>(header.num_points) * header.num_components;
  return true;
}

Status Invalid(const std::string &message) {
  return Status(Status::INVALID_PARAMETER, message);
}

Status DecodeError(const std::string &message) {
  return Status(Status::DRACO_ERROR, message);
}

bool EncodeDouble(double value, EncoderBuffer* buffer) {
  return buffer->Encode(value);
}

bool DecodeDouble(double* value, DecoderBuffer* buffer) {
  if (!buffer->Decode(value) || !IsFinite(*value)) {
    return false;
  }
  return true;
}

bool EncodeHeader(const EncodedSpatialBlockAttribute &encoded,
                  uint32_t num_blocks, EncoderBuffer* buffer) {
  if (!buffer->Encode(kMagic)) {
    return false;
  }
  uint8_t exact = encoded.exact ? 1 : 0;
  const uint8_t data_type = static_cast<uint8_t>(encoded.data_type);
  if (!buffer->Encode(kVersion) || !buffer->Encode(encoded.num_points) ||
      !buffer->Encode(encoded.num_components) || !buffer->Encode(data_type) ||
      !buffer->Encode(exact) || !buffer->Encode(encoded.quantization_step) ||
      !buffer->Encode(num_blocks)) {
    return false;
  }
  return true;
}

bool ReadHeader(DecoderBuffer* buffer, ParsedHeader* header) {
  char magic[sizeof(kMagic)];
  if (!buffer->Decode(magic, sizeof(magic)) ||
      std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    return false;
  }

  uint32_t version = 0;
  uint8_t exact = 0;
  if (!buffer->Decode(&version) || version != kVersion ||
      !buffer->Decode(&header->num_points) ||
      !buffer->Decode(&header->num_components) ||
      !buffer->Decode(&header->data_type) || !buffer->Decode(&exact) ||
      !buffer->Decode(&header->quantization_step) ||
      !buffer->Decode(&header->num_blocks)) {
    return false;
  }
  if (header->num_components == 0 || !IsValidDataType(header->data_type) ||
      exact > 1) {
    return false;
  }
  header->exact = exact != 0;
  if (header->exact) {
    if (header->quantization_step != 0.0) {
      return false;
    }
  } else if (!IsFinite(header->quantization_step) ||
             header->quantization_step <= 0.0) {
    return false;
  }
  if (header->num_points == 0) {
    return header->num_blocks == 0;
  }
  if (header->num_blocks == 0 || header->num_blocks > header->num_points) {
    return false;
  }
  return true;
}

void PredictValues(SpatialBlockPredictor predictor, int num_components,
                   const DecodeState &state, double* prediction) {
  if (!state.has_prev) {
    std::fill(prediction, prediction + num_components, 0.0);
    return;
  }
  if (predictor == SpatialBlockPredictor::kLinear && state.has_prev2) {
    for (int c = 0; c < num_components; ++c) {
      prediction[c] = 2.0 * state.prev[c] - state.prev2[c];
    }
    return;
  }
  std::copy(state.prev.begin(), state.prev.end(), prediction);
}

bool QuantizeResidual(double residual, double step, int64_t* quantized) {
  const double scaled = residual / step;
  if (!IsFinite(scaled)) {
    return false;
  }
  const long double rounded =
      std::round(static_cast<long double>(scaled));
  const long double min_value =
      static_cast<long double>(std::numeric_limits<int64_t>::min());
  const long double max_value =
      static_cast<long double>(std::numeric_limits<int64_t>::max());
  if (rounded < min_value || rounded > max_value) {
    return false;
  }
  *quantized = static_cast<int64_t>(rounded);
  return true;
}

bool UpdateState(const double* reconstructed, int num_components,
                 DecodeState* state) {
  if (state->has_prev) {
    state->prev2 = state->prev;
    state->has_prev2 = true;
  }
  state->prev.assign(reconstructed, reconstructed + num_components);
  state->has_prev = true;
  return true;
}

Status EvaluateCandidate(
    SpatialBlockPredictor predictor, const std::vector<double> &block_values,
    uint32_t num_points, int num_components, bool exact, double step,
    const DecodeState &initial_state, CandidateResult* result) {
  result->predictor = predictor;
  result->quantized.clear();
  result->stats = SpatialBlockResidualStats();
  result->stats.predictor = predictor;
  result->stats.num_points = num_points;
  result->stats.has_previous_context = initial_state.has_prev;
  result->stats.residual_count =
      static_cast<uint64_t>(num_points) * num_components;

  if (!exact) {
    result->quantized.resize(static_cast<size_t>(num_points) * num_components,
                             0);
  }

  DecodeState state = initial_state;
  std::vector<double> prediction(num_components);
  std::vector<double> reconstructed(num_components);
  double sum_abs_residual = 0.0;
  double sum_squared_residual = 0.0;
  double max_abs_residual = 0.0;
  double max_abs_error = 0.0;
  double boundary_abs_residual = 0.0;
  double cost = 0.0;

  for (uint32_t i = 0; i < num_points; ++i) {
    PredictValues(predictor, num_components, state, prediction.data());
    for (int c = 0; c < num_components; ++c) {
      const double original =
          block_values[static_cast<size_t>(i) * num_components + c];
      const double residual = original - prediction[c];
      if (!IsFinite(residual)) {
        return DecodeError("Non-finite prediction residual.");
      }
      const double abs_residual = std::fabs(residual);
      sum_abs_residual += abs_residual;
      sum_squared_residual += residual * residual;
      max_abs_residual = std::max(max_abs_residual, abs_residual);
      if (i == 0) {
        boundary_abs_residual =
            std::max(boundary_abs_residual, abs_residual);
      }

      if (exact) {
        reconstructed[c] = original;
        cost += abs_residual;
      } else {
        const size_t offset =
            static_cast<size_t>(i) * num_components + c;
        int64_t quantized = 0;
        if (!QuantizeResidual(residual, step, &quantized)) {
          return DecodeError("Residual is outside the supported range.");
        }
        result->quantized[offset] = quantized;
        reconstructed[c] =
            prediction[c] + static_cast<double>(quantized) * step;
        if (!IsFinite(reconstructed[c])) {
          return DecodeError("Non-finite reconstructed value.");
        }
        cost += std::fabs(static_cast<double>(quantized));
      }
      max_abs_error =
          std::max(max_abs_error, std::fabs(reconstructed[c] - original));
    }
    UpdateState(reconstructed.data(), num_components, &state);
  }

  if (!IsFinite(sum_abs_residual) || !IsFinite(sum_squared_residual) ||
      !IsFinite(cost)) {
    return DecodeError("Residual statistics overflowed.");
  }
  if (result->stats.residual_count > 0) {
    result->stats.rms_residual =
        std::sqrt(sum_squared_residual /
                  static_cast<double>(result->stats.residual_count));
  }
  result->stats.sum_abs_residual = sum_abs_residual;
  result->stats.max_abs_residual = max_abs_residual;
  result->stats.max_abs_reconstruction_error = max_abs_error;
  result->stats.boundary_abs_residual = boundary_abs_residual;
  result->stats.predictor = predictor;
  result->cost = cost;
  result->end_state = std::move(state);
  return OkStatus();
}

Status ReadBlockHeader(DecoderBuffer* buffer, const ParsedHeader &header,
                       uint32_t expected_block_index,
                       SpatialBlockPredictor* predictor,
                       uint32_t* num_points, uint8_t* flags,
                       std::vector<double>* seed_prev,
                       std::vector<double>* seed_prev2) {
  uint8_t predictor_byte = 0;
  if (!buffer->Decode(&predictor_byte) || !buffer->Decode(num_points) ||
      !buffer->Decode(flags)) {
    return DecodeError("Truncated block header.");
  }
  if (predictor_byte > static_cast<uint8_t>(SpatialBlockPredictor::kLinear)) {
    return DecodeError("Invalid block predictor.");
  }
  *predictor = static_cast<SpatialBlockPredictor>(predictor_byte);
  if (*num_points == 0 || *num_points > header.num_points) {
    return DecodeError("Invalid block point count.");
  }
  if (*flags & ~(kHasPrevious | kHasPreviousTwo)) {
    return DecodeError("Invalid block context flags.");
  }
  if ((*flags & kHasPreviousTwo) && !(*flags & kHasPrevious)) {
    return DecodeError("Second previous seed without previous seed.");
  }
  if (expected_block_index == 0 && *flags != 0) {
    return DecodeError("The first block must not have previous context.");
  }
  if (expected_block_index > 0 && !(*flags & kHasPrevious)) {
    return DecodeError("A non-first block must continue the previous context.");
  }

  seed_prev->clear();
  seed_prev2->clear();
  if (*flags & kHasPrevious) {
    seed_prev->resize(header.num_components);
    for (uint8_t c = 0; c < header.num_components; ++c) {
      if (!DecodeDouble(&(*seed_prev)[c], buffer)) {
        return DecodeError("Invalid previous sample seed.");
      }
    }
  }
  if (*flags & kHasPreviousTwo) {
    seed_prev2->resize(header.num_components);
    for (uint8_t c = 0; c < header.num_components; ++c) {
      if (!DecodeDouble(&(*seed_prev2)[c], buffer)) {
        return DecodeError("Invalid second previous sample seed.");
      }
    }
  }
  return OkStatus();
}

Status ReadBlock(DecoderBuffer* buffer, const ParsedHeader &header,
                 uint32_t block_index, std::vector<uint8_t>* seen_points,
                 BlockData* block) {
  SpatialBlockPredictor predictor = SpatialBlockPredictor::kPrevious;
  uint32_t num_points = 0;
  uint8_t flags = 0;
  Status status = ReadBlockHeader(buffer, header, block_index, &predictor,
                                  &num_points, &flags, &block->seed_prev,
                                  &block->seed_prev2);
  if (!status.ok()) {
    return status;
  }

  block->has_prev = (flags & kHasPrevious) != 0;
  block->has_prev2 = (flags & kHasPreviousTwo) != 0;
  block->point_indices.resize(num_points);
  block->values.resize(static_cast<size_t>(num_points) *
                       header.num_components);

  std::vector<int64_t> quantized;
  if (!header.exact) {
    quantized.resize(static_cast<size_t>(num_points) * header.num_components);
  }

  for (uint32_t i = 0; i < num_points; ++i) {
    uint32_t point_index = 0;
    if (!DecodeVarint<uint32_t>(&point_index, buffer) ||
        point_index >= header.num_points) {
      return DecodeError("Invalid point index.");
    }
    if ((*seen_points)[point_index]) {
      return DecodeError("Duplicate point index.");
    }
    (*seen_points)[point_index] = 1;
    block->point_indices[i] = point_index;

    for (uint8_t c = 0; c < header.num_components; ++c) {
      const size_t offset = static_cast<size_t>(i) * header.num_components + c;
      if (header.exact) {
        if (!DecodeDouble(&block->values[offset], buffer)) {
          return DecodeError("Invalid exact value.");
        }
      } else if (!DecodeVarint<int64_t>(&quantized[offset], buffer)) {
        return DecodeError("Invalid quantized residual.");
      }
    }
  }

  DecodeState state;
  state.has_prev = block->has_prev;
  state.has_prev2 = block->has_prev2;
  state.prev = block->seed_prev;
  state.prev2 = block->seed_prev2;

  std::vector<double> prediction(header.num_components);
  std::vector<double> reconstructed(header.num_components);
  for (uint32_t i = 0; i < num_points; ++i) {
    PredictValues(predictor, header.num_components, state, prediction.data());
    for (uint8_t c = 0; c < header.num_components; ++c) {
      const size_t offset = static_cast<size_t>(i) * header.num_components + c;
      if (header.exact) {
        reconstructed[c] = block->values[offset];
      } else {
        reconstructed[c] =
            prediction[c] +
            static_cast<double>(quantized[offset]) *
                header.quantization_step;
        block->values[offset] = reconstructed[c];
      }
      if (!IsFinite(reconstructed[c])) {
        return DecodeError("Non-finite reconstructed value.");
      }
    }
    UpdateState(reconstructed.data(), header.num_components, &state);
  }
  block->end_state = std::move(state);
  return OkStatus();
}

bool SeedsEqual(const std::vector<double> &left,
                const std::vector<double> &right) {
  return left == right;
}

}  // namespace

const char* GetSpatialBlockPredictorName(SpatialBlockPredictor predictor) {
  switch (predictor) {
    case SpatialBlockPredictor::kPrevious:
      return "previous";
    case SpatialBlockPredictor::kLinear:
      return "linear";
  }
  return "unknown";
}

Status SpatialBlockAttributeCodec::Encode(
    const PointCloud &point_cloud, int attribute_id,
    const SpatialBlockCodecOptions &options,
    EncodedSpatialBlockAttribute* encoded) {
  if (encoded == nullptr) {
    return Invalid("encoded must not be null.");
  }
  if (attribute_id < 0 || attribute_id >= point_cloud.num_attributes()) {
    return Invalid("Invalid attribute id.");
  }
  if (options.max_points_per_block == 0) {
    return Invalid("max_points_per_block must be greater than zero.");
  }
  if (!IsFinite(options.max_reconstruction_error) ||
      options.max_reconstruction_error < 0.0) {
    return Invalid("max_reconstruction_error must be finite and non-negative.");
  }
  if (options.predictors.empty()) {
    return Invalid("At least one predictor must be provided.");
  }
  for (SpatialBlockPredictor predictor : options.predictors) {
    if (predictor != SpatialBlockPredictor::kPrevious &&
        predictor != SpatialBlockPredictor::kLinear) {
      return Invalid("Unsupported block predictor.");
    }
  }

  const PointAttribute* position_attribute =
      point_cloud.GetNamedAttribute(GeometryAttribute::POSITION);
  if (position_attribute == nullptr ||
      position_attribute->num_components() < 3) {
    return Invalid("A three-component POSITION attribute is required.");
  }
  const PointAttribute* attribute = point_cloud.attribute(attribute_id);
  const int num_components = attribute->num_components();
  if (num_components <= 0 ||
      num_components > std::numeric_limits<uint8_t>::max()) {
    return Invalid("Unsupported number of attribute components.");
  }
  if (!IsValidDataType(static_cast<uint8_t>(attribute->data_type()))) {
    return Invalid("Unsupported attribute data type.");
  }
  if (attribute->data_type() == DT_INT64 ||
      attribute->data_type() == DT_UINT64) {
    return Invalid(
        "64-bit integer attributes cannot be represented by the double-valued "
        "interface.");
  }
  if (point_cloud.num_points() == 0) {
    return Invalid("The point cloud must contain at least one point.");
  }
  if (point_cloud.num_points() >
      std::numeric_limits<uint32_t>::max() / num_components) {
    return Invalid("The point cloud is too large.");
  }

  const uint32_t num_points = point_cloud.num_points();
  std::vector<std::array<double, 3>> positions(num_points);
  std::vector<double> attribute_values(
      static_cast<size_t>(num_points) * num_components);
  for (uint32_t i = 0; i < num_points; ++i) {
    const PointIndex point_index(i);
    if (!position_attribute->ConvertValue<double>(
            position_attribute->mapped_index(point_index), 3,
            positions[i].data())) {
      return Invalid("Failed to read a POSITION value.");
    }
    if (!attribute->ConvertValue<double>(attribute->mapped_index(point_index),
                                         num_components,
                                         &attribute_values[static_cast<size_t>(
                                             i) * num_components])) {
      return Invalid("Failed to read an attribute value.");
    }
    for (double value : positions[i]) {
      if (!IsFinite(value)) {
        return Invalid("POSITION values must be finite.");
      }
    }
    for (int c = 0; c < num_components; ++c) {
      if (!IsFinite(attribute_values[static_cast<size_t>(i) * num_components +
                                     c])) {
        return Invalid("Attribute values must be finite.");
      }
    }
  }

  std::vector<uint32_t> order(num_points);
  for (uint32_t i = 0; i < num_points; ++i) {
    order[i] = i;
  }
  std::stable_sort(order.begin(), order.end(),
                   [&positions](uint32_t left, uint32_t right) {
                     for (int c = 0; c < 3; ++c) {
                       if (positions[left][c] < positions[right][c]) {
                         return true;
                       }
                       if (positions[left][c] > positions[right][c]) {
                         return false;
                       }
                     }
                     return left < right;
                   });

  const bool exact = options.max_reconstruction_error == 0.0;
  const double step = exact ? 0.0 : options.max_reconstruction_error;
  const uint64_t num_blocks_64 =
      (static_cast<uint64_t>(num_points) + options.max_points_per_block - 1) /
      options.max_points_per_block;
  if (num_blocks_64 > std::numeric_limits<uint32_t>::max()) {
    return Invalid("Too many spatial blocks.");
  }
  const uint32_t num_blocks = static_cast<uint32_t>(num_blocks_64);

  EncoderBuffer buffer;
  encoded->num_points = num_points;
  encoded->num_components = static_cast<uint8_t>(num_components);
  encoded->data_type = attribute->data_type();
  encoded->normalized = attribute->normalized();
  encoded->max_reconstruction_error = options.max_reconstruction_error;
  encoded->quantization_step = step;
  encoded->exact = exact;
  encoded->blocks.clear();
  encoded->blocks.reserve(num_blocks);

  if (!EncodeHeader(*encoded, num_blocks, &buffer)) {
    return DecodeError("Failed to write header.");
  }

  DecodeState state;
  state.prev.resize(num_components);
  state.prev2.resize(num_components);
  for (uint32_t block_index = 0; block_index < num_blocks; ++block_index) {
    const uint32_t begin = block_index * options.max_points_per_block;
    const uint32_t end =
        std::min(num_points, begin + options.max_points_per_block);
    const uint32_t block_size = end - begin;

    std::vector<uint32_t> point_indices(block_size);
    std::vector<double> block_values(
        static_cast<size_t>(block_size) * num_components);
    std::array<double, 3> min_position = positions[order[begin]];
    std::array<double, 3> max_position = min_position;
    for (uint32_t i = 0; i < block_size; ++i) {
      const uint32_t point_index = order[begin + i];
      point_indices[i] = point_index;
      for (int c = 0; c < 3; ++c) {
        min_position[c] = std::min(min_position[c], positions[point_index][c]);
        max_position[c] = std::max(max_position[c], positions[point_index][c]);
      }
      for (int c = 0; c < num_components; ++c) {
        block_values[static_cast<size_t>(i) * num_components + c] =
            attribute_values[static_cast<size_t>(point_index) * num_components +
                             c];
      }
    }

    CandidateResult best;
    bool have_best = false;
    for (SpatialBlockPredictor predictor : options.predictors) {
      CandidateResult candidate;
      const Status status =
          EvaluateCandidate(predictor, block_values, block_size, num_components,
                            exact, step, state, &candidate);
      if (!status.ok()) {
        return status;
      }
      if (!have_best || candidate.cost < best.cost) {
        best = std::move(candidate);
        have_best = true;
      }
    }

    best.stats.block_index = block_index;
    best.stats.first_point_index = point_indices.front();
    best.stats.last_point_index = point_indices.back();
    best.stats.min_position = min_position;
    best.stats.max_position = max_position;

    const uint8_t predictor_byte = static_cast<uint8_t>(best.predictor);
    uint8_t flags = 0;
    if (state.has_prev) {
      flags |= kHasPrevious;
    }
    if (state.has_prev2) {
      flags |= kHasPreviousTwo;
    }
    if (!buffer.Encode(predictor_byte) || !buffer.Encode(block_size) ||
        !buffer.Encode(flags)) {
      return DecodeError("Failed to write block header.");
    }
    if (state.has_prev) {
      for (int c = 0; c < num_components; ++c) {
        if (!EncodeDouble(state.prev[c], &buffer)) {
          return DecodeError("Failed to write previous sample seed.");
        }
      }
    }
    if (state.has_prev2) {
      for (int c = 0; c < num_components; ++c) {
        if (!EncodeDouble(state.prev2[c], &buffer)) {
          return DecodeError("Failed to write second previous sample seed.");
        }
      }
    }

    for (uint32_t i = 0; i < block_size; ++i) {
      if (!EncodeVarint<uint32_t>(point_indices[i], &buffer)) {
        return DecodeError("Failed to write point index.");
      }
      if (exact) {
        for (int c = 0; c < num_components; ++c) {
          const size_t offset =
              static_cast<size_t>(i) * num_components + c;
          if (!EncodeDouble(block_values[offset], &buffer)) {
            return DecodeError("Failed to write exact value.");
          }
        }
      } else {
        for (int c = 0; c < num_components; ++c) {
          const size_t offset =
              static_cast<size_t>(i) * num_components + c;
          if (!EncodeVarint<int64_t>(best.quantized[offset], &buffer)) {
            return DecodeError("Failed to write quantized residual.");
          }
        }
      }
    }

    if (best.stats.max_abs_reconstruction_error >
        options.max_reconstruction_error) {
      return DecodeError("Block reconstruction error exceeds the threshold.");
    }
    encoded->blocks.push_back(best.stats);
    state = std::move(best.end_state);
  }

  encoded->data.assign(buffer.data(), buffer.data() + buffer.size());
  return OkStatus();
}

Status SpatialBlockAttributeCodec::Decode(
    const EncodedSpatialBlockAttribute &encoded,
    std::vector<double>* values) {
  return Decode(encoded.data.data(), encoded.data.size(), values);
}

Status SpatialBlockAttributeCodec::Decode(const uint8_t* data,
                                          size_t data_size,
                                          std::vector<double>* values) {
  if (data == nullptr || values == nullptr) {
    return Invalid("data and values must not be null.");
  }
  if (data_size == 0 ||
      data_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return Invalid("Invalid encoded data size.");
  }

  DecoderBuffer buffer;
  buffer.Init(reinterpret_cast<const char*>(data), data_size);
  ParsedHeader header;
  if (!ReadHeader(&buffer, &header)) {
    return DecodeError("Invalid or unsupported header.");
  }
  if (header.num_points > data_size) {
    return DecodeError("Point count exceeds the encoded data size.");
  }
  size_t value_count = 0;
  if (!GetValueCount(header, &value_count)) {
    return DecodeError("Encoded value count is too large.");
  }

  values->assign(value_count, 0.0);
  std::vector<uint8_t> seen_points(header.num_points, 0);
  DecodeState state;

  for (uint32_t block_index = 0; block_index < header.num_blocks;
       ++block_index) {
    BlockData block;
    const Status status =
        ReadBlock(&buffer, header, block_index, &seen_points, &block);
    if (!status.ok()) {
      return status;
    }

    if (block_index == 0) {
      if (block.has_prev) {
        return DecodeError("Unexpected context in the first block.");
      }
    } else {
      if (!block.has_prev || !state.has_prev ||
          !SeedsEqual(block.seed_prev, state.prev)) {
        return DecodeError("Block boundary seed mismatch.");
      }
      if (block.has_prev2 != state.has_prev2 ||
          (block.has_prev2 &&
           !SeedsEqual(block.seed_prev2, state.prev2))) {
        return DecodeError("Block boundary second seed mismatch.");
      }
    }

    for (size_t i = 0; i < block.point_indices.size(); ++i) {
      const size_t dst =
          static_cast<size_t>(block.point_indices[i]) * header.num_components;
      for (uint8_t c = 0; c < header.num_components; ++c) {
        (*values)[dst + c] =
            block.values[i * header.num_components + c];
      }
    }
    state = std::move(block.end_state);
  }

  if (buffer.remaining_size() != 0) {
    return DecodeError("Trailing bytes after the last block.");
  }
  for (uint8_t seen : seen_points) {
    if (seen == 0) {
      return DecodeError("The encoded stream does not cover every point.");
    }
  }
  return OkStatus();
}

Status SpatialBlockAttributeCodec::DecodeBlock(
    const EncodedSpatialBlockAttribute &encoded, uint32_t block_index,
    std::vector<double>* values, std::vector<uint32_t>* point_indices) {
  return DecodeBlock(encoded.data.data(), encoded.data.size(), block_index,
                     values, point_indices);
}

Status SpatialBlockAttributeCodec::DecodeBlock(
    const uint8_t* data, size_t data_size, uint32_t block_index,
    std::vector<double>* values, std::vector<uint32_t>* point_indices) {
  if (data == nullptr || values == nullptr) {
    return Invalid("data and values must not be null.");
  }
  if (data_size == 0 ||
      data_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return Invalid("Invalid encoded data size.");
  }

  DecoderBuffer buffer;
  buffer.Init(reinterpret_cast<const char*>(data), data_size);
  ParsedHeader header;
  if (!ReadHeader(&buffer, &header)) {
    return DecodeError("Invalid or unsupported header.");
  }
  if (header.num_points > data_size) {
    return DecodeError("Point count exceeds the encoded data size.");
  }
  size_t value_count = 0;
  if (!GetValueCount(header, &value_count)) {
    return DecodeError("Encoded value count is too large.");
  }
  (void)value_count;
  if (block_index >= header.num_blocks) {
    return Invalid("block_index is out of range.");
  }

  std::vector<uint8_t> seen_points(header.num_points, 0);
  for (uint32_t current = 0; current <= block_index; ++current) {
    BlockData block;
    const Status status =
        ReadBlock(&buffer, header, current, &seen_points, &block);
    if (!status.ok()) {
      return status;
    }
    if (current == block_index) {
      *values = std::move(block.values);
      if (point_indices != nullptr) {
        *point_indices = std::move(block.point_indices);
      }
      return OkStatus();
    }
  }
  return DecodeError("Failed to decode the requested block.");
}

}  // namespace draco

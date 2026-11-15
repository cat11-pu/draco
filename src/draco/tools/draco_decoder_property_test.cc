#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "draco/compression/decode.h"
#include "draco/compression/encode.h"
#include "draco/core/encoder_buffer.h"
#include "draco/core/status.h"
#include "draco/mesh/mesh.h"
#include "draco/point_cloud/point_cloud.h"

namespace {

constexpr int kMalformedCount = 200;
constexpr int kValidCount = 50;

enum class MutationKind {
  kTruncate = 0,
  kMisalign = 1,
  kIllegalAttribute = 2,
  kOversizedIndex = 3,
};

struct EncodedModel {
  std::unique_ptr<draco::Mesh> source;
  std::vector<char> data;
  int method = 0;
  int speed = 0;
};

struct Mutant {
  int id = 0;
  MutationKind kind = MutationKind::kTruncate;
  int seed = 0;
  int subkind = 0;
  size_t bit_offset = 0;
  std::string detail;
  std::vector<char> data;
};

struct ChildReport {
  bool crash = false;
  bool sanitizer_error = false;
  int exit_code = 0;
  int signal = 0;
  bool result_seen = false;
  bool accepted = false;
  int geometry_type = -1;
  int status_code = -1;
  int64_t decoded_bytes = 0;
  std::string message;
  std::string captured;
};

const char* MutationName(MutationKind kind) {
  switch (kind) {
    case MutationKind::kTruncate:
      return "truncate";
    case MutationKind::kMisalign:
      return "misalign";
    case MutationKind::kIllegalAttribute:
      return "illegal_attribute";
    case MutationKind::kOversizedIndex:
      return "oversized_index";
  }
  return "unknown";
}

uint32_t Mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

bool GetBit(const std::vector<char>& data, size_t bit_offset) {
  const size_t byte_offset = bit_offset >> 3;
  const int bit = static_cast<int>(bit_offset & 7);
  return ((static_cast<uint8_t>(data[byte_offset]) >> bit) & 1) != 0;
}

void SetBit(std::vector<char>* data, size_t bit_offset, bool value) {
  const size_t byte_offset = bit_offset >> 3;
  const int bit = static_cast<int>(bit_offset & 7);
  uint8_t byte = static_cast<uint8_t>((*data)[byte_offset]);
  if (value) {
    byte |= static_cast<uint8_t>(1U << bit);
  } else {
    byte &= static_cast<uint8_t>(~(1U << bit));
  }
  (*data)[byte_offset] = static_cast<char>(byte);
}

std::vector<char> DeleteBit(const std::vector<char>& data, size_t bit_offset) {
  const size_t input_bits = data.size() * 8;
  std::vector<char> out((input_bits - 1 + 7) / 8, 0);
  for (size_t i = 0; i < bit_offset; ++i) {
    SetBit(&out, i, GetBit(data, i));
  }
  for (size_t i = bit_offset; i + 1 < input_bits; ++i) {
    SetBit(&out, i, GetBit(data, i + 1));
  }
  return out;
}

std::vector<char> InsertBit(const std::vector<char>& data, size_t bit_offset,
                            bool value) {
  const size_t input_bits = data.size() * 8;
  std::vector<char> out((input_bits + 1 + 7) / 8, 0);
  for (size_t i = 0; i < bit_offset; ++i) {
    SetBit(&out, i, GetBit(data, i));
  }
  SetBit(&out, bit_offset, value);
  for (size_t i = bit_offset; i < input_bits; ++i) {
    SetBit(&out, i + 1, GetBit(data, i));
  }
  return out;
}

bool ReadVarint(const std::vector<char>& data, size_t* pos, uint32_t* value) {
  uint32_t result = 0;
  for (int shift = 0; shift <= 28; shift += 7) {
    if (*pos >= data.size()) {
      return false;
    }
    const uint8_t byte = static_cast<uint8_t>(data[(*pos)++]);
    result |= static_cast<uint32_t>(byte & 0x7f) << shift;
    if ((byte & 0x80) == 0) {
      *value = result;
      return true;
    }
  }
  return false;
}

int AddAttribute(draco::PointCloud* pc, draco::GeometryAttribute::Type type,
                 int8_t num_components, draco::DataType data_type,
                 bool normalized, int num_values) {
  draco::GeometryAttribute ga;
  ga.Init(type, nullptr, num_components, data_type, normalized,
          draco::DataTypeLength(data_type) * num_components, 0);
  return pc->AddAttribute(ga, true, num_values);
}

std::unique_ptr<draco::Mesh> BuildLegalMesh(int seed) {
  const int cells = 2 + seed % 8;
  const int side = cells + 1;
  const int num_points = side * side;
  const int num_faces = cells * cells * 2;

  std::unique_ptr<draco::Mesh> mesh(new draco::Mesh());
  mesh->set_num_points(num_points);
  const int pos_id =
      AddAttribute(mesh.get(), draco::GeometryAttribute::POSITION, 3,
                   draco::DT_FLOAT32, false, num_points);
  const int normal_id = AddAttribute(
      mesh.get(), draco::GeometryAttribute::NORMAL, 3, draco::DT_FLOAT32,
      false, num_points);
  const int color_id = AddAttribute(mesh.get(), draco::GeometryAttribute::COLOR,
                                    4, draco::DT_UINT8, true, num_points);
  const int tex_id =
      AddAttribute(mesh.get(), draco::GeometryAttribute::TEX_COORD, 2,
                   draco::DT_FLOAT32, false, num_points);
  if (pos_id < 0 || normal_id < 0 || color_id < 0 || tex_id < 0) {
    return nullptr;
  }

  for (int y = 0; y < side; ++y) {
    for (int x = 0; x < side; ++x) {
      const int point_id = y * side + x;
      const float fx = static_cast<float>(x);
      const float fy = static_cast<float>(y);
      const std::array<float, 3> position = {
          fx * 0.25f + static_cast<float>(seed) * 0.5f,
          fy * 0.25f - static_cast<float>(seed) * 0.125f,
          static_cast<float>((x * 17 + y * 31 + seed * 13) % 97) * 0.015625f};
      const std::array<float, 3> normal = {0.0f, 0.0f, 1.0f};
      const std::array<uint8_t, 4> color = {
          static_cast<uint8_t>((x * 23 + seed * 11) & 0xff),
          static_cast<uint8_t>((y * 29 + seed * 7) & 0xff),
          static_cast<uint8_t>((x * y * 3 + seed * 5 + 17) & 0xff), 255};
      const std::array<float, 2> tex = {
          static_cast<float>(x) / static_cast<float>(cells),
          static_cast<float>(y) / static_cast<float>(cells)};
      mesh->attribute(pos_id)->SetAttributeValue(draco::AttributeValueIndex(
                                                     point_id),
                                                 position.data());
      mesh->attribute(normal_id)->SetAttributeValue(
          draco::AttributeValueIndex(point_id), normal.data());
      mesh->attribute(color_id)->SetAttributeValue(
          draco::AttributeValueIndex(point_id), color.data());
      mesh->attribute(tex_id)->SetAttributeValue(
          draco::AttributeValueIndex(point_id), tex.data());
    }
  }

  for (int y = 0; y < cells; ++y) {
    for (int x = 0; x < cells; ++x) {
      const int v00 = y * side + x;
      const int v10 = v00 + 1;
      const int v01 = v00 + side;
      const int v11 = v01 + 1;
      mesh->AddFace({{draco::PointIndex(v00), draco::PointIndex(v10),
                      draco::PointIndex(v01)}});
      mesh->AddFace({{draco::PointIndex(v10), draco::PointIndex(v11),
                      draco::PointIndex(v01)}});
    }
  }
  if (mesh->num_faces() != static_cast<uint32_t>(num_faces)) {
    return nullptr;
  }
  return mesh;
}

std::unique_ptr<draco::PointCloud> BuildLegalPointCloud(int seed) {
  const int num_points = 17 + seed * 3;
  std::unique_ptr<draco::PointCloud> pc(new draco::PointCloud());
  pc->set_num_points(num_points);
  const int pos_id = AddAttribute(pc.get(), draco::GeometryAttribute::POSITION,
                                  3, draco::DT_FLOAT32, false, num_points);
  if (pos_id < 0) {
    return nullptr;
  }
  for (int i = 0; i < num_points; ++i) {
    const std::array<float, 3> position = {
        static_cast<float>(i) * 0.125f,
        static_cast<float>((i * 7 + seed) % 19) * 0.25f,
        static_cast<float>((i * 11 + seed * 3) % 23) * 0.0625f};
    pc->attribute(pos_id)->SetAttributeValue(draco::AttributeValueIndex(i),
                                             position.data());
  }
  return pc;
}

std::unique_ptr<draco::Mesh> BuildRawIndexMesh(int num_points) {
  std::unique_ptr<draco::Mesh> mesh(new draco::Mesh());
  mesh->set_num_points(num_points);
  const int pos_id =
      AddAttribute(mesh.get(), draco::GeometryAttribute::POSITION, 3,
                   draco::DT_FLOAT32, false, num_points);
  if (pos_id < 0) {
    return nullptr;
  }
  for (int i = 0; i < num_points; ++i) {
    const std::array<float, 3> position = {
        static_cast<float>(i), static_cast<float>(i % 5),
        static_cast<float>(i % 7)};
    mesh->attribute(pos_id)->SetAttributeValue(draco::AttributeValueIndex(i),
                                               position.data());
  }
  mesh->AddFace({{draco::PointIndex(0), draco::PointIndex(1),
                  draco::PointIndex(2)}});
  return mesh;
}

bool EncodeMesh(const draco::Mesh& mesh, int method, int speed,
                std::vector<char>* data) {
  draco::Encoder encoder;
  encoder.SetEncodingMethod(method);
  encoder.SetSpeedOptions(speed, speed);
  draco::EncoderBuffer buffer;
  const draco::Status status = encoder.EncodeMeshToBuffer(mesh, &buffer);
  if (!status.ok()) {
    return false;
  }
  data->assign(buffer.data(), buffer.data() + buffer.size());
  return true;
}

bool EncodePointCloud(const draco::PointCloud& pc, std::vector<char>* data) {
  draco::Encoder encoder;
  encoder.SetEncodingMethod(draco::POINT_CLOUD_SEQUENTIAL_ENCODING);
  encoder.SetSpeedOptions(10, 10);
  draco::EncoderBuffer buffer;
  const draco::Status status = encoder.EncodePointCloudToBuffer(pc, &buffer);
  if (!status.ok()) {
    return false;
  }
  data->assign(buffer.data(), buffer.data() + buffer.size());
  return true;
}

bool LocatePointCloudAttributeFields(const std::vector<char>& data,
                                     size_t* type, size_t* data_type,
                                     size_t* components, size_t* decoder_type) {
  size_t pos = 11;
  if (data.size() < pos + 4) {
    return false;
  }
  pos += 4;
  if (pos >= data.size()) {
    return false;
  }
  const uint8_t num_attribute_decoders =
      static_cast<uint8_t>(data[pos++]);
  if (num_attribute_decoders != 1) {
    return false;
  }
  uint32_t num_attributes = 0;
  if (!ReadVarint(data, &pos, &num_attributes) || num_attributes != 1 ||
      pos + 5 > data.size()) {
    return false;
  }
  *type = pos;
  *data_type = pos + 1;
  *components = pos + 2;
  pos += 4;
  uint32_t unique_id = 0;
  if (!ReadVarint(data, &pos, &unique_id)) {
    return false;
  }
  *decoder_type = pos;
  return pos < data.size();
}

bool LocateRawIndex(const std::vector<char>& data, size_t* index_byte,
                    uint32_t* num_points) {
  size_t pos = 11;
  uint32_t num_faces = 0;
  if (!ReadVarint(data, &pos, &num_faces) ||
      !ReadVarint(data, &pos, num_points) || pos >= data.size()) {
    return false;
  }
  const uint8_t connectivity_method = static_cast<uint8_t>(data[pos++]);
  if (connectivity_method != 1 || num_faces == 0 || *num_points == 0) {
    return false;
  }
  *index_byte = pos;
  return pos < data.size();
}

std::vector<EncodedModel> BuildValidModels() {
  std::vector<EncodedModel> models;
  models.reserve(kValidCount);
  for (int seed = 0; seed < kValidCount; ++seed) {
    EncodedModel model;
    model.source = BuildLegalMesh(seed);
    if (model.source == nullptr) {
      std::fprintf(stderr, "Failed to build valid mesh %d\n", seed);
      std::exit(2);
    }
    model.method = (seed % 2 == 0) ? draco::MESH_EDGEBREAKER_ENCODING
                                   : draco::MESH_SEQUENTIAL_ENCODING;
    model.speed = seed % 11;
    if (!EncodeMesh(*model.source, model.method, model.speed, &model.data)) {
      std::fprintf(stderr, "Failed to encode valid mesh %d\n", seed);
      std::exit(2);
    }
    models.push_back(std::move(model));
  }
  return models;
}

Mutant MakeMutant(int id, const std::vector<EncodedModel>& valid_models) {
  Mutant mutant;
  mutant.id = id;
  mutant.kind = static_cast<MutationKind>((id - 1) % 4);
  mutant.seed = (id * 7 + id / 4) % kValidCount;
  mutant.subkind = (id - 1) / 4;

  const uint32_t mixed = Mix(static_cast<uint32_t>(id * 2654435761U));
  if (mutant.kind == MutationKind::kTruncate) {
    const EncodedModel& model = valid_models[mutant.seed];
    mutant.data = model.data;
    const size_t first_payload_byte = 12;
    const size_t cut_byte =
        first_payload_byte +
        mixed % (mutant.data.size() - first_payload_byte);
    mutant.bit_offset = cut_byte * 8;
    mutant.detail = "cut at byte " + std::to_string(cut_byte);
    mutant.data.resize(cut_byte);
    return mutant;
  }

  if (mutant.kind == MutationKind::kMisalign) {
    const EncodedModel& model = valid_models[mutant.seed];
    const size_t total_bits = model.data.size() * 8;
    const size_t first_payload_bit = 12 * 8;
    mutant.bit_offset =
        first_payload_bit + mixed % (total_bits - first_payload_bit);
    const bool insert = (id & 1) != 0;
    mutant.data = insert ? InsertBit(model.data, mutant.bit_offset, true)
                         : DeleteBit(model.data, mutant.bit_offset);
    mutant.detail = insert ? "inserted one bit" : "deleted one bit";
    return mutant;
  }

  if (mutant.kind == MutationKind::kIllegalAttribute) {
    std::unique_ptr<draco::PointCloud> pc =
        BuildLegalPointCloud(mutant.subkind);
    if (pc == nullptr || !EncodePointCloud(*pc, &mutant.data)) {
      std::fprintf(stderr, "Failed to build illegal-attribute seed %d\n", id);
      std::exit(2);
    }
    size_t type = 0;
    size_t data_type = 0;
    size_t components = 0;
    size_t decoder_type = 0;
    if (!LocatePointCloudAttributeFields(mutant.data, &type, &data_type,
                                         &components, &decoder_type)) {
      std::fprintf(stderr, "Failed to locate attribute fields for seed %d\n",
                   id);
      std::exit(2);
    }
    const int variant = mutant.subkind % 5;
    if (variant == 0) {
      mutant.bit_offset = type * 8;
      mutant.data[type] = static_cast<char>(0xff);
      mutant.detail = "attribute type = 255";
    } else if (variant == 1) {
      mutant.bit_offset = data_type * 8;
      mutant.data[data_type] = static_cast<char>(0xff);
      mutant.detail = "data type = 255";
    } else if (variant == 2) {
      mutant.bit_offset = components * 8;
      mutant.data[components] = 0;
      mutant.detail = "num_components = 0";
    } else if (variant == 3) {
      mutant.bit_offset = components * 8;
      mutant.data[components] = static_cast<char>(0xff);
      mutant.detail = "num_components = 255";
    } else {
      mutant.bit_offset = decoder_type * 8;
      mutant.data[decoder_type] = static_cast<char>(0xff);
      mutant.detail = "sequential decoder type = 255";
    }
    return mutant;
  }

  const int num_points = (id & 1) == 0 ? 8 : 300;
  std::unique_ptr<draco::Mesh> mesh = BuildRawIndexMesh(num_points);
  if (mesh == nullptr ||
      !EncodeMesh(*mesh, draco::MESH_SEQUENTIAL_ENCODING, 10,
                  &mutant.data)) {
    std::fprintf(stderr, "Failed to build oversized-index seed %d\n", id);
    std::exit(2);
  }
  size_t index_byte = 0;
  uint32_t decoded_num_points = 0;
  if (!LocateRawIndex(mutant.data, &index_byte, &decoded_num_points) ||
      decoded_num_points != static_cast<uint32_t>(num_points)) {
    std::fprintf(stderr, "Failed to locate raw index for seed %d\n", id);
    std::exit(2);
  }
  mutant.bit_offset = index_byte * 8;
  if (decoded_num_points < 256) {
    mutant.data[index_byte] = static_cast<char>(0xff);
    mutant.detail = "uint8 index = 255, num_points = " +
                    std::to_string(decoded_num_points);
  } else {
    mutant.data[index_byte] = static_cast<char>(0xff);
    mutant.data[index_byte + 1] = static_cast<char>(0xff);
    mutant.detail = "uint16 index = 65535, num_points = " +
                    std::to_string(decoded_num_points);
  }
  return mutant;
}

std::string EscapeMessage(const std::string& message) {
  std::string escaped;
  escaped.reserve(message.size());
  for (const char c : message) {
    if (c == '\n' || c == '\r' || c == '\t') {
      escaped.push_back(' ');
    } else {
      escaped.push_back(c);
    }
  }
  return escaped;
}

[[noreturn]] void RunChild(int write_fd, const std::vector<char>& data) {
  if (dup2(write_fd, STDOUT_FILENO) < 0 || dup2(write_fd, STDERR_FILENO) < 0) {
    _exit(126);
  }
  close(write_fd);

  draco::DecoderBuffer buffer;
  buffer.Init(data.data(), data.size());
  auto type_or = draco::Decoder::GetEncodedGeometryType(&buffer);
  if (!type_or.ok()) {
    std::printf("RESULT\ttype=-1\taccepted=0\tcode=%d\tbytes=%lld\tmsg=%s\n",
                static_cast<int>(type_or.status().code()),
                static_cast<long long>(buffer.decoded_size()),
                EscapeMessage(type_or.status().error_msg()).c_str());
    std::fflush(stdout);
    std::exit(0);
  }

  const draco::EncodedGeometryType type = type_or.value();
  if (type == draco::TRIANGULAR_MESH) {
    draco::Decoder decoder;
    auto result = decoder.DecodeMeshFromBuffer(&buffer);
    std::printf("RESULT\ttype=%d\taccepted=%d\tcode=%d\tbytes=%lld\tmsg=%s\n",
                static_cast<int>(type), result.ok() ? 1 : 0,
                static_cast<int>(result.status().code()),
                static_cast<long long>(buffer.decoded_size()),
                EscapeMessage(result.status().error_msg()).c_str());
  } else if (type == draco::POINT_CLOUD) {
    draco::Decoder decoder;
    auto result = decoder.DecodePointCloudFromBuffer(&buffer);
    std::printf("RESULT\ttype=%d\taccepted=%d\tcode=%d\tbytes=%lld\tmsg=%s\n",
                static_cast<int>(type), result.ok() ? 1 : 0,
                static_cast<int>(result.status().code()),
                static_cast<long long>(buffer.decoded_size()),
                EscapeMessage(result.status().error_msg()).c_str());
  } else {
    std::printf("RESULT\ttype=%d\taccepted=0\tcode=-1\tbytes=%lld\tmsg=%s\n",
                static_cast<int>(type),
                static_cast<long long>(buffer.decoded_size()),
                "unsupported geometry type");
  }
  std::fflush(stdout);
  std::exit(0);
}

struct ChildProcessResult {
  int status = 0;
  std::string captured;
};

ChildProcessResult RunIsolated(const std::vector<char>& data) {
  int pipe_fds[2];
  if (pipe(pipe_fds) != 0) {
    std::perror("pipe");
    std::exit(2);
  }
  std::fflush(stdout);
  std::fflush(stderr);
  const pid_t pid = fork();
  if (pid < 0) {
    std::perror("fork");
    std::exit(2);
  }
  if (pid == 0) {
    close(pipe_fds[0]);
    RunChild(pipe_fds[1], data);
  }

  close(pipe_fds[1]);
  ChildProcessResult result;
  char chunk[4096];
  while (true) {
    const ssize_t count = read(pipe_fds[0], chunk, sizeof(chunk));
    if (count > 0) {
      result.captured.append(chunk, static_cast<size_t>(count));
    } else if (count == 0) {
      break;
    } else if (errno != EINTR) {
      break;
    }
  }
  close(pipe_fds[0]);
  if (waitpid(pid, &result.status, 0) < 0) {
    std::perror("waitpid");
    std::exit(2);
  }
  return result;
}

bool ParseIntField(const std::string& text, const std::string& key,
                   int64_t* value) {
  const std::string token = key + "=";
  const size_t start = text.find(token);
  if (start == std::string::npos) {
    return false;
  }
  const size_t value_start = start + token.size();
  size_t value_end = value_start;
  while (value_end < text.size() && text[value_end] != '\t' &&
         text[value_end] != '\n') {
    ++value_end;
  }
  if (value_end == value_start) {
    return false;
  }
  char* end = nullptr;
  const long long parsed =
      std::strtoll(text.c_str() + value_start, &end, 10);
  if (end == text.c_str() + value_start) {
    return false;
  }
  *value = parsed;
  return true;
}

std::string GetField(const std::string& text, const std::string& key) {
  const std::string token = key + "=";
  const size_t start = text.find(token);
  if (start == std::string::npos) {
    return "";
  }
  const size_t value_start = start + token.size();
  size_t value_end = value_start;
  while (value_end < text.size() && text[value_end] != '\t' &&
         text[value_end] != '\n') {
    ++value_end;
  }
  return text.substr(value_start, value_end - value_start);
}

ChildReport AnalyzeChild(const ChildProcessResult& process) {
  ChildReport report;
  report.captured = process.captured;
  report.exit_code = WIFEXITED(process.status)
                         ? WEXITSTATUS(process.status)
                         : -1;
  report.signal =
      WIFSIGNALED(process.status) ? WTERMSIG(process.status) : 0;
  report.crash = report.signal != 0 || report.exit_code != 0;
  report.sanitizer_error =
      process.captured.find("AddressSanitizer") != std::string::npos ||
      process.captured.find("LeakSanitizer") != std::string::npos ||
      process.captured.find("runtime error:") != std::string::npos;

  const size_t result_pos = process.captured.find("RESULT\t");
  if (result_pos != std::string::npos) {
    report.result_seen = true;
    const std::string line =
        process.captured.substr(result_pos,
                                process.captured.find('\n', result_pos) -
                                    result_pos);
    int64_t integer = 0;
    if (ParseIntField(line, "accepted", &integer)) {
      report.accepted = integer != 0;
    }
    if (ParseIntField(line, "type", &integer)) {
      report.geometry_type = static_cast<int>(integer);
    }
    if (ParseIntField(line, "code", &integer)) {
      report.status_code = static_cast<int>(integer);
    }
    if (ParseIntField(line, "bytes", &integer)) {
      report.decoded_bytes = integer;
    }
    report.message = GetField(line, "msg");
  }
  return report;
}

std::string AttributeMetadata(const draco::PointAttribute& attribute) {
  return std::to_string(static_cast<int>(attribute.attribute_type())) + ":" +
         std::to_string(static_cast<int>(attribute.data_type())) + ":" +
         std::to_string(attribute.num_components()) + ":" +
         (attribute.normalized() ? "1" : "0") + ":" +
         std::to_string(attribute.byte_stride());
}

std::string CornerValue(const draco::PointAttribute& attribute,
                        draco::PointIndex point) {
  const draco::AttributeValueIndex value_index =
      attribute.mapped_index(point);
  const int64_t byte_stride = attribute.byte_stride();
  if (value_index.value() >= attribute.size() || byte_stride < 0) {
    return "<invalid>";
  }
  const int64_t byte_pos = attribute.GetBytePos(value_index);
  if (byte_pos < 0 ||
      byte_pos + byte_stride >
          static_cast<int64_t>(attribute.buffer()->data_size())) {
    return "<invalid>";
  }
  return std::string(
      reinterpret_cast<const char*>(attribute.buffer()->data() + byte_pos),
      static_cast<size_t>(byte_stride));
}

std::string HexEncode(const std::string& bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (const unsigned char byte : bytes) {
    out.push_back(kHex[byte >> 4]);
    out.push_back(kHex[byte & 0xf]);
  }
  return out;
}

std::string DumpManifest(const draco::Mesh& mesh, int seed) {
  std::ostringstream out;
  out << "MODEL " << seed << "\n";
  out << "faces " << mesh.num_faces() << "\n";
  out << "points " << mesh.num_points() << "\n";
  out << "attrs " << mesh.num_attributes() << "\n";

  std::vector<int> attribute_ids;
  for (int att_id = 0; att_id < mesh.num_attributes(); ++att_id) {
    attribute_ids.push_back(att_id);
  }
  std::sort(attribute_ids.begin(), attribute_ids.end(),
            [&mesh](int lhs, int rhs) {
              const draco::PointAttribute* left = mesh.attribute(lhs);
              const draco::PointAttribute* right = mesh.attribute(rhs);
              if (left->attribute_type() != right->attribute_type()) {
                return left->attribute_type() < right->attribute_type();
              }
              return left->unique_id() < right->unique_id();
            });

  for (const int att_id : attribute_ids) {
    const draco::PointAttribute& attribute = *mesh.attribute(att_id);
    out << "attr " << static_cast<int>(attribute.attribute_type()) << " "
        << attribute.unique_id() << " "
        << static_cast<int>(attribute.data_type()) << " "
        << static_cast<int>(attribute.num_components()) << " "
        << (attribute.normalized() ? 1 : 0) << " " << attribute.size() << " "
        << attribute.byte_stride() << "\n";
  }
  for (draco::FaceIndex face_index(0); face_index < mesh.num_faces();
       ++face_index) {
    const draco::Mesh::Face& face = mesh.face(face_index);
    out << "face " << face[0].value() << " " << face[1].value() << " "
        << face[2].value() << "\n";
  }
  for (const int att_id : attribute_ids) {
    const draco::PointAttribute& attribute = *mesh.attribute(att_id);
    for (int point = 0; point < static_cast<int>(mesh.num_points()); ++point) {
      out << "value " << static_cast<int>(attribute.attribute_type()) << " "
          << attribute.unique_id() << " " << point << " "
          << HexEncode(CornerValue(attribute, draco::PointIndex(point)))
          << "\n";
    }
  }
  return out.str();
}

std::vector<std::string> CanonicalFaceItems(const draco::Mesh& mesh) {
  std::vector<std::string> items;
  items.reserve(mesh.num_faces());
  for (draco::FaceIndex face_index(0); face_index < mesh.num_faces();
       ++face_index) {
    const draco::Mesh::Face& face = mesh.face(face_index);
    std::string best;
    for (int rotation = 0; rotation < 3; ++rotation) {
      std::string serialized;
      for (int corner = 0; corner < 3; ++corner) {
        const draco::PointIndex point =
            face[(rotation + corner) % 3];
        for (int att_id = 0; att_id < mesh.num_attributes(); ++att_id) {
          const draco::PointAttribute* attribute = mesh.attribute(att_id);
          serialized += AttributeMetadata(*attribute);
          serialized.push_back('\x1f');
          serialized += CornerValue(*attribute, point);
          serialized.push_back('\x1e');
        }
      }
      if (rotation == 0 || serialized < best) {
        best = std::move(serialized);
      }
    }
    items.push_back(std::move(best));
  }
  std::sort(items.begin(), items.end());
  return items;
}

struct ComparisonResult {
  bool ok = false;
  size_t items_compared = 0;
  std::string detail;
};

ComparisonResult CompareMeshes(const draco::Mesh& expected,
                               const draco::Mesh& actual) {
  ComparisonResult result;
  if (expected.num_faces() != actual.num_faces()) {
    result.detail =
        "num_faces expected=" + std::to_string(expected.num_faces()) +
        " actual=" + std::to_string(actual.num_faces());
    return result;
  }
  if (expected.num_points() != actual.num_points()) {
    result.detail = "num_points expected=" +
                    std::to_string(expected.num_points()) +
                    " actual=" + std::to_string(actual.num_points());
    return result;
  }
  if (expected.num_attributes() != actual.num_attributes()) {
    result.detail = "num_attributes expected=" +
                    std::to_string(expected.num_attributes()) +
                    " actual=" + std::to_string(actual.num_attributes());
    return result;
  }
  for (int att_id = 0; att_id < expected.num_attributes(); ++att_id) {
    const std::string expected_metadata =
        AttributeMetadata(*expected.attribute(att_id));
    const std::string actual_metadata =
        AttributeMetadata(*actual.attribute(att_id));
    if (expected_metadata != actual_metadata) {
      result.detail = "attribute " + std::to_string(att_id) +
                      " metadata expected=" + expected_metadata +
                      " actual=" + actual_metadata;
      return result;
    }
  }

  const std::vector<std::string> expected_items = CanonicalFaceItems(expected);
  const std::vector<std::string> actual_items = CanonicalFaceItems(actual);
  result.items_compared = expected_items.size();
  if (expected_items.size() != actual_items.size()) {
    result.detail = "canonical face count mismatch";
    return result;
  }
  for (size_t i = 0; i < expected_items.size(); ++i) {
    if (expected_items[i] != actual_items[i]) {
      result.detail = "canonical face " + std::to_string(i) +
                      " differs (attribute/corner bytes)";
      return result;
    }
  }
  result.ok = true;
  result.detail = "all fields match";
  return result;
}

bool IsSanitizerFailure(const ChildReport& report) {
  return report.sanitizer_error || report.signal != 0 ||
         (report.exit_code != 0 && !report.result_seen);
}

bool ReadTextFile(const std::string& path, std::string* text) {
  std::ifstream input(path);
  if (!input) {
    return false;
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  *text = contents.str();
  return true;
}

std::string FirstManifestDifference(const std::string& expected,
                                    const std::string& actual) {
  std::istringstream expected_stream(expected);
  std::istringstream actual_stream(actual);
  std::string expected_line;
  std::string actual_line;
  std::string model;
  size_t line_number = 0;
  while (true) {
    const bool expected_ok =
        static_cast<bool>(std::getline(expected_stream, expected_line));
    const bool actual_ok =
        static_cast<bool>(std::getline(actual_stream, actual_line));
    if (!expected_ok && !actual_ok) {
      return "none";
    }
    ++line_number;
    if (expected_line.rfind("MODEL ", 0) == 0) {
      model = expected_line;
    }
    if (!expected_ok || !actual_ok || expected_line != actual_line) {
      return "line " + std::to_string(line_number) + " " + model +
             " expected=[" + expected_line + "] actual=[" + actual_line + "]";
    }
  }
}

struct ReferenceResult {
  bool ok = false;
  size_t records = 0;
  std::string detail;
};

ReferenceResult CompareWithReferenceDecoder(const std::string& artifacts,
                                            const std::string& candidate) {
  ReferenceResult result;
  const char* script_env = std::getenv("DRACO_PROPERTY_REFERENCE_SCRIPT");
  const std::string script =
      script_env == nullptr
          ? "src/draco/tools/draco_reference_decode.js"
          : script_env;
  const std::string model_dir = artifacts + "/valid_models";
  const std::string reference_path = artifacts + "/reference_manifest.txt";
  const std::string error_path = artifacts + "/reference_decoder.stderr";
  const std::string command = "node '" + script + "' '" + model_dir + "' > '" +
                              reference_path + "' 2> '" + error_path + "'";
  const int command_status = std::system(command.c_str());
  if (command_status != 0) {
    std::string error;
    ReadTextFile(error_path, &error);
    result.detail = "reference decoder failed: " + error;
    return result;
  }

  std::string reference;
  if (!ReadTextFile(reference_path, &reference)) {
    result.detail = "failed to read reference manifest";
    return result;
  }
  std::istringstream lines(reference);
  std::string line;
  while (std::getline(lines, line)) {
    if (!line.empty()) {
      ++result.records;
    }
  }
  result.detail = FirstManifestDifference(candidate, reference);
  result.ok = result.detail == "none";
  return result;
}

std::string KindSummaryName(MutationKind kind) {
  return MutationName(kind);
}

void WriteFile(const std::string& path, const std::vector<char>& data) {
  FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    std::perror(path.c_str());
    return;
  }
  if (!data.empty()) {
    std::fwrite(data.data(), 1, data.size(), file);
  }
  std::fclose(file);
}

void WriteTextFile(const std::string& path, const std::string& text) {
  FILE* file = std::fopen(path.c_str(), "w");
  if (file == nullptr) {
    std::perror(path.c_str());
    return;
  }
  std::fwrite(text.data(), 1, text.size(), file);
  std::fclose(file);
}

struct KindStats {
  int samples = 0;
  int accepted = 0;
  int rejected = 0;
  int crashes = 0;
  int sanitizer_errors = 0;
  size_t min_bit = static_cast<size_t>(-1);
  size_t max_bit = 0;
};

void UpdateStats(KindStats* stats, const Mutant& mutant,
                 const ChildReport& report) {
  ++stats->samples;
  if (report.accepted) {
    ++stats->accepted;
  } else if (report.result_seen) {
    ++stats->rejected;
  }
  if (IsSanitizerFailure(report)) {
    ++stats->crashes;
    if (report.sanitizer_error) {
      ++stats->sanitizer_errors;
    }
  }
  stats->min_bit = std::min(stats->min_bit, mutant.bit_offset);
  stats->max_bit = std::max(stats->max_bit, mutant.bit_offset);
}

int Run(int argc, char** argv) {
  (void)argc;
  (void)argv;
  const char* artifacts_env = std::getenv("DRACO_PROPERTY_ARTIFACTS");
  const std::string artifacts =
      artifacts_env == nullptr ? "property_test_artifacts" : artifacts_env;
  const std::string counterexample_dir = artifacts + "/counterexamples";
  const std::string accepted_malformed_dir =
      artifacts + "/accepted_malformed";
  const std::string valid_model_dir = artifacts + "/valid_models";
  const std::string mkdir_command =
      "mkdir -p '" + counterexample_dir + "' '" + accepted_malformed_dir +
      "' '" + valid_model_dir + "'";
  if (std::system(mkdir_command.c_str()) != 0) {
    std::fprintf(stderr, "Failed to create artifact directory\n");
    return 2;
  }

  std::printf("Building and encoding %d valid reference models...\n",
              kValidCount);
  std::fflush(stdout);
  const std::vector<EncodedModel> valid_models = BuildValidModels();

  int valid_passed = 0;
  int valid_failed = 0;
  size_t valid_items = 0;
  std::string valid_failure;
  std::ostringstream candidate_manifest;
  std::ostringstream valid_rows;
  valid_rows << "seed\tmethod\tspeed\tsource_faces\tsource_points\t"
                "decoded_faces\tdecoded_points\tattributes\tface_items\t"
                "result\tdetail\n";
  for (int seed = 0; seed < kValidCount; ++seed) {
    const EncodedModel& model = valid_models[seed];
    WriteFile(valid_model_dir + "/model_" + std::to_string(seed) + ".drc",
              model.data);
    const ChildProcessResult process = RunIsolated(model.data);
    const ChildReport report = AnalyzeChild(process);
    if (report.crash || report.sanitizer_error) {
      ++valid_failed;
      valid_rows << seed << '\t' << model.method << '\t' << model.speed << '\t'
                 << model.source->num_faces() << '\t'
                 << model.source->num_points() << "\t0\t0\t0\t0\tFAIL\t"
                 << "sanitizer_or_crash\n";
      if (valid_failure.empty()) {
        valid_failure = "valid seed " + std::to_string(seed) +
                        " sanitizer/crash: " + report.captured;
      }
      continue;
    }

    draco::DecoderBuffer buffer;
    buffer.Init(model.data.data(), model.data.size());
    draco::Decoder decoder;
    auto decoded = decoder.DecodeMeshFromBuffer(&buffer);
    if (!decoded.ok() || decoded.value() == nullptr) {
      ++valid_failed;
      valid_rows << seed << '\t' << model.method << '\t' << model.speed << '\t'
                 << model.source->num_faces() << '\t'
                 << model.source->num_points() << "\t0\t0\t0\t0\tFAIL\t"
                 << decoded.status().error_msg() << "\n";
      if (valid_failure.empty()) {
        valid_failure = "valid seed " + std::to_string(seed) +
                        " did not decode: " + decoded.status().error_msg();
      }
      continue;
    }
    candidate_manifest << DumpManifest(*decoded.value(), seed);
    const ComparisonResult comparison =
        CompareMeshes(*model.source, *decoded.value());
    valid_items += comparison.items_compared;
    if (!comparison.ok) {
      ++valid_failed;
      valid_rows << seed << '\t' << model.method << '\t' << model.speed << '\t'
                 << model.source->num_faces() << '\t'
                 << model.source->num_points() << '\t'
                 << decoded.value()->num_faces() << '\t'
                 << decoded.value()->num_points() << '\t'
                 << decoded.value()->num_attributes() << '\t'
                 << comparison.items_compared << "\tFAIL\t"
                 << comparison.detail << "\n";
      if (valid_failure.empty()) {
        valid_failure = "valid seed " + std::to_string(seed) + ": " +
                        comparison.detail;
      }
    } else {
      ++valid_passed;
      valid_rows << seed << '\t' << model.method << '\t' << model.speed << '\t'
                 << model.source->num_faces() << '\t'
                 << model.source->num_points() << '\t'
                 << decoded.value()->num_faces() << '\t'
                 << decoded.value()->num_points() << '\t'
                 << decoded.value()->num_attributes() << '\t'
                 << comparison.items_compared << "\tPASS\t"
                 << comparison.detail << "\n";
    }
  }
  const std::string candidate_manifest_text = candidate_manifest.str();
  WriteTextFile(artifacts + "/candidate_manifest.txt",
                candidate_manifest_text);
  const ReferenceResult reference =
      CompareWithReferenceDecoder(artifacts, candidate_manifest_text);
  int reference_failed = 0;
  if (!reference.ok) {
    reference_failed = 1;
    if (valid_failure.empty()) {
      valid_failure = "reference decoder mismatch: " + reference.detail;
    }
  }

  std::printf("Running %d malformed models (4-way alternating)...\n",
              kMalformedCount);
  std::fflush(stdout);
  std::array<KindStats, 4> stats;
  int failures = 0;
  int counterexamples = 0;
  std::ostringstream malformed_rows;
  malformed_rows << "id\tkind\tseed\tsubkind\tbit_offset\tbyte_offset\t"
                    "accepted\tstatus_code\tdecoded_bytes\tresult\tdetail\t"
                    "decoder_message\n";
  for (int id = 1; id <= kMalformedCount; ++id) {
    const Mutant mutant = MakeMutant(id, valid_models);
    const ChildProcessResult process = RunIsolated(mutant.data);
    const ChildReport report = AnalyzeChild(process);
    KindStats& kind_stats = stats[static_cast<size_t>(mutant.kind)];
    UpdateStats(&kind_stats, mutant, report);

    const bool failure = IsSanitizerFailure(report);
    if (report.accepted) {
      WriteFile(accepted_malformed_dir + "/case_" +
                    std::to_string(mutant.id) + "_" +
                    MutationName(mutant.kind) + "_bit_" +
                    std::to_string(mutant.bit_offset) + ".drc",
                mutant.data);
    }
    malformed_rows << mutant.id << '\t' << MutationName(mutant.kind) << '\t'
                   << mutant.seed << '\t' << mutant.subkind << '\t'
                   << mutant.bit_offset << '\t' << (mutant.bit_offset / 8)
                   << '\t' << (report.accepted ? 1 : 0) << '\t'
                   << report.status_code << '\t' << report.decoded_bytes << '\t'
                   << (failure ? "FAIL" : "PASS") << '\t' << mutant.detail
                   << '\t' << report.message << '\n';
    if (failure) {
      ++failures;
      ++counterexamples;
      const std::string stem =
          counterexample_dir + "/case_" + std::to_string(mutant.id) + "_" +
          MutationName(mutant.kind) + "_bit_" +
          std::to_string(mutant.bit_offset);
      WriteFile(stem + ".drc", mutant.data);
      std::ostringstream metadata;
      metadata << "id=" << mutant.id << "\n"
               << "kind=" << MutationName(mutant.kind) << "\n"
               << "seed=" << mutant.seed << "\n"
               << "subkind=" << mutant.subkind << "\n"
               << "bit_offset=" << mutant.bit_offset << "\n"
               << "byte_offset=" << (mutant.bit_offset / 8) << "\n"
               << "detail=" << mutant.detail << "\n"
               << "exit_code=" << report.exit_code << "\n"
               << "signal=" << report.signal << "\n"
               << "sanitizer_error=" << (report.sanitizer_error ? 1 : 0)
               << "\n\n"
               << report.captured;
      WriteTextFile(stem + ".txt", metadata.str());
      std::printf(
          "FAIL id=%d kind=%s bit_offset=%zu byte_offset=%zu exit=%d "
          "signal=%d\n%s\n",
          mutant.id, MutationName(mutant.kind), mutant.bit_offset,
          mutant.bit_offset / 8, report.exit_code, report.signal,
          report.captured.c_str());
    }
  }

  std::ostringstream summary;
  summary << "Draco decoder property test summary\n";
  summary << "malformed_total=" << kMalformedCount << "\n";
  summary << "malformed_failures=" << failures << "\n";
  summary << "counterexamples=" << counterexamples << "\n";
  for (size_t i = 0; i < stats.size(); ++i) {
    const KindStats& kind_stats = stats[i];
    summary << "kind=" << KindSummaryName(static_cast<MutationKind>(i))
            << " samples=" << kind_stats.samples
            << " accepted=" << kind_stats.accepted
            << " rejected=" << kind_stats.rejected
            << " crashes=" << kind_stats.crashes
            << " sanitizer_errors=" << kind_stats.sanitizer_errors
            << " bit_offset_range="
            << (kind_stats.samples == 0 ? 0 : kind_stats.min_bit) << ".."
            << kind_stats.max_bit << "\n";
  }
  summary << "valid_total=" << kValidCount << "\n";
  summary << "valid_passed=" << valid_passed << "\n";
  summary << "valid_failed=" << valid_failed << "\n";
  summary << "valid_canonical_face_items_compared=" << valid_items << "\n";
  summary << "reference_decoder=javascript/draco_decoder.wasm "
             "(Draco 1.5.9)\n";
  summary << "reference_decoder_match=" << (reference.ok ? 1 : 0) << "\n";
  summary << "reference_decoder_failed=" << reference_failed << "\n";
  summary << "reference_manifest_records_compared=" << reference.records
          << "\n";
  summary << "reference_decoder_detail=" << reference.detail << "\n";
  summary << "valid_failure="
          << (valid_failure.empty() ? "none" : valid_failure) << "\n";
  const std::string summary_text = summary.str();
  std::printf("%s", summary_text.c_str());
  WriteTextFile(artifacts + "/summary.txt", summary_text);
  WriteTextFile(artifacts + "/malformed_cases.tsv", malformed_rows.str());
  WriteTextFile(artifacts + "/valid_cases.tsv", valid_rows.str());

  std::ostringstream json;
  json << "{\n"
       << "  \"malformed_total\": " << kMalformedCount << ",\n"
       << "  \"malformed_failures\": " << failures << ",\n"
       << "  \"counterexamples\": " << counterexamples << ",\n"
       << "  \"valid_total\": " << kValidCount << ",\n"
       << "  \"valid_passed\": " << valid_passed << ",\n"
       << "  \"valid_failed\": " << valid_failed << ",\n"
       << "  \"valid_canonical_face_items_compared\": " << valid_items
       << ",\n"
       << "  \"reference_decoder\": \"javascript/draco_decoder.wasm "
          "(Draco 1.5.9)\",\n"
       << "  \"reference_decoder_match\": " << (reference.ok ? 1 : 0) << ",\n"
       << "  \"reference_decoder_failed\": " << reference_failed << ",\n"
       << "  \"reference_manifest_records_compared\": " << reference.records
       << "\n}\n";
  WriteTextFile(artifacts + "/summary.json", json.str());

  return (failures == 0 && valid_failed == 0 && reference.ok) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) { return Run(argc, argv); }

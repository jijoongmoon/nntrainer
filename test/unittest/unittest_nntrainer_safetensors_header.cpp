// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_safetensors_header.cpp
 * @date   14 May 2026
 * @brief  Unit tests for the .safetensors header parser.
 *
 *         The tests synthesise a tiny in-memory safetensors blob (8-byte
 *         LE header_size + UTF-8 JSON header + raw data) and round-trip
 *         it through parseSafetensorsHeader(). This pins down the parser
 *         independently of any real model file so it can be developed
 *         offline.
 *
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cstdint>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include <safetensors_header.h>

namespace {

// Pack `<u64 LE header_size><json_header_bytes><data_bytes>` into a single
// in-memory buffer so we can feed parseSafetensorsHeader() without touching
// disk.
std::vector<uint8_t> buildBlob(const std::string &json_header,
                               size_t data_bytes = 0) {
  std::vector<uint8_t> out;
  out.resize(8 + json_header.size() + data_bytes, 0);
  const uint64_t header_size = json_header.size();
  for (int i = 0; i < 8; ++i) {
    out[i] = static_cast<uint8_t>((header_size >> (8 * i)) & 0xFF);
  }
  std::memcpy(out.data() + 8, json_header.data(), json_header.size());
  return out;
}

} // namespace

TEST(SafetensorsHeader, parses_single_tensor_minimal) {
  // One tensor, F32, 2x3 shape, 24 bytes of data offsets.
  const std::string h = R"({"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}})";
  auto blob = buildBlob(h, 24);

  auto hdr = nntrainer::parseSafetensorsHeader(blob.data(), blob.size());

  EXPECT_EQ(hdr.header_size, h.size());
  EXPECT_EQ(hdr.data_block_offset, 8u + h.size());
  ASSERT_EQ(hdr.tensors.size(), 1u);
  const auto &t = hdr.tensors.at("weight");
  EXPECT_EQ(t.dtype, nntrainer::TensorDim::DataType::FP32);
  EXPECT_EQ(t.dtype_raw, "F32");
  ASSERT_EQ(t.shape.size(), 2u);
  EXPECT_EQ(t.shape[0], 2u);
  EXPECT_EQ(t.shape[1], 3u);
  EXPECT_EQ(t.data_offset_start, 0u);
  EXPECT_EQ(t.data_offset_end, 24u);
}

TEST(SafetensorsHeader, parses_multiple_tensors_with_mixed_dtypes) {
  // FC.weight is F32 (4*8=32 bytes), FC.bias is F16 (8 bytes). These are
  // exactly the per-weight dtype mix Chunk A's weight_dtype_map is meant
  // to consume.
  const std::string h =
    R"({"fc.weight":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]},)"
    R"("fc.bias":{"dtype":"F16","shape":[4],"data_offsets":[32,40]}})";
  auto blob = buildBlob(h, 40);

  auto hdr = nntrainer::parseSafetensorsHeader(blob.data(), blob.size());

  ASSERT_EQ(hdr.tensors.size(), 2u);
  EXPECT_EQ(hdr.tensors.at("fc.weight").dtype,
            nntrainer::TensorDim::DataType::FP32);
  EXPECT_EQ(hdr.tensors.at("fc.bias").dtype,
            nntrainer::TensorDim::DataType::FP16);
  EXPECT_EQ(hdr.tensors.at("fc.weight").data_offset_end, 32u);
  EXPECT_EQ(hdr.tensors.at("fc.bias").data_offset_start, 32u);
  EXPECT_EQ(hdr.tensors.at("fc.bias").data_offset_end, 40u);
}

TEST(SafetensorsHeader, parses_metadata_block) {
  const std::string h =
    R"({"__metadata__":{"format":"pt","model":"toy"},)"
    R"("w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
  auto blob = buildBlob(h, 4);

  auto hdr = nntrainer::parseSafetensorsHeader(blob.data(), blob.size());

  ASSERT_EQ(hdr.metadata.size(), 2u);
  EXPECT_EQ(hdr.metadata.at("format"), "pt");
  EXPECT_EQ(hdr.metadata.at("model"), "toy");
  ASSERT_EQ(hdr.tensors.size(), 1u);
  EXPECT_TRUE(hdr.tensors.count("w"));
}

TEST(SafetensorsHeader, dtype_mapping_known_tokens) {
  using DT = nntrainer::TensorDim::DataType;
  EXPECT_EQ(nntrainer::safetensorsDtypeToNntrainer("F32"), DT::FP32);
  EXPECT_EQ(nntrainer::safetensorsDtypeToNntrainer("F16"), DT::FP16);
  EXPECT_EQ(nntrainer::safetensorsDtypeToNntrainer("I8"),  DT::QINT8);
  EXPECT_EQ(nntrainer::safetensorsDtypeToNntrainer("U8"),  DT::UINT8);
  EXPECT_EQ(nntrainer::safetensorsDtypeToNntrainer("U16"), DT::UINT16);
  EXPECT_EQ(nntrainer::safetensorsDtypeToNntrainer("U32"), DT::UINT32);
}

TEST(SafetensorsHeader, dtype_mapping_unknown_is_NONE) {
  using DT = nntrainer::TensorDim::DataType;
  // BF16, F64, I32, BOOL etc. aren't in nntrainer's TensorDim::DataType.
  // Parser must surface them as NONE (caller can decide what to do — usually
  // dequantise / re-quantise via the converter pipeline).
  EXPECT_EQ(nntrainer::safetensorsDtypeToNntrainer("BF16"), DT::NONE);
  EXPECT_EQ(nntrainer::safetensorsDtypeToNntrainer("F64"),  DT::NONE);
  EXPECT_EQ(nntrainer::safetensorsDtypeToNntrainer("I32"),  DT::NONE);
}

TEST(SafetensorsHeader, header_keeps_dtype_raw_even_when_unknown) {
  // Even if we can't translate, the raw token must be preserved so the
  // loader can log a clear message ("safetensors says BF16 but nntrainer
  // can't consume it directly").
  const std::string h =
    R"({"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}})";
  auto blob = buildBlob(h, 2);

  auto hdr = nntrainer::parseSafetensorsHeader(blob.data(), blob.size());
  ASSERT_EQ(hdr.tensors.size(), 1u);
  EXPECT_EQ(hdr.tensors.at("w").dtype,
            nntrainer::TensorDim::DataType::NONE);
  EXPECT_EQ(hdr.tensors.at("w").dtype_raw, "BF16");
}

TEST(SafetensorsHeader, throws_on_too_small_buffer) {
  std::vector<uint8_t> tiny(4, 0);
  EXPECT_THROW(
    nntrainer::parseSafetensorsHeader(tiny.data(), tiny.size()),
    std::runtime_error);
}

TEST(SafetensorsHeader, throws_on_header_overflow) {
  // Claim a header_size that exceeds the buffer length.
  std::vector<uint8_t> bad(16, 0);
  uint64_t huge = 1000;
  for (int i = 0; i < 8; ++i)
    bad[i] = static_cast<uint8_t>((huge >> (8 * i)) & 0xFF);
  // Even if the bytes inside happened to be valid JSON, header_size>file
  // must be rejected before parsing.
  EXPECT_THROW(
    nntrainer::parseSafetensorsHeader(bad.data(), bad.size()),
    std::runtime_error);
}

TEST(SafetensorsHeader, throws_on_malformed_json) {
  const std::string h = R"({"w":{"dtype":"F32","shape":[1]"data_offsets":[0,4]}})";
  // missing comma after "shape":[1]
  auto blob = buildBlob(h, 4);
  EXPECT_THROW(
    nntrainer::parseSafetensorsHeader(blob.data(), blob.size()),
    std::runtime_error);
}

TEST(SafetensorsHeader, parses_from_file_round_trip) {
  // Round-trip through an actual tmp file — verifies the file-mode entry
  // point reads the header bytes correctly.
  const std::string h =
    R"({"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})";
  auto blob = buildBlob(h, 16);

  const std::string tmp = "/tmp/nntrainer_safetensors_test.bin";
  {
    std::ofstream out(tmp, std::ios::binary);
    out.write(reinterpret_cast<const char *>(blob.data()),
              static_cast<std::streamsize>(blob.size()));
  }

  auto hdr = nntrainer::parseSafetensorsHeaderFromFile(tmp);
  ASSERT_EQ(hdr.tensors.size(), 1u);
  EXPECT_EQ(hdr.tensors.at("weight").dtype,
            nntrainer::TensorDim::DataType::FP32);
  EXPECT_EQ(hdr.tensors.at("weight").shape.size(), 2u);
  std::remove(tmp.c_str());
}

int main(int argc, char **argv) {
  int result = -1;
  try {
    ::testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS" << std::endl;
  }
  return result;
}

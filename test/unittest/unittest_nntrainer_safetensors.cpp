// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_nntrainer_safetensors.cpp
 * @date   24 April 2026
 * @brief  Round-trip tests for MODEL_FORMAT_SAFETENSORS save/load.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <layer.h>
#include <model.h>
#include <nntrainer_error.h>
#include <nntrainer_test_util.h>

namespace {

/**
 * @brief Build a tiny {input -> fully_connected} model for roundtrip tests.
 */
static std::unique_ptr<ml::train::Model>
buildTinyModel(const std::string &input_shape, unsigned int unit) {
  auto model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"batch_size=1"});
  model->addLayer(ml::train::createLayer(
    "input", {"name=input", "input_shape=" + input_shape}));
  model->addLayer(ml::train::createLayer(
    "fully_connected",
    {"name=fc", "unit=" + std::to_string(unit), "bias_initializer=zeros"}));
  // Use INFERENCE mode: save/load roundtrip does not need a loss layer.
  EXPECT_EQ(model->compile(ml::train::ExecutionMode::INFERENCE), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(ml::train::ExecutionMode::INFERENCE),
            ML_ERROR_NONE);
  return model;
}

/**
 * @brief Compare two files byte-by-byte.
 */
static bool filesBitEqual(const std::string &a, const std::string &b) {
  std::ifstream fa(a, std::ios::binary);
  std::ifstream fb(b, std::ios::binary);
  if (!fa.good() || !fb.good())
    return false;
  std::vector<char> va((std::istreambuf_iterator<char>(fa)),
                       std::istreambuf_iterator<char>());
  std::vector<char> vb((std::istreambuf_iterator<char>(fb)),
                       std::istreambuf_iterator<char>());
  return va == vb;
}

} // namespace

/**
 * @brief save(SAFETENSORS) produces a readable file that load(SAFETENSORS)
 *        consumes without error on a freshly built identical model.
 */
TEST(nntrainer_safetensors, save_and_load_roundtrip_p) {
  const std::string path = "unittest_safetensors_roundtrip.safetensors";
  std::remove(path.c_str());

  auto writer = buildTinyModel("1:1:4", 2);
  ASSERT_NO_THROW(
    writer->save(path, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS));

  // File must exist and carry the 8B header-size prefix at minimum.
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(f.good());
  EXPECT_GT(static_cast<size_t>(f.tellg()), sizeof(uint64_t));
  f.close();

  auto reader = buildTinyModel("1:1:4", 2);
  EXPECT_NO_THROW(
    reader->load(path, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS));

  std::remove(path.c_str());
}

/**
 * @brief Re-saving a just-loaded safetensors file must produce a byte-for-byte
 *        identical payload (format is deterministic).
 */
TEST(nntrainer_safetensors, resave_is_byte_identical_p) {
  const std::string first = "unittest_safetensors_first.safetensors";
  const std::string second = "unittest_safetensors_second.safetensors";
  std::remove(first.c_str());
  std::remove(second.c_str());

  auto a = buildTinyModel("1:1:4", 2);
  ASSERT_NO_THROW(
    a->save(first, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS));

  auto b = buildTinyModel("1:1:4", 2);
  ASSERT_NO_THROW(
    b->load(first, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS));
  ASSERT_NO_THROW(
    b->save(second, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS));

  EXPECT_TRUE(filesBitEqual(first, second));

  std::remove(first.c_str());
  std::remove(second.c_str());
}

/**
 * @brief The on-disk layout must start with an 8-byte little-endian header
 *        size followed by a JSON object that references nntrainer metadata.
 */
TEST(nntrainer_safetensors, header_layout_p) {
  const std::string path = "unittest_safetensors_header.safetensors";
  std::remove(path.c_str());

  auto model = buildTinyModel("1:1:4", 3);
  ASSERT_NO_THROW(
    model->save(path, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS));

  std::ifstream f(path, std::ios::binary);
  ASSERT_TRUE(f.good());

  uint64_t header_size = 0;
  f.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
  ASSERT_EQ(f.gcount(), static_cast<std::streamsize>(sizeof(header_size)));
  ASSERT_GT(header_size, 0u);

  std::string header_json(static_cast<size_t>(header_size), '\0');
  f.read(&header_json[0], static_cast<std::streamsize>(header_size));
  ASSERT_EQ(f.gcount(), static_cast<std::streamsize>(header_size));

  EXPECT_NE(header_json.find("__metadata__"), std::string::npos);
  EXPECT_NE(header_json.find("\"format\":\"nntrainer\""), std::string::npos);
  EXPECT_NE(header_json.find("data_offsets"), std::string::npos);

  std::remove(path.c_str());
}

/**
 * @brief Loading a non-existent safetensors file must raise an exception
 *        rather than silently succeeding or crashing.
 */
TEST(nntrainer_safetensors, load_missing_file_n) {
  const std::string path = "unittest_safetensors_does_not_exist.safetensors";
  std::remove(path.c_str()); // ensure absent

  auto reader = buildTinyModel("1:1:4", 2);
  EXPECT_ANY_THROW(
    reader->load(path, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS));
}

/**
 * @brief Test entry point.
 */
int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}

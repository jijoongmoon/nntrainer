// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_q8_0_tensor.cpp
 * @date   14 May 2026
 * @brief  Unit tests for Q8_0_Tensor + the surrounding factory/dispatch
 *         plumbing (TensorDim::DataType::Q8_0, QScheme::Q8_0,
 *         ModelTensorDataTypeInfo::WQ80A32). Mirrors what Q4_0_Tensor
 *         already provides for the 4-bit path.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude (mirror of q4_0 plumbing by Donghyeon Jeong)
 * @bug    No known bugs except for NYI items
 */

#include <cstring>
#include <gtest/gtest.h>

#include <q8_0_tensor.h>
#include <quantizer.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace {

// 32 elements per Q8_0 block × 34 bytes/block. Pick a tensor wide enough to
// span multiple blocks so we exercise multi-block byte-size accounting.
constexpr unsigned int H = 4;
constexpr unsigned int W = 64;
constexpr unsigned int NUM_BLOCKS = (H * W) / QK8_0;
constexpr size_t EXPECTED_BYTES = NUM_BLOCKS * sizeof(nntrainer::block_q8_0);

nntrainer::TensorDim q80_dim() {
  return nntrainer::TensorDim(
    1, 1, H, W,
    {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::Q8_0});
}

// Build a deterministic Q8_0 buffer where each block has scale=1.0 (fp16 0x3C00)
// and qs[i] = (block_idx + i) % 256 cast to int8.
std::vector<uint8_t> makeSyntheticQ80Buffer() {
  std::vector<uint8_t> buf(EXPECTED_BYTES, 0);
  for (unsigned int b = 0; b < NUM_BLOCKS; ++b) {
    uint8_t *block = buf.data() + b * sizeof(nntrainer::block_q8_0);
    // fp16 1.0 = 0x3C00 (little-endian)
    block[0] = 0x00;
    block[1] = 0x3C;
    for (int l = 0; l < 32; ++l) {
      block[2 + l] = static_cast<uint8_t>((b + l) & 0xFF);
    }
  }
  return buf;
}

} // namespace

TEST(Q8_0_Tensor, datatype_enum_and_block_layout) {
  // The added enum + struct layout must be in sync with GGML's block_q8_0:
  //  - 2-byte fp16 scale + 32 int8 quants = 34 bytes.
  EXPECT_EQ(static_cast<int>(nntrainer::TensorDim::DataType::Q8_0),
            // Q8_0 was inserted right after Q4_0 in the enum.
            static_cast<int>(nntrainer::TensorDim::DataType::Q4_0) + 1);
  EXPECT_EQ(QK8_0, 32u);
  EXPECT_EQ(sizeof(nntrainer::block_q8_0), 34u);
  EXPECT_EQ(sizeof(nntrainer::block_q8_0), 34u);
}

TEST(Q8_0_Tensor, qscheme_value) {
  // Default-constructed Q8_0_Tensor should report QScheme::Q8_0.
  nntrainer::Q8_0_Tensor t("test", nntrainer::Tformat::NCHW);
  EXPECT_EQ(t.q_scheme(), nntrainer::QScheme::Q8_0);
}

TEST(Q8_0_Tensor, rejects_invalid_dim) {
  // batch != 1
  EXPECT_THROW(
    nntrainer::Q8_0_Tensor(
      nntrainer::TensorDim(
        2, 1, H, W,
        {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::Q8_0}),
      false),
    std::invalid_argument);

  // channel != 1
  EXPECT_THROW(
    nntrainer::Q8_0_Tensor(
      nntrainer::TensorDim(
        1, 2, H, W,
        {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::Q8_0}),
      false),
    std::invalid_argument);

  // width not divisible by 32
  EXPECT_THROW(
    nntrainer::Q8_0_Tensor(
      nntrainer::TensorDim(
        1, 1, H, 17,
        {nntrainer::Tformat::NCHW, nntrainer::TensorDim::DataType::Q8_0}),
      false),
    std::invalid_argument);
}

TEST(Q8_0_Tensor, allocate_and_size_accounting) {
  nntrainer::Q8_0_Tensor t(q80_dim(), /*alloc_now=*/true);
  EXPECT_EQ(t.size(), EXPECTED_BYTES);
  EXPECT_EQ(t.getMemoryBytes(), EXPECTED_BYTES);

  // getData must return a valid (non-null), readable buffer after allocate().
  void *data = t.getData();
  ASSERT_NE(data, nullptr);
}

TEST(Q8_0_Tensor, copy_from_buffer_is_byte_exact) {
  auto src = makeSyntheticQ80Buffer();
  nntrainer::Q8_0_Tensor t(q80_dim(), src.data());

  // After construction with a buffer, the tensor contents must match byte-wise.
  const uint8_t *dst = reinterpret_cast<const uint8_t *>(t.getData());
  ASSERT_NE(dst, nullptr);
  for (size_t i = 0; i < src.size(); ++i) {
    ASSERT_EQ(dst[i], src[i]) << "byte mismatch at offset " << i;
  }
}

TEST(Q8_0_Tensor, factory_dispatch_creates_q8_0_tensor) {
  // Going through nntrainer::Tensor (not Q8_0_Tensor directly) should still
  // produce a Q8_0-backed tensor that reports size()/getMemoryBytes() in
  // packed-Q8_0 units.
  nntrainer::Tensor t(q80_dim(), /*alloc_now=*/true);
  EXPECT_EQ(t.getDataType(), nntrainer::TensorDim::DataType::Q8_0);
  EXPECT_EQ(t.size(), EXPECTED_BYTES);
  EXPECT_EQ(t.getMemoryBytes(), EXPECTED_BYTES);
}

TEST(Q8_0_Tensor, set_zero_zeroes_every_byte) {
  auto src = makeSyntheticQ80Buffer();
  nntrainer::Q8_0_Tensor t(q80_dim(), src.data());
  t.setZero();

  const uint8_t *dst = reinterpret_cast<const uint8_t *>(t.getData());
  for (size_t i = 0; i < EXPECTED_BYTES; ++i) {
    ASSERT_EQ(dst[i], 0u) << "non-zero byte at offset " << i;
  }
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

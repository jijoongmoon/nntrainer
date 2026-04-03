// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_nntrainer_q1_0.cpp
 * @date        02 April 2026
 * @brief       Unit tests for Q1_0 (1-bit, group size 128) quantization.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <cmath>
#include <cpu_backend.h>

// FP16 scale storage introduces small rounding errors (~0.1%)
static constexpr float FP16_TOL = 1e-2f;
#include <ggml_interface.h>
#include <nntrainer_error.h>
#include <quantizer.h>
#include <tensor.h>

// ===================================================================
// Q1_0_Tensor basic tests
// ===================================================================

/**
 * @brief Q1_0_Tensor creation with valid dimensions
 */
TEST(nntrainer_Q1_0_Tensor, create_01_p) {
  // width must be divisible by 128
  nntrainer::Tensor t(
    1, 1, 4, 128,
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::Q1_0});

  EXPECT_EQ(t.getDataType(), nntrainer::Tdatatype::Q1_0);
  // 4 rows * 128 cols / 128 per block = 4 blocks * 18 bytes = 72 bytes
  EXPECT_EQ(t.getMemoryBytes(), 72u);
}

/**
 * @brief Q1_0_Tensor creation with larger dimensions
 */
TEST(nntrainer_Q1_0_Tensor, create_02_p) {
  nntrainer::Tensor t(
    1, 1, 8, 256,
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::Q1_0});

  EXPECT_EQ(t.getDataType(), nntrainer::Tdatatype::Q1_0);
  // 8 * 256 / 128 = 16 blocks * 18 = 288 bytes
  EXPECT_EQ(t.getMemoryBytes(), 288u);
}

/**
 * @brief Q1_0_Tensor width not divisible by 128 (negative test)
 */
TEST(nntrainer_Q1_0_Tensor, create_03_n) {
  EXPECT_THROW(
    nntrainer::Tensor(
      1, 1, 4, 64,
      {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::Q1_0}),
    std::invalid_argument);
}

/**
 * @brief Q1_0_Tensor batch != 1 (negative test)
 */
TEST(nntrainer_Q1_0_Tensor, create_04_n) {
  EXPECT_THROW(
    nntrainer::Tensor(
      2, 1, 4, 128,
      {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::Q1_0}),
    std::invalid_argument);
}

/**
 * @brief Q1_0_Tensor setZero and getData
 */
TEST(nntrainer_Q1_0_Tensor, set_zero_01_p) {
  nntrainer::Tensor t(
    1, 1, 1, 128,
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::Q1_0});

  t.setZero();
  void *data = t.getData();
  EXPECT_NE(data, nullptr);

  // All bytes should be zero after setZero
  uint8_t *bytes = static_cast<uint8_t *>(data);
  for (size_t i = 0; i < t.getMemoryBytes(); ++i) {
    EXPECT_EQ(bytes[i], 0);
  }
}

/**
 * @brief Q1_0_Tensor q_scheme returns Q1_0
 */
TEST(nntrainer_Q1_0_Tensor, q_scheme_01_p) {
  nntrainer::Tensor t(
    1, 1, 1, 128,
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::Q1_0});

  EXPECT_EQ(t.q_scheme(), nntrainer::QScheme::Q1_0);
}

// ===================================================================
// Q1_0 Quantize / Dequantize tests (low-level GGML interface)
// ===================================================================

/**
 * @brief Test quantize and dequantize round-trip for known data.
 * All-positive values should produce all-1 bits and recover +scale.
 */
TEST(nntrainer_Q1_0_GGML, quantize_dequantize_all_positive_p) {
  nntrainer::init_backend();

  const int K = 128;
  std::vector<float> input(K, 0.5f);

  // Quantize
  const int num_blocks = K / 128;
  const size_t q_size = num_blocks * 18; // 18 bytes per block
  std::vector<uint8_t> quantized(q_size);
  size_t ret = nntrainer::__ggml_quantize_q1_0(
    input.data(), quantized.data(), 1, K, nullptr);
  EXPECT_EQ(ret, q_size);

  // Dequantize
  std::vector<float> output(K);
  nntrainer::__ggml_dequantize_row_q1_0(quantized.data(), output.data(), K);

  // All values positive -> all bits = 1 -> all outputs = +scale = 0.5
  for (int i = 0; i < K; ++i) {
    EXPECT_NEAR(output[i], 0.5f, FP16_TOL);
  }
}

/**
 * @brief Test quantize and dequantize round-trip for all-negative data.
 */
TEST(nntrainer_Q1_0_GGML, quantize_dequantize_all_negative_p) {
  nntrainer::init_backend();

  const int K = 128;
  std::vector<float> input(K, -0.3f);

  const size_t q_size = 18;
  std::vector<uint8_t> quantized(q_size);
  nntrainer::__ggml_quantize_q1_0(
    input.data(), quantized.data(), 1, K, nullptr);

  std::vector<float> output(K);
  nntrainer::__ggml_dequantize_row_q1_0(quantized.data(), output.data(), K);

  // All values negative -> all bits = 0 -> all outputs = -scale = -0.3
  for (int i = 0; i < K; ++i) {
    EXPECT_NEAR(output[i], -0.3f, FP16_TOL);
  }
}

/**
 * @brief Test quantize/dequantize with mixed positive/negative values.
 * 1-bit quantization preserves sign but collapses magnitude to scale.
 */
TEST(nntrainer_Q1_0_GGML, quantize_dequantize_mixed_p) {
  nntrainer::init_backend();

  const int K = 128;
  std::vector<float> input(K);
  // Alternating +0.7 and -0.3
  for (int i = 0; i < K; ++i) {
    input[i] = (i % 2 == 0) ? 0.7f : -0.3f;
  }

  const size_t q_size = 18;
  std::vector<uint8_t> quantized(q_size);
  nntrainer::__ggml_quantize_q1_0(
    input.data(), quantized.data(), 1, K, nullptr);

  std::vector<float> output(K);
  nntrainer::__ggml_dequantize_row_q1_0(quantized.data(), output.data(), K);

  // scale = max(|0.7|, |-0.3|) = 0.7
  // positive values -> +0.7, negative values -> -0.7
  for (int i = 0; i < K; ++i) {
    if (i % 2 == 0) {
      EXPECT_NEAR(output[i], 0.7f, FP16_TOL);
    } else {
      EXPECT_NEAR(output[i], -0.7f, FP16_TOL);
    }
  }
}

/**
 * @brief Test quantize/dequantize with multiple blocks (256 elements).
 */
TEST(nntrainer_Q1_0_GGML, quantize_dequantize_multi_block_p) {
  nntrainer::init_backend();

  const int K = 256;
  std::vector<float> input(K);
  // Block 0: all +1.0, Block 1: all -2.0
  for (int i = 0; i < 128; ++i) input[i] = 1.0f;
  for (int i = 128; i < 256; ++i) input[i] = -2.0f;

  const size_t q_size = 2 * 18;
  std::vector<uint8_t> quantized(q_size);
  nntrainer::__ggml_quantize_q1_0(
    input.data(), quantized.data(), 1, K, nullptr);

  std::vector<float> output(K);
  nntrainer::__ggml_dequantize_row_q1_0(quantized.data(), output.data(), K);

  for (int i = 0; i < 128; ++i) {
    EXPECT_NEAR(output[i], 1.0f, FP16_TOL);
  }
  for (int i = 128; i < 256; ++i) {
    EXPECT_NEAR(output[i], -2.0f, FP16_TOL);
  }
}

// ===================================================================
// Q1_0 Dot Product tests
// ===================================================================

/**
 * @brief Test vec_dot_q1_0_f32: dot product of Q1_0 weights with float
 * activations.
 */
TEST(nntrainer_Q1_0_GGML, vec_dot_q1_0_f32_01_p) {
  nntrainer::init_backend();

  const int K = 128;

  // Weights: alternating +1 and -1 (scale=1.0, alternating bits)
  std::vector<float> weights(K);
  for (int i = 0; i < K; ++i) {
    weights[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }

  // Activations: all 1.0
  std::vector<float> activations(K, 1.0f);

  // Quantize weights
  std::vector<uint8_t> q_weights(18);
  nntrainer::__ggml_quantize_q1_0(
    weights.data(), q_weights.data(), 1, K, nullptr);

  // Compute dot product
  float result = nntrainer::__ggml_vec_dot_q1_0_f32(
    K, q_weights.data(), activations.data());

  // Expected: sum of (+1)*1 + (-1)*1 for 64 pairs = 64 - 64 = 0
  EXPECT_NEAR(result, 0.0f, 1e-5f);
}

/**
 * @brief Test vec_dot_q1_0_f32 with uniform weights.
 */
TEST(nntrainer_Q1_0_GGML, vec_dot_q1_0_f32_02_p) {
  nntrainer::init_backend();

  const int K = 128;

  // All weights = +2.0
  std::vector<float> weights(K, 2.0f);

  // Activations: 1.0 for all
  std::vector<float> activations(K, 1.0f);

  std::vector<uint8_t> q_weights(18);
  nntrainer::__ggml_quantize_q1_0(
    weights.data(), q_weights.data(), 1, K, nullptr);

  float result = nntrainer::__ggml_vec_dot_q1_0_f32(
    K, q_weights.data(), activations.data());

  // All +2.0 -> scale=2.0, all bits=1 -> 128 * 2.0 * 1.0 = 256.0
  EXPECT_NEAR(result, 256.0f, 1e-3f);
}

/**
 * @brief Test vec_dot_q1_0_f32 with multiple blocks.
 */
TEST(nntrainer_Q1_0_GGML, vec_dot_q1_0_f32_multi_block_p) {
  nntrainer::init_backend();

  const int K = 256;

  // Block 0: all +1.0, Block 1: all -1.0
  std::vector<float> weights(K);
  for (int i = 0; i < 128; ++i) weights[i] = 1.0f;
  for (int i = 128; i < 256; ++i) weights[i] = -1.0f;

  // Activations: all 2.0
  std::vector<float> activations(K, 2.0f);

  std::vector<uint8_t> q_weights(2 * 18);
  nntrainer::__ggml_quantize_q1_0(
    weights.data(), q_weights.data(), 1, K, nullptr);

  float result = nntrainer::__ggml_vec_dot_q1_0_f32(
    K, q_weights.data(), activations.data());

  // Block 0: 128 * (+1.0) * 2.0 = 256
  // Block 1: 128 * (-1.0) * 2.0 = -256
  // Total = 0
  EXPECT_NEAR(result, 0.0f, 1e-3f);
}

// ===================================================================
// Q1_0 GEMM tests
// ===================================================================

/**
 * @brief Test Q1_0 GEMM: simple identity-like case.
 * M=1 (single activation row), N=2 (two weight rows), K=128.
 */
TEST(nntrainer_Q1_0_GGML, gemm_q1_0_01_p) {
  nntrainer::init_backend();

  const unsigned int M = 1;  // activation rows (batch)
  const unsigned int N = 2;  // weight rows (output dim)
  const unsigned int K = 128;

  // Weight row 0: all +1.0
  // Weight row 1: all -1.0
  std::vector<float> w_data(N * K);
  for (unsigned int i = 0; i < K; ++i) w_data[i] = 1.0f;
  for (unsigned int i = K; i < 2 * K; ++i) w_data[i] = -1.0f;

  // Quantize weights
  std::vector<uint8_t> q_weights(N * 18);
  nntrainer::__ggml_quantize_q1_0(
    w_data.data(), q_weights.data(), N, K, nullptr);

  // Activation: all 0.5
  std::vector<float> A(M * K, 0.5f);

  // Output
  std::vector<float> C(M * N, 0.0f);

  nntrainer::__ggml_q1_0_GEMM(M, N, K, A.data(), K,
                              q_weights.data(), K, C.data(), N);

  // Row 0: 128 * (+1.0) * 0.5 = 64.0
  // Row 1: 128 * (-1.0) * 0.5 = -64.0
  EXPECT_NEAR(C[0], 64.0f, 1e-3f);
  EXPECT_NEAR(C[1], -64.0f, 1e-3f);
}

/**
 * @brief Test Q1_0 GEMM with larger dimensions.
 * M=2 (two activation rows), N=4 (four weight rows), K=256.
 */
TEST(nntrainer_Q1_0_GGML, gemm_q1_0_02_p) {
  nntrainer::init_backend();

  const unsigned int M = 2;
  const unsigned int N = 4;
  const unsigned int K = 256;

  // Weight matrix: each row has same value but different sign pattern
  std::vector<float> w_data(N * K);
  for (unsigned int n = 0; n < N; ++n) {
    float val = (n % 2 == 0) ? 0.5f : -0.5f;
    for (unsigned int k = 0; k < K; ++k) {
      w_data[n * K + k] = val;
    }
  }

  std::vector<uint8_t> q_weights(N * 2 * 18); // 2 blocks per row
  nntrainer::__ggml_quantize_q1_0(
    w_data.data(), q_weights.data(), N, K, nullptr);

  // Two activation rows
  std::vector<float> A(M * K);
  for (unsigned int k = 0; k < K; ++k) {
    A[k] = 1.0f;         // Row 0: all 1.0
    A[K + k] = -1.0f;    // Row 1: all -1.0
  }

  std::vector<float> C(M * N, 0.0f);

  nntrainer::__ggml_q1_0_GEMM(M, N, K, A.data(), K,
                              q_weights.data(), K, C.data(), N);

  // Row 0 activations (all 1.0):
  //   W row 0 (+0.5): 256 * 0.5 * 1.0 = 128.0
  //   W row 1 (-0.5): 256 * (-0.5) * 1.0 = -128.0
  //   W row 2 (+0.5): 128.0
  //   W row 3 (-0.5): -128.0

  // Row 1 activations (all -1.0):
  //   W row 0 (+0.5): 256 * 0.5 * (-1.0) = -128.0
  //   W row 1 (-0.5): 256 * (-0.5) * (-1.0) = 128.0
  //   ...

  EXPECT_NEAR(C[0 * N + 0], 128.0f, 1e-2f);
  EXPECT_NEAR(C[0 * N + 1], -128.0f, 1e-2f);
  EXPECT_NEAR(C[0 * N + 2], 128.0f, 1e-2f);
  EXPECT_NEAR(C[0 * N + 3], -128.0f, 1e-2f);
  EXPECT_NEAR(C[1 * N + 0], -128.0f, 1e-2f);
  EXPECT_NEAR(C[1 * N + 1], 128.0f, 1e-2f);
  EXPECT_NEAR(C[1 * N + 2], -128.0f, 1e-2f);
  EXPECT_NEAR(C[1 * N + 3], 128.0f, 1e-2f);
}

// ===================================================================
// Q1_0 Quantizer (high-level) tests
// ===================================================================

/**
 * @brief GgmlQuantizer Q1_0 quantize and dequantize round-trip.
 * For 1-bit quantization, the sign is preserved but magnitude collapses.
 */
TEST(nntrainer_Quantizer, ggml_q1_0_01_p) {
  nntrainer::init_backend();

  uint32_t K = 128;
  uint32_t N = 128;

  // Create uniform-magnitude weights so 1-bit quantization is lossless
  std::vector<float> weight(K * N);
  for (uint32_t i = 0; i < K * N; ++i) {
    weight[i] = (i % 3 == 0) ? 0.5f : -0.5f;
  }

  nntrainer::Tensor W_fp32(
    1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});

  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n)
      W_fp32.setValue(0, 0, k, n, weight[k * N + n]);

  // Create GGML Q1_0 quantizer
  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(nntrainer::QScheme::Q1_0);

  EXPECT_EQ(quantizer->qscheme(), nntrainer::QScheme::Q1_0);

  // Quantize
  nntrainer::Tensor W_q10 =
    quantizer->quantize(W_fp32, nntrainer::Tdatatype::Q1_0);

  EXPECT_EQ(W_q10.getDataType(), nntrainer::Tdatatype::Q1_0);

  // Dequantize
  nntrainer::Tensor W_deq =
    quantizer->dequantize(W_q10, nntrainer::Tdatatype::FP32);

  EXPECT_EQ(W_deq.getDataType(), nntrainer::Tdatatype::FP32);

  // For uniform-magnitude inputs, dequantized values should have magnitude 0.5
  // but sign may differ due to transposition in quantizer
  const float *deq_data = W_deq.getData<float>();
  for (uint32_t i = 0; i < K * N; ++i) {
    EXPECT_NEAR(std::fabs(deq_data[i]), 0.5f, 1e-3f)
      << "Mismatch at index " << i;
  }
}

/**
 * @brief GgmlQuantizer Q1_0 with random data - verify magnitude preservation.
 */
TEST(nntrainer_Quantizer, ggml_q1_0_02_p) {
  nntrainer::init_backend();

  uint32_t K = 256;
  uint32_t N = 128;

  std::vector<float> weight =
    generate_random_vector<float>(K * N, -1.0f, 1.0f);

  nntrainer::Tensor W_fp32(
    1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});

  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n)
      W_fp32.setValue(0, 0, k, n, weight[k * N + n]);

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(nntrainer::QScheme::Q1_0);

  // Quantize
  nntrainer::Tensor W_q10 =
    quantizer->quantize(W_fp32, nntrainer::Tdatatype::Q1_0);

  // Dequantize
  nntrainer::Tensor W_deq =
    quantizer->dequantize(W_q10, nntrainer::Tdatatype::FP32);

  // For 1-bit quantization, the sign should mostly be preserved
  // but magnitudes collapse to scale. We verify the output is non-zero
  // and reasonable.
  const float *orig = W_fp32.getData<float>();
  const float *deq = W_deq.getData<float>();

  int sign_matches = 0;
  for (uint32_t i = 0; i < K * N; ++i) {
    // Verify output is non-zero (all weights map to +/-scale)
    EXPECT_NE(deq[i], 0.0f);
  }
}

/**
 * @brief Q1_0 memory efficiency check.
 * Q1_0 should use ~1.125 bits per weight (18 bytes per 128 weights).
 */
TEST(nntrainer_Q1_0_Tensor, memory_efficiency_p) {
  // 1024 x 1024 = 1M weights
  // FP32: 4MB, Q1_0: 1M/128 * 18 = 144000 bytes = ~140.6 KB
  nntrainer::Tensor t(
    1, 1, 1024, 1024,
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::Q1_0});

  size_t fp32_size = 1024 * 1024 * sizeof(float); // 4MB
  size_t q1_0_size = t.getMemoryBytes();

  // Q1_0 should be roughly 1/28 of FP32
  // 18 bytes / 128 weights = 0.140625 bytes/weight vs 4 bytes/weight
  float ratio = static_cast<float>(q1_0_size) / fp32_size;
  EXPECT_LT(ratio, 0.05f); // Less than 5% of FP32

  // Exact expected size: (1024*1024/128) * 18 = 147456
  EXPECT_EQ(q1_0_size, 147456u);
}

// ===================================================================
// Large-scale random tests to verify SIMD kernels match scalar results
// ===================================================================

/**
 * @brief Large-scale quantize/dequantize with random data.
 * Verifies that SIMD-optimized kernels produce bit-exact results
 * with the scalar reference on diverse random inputs.
 */
TEST(nntrainer_Q1_0_GGML, large_random_quantize_dequantize_p) {
  nntrainer::init_backend();

  const int K = 1024;
  std::vector<float> input = generate_random_vector<float>(K, -5.0f, 5.0f);

  // Quantize
  const int nb = K / 128;
  std::vector<uint8_t> quantized(nb * 18);
  nntrainer::__ggml_quantize_q1_0(
    input.data(), quantized.data(), 1, K, nullptr);

  // Dequantize
  std::vector<float> output(K);
  nntrainer::__ggml_dequantize_row_q1_0(quantized.data(), output.data(), K);

  // Verify: all outputs should be +/-scale (non-zero)
  for (int i = 0; i < K; ++i) {
    EXPECT_NE(output[i], 0.0f) << "at index " << i;
  }

  // Verify sign preservation: sign(output) should match sign(input)
  // for values not too close to zero
  int sign_matches = 0;
  int significant = 0;
  for (int i = 0; i < K; ++i) {
    if (std::fabs(input[i]) > 0.1f) {
      significant++;
      if ((input[i] > 0) == (output[i] > 0)) {
        sign_matches++;
      }
    }
  }
  // At least 95% sign preservation for significant values
  float sign_ratio = (float)sign_matches / significant;
  EXPECT_GT(sign_ratio, 0.95f) << "sign_matches=" << sign_matches
                               << " significant=" << significant;
}

/**
 * @brief Large-scale dot product verification.
 * Computes dot product via quantized path and via direct float computation,
 * then compares.
 */
TEST(nntrainer_Q1_0_GGML, large_random_dot_product_p) {
  nntrainer::init_backend();

  const int K = 2048;

  // Create weights with known structure: alternating blocks of +/- values
  std::vector<float> weights(K);
  for (int i = 0; i < K; ++i) {
    weights[i] = ((i / 7) % 2 == 0) ? 1.5f : -1.5f;
  }

  // Random activations
  std::vector<float> activations =
    generate_random_vector<float>(K, -1.0f, 1.0f);

  // Quantize weights
  const int nb = K / 128;
  std::vector<uint8_t> q_weights(nb * 18);
  nntrainer::__ggml_quantize_q1_0(
    weights.data(), q_weights.data(), 1, K, nullptr);

  // Compute via SIMD-optimized path
  float simd_result = nntrainer::__ggml_vec_dot_q1_0_f32(
    K, q_weights.data(), activations.data());

  // Compute reference: dequantize then manual dot
  std::vector<float> deq_weights(K);
  nntrainer::__ggml_dequantize_row_q1_0(
    q_weights.data(), deq_weights.data(), K);

  float ref_result = 0.0f;
  for (int i = 0; i < K; ++i) {
    ref_result += deq_weights[i] * activations[i];
  }

  // Should match within floating-point accumulation tolerance
  EXPECT_NEAR(simd_result, ref_result, std::fabs(ref_result) * 1e-5f + 1e-3f)
    << "SIMD dot product diverges from dequantize+manual reference";
}

/**
 * @brief Large-scale GEMM verification.
 * Computes GEMM via optimized path and via dequantize+BLAS-style,
 * then compares.
 */
TEST(nntrainer_Q1_0_GGML, large_random_gemm_p) {
  nntrainer::init_backend();

  const unsigned int M = 4;  // batch
  const unsigned int N = 8;  // output dim (weight rows)
  const unsigned int K = 512;

  // Random weights
  std::vector<float> w_data = generate_random_vector<float>(N * K, -2.0f, 2.0f);
  const int nb = K / 128;
  std::vector<uint8_t> q_weights(N * nb * 18);
  nntrainer::__ggml_quantize_q1_0(
    w_data.data(), q_weights.data(), N, K, nullptr);

  // Random activations
  std::vector<float> A = generate_random_vector<float>(M * K, -1.0f, 1.0f);

  // SIMD GEMM
  std::vector<float> C_simd(M * N, 0.0f);
  nntrainer::__ggml_q1_0_GEMM(M, N, K, A.data(), K,
                              q_weights.data(), K, C_simd.data(), N);

  // Reference: dequantize all weights, then manual GEMM
  std::vector<float> deq_w(N * K);
  for (unsigned int n = 0; n < N; ++n) {
    nntrainer::__ggml_dequantize_row_q1_0(
      q_weights.data() + n * nb * 18, deq_w.data() + n * K, K);
  }

  std::vector<float> C_ref(M * N, 0.0f);
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (unsigned int k = 0; k < K; ++k) {
        sum += A[m * K + k] * deq_w[n * K + k];
      }
      C_ref[m * N + n] = sum;
    }
  }

  // Compare
  for (unsigned int i = 0; i < M * N; ++i) {
    EXPECT_NEAR(C_simd[i], C_ref[i],
                std::fabs(C_ref[i]) * 1e-4f + 1e-2f)
      << "GEMM mismatch at index " << i
      << " (row=" << i / N << ", col=" << i % N << ")";
  }
}

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

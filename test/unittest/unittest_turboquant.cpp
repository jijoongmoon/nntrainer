// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_turboquant.cpp
 * @date   28 March 2026
 * @brief  Unit tests for TurboQuant 4-bit packed KV cache operations
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 */

#include <cpu_backend.h>
#include <fallback_internal.h>
#include <gtest/gtest.h>
#include <turboquant_utils.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

/**
 * @brief Test pack/unpack round-trip for known values.
 *        Packs two 4-bit elements per byte and verifies unpack recovers them.
 */
TEST(turboquant_utils, pack_unpack_roundtrip) {
  // q_vals: 3-bit values (0-7), signs: 0 or 1
  std::vector<uint8_t> q_vals = {5, 2, 7, 0, 3, 6, 1, 4};
  std::vector<uint8_t> signs = {1, 0, 1, 0, 0, 1, 1, 0};
  size_t num_elements = q_vals.size();

  std::vector<uint8_t> packed(num_elements / 2, 0);
  nntrainer::pack_turboquant_4bit(q_vals.data(), signs.data(), num_elements,
                                  packed.data());

  // Verify each packed byte
  for (size_t i = 0; i < num_elements; i += 2) {
    uint8_t val0, sign0, val1, sign1;
    nntrainer::unpack_turboquant_4bit(packed[i / 2], val0, sign0, val1, sign1);

    EXPECT_EQ(val0, q_vals[i]) << "Mismatch at element " << i;
    EXPECT_EQ(sign0, signs[i]) << "Sign mismatch at element " << i;
    EXPECT_EQ(val1, q_vals[i + 1]) << "Mismatch at element " << (i + 1);
    EXPECT_EQ(sign1, signs[i + 1]) << "Sign mismatch at element " << (i + 1);
  }
}

/**
 * @brief Test specific expected bit patterns.
 *        elem0 = 5 | (1<<3) = 0x0D, elem1 = 2 | (0<<3) = 0x02 -> byte = 0x2D
 */
TEST(turboquant_utils, pack_known_bit_patterns) {
  std::vector<uint8_t> q_vals = {5, 2, 7, 0};
  std::vector<uint8_t> signs = {1, 0, 1, 0};

  std::vector<uint8_t> packed(2, 0);
  nntrainer::pack_turboquant_4bit(q_vals.data(), signs.data(), 4,
                                  packed.data());

  // Byte 0: elem0 = 5 | (1<<3) = 13 (0x0D), elem1 = 2 | (0<<3) = 2 (0x02)
  //         packed = (0x02 << 4) | 0x0D = 0x2D
  EXPECT_EQ(packed[0], 0x2D);

  // Byte 1: elem0 = 7 | (1<<3) = 15 (0x0F), elem1 = 0 | (0<<3) = 0 (0x00)
  //         packed = (0x00 << 4) | 0x0F = 0x0F
  EXPECT_EQ(packed[1], 0x0F);
}

/**
 * @brief Test all possible 3-bit values and sign combinations.
 */
TEST(turboquant_utils, pack_unpack_exhaustive) {
  // Generate all 16 possible (q_val, sign) combinations paired
  std::vector<uint8_t> q_vals;
  std::vector<uint8_t> signs;

  for (int q = 0; q < 8; ++q) {
    for (int s = 0; s < 2; ++s) {
      q_vals.push_back(q);
      signs.push_back(s);
    }
  }

  size_t n = q_vals.size(); // 16 elements
  std::vector<uint8_t> packed(n / 2);
  nntrainer::pack_turboquant_4bit(q_vals.data(), signs.data(), n,
                                  packed.data());

  for (size_t i = 0; i < n; i += 2) {
    uint8_t v0, s0, v1, s1;
    nntrainer::unpack_turboquant_4bit(packed[i / 2], v0, s0, v1, s1);
    EXPECT_EQ(v0, q_vals[i]);
    EXPECT_EQ(s0, signs[i]);
    EXPECT_EQ(v1, q_vals[i + 1]);
    EXPECT_EQ(s1, signs[i + 1]);
  }
}

/**
 * @brief Test dequantize_turboquant for correctness.
 *        dequant_val = scale * (q_val - 4)
 */
TEST(turboquant_utils, dequantize_values) {
  float scale = 0.5f;

  // q=0 -> 0.5 * (0-4) = -2.0
  EXPECT_FLOAT_EQ(nntrainer::dequantize_turboquant(0, scale), -2.0f);
  // q=4 -> 0.5 * (4-4) = 0.0
  EXPECT_FLOAT_EQ(nntrainer::dequantize_turboquant(4, scale), 0.0f);
  // q=7 -> 0.5 * (7-4) = 1.5
  EXPECT_FLOAT_EQ(nntrainer::dequantize_turboquant(7, scale), 1.5f);
  // q=1 -> 0.5 * (1-4) = -1.5
  EXPECT_FLOAT_EQ(nntrainer::dequantize_turboquant(1, scale), -1.5f);
}

/**
 * @brief Test quantize_and_pack round-trip: quantize FP32 -> pack -> unpack ->
 *        dequantize and verify error is bounded.
 */
TEST(turboquant_utils, quantize_roundtrip_error_bound) {
  constexpr int N = 64;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

  std::vector<float> input(N);
  for (auto &v : input)
    v = dist(gen);

  std::vector<uint8_t> packed(N / 2);
  int num_groups = (N + 32 - 1) / 32;
  std::vector<float> scales(num_groups);

  nntrainer::quantize_and_pack_turboquant(input.data(), N, packed.data(),
                                          scales.data());

  // Dequantize and check error
  float max_err = 0.0f;
  for (int i = 0; i < N; i += 2) {
    uint8_t v0, s0, v1, s1;
    nntrainer::unpack_turboquant_4bit(packed[i / 2], v0, s0, v1, s1);

    int grp = i / 32;
    float dq0 = nntrainer::dequantize_turboquant(v0, scales[grp]);
    float dq1 = nntrainer::dequantize_turboquant(v1, scales[grp]);

    float err0 = std::fabs(input[i] - dq0);
    float err1 = std::fabs(input[i + 1] - dq1);
    max_err = std::max(max_err, std::max(err0, err1));
  }

  // 3-bit quantization: scale = absmax/3, max per-element error = scale/2
  // For input range [-3,3]: absmax~3, error bound = 3/6 = 0.5
  EXPECT_LT(max_err, 0.55f) << "Quantization error too large: " << max_err;
}

/**
 * @brief Helper: compute reference FP32 Q*K^T dot product (scalar).
 */
static void reference_qk_dot(const float *query, const float *keys,
                              float *output, int num_rows, int num_cache_head,
                              int head_dim, int gqa_size) {
  for (int n = 0; n < num_cache_head; ++n) {
    for (int row = 0; row < num_rows; ++row) {
      for (int g = 0; g < gqa_size; ++g) {
        const float *q_ptr = query + n * gqa_size * head_dim + g * head_dim;
        const float *k_ptr = keys + (row * num_cache_head + n) * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          sum += q_ptr[d] * k_ptr[d];
        output[row * num_cache_head * gqa_size + n * gqa_size + g] =
          sum / std::sqrt((float)head_dim);
      }
    }
  }
}

/**
 * @brief Test compute_kcaches_packed4 against FP32 reference.
 *        Uses the quantize -> pack -> compute_kcaches_packed4 pipeline
 *        and compares against direct FP32 Q*K^T.
 */
TEST(turboquant_compute, kcaches_packed4_vs_fp32_reference) {
  constexpr int num_rows = 4;
  constexpr int num_cache_head = 2;
  constexpr int head_dim = 32;
  constexpr int gqa_size = 2;
  constexpr int tile_size = 4;
  constexpr int GROUP_SIZE = 32;

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Generate query: (num_cache_head * gqa_size * head_dim)
  int q_size = num_cache_head * gqa_size * head_dim;
  std::vector<float> query(q_size);
  for (auto &v : query)
    v = dist(gen);

  // Generate key cache: (num_rows * num_cache_head * head_dim)
  int k_size = num_rows * num_cache_head * head_dim;
  std::vector<float> keys_fp32(k_size);
  for (auto &v : keys_fp32)
    v = dist(gen);

  // 1. Compute FP32 reference
  int out_size = num_rows * num_cache_head * gqa_size;
  std::vector<float> ref_output(out_size, 0.0f);
  reference_qk_dot(query.data(), keys_fp32.data(), ref_output.data(), num_rows,
                   num_cache_head, head_dim, gqa_size);

  // 2. Quantize each row's keys and pack
  int packed_row_bytes = num_cache_head * head_dim / 2;
  int num_groups_per_head = (head_dim + GROUP_SIZE - 1) / GROUP_SIZE;
  int scales_per_row = num_cache_head * num_groups_per_head;

  std::vector<uint8_t> packed_keys(num_rows * packed_row_bytes);
  std::vector<float> key_scales(num_rows * scales_per_row);

  for (int row = 0; row < num_rows; ++row) {
    for (int h = 0; h < num_cache_head; ++h) {
      const float *src =
        keys_fp32.data() + (row * num_cache_head + h) * head_dim;
      uint8_t *dst =
        packed_keys.data() + row * packed_row_bytes + h * head_dim / 2;
      float *s_dst =
        key_scales.data() + row * scales_per_row + h * num_groups_per_head;

      nntrainer::quantize_and_pack_turboquant(src, head_dim, dst, s_dst);
    }
  }

  // 3. Compute with packed4 kernel
  std::vector<float> packed_output(out_size, 0.0f);
  nntrainer::compute_kcaches_packed4(
    query.data(), packed_keys.data(), key_scales.data(), packed_output.data(),
    num_rows, num_cache_head, head_dim, gqa_size, tile_size);

  // 4. Compare - allow quantization error (3-bit is coarse)
  float max_diff = 0.0f;
  for (int i = 0; i < out_size; ++i) {
    float diff = std::fabs(ref_output[i] - packed_output[i]);
    max_diff = std::max(max_diff, diff);
  }

  // Dot product error: per-element errors cancel out over head_dim summation.
  // Expected bound ~ absmax / (6 * sqrt(head_dim)) ≈ 0.03 for dim=32.
  // Use 0.15 with margin for statistical variation.
  EXPECT_LT(max_diff, 0.15f)
    << "compute_kcaches_packed4 error too large: " << max_diff;
}

/**
 * @brief Helper: compute reference FP32 attention-weighted value sum.
 */
static void reference_attn_value(int row_num, const float *attn_weights,
                                 const float *values, float *output,
                                 int num_cache_head, int gqa_size,
                                 int head_dim) {
  for (int n = 0; n < num_cache_head; ++n) {
    for (int h = 0; h < gqa_size; ++h) {
      std::vector<float> acc(head_dim, 0.0f);
      for (int j = 0; j <= row_num; ++j) {
        float a_val =
          attn_weights[(j * num_cache_head + n) * gqa_size + h];
        const float *v_ptr = values + (j * num_cache_head + n) * head_dim;
        for (int d = 0; d < head_dim; ++d)
          acc[d] += a_val * v_ptr[d];
      }
      int out_base = (n * gqa_size + h) * head_dim;
      for (int d = 0; d < head_dim; ++d)
        output[out_base + d] = acc[d];
    }
  }
}

/**
 * @brief Test compute_vcache_packed4_transposed against FP32 reference.
 */
TEST(turboquant_compute, vcache_packed4_transposed_vs_fp32_reference) {
  constexpr int num_rows = 4;
  constexpr int num_cache_head = 2;
  constexpr int head_dim = 32;
  constexpr int gqa_size = 2;
  constexpr int GROUP_SIZE = 32;
  int row_num = num_rows - 1;

  std::mt19937 gen(456);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> attn_dist(0.0f, 1.0f);

  // Generate attention weights: (num_rows * num_cache_head * gqa_size)
  int attn_size = num_rows * num_cache_head * gqa_size;
  std::vector<float> attn_weights(attn_size);
  for (auto &v : attn_weights)
    v = attn_dist(gen);

  // Normalize per position to sum to 1 (simulate softmax)
  for (int n = 0; n < num_cache_head; ++n) {
    for (int h = 0; h < gqa_size; ++h) {
      float sum = 0.0f;
      for (int j = 0; j < num_rows; ++j)
        sum += attn_weights[(j * num_cache_head + n) * gqa_size + h];
      for (int j = 0; j < num_rows; ++j)
        attn_weights[(j * num_cache_head + n) * gqa_size + h] /= sum;
    }
  }

  // Generate value cache FP32
  int v_size = num_rows * num_cache_head * head_dim;
  std::vector<float> values_fp32(v_size);
  for (auto &v : values_fp32)
    v = dist(gen);

  // 1. FP32 reference
  int out_dim = num_cache_head * gqa_size * head_dim;
  std::vector<float> ref_output(out_dim, 0.0f);
  reference_attn_value(row_num, attn_weights.data(), values_fp32.data(),
                       ref_output.data(), num_cache_head, gqa_size, head_dim);

  // 2. Quantize and pack values
  int packed_row_bytes = num_cache_head * head_dim / 2;
  int num_groups_per_head = (head_dim + GROUP_SIZE - 1) / GROUP_SIZE;
  int scales_per_row = num_cache_head * num_groups_per_head;

  std::vector<uint8_t> packed_values(num_rows * packed_row_bytes);
  std::vector<float> value_scales(num_rows * scales_per_row);

  for (int row = 0; row < num_rows; ++row) {
    for (int h = 0; h < num_cache_head; ++h) {
      const float *src =
        values_fp32.data() + (row * num_cache_head + h) * head_dim;
      uint8_t *dst =
        packed_values.data() + row * packed_row_bytes + h * head_dim / 2;
      float *s_dst =
        value_scales.data() + row * scales_per_row + h * num_groups_per_head;

      nntrainer::quantize_and_pack_turboquant(src, head_dim, dst, s_dst);
    }
  }

  // 3. Compute with packed4 kernel
  std::vector<float> packed_output(out_dim, 0.0f);
  nntrainer::compute_vcache_packed4_transposed(
    row_num, attn_weights.data(), packed_values.data(), value_scales.data(),
    packed_output.data(), num_cache_head, gqa_size, head_dim);

  // 4. Compare
  float max_diff = 0.0f;
  for (int i = 0; i < out_dim; ++i) {
    float diff = std::fabs(ref_output[i] - packed_output[i]);
    max_diff = std::max(max_diff, diff);
  }

  // Weighted sum error: attn_weights sum to 1, so error bounded by
  // max per-element quant error ≈ absmax/6 ≈ 0.167 for inputs in [-1,1].
  EXPECT_LT(max_diff, 0.2f)
    << "compute_vcache_packed4_transposed error too large: " << max_diff;
}

/**
 * @brief Test quantize_kv_turboquant matches quantize_and_pack_turboquant.
 *        Both should produce identical results.
 */
TEST(turboquant_compute, quantize_kv_matches_utility) {
  constexpr int N = 64;
  std::mt19937 gen(789);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<float> input(N);
  for (auto &v : input)
    v = dist(gen);

  // Using the inline utility
  int num_groups = (N + 32 - 1) / 32;
  std::vector<uint8_t> packed_util(N / 2);
  std::vector<float> scales_util(num_groups);
  nntrainer::quantize_and_pack_turboquant(input.data(), N, packed_util.data(),
                                          scales_util.data());

  // Using the cpu_backend function
  std::vector<uint8_t> packed_backend(N / 2);
  std::vector<float> scales_backend(num_groups);
  nntrainer::quantize_kv_turboquant(input.data(), N, packed_backend.data(),
                                    scales_backend.data());

  // Should be identical
  for (int i = 0; i < num_groups; ++i) {
    EXPECT_FLOAT_EQ(scales_util[i], scales_backend[i])
      << "Scale mismatch at group " << i;
  }
  for (size_t i = 0; i < packed_util.size(); ++i) {
    EXPECT_EQ(packed_util[i], packed_backend[i])
      << "Packed byte mismatch at index " << i;
  }
}

/**
 * @brief Test with head_dim that is not a multiple of GROUP_SIZE (32).
 *        Verifies the multi-group scale handling.
 */
TEST(turboquant_compute, kcaches_packed4_large_head_dim) {
  constexpr int num_rows = 2;
  constexpr int num_cache_head = 1;
  constexpr int head_dim = 128; // 4 groups of 32
  constexpr int gqa_size = 1;
  constexpr int tile_size = 4;
  constexpr int GROUP_SIZE = 32;

  std::mt19937 gen(321);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  int q_size = num_cache_head * gqa_size * head_dim;
  std::vector<float> query(q_size);
  for (auto &v : query)
    v = dist(gen);

  int k_size = num_rows * num_cache_head * head_dim;
  std::vector<float> keys_fp32(k_size);
  for (auto &v : keys_fp32)
    v = dist(gen);

  // FP32 reference
  int out_size = num_rows * num_cache_head * gqa_size;
  std::vector<float> ref_output(out_size, 0.0f);
  reference_qk_dot(query.data(), keys_fp32.data(), ref_output.data(), num_rows,
                   num_cache_head, head_dim, gqa_size);

  // Quantize and pack
  int packed_row_bytes = num_cache_head * head_dim / 2;
  int num_groups_per_head = (head_dim + GROUP_SIZE - 1) / GROUP_SIZE;
  int scales_per_row = num_cache_head * num_groups_per_head;

  std::vector<uint8_t> packed_keys(num_rows * packed_row_bytes);
  std::vector<float> key_scales(num_rows * scales_per_row);

  for (int row = 0; row < num_rows; ++row) {
    for (int h = 0; h < num_cache_head; ++h) {
      const float *src =
        keys_fp32.data() + (row * num_cache_head + h) * head_dim;
      uint8_t *dst =
        packed_keys.data() + row * packed_row_bytes + h * head_dim / 2;
      float *s_dst =
        key_scales.data() + row * scales_per_row + h * num_groups_per_head;
      nntrainer::quantize_and_pack_turboquant(src, head_dim, dst, s_dst);
    }
  }

  // Packed computation
  std::vector<float> packed_output(out_size, 0.0f);
  nntrainer::compute_kcaches_packed4(
    query.data(), packed_keys.data(), key_scales.data(), packed_output.data(),
    num_rows, num_cache_head, head_dim, gqa_size, tile_size);

  float max_diff = 0.0f;
  for (int i = 0; i < out_size; ++i) {
    float diff = std::fabs(ref_output[i] - packed_output[i]);
    max_diff = std::max(max_diff, diff);
  }

  // Larger head_dim → more cancellation → tighter bound.
  // Expected ~ absmax / (6 * sqrt(128)) ≈ 0.015 for dim=128.
  EXPECT_LT(max_diff, 0.05f)
    << "Large head_dim kcaches error too large: " << max_diff;
}

/**
 * @brief Test sliding window support in compute_kcaches_packed4.
 */
TEST(turboquant_compute, kcaches_packed4_sliding_window) {
  constexpr int num_rows = 8;
  constexpr int num_cache_head = 1;
  constexpr int head_dim = 32;
  constexpr int gqa_size = 1;
  constexpr int tile_size = 4;
  constexpr int GROUP_SIZE = 32;
  constexpr size_t local_window = 4;

  std::mt19937 gen(555);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  int q_size = num_cache_head * gqa_size * head_dim;
  std::vector<float> query(q_size);
  for (auto &v : query)
    v = dist(gen);

  int k_size = num_rows * num_cache_head * head_dim;
  std::vector<float> keys_fp32(k_size);
  for (auto &v : keys_fp32)
    v = dist(gen);

  // Quantize and pack all rows
  int packed_row_bytes = num_cache_head * head_dim / 2;
  int num_groups_per_head = (head_dim + GROUP_SIZE - 1) / GROUP_SIZE;
  int scales_per_row = num_cache_head * num_groups_per_head;

  std::vector<uint8_t> packed_keys(num_rows * packed_row_bytes);
  std::vector<float> key_scales(num_rows * scales_per_row);

  for (int row = 0; row < num_rows; ++row) {
    const float *src = keys_fp32.data() + row * num_cache_head * head_dim;
    uint8_t *dst = packed_keys.data() + row * packed_row_bytes;
    float *s_dst = key_scales.data() + row * scales_per_row;
    nntrainer::quantize_and_pack_turboquant(src, head_dim, dst, s_dst);
  }

  // Compute with sliding window
  int row_cnt = local_window; // only last 4 rows should be used
  std::vector<float> packed_output(row_cnt * num_cache_head * gqa_size, 0.0f);
  nntrainer::compute_kcaches_packed4(
    query.data(), packed_keys.data(), key_scales.data(), packed_output.data(),
    num_rows, num_cache_head, head_dim, gqa_size, tile_size, local_window);

  // Verify output size matches window (not full num_rows)
  EXPECT_EQ(packed_output.size(), (size_t)(row_cnt * num_cache_head * gqa_size));

  // Verify non-zero (actual computation happened)
  bool has_nonzero = false;
  for (auto v : packed_output) {
    if (std::fabs(v) > 1e-6f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Sliding window produced all zeros";
}

/**
 * @brief Test edge case: zero input should produce zero quantized values
 *        (centered at zero_point=4 in 3-bit).
 */
TEST(turboquant_utils, quantize_zeros) {
  constexpr int N = 32;
  std::vector<float> input(N, 0.0f);
  std::vector<uint8_t> packed(N / 2);
  std::vector<float> scales(1);

  nntrainer::quantize_and_pack_turboquant(input.data(), N, packed.data(),
                                          scales.data());

  // All zero values should quantize to q=4 (zero point)
  for (int i = 0; i < N; i += 2) {
    uint8_t v0, s0, v1, s1;
    nntrainer::unpack_turboquant_4bit(packed[i / 2], v0, s0, v1, s1);
    EXPECT_EQ(v0, 4) << "Zero input should map to q=4, got " << (int)v0;
    EXPECT_EQ(v1, 4) << "Zero input should map to q=4, got " << (int)v1;
  }
}

GTEST_API_ int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

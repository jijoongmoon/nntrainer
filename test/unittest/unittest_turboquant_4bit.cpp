// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_turboquant_4bit.cpp
 * @date   28 March 2026
 * @brief  Unit tests for TurboQuant 4-bit packing and quantization utilities.
 */

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

#include <turboquant_4bit.h>

/**
 * @brief Test pack/unpack round-trip with known values
 */
TEST(turboquant_4bit, pack_unpack_roundtrip_basic) {
  // 4 elements: q_vals in [0,7], signs in {0,1}
  int8_t q_vals[] = {5, 2, 7, 0};
  uint8_t signs[] = {1, 0, 1, 0};
  const size_t n = 4;

  // Pack
  std::vector<uint8_t> packed(n / 2);
  nntrainer::pack_turboquant_4bit(q_vals, signs, n, packed.data());

  // Expected:
  // elem0 = 5 | (1<<3) = 0x0D, elem1 = 2 | (0<<3) = 0x02 -> byte0 = 0x2D
  // elem0 = 7 | (1<<3) = 0x0F, elem1 = 0 | (0<<3) = 0x00 -> byte1 = 0x0F
  EXPECT_EQ(packed[0], 0x2D);
  EXPECT_EQ(packed[1], 0x0F);

  // Unpack
  std::vector<int8_t> q_out(n);
  std::vector<uint8_t> s_out(n);
  nntrainer::unpack_turboquant_4bit(packed.data(), q_out.data(), s_out.data(),
                                    n);

  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(q_out[i], q_vals[i]) << "q_vals mismatch at index " << i;
    EXPECT_EQ(s_out[i], signs[i]) << "signs mismatch at index " << i;
  }
}

/**
 * @brief Test pack/unpack round-trip for all possible 4-bit values
 */
TEST(turboquant_4bit, pack_unpack_all_values) {
  // Test all combinations of 3-bit (0~7) x sign (0,1)
  std::vector<int8_t> q_vals;
  std::vector<uint8_t> signs;

  for (int q = 0; q <= 7; ++q) {
    for (int s = 0; s <= 1; ++s) {
      q_vals.push_back(static_cast<int8_t>(q));
      signs.push_back(static_cast<uint8_t>(s));
    }
  }
  // 16 elements (even)
  size_t n = q_vals.size();
  ASSERT_EQ(n, 16u);

  std::vector<uint8_t> packed(n / 2);
  nntrainer::pack_turboquant_4bit(q_vals.data(), signs.data(), n,
                                  packed.data());

  std::vector<int8_t> q_out(n);
  std::vector<uint8_t> s_out(n);
  nntrainer::unpack_turboquant_4bit(packed.data(), q_out.data(), s_out.data(),
                                    n);

  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(q_out[i], q_vals[i]) << "q_vals mismatch at index " << i;
    EXPECT_EQ(s_out[i], signs[i]) << "signs mismatch at index " << i;
  }
}

/**
 * @brief Test pack/unpack round-trip with random data (large size)
 */
TEST(turboquant_4bit, pack_unpack_random_large) {
  const size_t n = 1024; // Typical head_dim * num_heads
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> q_dist(0, 7);
  std::uniform_int_distribution<int> s_dist(0, 1);

  std::vector<int8_t> q_vals(n);
  std::vector<uint8_t> signs(n);
  for (size_t i = 0; i < n; ++i) {
    q_vals[i] = static_cast<int8_t>(q_dist(rng));
    signs[i] = static_cast<uint8_t>(s_dist(rng));
  }

  std::vector<uint8_t> packed(n / 2);
  nntrainer::pack_turboquant_4bit(q_vals.data(), signs.data(), n,
                                  packed.data());

  std::vector<int8_t> q_out(n);
  std::vector<uint8_t> s_out(n);
  nntrainer::unpack_turboquant_4bit(packed.data(), q_out.data(), s_out.data(),
                                    n);

  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(q_out[i], q_vals[i]);
    EXPECT_EQ(s_out[i], signs[i]);
  }
}

/**
 * @brief Test quantize_fp32_to_3bit produces valid 3-bit values and signs
 */
TEST(turboquant_4bit, quantize_fp32_to_3bit_range) {
  std::vector<float> input = {-1.5f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f,
                               2.0f,  3.0f,  4.0f, 5.0f, 6.0f, 7.5f};
  size_t n = input.size();

  std::vector<int8_t> q_vals(n);
  std::vector<uint8_t> signs(n);
  float scale, zp, qjl_scale;

  nntrainer::quantize_fp32_to_3bit(input.data(), q_vals.data(), signs.data(),
                                   &scale, &zp, &qjl_scale, n);

  // All q_vals must be in [0, 7]
  for (size_t i = 0; i < n; ++i) {
    EXPECT_GE(q_vals[i], 0) << "q_val < 0 at index " << i;
    EXPECT_LE(q_vals[i], 7) << "q_val > 7 at index " << i;
  }

  // All signs must be 0 or 1
  for (size_t i = 0; i < n; ++i) {
    EXPECT_TRUE(signs[i] == 0 || signs[i] == 1)
      << "invalid sign at index " << i;
  }

  // Scale must be positive
  EXPECT_GT(scale, 0.0f);

  // QJL scale must be non-negative
  EXPECT_GE(qjl_scale, 0.0f);
}

/**
 * @brief Test full quantize -> pack -> unpack -> dequantize pipeline
 *        and verify reconstruction error is bounded.
 */
TEST(turboquant_4bit, full_pipeline_error_bound) {
  const size_t n = 128; // Typical head_dim
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

  std::vector<float> input(n);
  for (auto &v : input)
    v = dist(rng);

  // Quantize
  std::vector<int8_t> q_vals(n);
  std::vector<uint8_t> signs(n);
  float scale, zp, qjl_scale;
  nntrainer::quantize_fp32_to_3bit(input.data(), q_vals.data(), signs.data(),
                                   &scale, &zp, &qjl_scale, n);

  // Pack
  std::vector<uint8_t> packed(n / 2);
  nntrainer::pack_turboquant_4bit(q_vals.data(), signs.data(), n,
                                  packed.data());

  // Unpack + Dequantize
  std::vector<float> output(n);
  nntrainer::unpack_and_dequantize_4bit(packed.data(), output.data(), scale, zp,
                                        qjl_scale, n);

  // Verify error bound: with QJL correction, error should be < scale
  // (3-bit quantization error is at most scale/2, QJL corrects by qjl_scale)
  float max_error = 0.0f;
  float mse = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float err = std::fabs(input[i] - output[i]);
    max_error = std::max(max_error, err);
    mse += err * err;
  }
  mse /= n;

  // Max error should be less than the quantization step size
  EXPECT_LT(max_error, scale + qjl_scale + 1e-6f);

  // MSE should be reasonable for 3-bit quantization with correction
  float rmse = std::sqrt(mse);
  EXPECT_LT(rmse, scale); // RMSE should be well within one quantization step
}

/**
 * @brief Test quantize_and_pack_4bit convenience function
 */
TEST(turboquant_4bit, quantize_and_pack_convenience) {
  const size_t head_dim = 128;
  std::mt19937 rng(456);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

  std::vector<float> input(head_dim);
  for (auto &v : input)
    v = dist(rng);

  // Temp buffers
  std::vector<int8_t> q_buf(head_dim);
  std::vector<uint8_t> s_buf(head_dim);

  // Pack
  std::vector<uint8_t> packed(head_dim / 2);
  float scale, zp, qjl_scale;
  nntrainer::quantize_and_pack_4bit(input.data(), packed.data(), &scale, &zp,
                                    &qjl_scale, head_dim, q_buf.data(),
                                    s_buf.data());

  // Unpack + dequantize
  std::vector<float> output(head_dim);
  nntrainer::unpack_and_dequantize_4bit(packed.data(), output.data(), scale, zp,
                                        qjl_scale, head_dim);

  // Verify reconstruction
  for (size_t i = 0; i < head_dim; ++i) {
    float err = std::fabs(input[i] - output[i]);
    EXPECT_LT(err, scale + qjl_scale + 1e-6f);
  }
}

/**
 * @brief Test dequantize_4bit_qjl correction direction
 *        sign=1 should add positive correction, sign=0 negative
 */
TEST(turboquant_4bit, qjl_correction_direction) {
  float scale = 1.0f;
  float zp = 0.0f;
  float qjl_scale = 0.25f;
  int8_t q_val = 3;

  float base = 3.0f * 1.0f + 0.0f; // = 3.0f

  float with_pos =
    nntrainer::dequantize_4bit_qjl(q_val, 1, scale, zp, qjl_scale);
  float with_neg =
    nntrainer::dequantize_4bit_qjl(q_val, 0, scale, zp, qjl_scale);

  EXPECT_FLOAT_EQ(with_pos, base + qjl_scale); // 3.25
  EXPECT_FLOAT_EQ(with_neg, base - qjl_scale); // 2.75
}

/**
 * @brief Test constant input (all same values) - edge case
 */
TEST(turboquant_4bit, constant_input) {
  const size_t n = 16;
  std::vector<float> input(n, 3.14f);

  std::vector<int8_t> q_vals(n);
  std::vector<uint8_t> signs(n);
  float scale, zp, qjl_scale;

  nntrainer::quantize_fp32_to_3bit(input.data(), q_vals.data(), signs.data(),
                                   &scale, &zp, &qjl_scale, n);

  // All values are the same, so scale should be minimal
  // and all q_vals should be the same
  int8_t first_q = q_vals[0];
  for (size_t i = 1; i < n; ++i) {
    EXPECT_EQ(q_vals[i], first_q);
  }
}

/**
 * @brief Test dot product equivalence: FP32 dot product vs
 *        4-bit quantized dot product. This simulates the
 *        compute_kcaches scenario.
 */
TEST(turboquant_4bit, dot_product_approximation) {
  const size_t head_dim = 128;
  std::mt19937 rng(789);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  // Simulate query and key vectors
  std::vector<float> query(head_dim);
  std::vector<float> key(head_dim);
  for (size_t i = 0; i < head_dim; ++i) {
    query[i] = dist(rng);
    key[i] = dist(rng);
  }

  // Reference: FP32 dot product
  float ref_dot = 0.0f;
  for (size_t i = 0; i < head_dim; ++i) {
    ref_dot += query[i] * key[i];
  }

  // Quantize key, then compute dot product with dequantized key
  std::vector<int8_t> q_buf(head_dim);
  std::vector<uint8_t> s_buf(head_dim);
  std::vector<uint8_t> packed(head_dim / 2);
  float scale, zp, qjl_scale;

  nntrainer::quantize_and_pack_4bit(key.data(), packed.data(), &scale, &zp,
                                    &qjl_scale, head_dim, q_buf.data(),
                                    s_buf.data());

  std::vector<float> key_deq(head_dim);
  nntrainer::unpack_and_dequantize_4bit(packed.data(), key_deq.data(), scale,
                                        zp, qjl_scale, head_dim);

  float quant_dot = 0.0f;
  for (size_t i = 0; i < head_dim; ++i) {
    quant_dot += query[i] * key_deq[i];
  }

  // The relative error should be bounded
  // For 3-bit + QJL, typical relative error < 20% for random vectors
  float abs_error = std::fabs(ref_dot - quant_dot);
  float rel_error =
    (std::fabs(ref_dot) > 1e-6f) ? abs_error / std::fabs(ref_dot) : abs_error;

  // Generous bound for 3-bit quantization
  EXPECT_LT(rel_error, 0.3f)
    << "ref_dot=" << ref_dot << " quant_dot=" << quant_dot;
}

/**
 * @brief Test compute_kcaches_4bit: compare against FP32 reference dot product
 */
TEST(turboquant_4bit, compute_kcaches_4bit_basic) {
  const int num_rows = 3;       // 3 cached tokens
  const int num_cache_head = 2; // 2 KV heads
  const int head_dim = 8;       // small for manual verification
  const int gqa_size = 2;       // 4 Q heads total
  const int tile_size = 4;

  // Create query: [num_heads_Q * head_dim] = [4 * 8]
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<float> query(num_cache_head * gqa_size * head_dim);
  for (auto &v : query)
    v = dist(rng);

  // Create key cache in FP32, then quantize
  std::vector<float> key_fp32(num_rows * num_cache_head * head_dim);
  for (auto &v : key_fp32)
    v = dist(rng);

  // Quantize each head separately
  std::vector<uint8_t> packed_cache(num_rows * num_cache_head * head_dim / 2);
  std::vector<float> scales(num_rows * num_cache_head);
  std::vector<float> zero_points(num_rows * num_cache_head);
  std::vector<float> qjl_scales(num_rows * num_cache_head);
  std::vector<int8_t> q_buf(head_dim);
  std::vector<uint8_t> s_buf(head_dim);

  for (int row = 0; row < num_rows; ++row) {
    for (int n = 0; n < num_cache_head; ++n) {
      int offset = (row * num_cache_head + n) * head_dim;
      int packed_offset = (row * num_cache_head + n) * (head_dim / 2);
      int meta_idx = row * num_cache_head + n;

      nntrainer::quantize_and_pack_4bit(
        &key_fp32[offset], &packed_cache[packed_offset], &scales[meta_idx],
        &zero_points[meta_idx], &qjl_scales[meta_idx], head_dim, q_buf.data(),
        s_buf.data());
    }
  }

  // Run compute_kcaches_4bit
  int row_cnt = num_rows;
  std::vector<float> output(row_cnt * num_cache_head * gqa_size, 0.0f);

  nntrainer::compute_kcaches_4bit(
    query.data(), packed_cache.data(), output.data(), num_rows, num_cache_head,
    head_dim, gqa_size, tile_size, scales.data(), zero_points.data(),
    qjl_scales.data());

  // Compute reference: dequantize and compute dot products
  float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> tmp(head_dim);

  for (int n = 0; n < num_cache_head; ++n) {
    for (int row = 0; row < num_rows; ++row) {
      int packed_offset = (row * num_cache_head + n) * (head_dim / 2);
      int meta_idx = row * num_cache_head + n;

      nntrainer::unpack_and_dequantize_4bit(
        &packed_cache[packed_offset], tmp.data(), scales[meta_idx],
        zero_points[meta_idx], qjl_scales[meta_idx], head_dim);

      for (int g = 0; g < gqa_size; ++g) {
        const float *q_ptr = query.data() + n * gqa_size * head_dim + g * head_dim;
        float ref_sum = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
          ref_sum += q_ptr[i] * tmp[i];
        }
        ref_sum *= inv_sqrt_hd;

        float actual = output[row * num_cache_head * gqa_size + n * gqa_size + g];
        EXPECT_NEAR(ref_sum, actual, 1e-5f)
          << "Mismatch at row=" << row << " head=" << n << " g=" << g;
      }
    }
  }
}

/**
 * @brief Test compute_vcaches_4bit: verify weighted sum
 */
TEST(turboquant_4bit, compute_vcaches_4bit_basic) {
  const int num_rows = 3;
  const int num_cache_head = 2;
  const int head_dim = 8;
  const int gqa_size = 2;

  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Create value cache FP32, quantize
  std::vector<float> value_fp32(num_rows * num_cache_head * head_dim);
  for (auto &v : value_fp32)
    v = dist(rng);

  std::vector<uint8_t> packed_cache(num_rows * num_cache_head * head_dim / 2);
  std::vector<float> scales(num_rows * num_cache_head);
  std::vector<float> zero_points(num_rows * num_cache_head);
  std::vector<float> qjl_scales(num_rows * num_cache_head);
  std::vector<int8_t> q_buf(head_dim);
  std::vector<uint8_t> s_buf(head_dim);

  for (int row = 0; row < num_rows; ++row) {
    for (int n = 0; n < num_cache_head; ++n) {
      int offset = (row * num_cache_head + n) * head_dim;
      int packed_offset = (row * num_cache_head + n) * (head_dim / 2);
      int meta_idx = row * num_cache_head + n;

      nntrainer::quantize_and_pack_4bit(
        &value_fp32[offset], &packed_cache[packed_offset], &scales[meta_idx],
        &zero_points[meta_idx], &qjl_scales[meta_idx], head_dim, q_buf.data(),
        s_buf.data());
    }
  }

  // Attention weights: simple uniform for test
  // Shape: [row_cnt * num_cache_head * gqa_size]
  int row_num = num_rows - 1; // last token
  std::vector<float> attn_weights(num_rows * num_cache_head * gqa_size);
  // Make valid softmax-like weights (sum to ~1 per head)
  for (int n = 0; n < num_cache_head; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      float sum = 0.0f;
      for (int j = 0; j < num_rows; ++j) {
        float w = 1.0f / num_rows;
        attn_weights[j * gqa_size * num_cache_head + n * gqa_size + g] = w;
        sum += w;
      }
    }
  }

  // Run compute_vcaches_4bit
  std::vector<float> output(num_cache_head * gqa_size * head_dim, 0.0f);
  nntrainer::compute_vcaches_4bit(
    row_num, attn_weights.data(), packed_cache.data(), output.data(),
    num_cache_head, gqa_size, head_dim, scales.data(), zero_points.data(),
    qjl_scales.data());

  // Reference: dequantize and compute weighted sum
  std::vector<float> ref_output(num_cache_head * gqa_size * head_dim, 0.0f);
  std::vector<float> tmp(head_dim);

  for (int n = 0; n < num_cache_head; ++n) {
    for (int j = 0; j <= row_num; ++j) {
      int packed_offset = (j * num_cache_head + n) * (head_dim / 2);
      int meta_idx = j * num_cache_head + n;

      nntrainer::unpack_and_dequantize_4bit(
        &packed_cache[packed_offset], tmp.data(), scales[meta_idx],
        zero_points[meta_idx], qjl_scales[meta_idx], head_dim);

      for (int h = 0; h < gqa_size; ++h) {
        float a_val =
          attn_weights[j * gqa_size * num_cache_head + n * gqa_size + h];
        for (int d = 0; d < head_dim; ++d) {
          ref_output[(n * gqa_size + h) * head_dim + d] += a_val * tmp[d];
        }
      }
    }
  }

  for (size_t i = 0; i < ref_output.size(); ++i) {
    EXPECT_NEAR(ref_output[i], output[i], 1e-5f) << "Mismatch at index " << i;
  }
}

/**
 * @brief Test sliding window support in compute_kcaches_4bit
 */
TEST(turboquant_4bit, compute_kcaches_4bit_sliding_window) {
  const int num_rows = 10;
  const int num_cache_head = 1;
  const int head_dim = 4;
  const int gqa_size = 1;
  const int tile_size = 4;
  const size_t window_size = 3; // Only attend to last 3 tokens

  std::mt19937 rng(777);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> query(head_dim);
  for (auto &v : query)
    v = dist(rng);

  std::vector<float> key_fp32(num_rows * num_cache_head * head_dim);
  for (auto &v : key_fp32)
    v = dist(rng);

  std::vector<uint8_t> packed(num_rows * num_cache_head * head_dim / 2);
  std::vector<float> scales(num_rows * num_cache_head);
  std::vector<float> zero_points(num_rows * num_cache_head);
  std::vector<float> qjl_scales(num_rows * num_cache_head);
  std::vector<int8_t> q_buf(head_dim);
  std::vector<uint8_t> s_buf(head_dim);

  for (int row = 0; row < num_rows; ++row) {
    int offset = row * head_dim;
    int packed_offset = row * (head_dim / 2);
    nntrainer::quantize_and_pack_4bit(
      &key_fp32[offset], &packed[packed_offset], &scales[row],
      &zero_points[row], &qjl_scales[row], head_dim, q_buf.data(),
      s_buf.data());
  }

  // With sliding window: should only produce window_size outputs
  int row_cnt = static_cast<int>(window_size);
  std::vector<float> output(row_cnt * num_cache_head * gqa_size, 0.0f);

  nntrainer::compute_kcaches_4bit(query.data(), packed.data(), output.data(),
                                  num_rows, num_cache_head, head_dim, gqa_size,
                                  tile_size, scales.data(), zero_points.data(),
                                  qjl_scales.data(), window_size);

  // Verify only last 3 rows were processed
  int start_row = num_rows - static_cast<int>(window_size);
  float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> tmp(head_dim);

  for (int r = 0; r < row_cnt; ++r) {
    int row = start_row + r;
    int packed_offset = row * (head_dim / 2);
    nntrainer::unpack_and_dequantize_4bit(&packed[packed_offset], tmp.data(),
                                          scales[row], zero_points[row],
                                          qjl_scales[row], head_dim);
    float ref_sum = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      ref_sum += query[i] * tmp[i];
    }
    ref_sum *= inv_sqrt_hd;
    EXPECT_NEAR(ref_sum, output[r], 1e-5f) << "Mismatch at window row " << r;
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

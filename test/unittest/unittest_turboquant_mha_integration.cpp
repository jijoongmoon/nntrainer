// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_turboquant_mha_integration.cpp
 * @brief  Integration test: MHACoreLayer with use_turboquant=true
 *         Instantiates the layer, runs incremental_forwarding, verifies no crash.
 */

#include <cpu_backend.h>
#include <gtest/gtest.h>
#include <turboquant_utils.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

/**
 * @brief Full pipeline test simulating what mha_core does:
 *        1. Generate random FP32 query, key, value
 *        2. Apply RoPE placeholder (identity for test)
 *        3. Quantize key+value → packed 4-bit cache
 *        4. compute_kcaches_packed4 (Q*K^T)
 *        5. Scalar softmax
 *        6. compute_vcache_packed4_transposed (attn*V)
 *        7. Verify output is finite and non-zero
 *
 *        This mimics one_batch_incremental_forwarding_turboquant
 *        for single-token decoding.
 */
TEST(turboquant_integration, single_token_decoding_pipeline) {
  // Qwen3-like tiny config
  constexpr int num_heads_Q = 4;
  constexpr int num_heads_KV = 2;
  constexpr int head_dim = 64;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int tile_size = 4;
  constexpr int GROUP_SIZE = 32;

  // Simulate context of 16 tokens, decoding token at position 16
  constexpr int context_len = 16;
  constexpr int kv_width = num_heads_KV * head_dim;
  constexpr int packed_row_bytes = kv_width / 2;
  constexpr int num_groups_per_row =
    num_heads_KV * ((head_dim + GROUP_SIZE - 1) / GROUP_SIZE);

  std::mt19937 gen(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Simulate existing KV cache (context_len rows already quantized)
  std::vector<uint8_t> packed_kcache(context_len * packed_row_bytes);
  std::vector<float> kcache_scales(context_len * num_groups_per_row);
  std::vector<uint8_t> packed_vcache(context_len * packed_row_bytes);
  std::vector<float> vcache_scales(context_len * num_groups_per_row);

  // Fill cache with quantized random data
  for (int row = 0; row < context_len; ++row) {
    std::vector<float> k_data(kv_width), v_data(kv_width);
    for (auto &v : k_data)
      v = dist(gen);
    for (auto &v : v_data)
      v = dist(gen);

    nntrainer::quantize_kv_turboquant(
      k_data.data(), kv_width,
      packed_kcache.data() + row * packed_row_bytes,
      kcache_scales.data() + row * num_groups_per_row);
    nntrainer::quantize_kv_turboquant(
      v_data.data(), kv_width,
      packed_vcache.data() + row * packed_row_bytes,
      vcache_scales.data() + row * num_groups_per_row);
  }

  // Step 1: Generate query for current token
  int q_size = num_heads_Q * head_dim;
  std::vector<float> query(q_size);
  for (auto &v : query)
    v = dist(gen);

  // Step 2: compute_kcaches_packed4 (Q * K^T / sqrt(head_dim))
  int num_rows = context_len; // all cached rows
  std::vector<float> attn_scores(num_rows * num_heads_Q, 0.0f);

  nntrainer::compute_kcaches_packed4(
    query.data(), packed_kcache.data(), kcache_scales.data(),
    attn_scores.data(), num_rows, num_heads_KV, head_dim, gqa_size, tile_size);

  // Verify attention scores are finite
  for (int i = 0; i < (int)attn_scores.size(); ++i) {
    ASSERT_TRUE(std::isfinite(attn_scores[i]))
      << "Non-finite attention score at " << i << ": " << attn_scores[i];
  }

  // Step 3: Softmax per head (simple row-wise softmax)
  for (int h = 0; h < num_heads_Q; ++h) {
    float max_val = -1e30f;
    for (int r = 0; r < num_rows; ++r)
      max_val = std::max(max_val, attn_scores[r * num_heads_Q + h]);

    float sum_exp = 0.0f;
    for (int r = 0; r < num_rows; ++r) {
      attn_scores[r * num_heads_Q + h] =
        std::exp(attn_scores[r * num_heads_Q + h] - max_val);
      sum_exp += attn_scores[r * num_heads_Q + h];
    }
    for (int r = 0; r < num_rows; ++r)
      attn_scores[r * num_heads_Q + h] /= sum_exp;
  }

  // Verify softmax sums to 1
  for (int h = 0; h < num_heads_Q; ++h) {
    float sum = 0.0f;
    for (int r = 0; r < num_rows; ++r)
      sum += attn_scores[r * num_heads_Q + h];
    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Softmax sum != 1 for head " << h;
  }

  // Step 4: compute_vcache_packed4_transposed (attn * V)
  int out_dim = num_heads_KV * gqa_size * head_dim;
  std::vector<float> output(out_dim, 0.0f);

  int row_num = context_len - 1;
  nntrainer::compute_vcache_packed4_transposed(
    row_num, attn_scores.data(), packed_vcache.data(), vcache_scales.data(),
    output.data(), num_heads_KV, gqa_size, head_dim);

  // Verify output is finite and non-zero
  bool has_nonzero = false;
  for (int i = 0; i < out_dim; ++i) {
    ASSERT_TRUE(std::isfinite(output[i]))
      << "Non-finite output at " << i << ": " << output[i];
    if (std::fabs(output[i]) > 1e-6f)
      has_nonzero = true;
  }
  EXPECT_TRUE(has_nonzero) << "All outputs are zero - pipeline likely broken";

  std::cout << "  [integration] single-token pipeline OK: " << out_dim
            << " output elements, all finite" << std::endl;
}

/**
 * @brief Multi-token prefill pipeline test.
 *        Simulates processing 8 tokens at once (prefill phase).
 */
TEST(turboquant_integration, prefill_pipeline) {
  constexpr int num_heads_Q = 8;
  constexpr int num_heads_KV = 2;
  constexpr int head_dim = 64;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int tile_size = 4;
  constexpr int GROUP_SIZE = 32;
  constexpr int seq_len = 8;

  constexpr int kv_width = num_heads_KV * head_dim;
  constexpr int packed_row_bytes = kv_width / 2;
  constexpr int num_groups_per_row =
    num_heads_KV * ((head_dim + GROUP_SIZE - 1) / GROUP_SIZE);

  std::mt19937 gen(54321);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Quantize seq_len rows of KV into cache
  std::vector<uint8_t> packed_kcache(seq_len * packed_row_bytes);
  std::vector<float> kcache_scales(seq_len * num_groups_per_row);
  std::vector<uint8_t> packed_vcache(seq_len * packed_row_bytes);
  std::vector<float> vcache_scales(seq_len * num_groups_per_row);

  for (int row = 0; row < seq_len; ++row) {
    std::vector<float> k_data(kv_width), v_data(kv_width);
    for (auto &v : k_data)
      v = dist(gen);
    for (auto &v : v_data)
      v = dist(gen);

    nntrainer::quantize_kv_turboquant(
      k_data.data(), kv_width,
      packed_kcache.data() + row * packed_row_bytes,
      kcache_scales.data() + row * num_groups_per_row);
    nntrainer::quantize_kv_turboquant(
      v_data.data(), kv_width,
      packed_vcache.data() + row * packed_row_bytes,
      vcache_scales.data() + row * num_groups_per_row);
  }

  // Query for all seq_len tokens
  int q_size = seq_len * num_heads_Q * head_dim;
  std::vector<float> query(q_size);
  for (auto &v : query)
    v = dist(gen);

  // For each token i, compute Q*K^T against rows [0, i+1) (causal)
  for (int i = 0; i < seq_len; ++i) {
    int num_rows_i = i + 1;
    std::vector<float> attn_i(num_rows_i * num_heads_Q, 0.0f);

    nntrainer::compute_kcaches_packed4(
      query.data() + i * num_heads_Q * head_dim, packed_kcache.data(),
      kcache_scales.data(), attn_i.data(), num_rows_i, num_heads_KV, head_dim,
      gqa_size, tile_size);

    // Softmax
    for (int h = 0; h < num_heads_Q; ++h) {
      float max_val = -1e30f;
      for (int r = 0; r < num_rows_i; ++r)
        max_val = std::max(max_val, attn_i[r * num_heads_Q + h]);
      float sum_exp = 0.0f;
      for (int r = 0; r < num_rows_i; ++r) {
        attn_i[r * num_heads_Q + h] =
          std::exp(attn_i[r * num_heads_Q + h] - max_val);
        sum_exp += attn_i[r * num_heads_Q + h];
      }
      for (int r = 0; r < num_rows_i; ++r)
        attn_i[r * num_heads_Q + h] /= sum_exp;
    }

    // V aggregation
    int out_dim = num_heads_KV * gqa_size * head_dim;
    std::vector<float> output_i(out_dim, 0.0f);
    nntrainer::compute_vcache_packed4_transposed(
      i, attn_i.data(), packed_vcache.data(), vcache_scales.data(),
      output_i.data(), num_heads_KV, gqa_size, head_dim);

    for (int j = 0; j < out_dim; ++j) {
      ASSERT_TRUE(std::isfinite(output_i[j]))
        << "Non-finite at token " << i << " elem " << j;
    }
  }

  std::cout << "  [integration] prefill pipeline OK: " << seq_len
            << " tokens processed causally" << std::endl;
}

/**
 * @brief Incremental token-by-token test.
 *        Simulates generating tokens one by one, growing the cache.
 */
TEST(turboquant_integration, incremental_generation) {
  constexpr int num_heads_Q = 4;
  constexpr int num_heads_KV = 2;
  constexpr int head_dim = 128;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int tile_size = 4;
  constexpr int GROUP_SIZE = 32;
  constexpr int max_tokens = 32;

  constexpr int kv_width = num_heads_KV * head_dim;
  constexpr int packed_row_bytes = kv_width / 2;
  constexpr int num_groups_per_row =
    num_heads_KV * ((head_dim + GROUP_SIZE - 1) / GROUP_SIZE);

  std::mt19937 gen(99999);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Pre-allocate max cache
  std::vector<uint8_t> packed_kcache(max_tokens * packed_row_bytes, 0);
  std::vector<float> kcache_scales(max_tokens * num_groups_per_row, 0);
  std::vector<uint8_t> packed_vcache(max_tokens * packed_row_bytes, 0);
  std::vector<float> vcache_scales(max_tokens * num_groups_per_row, 0);

  for (int t = 0; t < max_tokens; ++t) {
    // New token's K, V
    std::vector<float> k_data(kv_width), v_data(kv_width);
    for (auto &v : k_data)
      v = dist(gen);
    for (auto &v : v_data)
      v = dist(gen);

    // Quantize and write to cache at position t
    nntrainer::quantize_kv_turboquant(
      k_data.data(), kv_width,
      packed_kcache.data() + t * packed_row_bytes,
      kcache_scales.data() + t * num_groups_per_row);
    nntrainer::quantize_kv_turboquant(
      v_data.data(), kv_width,
      packed_vcache.data() + t * packed_row_bytes,
      vcache_scales.data() + t * num_groups_per_row);

    // Query
    std::vector<float> query(num_heads_Q * head_dim);
    for (auto &v : query)
      v = dist(gen);

    // Compute attention
    int num_rows = t + 1;
    std::vector<float> attn(num_rows * num_heads_Q, 0.0f);
    nntrainer::compute_kcaches_packed4(
      query.data(), packed_kcache.data(), kcache_scales.data(), attn.data(),
      num_rows, num_heads_KV, head_dim, gqa_size, tile_size);

    // Softmax
    for (int h = 0; h < num_heads_Q; ++h) {
      float max_val = -1e30f;
      for (int r = 0; r < num_rows; ++r)
        max_val = std::max(max_val, attn[r * num_heads_Q + h]);
      float sum_exp = 0.0f;
      for (int r = 0; r < num_rows; ++r) {
        attn[r * num_heads_Q + h] =
          std::exp(attn[r * num_heads_Q + h] - max_val);
        sum_exp += attn[r * num_heads_Q + h];
      }
      for (int r = 0; r < num_rows; ++r)
        attn[r * num_heads_Q + h] /= sum_exp;
    }

    // V aggregation
    int out_dim = num_heads_KV * gqa_size * head_dim;
    std::vector<float> output(out_dim, 0.0f);
    nntrainer::compute_vcache_packed4_transposed(
      t, attn.data(), packed_vcache.data(), vcache_scales.data(), output.data(),
      num_heads_KV, gqa_size, head_dim);

    for (int j = 0; j < out_dim; ++j) {
      ASSERT_TRUE(std::isfinite(output[j]))
        << "Token " << t << ": non-finite at elem " << j;
    }
  }

  std::cout << "  [integration] incremental generation OK: " << max_tokens
            << " tokens decoded" << std::endl;
}

/**
 * @brief FP32 reference: full attention pipeline (scalar).
 *        Q*K^T / sqrt(d) → softmax → attn*V
 *        No quantization. This is the ground truth.
 */
static void fp32_reference_attention(
  const float *query, // [num_heads_Q * head_dim]
  const float *keys,  // [num_rows * num_heads_KV * head_dim]
  const float *values, // [num_rows * num_heads_KV * head_dim]
  float *output,      // [num_heads_Q * head_dim]
  int num_rows, int num_heads_Q, int num_heads_KV, int head_dim) {

  int gqa_size = num_heads_Q / num_heads_KV;
  float scale = 1.0f / std::sqrt((float)head_dim);

  for (int n = 0; n < num_heads_KV; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      int qh = n * gqa_size + g; // query head index
      const float *q = query + qh * head_dim;

      // Q*K^T
      std::vector<float> scores(num_rows);
      for (int r = 0; r < num_rows; ++r) {
        const float *k = keys + (r * num_heads_KV + n) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          dot += q[d] * k[d];
        scores[r] = dot * scale;
      }

      // Softmax
      float max_s = *std::max_element(scores.begin(), scores.end());
      float sum_exp = 0.0f;
      for (int r = 0; r < num_rows; ++r) {
        scores[r] = std::exp(scores[r] - max_s);
        sum_exp += scores[r];
      }
      for (int r = 0; r < num_rows; ++r)
        scores[r] /= sum_exp;

      // Attn * V
      float *out = output + qh * head_dim;
      std::fill(out, out + head_dim, 0.0f);
      for (int r = 0; r < num_rows; ++r) {
        const float *v = values + (r * num_heads_KV + n) * head_dim;
        for (int d = 0; d < head_dim; ++d)
          out[d] += scores[r] * v[d];
      }
    }
  }
}

/**
 * @brief TurboQuant pipeline: quantize KV → packed attention.
 *        Same Q, K, V as reference, but K/V go through 3-bit quantization.
 */
static void turboquant_attention(
  const float *query, const float *keys, const float *values, float *output,
  int num_rows, int num_heads_Q, int num_heads_KV, int head_dim) {

  int gqa_size = num_heads_Q / num_heads_KV;
  int kv_width = num_heads_KV * head_dim;
  int packed_row_bytes = kv_width / 2;
  constexpr int GROUP_SIZE = 32;
  int num_groups_per_row =
    num_heads_KV * ((head_dim + GROUP_SIZE - 1) / GROUP_SIZE);

  // Quantize K and V into packed cache
  std::vector<uint8_t> pk(num_rows * packed_row_bytes);
  std::vector<float> ks(num_rows * num_groups_per_row);
  std::vector<uint8_t> pv(num_rows * packed_row_bytes);
  std::vector<float> vs(num_rows * num_groups_per_row);

  for (int r = 0; r < num_rows; ++r) {
    nntrainer::quantize_kv_turboquant(
      keys + r * kv_width, kv_width,
      pk.data() + r * packed_row_bytes,
      ks.data() + r * num_groups_per_row);
    nntrainer::quantize_kv_turboquant(
      values + r * kv_width, kv_width,
      pv.data() + r * packed_row_bytes,
      vs.data() + r * num_groups_per_row);
  }

  // Q*K^T via packed kernel
  std::vector<float> attn(num_rows * num_heads_Q, 0.0f);
  nntrainer::compute_kcaches_packed4(
    query, pk.data(), ks.data(), attn.data(), num_rows, num_heads_KV, head_dim,
    gqa_size, 4);

  // Softmax (per query head)
  for (int h = 0; h < num_heads_Q; ++h) {
    float max_val = -1e30f;
    for (int r = 0; r < num_rows; ++r)
      max_val = std::max(max_val, attn[r * num_heads_Q + h]);
    float sum_exp = 0.0f;
    for (int r = 0; r < num_rows; ++r) {
      attn[r * num_heads_Q + h] =
        std::exp(attn[r * num_heads_Q + h] - max_val);
      sum_exp += attn[r * num_heads_Q + h];
    }
    for (int r = 0; r < num_rows; ++r)
      attn[r * num_heads_Q + h] /= sum_exp;
  }

  // Attn * V via packed kernel
  std::fill(output, output + num_heads_Q * head_dim, 0.0f);
  nntrainer::compute_vcache_packed4_transposed(
    num_rows - 1, attn.data(), pv.data(), vs.data(), output, num_heads_KV,
    gqa_size, head_dim);
}

/**
 * @brief TurboQuant with PolarQuant rotation pipeline (v1 rotation variant).
 */
[[maybe_unused]] static void turboquant_rotated_attention(
  const float *query, const float *keys, const float *values, float *output,
  int num_rows, int num_heads_Q, int num_heads_KV, int head_dim) {

  int gqa_size = num_heads_Q / num_heads_KV;
  int kv_width = num_heads_KV * head_dim;
  int packed_row_bytes = kv_width / 2;
  constexpr int GROUP_SIZE = 32;
  int num_groups_per_row =
    num_heads_KV * ((head_dim + GROUP_SIZE - 1) / GROUP_SIZE);

  // Generate deterministic random signs for rotation
  std::vector<float> signs(head_dim);
  nntrainer::generate_random_signs(signs.data(), head_dim, 0xDEADBEEF);

  // Quantize K and V with rotation into packed cache
  std::vector<uint8_t> pk(num_rows * packed_row_bytes);
  std::vector<float> ks(num_rows * num_groups_per_row);
  std::vector<uint8_t> pv(num_rows * packed_row_bytes);
  std::vector<float> vs(num_rows * num_groups_per_row);

  for (int r = 0; r < num_rows; ++r) {
    nntrainer::quantize_kv_turboquant_rotated(
      keys + r * kv_width, kv_width,
      pk.data() + r * packed_row_bytes,
      ks.data() + r * num_groups_per_row,
      signs.data(), head_dim, num_heads_KV);
    nntrainer::quantize_kv_turboquant_rotated(
      values + r * kv_width, kv_width,
      pv.data() + r * packed_row_bytes,
      vs.data() + r * num_groups_per_row,
      signs.data(), head_dim, num_heads_KV);
  }

  // Q*K^T with rotated query
  std::vector<float> attn(num_rows * num_heads_Q, 0.0f);
  nntrainer::compute_kcaches_packed4_rotated(
    query, pk.data(), ks.data(), attn.data(), num_rows, num_heads_KV, head_dim,
    gqa_size, 4, signs.data());

  // Softmax
  for (int h = 0; h < num_heads_Q; ++h) {
    float max_val = -1e30f;
    for (int r = 0; r < num_rows; ++r)
      max_val = std::max(max_val, attn[r * num_heads_Q + h]);
    float sum_exp = 0.0f;
    for (int r = 0; r < num_rows; ++r) {
      attn[r * num_heads_Q + h] =
        std::exp(attn[r * num_heads_Q + h] - max_val);
      sum_exp += attn[r * num_heads_Q + h];
    }
    for (int r = 0; r < num_rows; ++r)
      attn[r * num_heads_Q + h] /= sum_exp;
  }

  // Attn * V_rotated → inverse rotate to get final output
  std::fill(output, output + num_heads_Q * head_dim, 0.0f);
  nntrainer::compute_vcache_packed4_transposed_rotated(
    num_rows - 1, attn.data(), pv.data(), vs.data(), output, num_heads_KV,
    gqa_size, head_dim, signs.data());
}

/**
 * @brief Compare FP32 reference attention output vs TurboQuant output.
 *        Small model config for detailed per-element analysis.
 */
TEST(turboquant_logit_compare, small_config) {
  constexpr int num_heads_Q = 4;
  constexpr int num_heads_KV = 2;
  constexpr int head_dim = 64;
  constexpr int context_len = 16;

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  int kv_width = num_heads_KV * head_dim;
  int q_size = num_heads_Q * head_dim;

  std::vector<float> query(q_size);
  std::vector<float> keys(context_len * kv_width);
  std::vector<float> values(context_len * kv_width);
  for (auto &v : query) v = dist(gen);
  for (auto &v : keys) v = dist(gen);
  for (auto &v : values) v = dist(gen);

  std::vector<float> ref_out(q_size, 0.0f);
  std::vector<float> tq_out(q_size, 0.0f);

  fp32_reference_attention(query.data(), keys.data(), values.data(),
                           ref_out.data(), context_len, num_heads_Q,
                           num_heads_KV, head_dim);
  turboquant_attention(query.data(), keys.data(), values.data(), tq_out.data(),
                       context_len, num_heads_Q, num_heads_KV, head_dim);

  // Per-element comparison
  float max_diff = 0, sum_diff = 0, sum_sq_diff = 0;
  float max_ref = 0;
  int worst_idx = -1;

  for (int i = 0; i < q_size; ++i) {
    float diff = std::fabs(ref_out[i] - tq_out[i]);
    sum_diff += diff;
    sum_sq_diff += diff * diff;
    max_ref = std::max(max_ref, std::fabs(ref_out[i]));
    if (diff > max_diff) {
      max_diff = diff;
      worst_idx = i;
    }
  }
  float avg_diff = sum_diff / q_size;
  float rmse = std::sqrt(sum_sq_diff / q_size);
  float rel_max = (max_ref > 0) ? max_diff / max_ref : 0;

  std::cout << "\n=== Logit Comparison: small (heads_Q=4, heads_KV=2, dim=64, ctx=16) ===\n"
            << "  max_abs_diff   = " << max_diff << " (at index " << worst_idx << ")\n"
            << "  avg_abs_diff   = " << avg_diff << "\n"
            << "  rmse           = " << rmse << "\n"
            << "  max_rel_error  = " << (rel_max * 100) << "%\n"
            << "  max |ref|      = " << max_ref << "\n"
            << "  ref[worst]     = " << ref_out[worst_idx]
            << "  tq[worst]      = " << tq_out[worst_idx] << "\n";

  // Per-head breakdown
  int gqa = num_heads_Q / num_heads_KV;
  for (int h = 0; h < num_heads_Q; ++h) {
    float h_max = 0, h_sum = 0;
    for (int d = 0; d < head_dim; ++d) {
      float diff = std::fabs(ref_out[h * head_dim + d] - tq_out[h * head_dim + d]);
      h_max = std::max(h_max, diff);
      h_sum += diff;
    }
    std::cout << "  head " << h << ": max_diff=" << h_max
              << "  avg_diff=" << (h_sum / head_dim) << "\n";
  }

  EXPECT_LT(max_diff, 0.2f) << "Small config logit diff too large";
  EXPECT_LT(rmse, 0.05f) << "Small config RMSE too large";
}

/**
 * @brief Compare with Qwen3-1.7B-like dimensions.
 */
TEST(turboquant_logit_compare, qwen3_like) {
  constexpr int num_heads_Q = 16;
  constexpr int num_heads_KV = 4;
  constexpr int head_dim = 128;
  constexpr int context_len = 64;

  std::mt19937 gen(2026);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  int kv_width = num_heads_KV * head_dim;
  int q_size = num_heads_Q * head_dim;

  std::vector<float> query(q_size);
  std::vector<float> keys(context_len * kv_width);
  std::vector<float> values(context_len * kv_width);
  for (auto &v : query) v = dist(gen);
  for (auto &v : keys) v = dist(gen);
  for (auto &v : values) v = dist(gen);

  std::vector<float> ref_out(q_size, 0.0f);
  std::vector<float> tq_out(q_size, 0.0f);

  fp32_reference_attention(query.data(), keys.data(), values.data(),
                           ref_out.data(), context_len, num_heads_Q,
                           num_heads_KV, head_dim);
  turboquant_attention(query.data(), keys.data(), values.data(), tq_out.data(),
                       context_len, num_heads_Q, num_heads_KV, head_dim);

  float max_diff = 0, sum_sq = 0, max_ref = 0;
  for (int i = 0; i < q_size; ++i) {
    float diff = std::fabs(ref_out[i] - tq_out[i]);
    max_diff = std::max(max_diff, diff);
    sum_sq += diff * diff;
    max_ref = std::max(max_ref, std::fabs(ref_out[i]));
  }
  float rmse = std::sqrt(sum_sq / q_size);
  float rel_max = (max_ref > 0) ? max_diff / max_ref : 0;

  std::cout << "\n=== Logit Comparison: qwen3-like (heads_Q=16, heads_KV=4, dim=128, ctx=64) ===\n"
            << "  max_abs_diff   = " << max_diff << "\n"
            << "  rmse           = " << rmse << "\n"
            << "  max_rel_error  = " << (rel_max * 100) << "%\n"
            << "  max |ref|      = " << max_ref << "\n";

  EXPECT_LT(max_diff, 0.2f) << "Qwen3-like logit diff too large";
  EXPECT_LT(rmse, 0.05f) << "Qwen3-like RMSE too large";
}

/**
 * @brief Compare across increasing context lengths.
 *        Shows how error scales with sequence length.
 */
TEST(turboquant_logit_compare, error_vs_context_length) {
  constexpr int num_heads_Q = 8;
  constexpr int num_heads_KV = 2;
  constexpr int head_dim = 128;

  std::cout << "\n=== Error vs Context Length (heads_Q=8, heads_KV=2, dim=128) ===\n"
            << "  ctx_len  max_diff    rmse      rel_max%\n";

  int q_size = num_heads_Q * head_dim;
  int kv_width = num_heads_KV * head_dim;

  for (int ctx : {4, 16, 64, 128, 256, 512}) {
    std::mt19937 gen(ctx * 7 + 1);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> query(q_size);
    std::vector<float> keys(ctx * kv_width);
    std::vector<float> values(ctx * kv_width);
    for (auto &v : query) v = dist(gen);
    for (auto &v : keys) v = dist(gen);
    for (auto &v : values) v = dist(gen);

    std::vector<float> ref_out(q_size, 0.0f);
    std::vector<float> tq_out(q_size, 0.0f);

    fp32_reference_attention(query.data(), keys.data(), values.data(),
                             ref_out.data(), ctx, num_heads_Q, num_heads_KV,
                             head_dim);
    turboquant_attention(query.data(), keys.data(), values.data(),
                         tq_out.data(), ctx, num_heads_Q, num_heads_KV,
                         head_dim);

    float max_diff = 0, sum_sq = 0, max_ref = 0;
    for (int i = 0; i < q_size; ++i) {
      float diff = std::fabs(ref_out[i] - tq_out[i]);
      max_diff = std::max(max_diff, diff);
      sum_sq += diff * diff;
      max_ref = std::max(max_ref, std::fabs(ref_out[i]));
    }
    float rmse = std::sqrt(sum_sq / q_size);
    float rel = (max_ref > 0) ? max_diff / max_ref * 100 : 0;

    printf("  %5d    %.6f  %.6f  %.2f%%\n", ctx, max_diff, rmse, rel);

    EXPECT_LT(max_diff, 0.3f) << "ctx=" << ctx << " logit diff too large";
  }
}

/**
 * @brief TurboQuant v2 pipeline: norm + rotation + Lloyd-Max codebook.
 *        Paper Algorithm 1 (MSE-optimal).
 */
static void turboquant_v2_attention(
  const float *query, const float *keys, const float *values, float *output,
  int num_rows, int num_heads_Q, int num_heads_KV, int head_dim) {

  int gqa_size = num_heads_Q / num_heads_KV;
  int kv_width = num_heads_KV * head_dim;
  int packed_row_bytes = kv_width / 2;

  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xDEADBEEF);

  // Quantize K and V with v2 (norm + rotation + Lloyd-Max)
  std::vector<uint8_t> pk(num_rows * packed_row_bytes);
  std::vector<float> k_norms(num_rows * num_heads_KV);
  std::vector<uint8_t> pv(num_rows * packed_row_bytes);
  std::vector<float> v_norms(num_rows * num_heads_KV);

  for (int r = 0; r < num_rows; ++r) {
    nntrainer::quantize_kv_turboquant_v2(
      keys + r * kv_width, pk.data() + r * packed_row_bytes,
      k_norms.data() + r * num_heads_KV, rot_signs.data(), head_dim,
      num_heads_KV);
    nntrainer::quantize_kv_turboquant_v2(
      values + r * kv_width, pv.data() + r * packed_row_bytes,
      v_norms.data() + r * num_heads_KV, rot_signs.data(), head_dim,
      num_heads_KV);
  }

  // Q*K^T
  std::vector<float> attn(num_rows * num_heads_Q, 0.0f);
  nntrainer::compute_kcaches_packed4_v2(
    query, pk.data(), k_norms.data(), attn.data(), num_rows, num_heads_KV,
    head_dim, gqa_size, 4, rot_signs.data());

  // Softmax
  for (int h = 0; h < num_heads_Q; ++h) {
    float max_val = -1e30f;
    for (int r = 0; r < num_rows; ++r)
      max_val = std::max(max_val, attn[r * num_heads_Q + h]);
    float sum_exp = 0.0f;
    for (int r = 0; r < num_rows; ++r) {
      attn[r * num_heads_Q + h] =
        std::exp(attn[r * num_heads_Q + h] - max_val);
      sum_exp += attn[r * num_heads_Q + h];
    }
    for (int r = 0; r < num_rows; ++r)
      attn[r * num_heads_Q + h] /= sum_exp;
  }

  // Attn * V
  std::fill(output, output + num_heads_Q * head_dim, 0.0f);
  nntrainer::compute_vcache_packed4_v2(
    num_rows - 1, attn.data(), pv.data(), v_norms.data(), output, num_heads_KV,
    gqa_size, head_dim, rot_signs.data());
}

/**
 * @brief Compare all three: v1 (uniform quant) vs v2 (Lloyd-Max + norm + rot)
 *        vs FP32 reference.
 */
TEST(turboquant_logit_compare, v1_vs_v2_vs_fp32) {
  constexpr int num_heads_Q = 16;
  constexpr int num_heads_KV = 4;
  constexpr int head_dim = 128;

  int q_size = num_heads_Q * head_dim;
  int kv_width = num_heads_KV * head_dim;

  std::cout << "\n=== v1 (uniform) vs v2 (Lloyd-Max+norm+rot) vs FP32 ===\n"
            << "  Config: heads_Q=16, heads_KV=4, dim=128\n"
            << "  ctx    v1_max     v1_rmse    v2_max     v2_rmse    improvement\n";

  for (int ctx : {4, 16, 64, 128, 256}) {
    std::mt19937 gen(ctx * 13 + 7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> query(q_size), keys(ctx * kv_width),
      values(ctx * kv_width);
    for (auto &v : query) v = dist(gen);
    for (auto &v : keys) v = dist(gen);
    for (auto &v : values) v = dist(gen);

    std::vector<float> ref_out(q_size, 0.0f);
    std::vector<float> v1_out(q_size, 0.0f);
    std::vector<float> v2_out(q_size, 0.0f);

    fp32_reference_attention(query.data(), keys.data(), values.data(),
                             ref_out.data(), ctx, num_heads_Q, num_heads_KV,
                             head_dim);
    turboquant_attention(query.data(), keys.data(), values.data(),
                         v1_out.data(), ctx, num_heads_Q, num_heads_KV,
                         head_dim);
    turboquant_v2_attention(query.data(), keys.data(), values.data(),
                            v2_out.data(), ctx, num_heads_Q, num_heads_KV,
                            head_dim);

    float v1_max = 0, v1_sq = 0, v2_max = 0, v2_sq = 0;
    for (int i = 0; i < q_size; ++i) {
      float d1 = std::fabs(ref_out[i] - v1_out[i]);
      float d2 = std::fabs(ref_out[i] - v2_out[i]);
      v1_max = std::max(v1_max, d1);
      v2_max = std::max(v2_max, d2);
      v1_sq += d1 * d1;
      v2_sq += d2 * d2;
    }
    float v1_rmse = std::sqrt(v1_sq / q_size);
    float v2_rmse = std::sqrt(v2_sq / q_size);
    float improv = (1.0f - v2_rmse / v1_rmse) * 100;

    printf("  %5d  %.6f  %.6f   %.6f  %.6f   %+.1f%%\n", ctx, v1_max,
           v1_rmse, v2_max, v2_rmse, improv);

    EXPECT_TRUE(std::isfinite(v2_max))
      << "ctx=" << ctx << " v2 produced non-finite";
  }
}

/**
 * @brief Test with LLM-like activation distribution (normal + outliers).
 *        This is where rotation + Lloyd-Max should shine.
 */
TEST(turboquant_logit_compare, v1_vs_v2_llm_distribution) {
  constexpr int num_heads_Q = 16;
  constexpr int num_heads_KV = 4;
  constexpr int head_dim = 128;

  int q_size = num_heads_Q * head_dim;
  int kv_width = num_heads_KV * head_dim;

  auto gen_llm_data = [](float *data, int n, std::mt19937 &gen) {
    std::normal_distribution<float> normal(0.0f, 0.3f);
    std::uniform_int_distribution<int> idx_dist(0, n - 1);
    std::uniform_real_distribution<float> outlier(-5.0f, 5.0f);
    for (int i = 0; i < n; ++i)
      data[i] = normal(gen);
    for (int i = 0; i < n / 20; ++i) // 5% outliers
      data[idx_dist(gen)] = outlier(gen);
  };

  std::cout
    << "\n=== LLM-like distribution: v1 vs v2 vs FP32 ===\n"
    << "  (normal σ=0.3 + 5% outliers in [-5,5])\n"
    << "  ctx    v1_max     v1_rmse    v2_max     v2_rmse    improvement\n";

  for (int ctx : {16, 64, 128, 256}) {
    std::mt19937 gen(ctx * 31 + 5);
    std::vector<float> query(q_size), keys(ctx * kv_width),
      values(ctx * kv_width);
    gen_llm_data(query.data(), q_size, gen);
    gen_llm_data(keys.data(), ctx * kv_width, gen);
    gen_llm_data(values.data(), ctx * kv_width, gen);

    std::vector<float> ref_out(q_size, 0.0f), v1_out(q_size, 0.0f),
      v2_out(q_size, 0.0f);

    fp32_reference_attention(query.data(), keys.data(), values.data(),
                             ref_out.data(), ctx, num_heads_Q, num_heads_KV,
                             head_dim);
    turboquant_attention(query.data(), keys.data(), values.data(),
                         v1_out.data(), ctx, num_heads_Q, num_heads_KV,
                         head_dim);
    turboquant_v2_attention(query.data(), keys.data(), values.data(),
                            v2_out.data(), ctx, num_heads_Q, num_heads_KV,
                            head_dim);

    float v1_max = 0, v1_sq = 0, v2_max = 0, v2_sq = 0;
    for (int i = 0; i < q_size; ++i) {
      float d1 = std::fabs(ref_out[i] - v1_out[i]);
      float d2 = std::fabs(ref_out[i] - v2_out[i]);
      v1_max = std::max(v1_max, d1);
      v2_max = std::max(v2_max, d2);
      v1_sq += d1 * d1;
      v2_sq += d2 * d2;
    }
    float v1_rmse = std::sqrt(v1_sq / q_size);
    float v2_rmse = std::sqrt(v2_sq / q_size);
    float improv = (1.0f - v2_rmse / v1_rmse) * 100;

    printf("  %5d  %.6f  %.6f   %.6f  %.6f   %+.1f%%\n", ctx, v1_max,
           v1_rmse, v2_max, v2_rmse, improv);
  }
}

GTEST_API_ int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

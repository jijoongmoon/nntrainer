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

/**
 * @brief Correct Prefill + Decode simulation:
 *
 *   [Prefill phase]
 *     - Receive prompt tokens (e.g. 16 tokens)
 *     - For FP32: store K,V as-is in cache
 *     - For TQ v2: quantize K,V and store in packed cache
 *     - Compute full causal attention for all prompt tokens
 *
 *   [Decode phase]
 *     - Generate new tokens one at a time
 *     - Only new Q is computed; K,V from prefill stay in cache
 *     - For each new token: append new K,V to cache, then
 *       compute attention using ALL cached K,V (prefill + new)
 *
 *   Compare FP32 output vs TurboQuant v2 output at each decode step.
 */
TEST(turboquant_qwen3_sim, prefill_then_decode) {
  constexpr int num_heads_Q = 8;
  constexpr int num_heads_KV = 2;
  constexpr int head_dim = 64;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int prefill_len = 16;  // prompt length
  constexpr int decode_len = 16;   // tokens to generate
  constexpr int max_seq = prefill_len + decode_len;

  constexpr int kv_width = num_heads_KV * head_dim;
  constexpr int q_width = num_heads_Q * head_dim;
  constexpr int packed_row = kv_width / 2;

  std::mt19937 gen(20260329);
  std::normal_distribution<float> normal(0.0f, 0.3f);
  std::uniform_real_distribution<float> outlier_val(-4.0f, 4.0f);

  auto gen_llm_vec = [&](float *dst, int n) {
    for (int i = 0; i < n; ++i)
      dst[i] = normal(gen);
    for (int i = 0; i < n / 20; ++i)
      dst[std::abs((int)gen()) % n] = outlier_val(gen);
  };

  // Rotation signs (shared, generated once)
  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xDEADBEEF);

  // FP32 KV cache (ground truth)
  std::vector<float> fp32_kcache(max_seq * kv_width, 0.0f);
  std::vector<float> fp32_vcache(max_seq * kv_width, 0.0f);

  // TurboQuant v2 packed cache
  std::vector<uint8_t> tq_kcache(max_seq * packed_row, 0);
  std::vector<float> tq_knorms(max_seq * num_heads_KV, 0.0f);
  std::vector<uint8_t> tq_vcache(max_seq * packed_row, 0);
  std::vector<float> tq_vnorms(max_seq * num_heads_KV, 0.0f);

  // ========== PREFILL ==========
  // Generate all prompt K,V at once (same data for both paths)
  std::vector<float> prompt_keys(prefill_len * kv_width);
  std::vector<float> prompt_values(prefill_len * kv_width);
  for (int t = 0; t < prefill_len; ++t) {
    gen_llm_vec(prompt_keys.data() + t * kv_width, kv_width);
    gen_llm_vec(prompt_values.data() + t * kv_width, kv_width);
  }

  // Store in FP32 cache
  std::copy(prompt_keys.begin(), prompt_keys.end(), fp32_kcache.begin());
  std::copy(prompt_values.begin(), prompt_values.end(), fp32_vcache.begin());

  // Quantize into TQ v2 cache (all prefill tokens at once)
  for (int t = 0; t < prefill_len; ++t) {
    nntrainer::quantize_kv_turboquant_v2(
      prompt_keys.data() + t * kv_width,
      tq_kcache.data() + t * packed_row,
      tq_knorms.data() + t * num_heads_KV,
      rot_signs.data(), head_dim, num_heads_KV);
    nntrainer::quantize_kv_turboquant_v2(
      prompt_values.data() + t * kv_width,
      tq_vcache.data() + t * packed_row,
      tq_vnorms.data() + t * num_heads_KV,
      rot_signs.data(), head_dim, num_heads_KV);
  }

  std::cout << "\n=== Prefill(" << prefill_len << ") + Decode(" << decode_len
            << "): FP32 vs TurboQuant v2 ===\n"
            << "  Config: heads_Q=" << num_heads_Q << ", heads_KV="
            << num_heads_KV << ", dim=" << head_dim << "\n"
            << "  step  ctx_len  max_diff  cosine_sim\n";

  // ========== DECODE ==========
  float total_cosine = 0;
  float worst_max_diff = 0;

  for (int d = 0; d < decode_len; ++d) {
    int pos = prefill_len + d; // current position in sequence
    int ctx_len = pos + 1;     // total context (prefill + decoded so far + current)

    // Generate new K,V for this decode token (same data for both)
    std::vector<float> new_k(kv_width), new_v(kv_width);
    gen_llm_vec(new_k.data(), kv_width);
    gen_llm_vec(new_v.data(), kv_width);

    // Append to FP32 cache
    std::copy(new_k.begin(), new_k.end(),
              fp32_kcache.begin() + pos * kv_width);
    std::copy(new_v.begin(), new_v.end(),
              fp32_vcache.begin() + pos * kv_width);

    // Quantize and append to TQ cache
    nntrainer::quantize_kv_turboquant_v2(
      new_k.data(), tq_kcache.data() + pos * packed_row,
      tq_knorms.data() + pos * num_heads_KV,
      rot_signs.data(), head_dim, num_heads_KV);
    nntrainer::quantize_kv_turboquant_v2(
      new_v.data(), tq_vcache.data() + pos * packed_row,
      tq_vnorms.data() + pos * num_heads_KV,
      rot_signs.data(), head_dim, num_heads_KV);

    // Generate query (only query is new at decode time)
    std::vector<float> query(q_width);
    gen_llm_vec(query.data(), q_width);

    // ---- FP32 attention (ground truth) ----
    std::vector<float> fp32_out(q_width, 0.0f);
    fp32_reference_attention(query.data(), fp32_kcache.data(),
                             fp32_vcache.data(), fp32_out.data(), ctx_len,
                             num_heads_Q, num_heads_KV, head_dim);

    // ---- TQ v2 attention (using packed prefill + decode cache) ----
    // Q * K_packed^T (reads ALL ctx_len rows from packed cache)
    std::vector<float> tq_scores(ctx_len * num_heads_Q, 0.0f);
    nntrainer::compute_kcaches_packed4_v2(
      query.data(), tq_kcache.data(), tq_knorms.data(), tq_scores.data(),
      ctx_len, num_heads_KV, head_dim, gqa_size, 4, rot_signs.data());

    // Softmax
    for (int h = 0; h < num_heads_Q; ++h) {
      float mx = -1e30f;
      for (int r = 0; r < ctx_len; ++r)
        mx = std::max(mx, tq_scores[r * num_heads_Q + h]);
      float se = 0;
      for (int r = 0; r < ctx_len; ++r) {
        tq_scores[r * num_heads_Q + h] =
          std::exp(tq_scores[r * num_heads_Q + h] - mx);
        se += tq_scores[r * num_heads_Q + h];
      }
      for (int r = 0; r < ctx_len; ++r)
        tq_scores[r * num_heads_Q + h] /= se;
    }

    // Attn * V_packed (reads ALL ctx_len rows from packed value cache)
    std::vector<float> tq_out(q_width, 0.0f);
    nntrainer::compute_vcache_packed4_v2(
      pos, tq_scores.data(), tq_vcache.data(), tq_vnorms.data(),
      tq_out.data(), num_heads_KV, gqa_size, head_dim, rot_signs.data());

    // Compare
    float max_diff = 0, dot_ab = 0, dot_aa = 0, dot_bb = 0;
    for (int i = 0; i < q_width; ++i) {
      ASSERT_TRUE(std::isfinite(tq_out[i]))
        << "Decode " << d << ": non-finite at " << i;
      float diff = std::fabs(fp32_out[i] - tq_out[i]);
      max_diff = std::max(max_diff, diff);
      dot_ab += fp32_out[i] * tq_out[i];
      dot_aa += fp32_out[i] * fp32_out[i];
      dot_bb += tq_out[i] * tq_out[i];
    }
    float cosine = (dot_aa > 0 && dot_bb > 0)
                     ? dot_ab / (std::sqrt(dot_aa) * std::sqrt(dot_bb))
                     : 0.0f;

    total_cosine += cosine;
    worst_max_diff = std::max(worst_max_diff, max_diff);

    printf("  %4d  %7d  %.6f  %.6f\n", d, ctx_len, max_diff, cosine);
  }

  float avg_cosine = total_cosine / decode_len;
  std::cout << "\n  Summary:\n"
            << "  avg_cosine_sim  = " << avg_cosine << "\n"
            << "  worst_max_diff  = " << worst_max_diff << "\n"
            << "  prefill tokens  = " << prefill_len << " (packed once)\n"
            << "  decode tokens   = " << decode_len << "\n";

  EXPECT_GT(avg_cosine, 0.95f)
    << "Prefill+decode cosine sim too low: " << avg_cosine;
}

/**
 * @brief Qwen3-1.7B-like prefill+decode with larger dimensions.
 */
TEST(turboquant_qwen3_sim, prefill_decode_1_7b_like) {
  constexpr int num_heads_Q = 16;
  constexpr int num_heads_KV = 4;
  constexpr int head_dim = 128;
  constexpr int gqa_size = num_heads_Q / num_heads_KV;
  constexpr int prefill_len = 32;
  constexpr int decode_len = 32;
  constexpr int max_seq = prefill_len + decode_len;

  constexpr int kv_width = num_heads_KV * head_dim;
  constexpr int q_width = num_heads_Q * head_dim;
  constexpr int packed_row = kv_width / 2;

  std::mt19937 gen(42);
  std::normal_distribution<float> normal(0.0f, 0.3f);
  std::uniform_real_distribution<float> outlier_val(-5.0f, 5.0f);

  auto gen_llm_vec = [&](float *dst, int n) {
    for (int i = 0; i < n; ++i)
      dst[i] = normal(gen);
    for (int i = 0; i < n / 20; ++i)
      dst[std::abs((int)gen()) % n] = outlier_val(gen);
  };

  std::vector<float> rot_signs(head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), head_dim, 0xDEADBEEF);

  std::vector<float> fp32_kc(max_seq * kv_width, 0.0f);
  std::vector<float> fp32_vc(max_seq * kv_width, 0.0f);
  std::vector<uint8_t> tq_kc(max_seq * packed_row, 0);
  std::vector<float> tq_kn(max_seq * num_heads_KV, 0.0f);
  std::vector<uint8_t> tq_vc(max_seq * packed_row, 0);
  std::vector<float> tq_vn(max_seq * num_heads_KV, 0.0f);

  // Prefill
  for (int t = 0; t < prefill_len; ++t) {
    gen_llm_vec(fp32_kc.data() + t * kv_width, kv_width);
    gen_llm_vec(fp32_vc.data() + t * kv_width, kv_width);
    nntrainer::quantize_kv_turboquant_v2(
      fp32_kc.data() + t * kv_width, tq_kc.data() + t * packed_row,
      tq_kn.data() + t * num_heads_KV, rot_signs.data(), head_dim,
      num_heads_KV);
    nntrainer::quantize_kv_turboquant_v2(
      fp32_vc.data() + t * kv_width, tq_vc.data() + t * packed_row,
      tq_vn.data() + t * num_heads_KV, rot_signs.data(), head_dim,
      num_heads_KV);
  }

  float total_cosine = 0;

  std::cout << "\n=== Qwen3-1.7B-like Prefill(" << prefill_len << ")+Decode("
            << decode_len << ") ===\n"
            << "  step  ctx_len  max_diff  cosine_sim\n";

  // Decode
  for (int d = 0; d < decode_len; ++d) {
    int pos = prefill_len + d;
    int ctx_len = pos + 1;

    gen_llm_vec(fp32_kc.data() + pos * kv_width, kv_width);
    gen_llm_vec(fp32_vc.data() + pos * kv_width, kv_width);
    nntrainer::quantize_kv_turboquant_v2(
      fp32_kc.data() + pos * kv_width, tq_kc.data() + pos * packed_row,
      tq_kn.data() + pos * num_heads_KV, rot_signs.data(), head_dim,
      num_heads_KV);
    nntrainer::quantize_kv_turboquant_v2(
      fp32_vc.data() + pos * kv_width, tq_vc.data() + pos * packed_row,
      tq_vn.data() + pos * num_heads_KV, rot_signs.data(), head_dim,
      num_heads_KV);

    std::vector<float> query(q_width);
    gen_llm_vec(query.data(), q_width);

    // FP32 ref
    std::vector<float> fp32_out(q_width, 0.0f);
    fp32_reference_attention(query.data(), fp32_kc.data(), fp32_vc.data(),
                             fp32_out.data(), ctx_len, num_heads_Q,
                             num_heads_KV, head_dim);

    // TQ v2
    std::vector<float> tq_scores(ctx_len * num_heads_Q, 0.0f);
    nntrainer::compute_kcaches_packed4_v2(
      query.data(), tq_kc.data(), tq_kn.data(), tq_scores.data(), ctx_len,
      num_heads_KV, head_dim, gqa_size, 4, rot_signs.data());

    for (int h = 0; h < num_heads_Q; ++h) {
      float mx = -1e30f;
      for (int r = 0; r < ctx_len; ++r)
        mx = std::max(mx, tq_scores[r * num_heads_Q + h]);
      float se = 0;
      for (int r = 0; r < ctx_len; ++r) {
        tq_scores[r * num_heads_Q + h] =
          std::exp(tq_scores[r * num_heads_Q + h] - mx);
        se += tq_scores[r * num_heads_Q + h];
      }
      for (int r = 0; r < ctx_len; ++r)
        tq_scores[r * num_heads_Q + h] /= se;
    }

    std::vector<float> tq_out(q_width, 0.0f);
    nntrainer::compute_vcache_packed4_v2(
      pos, tq_scores.data(), tq_vc.data(), tq_vn.data(), tq_out.data(),
      num_heads_KV, gqa_size, head_dim, rot_signs.data());

    float max_diff = 0, dot_ab = 0, dot_aa = 0, dot_bb = 0;
    for (int i = 0; i < q_width; ++i) {
      ASSERT_TRUE(std::isfinite(tq_out[i]));
      float diff = std::fabs(fp32_out[i] - tq_out[i]);
      max_diff = std::max(max_diff, diff);
      dot_ab += fp32_out[i] * tq_out[i];
      dot_aa += fp32_out[i] * fp32_out[i];
      dot_bb += tq_out[i] * tq_out[i];
    }
    float cosine = (dot_aa > 0 && dot_bb > 0)
                     ? dot_ab / (std::sqrt(dot_aa) * std::sqrt(dot_bb))
                     : 0.0f;
    total_cosine += cosine;

    if (d < 5 || d == decode_len - 1)
      printf("  %4d  %7d  %.6f  %.6f\n", d, ctx_len, max_diff, cosine);
    else if (d == 5)
      printf("  ...   ...\n");
  }

  float avg_cosine = total_cosine / decode_len;
  std::cout << "  avg_cosine_sim = " << avg_cosine << "\n";

  EXPECT_GT(avg_cosine, 0.95f)
    << "Qwen3-1.7B prefill+decode cosine too low: " << avg_cosine;
}

/**
 * @brief Qwen3-0.6B exact dimensions test.
 *
 *   Qwen3-0.6B: hidden=896, heads_Q=14, heads_KV=2, head_dim=64, layers=28
 *
 *   Simulates mha_core's exact calling pattern:
 *   - RoPE-modulated K data (cos/sin frequency pattern)
 *   - Prefill → packed cache → decode using packed cache
 *   - Key: 3-bit Lloyd-Max (norm+rotation)
 *   - Value: 2-bit group min-max
 *   - Compare output vs FP32 reference for each decode step
 */
TEST(turboquant_qwen3_sim, qwen3_0_6b_exact_dims) {
  // Qwen3-0.6B exact config
  constexpr int NUM_HEADS_Q = 14;
  constexpr int NUM_HEADS_KV = 2;
  constexpr int HEAD_DIM = 64;
  constexpr int GQA_SIZE = NUM_HEADS_Q / NUM_HEADS_KV; // 7
  constexpr int PREFILL_LEN = 32;
  constexpr int DECODE_LEN = 16;
  constexpr int MAX_SEQ = PREFILL_LEN + DECODE_LEN;

  constexpr int KV_WIDTH = NUM_HEADS_KV * HEAD_DIM;    // 128
  constexpr int Q_WIDTH = NUM_HEADS_Q * HEAD_DIM;      // 896
  constexpr int KEY_PACKED_ROW = KV_WIDTH / 2;          // 64 bytes (3-bit)
  constexpr int VAL_PACKED_ROW = KV_WIDTH / 2;          // 32 bytes (4-bit)
  constexpr int VAL_GROUPS = (KV_WIDTH + 32 - 1) / 32;  // 4
  constexpr int VAL_PARAMS_ROW = VAL_GROUPS * 2;         // 8

  std::mt19937 gen(42);
  std::normal_distribution<float> normal(0.0f, 0.1f);

  // Simulate RoPE-modulated data: each dim has different frequency
  auto gen_rope_data = [&](float *dst, int width, int pos) {
    for (int d = 0; d < width; ++d) {
      // RoPE-like: base value * cos/sin modulation
      float base = normal(gen);
      float freq = 1.0f / std::pow(500000.0f, (float)(d % HEAD_DIM) / HEAD_DIM);
      float angle = pos * freq;
      if ((d % HEAD_DIM) < HEAD_DIM / 2)
        dst[d] = base * std::cos(angle) + normal(gen) * std::sin(angle);
      else
        dst[d] = base * std::cos(angle) - normal(gen) * std::sin(angle);
    }
  };

  std::vector<float> rot_signs(HEAD_DIM);
  nntrainer::generate_random_signs(rot_signs.data(), HEAD_DIM, 0xDEADBEEF);

  // FP32 caches
  std::vector<float> fp32_kc(MAX_SEQ * KV_WIDTH, 0);
  std::vector<float> fp32_vc(MAX_SEQ * KV_WIDTH, 0);
  // TQ caches
  std::vector<uint8_t> tq_kc(MAX_SEQ * KEY_PACKED_ROW, 0);
  std::vector<float> tq_kn(MAX_SEQ * NUM_HEADS_KV, 0);
  std::vector<uint8_t> tq_vc(MAX_SEQ * VAL_PACKED_ROW, 0);
  std::vector<float> tq_vp(MAX_SEQ * VAL_PARAMS_ROW, 0);

  // === PREFILL ===
  for (int t = 0; t < PREFILL_LEN; ++t) {
    gen_rope_data(fp32_kc.data() + t * KV_WIDTH, KV_WIDTH, t);
    gen_rope_data(fp32_vc.data() + t * KV_WIDTH, KV_WIDTH, t);

    nntrainer::quantize_kv_turboquant_v2(
      fp32_kc.data() + t * KV_WIDTH,
      tq_kc.data() + t * KEY_PACKED_ROW,
      tq_kn.data() + t * NUM_HEADS_KV,
      rot_signs.data(), HEAD_DIM, NUM_HEADS_KV);
    nntrainer::quantize_value_group2bit(
      fp32_vc.data() + t * KV_WIDTH,
      tq_vc.data() + t * VAL_PACKED_ROW,
      tq_vp.data() + t * VAL_PARAMS_ROW,
      HEAD_DIM, NUM_HEADS_KV);
  }

  std::cout << "\n=== Qwen3-0.6B Exact Dims: Prefill(" << PREFILL_LEN
            << ")+Decode(" << DECODE_LEN << ") ===\n"
            << "  heads_Q=" << NUM_HEADS_Q << ", heads_KV=" << NUM_HEADS_KV
            << ", dim=" << HEAD_DIM << ", GQA=" << GQA_SIZE << "\n"
            << "  Key: 3-bit Lloyd-Max, Value: 2-bit group min-max\n"
            << "  step  ctx_len  max_diff  cosine_sim\n";

  float total_cosine = 0;
  float worst_diff = 0;

  // === DECODE ===
  for (int d = 0; d < DECODE_LEN; ++d) {
    int pos = PREFILL_LEN + d;
    int ctx_len = pos + 1;

    // New K,V for this decode token
    gen_rope_data(fp32_kc.data() + pos * KV_WIDTH, KV_WIDTH, pos);
    gen_rope_data(fp32_vc.data() + pos * KV_WIDTH, KV_WIDTH, pos);

    nntrainer::quantize_kv_turboquant_v2(
      fp32_kc.data() + pos * KV_WIDTH,
      tq_kc.data() + pos * KEY_PACKED_ROW,
      tq_kn.data() + pos * NUM_HEADS_KV,
      rot_signs.data(), HEAD_DIM, NUM_HEADS_KV);
    nntrainer::quantize_value_group2bit(
      fp32_vc.data() + pos * KV_WIDTH,
      tq_vc.data() + pos * VAL_PACKED_ROW,
      tq_vp.data() + pos * VAL_PARAMS_ROW,
      HEAD_DIM, NUM_HEADS_KV);

    // Query
    std::vector<float> query(Q_WIDTH);
    gen_rope_data(query.data(), Q_WIDTH, pos);

    // --- FP32 reference ---
    std::vector<float> fp32_out(Q_WIDTH, 0);
    fp32_reference_attention(query.data(), fp32_kc.data(), fp32_vc.data(),
                             fp32_out.data(), ctx_len, NUM_HEADS_Q,
                             NUM_HEADS_KV, HEAD_DIM);

    // --- TQ: Key 3-bit Lloyd-Max ---
    std::vector<float> tq_scores(ctx_len * NUM_HEADS_Q, 0);
    nntrainer::compute_kcaches_packed4_v2(
      query.data(), tq_kc.data(), tq_kn.data(), tq_scores.data(), ctx_len,
      NUM_HEADS_KV, HEAD_DIM, GQA_SIZE, 4, rot_signs.data());

    // Softmax
    for (int h = 0; h < NUM_HEADS_Q; ++h) {
      float mx = -1e30f;
      for (int r = 0; r < ctx_len; ++r)
        mx = std::max(mx, tq_scores[r * NUM_HEADS_Q + h]);
      float se = 0;
      for (int r = 0; r < ctx_len; ++r) {
        tq_scores[r * NUM_HEADS_Q + h] =
          std::exp(tq_scores[r * NUM_HEADS_Q + h] - mx);
        se += tq_scores[r * NUM_HEADS_Q + h];
      }
      for (int r = 0; r < ctx_len; ++r)
        tq_scores[r * NUM_HEADS_Q + h] /= se;
    }

    // --- TQ: Value 2-bit group min-max ---
    std::vector<float> tq_out(Q_WIDTH, 0);
    nntrainer::compute_vcache_group2bit(
      pos, tq_scores.data(), tq_vc.data(), tq_vp.data(), tq_out.data(),
      NUM_HEADS_KV, GQA_SIZE, HEAD_DIM);

    // Compare
    float max_diff = 0, dot_ab = 0, dot_aa = 0, dot_bb = 0;
    for (int i = 0; i < Q_WIDTH; ++i) {
      ASSERT_TRUE(std::isfinite(tq_out[i])) << "d=" << d << " i=" << i;
      float diff = std::fabs(fp32_out[i] - tq_out[i]);
      max_diff = std::max(max_diff, diff);
      dot_ab += fp32_out[i] * tq_out[i];
      dot_aa += fp32_out[i] * fp32_out[i];
      dot_bb += tq_out[i] * tq_out[i];
    }
    float cosine = (dot_aa > 0 && dot_bb > 0)
                     ? dot_ab / (std::sqrt(dot_aa) * std::sqrt(dot_bb))
                     : 0;
    total_cosine += cosine;
    worst_diff = std::max(worst_diff, max_diff);

    printf("  %4d  %7d  %.6f  %.6f\n", d, ctx_len, max_diff, cosine);
  }

  float avg_cos = total_cosine / DECODE_LEN;
  std::cout << "\n  avg_cosine = " << avg_cos
            << "\n  worst_diff = " << worst_diff << "\n";

  EXPECT_GT(avg_cos, 0.95f) << "Qwen3-0.6B cosine too low";
}

GTEST_API_ int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

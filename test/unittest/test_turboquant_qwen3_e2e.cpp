// SPDX-License-Identifier: Apache-2.0
/**
 * @brief TurboQuant end-to-end test: Qwen3 model with random weights.
 *        Runs same model twice (FP16 KV cache vs TurboQuant v2 KV cache)
 *        and compares the attention outputs at each layer.
 */
#include <cpu_backend.h>
#include <turboquant_utils.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Simulate one full Qwen3 decoder layer's attention:
//   Q_proj(hidden) → Q
//   K_proj(hidden) → K → RoPE → cache
//   V_proj(hidden) → V → cache
//   Attention(Q, cached_K, cached_V) → output

struct Qwen3Config {
  int hidden_size;
  int num_heads_q;
  int num_heads_kv;
  int head_dim;
  int num_layers;
  int prefill_len;
  int decode_len;
};

void run_attention_comparison(const Qwen3Config &cfg) {
  int gqa_size = cfg.num_heads_q / cfg.num_heads_kv;
  int kv_width = cfg.num_heads_kv * cfg.head_dim;
  int q_width = cfg.num_heads_q * cfg.head_dim;
  int max_seq = cfg.prefill_len + cfg.decode_len;
  int key_packed_row = kv_width / 2;
  int val_packed_row = kv_width / 2; // 4-bit
  int val_groups = (kv_width + 32 - 1) / 32;
  int val_params_row = val_groups * 2;

  std::mt19937 gen(42);

  // Random weight matrices for Q/K/V projections (simplified)
  std::normal_distribution<float> weight_dist(0.0f, 0.02f);
  std::vector<float> W_q(cfg.hidden_size * q_width);
  std::vector<float> W_k(cfg.hidden_size * kv_width);
  std::vector<float> W_v(cfg.hidden_size * kv_width);
  for (auto &v : W_q) v = weight_dist(gen);
  for (auto &v : W_k) v = weight_dist(gen);
  for (auto &v : W_v) v = weight_dist(gen);

  // Rotation signs
  std::vector<float> rot_signs(cfg.head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), cfg.head_dim, 0xDEADBEEF);

  // Caches
  std::vector<float> fp32_kc(max_seq * kv_width, 0);
  std::vector<float> fp32_vc(max_seq * kv_width, 0);
  std::vector<uint8_t> tq_kc(max_seq * key_packed_row, 0);
  std::vector<float> tq_kn(max_seq * cfg.num_heads_kv, 0);
  std::vector<uint8_t> tq_vc(max_seq * val_packed_row, 0);
  std::vector<float> tq_vp(max_seq * val_params_row, 0);

  auto matmul = [](const float *A, const float *B, float *C, int M, int N,
                   int K) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        float sum = 0;
        for (int k = 0; k < K; ++k)
          sum += A[i * K + k] * B[k * N + j];
        C[i * N + j] = sum;
      }
    }
  };

  // Simple RoPE
  auto apply_rope = [&](float *data, int width, int head_dim, int pos) {
    for (int h = 0; h < width / head_dim; ++h) {
      float *head = data + h * head_dim;
      for (int d = 0; d < head_dim / 2; ++d) {
        float freq = 1.0f / std::pow(500000.0f, 2.0f * d / head_dim);
        float angle = pos * freq;
        float cos_a = std::cos(angle), sin_a = std::sin(angle);
        float x0 = head[d], x1 = head[d + head_dim / 2];
        head[d] = x0 * cos_a - x1 * sin_a;
        head[d + head_dim / 2] = x0 * sin_a + x1 * cos_a;
      }
    }
  };

  std::normal_distribution<float> input_dist(0.0f, 0.1f);
  float total_cosine = 0;
  float worst_diff = 0;
  int total_steps = 0;

  std::cout << "  Qwen3 (h=" << cfg.hidden_size << " Q=" << cfg.num_heads_q
            << " KV=" << cfg.num_heads_kv << " d=" << cfg.head_dim
            << " GQA=" << gqa_size << ")\n"
            << "  Prefill=" << cfg.prefill_len
            << " Decode=" << cfg.decode_len << "\n"
            << "  step  ctx  max_diff  cosine\n";

  for (int t = 0; t < max_seq; ++t) {
    // Generate random hidden state
    std::vector<float> hidden(cfg.hidden_size);
    for (auto &v : hidden) v = input_dist(gen);

    // Project Q, K, V
    std::vector<float> q(q_width), k(kv_width), v(kv_width);
    matmul(hidden.data(), W_q.data(), q.data(), 1, q_width, cfg.hidden_size);
    matmul(hidden.data(), W_k.data(), k.data(), 1, kv_width, cfg.hidden_size);
    matmul(hidden.data(), W_v.data(), v.data(), 1, kv_width, cfg.hidden_size);

    // RoPE on Q and K
    apply_rope(q.data(), q_width, cfg.head_dim, t);
    apply_rope(k.data(), kv_width, cfg.head_dim, t);

    // Store in FP32 cache
    std::copy(k.begin(), k.end(), fp32_kc.begin() + t * kv_width);
    std::copy(v.begin(), v.end(), fp32_vc.begin() + t * kv_width);

    // Quantize into TQ cache
    nntrainer::quantize_kv_turboquant_v2(
      k.data(), tq_kc.data() + t * key_packed_row,
      tq_kn.data() + t * cfg.num_heads_kv, rot_signs.data(), cfg.head_dim,
      cfg.num_heads_kv);
    nntrainer::quantize_value_group2bit(
      v.data(), tq_vc.data() + t * val_packed_row,
      tq_vp.data() + t * val_params_row, cfg.head_dim, cfg.num_heads_kv);

    // Only compare during decode phase
    if (t < cfg.prefill_len)
      continue;

    int ctx = t + 1;

    // FP32 attention
    std::vector<float> fp32_out(q_width, 0);
    for (int nh = 0; nh < cfg.num_heads_kv; ++nh) {
      for (int g = 0; g < gqa_size; ++g) {
        int qh = nh * gqa_size + g;
        const float *qp = q.data() + qh * cfg.head_dim;
        std::vector<float> scores(ctx);
        float scale = 1.0f / std::sqrt((float)cfg.head_dim);
        for (int r = 0; r < ctx; ++r) {
          const float *kp = fp32_kc.data() + (r * cfg.num_heads_kv + nh) * cfg.head_dim;
          float dot = 0;
          for (int d = 0; d < cfg.head_dim; ++d) dot += qp[d] * kp[d];
          scores[r] = dot * scale;
        }
        float mx = *std::max_element(scores.begin(), scores.end());
        float se = 0;
        for (auto &s : scores) { s = std::exp(s - mx); se += s; }
        for (auto &s : scores) s /= se;
        float *out = fp32_out.data() + qh * cfg.head_dim;
        for (int r = 0; r < ctx; ++r) {
          const float *vp = fp32_vc.data() + (r * cfg.num_heads_kv + nh) * cfg.head_dim;
          for (int d = 0; d < cfg.head_dim; ++d) out[d] += scores[r] * vp[d];
        }
      }
    }

    // TQ attention
    std::vector<float> tq_scores(ctx * cfg.num_heads_q, 0);
    nntrainer::compute_kcaches_packed4_v2(
      q.data(), tq_kc.data(), tq_kn.data(), tq_scores.data(), ctx,
      cfg.num_heads_kv, cfg.head_dim, gqa_size, 4, rot_signs.data());

    for (int h = 0; h < cfg.num_heads_q; ++h) {
      float mx = -1e30f;
      for (int r = 0; r < ctx; ++r)
        mx = std::max(mx, tq_scores[r * cfg.num_heads_q + h]);
      float se = 0;
      for (int r = 0; r < ctx; ++r) {
        tq_scores[r * cfg.num_heads_q + h] =
          std::exp(tq_scores[r * cfg.num_heads_q + h] - mx);
        se += tq_scores[r * cfg.num_heads_q + h];
      }
      for (int r = 0; r < ctx; ++r)
        tq_scores[r * cfg.num_heads_q + h] /= se;
    }

    std::vector<float> tq_out(q_width, 0);
    nntrainer::compute_vcache_group2bit(
      t, tq_scores.data(), tq_vc.data(), tq_vp.data(), tq_out.data(),
      cfg.num_heads_kv, gqa_size, cfg.head_dim);

    // Compare
    float max_diff = 0, dot_ab = 0, dot_aa = 0, dot_bb = 0;
    for (int i = 0; i < q_width; ++i) {
      float diff = std::fabs(fp32_out[i] - tq_out[i]);
      max_diff = std::max(max_diff, diff);
      dot_ab += fp32_out[i] * tq_out[i];
      dot_aa += fp32_out[i] * fp32_out[i];
      dot_bb += tq_out[i] * tq_out[i];
    }
    float cosine = (dot_aa > 0 && dot_bb > 0)
                     ? dot_ab / (std::sqrt(dot_aa) * std::sqrt(dot_bb)) : 0;
    total_cosine += cosine;
    worst_diff = std::max(worst_diff, max_diff);
    total_steps++;

    int step = t - cfg.prefill_len;
    if (step < 5 || step == cfg.decode_len - 1)
      printf("  %4d  %4d  %.6f  %.6f\n", step, ctx, max_diff, cosine);
    else if (step == 5)
      printf("  ...   ...\n");
  }

  float avg_cos = total_cosine / total_steps;
  printf("\n  avg_cosine = %.6f\n  worst_diff = %.6f\n\n", avg_cos, worst_diff);
}

/**
 * @brief Hybrid cache attention: recent exact buffer + compressed history.
 *        Paper's actual design: newest buffer_size tokens in FP16,
 *        older tokens in Key 3-bit + Value 2-bit.
 */
void run_hybrid_cache_comparison(const Qwen3Config &cfg, int buffer_size) {
  int gqa_size = cfg.num_heads_q / cfg.num_heads_kv;
  int kv_width = cfg.num_heads_kv * cfg.head_dim;
  int q_width = cfg.num_heads_q * cfg.head_dim;
  int max_seq = cfg.prefill_len + cfg.decode_len;
  int key_packed_row = kv_width / 2;
  int val_packed_row = kv_width / 4; // 2-bit: 4 per byte
  int val_groups = (kv_width + 32 - 1) / 32;
  int val_params_row = val_groups * 2;

  std::mt19937 gen(42);
  std::normal_distribution<float> weight_dist(0.0f, 0.02f);
  std::normal_distribution<float> input_dist(0.0f, 0.1f);

  std::vector<float> W_q(cfg.hidden_size * q_width);
  std::vector<float> W_k(cfg.hidden_size * kv_width);
  std::vector<float> W_v(cfg.hidden_size * kv_width);
  for (auto &v : W_q) v = weight_dist(gen);
  for (auto &v : W_k) v = weight_dist(gen);
  for (auto &v : W_v) v = weight_dist(gen);

  std::vector<float> rot_signs(cfg.head_dim);
  nntrainer::generate_random_signs(rot_signs.data(), cfg.head_dim, 0xDEADBEEF);

  auto matmul = [](const float *A, const float *B, float *C, int M, int N,
                   int K) {
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        float s = 0;
        for (int k = 0; k < K; ++k) s += A[i*K+k] * B[k*N+j];
        C[i*N+j] = s;
      }
  };

  auto apply_rope = [&](float *data, int width, int head_dim, int pos) {
    for (int h = 0; h < width / head_dim; ++h) {
      float *hd = data + h * head_dim;
      for (int d = 0; d < head_dim / 2; ++d) {
        float freq = 1.0f / std::pow(500000.0f, 2.0f * d / head_dim);
        float angle = pos * freq;
        float c = std::cos(angle), s = std::sin(angle);
        float x0 = hd[d], x1 = hd[d + head_dim/2];
        hd[d] = x0*c - x1*s;
        hd[d + head_dim/2] = x0*s + x1*c;
      }
    }
  };

  // FP32 full cache (reference)
  std::vector<float> fp32_kc(max_seq * kv_width, 0);
  std::vector<float> fp32_vc(max_seq * kv_width, 0);

  // Compressed history (older tokens, quantized)
  std::vector<uint8_t> comp_kc(max_seq * key_packed_row, 0);
  std::vector<float> comp_kn(max_seq * cfg.num_heads_kv, 0);
  std::vector<uint8_t> comp_vc(max_seq * val_packed_row, 0);
  std::vector<float> comp_vp(max_seq * val_params_row, 0);

  // Exact buffer (recent tokens, FP16-equivalent = FP32 here)
  // Just reuse fp32 cache for exact part, index by position

  float total_cosine = 0;
  int total_steps = 0;

  std::cout << "  Qwen3 (h=" << cfg.hidden_size << " Q=" << cfg.num_heads_q
            << " KV=" << cfg.num_heads_kv << " d=" << cfg.head_dim
            << ") buffer=" << buffer_size << "\n"
            << "  Key: 3-bit Lloyd-Max, Value: 2-bit group min-max + exact buffer\n"
            << "  step  ctx  comp  exact  max_diff  cosine\n";

  for (int t = 0; t < max_seq; ++t) {
    std::vector<float> hidden(cfg.hidden_size);
    for (auto &v : hidden) v = input_dist(gen);

    std::vector<float> q(q_width), k(kv_width), v(kv_width);
    matmul(hidden.data(), W_q.data(), q.data(), 1, q_width, cfg.hidden_size);
    matmul(hidden.data(), W_k.data(), k.data(), 1, kv_width, cfg.hidden_size);
    matmul(hidden.data(), W_v.data(), v.data(), 1, kv_width, cfg.hidden_size);

    apply_rope(q.data(), q_width, cfg.head_dim, t);
    apply_rope(k.data(), kv_width, cfg.head_dim, t);

    // Store in FP32 reference cache
    std::copy(k.begin(), k.end(), fp32_kc.begin() + t * kv_width);
    std::copy(v.begin(), v.end(), fp32_vc.begin() + t * kv_width);

    // Quantize into compressed cache (all tokens, even recent ones)
    nntrainer::quantize_kv_turboquant_v2(
      k.data(), comp_kc.data() + t * key_packed_row,
      comp_kn.data() + t * cfg.num_heads_kv,
      rot_signs.data(), cfg.head_dim, cfg.num_heads_kv);
    nntrainer::value_quantize_group2bit(
      v.data(), kv_width, comp_vc.data() + t * val_packed_row,
      comp_vp.data() + t * val_params_row);

    if (t < cfg.prefill_len) continue;

    int ctx = t + 1;
    int exact_start = std::max(0, ctx - buffer_size);
    int compressed_len = exact_start; // [0, exact_start) = compressed
    int exact_len = ctx - exact_start; // [exact_start, ctx) = exact

    // === FP32 reference ===
    std::vector<float> fp32_out(q_width, 0);
    for (int nh = 0; nh < cfg.num_heads_kv; ++nh) {
      for (int g = 0; g < gqa_size; ++g) {
        int qh = nh * gqa_size + g;
        const float *qp = q.data() + qh * cfg.head_dim;
        std::vector<float> scores(ctx);
        float scale = 1.0f / std::sqrt((float)cfg.head_dim);
        for (int r = 0; r < ctx; ++r) {
          const float *kp = fp32_kc.data() + (r*cfg.num_heads_kv+nh)*cfg.head_dim;
          float dot = 0;
          for (int d = 0; d < cfg.head_dim; ++d) dot += qp[d]*kp[d];
          scores[r] = dot * scale;
        }
        float mx = *std::max_element(scores.begin(), scores.end());
        float se = 0;
        for (auto &s : scores) { s = std::exp(s-mx); se += s; }
        for (auto &s : scores) s /= se;
        float *out = fp32_out.data() + qh * cfg.head_dim;
        for (int r = 0; r < ctx; ++r) {
          const float *vp2 = fp32_vc.data() + (r*cfg.num_heads_kv+nh)*cfg.head_dim;
          for (int d = 0; d < cfg.head_dim; ++d) out[d] += scores[r]*vp2[d];
        }
      }
    }

    // === Hybrid TQ: compressed history + exact buffer ===
    // Step 1: Compute scores for ALL positions
    std::vector<float> hybrid_scores(ctx * cfg.num_heads_q, 0);

    // Compressed portion [0, compressed_len): use quantized key
    if (compressed_len > 0) {
      nntrainer::compute_kcaches_packed4_v2(
        q.data(), comp_kc.data(), comp_kn.data(), hybrid_scores.data(),
        compressed_len, cfg.num_heads_kv, cfg.head_dim, gqa_size, 4,
        rot_signs.data());
    }

    // Exact portion [exact_start, ctx): use FP32 key (dot product directly)
    {
      float scale = 1.0f / std::sqrt((float)cfg.head_dim);
      for (int nh = 0; nh < cfg.num_heads_kv; ++nh) {
        for (int g = 0; g < gqa_size; ++g) {
          int qh = nh * gqa_size + g;
          const float *qp = q.data() + qh * cfg.head_dim;
          for (int r = exact_start; r < ctx; ++r) {
            const float *kp = fp32_kc.data() + (r*cfg.num_heads_kv+nh)*cfg.head_dim;
            float dot = 0;
            for (int d = 0; d < cfg.head_dim; ++d) dot += qp[d]*kp[d];
            hybrid_scores[r * cfg.num_heads_q + qh] = dot * scale;
          }
        }
      }
    }

    // Softmax over ALL scores
    for (int h = 0; h < cfg.num_heads_q; ++h) {
      float mx = -1e30f;
      for (int r = 0; r < ctx; ++r)
        mx = std::max(mx, hybrid_scores[r*cfg.num_heads_q+h]);
      float se = 0;
      for (int r = 0; r < ctx; ++r) {
        hybrid_scores[r*cfg.num_heads_q+h] =
          std::exp(hybrid_scores[r*cfg.num_heads_q+h] - mx);
        se += hybrid_scores[r*cfg.num_heads_q+h];
      }
      for (int r = 0; r < ctx; ++r)
        hybrid_scores[r*cfg.num_heads_q+h] /= se;
    }

    // Step 2: Value aggregation - compressed + exact
    std::vector<float> hybrid_out(q_width, 0);

    // Compressed value portion [0, compressed_len)
    if (compressed_len > 0) {
      nntrainer::compute_vcache_group2bit(
        compressed_len - 1, hybrid_scores.data(), comp_vc.data(),
        comp_vp.data(), hybrid_out.data(), cfg.num_heads_kv, gqa_size,
        cfg.head_dim);
    }

    // Exact value portion [exact_start, ctx)
    for (int nh = 0; nh < cfg.num_heads_kv; ++nh) {
      for (int g = 0; g < gqa_size; ++g) {
        int qh = nh * gqa_size + g;
        float *out = hybrid_out.data() + qh * cfg.head_dim;
        for (int r = exact_start; r < ctx; ++r) {
          float a = hybrid_scores[r * cfg.num_heads_q + qh];
          const float *vp2 = fp32_vc.data() + (r*cfg.num_heads_kv+nh)*cfg.head_dim;
          for (int d = 0; d < cfg.head_dim; ++d)
            out[d] += a * vp2[d];
        }
      }
    }

    // Compare
    float max_diff = 0, dot_ab = 0, dot_aa = 0, dot_bb = 0;
    for (int i = 0; i < q_width; ++i) {
      float diff = std::fabs(fp32_out[i] - hybrid_out[i]);
      max_diff = std::max(max_diff, diff);
      dot_ab += fp32_out[i]*hybrid_out[i];
      dot_aa += fp32_out[i]*fp32_out[i];
      dot_bb += hybrid_out[i]*hybrid_out[i];
    }
    float cosine = (dot_aa>0 && dot_bb>0) ? dot_ab/(std::sqrt(dot_aa)*std::sqrt(dot_bb)) : 0;
    total_cosine += cosine;
    total_steps++;

    int step = t - cfg.prefill_len;
    if (step < 5 || step == cfg.decode_len - 1)
      printf("  %4d  %4d  %4d  %5d  %.6f  %.6f\n",
             step, ctx, compressed_len, exact_len, max_diff, cosine);
    else if (step == 5)
      printf("  ...   ...\n");
  }

  float avg_cos = total_cosine / total_steps;
  printf("\n  avg_cosine = %.6f (buffer=%d)\n\n", avg_cos, buffer_size);
}

int main() {
  std::cout << "=== TurboQuant v2 End-to-End: Random Weight Qwen3 ===\n\n";

  // Full quantization (no buffer) - current approach
  std::cout << "--- Qwen3-0.6B: Full Quantization (Key 3-bit + Value 4-bit) ---\n";
  run_attention_comparison({896, 14, 2, 64, 28, 32, 16});

  // Hybrid cache (paper method): Key 3-bit + Value 2-bit + exact buffer
  std::cout << "--- Qwen3-0.6B: Hybrid Cache (buffer=16, Value 2-bit) ---\n";
  run_hybrid_cache_comparison({896, 14, 2, 64, 28, 32, 16}, 16);

  std::cout << "--- Qwen3-0.6B: Hybrid Cache (buffer=32, Value 2-bit) ---\n";
  run_hybrid_cache_comparison({896, 14, 2, 64, 28, 64, 16}, 32);

  std::cout << "--- Qwen3-1.7B: Hybrid Cache (buffer=32, Value 2-bit) ---\n";
  run_hybrid_cache_comparison({1536, 12, 4, 128, 28, 64, 16}, 32);

  std::cout << "--- Qwen3-4B: Hybrid Cache (buffer=32, Value 2-bit) ---\n";
  run_hybrid_cache_comparison({2560, 32, 4, 128, 40, 48, 8}, 32);

  return 0;
}

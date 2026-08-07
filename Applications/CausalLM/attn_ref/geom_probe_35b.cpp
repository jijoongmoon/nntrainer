// SPDX-License-Identifier: Apache-2.0
/**
 * @file    geom_probe_35b.cpp
 * @brief   Do the existing CUDA device arms actually BIND at the real
 *          Qwen3.6-35B geometry? (build-plan item 2)
 *
 * Both arms decline silently by design -- the caller just falls through to a
 * host path. On this hardware a silent host fall-through is not merely slow:
 * inside a CUDA-graph capture it is a wrong-answer bug (the host op reads
 * buffers whose producing kernels have not run, and is never recorded). So the
 * graph shape for the 35B must be decided knowing which of these bind:
 *
 *   (i)  cuda_attention_interleaved_fp16 at head_dim=256, nH=16, nKV=2.
 *        gemma4 exercises head_dim 256, but at nH=8/nKV=1; GQA 8:1 at d=256 is
 *        new. If it declines, the full-attention mixer needs a different node
 *        decomposition.
 *   (ii) cuda_fc_dense_gemm_fp16 at the FP16-dense GDN projection shapes the
 *        manifest actually contains. N=1 (shared_gate_lin) and N=32 (in_proj_b,
 *        in_proj_a) are degenerate cuBLAS GEMV shapes nothing in-tree has run.
 *        If they decline, those FCs must fold into the GDN kernels instead of
 *        living as graph nodes.
 *
 * Correctness is checked too, not just "returned true": the attention probe
 * diffs the device output against a host reference, because an arm that binds
 * and computes the wrong thing is worse than one that declines.
 *
 * Standalone: links libnntrainer.so, no graph, no model, no meson change.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include <cuda_attention.h>
#include <cuda_context_manager.h>
#include <cuda_fc_dense.h>
#include <cuda_stream_manager.h>

namespace {

int failures = 0;

/** @brief fp16 bit-pattern helpers (avoid depending on _FP16 being enabled) */
unsigned short f2h(float f) {
  unsigned int x;
  std::memcpy(&x, &f, 4);
  const unsigned int sign = (x >> 16) & 0x8000u;
  int exp = (int)((x >> 23) & 0xFF) - 127 + 15;
  unsigned int man = x & 0x7FFFFFu;
  if (exp <= 0)
    return (unsigned short)sign;
  if (exp >= 31)
    return (unsigned short)(sign | 0x7C00u);
  return (unsigned short)(sign | ((unsigned)exp << 10) | (man >> 13));
}
float h2f(unsigned short h) {
  const unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
  const int exp = (h >> 10) & 0x1F;
  const unsigned int man = h & 0x3FFu;
  unsigned int x;
  if (exp == 0)
    x = sign;
  else if (exp == 31)
    x = sign | 0x7F800000u | (man << 13);
  else
    x = sign | ((unsigned)(exp - 15 + 127) << 23) | (man << 13);
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

/** @brief allocate host-mapped like the runtime pools do on integrated */
void *alloc_pool(size_t bytes) {
  void *p = nullptr;
  if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped) != cudaSuccess)
    return nullptr;
  return p;
}

void report(const char *what, bool ok, const char *extra = "") {
  std::printf("%s %-58s %s\n", ok ? "[PASS]" : "[FAIL]", what, extra);
  if (!ok)
    ++failures;
}

/* ---------------------------------------------------------------- (ii) FC */

void probe_dense_fc() {
  // The FP16-dense projections the 35B manifest actually carries, as [K,N].
  // Every GDN layer runs all of these at M=1 per decode token.
  struct Shape {
    unsigned K, N;
    const char *name;
  };
  const Shape shapes[] = {
    {2048, 8192, "GDN in_proj_qkv  K2048 N8192"},
    {2048, 4096, "GDN in_proj_z    K2048 N4096"},
    {2048, 32, "GDN in_proj_b/a  K2048 N32   (degenerate N)"},
    {4096, 2048, "GDN out_proj     K4096 N2048"},
    {2048, 1, "shared_gate_lin  K2048 N1    (degenerate N)"},
  };
  const unsigned M = 1; // decode

  for (const auto &s : shapes) {
    const size_t xb = (size_t)M * s.K * 2;
    const size_t wb = (size_t)s.K * s.N * 2;
    const size_t yb = (size_t)M * s.N * 2;
    auto *X = (unsigned short *)alloc_pool(xb);
    auto *W = (unsigned short *)alloc_pool(wb);
    auto *Y = (unsigned short *)alloc_pool(yb);
    if (!X || !W || !Y) {
      report(s.name, false, "(host-mapped alloc failed)");
      continue;
    }
    for (unsigned i = 0; i < M * s.K; ++i)
      X[i] = f2h(((i * 37) % 13) * 0.01f - 0.06f);
    for (size_t i = 0; i < (size_t)s.K * s.N; ++i)
      W[i] = f2h(((i * 17) % 11) * 0.01f - 0.05f);
    std::memset(Y, 0, yb);

    const bool ok = nntrainer::cuda::cuda_fc_dense_gemm_fp16(X, W, Y, M, s.N,
                                                             s.K);
    nntrainer::cuda::StreamManager::Global().finish();

    // Only trust "true" if the numbers are right: check one output column
    // against a host dot product.
    bool numeric = ok;
    if (ok) {
      const unsigned n = 0;
      float ref = 0.0f;
      for (unsigned k = 0; k < s.K; ++k)
        ref += h2f(X[k]) * h2f(W[(size_t)k * s.N + n]);
      const float got = h2f(Y[n]);
      const float tol = 1e-2f * (std::fabs(ref) + 1.0f);
      numeric = std::fabs(got - ref) <= tol;
      char buf[128];
      std::snprintf(buf, sizeof(buf), "(y[0] got %.4f ref %.4f)", got, ref);
      report(s.name, numeric, buf);
    } else {
      report(s.name, false, "(declined -> caller would host-dot)");
    }
    cudaFreeHost(X);
    cudaFreeHost(W);
    cudaFreeHost(Y);
  }
}

/* --------------------------------------------------------- (i) attention */

void probe_attention() {
  // 35B full-attention geometry.
  const int HQ = 16, HKV = 2, D = 256;
  const int N_q = 1; // decode step
  const int cases[] = {64, 512, 2048};

  for (int N_kv : cases) {
    const size_t qb = (size_t)N_q * HQ * D * 2;
    const size_t kb = (size_t)N_kv * HKV * D * 2;
    const size_t ob = (size_t)N_q * HQ * D * 2;
    auto *Q = (unsigned short *)alloc_pool(qb);
    auto *K = (unsigned short *)alloc_pool(kb);
    auto *V = (unsigned short *)alloc_pool(kb);
    auto *O = (unsigned short *)alloc_pool(ob);
    if (!Q || !K || !V || !O) {
      report("attention alloc", false);
      continue;
    }
    for (size_t i = 0; i < (size_t)N_q * HQ * D; ++i)
      Q[i] = f2h(((i * 29) % 7) * 0.05f - 0.15f);
    for (size_t i = 0; i < (size_t)N_kv * HKV * D; ++i) {
      K[i] = f2h(((i * 13) % 9) * 0.04f - 0.16f);
      V[i] = f2h(((i * 23) % 5) * 0.06f - 0.12f);
    }
    std::memset(O, 0, ob);

    // Call convention copied from the real caller (mha_core.cpp:3092):
    // window is INT_MAX when the layer is NOT sliding-window (0 would mask
    // every key and yield zeros), and a decode step's query sits at the LAST
    // cache position, so cache_from = N_kv - 1.
    const int cache_from = N_kv - N_q;
    const bool ok = nntrainer::cuda::cuda_attention_interleaved_fp16(
      Q, K, V, O, HQ, HKV, N_q, N_kv, cache_from, D, /*window=*/INT_MAX,
      /*softcap=*/0.0f, /*ring_cap=*/0);
    nntrainer::cuda::StreamManager::Global().finish();

    char name[96];
    std::snprintf(name, sizeof(name), "attn d=%d HQ=%d HKV=%d N_kv=%-4d", D, HQ,
                  HKV, N_kv);
    if (!ok) {
      report(name, false, "(declined -> mha_core would host-path)");
      cudaFreeHost(Q); cudaFreeHost(K); cudaFreeHost(V); cudaFreeHost(O);
      continue;
    }

    // Host reference for head 0 (GQA: q head h reads kv head h / (HQ/HKV)).
    const int gqa = HQ / HKV;
    std::vector<float> s(N_kv);
    const float scale = 1.0f / std::sqrt((float)D);
    const int kvh = 0 / gqa;
    float m = -1e30f;
    for (int t = 0; t < N_kv; ++t) {
      float acc = 0.0f;
      for (int d = 0; d < D; ++d)
        acc += h2f(Q[d]) * h2f(K[((size_t)t * HKV + kvh) * D + d]);
      s[t] = acc * scale;
      m = std::max(m, s[t]);
    }
    float sum = 0.0f;
    for (int t = 0; t < N_kv; ++t) {
      s[t] = std::exp(s[t] - m);
      sum += s[t];
    }
    float ref0 = 0.0f;
    for (int t = 0; t < N_kv; ++t)
      ref0 += (s[t] / sum) * h2f(V[((size_t)t * HKV + kvh) * D + 0]);

    const float got0 = h2f(O[0]);
    bool all_zero = true;
    for (size_t i = 0; i < (size_t)N_q * HQ * D && all_zero; ++i)
      if (O[i] != 0)
        all_zero = false;
    // Relative tolerance. An absolute one would let an all-zero output "pass"
    // whenever the reference happens to be small -- which is precisely the
    // silent-wrong-answer case this probe exists to catch.
    const bool numeric =
      !all_zero && std::fabs(got0 - ref0) <= 5e-2f * std::fabs(ref0);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "(o[0] got %.6f ref %.6f%s)", got0, ref0,
                  all_zero ? "  ALL-ZERO OUTPUT" : "");
    report(name, numeric, buf);

    cudaFreeHost(Q); cudaFreeHost(K); cudaFreeHost(V); cudaFreeHost(O);
  }
}

} // namespace

int main() {
  std::printf("=== 35B geometry probe: do the existing device arms bind? ===\n");
  std::printf("integrated=%d\n",
              (int)nntrainer::cuda::ContextManager::Global().isIntegrated());

  std::printf("\n-- (ii) cuda_fc_dense_gemm_fp16, M=1, GDN FP16-dense shapes --\n");
  probe_dense_fc();

  std::printf("\n-- (i) cuda_attention_interleaved_fp16, 35B full-attn geometry --\n");
  probe_attention();

  std::printf("\n%s (%d failed)\n",
              failures ? "=== SOME ARMS DECLINE ===" : "=== ALL ARMS BIND ===",
              failures);
  return failures ? 1 : 0;
}

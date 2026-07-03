// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_mm.cpp
 * @date   18 Jun 2026
 * @brief  HMX matrix-multiply operations backed by HexKL (Phase 1).
 */

#ifdef ENABLE_HEXKL
#ifdef ENABLE_FP16

#include <hexkl_mm.h>
#include <htp_backend.h>

#include <remote.h>
#include <sdkl.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
struct WHEntry {
  void *npu;
  unsigned int N, K;
  size_t bytes;
};

// Weight cache with a hard cap on total NPU bytes to avoid DMA exhaustion.
// When the cap is exceeded, the oldest entry (by insertion order) is evicted.
static constexpr size_t WH_CACHE_MAX_BYTES = 64 * 1024 * 1024; // 64 MB

struct WHCache {
  std::map<const void *, WHEntry> cache;
  // Insertion-order tracking for FIFO eviction.
  std::vector<const void *> order;
  size_t total_bytes = 0;
  std::mutex mtx;

  void evict_one() {
    if (order.empty())
      return;
    const void *oldest = order.front();
    order.erase(order.begin());
    auto it = cache.find(oldest);
    if (it != cache.end()) {
      total_bytes -= it->second.bytes;
      sdkl_npu_free(it->second.npu);
      cache.erase(it);
    }
  }

  ~WHCache() {
    for (auto &e : cache)
      sdkl_npu_free(e.second.npu);
  }
} g_wh_cache;
} // namespace

namespace nntrainer {
namespace hmx {

void shgemm_f32f16_f32(const unsigned int TStorageOrder, bool TransA,
                       bool TransB, const unsigned int M, const unsigned int N,
                       const unsigned int K, const float alpha, const float *A,
                       const unsigned int lda, const _FP16 *B,
                       const unsigned int ldb, const float beta, float *C,
                       const unsigned int ldc) {
  // Constraints (§2, sdkl.h 6.4.0.1):
  //   (1) 2D only — calculateFlattenDot already flattened 4D to M/N/K
  //   (2) sdkl_npu_mm_f32f16_f32: result = X * W^T
  //       where X is FP32 input [M×K], W is FP16 weight [N×K] in WH layout,
  //       and the FP32 output is A [M×N].
  //   (3) B is [N, K] row-major — must convert RM→WH before NPU call.
  //   (4) All buffers must be allocated with sdkl_npu_alloc() for alignment.
  //
  // Naming: the sdkl API uses (A=output, X=input, W=weight).
  //   Our nntrainer parameters use (A=input, B=weight, C=output).
  //   Internally we call: sdkl_npu_mm_f32f16_f32(domain,
  //     n_row=M, n_col=N, n_inner=K,
  //     A_npu /*out*/, X_npu /*A input*/, W_npu /*B weight WH*/)
  //
  // alpha/beta: nntrainer always passes alpha=1.0, beta=0.0 for
  //   dotFloat32Float16. General alpha/beta are supported below.
  //
  // TransA/TransB/TStorageOrder: forwarded from HtpComputeOps::shgemm which
  //   mirrors CpuComputeOps. sdkl_npu_mm_f32f16_f32 computes input × weight^T
  //   natively, so TransB=true is the expected call pattern (weight [N,K]).

  int domain = HtpBackend::global().domain();

  // HMX F16 tile alignment constraints (sdkl 6.4, verified on V79/V81):
  //   N (n_col) must be a multiple of 32 — this is a hard weight-layout
  //   requirement that callers must satisfy.
  //   M (n_row) is padded internally: we round Mp = ceil(M/32)*32, allocate
  //   Mp rows for X and A buffers, and only copy back the real M output rows.
  if (N % 32 != 0)
    throw std::runtime_error(
      "shgemm_f32f16_f32: N must be a multiple of 32 (N=" + std::to_string(N) +
      ")");

  // Round M up to the next multiple of 32 for NPU tile alignment.
  const unsigned int Mp = (M + 31u) & ~31u;

  // --- Allocate NPU-accessible staging buffers ---
  // sdkl_npu_alloc(size_t size, void** buffer_ptr): returns 0 on success.
  // sdkl_npu_free(void* addr): no domain parameter.
  void *X_npu = nullptr; // FP32 input (nntrainer's A), Mp*K rows
  void *A_npu = nullptr; // FP32 output (nntrainer's C), Mp*N rows

  auto cleanup = [&]() {
    if (X_npu)
      sdkl_npu_free(X_npu);
    if (A_npu)
      sdkl_npu_free(A_npu);
  };

  int err = 0;

  err = sdkl_npu_alloc(Mp * K * sizeof(float), &X_npu);
  if (err != 0) {
    cleanup();
    throw std::runtime_error("sdkl_npu_alloc X failed: " + std::to_string(err));
  }

  err = sdkl_npu_alloc(Mp * N * sizeof(float), &A_npu);
  if (err != 0) {
    cleanup();
    throw std::runtime_error("sdkl_npu_alloc A failed: " + std::to_string(err));
  }

  // --- Copy input into NPU memory ---
  // X: zero-fill the full Mp*K buffer, then copy the real M rows.
  std::memset(X_npu, 0, Mp * K * sizeof(float));
  std::memcpy(X_npu, A, M * K * sizeof(float));

  // M>1 (prefill): bypass WHCache — weight accessed once per forward pass.
  // Alloc W_npu transiently to keep peak NPU DMA below the ~64 MB budget.
  // WHCache below is kept for potential future decode-NPU scenarios.
  if (M > 1) {
    const size_t w_bytes = N * K * sizeof(_FP16);
    void *W_npu = nullptr;
    err = sdkl_npu_alloc(w_bytes, &W_npu);
    if (err != 0 || W_npu == nullptr) {
      cleanup();
      throw std::runtime_error(
        "shgemm_f32f16_f32: prefill weight NPU alloc failed (err=" +
        std::to_string(err) + ")");
    }
    std::memcpy(W_npu, B, w_bytes);
    int werr =
      sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K, (_Float16 *)W_npu);
    if (werr != 0) {
      sdkl_npu_free(W_npu);
      cleanup();
      throw std::runtime_error("rm_to_wh_f16_inplace failed (prefill): " +
                               std::to_string(werr));
    }
    if (beta != 0.0f) {
      sdkl_npu_free(W_npu);
      cleanup();
      throw std::runtime_error(
        "shgemm_f32f16_f32: beta != 0 not supported (Phase 1)");
    }
    std::memset(A_npu, 0, Mp * N * sizeof(float));
    err = sdkl_npu_mm_f32f16_f32(
      domain, (int)Mp, (int)N, (int)K, static_cast<float *>(A_npu),
      static_cast<const float *>(X_npu), static_cast<const _Float16 *>(W_npu));
    sdkl_npu_free(W_npu);
    if (err != 0) {
      cleanup();
      throw std::runtime_error("sdkl_npu_mm_f32f16_f32 failed (prefill): " +
                               std::to_string(err));
    }
    const float *A_npu_f32 = static_cast<const float *>(A_npu);
    if (alpha == 1.0f) {
      std::memcpy(C, A_npu_f32, M * N * sizeof(float));
    } else {
      for (unsigned int i = 0; i < M * N; ++i)
        C[i] = alpha * A_npu_f32[i];
    }
    cleanup();
    return;
  }

  // --- Weight cache: convert RM->WH once per unique weight pointer ---
  // FSU is off for engine=htp (transformer FSU guard), so B pointer is stable.
  // Cache is size-limited (WH_CACHE_MAX_BYTES) to avoid NPU DMA exhaustion;
  // FIFO eviction removes the oldest entry when the limit is exceeded.
  void *W_npu_cached = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_wh_cache.mtx);
    auto it = g_wh_cache.cache.find(B);
    if (it == g_wh_cache.cache.end() || it->second.N != N ||
        it->second.K != K) {
      // Cache miss: evict stale entry for same B pointer if present.
      if (it != g_wh_cache.cache.end()) {
        g_wh_cache.total_bytes -= it->second.bytes;
        sdkl_npu_free(it->second.npu);
        auto oit =
          std::find(g_wh_cache.order.begin(), g_wh_cache.order.end(), B);
        if (oit != g_wh_cache.order.end())
          g_wh_cache.order.erase(oit);
        g_wh_cache.cache.erase(it);
      }
      // Evict oldest entries until we are below the cap.
      const size_t new_bytes = N * K * sizeof(_FP16);
      while (g_wh_cache.total_bytes + new_bytes > WH_CACHE_MAX_BYTES &&
             !g_wh_cache.order.empty())
        g_wh_cache.evict_one();

      void *new_npu = nullptr;
      err = sdkl_npu_alloc(new_bytes, &new_npu);
      if (err != 0 || new_npu == nullptr) {
        cleanup();
        throw std::runtime_error(
          "shgemm_f32f16_f32: weight NPU alloc failed (err=" +
          std::to_string(err) + ")");
      }
      std::memcpy(new_npu, B, new_bytes);
      int werr = sdkl_cpu_rm_to_wh_f16_inplace((size_t)N, (size_t)K,
                                               (_Float16 *)new_npu);
      if (werr != 0) {
        sdkl_npu_free(new_npu);
        cleanup();
        throw std::runtime_error("rm_to_wh_f16_inplace failed: " +
                                 std::to_string(werr));
      }
      g_wh_cache.cache[B] = {new_npu, N, K, new_bytes};
      g_wh_cache.order.push_back(B);
      g_wh_cache.total_bytes += new_bytes;
      W_npu_cached = new_npu;
    } else {
      W_npu_cached = it->second.npu;
    }
  }
  const _Float16 *W_wh = static_cast<const _Float16 *>(W_npu_cached);

  // sdkl_npu_mm_f32f16_f32 is write-only (not accumulate); beta != 0 is
  // unsupported in Phase 1. Phase 2: support accumulation if the SDK adds an
  // accumulate mode.
  if (beta != 0.0f) {
    cleanup();
    throw std::runtime_error("shgemm_f32f16_f32: beta != 0 is not supported in "
                             "Phase 1 (NPU write-only)");
  }
  std::memset(A_npu, 0, Mp * N * sizeof(float));

  // --- NPU matrix multiply ---
  // sdkl_npu_mm_f32f16_f32(int domain,
  //                         int n_row, int n_col, int n_inner,
  //                         float* A   /*output FP32*/,
  //                         const float* X  /*input FP32*/,
  //                         const _Float16* W /*weight FP16 WH*/)
  // Note: no lda/ldb/ldc — contiguous row-major assumed.
  // Use Mp (padded row count) so the NPU sees a 32-aligned M.
  err = sdkl_npu_mm_f32f16_f32(domain, (int)Mp, (int)N, (int)K,
                               static_cast<float *>(A_npu),
                               static_cast<const float *>(X_npu), W_wh);
  if (err != 0) {
    cleanup();
    throw std::runtime_error("sdkl_npu_mm_f32f16_f32 failed: " +
                             std::to_string(err));
  }

  // --- Copy result back and apply alpha ---
  // Only copy M real output rows (not the Mp padded rows).
  const float *A_npu_f32 = static_cast<const float *>(A_npu);
  if (alpha == 1.0f) {
    std::memcpy(C, A_npu_f32, M * N * sizeof(float));
  } else {
    for (unsigned int i = 0; i < M * N; ++i)
      C[i] = alpha * A_npu_f32[i];
  }

  cleanup();
}

} // namespace hmx
} // namespace nntrainer

#endif // ENABLE_FP16

// The includes below are repeated outside ENABLE_FP16 so that
// shgemm_u8i8_i32 can be compiled even when ENABLE_FP16 is not set.
// C++ system headers are idempotent; project headers carry their own guards.
#include <hexkl_mm.h>
#include <htp_backend.h>

#include <sdkl.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace nntrainer {
namespace hmx {

namespace {

size_t checkedMulSizeT(size_t lhs, size_t rhs, const char *label) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::runtime_error(
      std::string("shgemm_u8i8_i32: overflow computing ") + label);
  }

  return lhs * rhs;
}

size_t checkedBytes(size_t count, size_t elem_size, const char *label) {
  return checkedMulSizeT(count, elem_size, label);
}

int checkedSdkDim(unsigned int dim, const char *label) {
  if (dim > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string("shgemm_u8i8_i32: ") + label +
                             " exceeds SDK int range");
  }

  return static_cast<int>(dim);
}

} // namespace

// U8 activations × I8 weights (WH-layout) → I32 accumulator → FP32 output.
//
// Quantization scheme (per-tensor activation, per-channel weight):
//   act_scale = max_abs(A) / 127.0f
//   X_u8[i]  = clamp(round(A[i] / act_scale) + 128, 0, 255)   (zp=128)
//   sdkl call: C_i32 = X_u8 * B_wh^T   (accumulator, no bias)
//   C[m,n]   = act_scale * wt_scale[n] * (float(C_i32[m,n]) -
//   float(zp_corr[n]))
//
// Low-level NPU entry point. B_wh must already be in WH layout.
// SDKL requires M%64==0, so pad activation/output rows internally and copy
// only the real M rows back to the caller.
void shgemm_u8i8_i32(unsigned int M, unsigned int N, unsigned int K,
                     const float *A, const int8_t *B_wh, const float *wt_scale,
                     const int32_t *zp_corr, float *C) {
  if (N % 32 != 0)
    throw std::runtime_error("shgemm_u8i8_i32: N must be a multiple of 32 (N=" +
                             std::to_string(N) + ")");

  const size_t M_sz = M;
  const size_t Mp_sz = ((M_sz + 63U) / 64U) * 64U;
  const size_t N_sz = N;
  const size_t K_sz = K;
  const size_t mk_elems = checkedMulSizeT(M_sz, K_sz, "M*K elements");
  const size_t mpk_elems = checkedMulSizeT(Mp_sz, K_sz, "Mp*K elements");
  const size_t mpn_elems = checkedMulSizeT(Mp_sz, N_sz, "Mp*N elements");
  const size_t nk_elems = checkedMulSizeT(N_sz, K_sz, "N*K elements");
  const size_t x_bytes =
    checkedBytes(mpk_elems, sizeof(uint8_t), "X_npu bytes");
  const size_t c_bytes =
    checkedBytes(mpn_elems, sizeof(int32_t), "C_npu bytes");
  const size_t w_bytes = checkedBytes(nk_elems, sizeof(int8_t), "W_npu bytes");
  const int sdk_M = checkedSdkDim(static_cast<unsigned int>(Mp_sz), "Mp");
  const int sdk_N = checkedSdkDim(N, "N");
  const int sdk_K = checkedSdkDim(K, "K");

  int domain = HtpBackend::global().domain();

  // --- Compute per-tensor activation scale ---
  float max_abs = 0.0f;
  for (size_t i = 0; i < mk_elems; ++i) {
    if (!std::isfinite(A[i]))
      continue;

    float v = std::fabs(A[i]);
    if (v > max_abs)
      max_abs = v;
  }
  // Guard against zero, non-finite, and underflowing scale values so the
  // quantization path stays finite and deterministic.
  constexpr float kActQuantMax = 127.0f;
  constexpr float kActZeroPoint = 128.0f;
  const double scale_candidate =
    static_cast<double>(max_abs) / static_cast<double>(kActQuantMax);
  const float act_scale =
    (std::isfinite(scale_candidate) &&
     scale_candidate >= static_cast<double>(std::numeric_limits<float>::min()))
      ? static_cast<float>(scale_candidate)
      : 1.0f;
  const float inv_act_scale = 1.0f / act_scale;

  // --- Allocate NPU-accessible staging buffers ---
  void *X_npu = nullptr; // uint8  [M * K]
  void *C_npu = nullptr; // int32  [M * N]
  void *W_npu = nullptr; // int8   [N * K] (WH layout, copied as-is)

  auto cleanup = [&]() {
    if (X_npu)
      sdkl_npu_free(X_npu);
    if (C_npu)
      sdkl_npu_free(C_npu);
    if (W_npu)
      sdkl_npu_free(W_npu);
  };

  int err = 0;

  err = sdkl_npu_alloc(x_bytes, &X_npu);
  if (err != 0 || X_npu == nullptr) {
    cleanup();
    throw std::runtime_error("shgemm_u8i8_i32: sdkl_npu_alloc X_npu failed: " +
                             std::to_string(err));
  }

  err = sdkl_npu_alloc(c_bytes, &C_npu);
  if (err != 0 || C_npu == nullptr) {
    cleanup();
    throw std::runtime_error("shgemm_u8i8_i32: sdkl_npu_alloc C_npu failed: " +
                             std::to_string(err));
  }

  err = sdkl_npu_alloc(w_bytes, &W_npu);
  if (err != 0 || W_npu == nullptr) {
    cleanup();
    throw std::runtime_error("shgemm_u8i8_i32: sdkl_npu_alloc W_npu failed: " +
                             std::to_string(err));
  }

  // --- FP32 → U8 quantization (zp = 128) ---
  uint8_t *X_u8 = static_cast<uint8_t *>(X_npu);
  for (size_t m = 0; m < Mp_sz; ++m) {
    const size_t row_offset = m * K_sz;
    if (m >= M_sz) {
      std::memset(X_u8 + row_offset, static_cast<int>(kActZeroPoint), K_sz);
      continue;
    }

    for (size_t k = 0; k < K_sz; ++k) {
      const float a = A[row_offset + k];
      if (!std::isfinite(a)) {
        X_u8[row_offset + k] = static_cast<uint8_t>(kActZeroPoint);
        continue;
      }

      float q = std::round(a * inv_act_scale) + kActZeroPoint;
      if (q < 0.0f)
        q = 0.0f;
      if (q > 255.0f)
        q = 255.0f;
      X_u8[row_offset + k] = static_cast<uint8_t>(q);
    }
  }

  // --- Copy WH-layout weight directly (no layout conversion needed) ---
  std::memcpy(W_npu, B_wh, w_bytes);

  // --- NPU matrix multiply ---
  // sdkl_npu_mm_u8i8_i32(domain, n_row, n_col, n_inner, A/*i32 out*/,
  //                       X/*u8 in*/, W/*i8 WH weight*/)
  std::memset(C_npu, 0, c_bytes);
  err = sdkl_npu_mm_u8i8_i32(
    domain, sdk_M, sdk_N, sdk_K, static_cast<int32_t *>(C_npu),
    static_cast<const uint8_t *>(X_npu), static_cast<const int8_t *>(W_npu));
  if (err != 0) {
    cleanup();
    throw std::runtime_error("sdkl_npu_mm_u8i8_i32 failed: " +
                             std::to_string(err));
  }

  // --- Dequantize: I32 accumulator → FP32 ---
  // C[m,n] = act_scale * wt_scale[n] * (C_i32[m,n] - zp_corr[n])
  const int32_t *C_i32 = static_cast<const int32_t *>(C_npu);
  for (size_t m = 0; m < M_sz; ++m) {
    const size_t row_offset = m * N_sz;
    for (size_t n = 0; n < N_sz; ++n) {
      C[row_offset + n] = act_scale * wt_scale[n] *
                          (static_cast<float>(C_i32[row_offset + n]) -
                           static_cast<float>(zp_corr[n]));
    }
  }

  cleanup();
}

} // namespace hmx
} // namespace nntrainer

#endif // ENABLE_HEXKL

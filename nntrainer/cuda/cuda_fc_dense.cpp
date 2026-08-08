// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_fc_dense.cpp
 * @date    06 Aug 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Dense FC GEMM implementation (see cuda_fc_dense.h).
 */

#include "cuda_fc_dense.h"

#include "cuda_blas_manager.h"
#include "cuda_context_manager.h"
#include "cuda_stream_manager.h"

#include <nntrainer_log.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

#include <cuda_runtime.h>

namespace nntrainer::cuda {

namespace {

/** NNTR_FC_CUDA_DENSE=0 opts the whole dense device arm out (the caller then
 *  falls back exactly as it did before this file existed). Read once. */
bool dense_on() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_FC_CUDA_DENSE");
    return !(e && e[0] == '0');
  }();
  return on;
}

/** NNTR_FC_CUDA_DENSE_DBG=1 traces the first few calls (which shapes actually
 *  took the device arm) -- the fall-through is otherwise silent by design. */
void trace(const char *kind, unsigned M, unsigned N, unsigned K, bool ok) {
  static const bool dbg = []() {
    const char *e = std::getenv("NNTR_FC_CUDA_DENSE_DBG");
    return e && e[0] == '1';
  }();
  if (!dbg)
    return;
  static int n = 0;
  if (n++ < 16)
    std::fprintf(stderr, "[CUDA-FC-DENSE] %s M=%u N=%u K=%u -> %s\n", kind, M,
                 N, K, ok ? "ok" : "FAIL");
}

/**
 * @brief The one orientation note for both entry points.
 *
 * cuBLAS is column-major. A row-major C[M,N] = A[M,K]*B[K,N] is the SAME bytes
 * as the column-major C'[N,M] = B'[N,K] * A'[K,M], where B' is the row-major B
 * reinterpreted column-major (ld = N) and A' the row-major A (ld = K). So one
 * OP_N/OP_N call with the operands swapped and (m,n) = (N,M) computes it with
 * no transpose and no copy. Identical to the int8 path in
 * BlasManager::igemmRowMajor(), kept spelled out here because getting it wrong
 * is silent (a transposed result is still a plausible-looking tensor).
 */
bool gemm_ex(int M, int N, int K, const void *A, cudaDataType a_type,
             const void *B, void *C, cudaDataType c_type,
             cublasComputeType_t compute) {
  cublasHandle_t h = BlasManager::Global().handle();
  if (h == nullptr)
    return false;
  const float alpha = 1.0f, beta = 0.0f;
  // The handle is bound to the backend stream in BlasManager::initialize(), so
  // this orders with the dequant / copy kernels around it without an extra
  // sync. Re-bind anyway: a caller that switched streams since would otherwise
  // silently race, and the call is a few nanoseconds.
  cublasSetStream(h, StreamManager::Global().GetStream());

  // M-CHUNKING. cublasGemmEx FAILS (status 13, EXECUTION_FAILED, followed by a
  // sticky illegal memory access that kills the context) on Orin/sm_87 for tall
  // GEMMs at a degenerate N -- measured at M=2774 N=32 K=2048, which is the
  // GDN in_proj_b / in_proj_a projection during a 4096-token prefill chunk. The
  // same shape at M<=2048 is fine, and the geometry probe binds it at every
  // shape, so this is an algo-selection failure at large M, not a caller error.
  // It is the same class of sm_87 defect the int8 path already works around by
  // chunking K (cuda_blas_manager.cpp), and the fix is the same shape: keep
  // every call inside the regime that works.
  //
  // Row-slicing is free here, unlike the int8 K-chunk: A is [M,K] row-major and
  // C is [M,N] row-major, so a row block of either is CONTIGUOUS. No repack, no
  // accumulation -- each chunk writes its own disjoint rows of C with beta=0.
  static const int mchunk = []() {
    const char *e = std::getenv("NNTR_CUBLAS_MCHUNK");
    if (e) {
      int v = std::atoi(e);
      return v > 0 ? v : (1 << 28);
    }
    // Discrete GPUs run the full-M call correctly; only integrated needs this.
    return ContextManager::Global().isIntegrated() ? 2048 : (1 << 28);
  }();
  const size_t a_elem = (a_type == CUDA_R_16F) ? 2u : 4u;
  const size_t c_elem = (c_type == CUDA_R_16F) ? 2u : 4u;
  for (int m0 = 0; m0 < M; m0 += mchunk) {
    const int mc = (M - m0 < mchunk) ? (M - m0) : mchunk;
    const void *Ai =
      static_cast<const char *>(A) + (size_t)m0 * (size_t)K * a_elem;
    void *Ci = static_cast<char *>(C) + (size_t)m0 * (size_t)N * c_elem;
    cublasStatus_t s = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, mc, K,
                                    &alpha, B, a_type, N, Ai, a_type, K, &beta,
                                    Ci, c_type, N, compute, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) {
      ml_loge("[CUDA] dense cublasGemmEx failed: %d (M=%d[%d..+%d] N=%d K=%d)",
              (int)s, M, m0, mc, N, K);
      return false;
    }
  }
  // Drain, exactly as every other device op does (cuda_fc_qint4.cpp:1601 and
  // friends). The stream binding above only orders this GEMM against other
  // DEVICE work; it says nothing about the HOST. On integrated hardware
  // cuda_async_mode() hard-returns false, so maybeFinish() is a full finish()
  // and that per-op drain is the entire mechanism this path relies on for
  // host/device ordering -- the caller in CudaComputeOps::fc returns straight
  // into the next op, which is frequently host code reading this UVM output.
  //
  // Omitting it was a live race, not a theoretical one: the real graph came out
  // WRONG and DIFFERENT on every run (max|d| 5-11 against a 3.8e-4 CPU
  // reference, argmax wandering) and went bit-identical the moment this arm was
  // disabled. It is invisible to the host-op detectors -- the work IS on the
  // device, it is just read too early.
  StreamManager::Global().maybeFinish();
  return true;
}

} // namespace

bool cuda_fc_dense_gemm_fp16(const void *Xh, const void *Wh, void *Yh,
                             unsigned int M, unsigned int N, unsigned int K) {
  if (!dense_on() || Xh == nullptr || Wh == nullptr || Yh == nullptr || M == 0 ||
      N == 0 || K == 0)
    return false;
  // fp16 in / fp16 out with an FP32 ACCUMULATE (CUBLAS_COMPUTE_32F). A
  // 16F accumulate would round every partial sum at 11 bits of mantissa,
  // which over K in the thousands is a visible drift from the host dot()
  // this arm replaces; the 32F accumulate is the closer match and costs
  // nothing on Tensor Cores.
  const bool ok = gemm_ex((int)M, (int)N, (int)K, Xh, CUDA_R_16F, Wh, Yh,
                          CUDA_R_16F, CUBLAS_COMPUTE_32F);
  trace("fp16", M, N, K, ok);
  return ok;
}

bool cuda_fc_dense_gemm_fp16_f32out(const void *Xh, const void *Wh, float *Y,
                                    unsigned int M, unsigned int N,
                                    unsigned int K) {
  if (!dense_on() || Xh == nullptr || Wh == nullptr || Y == nullptr || M == 0 ||
      N == 0 || K == 0)
    return false;
  // Same fp32 accumulate as the fp16 entry point, just not rounded back down
  // on store. cublasGemmEx supports 16F/16F -> 32F with COMPUTE_32F directly,
  // so this costs an output buffer, not a conversion pass.
  const bool ok = gemm_ex((int)M, (int)N, (int)K, Xh, CUDA_R_16F, Wh, Y,
                          CUDA_R_32F, CUBLAS_COMPUTE_32F);
  trace("fp16->fp32", M, N, K, ok);
  return ok;
}

bool cuda_fc_dense_gemm_fp32(const float *X, const float *W, float *Y,
                             unsigned int M, unsigned int N, unsigned int K) {
  if (!dense_on() || X == nullptr || W == nullptr || Y == nullptr || M == 0 ||
      N == 0 || K == 0)
    return false;
  // CUBLAS_COMPUTE_32F (not _32F_FAST_TF32): TF32 would silently truncate the
  // fp32 mantissa to 10 bits, which is a numerics change no caller asked for.
  const bool ok = gemm_ex((int)M, (int)N, (int)K, X, CUDA_R_32F, W, Y,
                          CUDA_R_32F, CUBLAS_COMPUTE_32F);
  trace("fp32", M, N, K, ok);
  return ok;
}

namespace {
// Armed warm-up request, largest shape per dtype (0 = none seen).
std::mutex g_warm_mtx;
struct WarmReq {
  unsigned int N = 0, K = 0, M = 0;
} g_warm[2];
bool g_warm_done[2] = {false, false};
} // namespace

void cuda_fc_dense_warmup(bool fp16, unsigned int N, unsigned int K,
                          unsigned int M) {
  if (!dense_on() || N == 0 || K == 0 || M == 0)
    return;
  // Called from the per-weight prebuild seam, i.e. from the PARALLEL load
  // workers: the record needs a real lock, not an unguarded flag.
  std::lock_guard<std::mutex> lk(g_warm_mtx);
  WarmReq &r = g_warm[fp16 ? 1 : 0];
  // Keep the LARGEST shape seen. cuBLAS picks its kernel per shape class, and
  // the bigger weight is the one whose module load is worth pre-paying.
  if ((size_t)N * K > (size_t)r.N * r.K) {
    r.N = N;
    r.K = K;
  }
  if (M > r.M)
    r.M = M;
}

void cuda_fc_dense_warmup_run() {
  for (int slot = 0; slot < 2; ++slot) {
    WarmReq r;
    {
      std::lock_guard<std::mutex> lk(g_warm_mtx);
      if (g_warm_done[slot] || g_warm[slot].N == 0)
        continue;
      g_warm_done[slot] = true;
      r = g_warm[slot];
    }
    const bool fp16 = (slot == 1);
    const size_t esz = fp16 ? 2u : 4u;
    void *A = nullptr, *B = nullptr, *C = nullptr;
    const size_t a_b = (size_t)r.M * r.K * esz, b_b = (size_t)r.K * r.N * esz,
                 c_b = (size_t)r.M * r.N * esz;
    // Own scratch rather than the real weight: this runs on whatever thread
    // reaches it first, and cuBLAS's kernel choice depends on the SHAPE, not
    // on the values -- so borrowing a live tensor would only add a race.
    if (cudaMalloc(&A, a_b) == cudaSuccess &&
        cudaMalloc(&B, b_b) == cudaSuccess &&
        cudaMalloc(&C, c_b) == cudaSuccess) {
      cudaStream_t stream = StreamManager::Global().GetStream();
      cudaMemsetAsync(A, 0, a_b, stream);
      cudaMemsetAsync(B, 0, b_b, stream);
      const cudaDataType dt = fp16 ? CUDA_R_16F : CUDA_R_32F;
      // Both shape classes the LLM graphs use: M>1 (prefill) picks a tiled
      // Tensor-Core kernel, M==1 (decode) a GEMV -- different cuBLAS modules,
      // so warming only one leaves the other's load on the critical path.
      (void)gemm_ex((int)r.M, (int)r.N, (int)r.K, A, dt, B, C, dt,
                    CUBLAS_COMPUTE_32F);
      (void)gemm_ex(1, (int)r.N, (int)r.K, A, dt, B, C, dt,
                    CUBLAS_COMPUTE_32F);
      cudaStreamSynchronize(stream);
    }
    if (A)
      cudaFree(A);
    if (B)
      cudaFree(B);
    if (C)
      cudaFree(C);
  }
}

} // namespace nntrainer::cuda

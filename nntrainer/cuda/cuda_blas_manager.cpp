// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_blas_manager.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   cuBLAS handle management implementation.
 */

#include "cuda_blas_manager.h"
#include "cuda_context_manager.h"
#include "cuda_stream_manager.h"

#include <nntrainer_log.h>

namespace nntrainer::cuda {

void BlasManager::initialize() noexcept {
  ContextManager::Global().EnsureCurrent();
  if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
    ml_loge("[CUDA] cublasCreate failed");
    handle_ = nullptr;
    return;
  }
  // bind to the backend stream so GEMMs order with the dequant / copy kernels.
  cublasSetStream(handle_, StreamManager::Global().GetStream());
  ok_ = true;
}

bool BlasManager::sgemmRowMajor(int M, int N, int K, const float *X,
                                const float *W, float *Y) {
  if (!ok_)
    return false;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Column-major C[N,M] = W_view[N,K] * X_view[K,M] = (X*W)^T, read back
  // row-major as Y[M,N] = X*W. (orientation validated vs CPU reference)
  cublasStatus_t s =
    cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, W, N, X, K,
                &beta, Y, N);
  if (s != CUBLAS_STATUS_SUCCESS) {
    ml_loge("[CUDA] cublasSgemm failed: %d", (int)s);
    return false;
  }
  return true;
}

bool BlasManager::igemmRowMajor(int M, int N, int K, const signed char *A,
                                const signed char *B, int *C) {
  static int dbg = -2;
  if (dbg == -2)
    dbg = (std::getenv("NNTR_IGEMM_DBG") != nullptr) ? 1 : 0;
  if (!ok_) {
    if (dbg)
      fprintf(stderr, "[IGEMM] ok_=false (BlasManager not initialized)\n");
    return false;
  }
  const int alpha = 1;
  const int beta = 0;
  // Mirror sgemmRowMajor's orientation: column-major C[N,M] = B_view[N,K] *
  // A_view[K,M] reads back row-major as C[M,N] = A*B. B is int8 weight [K,N]
  // (ld=N), A is int8 act [M,K] (ld=K). int8 in / int32 accumulate -> IMMA
  // Tensor Cores on sm_75+ (N,K are multiples of 32 for the e2b dims).
  cublasStatus_t s = cublasGemmEx(
    handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_8I, N, A,
    CUDA_R_8I, K, &beta, C, CUDA_R_32I, N, CUBLAS_COMPUTE_32I,
    CUBLAS_GEMM_DEFAULT);
  if (s != CUBLAS_STATUS_SUCCESS) {
    if (dbg)
      fprintf(stderr, "[IGEMM] cublasGemmEx int8 status=%d (M=%d N=%d K=%d)\n",
              (int)s, M, N, K);
    ml_loge("[CUDA] cublasGemmEx int8 failed: %d", (int)s);
    return false;
  }
  if (dbg) {
    static int once = 0;
    if (once++ < 3)
      fprintf(stderr, "[IGEMM] OK M=%d N=%d K=%d\n", M, N, K);
  }
  return true;
}

BlasManager::~BlasManager() {
  if (handle_)
    cublasDestroy(handle_);
}

} // namespace nntrainer::cuda

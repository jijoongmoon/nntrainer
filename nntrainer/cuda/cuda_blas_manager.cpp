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

BlasManager::~BlasManager() {
  if (handle_)
    cublasDestroy(handle_);
}

} // namespace nntrainer::cuda

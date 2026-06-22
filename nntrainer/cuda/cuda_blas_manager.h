// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_blas_manager.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Owns the process-lifetime cuBLAS handle, bound to the CUDA backend
 *          stream. Kept separate from StreamManager so cublas_v2.h stays out of
 *          the core runtime headers. Used by CudaFcLayer (and later GEMM ops).
 */

#ifndef __CUDA_BLAS_MANAGER_H__
#define __CUDA_BLAS_MANAGER_H__

#include <cublas_v2.h>

#include "singleton.h"

namespace nntrainer::cuda {

/**
 * @class BlasManager
 * @brief Singleton cuBLAS handle bound to the backend stream.
 */
class BlasManager : public Singleton<BlasManager> {
public:
  /**
   * @brief raw cuBLAS handle (nullptr if init failed)
   */
  cublasHandle_t handle() const { return handle_; }

  /**
   * @brief Row-major Y[M,N] = X[M,K] * W[K,N].
   *
   * nntrainer's FC weight is laid out [K,N] (weight_dim (1,1,K,N)), activation
   * X is [M,K] row-major, output Y is [M,N] row-major. cuBLAS is column-major,
   * so we feed the row-major buffers with swapped operands (the row-major [M,N]
   * == column-major [N,M] identity), validated against a CPU reference.
   *
   * @return true on CUBLAS_STATUS_SUCCESS
   */
  bool sgemmRowMajor(int M, int N, int K, const float *X, const float *W,
                     float *Y);

  /**
   * @brief Destroy the cuBLAS handle.
   */
  ~BlasManager() override;

protected:
  /**
   * @brief Singleton hook: create the handle and bind it to the backend stream.
   */
  void initialize() noexcept override;

private:
  cublasHandle_t handle_{nullptr};
  bool ok_{false};
};

} // namespace nntrainer::cuda

#endif // __CUDA_BLAS_MANAGER_H__

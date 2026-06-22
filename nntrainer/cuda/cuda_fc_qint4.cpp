// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_fc_qint4.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Fused QINT4 dequant-GEMM implementation (NVRTC kernel).
 */

#include "cuda_fc_qint4.h"

#include <cuda_context.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

namespace nntrainer::cuda {

// One thread per output element Y[m,n]; loops K reading the int4 weight at the
// row-major [K,N] linear index i = k*N + n, dequantizing inline (even i = high
// nibble via arithmetic >>4, odd i = low nibble sign-extended -- matches
// Int4QTensor::getValue) and scaling by scale[i/group]. float accumulation.
static const char *FC_QINT4_SRC = R"CU(
extern "C" __global__ void fc_qint4_gemm(const float *X,
                                         const unsigned char *nib,
                                         const float *sc, float *Y, int M, int N,
                                         int K, int group) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    int i = k * N + n;
    signed char byte = (signed char)nib[i >> 1];
    int v;
    if ((i & 1) == 0)
      v = byte >> 4;
    else {
      signed char t = (signed char)(byte << 4);
      v = t >> 4;
    }
    acc += X[m * K + k] * ((float)v * sc[i / group]);
  }
  Y[m * N + n] = acc;
}
)CU";

bool cuda_fc_qint4_gemm_fp32(const float *X, const unsigned char *nibbles,
                             const float *scales, float *Y, unsigned int M,
                             unsigned int N, unsigned int K,
                             unsigned int group) {
  if (M == 0 || N == 0 || K == 0)
    return true;

  auto kernel =
    CudaContext::Global().registerCudaKernel(FC_QINT4_SRC, "fc_qint4_gemm");
  if (!kernel) {
    ml_loge("[CUDA] fc_qint4: kernel registration failed");
    return false;
  }

  int m = (int)M, n = (int)N, k = (int)K, g = (int)group;
  kernel->SetKernelArguments(0, &X, sizeof(X));
  kernel->SetKernelArguments(1, &nibbles, sizeof(nibbles));
  kernel->SetKernelArguments(2, &scales, sizeof(scales));
  kernel->SetKernelArguments(3, &Y, sizeof(Y));
  kernel->SetKernelArguments(4, &m, sizeof(m));
  kernel->SetKernelArguments(5, &n, sizeof(n));
  kernel->SetKernelArguments(6, &k, sizeof(k));
  kernel->SetKernelArguments(7, &g, sizeof(g));

  const int block[3] = {16, 16, 1};
  const int grid[3] = {((int)N + 15) / 16, ((int)M + 15) / 16, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().finish();
  return true;
}

} // namespace nntrainer::cuda

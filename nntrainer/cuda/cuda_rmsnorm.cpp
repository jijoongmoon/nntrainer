// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_rmsnorm.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device RMSNorm op implementation (NVRTC kernel, validated math).
 */

#include "cuda_rmsnorm.h"

#include <cuda_context.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

namespace nntrainer::cuda {

// One block per row; block-reduces the sum of squares in FP32; scales by
// rsqrt(mean+eps) and folds the raw gamma (no (1+gamma) bias). has_gamma=0
// skips the gamma read (gamma-free v_norm).
static const char *RMSNORM_FP32_SRC = R"CU(
extern "C" __global__ void rmsnorm_fp32(const float *x, const float *gamma,
                                        float *y, int width, float eps,
                                        int has_gamma) {
  int row = blockIdx.x;
  const float *xr = x + (size_t)row * width;
  float *yr = y + (size_t)row * width;
  __shared__ float sdata[256];
  float partial = 0.f;
  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    float v = xr[k];
    partial += v * v;
  }
  sdata[threadIdx.x] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(sdata[0] / (float)width + eps);
  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    float g = has_gamma ? gamma[k] : 1.0f;
    yr[k] = xr[k] * inv * g;
  }
}
)CU";

bool cuda_rmsnorm_fp32(const float *in, const float *gamma, float *out,
                       float eps, unsigned int rows, unsigned int width) {
  if (rows == 0 || width == 0)
    return true;

  auto kernel =
    CudaContext::Global().registerCudaKernel(RMSNORM_FP32_SRC, "rmsnorm_fp32");
  if (!kernel) {
    ml_loge("[CUDA] rmsnorm: kernel registration failed");
    return false;
  }

  int w = (int)width;
  int has_gamma = (gamma != nullptr) ? 1 : 0;
  const float *gamma_ptr = gamma; // may be null; never dereferenced when 0

  kernel->SetKernelArguments(0, &in, sizeof(in));
  kernel->SetKernelArguments(1, &gamma_ptr, sizeof(gamma_ptr));
  kernel->SetKernelArguments(2, &out, sizeof(out));
  kernel->SetKernelArguments(3, &w, sizeof(w));
  kernel->SetKernelArguments(4, &eps, sizeof(eps));
  kernel->SetKernelArguments(5, &has_gamma, sizeof(has_gamma));

  const int block[3] = {256, 1, 1};
  const int grid[3] = {(int)rows, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().finish();
  return true;
}

} // namespace nntrainer::cuda

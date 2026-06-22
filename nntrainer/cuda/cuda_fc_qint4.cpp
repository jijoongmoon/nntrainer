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

#include <cuda_runtime.h>

#include <mutex>
#include <unordered_map>

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

// One thread per output Y[m,n]. Loops K, decoding the signed int4 weight for
// (n, k) straight from the KAI Section-A super-row payload by inverting
// Int4Utils::packPlainToSectionA (NR=4, KR=16, SR=2, 16-K interleave,
// block_length=8, byte = plain_nibbles ^ 0x88). Constants are baked in. The
// per-output-channel fp16 scale (one per n) is read once and converted to fp32
// with a self-contained half->float (no NVRTC header dependency). float accum.
//   k_internal = roundup(K,32); for a loaded weight K%32==0 so k_internal==K.
//   stride (nibble_bytes_per_super_row) = NR*(k_internal/2) = 2*k_internal.
// Inverse index for (n,k): a=k/32, r=k%32, is_high=(r>=16),
//   kb = is_high? r-16 : r,  t = 16*a + kb,
//   byte_off = (n/4)*2*k_internal + ((t/8)*4 + (n%4))*8 + (t%8),
//   nib = is_high ? hi-nibble : lo-nibble of section_a[byte_off],
//   int4 = (nib ^ 8) - 8.
static const char *FC_QINT4_SECA_SRC = R"CU(
extern "C" {

__device__ __forceinline__ float seca_h2f(unsigned short h) {
  unsigned int sign = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int exp = (h >> 10) & 0x1Fu;
  unsigned int mant = h & 0x3FFu;
  unsigned int out;
  if (exp == 0u) {
    if (mant == 0u) {
      out = sign;
    } else {
      int e = -1;
      do { mant <<= 1; e++; } while ((mant & 0x400u) == 0u);
      mant &= 0x3FFu;
      out = sign | ((unsigned int)(127 - 15 - e) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    out = sign | 0x7F800000u | (mant << 13);
  } else {
    out = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
  }
  return __int_as_float((int)out);
}

__global__ void fc_qint4_seca_gemm(const float *X, const unsigned char *secA,
                                   const unsigned short *sc, float *Y, int M,
                                   int N, int K, int k_internal) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  const int stride = 2 * k_internal;        // bytes per super-row
  const int base = (n >> 2) * stride;        // super-row start
  const int nr_idx = n & 3;
  const float *xr = X + (long)m * K;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    int a = k >> 5;          // k / 32
    int r = k & 31;          // k % 32
    int is_high = (r >= 16) ? 1 : 0;
    int kb = is_high ? (r - 16) : r;
    int t = (a << 4) + kb;   // 16*a + kb
    int byte_off = base + (((t >> 3) << 2) + nr_idx) * 8 + (t & 7);
    unsigned char s = secA[byte_off];
    int nib = is_high ? ((s >> 4) & 0xF) : (s & 0xF);
    int q = (nib ^ 8) - 8;   // signed int4 in [-8,7]
    acc += xr[k] * (float)q;
  }
  Y[(long)m * N + n] = acc * seca_h2f(sc[n]);
}

}
)CU";

bool cuda_fc_qint4_sectionA_gemm_fp32(const float *X,
                                      const unsigned char *section_a,
                                      const unsigned short *scales_fp16,
                                      float *Y, unsigned int M, unsigned int N,
                                      unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;

  auto kernel = CudaContext::Global().registerCudaKernel(FC_QINT4_SECA_SRC,
                                                         "fc_qint4_seca_gemm");
  if (!kernel) {
    ml_loge("[CUDA] fc_qint4_seca: kernel registration failed");
    return false;
  }

  int m = (int)M, n = (int)N, k = (int)K;
  int k_internal = (int)(((K + 31u) / 32u) * 32u);
  kernel->SetKernelArguments(0, &X, sizeof(X));
  kernel->SetKernelArguments(1, &section_a, sizeof(section_a));
  kernel->SetKernelArguments(2, &scales_fp16, sizeof(scales_fp16));
  kernel->SetKernelArguments(3, &Y, sizeof(Y));
  kernel->SetKernelArguments(4, &m, sizeof(m));
  kernel->SetKernelArguments(5, &n, sizeof(n));
  kernel->SetKernelArguments(6, &k, sizeof(k));
  kernel->SetKernelArguments(7, &k_internal, sizeof(k_internal));

  const int block[3] = {16, 16, 1};
  const int grid[3] = {((int)N + 15) / 16, ((int)M + 15) / 16, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().finish();
  return true;
}

namespace {
// Device mirror of a host-resident QINT4 weight + reusable activation/output
// staging buffers. Weights are constant for the model lifetime, so the
// Section-A payload + scales are uploaded once and cached by host pointer.
struct DevWeight {
  unsigned char *d_secA = nullptr;
  unsigned short *d_sc = nullptr;
};
std::unordered_map<const void *, DevWeight> g_qint4_weight_cache;
float *g_stage_x = nullptr;
size_t g_stage_x_cap = 0;
float *g_stage_y = nullptr;
size_t g_stage_y_cap = 0;
std::mutex g_qint4_mtx;

bool ensure_stage(float **buf, size_t *cap, size_t bytes) {
  if (bytes <= *cap)
    return true;
  if (*buf)
    cudaFree(*buf);
  if (cudaMalloc(buf, bytes) != cudaSuccess) {
    *buf = nullptr;
    *cap = 0;
    return false;
  }
  *cap = bytes;
  return true;
}
} // namespace

bool cuda_fc_qint4_sectionA_gemm_fp32_resident(const float *host_X,
                                               const unsigned char *host_secA,
                                               const unsigned short *host_scales,
                                               float *host_Y, unsigned int M,
                                               unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  std::lock_guard<std::mutex> lk(g_qint4_mtx);

  // 1) device weight (upload once, cache by host pointer).
  auto it = g_qint4_weight_cache.find(host_secA);
  if (it == g_qint4_weight_cache.end()) {
    const unsigned k_internal = ((K + 31u) / 32u) * 32u;
    const size_t secA_bytes =
      (size_t)(((N + 3u) / 4u) * 4u) * (k_internal / 2u);
    DevWeight dw;
    if (cudaMalloc(&dw.d_secA, secA_bytes) != cudaSuccess)
      return false;
    if (cudaMalloc(&dw.d_sc, sizeof(unsigned short) * (size_t)N) !=
        cudaSuccess) {
      cudaFree(dw.d_secA);
      return false;
    }
    cudaMemcpy(dw.d_secA, host_secA, secA_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dw.d_sc, host_scales, sizeof(unsigned short) * (size_t)N,
               cudaMemcpyHostToDevice);
    it = g_qint4_weight_cache.emplace(host_secA, dw).first;
  }

  // 2) stage activation in, output buffer out (grown as needed).
  const size_t xb = sizeof(float) * (size_t)M * K;
  const size_t yb = sizeof(float) * (size_t)M * N;
  if (!ensure_stage(&g_stage_x, &g_stage_x_cap, xb) ||
      !ensure_stage(&g_stage_y, &g_stage_y_cap, yb))
    return false;
  cudaMemcpy(g_stage_x, host_X, xb, cudaMemcpyHostToDevice);

  // 3) device GEMM (synchronizes the backend stream internally).
  if (!cuda_fc_qint4_sectionA_gemm_fp32(g_stage_x, it->second.d_secA,
                                        it->second.d_sc, g_stage_y, M, N, K))
    return false;

  // 4) output back to the host tensor.
  cudaMemcpy(host_Y, g_stage_y, yb, cudaMemcpyDeviceToHost);
  return true;
}

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

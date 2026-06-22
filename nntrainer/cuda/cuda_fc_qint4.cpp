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

// ===========================================================================
// w4a8 dp4a fast path
// ===========================================================================
// Three NVRTC kernels (one module): per-row int8 activation quant, a one-time
// Section-A -> plain row-major int4 repack (reuses the validated inverse
// mapping), and a __dp4a int8xint4 GEMM. Compiled for the device arch
// (compute_89 on Ada), so __dp4a lowers to the dp4a PTX instruction.
static const char *FC_QINT4_DP4A_SRC = R"CU(
extern "C" {

__device__ __forceinline__ float dp4a_h2f(unsigned short h) {
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

// signed int4 weight for (output n, input k) from the Section-A payload.
__device__ __forceinline__ int seca_decode(const unsigned char *secA, int n,
                                            int k, int k_internal) {
  int a = k >> 5, r = k & 31;
  int is_high = (r >= 16) ? 1 : 0;
  int kb = is_high ? (r - 16) : r;
  int t = (a << 4) + kb;
  int nr_idx = n & 3;
  int byte_off = (n >> 2) * (2 * k_internal) + (((t >> 3) << 2) + nr_idx) * 8 +
                 (t & 7);
  unsigned char s = secA[byte_off];
  int nib = is_high ? ((s >> 4) & 0xF) : (s & 0xF);
  return (nib ^ 8) - 8;
}

// per-row symmetric int8 quant of the activation (one block per row).
__global__ void act_quant_i8(const float *X, signed char *q8, float *ascale,
                             int M, int K) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  __shared__ float sm[256];
  const float *xr = X + (long)m * K;
  float local = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    local = fmaxf(local, fabsf(xr[k]));
  sm[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
    __syncthreads();
  }
  float amax = sm[0];
  float inv = amax > 0.f ? 127.0f / amax : 0.f;
  if (threadIdx.x == 0)
    ascale[m] = amax > 0.f ? amax / 127.0f : 1.0f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = __float2int_rn(xr[k] * inv);
    q = max(-127, min(127, q));
    q8[(long)m * K + k] = (signed char)q;
  }
}

// Section-A -> plain row-major int4 [N, ceil(K/2)]: byte[n][kb] low nibble =
// int4(n, 2kb), high nibble = int4(n, 2kb+1), each stored two's-complement.
__global__ void repack_seca_i4(const unsigned char *secA, signed char *plain,
                               int N, int K, int k_internal) {
  int kb = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  int Kh = (K + 1) >> 1;
  if (n >= N || kb >= Kh)
    return;
  int k0 = 2 * kb, k1 = 2 * kb + 1;
  int v0 = seca_decode(secA, n, k0, k_internal);
  int v1 = (k1 < K) ? seca_decode(secA, n, k1, k_internal) : 0;
  plain[(long)n * Kh + kb] = (signed char)((v0 & 0xF) | ((v1 & 0xF) << 4));
}

// Y[m,n] = ascale[m] * w_scale[n] * sum_k q8[m,k] * int4(n,k), via __dp4a.
__global__ void dp4a_gemm(const signed char *q8, const signed char *plain,
                          const float *ascale, const unsigned short *wscale,
                          float *Y, int M, int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  int Kh = (K + 1) >> 1;
  const signed char *qrow = q8 + (long)m * K;
  const signed char *wrow = plain + (long)n * Kh;
  int acc = 0, k = 0;
  for (; k + 4 <= K; k += 4) {
    int a = *(const int *)(qrow + k); // lanes = act k,k+1,k+2,k+3
    int kb = k >> 1;
    int b0 = (unsigned char)wrow[kb];     // k(low), k+1(high)
    int b1 = (unsigned char)wrow[kb + 1]; // k+2(low), k+3(high)
    int w0 = ((int)(signed char)(b0 << 4)) >> 4;
    int w1 = ((int)(signed char)b0) >> 4;
    int w2 = ((int)(signed char)(b1 << 4)) >> 4;
    int w3 = ((int)(signed char)b1) >> 4;
    int w = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) |
            ((w3 & 0xFF) << 24);
    acc = __dp4a(a, w, acc);
  }
  for (; k < K; ++k) { // tail (none when K%32==0)
    int kb = k >> 1;
    int b = (unsigned char)wrow[kb];
    int wv = (k & 1) ? (((int)(signed char)b) >> 4)
                     : (((int)(signed char)(b << 4)) >> 4);
    acc += (int)qrow[k] * wv;
  }
  Y[(long)m * N + n] = (float)acc * ascale[m] * dp4a_h2f(wscale[n]);
}

// Register-blocked dp4a GEMM: a 64x64 output tile per block; each of the 256
// threads accumulates a 4x4 micro-tile in registers, so a K-chunk of 32 staged
// once into shared memory feeds 16 dp4a per thread before the next load -- much
// higher arithmetic intensity than the 1-output-per-thread tiled kernel.
#define RB_BM 64
#define RB_BN 64
#define RB_BK 32
#define RB_TM 4
#define RB_TN 4
__global__ void dp4a_gemm_reg(const signed char *q8, const signed char *plain,
                              const float *ascale, const unsigned short *wscale,
                              float *Y, int M, int N, int K) {
  __shared__ signed char As[RB_BM][RB_BK];
  __shared__ signed char Ws[RB_BN][RB_BK];
  int tx = threadIdx.x, ty = threadIdx.y; // 0..15 each
  int tid = ty * 16 + tx;
  int blockM = blockIdx.y * RB_BM, blockN = blockIdx.x * RB_BN;
  int Kh = (K + 1) >> 1;
  int acc[RB_TM][RB_TN];
#pragma unroll
  for (int i = 0; i < RB_TM; i++)
#pragma unroll
    for (int j = 0; j < RB_TN; j++)
      acc[i][j] = 0;
  for (int k0 = 0; k0 < K; k0 += RB_BK) {
    for (int e = tid; e < RB_BM * RB_BK; e += 256) {
      int i = e / RB_BK, j = e % RB_BK;
      int mm = blockM + i, kk = k0 + j;
      As[i][j] = (mm < M && kk < K) ? q8[(long)mm * K + kk] : (signed char)0;
    }
    for (int e = tid; e < RB_BN * RB_BK; e += 256) {
      int i = e / RB_BK, j = e % RB_BK;
      int nn = blockN + i, kk = k0 + j;
      signed char wv = 0;
      if (nn < N && kk < K) {
        unsigned char b = (unsigned char)plain[(long)nn * Kh + (kk >> 1)];
        wv = (kk & 1) ? (((signed char)b) >> 4)
                      : (((signed char)(b << 4)) >> 4);
      }
      Ws[i][j] = wv;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < RB_BK; kk += 4) {
      int af[RB_TM], wf[RB_TN];
#pragma unroll
      for (int i = 0; i < RB_TM; i++)
        af[i] = *(const int *)&As[ty * RB_TM + i][kk];
#pragma unroll
      for (int j = 0; j < RB_TN; j++)
        wf[j] = *(const int *)&Ws[tx * RB_TN + j][kk];
#pragma unroll
      for (int i = 0; i < RB_TM; i++)
#pragma unroll
        for (int j = 0; j < RB_TN; j++)
          acc[i][j] = __dp4a(af[i], wf[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < RB_TM; i++) {
    int row = blockM + ty * RB_TM + i;
    if (row >= M)
      continue;
    float as = ascale[row];
#pragma unroll
    for (int j = 0; j < RB_TN; j++) {
      int col = blockN + tx * RB_TN + j;
      if (col < N)
        Y[(long)row * N + col] = (float)acc[i][j] * as * dp4a_h2f(wscale[col]);
    }
  }
}

}
)CU";

namespace {
// cached plain-int4 repack of each Section-A weight (keyed by host/UVM pointer).
std::unordered_map<const void *, signed char *> g_dp4a_plain_cache;
signed char *g_dp4a_q8 = nullptr;
size_t g_dp4a_q8_cap = 0;
float *g_dp4a_ascale = nullptr;
size_t g_dp4a_ascale_cap = 0;
std::mutex g_dp4a_mtx;
} // namespace

bool cuda_fc_qint4_sectionA_dp4a_gemm_fp32(const float *X,
                                           const unsigned char *section_a,
                                           const unsigned short *scales_fp16,
                                           float *Y, unsigned int M,
                                           unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;

  auto kq = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "act_quant_i8");
  auto kr = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "repack_seca_i4");
  // register-blocked 64x64-tile GEMM for batched M (prefill); the per-output
  // kernel for tiny M (decode), where a big tile would mostly idle.
  const bool tiled = (M >= 8);
  auto kg = CudaContext::Global().registerCudaKernel(
    FC_QINT4_DP4A_SRC, tiled ? "dp4a_gemm_reg" : "dp4a_gemm");
  if (!kq || !kr || !kg) {
    ml_loge("[CUDA] fc_qint4 dp4a: kernel registration failed");
    return false;
  }

  std::lock_guard<std::mutex> lk(g_dp4a_mtx);

  const int m = (int)M, n = (int)N, k = (int)K;
  const int k_internal = (int)(((K + 31u) / 32u) * 32u);
  const size_t Kh = (K + 1u) / 2u;

  // 1) one-time Section-A -> plain int4 repack (cached on device).
  auto it = g_dp4a_plain_cache.find(section_a);
  if (it == g_dp4a_plain_cache.end()) {
    signed char *plain = nullptr;
    if (cudaMalloc(&plain, (size_t)N * Kh) != cudaSuccess)
      return false;
    kr->SetKernelArguments(0, &section_a, sizeof(section_a));
    kr->SetKernelArguments(1, &plain, sizeof(plain));
    kr->SetKernelArguments(2, &n, sizeof(n));
    kr->SetKernelArguments(3, &k, sizeof(k));
    kr->SetKernelArguments(4, &k_internal, sizeof(k_internal));
    const int rb[3] = {16, 16, 1};
    const int rg[3] = {((int)Kh + 15) / 16, ((int)N + 15) / 16, 1};
    if (!StreamManager::Global().DispatchCommand(*kr, rg, rb)) {
      cudaFree(plain);
      return false;
    }
    it = g_dp4a_plain_cache.emplace(section_a, plain).first;
  }
  signed char *plain = it->second;

  // 2) int8 activation quant (per-row), into grown device scratch.
  const size_t q8b = (size_t)M * K;
  const size_t asb = sizeof(float) * (size_t)M;
  if (q8b > g_dp4a_q8_cap) {
    if (g_dp4a_q8)
      cudaFree(g_dp4a_q8);
    if (cudaMalloc(&g_dp4a_q8, q8b) != cudaSuccess) {
      g_dp4a_q8 = nullptr;
      g_dp4a_q8_cap = 0;
      return false;
    }
    g_dp4a_q8_cap = q8b;
  }
  if (asb > g_dp4a_ascale_cap) {
    if (g_dp4a_ascale)
      cudaFree(g_dp4a_ascale);
    if (cudaMalloc(&g_dp4a_ascale, asb) != cudaSuccess) {
      g_dp4a_ascale = nullptr;
      g_dp4a_ascale_cap = 0;
      return false;
    }
    g_dp4a_ascale_cap = asb;
  }
  kq->SetKernelArguments(0, &X, sizeof(X));
  kq->SetKernelArguments(1, &g_dp4a_q8, sizeof(g_dp4a_q8));
  kq->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kq->SetKernelArguments(3, &m, sizeof(m));
  kq->SetKernelArguments(4, &k, sizeof(k));
  const int qb[3] = {256, 1, 1};
  const int qg[3] = {(int)M, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kq, qg, qb))
    return false;

  // 3) dp4a GEMM.
  kg->SetKernelArguments(0, &g_dp4a_q8, sizeof(g_dp4a_q8));
  kg->SetKernelArguments(1, &plain, sizeof(plain));
  kg->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kg->SetKernelArguments(3, &scales_fp16, sizeof(scales_fp16));
  kg->SetKernelArguments(4, &Y, sizeof(Y));
  kg->SetKernelArguments(5, &m, sizeof(m));
  kg->SetKernelArguments(6, &n, sizeof(n));
  kg->SetKernelArguments(7, &k, sizeof(k));
  const int gb[3] = {16, 16, 1};
  // reg-blocked kernel: 64x64 output tile/block; per-output kernel: 16x16.
  const int tile = tiled ? 64 : 16;
  const int grid[3] = {((int)N + tile - 1) / tile, ((int)M + tile - 1) / tile,
                       1};
  if (!StreamManager::Global().DispatchCommand(*kg, grid, gb))
    return false;
  StreamManager::Global().finish();
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

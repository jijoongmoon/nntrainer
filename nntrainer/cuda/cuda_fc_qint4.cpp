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

#include <cuda_blas_manager.h>
#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <thread_manager.h>

namespace nntrainer::cuda {

// Per-op cudaStreamSynchronize is ~90% of inference wall time (nsys): each GPU
// op drains the stream, fully serializing CPU and GPU. This drain is a sync
// point hook for the future selective-sync work (sync only before a HOST
// consumer reads a UVM output, not after every FC).
//
// NNTR_CUDA_ASYNC=1 drops the drains -- EXPERIMENTAL/UNSAFE: it makes decode
// ~40% faster but produces GARBAGE, because the host ops between FCs (RoPE,
// attention, geglu) then read UVM the GPU is still writing -- the
// concurrentManagedAccess page-fault does NOT order a host read against an
// in-flight kernel write. The coherent path to that speedup is to move those
// decode host ops onto the GPU too (GPU RoPE/geglu, the GPU attention exists)
// so the whole decode step is one ordered GPU chain drained once per token.
// Default (sync) is coherent.
static inline void maybe_finish() {
  static const bool async = []() {
    const char *e = std::getenv("NNTR_CUDA_ASYNC");
    if (e == nullptr || e[0] != '1')
      return false;
    // Integrated GPU (Tegra/Orin): force sync -- async is non-coherent on the
    // shared-memory iGPU (see cuda_stream_manager cuda_async_mode()).
    return !ContextManager::Global().isIntegrated();
  }();
  if (!async)
    StreamManager::Global().finish();
}

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
  maybe_finish();
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
  // cudaMalloc/cudaFree inside a CUDA-graph stream capture invalidates the
  // capture. The fp32-resident staging buffers are pre-grown at load by
  // cuda_fc_qint4_dp4a_prewarm() so this branch must not run under capture; if
  // it ever would (an under-sized prewarm), bail so the caller falls back
  // rather than corrupting the graph.
  if (StreamManager::Global().isCapturing())
    return false;
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
  StreamManager::Global().finishIfAsync();
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

// float -> fp16 (IEEE half), round to nearest even.
__device__ __forceinline__ unsigned short dp4a_f2h(float f) {
  unsigned int x = (unsigned int)__float_as_int(f);
  unsigned int sign = (x >> 16) & 0x8000u;
  int e = (int)((x >> 23) & 0xFFu);
  unsigned int mant = x & 0x7FFFFFu;
  if (e == 0xFF)
    return (unsigned short)(sign | 0x7C00u | (mant ? 0x200u : 0u)); // inf/nan
  int exp = e - 127 + 15;
  if (exp >= 0x1F)
    return (unsigned short)(sign | 0x7C00u); // overflow -> inf
  if (exp <= 0) {
    if (exp < -10)
      return (unsigned short)sign; // underflow -> 0
    mant |= 0x800000u;
    int shift = 14 - exp;
    unsigned int h = mant >> shift;
    unsigned int rem = mant & ((1u << shift) - 1u);
    unsigned int half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1u)))
      h++;
    return (unsigned short)(sign | h);
  }
  unsigned int h = ((unsigned int)exp << 10) | (mant >> 13);
  unsigned int rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u)))
    h++;
  return (unsigned short)(sign | h);
}

// asymmetric int8 quant params for a row's [min,max] (range forced to include
// 0, nudged zero-point) -- mirrors the OpenCL v8c act-quant. Returns recip
// (dequant scale) and zp; sets scale_q (quant multiplier) by reference.
__device__ __forceinline__ void asym_qparams(float fmn, float fmx,
                                             float &scale_q, float &recip,
                                             int &zp) {
  float rmin = fminf(0.f, fmn), rmax = fmaxf(0.f, fmx);
  float range = rmax - rmin;
  scale_q = range > 0.f ? 255.f / range : 1.f;
  recip = range > 0.f ? range / 255.f : 1.f;
  float dmin = rmin * scale_q, dmax = rmax * scale_q;
  float zp_lo = -128.f - dmin, zp_hi = 127.f - dmax;
  float zp_f = ((-128.f + dmin) + (127.f + dmax) > 0.f) ? zp_lo : zp_hi;
  zp_f = fmaxf(-128.f, fminf(127.f, zp_f));
  zp = (int)rintf(zp_f);
}

// per-row asymmetric int8 quant of an fp16 activation (one block per row).
// stores recip in ascale[m], zero-point in azp[m].
__global__ void act_quant_i8_h(const unsigned short *Xh, signed char *q8,
                               float *ascale, int *azp, int M, int K) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  __shared__ float smn[256];
  __shared__ float smx[256];
  const unsigned short *xr = Xh + (long)m * K;
  float lmn = 0.f, lmx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = dp4a_h2f(xr[k]);
    lmn = fminf(lmn, v);
    lmx = fmaxf(lmx, v);
  }
  smn[threadIdx.x] = lmn;
  smx[threadIdx.x] = lmx;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x + s]);
      smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x + s]);
    }
    __syncthreads();
  }
  float scale_q, recip;
  int zp;
  asym_qparams(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) {
    ascale[m] = recip;
    azp[m] = zp;
  }
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = (int)rintf(dp4a_h2f(xr[k]) * scale_q) + zp;
    q = max(-128, min(127, q));
    q8[(long)m * K + k] = (signed char)q;
  }
}

// per-output-channel weight row-sum (sum of signed int4) for the activation
// zero-point correction: Y -= recip[m]*scale_w[n]*zp[m]*rowsum_w[n].
__global__ void weight_rowsum(const signed char *plain, int *rowsum, int N,
                              int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N)
    return;
  int Kh = (K + 1) >> 1;
  const signed char *wrow = plain + (long)n * Kh;
  int s = 0;
  for (int kb = 0; kb < Kh; ++kb) {
    int b = (unsigned char)wrow[kb];
    int k0 = 2 * kb, k1 = 2 * kb + 1;
    if (k0 < K)
      s += ((int)(signed char)(b << 4)) >> 4;
    if (k1 < K)
      s += ((int)(signed char)b) >> 4;
  }
  rowsum[n] = s;
}

// float buffer -> fp16 buffer.
__global__ void cvt_f2h(const float *src, unsigned short *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = dp4a_f2h(src[i]);
}

// fp16 buffer -> float buffer.
__global__ void cvt_h2f(const unsigned short *src, float *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = dp4a_h2f(src[i]);
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

// per-row asymmetric int8 quant of the activation (one block per row).
__global__ void act_quant_i8(const float *X, signed char *q8, float *ascale,
                             int *azp, int M, int K) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  __shared__ float smn[256];
  __shared__ float smx[256];
  const float *xr = X + (long)m * K;
  float lmn = 0.f, lmx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = xr[k];
    lmn = fminf(lmn, v);
    lmx = fmaxf(lmx, v);
  }
  smn[threadIdx.x] = lmn;
  smx[threadIdx.x] = lmx;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x + s]);
      smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x + s]);
    }
    __syncthreads();
  }
  float scale_q, recip;
  int zp;
  asym_qparams(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) {
    ascale[m] = recip;
    azp[m] = zp;
  }
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = (int)rintf(xr[k] * scale_q) + zp;
    q = max(-128, min(127, q));
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

// Y[m,n] = recip[m]*w_scale[n]*(sum_k q8[m,k]*int4(n,k) - zp[m]*rowsum_w[n]),
// the asymmetric-activation dequant (zp from act_quant, rowsum_w from the
// weight). via __dp4a.
__global__ void dp4a_gemm(const signed char *q8, const signed char *plain,
                          const float *ascale, const int *azp,
                          const int *wrowsum, const unsigned short *wscale,
                          float *Y, int M, int N, int K, int out_fp16) {
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
  float r = (float)(acc - azp[m] * wrowsum[n]) * ascale[m] * dp4a_h2f(wscale[n]);
  if (out_fp16)
    ((unsigned short *)Y)[(long)m * N + n] = dp4a_f2h(r);
  else
    Y[(long)m * N + n] = r;
}

// Dedicated M=1 decode GEMV: one block per output n, threads split K and
// block-reduce. The tiled dp4a_gemm wastes 15/16 rows of its 16x16 block at M=1
// (94% idle) and reads weight rows with a stride; here every thread is active
// and reads a contiguous K-slice of one weight row (coalesced). Activation row
// is row 0 (q8). out_fp16 folds the fp16 conversion in.
__global__ void dp4a_gemv(const signed char *q8, const signed char *plain,
                          const float *ascale, const int *azp,
                          const int *wrowsum, const unsigned short *wscale,
                          float *Y, int N, int K, int out_fp16) {
  // One WARP per output n (warps_per_block outputs per block) -> N/warps_per_block
  // blocks instead of N, amortizing the per-block launch/epilogue overhead that
  // dominated the old one-block-per-tiny-output design. No shared memory, no
  // __syncthreads: each lane reads a coalesced K-slice of the weight row and the
  // warp-shuffle reduces. dp4a int32 accumulate is integer-associative so the
  // result is BIT-IDENTICAL to the block-reduce version. (llama.cpp MMVQ shape.)
  const int warps_per_block = blockDim.x >> 5;
  int n = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  if (n >= N)
    return;
  const int lane = threadIdx.x & 31;
  int Kh = (K + 1) >> 1;
  const signed char *wrow = plain + (long)n * Kh;
  int acc = 0;
  for (int k = lane * 4; k + 4 <= K; k += 32 * 4) {
    int a = *(const int *)(q8 + k);
    int kb = k >> 1; // = lane*2 -> 2-byte aligned for the short load (K even)
    unsigned int w16 = *(const unsigned short *)(wrow + kb);
    int b0 = w16 & 0xFF;
    int b1 = (w16 >> 8) & 0xFF;
    int w0 = ((int)(signed char)(b0 << 4)) >> 4;
    int w1 = ((int)(signed char)b0) >> 4;
    int w2 = ((int)(signed char)(b1 << 4)) >> 4;
    int w3 = ((int)(signed char)b1) >> 4;
    int w = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) |
            ((w3 & 0xFF) << 24);
    acc = __dp4a(a, w, acc);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) {
    float r = (float)(acc - azp[0] * wrowsum[n]) * ascale[0] *
              dp4a_h2f(wscale[n]);
    if (out_fp16)
      ((unsigned short *)Y)[n] = dp4a_f2h(r);
    else
      Y[n] = r;
  }
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
                              const float *ascale, const int *azp,
                              const int *wrowsum, const unsigned short *wscale,
                              float *Y, int M, int N, int K, int out_fp16) {
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
    int zp = azp[row];
#pragma unroll
    for (int j = 0; j < RB_TN; j++) {
      int col = blockN + tx * RB_TN + j;
      if (col < N) {
        float r =
          (float)(acc[i][j] - zp * wrowsum[col]) * as * dp4a_h2f(wscale[col]);
        if (out_fp16)
          ((unsigned short *)Y)[(long)row * N + col] = dp4a_f2h(r);
        else
          Y[(long)row * N + col] = r;
      }
    }
  }
}

// === cuBLAS INT8 IMMA (Tensor Core) prefill FC support ===
// The __dp4a kernels run on the int ALU (ceiling ~21 TOPS on Ada). cuBLAS int8
// IMMA runs on the Tensor Cores (~30 TOPS measured, ~10x our dp4a GEMM). These
// three kernels feed it: unpack the int4 weight -> int8 ONCE (cached), and the
// int32 GEMM result is bit-identical to the __dp4a acc, so the SAME dequant
// applies in the epilogue.

// int4 Section-A weight -> int8 [K,N] (w8[k*N+n] = int4(n,k)). Unpacked once and
// cached (weights are static), so cuBLAS reads contiguous int8 -- doing this per
// call would add a memory pass that erases the Tensor-Core win.
__global__ void repack_seca_i8_kn(const unsigned char *secA, signed char *w8,
                                  int N, int K, int k_internal) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (n >= N || k >= K)
    return;
  w8[(long)k * N + n] = (signed char)seca_decode(secA, n, k, k_internal);
}

// per-output-channel sum of the int8 weight column (k-strided), for the
// activation zero-point correction. one thread per output channel n.
__global__ void weight_rowsum_kn(const signed char *w8, int *rowsum, int N,
                                 int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N)
    return;
  long s = 0;
  for (int k = 0; k < K; ++k)
    s += (int)w8[(long)k * N + n];
  rowsum[n] = (int)s;
}

// dequant epilogue for the int8 IMMA GEMM: C is the int32 dot-product (== the
// __dp4a acc, bit-identical). Y[m,n]=(C - zp[m]*rowsum[n])*recip[m]*wscale[n].
__global__ void dequant_i32_fp16(const int *C, const float *ascale,
                                 const int *azp, const int *wrowsum,
                                 const unsigned short *wscale, unsigned short *Y,
                                 int M, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  float r = (float)(C[(long)m * N + n] - azp[m] * wrowsum[n]) * ascale[m] *
            dp4a_h2f(wscale[n]);
  Y[(long)m * N + n] = dp4a_f2h(r);
}

}
)CU";

namespace {
// cached plain-int4 repack of each Section-A weight (keyed by host/UVM pointer).
struct DevWeightQ {
  signed char *plain = nullptr; // plain row-major int4 [N, ceil(K/2)]
  int *rowsum = nullptr;        // per-channel sum of signed int4 [N]
};
std::unordered_map<const void *, DevWeightQ> g_dp4a_plain_cache;
// int8-unpacked weight [K,N] + per-channel rowsum, for the cuBLAS int8 path
// (keyed by the Section-A host/UVM pointer; unpacked once, weights are static).
struct DevWeightI8 {
  signed char *w8 = nullptr; // int8 weight [K,N] (w8[k*N+n] = int4(n,k))
  int *rowsum = nullptr;     // per-channel sum of int8 weight [N]
};
std::unordered_map<const void *, DevWeightI8> g_i8_weight_cache;
int *g_i8_c = nullptr; // int32 GEMM output scratch [M,N]
size_t g_i8_c_cap = 0;
signed char *g_dp4a_q8 = nullptr;
size_t g_dp4a_q8_cap = 0;
float *g_dp4a_ascale = nullptr; // per-row recip (dequant scale)
size_t g_dp4a_ascale_cap = 0;
int *g_dp4a_azp = nullptr; // per-row activation zero-point
size_t g_dp4a_azp_cap = 0;
float *g_dp4a_yf = nullptr; // float Y staging for the fp16-output path
size_t g_dp4a_yf_cap = 0;
float *g_dp4a_xf = nullptr; // float X staging for the naive fp16 path
size_t g_dp4a_xf_cap = 0;
// fp16 X staging for a HOST-resident input on the device GPU qint4 path: when
// the FC input pointer is host memory (e.g. NNTR_CUDA_M2B feeds the token via
// pinned host memory), the fp16 dp4a/cublas kernels still need a device X. The
// M*K fp16 input is copied H2D into this buffer and the device pointer is used
// instead of falling to the i8mm host dot (which SIGILLs on Orin).
unsigned short *g_stage_xh = nullptr;
size_t g_stage_xh_cap = 0;
std::mutex g_dp4a_mtx;

// repack (cached) + GEMM into a device float Y, using the already-staged
// q8/ascale scratch. Caller holds g_dp4a_mtx and has run act-quant.
bool dp4a_repack_and_gemm(const unsigned char *section_a,
                          const unsigned short *scales_fp16, float *Yf,
                          unsigned int M, unsigned int N, unsigned int K,
                          int out_fp16 = 0) {
  const int n = (int)N, k = (int)K;
  const int k_internal = (int)(((K + 31u) / 32u) * 32u);
  const size_t Kh = (K + 1u) / 2u;
  auto kr = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "repack_seca_i4");
  auto krs = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "weight_rowsum");
  const bool gemv = (M == 1);
  const bool tiled = (M >= 8);
  auto kg = CudaContext::Global().registerCudaKernel(
    FC_QINT4_DP4A_SRC,
    gemv ? "dp4a_gemv" : (tiled ? "dp4a_gemm_reg" : "dp4a_gemm"));
  if (!kr || !krs || !kg)
    return false;

  auto it = g_dp4a_plain_cache.find(section_a);
  if (it == g_dp4a_plain_cache.end()) {
    DevWeightQ dw;
    if (cudaMalloc(&dw.plain, (size_t)N * Kh) != cudaSuccess)
      return false;
    if (cudaMalloc(&dw.rowsum, sizeof(int) * (size_t)N) != cudaSuccess) {
      cudaFree(dw.plain);
      return false;
    }
    kr->SetKernelArguments(0, &section_a, sizeof(section_a));
    kr->SetKernelArguments(1, &dw.plain, sizeof(dw.plain));
    kr->SetKernelArguments(2, &n, sizeof(n));
    kr->SetKernelArguments(3, &k, sizeof(k));
    kr->SetKernelArguments(4, &k_internal, sizeof(k_internal));
    const int rb[3] = {16, 16, 1};
    const int rg[3] = {((int)Kh + 15) / 16, ((int)N + 15) / 16, 1};
    if (!StreamManager::Global().DispatchCommand(*kr, rg, rb)) {
      cudaFree(dw.plain);
      cudaFree(dw.rowsum);
      return false;
    }
    // per-channel weight row-sum (for the activation zero-point correction).
    krs->SetKernelArguments(0, &dw.plain, sizeof(dw.plain));
    krs->SetKernelArguments(1, &dw.rowsum, sizeof(dw.rowsum));
    krs->SetKernelArguments(2, &n, sizeof(n));
    krs->SetKernelArguments(3, &k, sizeof(k));
    const int sb[3] = {128, 1, 1};
    const int sg[3] = {((int)N + 127) / 128, 1, 1};
    if (!StreamManager::Global().DispatchCommand(*krs, sg, sb)) {
      cudaFree(dw.plain);
      cudaFree(dw.rowsum);
      return false;
    }
    it = g_dp4a_plain_cache.emplace(section_a, dw).first;
  }
  signed char *plain = it->second.plain;
  int *wrowsum = it->second.rowsum;

  const int mm = (int)M;
  kg->SetKernelArguments(0, &g_dp4a_q8, sizeof(g_dp4a_q8));
  kg->SetKernelArguments(1, &plain, sizeof(plain));
  kg->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kg->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
  kg->SetKernelArguments(4, &wrowsum, sizeof(wrowsum));
  kg->SetKernelArguments(5, &scales_fp16, sizeof(scales_fp16));
  kg->SetKernelArguments(6, &Yf, sizeof(Yf));
  if (gemv) {
    // dp4a_gemv: one WARP per output, 4 warps (128 threads) per block -> ceil(N/4)
    // blocks instead of N (4x fewer per-block launch/epilogue overheads).
    kg->SetKernelArguments(7, &n, sizeof(n));
    kg->SetKernelArguments(8, &k, sizeof(k));
    kg->SetKernelArguments(9, &out_fp16, sizeof(out_fp16));
    const int gvb[3] = {128, 1, 1};
    const int gvg[3] = {((int)N + 3) / 4, 1, 1};
    return StreamManager::Global().DispatchCommand(*kg, gvg, gvb);
  }
  kg->SetKernelArguments(7, &mm, sizeof(mm));
  kg->SetKernelArguments(8, &n, sizeof(n));
  kg->SetKernelArguments(9, &k, sizeof(k));
  kg->SetKernelArguments(10, &out_fp16, sizeof(out_fp16));
  const int gb[3] = {16, 16, 1};
  const int tile = tiled ? 64 : 16;
  const int gg[3] = {((int)N + tile - 1) / tile, ((int)M + tile - 1) / tile, 1};
  return StreamManager::Global().DispatchCommand(*kg, gg, gb);
}

bool ensure_buf(void **buf, size_t *cap, size_t bytes) {
  if (bytes <= *cap)
    return true;
  // cudaMalloc/cudaFree inside a CUDA-graph stream capture invalidates the
  // capture. The dp4a decode scratch is pre-grown at load by
  // cuda_fc_qint4_dp4a_prewarm() so this branch must not run under capture; if
  // it ever would (an under-sized prewarm), bail so the caller falls back
  // rather than corrupting the graph.
  if (StreamManager::Global().isCapturing())
    return false;
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

// stage q8 + ascale + azp scratch (caller holds the mutex). False on OOM.
// +256B tail pad on the int8 activation: the cuBLAS int8 IMMA GEMM reads A with
// wide vectorized (>=16B) Tensor-Core loads that can run past the last real
// element; an exactly-sized buffer (esp. large K=6144 down-proj) then faults
// with cudaErrorIllegalAddress. The pad keeps those reads in mapped memory.
static constexpr size_t FC_I8_TAIL_PAD = 256;
static bool dp4a_stage_scratch(unsigned int M, unsigned int K) {
  return ensure_buf((void **)&g_dp4a_q8, &g_dp4a_q8_cap,
                    (size_t)M * K + FC_I8_TAIL_PAD) &&
         ensure_buf((void **)&g_dp4a_ascale, &g_dp4a_ascale_cap,
                    sizeof(float) * (size_t)M) &&
         ensure_buf((void **)&g_dp4a_azp, &g_dp4a_azp_cap,
                    sizeof(int) * (size_t)M);
}

// Pre-grow ALL the static dp4a decode scratch buffers to the model's max decode
// capacity at load. The M=1 dp4a decode FC path is reached under graph capture
// once NNTR_CUDA_GRAPH is on; a cudaMalloc/Free inside
// cudaStreamBeginCapture..EndCapture invalidates the capture and surfaces as
// "NvMapMemAllocInternalTagged failed: error 12". Warming here (before any
// capture) makes every captured ensure_buf a pure cap-hit, so the dp4a path
// stays usable under the graph. ensure_buf's isCapturing() guard is the safety
// net if a model exceeds these bounds. Idempotent (cap check). False on OOM.
//
// maxM   max decode token rows (1 for decode; larger is a harmless over-grow)
// maxK   max FC input dim  (hidden DIM; covers every decode FC's K)
// maxN   max FC output dim (max(vocab, intermediate); covers lm_head + FFN)
bool cuda_fc_qint4_dp4a_prewarm(unsigned int maxM, unsigned int maxK,
                                unsigned int maxN) {
  if (maxM == 0 || maxK == 0 || maxN == 0)
    return true;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // q8/ascale/azp staging: exact sizes dp4a_stage_scratch() computes.
  if (!dp4a_stage_scratch(maxM, maxK))
    return false;
  // float X/Y staging: exact sizes the fp16-naive / fp32-resident paths use
  // (g_dp4a_xf = M*K floats, g_dp4a_yf = M*N floats). yf is grown to maxM*maxN
  // so the largest decode FC (lm_head N = vocab) is covered.
  // g_stage_xh (M*K fp16) covers the host-resident-input staging on the fp16 GPU
  // path, so that copy is also a pure cap-hit under capture.
  return ensure_buf((void **)&g_dp4a_xf, &g_dp4a_xf_cap,
                    sizeof(float) * (size_t)maxM * maxK) &&
         ensure_buf((void **)&g_dp4a_yf, &g_dp4a_yf_cap,
                    sizeof(float) * (size_t)maxM * maxN) &&
         ensure_buf((void **)&g_stage_xh, &g_stage_xh_cap,
                    sizeof(unsigned short) * (size_t)maxM * maxK);
}

// Stage a HOST-resident M*K fp16 activation into a device buffer for the fp16
// GPU qint4 path. Copies host_Xh H2D (async, on the backend stream so it is
// ordered before the kernels that read it) into the reusable g_stage_xh buffer
// and returns the device pointer. Returns nullptr if the buffer can't be grown
// (OOM, or a capture before prewarm sized it) so the caller falls back to the
// host path. The copy is enqueued on the backend stream; the subsequent kernel
// launch on the same stream sees it complete.
const unsigned short *cuda_fc_qint4_stage_host_x_fp16(const unsigned short *host_Xh,
                                                      unsigned int M,
                                                      unsigned int K) {
  if (host_Xh == nullptr || M == 0 || K == 0)
    return nullptr;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  const size_t bytes = sizeof(unsigned short) * (size_t)M * K;
  if (!ensure_buf((void **)&g_stage_xh, &g_stage_xh_cap, bytes))
    return nullptr; // OOM, or capturing before the buffer was prewarmed
  if (cudaMemcpyAsync(g_stage_xh, host_Xh, bytes, cudaMemcpyHostToDevice,
                      StreamManager::Global().GetStream()) != cudaSuccess) {
    cudaGetLastError();
    return nullptr;
  }
  return g_stage_xh;
}

// Stage a HOST-resident QINT4 section-A weight + scales into cached device
// buffers and return the device pointers. A model-load timing race can leave a
// weight in plain host memory (cudaPointerGetAttributes type Unregistered)
// instead of the managed pool; the dp4a repack kernel reads section_a on the
// GPU, so a host pointer makes the cudaFcGemm gate fall to the i8mm host dot,
// which SIGILLs on Orin (no i8mm). Reuses g_qint4_weight_cache (weights are
// constant, uploaded once, keyed by host section_a). Uploads happen on the
// first/prefill forward, NOT under graph capture; bails if asked to allocate
// under capture so the caller can fall back. Returns false on failure.
bool cuda_fc_qint4_stage_host_weight(const unsigned char *host_secA,
                                     const unsigned short *host_scales,
                                     unsigned int N, unsigned int K,
                                     const unsigned char **dev_secA,
                                     const unsigned short **dev_scales) {
  if (host_secA == nullptr || host_scales == nullptr || N == 0 || K == 0)
    return false;
  std::lock_guard<std::mutex> lk(g_qint4_mtx);
  auto it = g_qint4_weight_cache.find(host_secA);
  if (it == g_qint4_weight_cache.end()) {
    if (StreamManager::Global().isCapturing())
      return false;
    const unsigned k_internal = ((K + 31u) / 32u) * 32u;
    const size_t secA_bytes =
      (size_t)(((N + 3u) / 4u) * 4u) * (k_internal / 2u);
    DevWeight dw;
    if (cudaMalloc(&dw.d_secA, secA_bytes) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    if (cudaMalloc(&dw.d_sc, sizeof(unsigned short) * (size_t)N) !=
        cudaSuccess) {
      cudaFree(dw.d_secA);
      cudaGetLastError();
      return false;
    }
    cudaMemcpy(dw.d_secA, host_secA, secA_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dw.d_sc, host_scales, sizeof(unsigned short) * (size_t)N,
               cudaMemcpyHostToDevice);
    it = g_qint4_weight_cache.emplace(host_secA, dw).first;
  }
  *dev_secA = it->second.d_secA;
  *dev_scales = it->second.d_sc;
  return true;
}

// CPU mirror of the device seca_decode: signed int4 weight for (output channel
// n, input k) from the KAI Section-A payload. Must match the GPU exactly so the
// prewarmed cache is bit-identical to the repack_seca_i4 kernel.
static inline int seca_decode_cpu(const unsigned char *secA, int n, int k,
                                  int k_internal) {
  int a = k >> 5, r = k & 31;
  int is_high = (r >= 16) ? 1 : 0;
  int kb = is_high ? (r - 16) : r;
  int t = (a << 4) + kb;
  int nr_idx = n & 3;
  long byte_off = (long)(n >> 2) * (2 * k_internal) +
                  (((t >> 3) << 2) + nr_idx) * 8 + (t & 7);
  unsigned char s = secA[byte_off];
  int nib = is_high ? ((s >> 4) & 0xF) : (s & 0xF);
  return (nib ^ 8) - 8;
}

// Prewarm the dp4a plain-int4 weight cache on the CPU at LOAD (nntrainer
// ThreadManager-parallel), so the first inference does not pay the one-time
// Section-A -> plain int4 repack (nsys: ~38% of the cold-run GPU time) and the
// GPU is free of it. Mirrors repack_seca_i4 + weight_rowsum bit-exactly, then
// uploads the plain int4 + per-channel rowsum to the device cache (keyed by the
// Section-A pointer, same key the dp4a path looks up at forward). Idempotent.
bool cuda_fc_qint4_prewarm(const unsigned char *section_a, unsigned int N,
                           unsigned int K) {
  if (section_a == nullptr || N == 0 || K == 0)
    return true;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  if (g_dp4a_plain_cache.count(section_a))
    return true; // already cached
  const int k_internal = (int)(((K + 31u) / 32u) * 32u);
  const size_t Kh = (K + 1u) / 2u;
  std::vector<signed char> plain((size_t)N * Kh);
  std::vector<int> rowsum(N, 0);
  auto &tm = nntrainer::ThreadManager::Global();
  tm.parallel_for(0, (size_t)N, [&](size_t n) {
    signed char *prow = plain.data() + n * Kh;
    long acc = 0;
    for (size_t kb = 0; kb < Kh; ++kb) {
      int k0 = 2 * (int)kb, k1 = k0 + 1;
      int v0 = seca_decode_cpu(section_a, (int)n, k0, k_internal);
      int v1 =
        (k1 < (int)K) ? seca_decode_cpu(section_a, (int)n, k1, k_internal) : 0;
      prow[kb] = (signed char)((v0 & 0xF) | ((v1 & 0xF) << 4));
      acc += v0 + ((k1 < (int)K) ? v1 : 0);
    }
    rowsum[n] = (int)acc;
  });
  DevWeightQ dw;
  if (cudaMalloc(&dw.plain, (size_t)N * Kh) != cudaSuccess)
    return false;
  if (cudaMalloc(&dw.rowsum, sizeof(int) * (size_t)N) != cudaSuccess) {
    cudaFree(dw.plain);
    return false;
  }
  cudaMemcpy(dw.plain, plain.data(), (size_t)N * Kh, cudaMemcpyHostToDevice);
  cudaMemcpy(dw.rowsum, rowsum.data(), sizeof(int) * (size_t)N,
             cudaMemcpyHostToDevice);
  g_dp4a_plain_cache.emplace(section_a, dw);

  // Also prewarm the cuBLAS int8 [K,N] weight cache when the cuBLAS prefill FC
  // path is on: otherwise its one-time GPU repack (repack_seca_i8_kn, ~32% of
  // cold prefill GPU time) runs on the first prefill instead of at load. Mirrors
  // repack_seca_i8_kn (w8[k*N+n]=int4(n,k)) + weight_rowsum_kn bit-exactly.
  static const char *_cb = std::getenv("NNTR_FC_CUDA_CUBLAS");
  if (_cb && _cb[0] != '0' && !g_i8_weight_cache.count(section_a)) {
    std::vector<signed char> w8((size_t)K * N);
    std::vector<int> rs8(N, 0);
    tm.parallel_for(0, (size_t)N, [&](size_t n) {
      long acc = 0;
      for (int kk = 0; kk < (int)K; ++kk) {
        int v = seca_decode_cpu(section_a, (int)n, kk, k_internal);
        w8[(long)kk * N + n] = (signed char)v;
        acc += v;
      }
      rs8[n] = (int)acc;
    });
    DevWeightI8 dw8;
    if (cudaMalloc(&dw8.w8, (size_t)K * N) == cudaSuccess &&
        cudaMalloc(&dw8.rowsum, sizeof(int) * (size_t)N) == cudaSuccess) {
      cudaMemcpy(dw8.w8, w8.data(), (size_t)K * N, cudaMemcpyHostToDevice);
      cudaMemcpy(dw8.rowsum, rs8.data(), sizeof(int) * (size_t)N,
                 cudaMemcpyHostToDevice);
      g_i8_weight_cache.emplace(section_a, dw8);
    } else if (dw8.w8) {
      cudaFree(dw8.w8);
    }
  }
  return true;
}

bool cuda_fc_qint4_sectionA_dp4a_gemm_fp32(const float *X,
                                           const unsigned char *section_a,
                                           const unsigned short *scales_fp16,
                                           float *Y, unsigned int M,
                                           unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  auto kq = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "act_quant_i8");
  if (!kq) {
    ml_loge("[CUDA] fc_qint4 dp4a: kernel registration failed");
    return false;
  }
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  if (!dp4a_stage_scratch(M, K))
    return false;
  int m = (int)M, k = (int)K;
  kq->SetKernelArguments(0, &X, sizeof(X));
  kq->SetKernelArguments(1, &g_dp4a_q8, sizeof(g_dp4a_q8));
  kq->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kq->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
  kq->SetKernelArguments(4, &m, sizeof(m));
  kq->SetKernelArguments(5, &k, sizeof(k));
  const int qb[3] = {256, 1, 1};
  const int qg[3] = {(int)M, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kq, qg, qb))
    return false;
  if (!dp4a_repack_and_gemm(section_a, scales_fp16, Y, M, N, K))
    return false;
  maybe_finish();
  return true;
}

bool cuda_fc_qint4_sectionA_dp4a_gemm_fp16(const unsigned short *Xh,
                                           const unsigned char *section_a,
                                           const unsigned short *scales_fp16,
                                           unsigned short *Yh, unsigned int M,
                                           unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  auto kqh = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "act_quant_i8_h");
  auto kc =
    CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC, "cvt_f2h");
  if (!kqh || !kc) {
    ml_loge("[CUDA] fc_qint4 dp4a fp16: kernel registration failed");
    return false;
  }
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // No float Y staging here: the GEMM writes fp16 directly (out_fp16=1 below),
  // so g_dp4a_yf is unused on this path. Allocating it lazily would cudaMalloc
  // inside a CUDA-graph capture (NNTR_CUDA_GRAPH) on the first captured decode
  // token and invalidate the graph -- so it is deliberately NOT sized here.
  if (!dp4a_stage_scratch(M, K))
    return false;
  int m = (int)M, k = (int)K;
  // 1) int8 activation quant from the fp16 input.
  kqh->SetKernelArguments(0, &Xh, sizeof(Xh));
  kqh->SetKernelArguments(1, &g_dp4a_q8, sizeof(g_dp4a_q8));
  kqh->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kqh->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
  kqh->SetKernelArguments(4, &m, sizeof(m));
  kqh->SetKernelArguments(5, &k, sizeof(k));
  const int qb[3] = {256, 1, 1};
  const int qg[3] = {(int)M, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kqh, qg, qb))
    return false;
  // 2) repack + GEMM writing fp16 directly: the float->fp16 conversion is folded
  // into the GEMM epilogue (out_fp16=1), removing the separate cvt_f2h kernel +
  // the FP32 staging buffer (one fewer kernel per FC -- a decode launch-overhead
  // win). (void)kc keeps the registration check above harmless.
  (void)kc;
  if (!dp4a_repack_and_gemm(section_a, scales_fp16,
                            reinterpret_cast<float *>(Yh), M, N, K,
                            /*out_fp16=*/1))
    return false;
  maybe_finish();
  return true;
}

// w4a8 on the INT8 Tensor Cores via cuBLAS (prefill FC). Same quant scheme as
// the dp4a path -- per-row asym int8 activation + symmetric int4 weight -- but
// the int8xint8->int32 GEMM runs on IMMA Tensor Cores instead of __dp4a on the
// int ALU (~10x the GEMM throughput at prefill M). The int32 accumulate is
// exact so the result is bit-identical to dp4a; the int4->int8 weight unpack is
// cached (one-time) to keep it off the per-call critical path.
bool cuda_fc_qint4_sectionA_cublas_i8_gemm_fp16(
  const unsigned short *Xh, const unsigned char *section_a,
  const unsigned short *scales_fp16, unsigned short *Yh, unsigned int M,
  unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  auto kqh = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "act_quant_i8_h");
  auto krp = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "repack_seca_i8_kn");
  auto krs = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "weight_rowsum_kn");
  auto kde = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "dequant_i32_fp16");
  if (!kqh || !krp || !krs || !kde) {
    ml_loge("[CUDA] fc_qint4 cublas-i8: kernel registration failed");
    return false;
  }
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // cuBLAS int8 IMMA requires the GEMM dims to be multiples of 32 (measured:
  // M=260/272 -> CUBLAS_STATUS_NOT_SUPPORTED, 256/320/512 OK). The prefill token
  // count M is arbitrary (e.g. 511), so pad the activation row count up to a
  // multiple of 32 for the GEMM only -- the extra rows are computed from
  // (harmless int8) scratch and ignored by the epilogue, which writes just the
  // real M rows. N and K are multiples of 32 by the load invariant.
  const unsigned Mpad = ((M + 31u) / 32u) * 32u;
  if (!dp4a_stage_scratch(Mpad, K))
    return false;
  const int m = (int)M, n = (int)N, k = (int)K, mpad = (int)Mpad;
  const int k_internal = (int)(((K + 31u) / 32u) * 32u);

  // 1) int8 activation quant from the fp16 input (reuse the dp4a quantizer).
  kqh->SetKernelArguments(0, &Xh, sizeof(Xh));
  kqh->SetKernelArguments(1, &g_dp4a_q8, sizeof(g_dp4a_q8));
  kqh->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kqh->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
  kqh->SetKernelArguments(4, &m, sizeof(m));
  kqh->SetKernelArguments(5, &k, sizeof(k));
  const int qb[3] = {256, 1, 1};
  const int qg[3] = {(int)M, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kqh, qg, qb))
    return false;

  // 2) int8 weight [K,N] + per-channel rowsum (cached one-time unpack).
  auto it = g_i8_weight_cache.find(section_a);
  if (it == g_i8_weight_cache.end()) {
    DevWeightI8 dw;
    if (cudaMalloc(&dw.w8, (size_t)N * K + FC_I8_TAIL_PAD) != cudaSuccess)
      return false;
    if (cudaMalloc(&dw.rowsum, sizeof(int) * (size_t)N) != cudaSuccess) {
      cudaFree(dw.w8);
      return false;
    }
    krp->SetKernelArguments(0, &section_a, sizeof(section_a));
    krp->SetKernelArguments(1, &dw.w8, sizeof(dw.w8));
    krp->SetKernelArguments(2, &n, sizeof(n));
    krp->SetKernelArguments(3, &k, sizeof(k));
    krp->SetKernelArguments(4, &k_internal, sizeof(k_internal));
    const int pb[3] = {16, 16, 1};
    const int pg[3] = {((int)N + 15) / 16, ((int)K + 15) / 16, 1};
    if (!StreamManager::Global().DispatchCommand(*krp, pg, pb)) {
      cudaFree(dw.w8);
      cudaFree(dw.rowsum);
      return false;
    }
    krs->SetKernelArguments(0, &dw.w8, sizeof(dw.w8));
    krs->SetKernelArguments(1, &dw.rowsum, sizeof(dw.rowsum));
    krs->SetKernelArguments(2, &n, sizeof(n));
    krs->SetKernelArguments(3, &k, sizeof(k));
    const int sb[3] = {128, 1, 1};
    const int sg[3] = {((int)N + 127) / 128, 1, 1};
    if (!StreamManager::Global().DispatchCommand(*krs, sg, sb)) {
      cudaFree(dw.w8);
      cudaFree(dw.rowsum);
      return false;
    }
    it = g_i8_weight_cache.emplace(section_a, dw).first;
  }

  // 3) int32 GEMM output scratch [Mpad,N] (+tail pad: IMMA can write/read C in
  // wide vectorized tiles past the last element on large shapes).
  if (!ensure_buf((void **)&g_i8_c, &g_i8_c_cap,
                  sizeof(int) * (size_t)Mpad * N + FC_I8_TAIL_PAD))
    return false;

  // 4) INT8 IMMA GEMM on the Tensor Cores (Mpad rows; same backend stream as
  // the kernels). C is [Mpad,N] row-major; the real M rows are at the same
  // offsets so the epilogue reads C[m*N+n] for m<M directly.
  if (!BlasManager::Global().igemmRowMajor(mpad, n, k, g_dp4a_q8, it->second.w8,
                                           g_i8_c))
    return false;

  // 5) dequant epilogue (bit-identical math to the dp4a kernel) -> fp16 Y.
  int *rowsum = it->second.rowsum;
  kde->SetKernelArguments(0, &g_i8_c, sizeof(g_i8_c));
  kde->SetKernelArguments(1, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kde->SetKernelArguments(2, &g_dp4a_azp, sizeof(g_dp4a_azp));
  kde->SetKernelArguments(3, &rowsum, sizeof(rowsum));
  kde->SetKernelArguments(4, &scales_fp16, sizeof(scales_fp16));
  kde->SetKernelArguments(5, &Yh, sizeof(Yh));
  kde->SetKernelArguments(6, &m, sizeof(m));
  kde->SetKernelArguments(7, &n, sizeof(n));
  const int db[3] = {16, 16, 1};
  const int dg[3] = {((int)N + 15) / 16, ((int)M + 15) / 16, 1};
  if (!StreamManager::Global().DispatchCommand(*kde, dg, db))
    return false;
  maybe_finish();
  // Catch an ASYNC failure in the cuBLAS IMMA GEMM / epilogue (the sync cuBLAS
  // status was already checked). On Orin a large-M IMMA can fault at runtime and
  // leave a STICKY cuda error -- which then makes the NEXT layer's
  // cudaPointerGetAttributes (rms_norm dev_ok gate) fail, dropping rms_norm to
  // its host path that reads device/managed activations under cMA=0 -> SIGSEGV.
  // Clearing + returning false makes the caller fall back to the (correct) dp4a
  // GEMM cleanly instead of corrupting the rest of the forward.
  {
    cudaError_t _e = cudaGetLastError();
    if (_e != cudaSuccess) {
      if (std::getenv("NNTR_IGEMM_DBG"))
        std::fprintf(stderr,
                     "[IGEMM] async error after GEMM M=%d N=%d K=%d: %s -> dp4a "
                     "fallback\n",
                     m, n, k, cudaGetErrorString(_e));
      return false;
    }
  }
  return true;
}

// Diagnostic / high-accuracy fp16 path: FP32-precision activation (no int8
// quant). fp16 -> fp32, naive Section-A FP32-act GEMM, fp32 -> fp16. Used when
// NNTR_FC_CUDA_DP4A=0 with an fp16 activation.
bool cuda_fc_qint4_sectionA_gemm_fp16_naive(const unsigned short *Xh,
                                            const unsigned char *section_a,
                                            const unsigned short *scales_fp16,
                                            unsigned short *Yh, unsigned int M,
                                            unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  auto kh2f = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                       "cvt_h2f");
  auto kf2h =
    CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC, "cvt_f2h");
  if (!kh2f || !kf2h)
    return false;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  const size_t xn = (size_t)M * K, yn = (size_t)M * N;
  if (!ensure_buf((void **)&g_dp4a_xf, &g_dp4a_xf_cap, sizeof(float) * xn) ||
      !ensure_buf((void **)&g_dp4a_yf, &g_dp4a_yf_cap, sizeof(float) * yn))
    return false;
  int xni = (int)xn, yni = (int)yn;
  const int cb[3] = {256, 1, 1};
  kh2f->SetKernelArguments(0, &Xh, sizeof(Xh));
  kh2f->SetKernelArguments(1, &g_dp4a_xf, sizeof(g_dp4a_xf));
  kh2f->SetKernelArguments(2, &xni, sizeof(xni));
  const int xg[3] = {((int)xn + 255) / 256, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kh2f, xg, cb))
    return false;
  // naive Section-A FP32-act GEMM (mutex-free; its own dispatch + finish).
  if (!cuda_fc_qint4_sectionA_gemm_fp32(g_dp4a_xf, section_a, scales_fp16,
                                        g_dp4a_yf, M, N, K))
    return false;
  kf2h->SetKernelArguments(0, &g_dp4a_yf, sizeof(g_dp4a_yf));
  kf2h->SetKernelArguments(1, &Yh, sizeof(Yh));
  kf2h->SetKernelArguments(2, &yni, sizeof(yni));
  const int yg[3] = {((int)yn + 255) / 256, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kf2h, yg, cb))
    return false;
  maybe_finish();
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
  maybe_finish();
  return true;
}

// ---------------------------------------------------------------------------
// Q6_K lm_head GEMV (port of the OpenCL kernel_mul_mv_q6_K_f32 = llama.cpp's
// mul_mv_q6_K). One CUDA block = N_SIMDGROUP(2) output rows x N_SIMDWIDTH(16)
// lanes; the 16 lanes split each 256-element super-block, accumulate over all
// blocks of the row, then reduce. Reads the FP16 hidden + (managed) Q6_K weight
// directly on the device and writes FP16 logits to the device output -- no host
// bounce, so it works under a device-only activation pool (NNTR_CUDA_DEV_ACT)
// where the host Q6_K GEMV would fault. gemma2/qwen3 keep the Q6_K lm_head;
// this is the GPU path gemma4 gets from its QINT4 untied lm_head.
static const char *Q6K_GEMV_SRC = R"CU(
#define QK_K 256
typedef unsigned char  u8;
typedef signed char    s8;
typedef unsigned short u16;

typedef struct { u8 ql[128]; u8 qh[64]; s8 scales[16]; u16 d; } block_q6_K;

__device__ __forceinline__ float h2f(u16 h) {
  unsigned int s = (unsigned int)(h & 0x8000) << 16;
  unsigned int e = (h >> 10) & 0x1F;
  unsigned int m = h & 0x3FF;
  unsigned int out;
  if (e == 0) {
    if (m == 0) { out = s; }
    else { e = 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF;
           out = s | ((e + 112) << 23) | (m << 13); }
  } else if (e == 31) { out = s | 0x7F800000u | (m << 13); }
  else { out = s | ((e + 112) << 23) | (m << 13); }
  return __int_as_float((int)out);
}

__device__ __forceinline__ u16 f2h(float f) {
  unsigned int x = (unsigned int)__float_as_int(f);
  unsigned int sign = (x >> 16) & 0x8000u;
  int exp = (int)((x >> 23) & 0xFF) - 127 + 15;
  unsigned int man = x & 0x7FFFFFu;
  if (exp <= 0) {
    if (exp < -10) return (u16)sign;
    man |= 0x800000u;
    unsigned int shift = (unsigned int)(14 - exp);
    unsigned int half = man >> shift;
    if ((man >> (shift - 1)) & 1u) half += 1; // round to nearest
    return (u16)(sign | half);
  } else if (exp >= 31) {
    return (u16)(sign | 0x7C00u);
  }
  unsigned int half = (unsigned int)(exp << 10) | (man >> 13);
  if ((man >> 12) & 1u) half += 1; // round to nearest
  return (u16)(sign | half);
}

extern "C" __global__ void q6k_gemv(const void *src0, const u16 *src1, u16 *dst,
                                    int ne00, int ne01) {
  const int N_SIMDWIDTH = 16;
  __shared__ float red[2][16];
  int nb = ne00 / QK_K;
  int row_group = threadIdx.x / N_SIMDWIDTH;
  int lane = threadIdx.x % N_SIMDWIDTH;
  int row = blockIdx.x * 2 + row_group;
  const block_q6_K *x = (const block_q6_K *)src0 + (long)row * nb;
  const u16 *yy = src1;
  u8 kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
  int tid = lane;
  int ip = tid / 8, il = tid % 8, l0 = 4 * il;
  int is = 8 * ip + l0 / 16;
  int y_offset = 128 * ip + l0;
  int q_offset_l = 64 * ip + l0;
  int q_offset_h = 32 * ip + l0;
  float sumf = 0.0f;
  if (row < ne01) {
    for (int i = 0; i < nb; i++) {
      const u8 *q1 = x[i].ql + q_offset_l;
      const u8 *q2 = q1 + QK_K / 8;
      const u8 *qh = x[i].qh + q_offset_h;
      const s8 *sc = x[i].scales + is;
      const u16 *y = yy + i * QK_K + y_offset;
      float dall = h2f(x[i].d);
      float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (int j = 0; j < 4; j++) {
        s0 += h2f(y[j + 0])  * ((float)((q1[j] & 0xF) | ((qh[j] & kmask1) << 4)) - 32.f);
        s1 += h2f(y[j + 32]) * ((float)((q2[j] & 0xF) | ((qh[j] & kmask2) << 2)) - 32.f);
        s2 += h2f(y[j + 64]) * ((float)((q1[j] >> 4)  | ((qh[j] & kmask3) >> 0)) - 32.f);
        s3 += h2f(y[j + 96]) * ((float)((q2[j] >> 4)  | ((qh[j] & kmask4) >> 2)) - 32.f);
      }
      sumf += dall * (s0 * sc[0] + s1 * sc[2] + s2 * sc[4] + s3 * sc[6]);
    }
  }
  red[row_group][lane] = sumf;
  __syncthreads();
  for (int off = N_SIMDWIDTH / 2; off > 0; off >>= 1) {
    if (lane < off) red[row_group][lane] += red[row_group][lane + off];
    __syncthreads();
  }
  if (lane == 0 && row < ne01)
    dst[row] = f2h(red[row_group][0]);
}
)CU";

bool lmhead_gemv_q6_k_cuda(const void *w_q6k_dev,
                           const unsigned short *hidden_fp16_dev,
                           unsigned short *logits_fp16_dev, int vocab,
                           int hidden) {
  if (vocab <= 0 || hidden <= 0 || (hidden % 256) != 0)
    return false;
  auto kernel = CudaContext::Global().registerCudaKernel(Q6K_GEMV_SRC, "q6k_gemv");
  if (!kernel)
    return false;
  kernel->SetKernelArguments(0, &w_q6k_dev, sizeof(w_q6k_dev));
  kernel->SetKernelArguments(1, &hidden_fp16_dev, sizeof(hidden_fp16_dev));
  kernel->SetKernelArguments(2, &logits_fp16_dev, sizeof(logits_fp16_dev));
  kernel->SetKernelArguments(3, &hidden, sizeof(hidden));
  kernel->SetKernelArguments(4, &vocab, sizeof(vocab));
  const int block[3] = {32, 1, 1};
  const int grid[3] = {(vocab + 1) / 2, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  maybe_finish();
  return true;
}

} // namespace nntrainer::cuda

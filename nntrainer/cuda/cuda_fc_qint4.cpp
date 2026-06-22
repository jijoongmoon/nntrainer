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

#include <cstdlib>
#include <mutex>
#include <unordered_map>

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
    return e != nullptr && e[0] == '1';
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
  Y[(long)m * N + n] =
    (float)(acc - azp[m] * wrowsum[n]) * ascale[m] * dp4a_h2f(wscale[n]);
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
    int zp = azp[row];
#pragma unroll
    for (int j = 0; j < RB_TN; j++) {
      int col = blockN + tx * RB_TN + j;
      if (col < N)
        Y[(long)row * N + col] = (float)(acc[i][j] - zp * wrowsum[col]) * as *
                                 dp4a_h2f(wscale[col]);
    }
  }
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
std::mutex g_dp4a_mtx;

// repack (cached) + GEMM into a device float Y, using the already-staged
// q8/ascale scratch. Caller holds g_dp4a_mtx and has run act-quant.
bool dp4a_repack_and_gemm(const unsigned char *section_a,
                          const unsigned short *scales_fp16, float *Yf,
                          unsigned int M, unsigned int N, unsigned int K) {
  const int n = (int)N, k = (int)K;
  const int k_internal = (int)(((K + 31u) / 32u) * 32u);
  const size_t Kh = (K + 1u) / 2u;
  auto kr = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "repack_seca_i4");
  auto krs = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "weight_rowsum");
  const bool tiled = (M >= 8);
  auto kg = CudaContext::Global().registerCudaKernel(
    FC_QINT4_DP4A_SRC, tiled ? "dp4a_gemm_reg" : "dp4a_gemm");
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
  kg->SetKernelArguments(7, &mm, sizeof(mm));
  kg->SetKernelArguments(8, &n, sizeof(n));
  kg->SetKernelArguments(9, &k, sizeof(k));
  const int gb[3] = {16, 16, 1};
  const int tile = tiled ? 64 : 16;
  const int gg[3] = {((int)N + tile - 1) / tile, ((int)M + tile - 1) / tile, 1};
  return StreamManager::Global().DispatchCommand(*kg, gg, gb);
}

bool ensure_buf(void **buf, size_t *cap, size_t bytes) {
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

// stage q8 + ascale + azp scratch (caller holds the mutex). False on OOM.
static bool dp4a_stage_scratch(unsigned int M, unsigned int K) {
  return ensure_buf((void **)&g_dp4a_q8, &g_dp4a_q8_cap, (size_t)M * K) &&
         ensure_buf((void **)&g_dp4a_ascale, &g_dp4a_ascale_cap,
                    sizeof(float) * (size_t)M) &&
         ensure_buf((void **)&g_dp4a_azp, &g_dp4a_azp_cap,
                    sizeof(int) * (size_t)M);
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
  const size_t yn = (size_t)M * N;
  if (!dp4a_stage_scratch(M, K) ||
      !ensure_buf((void **)&g_dp4a_yf, &g_dp4a_yf_cap, sizeof(float) * yn))
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
  // 2) repack + GEMM into the float staging buffer.
  if (!dp4a_repack_and_gemm(section_a, scales_fp16, g_dp4a_yf, M, N, K))
    return false;
  // 3) float -> fp16 output.
  int yni = (int)yn;
  kc->SetKernelArguments(0, &g_dp4a_yf, sizeof(g_dp4a_yf));
  kc->SetKernelArguments(1, &Yh, sizeof(Yh));
  kc->SetKernelArguments(2, &yni, sizeof(yni));
  const int cb[3] = {256, 1, 1};
  const int cg[3] = {((int)yn + 255) / 256, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kc, cg, cb))
    return false;
  maybe_finish();
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

} // namespace nntrainer::cuda

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_fc_qint4.cpp
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Fused QS4CX dequant-GEMM implementation (NVRTC kernel).
 */

#include "cuda_fc_qint4.h"
#include "cuda_pack_cache.h"

#include <cuda_common.h> // cuda_vec4_rows_ok
#include <cuda_blas_manager.h>
#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

#include <cstdint>
#if defined(_WIN32)
#include <windows.h> // DiscardVirtualMemory
#else
#include <sys/mman.h> // madvise
#endif
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread_manager.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <fp16.h>

namespace nntrainer::cuda {

// One thread per output element Y[m,n]; loops K reading the int4 weight from
// the QS4CX PLAIN payload (row-major [N][Kh] bytes, even k = low nibble, stored
// uint4 = int4+8), dequantizing inline and scaling by the per-channel fp16
// scale. float accumulation. cvt_h2f / cvt_f2h are the fp16<->fp32 element
// converters the fp16-activation path stages through.
static const char *FC_QINT4_PLAIN_SRC = R"CU(
extern "C" {

__device__ __forceinline__ float plain_h2f(unsigned short h) {
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

__device__ __forceinline__ unsigned short plain_f2h(float f) {
  unsigned int x = (unsigned int)__float_as_int(f), s = (x >> 16) & 0x8000u,
               mant = x & 0x7FFFFFu;
  int e = (int)((x >> 23) & 0xFFu);
  if (e == 0xFF) return (unsigned short)(s | 0x7C00u | (mant ? 0x200u : 0u));
  int exp = e - 127 + 15;
  if (exp >= 0x1F) return (unsigned short)(s | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10) return (unsigned short)s;
    mant |= 0x800000u; int sh = 14 - exp;
    unsigned int hh = mant >> sh, rem = mant & ((1u << sh) - 1u),
                 half = 1u << (sh - 1);
    if (rem > half || (rem == half && (hh & 1u))) hh++;
    return (unsigned short)(s | hh);
  }
  unsigned int hh = ((unsigned int)exp << 10) | (mant >> 13), rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (hh & 1u))) hh++;
  return (unsigned short)(s | hh);
}

__global__ void fc_qint4_plain_gemm(const float *X, const unsigned char *W,
                                    const unsigned short *sc, float *Y, int M,
                                    int N, int K, int Kh) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  const unsigned char *wrow = W + (long)n * Kh;
  const float *xr = X + (long)m * K;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    unsigned char b = wrow[k >> 1];
    int nib = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
    acc += xr[k] * (float)(nib - 8);
  }
  Y[(long)m * N + n] = acc * plain_h2f(sc[n]);
}

__global__ void cvt_f2h(const float *src, unsigned short *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = plain_f2h(src[i]);
}

__global__ void cvt_h2f(const unsigned short *src, float *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = plain_h2f(src[i]);
}

}
)CU";

static const char *FC_QINT4_DP4A_SRC =
  R"CU(
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

// Hardware half<->float conversion, reachable from NVRTC without cuda_fp16.h.
//
// The scalar software routines above are ~20 integer ops each; the hardware
// instruction is one. On the DECODE row shapes (one block, a few thousand
// elements) that difference is the whole kernel: measured on RTX 5060, the
// row-at-a-time norm/quant kernels spend far more time converting than moving
// their few KB. Verified bit-identical to dp4a_h2f / dp4a_f2h over all 65536
// half patterns and 4M random floats, so the vectorized kernels below can use
// them without changing any value.
__device__ __forceinline__ float vq_h2f(unsigned short h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short vq_f2h(float f) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
// gamma is a weight at an arbitrary 2-byte offset in the model blob, so it is
// routinely NOT vector-aligned even when the activation rows are. Reading it
// with four scalar loads keeps the activation traffic vectorized instead of
// dropping the whole row to the scalar kernel. (has_gamma: 0 none, 1 vector,
// 2 scalar.)
__device__ __forceinline__ float4 vq_gather4(const unsigned short *g) {
  return make_float4(vq_h2f(g[0]), vq_h2f(g[1]), vq_h2f(g[2]), vq_h2f(g[3]));
}
__device__ __forceinline__ float4 vq_load4(uint2 r) {
  return make_float4(vq_h2f((unsigned short)(r.x & 0xFFFFu)),
                     vq_h2f((unsigned short)(r.x >> 16)),
                     vq_h2f((unsigned short)(r.y & 0xFFFFu)),
                     vq_h2f((unsigned short)(r.y >> 16)));
}
// Warp-shuffle reduce + one shared round over the warp results. blockDim.x
// must be a multiple of 32 and at most 1024. IDENT pads the lanes past the
// warp count in the final round, so it must be OP's identity.
#define VQ_REDUCE(scratch, val, OP, IDENT)                                     \
  do {                                                                         \
    for (int _o = 16; _o > 0; _o >>= 1)                                        \
      val = OP(val, __shfl_down_sync(0xffffffffu, val, _o));                   \
    if ((threadIdx.x & 31) == 0)                                               \
      scratch[threadIdx.x >> 5] = val;                                         \
    __syncthreads();                                                           \
    if (threadIdx.x < 32) {                                                    \
      float _a =                                                               \
        (threadIdx.x < (blockDim.x >> 5)) ? scratch[threadIdx.x] : (IDENT);    \
      for (int _o = 16; _o > 0; _o >>= 1)                                      \
        _a = OP(_a, __shfl_down_sync(0xffffffffu, _a, _o));                    \
      if (threadIdx.x == 0)                                                    \
        scratch[0] = _a;                                                       \
    }                                                                          \
    __syncthreads();                                                           \
  } while (0)
__device__ __forceinline__ float vq_add(float a, float b) { return a + b; }
#define VQ_POSINF __int_as_float(0x7F800000)
#define VQ_NEGINF __int_as_float((int)0xFF800000)

// Per-thread carry of the decoded row: with 4 halves per slot and 512 threads
// this covers K up to 16384 without a second global read; wider rows fall back
// to re-reading (still correct, just one more pass over an L1-hot row).
#define VQ_NCARRY 8

// Vectorized per-row asymmetric int8 activation quant. BIT-IDENTICAL to
// act_quant_i8_h: min/max are order-independent, the conversions are the same
// values, and the rint/clamp is unchanged.
__global__ void act_quant_i8_h_v4(const unsigned short *Xh, signed char *q8,
                                  float *ascale, int *azp, int M, int K) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  const uint2 *xv = (const uint2 *)(Xh + (long)m * K);
  int *q32 = (int *)(q8 + (long)m * K);
  const int nv = K >> 2;
  __shared__ float smn[32];
  __shared__ float smx[32];
  float lmn = 0.f, lmx = 0.f;
  float4 carry[VQ_NCARRY];
  int nc = 0;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 f = vq_load4(xv[i]);
    if (nc < VQ_NCARRY)
      carry[nc++] = f;
    lmn = fminf(lmn, fminf(fminf(f.x, f.y), fminf(f.z, f.w)));
    lmx = fmaxf(lmx, fmaxf(fmaxf(f.x, f.y), fmaxf(f.z, f.w)));
  }
  VQ_REDUCE(smn, lmn, fminf, VQ_POSINF);
  VQ_REDUCE(smx, lmx, fmaxf, VQ_NEGINF);
  float scale_q, recip;
  int zp;
  asym_qparams(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) {
    ascale[m] = recip;
    azp[m] = zp;
  }
  nc = 0;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 f = (nc < VQ_NCARRY) ? carry[nc++] : vq_load4(xv[i]);
    int q0 = max(-128, min(127, (int)rintf(f.x * scale_q) + zp));
    int q1 = max(-128, min(127, (int)rintf(f.y * scale_q) + zp));
    int q2 = max(-128, min(127, (int)rintf(f.z * scale_q) + zp));
    int q3 = max(-128, min(127, (int)rintf(f.w * scale_q) + zp));
    q32[i] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) |
             ((q3 & 0xFF) << 24);
  }
}

// Vectorized RMSNorm + int8 quant of the normed row (see rmsnorm_quant_i8_h
// below for the fusion rationale). The sum of squares is reduced in a
// different ORDER than the scalar kernels (vector-of-4 per thread, then warp
// shuffles), so `inv` can differ by an ulp -- the one place this lever is not
// bit-identical. Everything downstream of `inv` is.
__global__ void rmsnorm_quant_i8_h_v4(const unsigned short *x,
                                      const unsigned short *gamma,
                                      unsigned short *y, signed char *q8,
                                      float *ascale, int *azp, int M, int K,
                                      float eps, int has_gamma) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  const uint2 *xv = (const uint2 *)(x + (long)m * K);
  const uint2 *gv = (const uint2 *)gamma;
  uint2 *yv = (uint2 *)(y + (long)m * K);
  int *q32 = (int *)(q8 + (long)m * K);
  const int nv = K >> 2;
  __shared__ float ssq[32];
  __shared__ float smn[32];
  __shared__ float smx[32];
  float4 carry[VQ_NCARRY];
  int nc = 0;
  float p = 0.f;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 f = vq_load4(xv[i]);
    if (nc < VQ_NCARRY)
      carry[nc++] = f;
    p += f.x * f.x + f.y * f.y + f.z * f.z + f.w * f.w;
  }
  VQ_REDUCE(ssq, p, vq_add, 0.f);
  const float inv = rsqrtf(ssq[0] / (float)K + eps);

  float lmn = 0.f, lmx = 0.f;
  nc = 0;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    const int slot = (nc < VQ_NCARRY) ? nc++ : -1;
    float4 f = (slot >= 0) ? carry[slot] : vq_load4(xv[i]);
    float4 g = make_float4(1.f, 1.f, 1.f, 1.f);
    if (has_gamma == 1)
      g = vq_load4(gv[i]);
    else if (has_gamma == 2)
      g = vq_gather4(gamma + 4 * i);
    unsigned short h0 = vq_f2h(f.x * inv * g.x), h1 = vq_f2h(f.y * inv * g.y);
    unsigned short h2 = vq_f2h(f.z * inv * g.z), h3 = vq_f2h(f.w * inv * g.w);
    uint2 o;
    o.x = (unsigned int)h0 | ((unsigned int)h1 << 16);
    o.y = (unsigned int)h2 | ((unsigned int)h3 << 16);
    yv[i] = o;
    // Quantize the ROUNDED output, exactly what a following act_quant would
    // read back. The carry slot is recycled here: its input value is already
    // consumed for this element.
    float4 r = make_float4(vq_h2f(h0), vq_h2f(h1), vq_h2f(h2), vq_h2f(h3));
    if (slot >= 0)
      carry[slot] = r;
    lmn = fminf(lmn, fminf(fminf(r.x, r.y), fminf(r.z, r.w)));
    lmx = fmaxf(lmx, fmaxf(fmaxf(r.x, r.y), fmaxf(r.z, r.w)));
  }
  VQ_REDUCE(smn, lmn, fminf, VQ_POSINF);
  VQ_REDUCE(smx, lmx, fmaxf, VQ_NEGINF);
  float scale_q, recip;
  int zp;
  asym_qparams(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) {
    ascale[m] = recip;
    azp[m] = zp;
  }
  nc = 0;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 r = (nc < VQ_NCARRY) ? carry[nc++] : vq_load4(yv[i]);
    int q0 = max(-128, min(127, (int)rintf(r.x * scale_q) + zp));
    int q1 = max(-128, min(127, (int)rintf(r.y * scale_q) + zp));
    int q2 = max(-128, min(127, (int)rintf(r.z * scale_q) + zp));
    int q3 = max(-128, min(127, (int)rintf(r.w * scale_q) + zp));
    q32[i] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) |
             ((q3 & 0xFF) << 24);
  }
}

// RMSNorm fused with the int8 activation quant its consumer FC needs.
//
// The decode norm and the quant that follows it are two single-block kernels
// over the same 1..8K-element row: at decode M=1 each is far below the launch
// granularity of the GPU, so the pair costs about twice its own arithmetic.
// Folding them removes one node per (norm -> FC-group) pair from the decode
// graph.
//
// Deliberately BIT-IDENTICAL to rmsnorm_fp16 followed by act_quant_i8_h:
//   - phase 1 reduces the sum of squares with the SAME per-thread stride and
//     the SAME shared-memory pairing tree, so the fp32 accumulation order (and
//     therefore `inv`) is unchanged;
//   - phase 2 writes exactly rmsnorm_fp16's y, and tracks min/max of the
//     ROUNDED fp16 it just stored -- the very values act_quant_i8_h would read
//     back -- so the quant params come out of asym_qparams unchanged;
//   - phase 3 re-reads those stores (each thread reads only its own) and
//     applies the identical rint/clamp.
// The equality is what lets the fused path be the default with a plain
// killswitch: no golden movement to argue about.
__global__ void rmsnorm_quant_i8_h(const unsigned short *x,
                                   const unsigned short *gamma,
                                   unsigned short *y, signed char *q8,
                                   float *ascale, int *azp, int M, int K,
                                   float eps, int has_gamma) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  const unsigned short *xr = x + (long)m * K;
  unsigned short *yr = y + (long)m * K;
  __shared__ float sdata[256];
  __shared__ float smx[256];
  float partial = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = dp4a_h2f(xr[k]);
    partial += v * v;
  }
  sdata[threadIdx.x] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(sdata[0] / (float)K + eps);
  __syncthreads(); // sdata[0] consumed; the arrays are reused below

  float lmn = 0.f, lmx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float g = has_gamma ? dp4a_h2f(gamma[k]) : 1.0f;
    unsigned short h = dp4a_f2h(dp4a_h2f(xr[k]) * inv * g);
    yr[k] = h;
    float v = dp4a_h2f(h);
    lmn = fminf(lmn, v);
    lmx = fmaxf(lmx, v);
  }
  sdata[threadIdx.x] = lmn;
  smx[threadIdx.x] = lmx;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] = fminf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
      smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x + s]);
    }
    __syncthreads();
  }
  float scale_q, recip;
  int zp;
  asym_qparams(sdata[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) {
    ascale[m] = recip;
    azp[m] = zp;
  }
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = (int)rintf(dp4a_h2f(yr[k]) * scale_q) + zp;
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

// signed int4 weight for (output n, input k) from the QS4CX plain payload
// (row-major [N][Kh] bytes, even k = low nibble, stored uint4 = int4+8).
__device__ __forceinline__ int plain_decode(const unsigned char *qw, int n,
                                            int k, int Kh) {
  unsigned char b = qw[(long)n * Kh + (k >> 1)];
  int nib = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
  return nib - 8;
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

// QS4CX plain -> signed packed int4 [N, ceil(K/2)]: byte[n][kb] low nibble =
// int4(n, 2kb), high nibble = int4(n, 2kb+1), each stored two's-complement.
// The source has the SAME [N][Kh] byte indexing with uint4 = int4+8 nibbles,
// and (x-8)&0xF == x^8 on a 4-bit lane, so the whole repack is one byte-wise
// XOR with 0x88 (an odd-K pad nibble 8 becomes signed 0, as before).
__global__ void repack_plain_i4(const unsigned char *qw, signed char *packed,
                                int N, int Kh) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (long long)N * Kh)
    packed[i] = (signed char)(qw[i] ^ 0x88);
}

)CU"
  // NOTE: split here into two adjacent raw-string literals — MSVC caps a single
  // string literal at 16380 bytes (C2026); the two concatenate
  // byte-identically.
  R"CU(
// Y[m,n] = recip[m]*w_scale[n]*(sum_k q8[m,k]*int4(n,k) - zp[m]*rowsum_w[n]),
// the asymmetric-activation dequant (zp from act_quant, rowsum_w from the
// weight). via __dp4a.
// `raw` (launch-uniform): 1 = `plain` is the QS4CX payload itself, so the
// offset-binary bias is removed here with an XOR and rowsum is accumulated in
// this same pass instead of being read from the derived cache. See
// dp4a_gemv_raw for why that cache is worth not building.
__global__ void dp4a_gemm(const signed char *q8, const signed char *plain,
                          const float *ascale, const int *azp,
                          const int *wrowsum, const unsigned short *wscale,
                          float *Y, int M, int N, int K, int out_fp16,
                          int raw) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  int Kh = (K + 1) >> 1;
  const int xr = raw ? 0x88 : 0;
  const signed char *qrow = q8 + (long)m * K;
  const signed char *wrow = plain + (long)n * Kh;
  int acc = 0, rs = 0, k = 0;
  for (; k + 4 <= K; k += 4) {
    int a = *(const int *)(qrow + k); // lanes = act k,k+1,k+2,k+3
    int kb = k >> 1;
    int b0 = ((unsigned char)wrow[kb]) ^ xr;     // k(low), k+1(high)
    int b1 = ((unsigned char)wrow[kb + 1]) ^ xr; // k+2(low), k+3(high)
    int w0 = ((int)(signed char)(b0 << 4)) >> 4;
    int w1 = ((int)(signed char)b0) >> 4;
    int w2 = ((int)(signed char)(b1 << 4)) >> 4;
    int w3 = ((int)(signed char)b1) >> 4;
    int w = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) |
            ((w3 & 0xFF) << 24);
    acc = __dp4a(a, w, acc);
    rs = __dp4a(0x01010101, w, rs);
  }
  for (; k < K; ++k) { // tail (none when K%32==0)
    int kb = k >> 1;
    int b = ((unsigned char)wrow[kb]) ^ xr;
    int wv = (k & 1) ? (((int)(signed char)b) >> 4)
                     : (((int)(signed char)(b << 4)) >> 4);
    acc += (int)qrow[k] * wv;
    rs += wv;
  }
  const int rsum = raw ? rs : wrowsum[n];
  float r = (float)(acc - azp[m] * rsum) * ascale[m] * dp4a_h2f(wscale[n]);
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
  // K4 = the input channels covered by whole groups of 4; the dp4a loop can
  // only consume those. k is a multiple of 4 so kb is even, but the 2-byte
  // weight load also needs the ROW base aligned, and wrow = plain + n*Kh is
  // 2-byte aligned only when Kh is even. Odd Kh (K % 4 == 1 or 2) leaves every
  // odd n on an odd address, which aborts the launch with "misaligned address"
  // rather than just computing wrong -- so those shapes read the byte pair
  // directly. The predicate is launch-uniform and loop-invariant: no warp
  // divergence, and K % 4 == 0 (every LLM projection width) keeps the wide load.
  const int K4 = K & ~3;
  const bool wide_w = ((Kh & 1) == 0);
  for (int k = lane * 4; k < K4; k += 32 * 4) {
    int a = *(const int *)(q8 + k);
    int kb = k >> 1;
    int b0, b1;
    if (wide_w) {
      unsigned int w16 = *(const unsigned short *)(wrow + kb);
      b0 = w16 & 0xFF;
      b1 = (w16 >> 8) & 0xFF;
    } else {
      b0 = (unsigned char)wrow[kb];
      b1 = (unsigned char)wrow[kb + 1];
    }
    int w0 = ((int)(signed char)(b0 << 4)) >> 4;
    int w1 = ((int)(signed char)b0) >> 4;
    int w2 = ((int)(signed char)(b1 << 4)) >> 4;
    int w3 = ((int)(signed char)b1) >> 4;
    int w = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) |
            ((w3 & 0xFF) << 24);
    acc = __dp4a(a, w, acc);
  }
  // Scalar tail for K % 4 != 0: the loop above consumes whole groups of 4, so
  // without this the last 1..3 input channels are dropped from every output.
  // One channel per lane over lanes 0..(K-K4-1) -- each is added exactly once
  // by the warp reduction below. Same nibble decode as dp4a_gemm's own tail.
  {
    int k = K4 + lane;
    if (k < K) {
      int b = (unsigned char)wrow[k >> 1];
      int wv = (k & 1) ? (((int)(signed char)b) >> 4)
                       : (((int)(signed char)(b << 4)) >> 4);
      acc += (int)q8[k] * wv;
    }
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

// Same math as dp4a_gemv, reading the QS4CX payload DIRECTLY instead of the
// derived DevWeightQ cache. The two things that cache precomputes are done
// inline here, and both are nearly free on a bandwidth-bound kernel:
//
//   - offset-binary -> two's complement. repack_plain_i4 does this as
//     `byte ^ 0x88` into a SECOND FULL COPY of every weight -- 15.1 GiB for
//     this model's experts alone, to avoid one XOR per 16-bit load.
//   - rowsum[n] = sum_k int4(n,k). weight_rowsum makes a whole separate pass
//     over the same bytes this loop already has in registers; one extra
//     __dp4a against 0x01010101 produces it in place.
//
// Both accumulators are int32 and cover exactly k in [0,K) -- the same range
// weight_rowsum uses (it skips the odd-K pad nibble, and so does the tail
// below) -- so dp4a's integer associativity makes this BIT-IDENTICAL to the
// cached path, not merely equivalent.
__global__ void dp4a_gemv_raw(const signed char *q8, const unsigned char *plain,
                              const float *ascale, const int *azp,
                              const unsigned short *wscale, float *Y, int N,
                              int K, int out_fp16) {
  const int warps_per_block = blockDim.x >> 5;
  int n = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  if (n >= N)
    return;
  const int lane = threadIdx.x & 31;
  int Kh = (K + 1) >> 1;
  const unsigned char *wrow = plain + (long)n * Kh;
  int acc = 0, rs = 0;
  const int K4 = K & ~3;
  const bool wide_w = ((Kh & 1) == 0);
  for (int k = lane * 4; k < K4; k += 32 * 4) {
    int a = *(const int *)(q8 + k);
    int kb = k >> 1;
    int b0, b1;
    if (wide_w) {
      unsigned int w16 = (*(const unsigned short *)(wrow + kb)) ^ 0x8888u;
      b0 = (int)(w16 & 0xFF);
      b1 = (int)((w16 >> 8) & 0xFF);
    } else {
      b0 = (int)(((unsigned int)wrow[kb]) ^ 0x88u);
      b1 = (int)(((unsigned int)wrow[kb + 1]) ^ 0x88u);
    }
    int w0 = ((int)(signed char)(b0 << 4)) >> 4;
    int w1 = ((int)(signed char)b0) >> 4;
    int w2 = ((int)(signed char)(b1 << 4)) >> 4;
    int w3 = ((int)(signed char)b1) >> 4;
    int w = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) |
            ((w3 & 0xFF) << 24);
    acc = __dp4a(a, w, acc);
    rs = __dp4a(0x01010101, w, rs);
  }
  {
    int k = K4 + lane;
    if (k < K) {
      int b = (int)(((unsigned int)wrow[k >> 1]) ^ 0x88u);
      int wv = (k & 1) ? (((int)(signed char)b) >> 4)
                       : (((int)(signed char)(b << 4)) >> 4);
      acc += (int)q8[k] * wv;
      rs += wv;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, o);
    rs += __shfl_down_sync(0xffffffffu, rs, o);
  }
  if (lane == 0) {
    float r = (float)(acc - azp[0] * rs) * ascale[0] * dp4a_h2f(wscale[n]);
    if (out_fp16)
      ((unsigned short *)Y)[n] = dp4a_f2h(r);
    else
      Y[n] = r;
  }
}

// Fused decode GEMV: the per-row asym int8 activation quant folded into the
// GEMV kernel itself (the ML Drift paper's 3.7 decode prescription: quantize
// inside the operational kernel). Every block redundantly reduces the row's
// min/max -- fmin/fmax are order- and partition-independent, so scale/zp and
// the quantized bytes are BIT-IDENTICAL to the two-kernel (act-quant + GEMV)
// path, and the dp4a accumulate is integer -- the output matches exactly.
// The activation row (K fp16, a few KB) re-reads from L2 per block; the win
// is one launch (+ per-op drain) less per FC call and no q8/ascale/azp
// global round-trip. M==1 only; dynamic shared = K bytes for the q8 row.
// ------------------------------------------------------------------ IMMA --
// Same tile, same staging, Tensor Cores instead of the int ALU.
//
// dp4a_gemm_reg already reads the PACKED int4 weight and unpacks it into
// shared memory during staging, so nothing is ever materialised in global
// memory -- the property that makes vLLM's Marlin fast. What it lacks is the
// instruction: __dp4a is the int32 ALU at ~21 TOPS, while the same operands on
// mma.sync.s8 are ~137 TOPS. So only the inner product changes here; the loads
// above and the epilogue below are dp4a_gemm_reg's, untouched.
//
// That combination is strictly better than Marlin ON THIS HARDWARE: identical
// weight DRAM traffic (K*N/2), HALF the activation traffic (int8 vs bf16), and
// twice the math rate (IMMA s8 vs HMMA bf16).
//
// VALIDATION IS FREE: int32 accumulation is exact and associative, so a
// correct fragment mapping is BIT-IDENTICAL to dp4a_gemm_reg. Any layout error
// shows up immediately as wrong output; there is no silent-drift regime.
//
// Fragment layout for mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32, with
// g = lane>>2 (0..7) and t = lane&3 (0..3), all per warp:
//   A [16m x 32k] row-major, 4 regs: a0 = A[g][4t..+3], a1 = A[g+8][4t..+3],
//                                    a2 = A[g][16+4t..+3], a3 = A[g+8][16+4t..+3]
//   B [32k x 8n] col-major,  2 regs: b0 = B[4t..+3][g],   b1 = B[16+4t..+3][g]
//   C [16m x 8n],            4 regs: c0,c1 = C[g][2t],C[g][2t+1]
//                                    c2,c3 = C[g+8][2t], C[g+8][2t+1]
// Ws is [n][k], i.e. contiguous in k for a fixed n -- which IS the col-major B
// operand, so the B fragment is a straight 4-byte load with no transpose.
//
// 8 warps cover the 64x64 tile as 4 warp-rows (16 m each) x 2 warp-cols (32 n
// each); each warp runs 4 mma per k-step, one per 8-wide n subtile.
// Own tile constants: this kernel sits ahead of dp4a_gemm_reg's RB_* defines
// in the source string. Same values -- the tile is deliberately identical, so
// the two are directly comparable.
#define IM_BM 64
#define IM_BN 64
#define IM_BK 32
__global__ __launch_bounds__(256, 2)
void imma_gemm_reg(const signed char *q8, const unsigned char *plain,
                   const float *ascale, const int *azp, const int *wrowsum,
                   const unsigned short *wscale, float *Y, int M, int N, int K,
                   int out_fp16, int raw) {
  __shared__ signed char As[IM_BM][IM_BK];
  __shared__ signed char Ws[IM_BN][IM_BK];
  const int tid = threadIdx.x;          // 256 threads, 1-D
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1; // 4 x 2 warp grid over 64x64
  const int g = lane >> 2, t = lane & 3;
  const int blockM = blockIdx.y * IM_BM, blockN = blockIdx.x * IM_BN;
  const int Kh = (K + 1) >> 1;
  const int xr = raw ? 0x88 : 0;

  int acc[4][4]; // [n subtile][c reg]
#pragma unroll
  for (int s = 0; s < 4; ++s)
#pragma unroll
    for (int r = 0; r < 4; ++r)
      acc[s][r] = 0;
  int rs[4] = {0, 0, 0, 0}; // rowsum per n subtile, for the raw path

  for (int k0 = 0; k0 < K; k0 += IM_BK) {
    for (int q = tid; q < IM_BM * IM_BK; q += 256) {
      int i = q / IM_BK, j = q % IM_BK;
      int mm = blockM + i, kk = k0 + j;
      As[i][j] = (mm < M && kk < K) ? q8[(long)mm * K + kk] : (signed char)0;
    }
    for (int q = tid; q < IM_BN * IM_BK; q += 256) {
      int i = q / IM_BK, j = q % IM_BK;
      int nn = blockN + i, kk = k0 + j;
      signed char wv = 0;
      if (nn < N && kk < K) {
        unsigned char b =
          (unsigned char)(((unsigned int)plain[(long)nn * Kh + (kk >> 1)]) ^ xr);
        wv = (kk & 1) ? (((signed char)b) >> 4) : (((signed char)(b << 4)) >> 4);
      }
      Ws[i][j] = wv;
    }
    __syncthreads();

    int a[4];
    a[0] = *(const int *)&As[wm * 16 + g][4 * t];
    a[1] = *(const int *)&As[wm * 16 + g + 8][4 * t];
    a[2] = *(const int *)&As[wm * 16 + g][16 + 4 * t];
    a[3] = *(const int *)&As[wm * 16 + g + 8][16 + 4 * t];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int ncol = wn * 32 + s * 8 + g;
      int b[2];
      b[0] = *(const int *)&Ws[ncol][4 * t];
      b[1] = *(const int *)&Ws[ncol][16 + 4 * t];
      if (raw) {
        rs[s] = __dp4a(0x01010101, b[0], rs[s]);
        rs[s] = __dp4a(0x01010101, b[1], rs[s]);
      }
      asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(acc[s][0]), "=r"(acc[s][1]), "=r"(acc[s][2]), "=r"(acc[s][3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(acc[s][0]), "r"(acc[s][1]), "r"(acc[s][2]), "r"(acc[s][3]));
    }
    __syncthreads();
  }

  // Epilogue. Each lane owns C[g][2t], C[g][2t+1], C[g+8][2t], C[g+8][2t+1]
  // of every 8-wide n subtile. The rowsum a lane accumulated belongs to column
  // (wn*32 + s*8 + g), which is NOT one of the columns it writes, so it is
  // exchanged through shared memory rather than used directly.
  __shared__ int rsh[IM_BN];
  if (raw) {
    // Column ncol's rowsum is SPLIT across the four lanes that share g: lane
    // (g,t) saw only k in {4t..4t+3, 16+4t..+3}. Those four are consecutive
    // lanes, so a two-step shuffle-down leaves the full sum in t==0.
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      int v = rs[s];
      v += __shfl_down_sync(0xffffffffu, v, 1);
      v += __shfl_down_sync(0xffffffffu, v, 2);
      if (t == 0)
        rsh[wn * 32 + s * 8 + g] = v;
    }
    __syncthreads();
  }
#pragma unroll
  for (int s = 0; s < 4; ++s) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int row = blockM + wm * 16 + g + ((r >> 1) ? 8 : 0);
      const int col = blockN + wn * 32 + s * 8 + 2 * t + (r & 1);
      if (row < M && col < N) {
        const int rsum = raw ? rsh[wn * 32 + s * 8 + 2 * t + (r & 1)]
                             : wrowsum[col];
        float v = (float)(acc[s][r] - azp[row] * rsum) * ascale[row] *
                  dp4a_h2f(wscale[col]);
        if (out_fp16)
          ((unsigned short *)Y)[(long)row * N + col] = dp4a_f2h(v);
        else
          Y[(long)row * N + col] = v;
      }
    }
  }
}

// ------------------------------------------------------------- IMMA v2 ----
// imma_gemm_reg with the memory pipeline it never had. Same fragment layout
// (documented above), same bit-exact result -- what changes is everything
// AROUND the mma, because v1 spends ~80 instructions per k-step to feed 4 of
// them:
//
//  1. VECTOR STAGING. v1 stages 2 KB of A and 2 KB of W a BYTE at a time: 8
//     LDG.8 + 8 STS.8 per thread for A, and for W another 8 loads each with a
//     (kk&1) nibble select. Here a thread moves 16 B of A (one 16-byte load ->
//     one 16-byte store) and 8 B of packed W (one 8-byte load -> 16 nibbles
//     unpacked four at a time -> one 16-byte store): 2 loads, 2 stores, ~12
//     ALU for the whole k-step.
//
//  2. NIBBLE UNPACK, FOUR AT A TIME. __byte_perm interleaves the low and high
//     nibble planes, and the offset-binary fix is exact in plain 32-bit
//     integer ops -- no SIMD intrinsic, no inter-byte borrow:
//        ((x | 0x80808080) - 0x08080808) ^ 0x80808080  ==  per-byte (x - 8)
//     Every byte of (x|0x80) is >= 128, so subtracting 8 can never borrow into
//     the neighbour; the final XOR removes the bias. Verified both ways: for
//     x in [0,7] the sum lands in [120,127] (bit 7 clear, XOR adds 128 -> x-8
//     mod 256) and for x in [8,15] in [128,135] (bit 7 set, XOR subtracts 128
//     -> x-8). cx folds in the cached path's extra XOR, exactly as v1's xr.
//
//  3. DOUBLE-BUFFERED SHARED + REGISTER PREFETCH. v1 runs load / sync /
//     compute / sync, so the global load latency is fully exposed -- and with
//     8 warps on a 16-SM part there is no occupancy to hide it behind. Here
//     the load for tile k+1 is ISSUED BEFORE the mma work on tile k, so its
//     several-hundred-cycle latency overlaps the compute, and ONE barrier per
//     k-step separates "everyone finished reading buffer X" from "someone
//     writes buffer X".
//
//  4. BK 32 -> 64. Halves the barrier count and doubles the bytes in flight
//     per thread.
//
//  5. NO BANK CONFLICTS. Row stride is BK+16 = 80 B = 20 words. A fragment
//     load reads (row, 4t) for row = base+g, g in 0..7, t in 0..3, so the bank
//     is (row*20 + t) % 32 and 20*g mod 32 over g = 0..7 is 0,20,8,28,16,4,
//     24,12 -- eight distinct 4-bank groups that exactly partition the 32
//     banks. A stride of BK itself (16 words) collides 4 ways.
//
//  6. ROWSUM OFF THE MATH PATH. v1 derives the raw-path rowsum from the B
//     fragments inside the k loop, where all FOUR warp-rows redundantly
//     compute the same per-column sum. Here the staging thread owns the same
//     column for every k-step, so it accumulates privately and the four
//     threads of a column reduce once, at the end.
//
// Requires K % 64 == 0 (no k tail, and it makes every vector access aligned)
// and an 8-byte-aligned payload; the caller gates on both and falls back to
// imma_gemm_reg otherwise.
#define IP_BM 64
#define IP_BN 64
#define IP_BK 64
#define IP_LD 80 /* row stride in BYTES: BK + 16, see note 5 */
struct alignas(16) IPv16 {
  unsigned int a, b, c, d;
};
struct alignas(8) IPv8 {
  unsigned int a, b;
};
// per-byte (nibble - 8) with the cached path's two's-complement fix folded in
__device__ __forceinline__ unsigned int ip_nib2i8(unsigned int x,
                                                  unsigned int cx) {
  x ^= cx;
  return (((x | 0x80808080u) - 0x08080808u) ^ 0x80808080u);
}
__global__ __launch_bounds__(256, 2)
void imma_gemm_pipe(const signed char *q8, const unsigned char *plain,
                    const float *ascale, const int *azp, const int *wrowsum,
                    const unsigned short *wscale, float *Y, int M, int N, int K,
                    int out_fp16, int raw) {
  __shared__ IPv16 Asv[2][IP_BM * IP_LD / 16];
  __shared__ IPv16 Wsv[2][IP_BN * IP_LD / 16];
  __shared__ int rsh[IP_BN];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1; // 4 x 2 warp grid over 64x64
  const int g = lane >> 2, t = lane & 3;
  const int blockM = blockIdx.y * IP_BM, blockN = blockIdx.x * IP_BN;
  const int Kh = K >> 1;
  const unsigned int cx = raw ? 0u : 0x08080808u;

  // Staging: 4 threads per row, 16 bytes of k each. A thread keeps the SAME
  // row for every k-step, which is what lets the rowsum stay private (note 6).
  const int srow = tid >> 2, ssub = tid & 3;
  const int arow = blockM + srow, wrow = blockN + srow;
  const bool arow_ok = arow < M, wrow_ok = wrow < N;
  const signed char *aptr = q8 + (long)arow * K + (ssub << 4);
  const unsigned char *wptr = plain + (long)wrow * Kh + (ssub << 3);
  const int sdst = srow * IP_LD + (ssub << 4); // byte offset, 16-B aligned

  int acc[4][4];
#pragma unroll
  for (int s = 0; s < 4; ++s)
#pragma unroll
    for (int r = 0; r < 4; ++r)
      acc[s][r] = 0;
  int rsacc = 0;

  IPv16 ra;
  IPv8 rw;
  IPv16 wv;

#define IP_LOAD(K0)                                                            \
  do {                                                                         \
    if (arow_ok) {                                                             \
      ra = *(const IPv16 *)(aptr + (K0));                                      \
    } else {                                                                   \
      ra.a = ra.b = ra.c = ra.d = 0u;                                          \
    }                                                                          \
    if (wrow_ok) {                                                             \
      rw = *(const IPv8 *)(wptr + ((K0) >> 1));                                \
    } else {                                                                   \
      rw.a = rw.b = 0u;                                                        \
    }                                                                          \
  } while (0)

  // 8 payload bytes -> 16 int8. lo/hi are the even/odd nibble planes; the two
  // byte_perms re-interleave them into k order.
#define IP_STORE(BUF)                                                          \
  do {                                                                         \
    unsigned int lo = rw.a & 0x0F0F0F0Fu;                                      \
    unsigned int hi = (rw.a >> 4) & 0x0F0F0F0Fu;                               \
    wv.a = ip_nib2i8(__byte_perm(lo, hi, 0x5140), cx);                         \
    wv.b = ip_nib2i8(__byte_perm(lo, hi, 0x7362), cx);                         \
    lo = rw.b & 0x0F0F0F0Fu;                                                   \
    hi = (rw.b >> 4) & 0x0F0F0F0Fu;                                            \
    wv.c = ip_nib2i8(__byte_perm(lo, hi, 0x5140), cx);                         \
    wv.d = ip_nib2i8(__byte_perm(lo, hi, 0x7362), cx);                         \
    if (raw) {                                                                 \
      rsacc = __dp4a(0x01010101, (int)wv.a, rsacc);                            \
      rsacc = __dp4a(0x01010101, (int)wv.b, rsacc);                            \
      rsacc = __dp4a(0x01010101, (int)wv.c, rsacc);                            \
      rsacc = __dp4a(0x01010101, (int)wv.d, rsacc);                            \
    }                                                                          \
    *(IPv16 *)((signed char *)Asv[BUF] + sdst) = ra;                           \
    *(IPv16 *)((signed char *)Wsv[BUF] + sdst) = wv;                           \
  } while (0)

  IP_LOAD(0);
  IP_STORE(0);
  __syncthreads();

  int cur = 0;
  for (int k0 = 0; k0 < K; k0 += IP_BK) {
    const int knext = k0 + IP_BK;
    // Issue the next tile's global loads FIRST: they are in flight for the
    // whole compute below.
    if (knext < K)
      IP_LOAD(knext);

    const signed char *Ab =
      (const signed char *)Asv[cur] + (wm * 16 + g) * IP_LD;
    const signed char *Bb = (const signed char *)Wsv[cur] + (wn * 32 + g) * IP_LD;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int ko = h * 32 + 4 * t;
      int a0 = *(const int *)(Ab + ko);
      int a1 = *(const int *)(Ab + 8 * IP_LD + ko);
      int a2 = *(const int *)(Ab + ko + 16);
      int a3 = *(const int *)(Ab + 8 * IP_LD + ko + 16);
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const signed char *Bs = Bb + s * 8 * IP_LD;
        int b0 = *(const int *)(Bs + ko);
        int b1 = *(const int *)(Bs + ko + 16);
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=r"(acc[s][0]), "=r"(acc[s][1]), "=r"(acc[s][2]), "=r"(acc[s][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
            "r"(acc[s][0]), "r"(acc[s][1]), "r"(acc[s][2]), "r"(acc[s][3]));
      }
    }

    if (knext < K) {
      // Safe without a second barrier: buffer cur^1 was last READ in the
      // previous iteration, and that iteration's barrier already separated
      // those reads from these writes.
      IP_STORE(cur ^ 1);
      __syncthreads();
      cur ^= 1;
    }
  }
#undef IP_LOAD
#undef IP_STORE

  if (raw) {
    // The four threads of a column are consecutive lanes, so two shuffle-downs
    // leave the whole column sum in the (tid&3)==0 lane.
    int v = rsacc;
    v += __shfl_down_sync(0xffffffffu, v, 1);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    if (ssub == 0)
      rsh[srow] = v;
    __syncthreads();
  }
#pragma unroll
  for (int s = 0; s < 4; ++s) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int row = blockM + wm * 16 + g + ((r >> 1) ? 8 : 0);
      const int cb = wn * 32 + s * 8 + 2 * t + (r & 1);
      const int col = blockN + cb;
      if (row < M && col < N) {
        const int rsum = raw ? rsh[cb] : wrowsum[col];
        float v = (float)(acc[s][r] - azp[row] * rsum) * ascale[row] *
                  dp4a_h2f(wscale[col]);
        if (out_fp16)
          ((unsigned short *)Y)[(long)row * N + col] = dp4a_f2h(v);
        else
          Y[(long)row * N + col] = v;
      }
    }
  }
}

// ---------------------------------------------------- grouped MoE IMMA ----
// imma_gemm_pipe with the M axis driven by a PADDED per-expert work list
// (vLLM's moe_align_block_size shape): every expert's assigned rows are
// bucketed into a contiguous "gathered" row space padded per expert to a
// multiple of IP_BM, so block `by` always owns gathered rows
// [by*64, by*64+64) and needs only ONE int of steering data:
// block_expert[by], the expert whose weight this block multiplies, with -1
// meaning "padding block, discard". The HOST never reads per-expert counts;
// the grid is sized to the data-independent worst case (T*topk/64 + E).
//
// Differences from imma_gemm_pipe, and nothing else differs:
//  - B base + per-channel fp16 scale come from per-expert POINTER TABLES
//    (the 35B's 30,720 expert weights are separate tensors, not one [E,N,K]).
//  - A rows load through toks[] (gathered row -> source token row of q8);
//    tokid == nullptr is the DIRECT mode for the down projection, whose
//    input (the SwiGLU output) already lives in gathered space. toks < 0 is
//    an intra-block padding row: its A stages as zeros and its output row is
//    written as EXACT 0 (keeps the gathered buffers deterministic; SwiGLU of
//    0 is 0, so padded rows stay clean through the whole chain).
//  - raw path only (cx = 0, in-kernel rowsum): expert weights carry no
//    DevWeightQ cache by default (NNTR_CUDA_FC_NOCACHE) and never should --
//    a second copy of 15 GiB of experts is the documented anti-pattern.
//  - ascale/azp index by SOURCE TOKEN (per-row quant params of the shared
//    layer activation), not by gathered row.
// Bit-exactness contract: identical K-order int32 accumulation and identical
// epilogue scalars as the per-expert imma_gemm_pipe call on the same rows,
// so the result bytes must MATCH the per-expert path exactly (int32
// accumulation is associative; any diff is a mapping bug, never rounding).
__global__ __launch_bounds__(256, 2)
void imma_moe_grouped(const signed char *q8, const int *tokid,
                      const unsigned long long *wp_tab,
                      const unsigned long long *ws_tab,
                      const int *block_expert, const float *ascale,
                      const int *azp, float *Y, int N, int K, int out_fp16) {
  const int e = block_expert[blockIdx.y];
  if (e < 0)
    return;
  const unsigned char *plain =
    (const unsigned char *)(unsigned long long)wp_tab[e];
  const unsigned short *wscale =
    (const unsigned short *)(unsigned long long)ws_tab[e];
  __shared__ IPv16 Asv[2][IP_BM * IP_LD / 16];
  __shared__ IPv16 Wsv[2][IP_BN * IP_LD / 16];
  __shared__ int rsh[IP_BN];
  __shared__ int toks[IP_BM];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1; // 4 x 2 warp grid over 64x64
  const int g = lane >> 2, t = lane & 3;
  const int blockM = blockIdx.y * IP_BM, blockN = blockIdx.x * IP_BN;
  const int Kh = K >> 1;
  const unsigned int cx = 0u; // raw payload: offset-binary nibbles

  if (tid < IP_BM)
    toks[tid] = tokid ? tokid[blockM + tid] : (blockM + tid);
  __syncthreads();

  const int srow = tid >> 2, ssub = tid & 3;
  const int atok = toks[srow];
  const int wrow = blockN + srow;
  const bool arow_ok = atok >= 0, wrow_ok = wrow < N;
  const signed char *aptr = q8 + (long)atok * K + (ssub << 4);
  const unsigned char *wptr = plain + (long)wrow * Kh + (ssub << 3);
  const int sdst = srow * IP_LD + (ssub << 4);

  int acc[4][4];
#pragma unroll
  for (int s = 0; s < 4; ++s)
#pragma unroll
    for (int r = 0; r < 4; ++r)
      acc[s][r] = 0;
  int rsacc = 0;

  IPv16 ra;
  IPv8 rw;
  IPv16 wv;

#define IPG_LOAD(K0)                                                           \
  do {                                                                         \
    if (arow_ok) {                                                             \
      ra = *(const IPv16 *)(aptr + (K0));                                      \
    } else {                                                                   \
      ra.a = ra.b = ra.c = ra.d = 0u;                                          \
    }                                                                          \
    if (wrow_ok) {                                                             \
      rw = *(const IPv8 *)(wptr + ((K0) >> 1));                                \
    } else {                                                                   \
      rw.a = rw.b = 0u;                                                        \
    }                                                                          \
  } while (0)

#define IPG_STORE(BUF)                                                         \
  do {                                                                         \
    unsigned int lo = rw.a & 0x0F0F0F0Fu;                                      \
    unsigned int hi = (rw.a >> 4) & 0x0F0F0F0Fu;                               \
    wv.a = ip_nib2i8(__byte_perm(lo, hi, 0x5140), cx);                         \
    wv.b = ip_nib2i8(__byte_perm(lo, hi, 0x7362), cx);                         \
    lo = rw.b & 0x0F0F0F0Fu;                                                   \
    hi = (rw.b >> 4) & 0x0F0F0F0Fu;                                            \
    wv.c = ip_nib2i8(__byte_perm(lo, hi, 0x5140), cx);                         \
    wv.d = ip_nib2i8(__byte_perm(lo, hi, 0x7362), cx);                         \
    rsacc = __dp4a(0x01010101, (int)wv.a, rsacc);                              \
    rsacc = __dp4a(0x01010101, (int)wv.b, rsacc);                              \
    rsacc = __dp4a(0x01010101, (int)wv.c, rsacc);                              \
    rsacc = __dp4a(0x01010101, (int)wv.d, rsacc);                              \
    *(IPv16 *)((signed char *)Asv[BUF] + sdst) = ra;                           \
    *(IPv16 *)((signed char *)Wsv[BUF] + sdst) = wv;                           \
  } while (0)

  IPG_LOAD(0);
  IPG_STORE(0);
  __syncthreads();

  int cur = 0;
  for (int k0 = 0; k0 < K; k0 += IP_BK) {
    const int knext = k0 + IP_BK;
    if (knext < K)
      IPG_LOAD(knext);

    const signed char *Ab =
      (const signed char *)Asv[cur] + (wm * 16 + g) * IP_LD;
    const signed char *Bb =
      (const signed char *)Wsv[cur] + (wn * 32 + g) * IP_LD;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int ko = h * 32 + 4 * t;
      int a0 = *(const int *)(Ab + ko);
      int a1 = *(const int *)(Ab + 8 * IP_LD + ko);
      int a2 = *(const int *)(Ab + ko + 16);
      int a3 = *(const int *)(Ab + 8 * IP_LD + ko + 16);
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const signed char *Bs = Bb + s * 8 * IP_LD;
        int b0 = *(const int *)(Bs + ko);
        int b1 = *(const int *)(Bs + ko + 16);
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=r"(acc[s][0]), "=r"(acc[s][1]), "=r"(acc[s][2]), "=r"(acc[s][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
            "r"(acc[s][0]), "r"(acc[s][1]), "r"(acc[s][2]), "r"(acc[s][3]));
      }
    }

    if (knext < K) {
      IPG_STORE(cur ^ 1);
      __syncthreads();
      cur ^= 1;
    }
  }
#undef IPG_LOAD
#undef IPG_STORE

  {
    int v = rsacc;
    v += __shfl_down_sync(0xffffffffu, v, 1);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    if (ssub == 0)
      rsh[srow] = v;
    __syncthreads();
  }
#pragma unroll
  for (int s = 0; s < 4; ++s) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int lrow = wm * 16 + g + ((r >> 1) ? 8 : 0);
      const int cb = wn * 32 + s * 8 + 2 * t + (r & 1);
      const int col = blockN + cb;
      if (col < N) {
        const int tok = toks[lrow];
        float v = 0.0f;
        if (tok >= 0)
          v = (float)(acc[s][r] - azp[tok] * rsh[cb]) * ascale[tok] *
              dp4a_h2f(wscale[col]);
        if (out_fp16)
          ((unsigned short *)Y)[(long)(blockM + lrow) * N + col] = dp4a_f2h(v);
        else
          Y[(long)(blockM + lrow) * N + col] = v;
      }
    }
  }
}

__global__ void dp4a_gemv_fused_h(const unsigned short *Xh,
                                  const signed char *plain,
                                  const int *wrowsum,
                                  const unsigned short *wscale, float *Y,
                                  int N, int K, int out_fp16) {
  extern __shared__ signed char q8s[]; // K bytes (launch-sized), 4-aligned
  __shared__ float smn[128];
  __shared__ float smx[128];
  // Phase A: cooperative row min/max -> qparams -> quantize into shared.
  // All threads participate (the n>=N early-out must come AFTER the syncs).
  float lmn = 0.f, lmx = 0.f;
  for (int kk = threadIdx.x; kk < K; kk += blockDim.x) {
    float v = dp4a_h2f(Xh[kk]);
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
  for (int kk = threadIdx.x; kk < K; kk += blockDim.x) {
    int q = (int)rintf(dp4a_h2f(Xh[kk]) * scale_q) + zp;
    q = max(-128, min(127, q));
    q8s[kk] = (signed char)q;
  }
  __syncthreads();
  // Phase B: the dp4a_gemv body over the shared q8 row (same warp mapping).
  const int warps_per_block = blockDim.x >> 5;
  int n = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  if (n >= N)
    return;
  const int lane = threadIdx.x & 31;
  int Kh = (K + 1) >> 1;
  const signed char *wrow = plain + (long)n * Kh;
  int acc = 0;
  for (int kk = lane * 4; kk + 4 <= K; kk += 32 * 4) {
    int a = *(const int *)(q8s + kk);
    int kb = kk >> 1;
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
    float r = (float)(acc - zp * wrowsum[n]) * recip * dp4a_h2f(wscale[n]);
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
// `raw` as in dp4a_gemm: the XOR moves onto the staging load below, and the
// per-column rowsum is accumulated off `wf[j]`, which the inner loop already
// holds in a register. Threads sharing a `tx` recompute the same rowsum, but
// that is register arithmetic on data already staged -- no extra traffic.
__global__ void dp4a_gemm_reg(const signed char *q8, const signed char *plain,
                              const float *ascale, const int *azp,
                              const int *wrowsum, const unsigned short *wscale,
                              float *Y, int M, int N, int K, int out_fp16,
                              int raw) {
  __shared__ signed char As[RB_BM][RB_BK];
  __shared__ signed char Ws[RB_BN][RB_BK];
  int tx = threadIdx.x, ty = threadIdx.y; // 0..15 each
  int tid = ty * 16 + tx;
  int blockM = blockIdx.y * RB_BM, blockN = blockIdx.x * RB_BN;
  int Kh = (K + 1) >> 1;
  const int xr = raw ? 0x88 : 0;
  int acc[RB_TM][RB_TN];
  int rs[RB_TN];
#pragma unroll
  for (int j = 0; j < RB_TN; j++)
    rs[j] = 0;
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
        unsigned char b =
          (unsigned char)(((unsigned char)plain[(long)nn * Kh + (kk >> 1)]) ^
                          xr);
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
      // rowsum off the SAME staged tile: Ws is zero where kk >= K, so this
      // sums exactly k in [0,K) -- the range weight_rowsum uses.
#pragma unroll
      for (int j = 0; j < RB_TN; j++)
        rs[j] = __dp4a(0x01010101, wf[j], rs[j]);
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
        const int rsum = raw ? rs[j] : wrowsum[col];
        float r = (float)(acc[i][j] - zp * rsum) * as * dp4a_h2f(wscale[col]);
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

// int4 plain weight -> int8 [K,N] (w8[k*N+n] = int4(n,k)). Unpacked once and
// cached (weights are static), so cuBLAS reads contiguous int8 -- doing this per
// call would add a memory pass that erases the Tensor-Core win.
__global__ void repack_plain_i8_kn(const unsigned char *qw, signed char *w8,
                                   int N, int K, int Kh) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (n >= N || k >= K)
    return;
  w8[(long)k * N + n] = (signed char)plain_decode(qw, n, k, Kh);
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

// [single weight copy] The QS4CX plain payload is consumed by the CUDA FC path
// directly (the OpenCL v8c kernel consumes it the same way) -- no host/UVM copy
// of the nibble payload. The only per-weight side allocation is this N-entry
// fp16 scale buffer: the dequant kernel reads the per-channel scale on device
// every call, while the tensor stores fp32 scales. Built once at first use,
// cached by the fp32-scale pointer with no erase (weights live for the process
// lifetime), never under a graph capture (a cudaMallocManaged inside capture
// invalidates it).
bool cuda_fc_qs4cx_scales_to_uvm_fp16(const float *fp32_scales, unsigned int N,
                                      const unsigned short **out_sc) {
  static std::map<const void *, unsigned short *> cache;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = cache.find(fp32_scales);
  if (it == cache.end()) {
    if (StreamManager::Global().isCapturing()) {
      // Declining here skips the WHOLE QS4CX arm of the FC dispatcher, so the
      // captured graph would be missing this layer entirely.
      StreamManager::Global().markCaptureDoomed(
        "a QS4CX fp16 scale buffer was not prewarmed and cannot be allocated "
        "under capture");
      return false;
    }
    unsigned short *usc = nullptr;
    // [WDDM coherence] This buffer is host-WRITTEN once and device-READ every
    // FC call -- the pattern that is incoherent on cMA==0 managed memory. Use
    // pinned host-mapped (zero-copy, UVA same-pointer) there; managed
    // elsewhere.
    static const bool host_mapped = []() {
      const char *e = std::getenv("NNTR_CUDA_HOST_MAPPED");
      if (e != nullptr)
        return e[0] == '1';
      return !ContextManager::Global().concurrentManagedAccess();
    }();
    if (host_mapped) {
      if (cudaHostAlloc(&usc, sizeof(unsigned short) * (size_t)N,
                        cudaHostAllocMapped) != cudaSuccess)
        return false;
    } else if (cudaMallocManaged(&usc, sizeof(unsigned short) * (size_t)N) !=
               cudaSuccess)
      return false;
    for (unsigned int n = 0; n < N; ++n)
      usc[n] = compute_fp32_to_fp16(fp32_scales[n]);
    it = cache.emplace(fp32_scales, usc).first;
  }
  *out_sc = it->second;
  return true;
}

namespace {
// Reusable fp32 activation/output staging for the fp16-naive path (the plain
// GEMM is fp32-in/fp32-out). Grown on demand, kept for reuse.
float *g_stage_xf = nullptr;
size_t g_stage_xf_cap = 0;
float *g_stage_yf = nullptr;
size_t g_stage_yf_cap = 0;
std::mutex g_stage_mtx;

bool ensure_buf(void **buf, size_t *cap, size_t bytes) {
  if (bytes <= *cap)
    return true;
  // cudaMalloc/cudaFree inside a CUDA-graph stream capture invalidates it; bail
  // so the caller falls back rather than corrupting the graph. Refusing keeps
  // the capture VALID, which is the dangerous part: the caller's op drops out
  // and the graph is instantiated without it. Doom the capture so it is thrown
  // away instead of replayed with a hole in it.
  if (StreamManager::Global().isCapturing()) {
    StreamManager::Global().markCaptureDoomed(
      "an FC scratch buffer was under-sized by the prewarm and cannot grow "
      "under capture");
    return false;
  }
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

// --- w4a8 dp4a fast path (the default int4 FC decode path) ---------------
/**
 * @brief Cached signed-packed-int4 repack of a QS4CX weight, keyed by the
 * plain host/UVM payload pointer (weight.getData()). Weights are static for
 * the model lifetime, so the derived device cache is built once and never
 * erased.
 */
struct DevWeightQ {
  signed char *plain = nullptr; // signed packed int4 [N, ceil(K/2)]
  int *rowsum = nullptr;        // per-channel sum of signed int4 [N]
};
std::unordered_map<const void *, DevWeightQ> g_dp4a_plain_cache;
// per-row int8 activation quant scratch (q8 + recip scale + zero-point).
signed char *g_dp4a_q8 = nullptr;
size_t g_dp4a_q8_cap = 0;
float *g_dp4a_ascale = nullptr; // per-row recip (dequant scale)
size_t g_dp4a_ascale_cap = 0;
int *g_dp4a_azp = nullptr; // per-row activation zero-point
size_t g_dp4a_azp_cap = 0;
std::mutex g_dp4a_mtx;
// +256B tail pad on the int8 activation scratch: the cuBLAS int8 IMMA GEMM (a
// later change) reads A with wide (>=16B) Tensor-Core loads that can run past
// the last element; sizing the shared scratch with the pad here keeps that
// change a pure add. The __dp4a path itself does not over-read.
static constexpr size_t FC_I8_TAIL_PAD = 256;

/**
 * @brief May a device kernel READ this plain QS4CX payload?
 *
 * Only a derived-cache HIT is a pure pointer-value lookup. Both builders below
 * bind the plain payload straight into a device kernel on a cache MISS --
 * ensure_dp4a_cache_locked() into repack_plain_i4, ensure_i8_cache_locked()
 * into repack_plain_i8_kn -- so a miss must first prove the bytes are readable
 * from the device: really device-accessible, and not a [pool-bypass] payload
 * whose pages were discarded (those read back as zero-filled, silently).
 *
 * Refusing at the builder rather than at the dispatcher's entry gate covers
 * every caller, present and future, and keeps the rule next to the kernel
 * argument it protects.
 */
bool plain_bindable(const unsigned char *plain_w) {
  const bool dev = dev_accessible(plain_w);
  const bool dropped = cuda_fc_qs4cx_plain_dropped(plain_w);
  if (dev && !dropped)
    return true;
  // Once per process: this is a configuration report, not a per-call event.
  static bool warned = false;
  if (!warned) {
    warned = true;
    ml_logw("[CUDA] fc_qint4: a derived weight cache is missing for a payload "
            "no kernel may read (device-accessible=%d, pages dropped=%d), so "
            "the build is refused and the FC falls back to a path that needs "
            "only the already-built cache. Build the caches at load time "
            "(NNTR_CUDA_PREWARM=1) or keep the payload device-resident "
            "(NNTR_QS4CX_HEAP_BYPASS=0 / NNTR_CUDA_DROP_PLAIN=0).",
            (int)dev, (int)dropped);
  }
  return false;
}

DevWeightQ *ensure_dp4a_cache_locked(const unsigned char *plain_w,
                                     unsigned int N, unsigned int K) {
  auto it = g_dp4a_plain_cache.find(plain_w);
  if (it != g_dp4a_plain_cache.end())
    return &it->second;
  // MISS: repack_plain_i4 below dereferences plain_w on the device.
  if (!plain_bindable(plain_w))
    return nullptr;
  const int n = (int)N, k = (int)K;
  const size_t Kh = (K + 1u) / 2u;
  auto kr = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "repack_plain_i4");
  auto krs = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "weight_rowsum");
  if (!kr || !krs)
    return nullptr;
  DevWeightQ dw;
  if (cudaMalloc(&dw.plain, (size_t)N * Kh) != cudaSuccess)
    return nullptr;
  if (cudaMalloc(&dw.rowsum, sizeof(int) * (size_t)N) != cudaSuccess) {
    cudaFree(dw.plain);
    return nullptr;
  }
  const int khi = (int)Kh;
  kr->SetKernelArguments(0, &plain_w, sizeof(plain_w));
  kr->SetKernelArguments(1, &dw.plain, sizeof(dw.plain));
  kr->SetKernelArguments(2, &n, sizeof(n));
  kr->SetKernelArguments(3, &khi, sizeof(khi));
  const int rb[3] = {256, 1, 1};
  const int rg[3] = {(int)(((size_t)N * Kh + 255) / 256), 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kr, rg, rb)) {
    cudaFree(dw.plain);
    cudaFree(dw.rowsum);
    return nullptr;
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
    return nullptr;
  }
  it = g_dp4a_plain_cache.emplace(plain_w, dw).first;
  return &it->second;
}

/** Do not BUILD the DevWeightQ cache in-path; read the QS4CX payload directly
 *  instead. An already-built cache is still used (see the call site).
 *  DEFAULT ON; NNTR_CUDA_FC_NOCACHE=0 restores the old always-build behaviour.
 *  Read once. */
static bool fc_nocache_on() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_CUDA_FC_NOCACHE");
    return !(e && e[0] == '0');
  }();
  return on;
}

// NNTR_CUDA_IMMA_TILE: run the register-blocked tile on the int8 Tensor Cores
// instead of the int ALU.
//   0 = dp4a only
//   1 = imma_gemm_reg   (v1: byte staging, two barriers per k-step)
//   2 = imma_gemm_pipe  (v2: vector staging, double-buffered, BK=64)
//
// DEFAULT 2, and the reason is measured on both models plus a bit-exactness
// harness (7 shapes x raw/cached x fp32/fp16, all identical to dp4a_gemm_reg
// -- int32 accumulation is associative, so a correct fragment mapping has no
// tolerance to argue about):
//
//   kernel alone   MoE gate/up 128x512x2048   dp4a 0.228ms  v1 0.125  v2 0.026
//                  attention   256x2048x4096  dp4a 2.540ms  v1 1.350  v2 0.223
//   gemma4 e2e     prefill  390.4 -> 2,828.2 TPS   (identical output text)
//   35B e2e        prefill  237.1 ->   263.1 TPS   (+11%: the expert GEMMs are
//                                                   only 3.2 s of a 35 s
//                                                   prefill -- host routing is
//                                                   13.6 s of it)
//
// It is defaulted rather than left opt-in because a measured win behind a flag
// is a trap for the next reader; =1 and =0 remain for A/B.
static int imma_tile_level() {
  static const int v = []() {
    const char *e = std::getenv("NNTR_CUDA_IMMA_TILE");
    return (e != nullptr) ? std::atoi(e) : 2;
  }();
  return v;
}

// repack (cached) + GEMM into a device float Y, using the already-staged
// q8/ascale scratch. Caller holds g_dp4a_mtx and has run act-quant.
bool dp4a_repack_and_gemm(const unsigned char *plain_w,
                          const unsigned short *scales_fp16, float *Yf,
                          unsigned int M, unsigned int N, unsigned int K,
                          int out_fp16 = 0,
                          const unsigned short *Xh_fused = nullptr) {
  const int n = (int)N, k = (int)K;
  const bool gemv = (M == 1);
  // Fused decode path: activation quant folded into the GEMV (bit-identical
  // output, one launch fewer, no q8 scratch round-trip). Caller passes the
  // fp16 activation row instead of pre-staging g_dp4a_q8.
  const bool fused = gemv && Xh_fused != nullptr;
  // MEASURED, do not "fix" this by occupancy. dp4a_gemm_reg's 64x64 tile gives
  // only ceil(N/64)*ceil(M/64) = 8 blocks at the MoE prefill shape (M~42,
  // N=512) on a 16-SM part, and it still beats the 16x16 dp4a_gemm's 96 blocks
  // by 1.4x on the whole layer (qwen_moe 7,940 ms vs 11,104 ms over a
  // 1,341-token prefill). The register-blocked tile stages coalesced tiles into
  // shared memory and yields 16 outputs per thread; the 16x16 kernel is one
  // output per thread with a serial K loop and strided weight reads. The tile
  // shape is right -- what is missing is enough of them per launch, which is a
  // grouped kernel (all experts in one grid), not a smaller tile.
  const bool tiled = (M >= 8);
  const int imma_tile = imma_tile_level();
  const bool use_imma = imma_tile > 0 && tiled && !gemv && !fused;
  // NNTR_CUDA_FC_NOCACHE=1: skip the DevWeightQ cache entirely for the decode
  // GEMV and read the QS4CX payload in place. That cache is a byte-for-byte
  // XOR copy of the weights plus an N-int rowsum, so on this model it is 15.1
  // GiB of the resident set spent avoiding one instruction per load. It is
  // also built LAZILY for MoE experts -- the load-time prewarm walk filters on
  // layer type "fully_connected" and the MoE node is "qwen_moe" -- so a
  // profile shows repack_plain_i4 + weight_rowsum as 18% of GPU time, and a
  // long prefill that touches all 30,720 experts pays it in full.
  // The GEMV needs its own kernel (it has no wrowsum parameter at all); the two
  // GEMMs take a launch-uniform `raw` flag instead, so there is one code path
  // rather than two copies of a register-blocked tile.
  //
  // The flag means "do not BUILD the cache", not "do not use it". If one is
  // already there we take it -- that is strictly better on both counts, and
  // the distinction is measurable: gemma4's FCs are layer type
  // "fully_connected" so the load-time walk prewarms every one of them, and
  // forcing the raw path there cost 6.2% of decode (64.2 -> 60.2 TPS) while
  // freeing nothing, because the cache had already been allocated. The 35B's
  // experts are the opposite case -- never prewarmed, 15.1 GiB, built in-path.
  // Same rule serves both.
  const bool nocache = fc_nocache_on() && !fused &&
                       g_dp4a_plain_cache.find(plain_w) ==
                         g_dp4a_plain_cache.end();
  const bool raw = nocache && gemv;
  // v2 has no k tail and assumes every vector access is aligned: K a multiple
  // of BK=64 makes each A row 16-B aligned and each payload row 32-B aligned,
  // and the payload BASE must be 8-B aligned. The cached copy is a cudaMalloc
  // so it always qualifies; the in-place QS4CX payload is whatever the loader
  // gave it, hence the check on the pointer actually passed. Every K in these
  // models (512/1024/2048/4096) qualifies; anything else falls back to v1.
  const void *w_gate = nocache ? (const void *)plain_w : nullptr;
  const bool use_pipe =
    use_imma && imma_tile >= 2 && (K % 64u) == 0u &&
    (!nocache || (reinterpret_cast<uintptr_t>(w_gate) & 7u) == 0u);
  auto kg = CudaContext::Global().registerCudaKernel(
    FC_QINT4_DP4A_SRC, fused     ? "dp4a_gemv_fused_h"
                       : raw     ? "dp4a_gemv_raw"
                       : gemv    ? "dp4a_gemv"
                       : use_pipe ? "imma_gemm_pipe"
                       : use_imma ? "imma_gemm_reg"
                                  : (tiled ? "dp4a_gemm_reg" : "dp4a_gemm"));
  if (!kg)
    return false;

  signed char *plain = nullptr;
  int *wrowsum = nullptr;
  if (!nocache) {
    DevWeightQ *dwp = ensure_dp4a_cache_locked(plain_w, N, K);
    if (!dwp)
      return false;
    plain = dwp->plain;
    wrowsum = dwp->rowsum;
  }
  // When nocache, the kernel reads the payload itself and derives rowsum; the
  // cache pointers stay null and are never dereferenced.
  const void *w_arg = nocache ? (const void *)plain_w : (const void *)plain;
  const int raw_i = nocache ? 1 : 0;

  const int mm = (int)M;
  if (raw) {
    kg->SetKernelArguments(0, &g_dp4a_q8, sizeof(g_dp4a_q8));
    kg->SetKernelArguments(1, &plain_w, sizeof(plain_w));
    kg->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
    kg->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
    kg->SetKernelArguments(4, &scales_fp16, sizeof(scales_fp16));
    kg->SetKernelArguments(5, &Yf, sizeof(Yf));
    kg->SetKernelArguments(6, &n, sizeof(n));
    kg->SetKernelArguments(7, &k, sizeof(k));
    kg->SetKernelArguments(8, &out_fp16, sizeof(out_fp16));
    const int gvb[3] = {128, 1, 1};
    const int gvg[3] = {((int)N + 3) / 4, 1, 1};
    return StreamManager::Global().DispatchCommand(*kg, gvg, gvb);
  }
  if (fused) {
    kg->SetKernelArguments(0, &Xh_fused, sizeof(Xh_fused));
    kg->SetKernelArguments(1, &plain, sizeof(plain));
    kg->SetKernelArguments(2, &wrowsum, sizeof(wrowsum));
    kg->SetKernelArguments(3, &scales_fp16, sizeof(scales_fp16));
    kg->SetKernelArguments(4, &Yf, sizeof(Yf));
    kg->SetKernelArguments(5, &n, sizeof(n));
    kg->SetKernelArguments(6, &k, sizeof(k));
    kg->SetKernelArguments(7, &out_fp16, sizeof(out_fp16));
    const int gvb[3] = {128, 1, 1};
    const int gvg[3] = {((int)N + 3) / 4, 1, 1};
    return StreamManager::Global().DispatchCommand(*kg, gvg, gvb,
                                                   (unsigned int)K);
  }
  kg->SetKernelArguments(0, &g_dp4a_q8, sizeof(g_dp4a_q8));
  kg->SetKernelArguments(1, &w_arg, sizeof(w_arg));
  kg->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kg->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
  kg->SetKernelArguments(4, &wrowsum, sizeof(wrowsum));
  kg->SetKernelArguments(5, &scales_fp16, sizeof(scales_fp16));
  kg->SetKernelArguments(6, &Yf, sizeof(Yf));
  if (gemv) {
    // dp4a_gemv: one WARP per output, 4 warps (128 threads) per block ->
    // ceil(N/4) blocks instead of N (4x fewer per-block launch/epilogue
    // overheads).
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
  kg->SetKernelArguments(11, &raw_i, sizeof(raw_i));
  if (use_imma) {
    // Same 64x64 output tile as dp4a_gemm_reg for both v1 and v2, but 256
    // threads as a 1-D block: the warp, not the thread, is the unit that owns
    // an mma fragment. Only the K depth differs (v2 stages 64 at a time).
    const int ib[3] = {256, 1, 1};
    const int ig[3] = {((int)N + 63) / 64, ((int)M + 63) / 64, 1};
    return StreamManager::Global().DispatchCommand(*kg, ig, ib);
  }
  const int gb[3] = {16, 16, 1};
  const int tile = tiled ? 64 : 16;
  const int gg[3] = {((int)N + tile - 1) / tile, ((int)M + tile - 1) / tile, 1};
  return StreamManager::Global().DispatchCommand(*kg, gg, gb);
}

static bool dp4a_stage_scratch(unsigned int M, unsigned int K) {
  return ensure_buf((void **)&g_dp4a_q8, &g_dp4a_q8_cap,
                    (size_t)M * K + FC_I8_TAIL_PAD) &&
         ensure_buf((void **)&g_dp4a_ascale, &g_dp4a_ascale_cap,
                    sizeof(float) * (size_t)M) &&
         ensure_buf((void **)&g_dp4a_azp, &g_dp4a_azp_cap,
                    sizeof(int) * (size_t)M);
}


// --- cuBLAS int8 IMMA (Tensor Core) prefill weight cache ------------------
/**
 * @brief int8-unpacked weight [K,N] + per-channel rowsum for the cuBLAS int8
 * path, keyed by the QS4CX plain payload pointer (unpacked once, weights are
 * static).
 */
struct DevWeightI8 {
  signed char *w8 = nullptr; // int8 weight [K,N] (w8[k*N+n] = int4(n,k))
  int *rowsum = nullptr;     // per-channel sum of int8 weight [N]
};
std::unordered_map<const void *, DevWeightI8> g_i8_weight_cache;
// FCs exempted from the EAGER cuBLAS-i8 cache build (skip_prefill towers /
// untied lm_head decode at M=1 never reach the M>=32 cuBLAS gate; their [K,N]
// int8 cache would be dead VRAM). The lazy runtime build self-heals if one is
// ever reached anyway.
std::unordered_set<const void *> g_i8_exempt;

// --- prewarm cost accounting ----------------------------------------------
// Split of the load-time prewarm lap into the parts a persistent pack cache
// CAN remove (host derive: permute/unpack + row sums, and the miss-path tee)
// and the part it CANNOT (the H2D upload, which happens either way). This is
// what decides whether the disk cache is worth its bytes on this lane, so it
// is permanent instrumentation, not a probe -- read via
// cuda_fc_qs4cx_prewarm_stats().
double g_prewarm_derive_ms = 0.0;
double g_prewarm_upload_ms = 0.0;
double g_prewarm_tee_ms = 0.0;
double g_prewarm_hit_ms = 0.0;
size_t g_prewarm_derive_bytes = 0;
size_t g_prewarm_hit_bytes = 0;
double g_stat_clock() {
  return std::chrono::duration<double, std::milli>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}
int *g_i8_c = nullptr; // int32 GEMM output scratch [Mpad,N]
size_t g_i8_c_cap = 0;
// act-quant handoff: whoever last filled g_dp4a_q8 records WHAT it quantized
// (activation pointer + K) and the stream dispatch count at that moment. A
// consumer FC may reuse the staged quant only if both still match -- the
// pointer alone is forgeable by the activation pool (a recycled buffer reuses
// the address), the sequence number is not: any kernel dispatched in between
// bumps it and the FC re-quantizes. Written by the fused norm+quant producer,
// by the dp4a decode path, and (under NNTR_QUANT_DEDUP) by the cuBLAS prefill
// path.
const void *g_last_quant_xh = nullptr;
int g_last_quant_k = 0;
unsigned long long g_last_quant_seq = 0;
bool g_last_quant_valid = false;

// NNTR_CUDA_FUSED_NORMQ: fold the decode RMSNorm and the int8 activation quant
// of the FC group it feeds into one kernel, and let the sibling FCs of that
// group consume the staged quant instead of recomputing it. Bit-identical to
// the split path (see rmsnorm_quant_i8_h), so it is the DEFAULT; =0 restores
// the separate rmsnorm_fp16 + act_quant_i8_h launches.
bool fused_normq_on() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_CUDA_FUSED_NORMQ");
    return !(e != nullptr && e[0] == '0');
  }();
  return v;
}

// Publish the staged quant as reusable by the very next FC on @p xh.
void mark_quant_staged(const void *xh, int k) {
  g_last_quant_xh = xh;
  g_last_quant_k = k;
  g_last_quant_seq = StreamManager::Global().dispatchSeq();
  g_last_quant_valid = true;
}

// True when g_dp4a_q8 already holds the int8 quant of (xh, k).
bool quant_staged_for(const void *xh, int k) {
  return g_last_quant_valid && xh == g_last_quant_xh && k == g_last_quant_k &&
         StreamManager::Global().dispatchSeq() == g_last_quant_seq;
}

/**
 * @brief allocate the [K,N] int8 weight buffer for the cuBLAS IMMA GEMM.
 * @note There are two builders for this exact buffer -- the lazy one below and
 * the eager load-time one in cuda_fc_qs4cx_prewarm() -- feeding one consumer
 * whose vectorized Tensor-Core loads run past the last element, so the
 * FC_I8_TAIL_PAD must be identical in both. Allocate in one place so the two
 * cannot drift apart.
 */
static cudaError_t alloc_i8_weight(signed char **w8, unsigned int N,
                                   unsigned int K) {
  return cudaMalloc(w8, (size_t)N * K + FC_I8_TAIL_PAD);
}

static DevWeightI8 *ensure_i8_cache_locked(const unsigned char *plain_w,
                                           unsigned int N, unsigned int K) {
  auto it = g_i8_weight_cache.find(plain_w);
  if (it != g_i8_weight_cache.end())
    return &it->second;
  // MISS: repack_plain_i8_kn below dereferences plain_w on the device. The i8
  // [K,N] cache is a SEPARATE map from the dp4a one, so the dispatcher's
  // "derived cache exists" entry ticket (cuda_fc_qs4cx_has_cache(), dp4a only)
  // does NOT imply this entry exists. Refusing here makes
  // cuda_fc_qs4cx_cublas_i8_gemm_fp16() report failure, and the caller's
  // fall-through hands the call to dp4a -- which under that ticket IS a pure
  // cache hit.
  if (!plain_bindable(plain_w))
    return nullptr;
  const int n = (int)N, k = (int)K, kh = (int)((K + 1u) / 2u);
  auto krp = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "repack_plain_i8_kn");
  auto krs = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "weight_rowsum_kn");
  if (!krp || !krs)
    return nullptr;
  DevWeightI8 dw;
  if (alloc_i8_weight(&dw.w8, N, K) != cudaSuccess)
    return nullptr;
  if (cudaMalloc(&dw.rowsum, sizeof(int) * (size_t)N) != cudaSuccess) {
    cudaFree(dw.w8);
    return nullptr;
  }
  krp->SetKernelArguments(0, &plain_w, sizeof(plain_w));
  krp->SetKernelArguments(1, &dw.w8, sizeof(dw.w8));
  krp->SetKernelArguments(2, &n, sizeof(n));
  krp->SetKernelArguments(3, &k, sizeof(k));
  krp->SetKernelArguments(4, &kh, sizeof(kh));
  const int pb[3] = {16, 16, 1};
  const int pg[3] = {((int)N + 15) / 16, ((int)K + 15) / 16, 1};
  if (!StreamManager::Global().DispatchCommand(*krp, pg, pb)) {
    cudaFree(dw.w8);
    cudaFree(dw.rowsum);
    return nullptr;
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
    return nullptr;
  }
  it = g_i8_weight_cache.emplace(plain_w, dw).first;
  return &it->second;
}

} // namespace

// One grouped-MoE GEMM launch: every expert's rows in one grid (see the
// kernel comment). All buffers are caller-owned (cuda_moe.cpp) -- this
// function touches none of the dp4a scratch, so it takes no lock. N and K
// must be multiples of 64 (both 35B projections qualify: 512/2048); the
// per-expert payload alignment is the CALLER's table-build-time check.
bool cuda_fc_qs4cx_moe_grouped_gemm(
  const signed char *q8, const int *tokid, const unsigned long long *wp_tab,
  const unsigned long long *ws_tab, const int *block_expert,
  const float *ascale, const int *azp, void *Y, unsigned int n_mblocks,
  unsigned int N, unsigned int K, int out_fp16) {
  if (q8 == nullptr || wp_tab == nullptr || ws_tab == nullptr ||
      block_expert == nullptr || Y == nullptr || n_mblocks == 0 ||
      (N & 63u) != 0u || (K & 63u) != 0u)
    return false;
  auto kg = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "imma_moe_grouped");
  if (!kg)
    return false;
  const int n = (int)N, k = (int)K;
  kg->SetKernelArguments(0, &q8, sizeof(q8));
  kg->SetKernelArguments(1, &tokid, sizeof(tokid));
  kg->SetKernelArguments(2, &wp_tab, sizeof(wp_tab));
  kg->SetKernelArguments(3, &ws_tab, sizeof(ws_tab));
  kg->SetKernelArguments(4, &block_expert, sizeof(block_expert));
  kg->SetKernelArguments(5, &ascale, sizeof(ascale));
  kg->SetKernelArguments(6, &azp, sizeof(azp));
  kg->SetKernelArguments(7, &Y, sizeof(Y));
  kg->SetKernelArguments(8, &n, sizeof(n));
  kg->SetKernelArguments(9, &k, sizeof(k));
  kg->SetKernelArguments(10, &out_fp16, sizeof(out_fp16));
  const int ib[3] = {256, 1, 1};
  const int ig[3] = {(int)(N / 64u), (int)n_mblocks, 1};
  return StreamManager::Global().DispatchCommand(*kg, ig, ib);
}

bool cuda_fc_qs4cx_gemm_fp32(const float *X, const unsigned char *plain_w,
                             const unsigned short *scales_fp16, float *Y,
                             unsigned int M, unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;

  auto kernel = CudaContext::Global().registerCudaKernel(FC_QINT4_PLAIN_SRC,
                                                         "fc_qint4_plain_gemm");
  if (!kernel) {
    ml_loge("[CUDA] fc_qint4_plain: kernel registration failed");
    return false;
  }

  int m = (int)M, n = (int)N, k = (int)K;
  int kh = (int)((K + 1u) / 2u);
  kernel->SetKernelArguments(0, &X, sizeof(X));
  kernel->SetKernelArguments(1, &plain_w, sizeof(plain_w));
  kernel->SetKernelArguments(2, &scales_fp16, sizeof(scales_fp16));
  kernel->SetKernelArguments(3, &Y, sizeof(Y));
  kernel->SetKernelArguments(4, &m, sizeof(m));
  kernel->SetKernelArguments(5, &n, sizeof(n));
  kernel->SetKernelArguments(6, &k, sizeof(k));
  kernel->SetKernelArguments(7, &kh, sizeof(kh));

  const int block[3] = {16, 16, 1};
  const int grid[3] = {((int)N + 15) / 16, ((int)M + 15) / 16, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

bool cuda_fc_qs4cx_gemm_fp16_naive(const unsigned short *Xh,
                                   const unsigned char *plain_w,
                                   const unsigned short *scales_fp16,
                                   unsigned short *Yh, unsigned int M,
                                   unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  auto kh2f =
    CudaContext::Global().registerCudaKernel(FC_QINT4_PLAIN_SRC, "cvt_h2f");
  auto kf2h =
    CudaContext::Global().registerCudaKernel(FC_QINT4_PLAIN_SRC, "cvt_f2h");
  if (!kh2f || !kf2h)
    return false;
  std::lock_guard<std::mutex> lk(g_stage_mtx);
  const size_t xn = (size_t)M * K, yn = (size_t)M * N;
  if (!ensure_buf((void **)&g_stage_xf, &g_stage_xf_cap, sizeof(float) * xn) ||
      !ensure_buf((void **)&g_stage_yf, &g_stage_yf_cap, sizeof(float) * yn))
    return false;
  int xni = (int)xn, yni = (int)yn;
  const int cb[3] = {256, 1, 1};
  kh2f->SetKernelArguments(0, &Xh, sizeof(Xh));
  kh2f->SetKernelArguments(1, &g_stage_xf, sizeof(g_stage_xf));
  kh2f->SetKernelArguments(2, &xni, sizeof(xni));
  const int xg[3] = {((int)xn + 255) / 256, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kh2f, xg, cb))
    return false;
  // naive plain-decode FP32-act GEMM (its own dispatch + drain).
  if (!cuda_fc_qs4cx_gemm_fp32(g_stage_xf, plain_w, scales_fp16, g_stage_yf, M,
                               N, K))
    return false;
  kf2h->SetKernelArguments(0, &g_stage_yf, sizeof(g_stage_yf));
  kf2h->SetKernelArguments(1, &Yh, sizeof(Yh));
  kf2h->SetKernelArguments(2, &yni, sizeof(yni));
  const int yg[3] = {((int)yn + 255) / 256, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kf2h, yg, cb))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

// [pool-bypass] True when the dp4a derived cache for this plain pointer
// already exists -- the dispatch then only needs the pointer VALUE as a key,
// so a host-heap (non-device-accessible) payload is fine and the host->device
// weight staging can be skipped entirely.
bool cuda_fc_qs4cx_has_cache(const unsigned char *plain_w) {
  if (plain_w == nullptr)
    return false;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  return g_dp4a_plain_cache.count(plain_w) != 0;
}

// [pool-bypass] Drop the plain payload's fully-owned pages after every derived
// device cache (dp4a packed + cuBLAS int8 + fp16 scales) exists -- the CUDA
// forward then only compares the pointer VALUE as a cache key, never
// dereferencing the bytes. Only meaningful when the payload is ordinary heap
// (NNTR_QS4CX_HEAP_BYPASS): madvise on a managed/UVM pool page fails EINVAL
// harmlessly. Refuses to run when the naive diagnostic path is selected
// (NNTR_FC_CUDA_DP4A=0 reads the plain payload per call). Inward page
// alignment protects neighboring heap metadata. x86-only like the bypass.
namespace {
// Payloads whose pages this process has discarded. Reading them back gives
// zero-filled pages, so the consumers that DO dereference the payload (the
// derived-cache builders via plain_bindable(); the naive plain GEMM; the host
// dot() tail in CudaComputeOps::fc) must be able to tell and refuse instead of
// silently computing against zeros.
//
// LOCK ORDER: g_dp4a_mtx BEFORE g_dropped_mtx, never the reverse. The nesting
// sites are plain_bindable() (called with g_dp4a_mtx held) and
// cuda_fc_qs4cx_release_weight_caches() / cuda_fc_qs4cx_prewarm() (which hold
// g_dp4a_mtx across their clear/erase). Nothing takes g_dropped_mtx first and
// then g_dp4a_mtx: cuda_fc_qs4cx_plain_dropped() and
// cuda_fc_qs4cx_drop_plain_pages() take g_dropped_mtx alone.
std::unordered_set<const void *> g_dropped_plain;
std::mutex g_dropped_mtx;

// Forget a drop mark. Called when a payload at this address has just been read
// successfully to (re)build its derived cache, so any mark left over from a
// previous model generation that happened to land on the same heap address
// describes bytes that no longer exist. Without this the mark is immortal and
// the host-dot() refusal in CudaComputeOps::fc would abort a reloaded model
// that is holding perfectly valid data.
void forget_dropped_plain(const unsigned char *plain_w) {
  if (plain_w == nullptr)
    return;
  std::lock_guard<std::mutex> lk(g_dropped_mtx);
  g_dropped_plain.erase(plain_w);
}
} // namespace

bool cuda_fc_qs4cx_plain_dropped(const unsigned char *plain_w) {
  if (plain_w == nullptr)
    return false;
  std::lock_guard<std::mutex> lk(g_dropped_mtx);
  return g_dropped_plain.count(plain_w) != 0;
}

bool cuda_fc_qs4cx_drop_plain_pages(const unsigned char *plain_w,
                                    unsigned int N, unsigned int K) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
  if (plain_w == nullptr || N == 0 || K == 0)
    return false;
  static const bool naive = []() {
    const char *e = std::getenv("NNTR_FC_CUDA_DP4A");
    return e != nullptr && e[0] == '0';
  }();
  if (naive)
    return false;
  const size_t payload =
    (size_t)N * (((size_t)K + 1) / 2) + (size_t)N * sizeof(float);
  const size_t page = 4096;
  uintptr_t lo = ((uintptr_t)plain_w + page - 1) & ~(page - 1);
  uintptr_t hi = ((uintptr_t)plain_w + payload) & ~(page - 1);
  if (hi <= lo)
    return false;
#if defined(_WIN32)
  const bool dropped =
    DiscardVirtualMemory((void *)lo, (SIZE_T)(hi - lo)) == ERROR_SUCCESS;
#else
  const bool dropped =
    ::madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED) == 0;
#endif
  if (dropped) {
    std::lock_guard<std::mutex> lk(g_dropped_mtx);
    g_dropped_plain.insert(plain_w);
  }
  return dropped;
#else
  (void)plain_w;
  (void)N;
  (void)K;
  return false;
#endif
}

bool cuda_fc_qs4cx_fused_normq_enabled() { return fused_normq_on(); }

bool cuda_fc_qs4cx_rmsnorm_prequant_fp16(const unsigned short *x,
                                         const unsigned short *gamma,
                                         unsigned short *y, float eps,
                                         unsigned int rows,
                                         unsigned int width) {
  if (!fused_normq_on())
    return false;
  if (rows == 0 || width == 0)
    return false;
  const bool vec4 =
    cuda_vec4_rows_small(rows) && cuda_vec4_rows_ok(width, x, y);
  auto k = CudaContext::Global().registerCudaKernel(
    FC_QINT4_DP4A_SRC, vec4 ? "rmsnorm_quant_i8_h_v4" : "rmsnorm_quant_i8_h");
  if (!k)
    return false;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // Sizing the quant scratch is a cudaMalloc, which is illegal mid-capture --
  // ensure_buf refuses there and we hand the row back to the plain norm. In
  // practice prefill has already grown the scratch past a single decode row by
  // the time the decode graph is captured, so this is a cold-start guard, not
  // the steady state.
  if (!dp4a_stage_scratch(rows, width))
    return false;
  int m = (int)rows, kk = (int)width;
  int has_gamma = (gamma == nullptr)         ? 0
                  : (!vec4 || cuda_vec4_rows_ok(4, gamma)) ? 1
                                             : 2;
  k->SetKernelArguments(0, &x, sizeof(x));
  k->SetKernelArguments(1, &gamma, sizeof(gamma));
  k->SetKernelArguments(2, &y, sizeof(y));
  k->SetKernelArguments(3, &g_dp4a_q8, sizeof(g_dp4a_q8));
  k->SetKernelArguments(4, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  k->SetKernelArguments(5, &g_dp4a_azp, sizeof(g_dp4a_azp));
  k->SetKernelArguments(6, &m, sizeof(m));
  k->SetKernelArguments(7, &kk, sizeof(kk));
  k->SetKernelArguments(8, &eps, sizeof(eps));
  k->SetKernelArguments(9, &has_gamma, sizeof(has_gamma));
  const int block[3] = {vec4 ? 512 : 256, 1, 1};
  const int grid[3] = {(int)rows, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*k, grid, block))
    return false;
  mark_quant_staged(y, kk);
  StreamManager::Global().maybeFinish();
  return true;
}

bool cuda_fc_qs4cx_dp4a_gemm_fp32(const float *X, const unsigned char *plain_w,
                                  const unsigned short *scales_fp16, float *Y,
                                  unsigned int M, unsigned int N,
                                  unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  auto kq =
    CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC, "act_quant_i8");
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
  if (!dp4a_repack_and_gemm(plain_w, scales_fp16, Y, M, N, K))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

bool cuda_fc_qs4cx_dp4a_gemm_fp16(const unsigned short *Xh,
                                  const unsigned char *plain_w,
                                  const unsigned short *scales_fp16,
                                  unsigned short *Yh, unsigned int M,
                                  unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  const bool q_vec4 = fused_normq_on() && cuda_vec4_rows_small(M) &&
                      cuda_vec4_rows_ok(K, Xh);
  auto kqh = CudaContext::Global().registerCudaKernel(
    FC_QINT4_DP4A_SRC, q_vec4 ? "act_quant_i8_h_v4" : "act_quant_i8_h");
  auto kc =
    CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC, "cvt_f2h");
  if (!kqh || !kc) {
    ml_loge("[CUDA] fc_qint4 dp4a fp16: kernel registration failed");
    return false;
  }
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // Fused decode (M==1): fold the activation quant into the GEMV kernel
  // (the ML Drift paper's 3.7 decode form). Output verified BIT-IDENTICAL
  // to the two-kernel path (fmin/fmax partition-independence + integer
  // dp4a), but on RTX 5060 it measured a stable ~31% decode-TPS LOSS on
  // BOTH the SAFE (per-op drain) and M2B-graph profiles (gemma2 1K:
  // 63-64 -> 43.9 TPS, 3x back-to-back each): the per-block Phase-A
  // barriers serialize the weight streaming that the split kernels
  // pipeline naturally, and on this class of GPU the saved launch +
  // q8 round-trip is cheaper than that stall. Default OFF; opt-in via
  // NNTR_CUDA_FC_FUSED_DECQ=1 for environments where the paper's premise
  // holds (launch/sync-tax-bound: WDDM without graphs, mobile-class) --
  // measure before enabling.
  static const bool _fused_decq = []() {
    const char *e = std::getenv("NNTR_CUDA_FC_FUSED_DECQ");
    return e && e[0] && e[0] != '0';
  }();
  if (M == 1 && _fused_decq) {
    if (!dp4a_repack_and_gemm(plain_w, scales_fp16,
                              reinterpret_cast<float *>(Yh), M, N, K,
                              /** out_fp16 */ 1, Xh))
      return false;
    StreamManager::Global().maybeFinish();
    return true;
  }
  // No float Y staging here: the GEMM writes fp16 directly (out_fp16=1 below),
  // so g_dp4a_yf is unused on this path. Allocating it lazily would cudaMalloc
  // inside a CUDA-graph capture (NNTR_CUDA_GRAPH) on the first captured decode
  // token and invalidate the graph -- so it is deliberately NOT sized here.
  if (!dp4a_stage_scratch(M, K))
    return false;
  int m = (int)M, k = (int)K;
  // 1) int8 activation quant from the fp16 input -- unless g_dp4a_q8 already
  // holds exactly this activation. That happens for every FC group fed by a
  // norm (q/k/v off attention_norm, gate/up off ffn_norm): the fused
  // norm+quant staged it, and the sibling FCs after the first one would
  // otherwise recompute an identical buffer. The guard is pointer + K + "no
  // kernel dispatched since", so a recycled pool address cannot impersonate
  // the staged row.
  if (!quant_staged_for(Xh, k)) {
    kqh->SetKernelArguments(0, &Xh, sizeof(Xh));
    kqh->SetKernelArguments(1, &g_dp4a_q8, sizeof(g_dp4a_q8));
    kqh->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
    kqh->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
    kqh->SetKernelArguments(4, &m, sizeof(m));
    kqh->SetKernelArguments(5, &k, sizeof(k));
    const int qb[3] = {q_vec4 ? 512 : 256, 1, 1};
    const int qg[3] = {(int)M, 1, 1};
    if (!StreamManager::Global().DispatchCommand(*kqh, qg, qb))
      return false;
  }
  // 2) repack + GEMM writing fp16 directly: the float->fp16 conversion is
  // folded into the GEMM epilogue (out_fp16=1), removing the separate cvt_f2h
  // kernel + the FP32 staging buffer (one fewer kernel per FC -- a decode
  // launch-overhead win). (void)kc keeps the registration check above harmless.
  (void)kc;
  if (!dp4a_repack_and_gemm(plain_w, scales_fp16, reinterpret_cast<float *>(Yh),
                            M, N, K,
                            /** out_fp16= */ 1))
    return false;
  // Re-stamp the handoff past this FC's own dispatches so the NEXT sibling on
  // the same activation still sees a valid staging (the GEMM bumped the
  // sequence). With the lever off nothing is ever published, so no FC can
  // skip its quant.
  if (fused_normq_on())
    mark_quant_staged(Xh, k);
  StreamManager::Global().maybeFinish();
  return true;
}

// [i8-jit] Optional transient JIT int8 weight unpack (NNTR_CUDA_I8_JIT): unpack
// the resident dp4a packed-int4 weight to int8 on the GPU per-prefill into a
// reusable scratch, instead of keeping a persistent per-weight int8 cache --
// trades a small per-call unpack cost for the cache's VRAM. Opt-in, default
// off.
// DEFAULT ON. The persistent per-weight int8 cache it replaces loses on BOTH
// axes for a MoE model -- measured at 1,341 tokens: JIT 28.7 GB peak /
// 124.6 TPS against the cache's 37.6 GB / 110.0 TPS -- because 30,720 expert
// weights would need ~30 GiB of [K,N] int8 to cache and each one is used for
// only m_e rows before the next expert's turn.
//
// The per-call unpack looks expensive and is not: the int8 scratch is K*N =
// 1 MB, which fits Orin's 4 MB L2, so the write and the GEMM's read of it stay
// in cache and only the 512 KB int4 payload comes from DRAM. Profiled at 334 ms
// (2.3%) against the IMMA GEMMs' 2,850 ms. And it is still clearly better than
// skipping the Tensor Cores: same 8,353-token prefill, qwen_moe 22,548 ms with
// this path vs 31,608 ms forced onto dp4a.
//
// It remains a workaround for using a library GEMM. cuBLAS int8 accepts only
// int8 operands, so the weights must be materialised; a fused w4a16 kernel
// (vLLM's Marlin) dequantises in-register inside the k-loop and never
// materialises anything. That is the structural gap, not this flag.
static inline bool i8_jit_on() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_CUDA_I8_JIT");
    return !(e != nullptr && e[0] == '0');
  }();
  return v;
}

// NNTR_FC_CUDA_CUBLAS: the int8-IMMA prefill GEMM. Default ON -- it is ~10x the
// dp4a int-ALU GEMM and bit-identical -- with an explicit =0 opting out. The
// test is VALUE-checked, not presence-checked, so =0 disables instead of
// enabling; and it lives here, next to the path it governs, so every consumer
// (the runtime dispatch and the load-time cache prewarm) reads one answer.
static inline bool cublas_i8_on() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_FC_CUDA_CUBLAS");
    return !(e != nullptr && e[0] == '0');
  }();
  return v;
}

// Tiled transpose-unpack: dp4a packed [N, Kh] (byte = plain^0x88, nibbles =
// two's-complement signed 4-bit) -> int8 [K, N]. Reads coalesced along Kh,
// writes coalesced along N via the shared tile.
static const char *I8_JIT_SRC = R"CU(
// Byte-granular fallback for shapes the _v4 tile cannot take. Same contract as
// _v4: RAW payload in, int8 [K,N] and the per-channel rowsum out.
extern "C" __global__ void i8_jit_unpack(const unsigned char *q4,
                                         signed char *w8, int *rowsum, int N,
                                         int K, int Kh) {
  __shared__ signed char t[32][65];
  int nn0 = blockIdx.y * 32, kh0 = blockIdx.x * 32;
  int nn = nn0 + threadIdx.y, kh = kh0 + threadIdx.x;
  if (nn < N && kh < Kh) {
    unsigned char b = q4[(long long)nn * Kh + kh];
    int lo = (int)(b & 0xFu) - 8;
    int hi = (int)((b >> 4) & 0xFu) - 8;
    t[threadIdx.y][2 * threadIdx.x] = (signed char)lo;
    t[threadIdx.y][2 * threadIdx.x + 1] = (signed char)hi;
    // An odd-K pad nibble is stored as 8, i.e. int4 0, so it contributes
    // nothing here and the sum matches weight_rowsum's k in [0,K) range.
    if (rowsum && (lo + hi))
      atomicAdd(&rowsum[nn], lo + hi);
  }
  __syncthreads();
  int k0 = kh0 * 2, wn = nn0 + threadIdx.x;
  for (int kk = threadIdx.y; kk < 64; kk += 32) {
    int k = k0 + kk;
    if (k < K && wn < N)
      w8[(long long)k * N + wn] = t[threadIdx.x][kk];
  }
}

// Vectorized variant (K%8==0 && N%4==0 -- every FC shape this path accepts):
// 64n x 64k tile, 256 threads; uint (4-byte) global loads along Kh and int
// (4-byte) coalesced global stores along N -- runs the ~1.8GB/prefill unpack
// traffic at near-memcpy bandwidth instead of byte-granular transactions.
// Reads the RAW QS4CX payload (offset-binary, value = int4 + 8), not the
// XOR'd DevWeightQ copy -- so `repack_plain_i4` and its 15.1 GiB of derived
// weights are not needed on this path at all. The decode is one subtraction
// instead of the old `((b&0xF)^8)&0xF)-8`, which was undoing a XOR that only
// existed because the input had already been XOR'd into a second buffer.
//
// It also emits the per-output-channel ROWSUM, which is the only other thing
// the dp4a cache was being built for here. That is free: this kernel already
// reads every weight byte. Block-local accumulation into shared memory first,
// so the global traffic is 64 int atomics per block rather than 512. Integer
// addition is exact and associative, so the atomics are deterministic in value
// -- unlike an fp atomicAdd, which is why that was refused elsewhere.
extern "C" __global__ void i8_jit_unpack_v4(const unsigned char *q4,
                                            signed char *w8, int *rowsum, int N,
                                            int K, int Kh) {
  // [k_local][n_local], row stride 68 so the 4-byte store below is aligned --
  // 68 is a multiple of 4, but the ARRAY BASE also has to be, and a
  // `signed char` array is only guaranteed byte alignment. Say so explicitly
  // rather than relying on what ptxas happens to do.
  __shared__ alignas(4) signed char t[64][68];
  __shared__ int rs[64];            // partial rowsum for this tile's 64 n
  const int nn0 = blockIdx.y * 64;
  const int kh0 = blockIdx.x * 32; // bytes of Kh covered by this tile
  const int tid = threadIdx.x;     // 256 threads
  if (tid < 64)
    rs[tid] = 0;
  __syncthreads();
  for (int rep = 0; rep < 2; ++rep) {
    int idx = tid + rep * 256;
    int nn = idx >> 3;   // 0..63
    int kb4 = idx & 7;   // which 4-byte group in the 32-byte span
    int n = nn0 + nn;
    int khb = kh0 + kb4 * 4;
    int part = 0;
    if (n < N && khb + 3 < Kh) {
      unsigned int v = *reinterpret_cast<const unsigned int *>(
        q4 + (long long)n * Kh + khb);
      int kl = kb4 * 8;
      for (int j = 0; j < 4; ++j) {
        unsigned int b = (v >> (8 * j)) & 0xFFu;
        int lo = (int)(b & 0xFu) - 8;
        int hi = (int)((b >> 4) & 0xFu) - 8;
        t[kl + 2 * j][nn] = (signed char)lo;
        t[kl + 2 * j + 1][nn] = (signed char)hi;
        part += lo + hi;
      }
    } else if (n < N) { // Kh tail (unused when K%8==0, kept for safety)
      for (int j = 0; j < 4; ++j) {
        int kb = khb + j;
        if (kb < Kh) {
          unsigned char b = q4[(long long)n * Kh + kb];
          int kl = kb4 * 8 + 2 * j;
          int lo = (int)(b & 0xFu) - 8;
          int hi = (int)((b >> 4) & 0xFu) - 8;
          t[kl][nn] = (signed char)lo;
          t[kl + 1][nn] = (signed char)hi;
          part += lo + hi;
        }
      }
    }
    if (n < N && part)
      atomicAdd(&rs[nn], part);
  }
  __syncthreads();
  if (rowsum && tid < 64 && nn0 + tid < N && rs[tid])
    atomicAdd(&rowsum[nn0 + tid], rs[tid]);
  const int k0 = kh0 * 2;
  for (int rep = 0; rep < 4; ++rep) {
    int idx = tid + rep * 256;
    int kl = idx >> 4; // 0..63
    int ni = idx & 15; // 16 ints cover 64 n
    int k = k0 + kl;
    int n = nn0 + ni * 4;
    if (k < K && n + 3 < N) {
      int val = *reinterpret_cast<const int *>(&t[kl][ni * 4]);
      *reinterpret_cast<int *>(w8 + (long long)k * N + n) = val;
    } else if (k < K) {
      for (int j = 0; j < 4; ++j)
        if (n + j < N)
          w8[(long long)k * N + n + j] = t[kl][ni * 4 + j];
    }
  }
}
)CU";

// w4a8 on the INT8 Tensor Cores via cuBLAS (prefill FC). Same quant scheme as
// the dp4a path -- per-row asym int8 activation + symmetric int4 weight -- but
// the int8xint8->int32 GEMM runs on IMMA Tensor Cores instead of __dp4a on the
// int ALU (~10x the GEMM throughput at prefill M). The int32 accumulate is
// exact so the result is bit-identical to dp4a; the int4->int8 weight unpack is
// cached (one-time) to keep it off the per-call critical path.

bool cuda_fc_qs4cx_cublas_i8_gemm_fp16(const unsigned short *Xh,
                                       const unsigned char *plain_w,
                                       const unsigned short *scales_fp16,
                                       unsigned short *Yh, unsigned int M,
                                       unsigned int N, unsigned int K) {
  // NNTR_FC_CUDA_CUBLAS=0 opts this path out; report failure so the caller's
  // fall-through chain (dp4a, then the naive GEMM) takes over. Enforcing the
  // lever at the entry point covers every caller -- the runtime FC dispatch
  // only knows the M>=32 prefill shape, not the lever.
  if (!cublas_i8_on())
    return false;
  if (M == 0 || N == 0 || K == 0)
    return true;
  // Stand aside for imma_gemm_pipe when it is eligible. Same int8 Tensor
  // Cores, but it reads the packed int4 straight out of the weight and unpacks
  // into shared memory while staging, so it never materialises the K*N int8
  // scratch this path needs -- and it wins by 7.2x end to end on gemma4. The
  // conditions MIRROR use_pipe in dp4a_repack_and_gemm and are deliberately
  // the stricter reading: the alignment test is on plain_w even though a
  // cached copy would make it moot, so an odd shape stays on cuBLAS rather
  // than falling to v1. Returning false hands the call to the dp4a arm, which
  // is where the tile lives (see the fall-through chain in
  // cuda_compute_ops.cpp).
  if (imma_tile_level() >= 2 && M >= 8u && (K % 64u) == 0u &&
      (reinterpret_cast<uintptr_t>(plain_w) & 7u) == 0u)
    return false;
  const bool q_vec4 = fused_normq_on() && cuda_vec4_rows_small(M) &&
                      cuda_vec4_rows_ok(K, Xh);
  auto kqh = CudaContext::Global().registerCudaKernel(
    FC_QINT4_DP4A_SRC, q_vec4 ? "act_quant_i8_h_v4" : "act_quant_i8_h");
  auto kde = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "dequant_i32_fp16");
  if (!kqh || !kde) {
    ml_loge("[CUDA] fc_qint4 cublas-i8: kernel registration failed");
    return false;
  }
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // cuBLAS int8 IMMA requires the GEMM dims to be multiples of 32 (measured:
  // M=260/272 -> CUBLAS_STATUS_NOT_SUPPORTED, 256/320/512 OK). The prefill
  // token count M is arbitrary (e.g. 511), so pad the activation row count up
  // to a multiple of 32 for the GEMM only -- the extra rows are computed from
  // (harmless int8) scratch and ignored by the epilogue, which writes just the
  // real M rows. N and K are multiples of 32 by the load invariant.
  const unsigned Mpad = ((M + 31u) / 32u) * 32u;
  if (!dp4a_stage_scratch(Mpad, K))
    return false;
  const int m = (int)M, n = (int)N, k = (int)K, mpad = (int)Mpad;

  // 1) int8 activation quant from the fp16 input (reuse the dp4a quantizer).
  // Skip when this exact (Xh,K) was just quantized into g_dp4a_q8 by a sibling
  // FC (q/k/v share attention_norm; gate/up share ffn_norm) -- the buffer still
  // holds it. See g_last_quant_xh above.
  // Opt-in: measured gain is within the thermal noise floor on Orin (act_quant
  // is not on the critical path -- the GEMM is), so default OFF; correct +
  // ready if a less-throttled host or a power budget makes the redundant
  // launches matter.
  // DEFAULT ON since the 35B. The old default-off was measured on ATTENTION,
  // where q/k/v share attention_norm and the saving is one act-quant in three
  // across a handful of FCs per layer -- genuinely inside the thermal noise.
  // A MoE layer is a different population: gate and up share the gathered
  // activation for EVERY routed expert, so the dedup removes one launch in
  // three out of 61,440 fc calls per prefill. Measured on an 8,353-token
  // prefill at chunk 4096: 234.3 -> 238.7 TPS, qwen_moe 22,994 -> 22,409 ms,
  // with act_quant_i8_h at 22,209 launches / 903 ms before. Output unchanged
  // (the reuse is exact -- same buffer, same K, guarded by dispatchSeq).
  // NNTR_QUANT_DEDUP=0 opts out.
  static const bool quant_dedup = []() {
    const char *e = std::getenv("NNTR_QUANT_DEDUP");
    return !(e != nullptr && e[0] == '0');
  }();
  const bool reuse_quant = quant_dedup && quant_staged_for(Xh, k);
  if (!reuse_quant) {
    kqh->SetKernelArguments(0, &Xh, sizeof(Xh));
    kqh->SetKernelArguments(1, &g_dp4a_q8, sizeof(g_dp4a_q8));
    kqh->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
    kqh->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
    kqh->SetKernelArguments(4, &m, sizeof(m));
    kqh->SetKernelArguments(5, &k, sizeof(k));
    const int qb[3] = {q_vec4 ? 512 : 256, 1, 1};
    const int qg[3] = {(int)M, 1, 1};
    if (!StreamManager::Global().DispatchCommand(*kqh, qg, qb))
      return false;
  }

  // 2) int8 weight [K,N] + per-channel rowsum. [i8-jit] JIT mode transpose-
  // unpacks the RESIDENT dp4a packed copy into a reusable scratch (nothing
  // stays resident; rowsum shared with the dp4a cache -- same values); else
  // the persistent per-weight cache (one-time unpack).
  signed char *w8src = nullptr;
  int *rowsum = nullptr;
  if (i8_jit_on()) {
    // No DevWeightQ here. The unpack reads the RAW payload and emits the
    // rowsum in the same pass, so this path never builds the derived cache --
    // which a profile showed costing weight_rowsum 748 ms + repack_plain_i4
    // 253 ms (15.3% of ALL GPU time) plus ~19,000 cudaMallocs on a 2,774-token
    // prefill, because every routed expert misses it and every miss allocates
    // twice. The payload must still be provably device-readable, which
    // ensure_dp4a_cache_locked used to check on our behalf.
    if (!plain_bindable(plain_w))
      return false;
    static signed char *jit_w8 = nullptr;
    static size_t jit_cap = 0;
    static int *jit_rs = nullptr;
    static size_t jit_rs_cap = 0;
    if (!ensure_buf((void **)&jit_w8, &jit_cap, (size_t)K * N) ||
        !ensure_buf((void **)&jit_rs, &jit_rs_cap, sizeof(int) * (size_t)N))
      return false;
    // The rowsum is accumulated with atomics, so it must start at zero.
    if (cudaMemsetAsync(jit_rs, 0, sizeof(int) * (size_t)N,
                        StreamManager::Global().GetStream()) != cudaSuccess)
      return false;
    // Vectorized transpose for 8|K && 4|N; byte-granular fallback otherwise.
    //
    // The THIRD condition is not optional and its absence was a hard failure,
    // not a slow path: _v4 reads the payload as `unsigned int`, so besides
    // 4|Kh (which 8|K gives) the payload BASE must be 4-byte aligned. A QS4CX
    // tensor starts wherever the manifest put it in the weight arena, and
    // nothing upstream rounds that up. gemma4 has such a tensor; the 35B does
    // not, which is why this shipped. The symptom is the worst kind: an
    // unaligned global read raises CUDA_ERROR_MISALIGNED_ADDRESS, which is
    // STICKY -- every later call in the context fails, cublasCreate included,
    // so the model dies far from here with "pack before run model" and a
    // cuModuleLoadData that cannot possibly be misaligned. See the family in
    // §5 of the hand-off: an error message that is the symptom of a dead
    // context.
    const bool vec_ok = ((K & 7u) == 0u) && ((N & 3u) == 0u) &&
                        ((reinterpret_cast<uintptr_t>(plain_w) & 3u) == 0u);
    auto ku = CudaContext::Global().registerCudaKernel(
      I8_JIT_SRC, vec_ok ? "i8_jit_unpack_v4" : "i8_jit_unpack");
    if (!ku)
      return false;
    const int khi = (int)((K + 1u) / 2u);
    ku->SetKernelArguments(0, &plain_w, sizeof(plain_w));
    ku->SetKernelArguments(1, &jit_w8, sizeof(jit_w8));
    ku->SetKernelArguments(2, &jit_rs, sizeof(jit_rs));
    ku->SetKernelArguments(3, &n, sizeof(n));
    ku->SetKernelArguments(4, &k, sizeof(k));
    ku->SetKernelArguments(5, &khi, sizeof(khi));
    const int ub[3] = {vec_ok ? 256 : 32, vec_ok ? 1 : 32, 1};
    const int ug[3] = {(khi + 31) / 32,
                       vec_ok ? ((int)N + 63) / 64 : ((int)N + 31) / 32, 1};
    if (!StreamManager::Global().DispatchCommand(*ku, ug, ub))
      return false;
    w8src = jit_w8;
    rowsum = jit_rs;
  } else {
    // int8 weight [K,N] + per-channel rowsum from the persistent per-weight
    // cache (one-time unpack; weights are static).
    DevWeightI8 *dw8 = ensure_i8_cache_locked(plain_w, N, K);
    if (!dw8)
      return false;
    w8src = dw8->w8;
    rowsum = dw8->rowsum;
  }

  // 3) int32 GEMM output scratch [Mpad,N] (+tail pad: IMMA can write/read C in
  // wide vectorized tiles past the last element on large shapes).
  if (!ensure_buf((void **)&g_i8_c, &g_i8_c_cap,
                  sizeof(int) * (size_t)Mpad * N + FC_I8_TAIL_PAD))
    return false;

  // 4) INT8 IMMA GEMM on the Tensor Cores (Mpad rows; same backend stream as
  // the kernels). C is [Mpad,N] row-major; the real M rows are at the same
  // offsets so the epilogue reads C[m*N+n] for m<M directly.
  if (!BlasManager::Global().igemmRowMajor(mpad, n, k, g_dp4a_q8, w8src,
                                           g_i8_c))
    return false;

  // 5) dequant epilogue (bit-identical math to the dp4a kernel) -> fp16 Y.
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
  // Re-stamp past this FC's own dispatches (the epilogue bumped the sequence)
  // so a sibling prefill FC on the same activation can still reuse the quant.
  if (quant_dedup)
    mark_quant_staged(Xh, k);
  StreamManager::Global().maybeFinish();
  // Catch an ASYNC failure in the cuBLAS IMMA GEMM / epilogue (the sync cuBLAS
  // status was already checked). On Orin a large-M IMMA can fault at runtime
  // and leave a STICKY cuda error -- which then makes the NEXT layer's
  // cudaPointerGetAttributes (rms_norm dev_ok gate) fail, dropping rms_norm to
  // its host path that reads device/managed activations under cMA=0 -> SIGSEGV.
  // Clearing + returning false makes the caller fall back to the (correct) dp4a
  // GEMM cleanly instead of corrupting the rest of the forward.
  {
    cudaError_t _e = cudaGetLastError();
    if (_e != cudaSuccess) {
      if (std::getenv("NNTR_IGEMM_DBG"))
        std::fprintf(
          stderr,
          "[IGEMM] async error after GEMM M=%d N=%d K=%d: %s -> dp4a "
          "fallback\n",
          m, n, k, cudaGetErrorString(_e));
      return false;
    }
  }
  return true;
}
// [wprefetch] Migrate a QS4CX weight's managed plain payload (+ fp32 scale
// tail) to the device with cudaMemPrefetchAsync, so the FC bytes leave host
// RSS and the GEMM reads them from VRAM. Discrete only (managed pages migrate).
bool cuda_fc_qs4cx_prefetch_weight(const unsigned char *plain_w, unsigned int N,
                                   unsigned int K) {
  if (plain_w == nullptr || N == 0 || K == 0)
    return false;
  if (ContextManager::Global().isIntegrated())
    return false;
  cudaPointerAttributes attr{};
  if (cudaPointerGetAttributes(&attr, plain_w) != cudaSuccess ||
      attr.type != cudaMemoryTypeManaged) {
    cudaGetLastError();
    return false;
  }
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  const size_t bytes = (size_t)N * ((K + 1u) / 2u) + (size_t)N * sizeof(float);
  // CUDA 13 signature (cudaMemLocation + flags).
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeDevice;
  loc.id = dev;
  if (cudaMemPrefetchAsync(plain_w, bytes, loc, /** flags */ 0,
                           StreamManager::Global().GetStream()) !=
      cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
}

// --- load-time prewarm + teardown lifecycle -------------------------------

// Mark a weight exempt from the eager load-time cuBLAS-i8 [K,N] build (see
// g_i8_exempt). Called at load time before the prewarm walk.
void cuda_fc_qs4cx_prewarm_exempt_i8(const void *plain_w) {
  g_i8_exempt.insert(plain_w);
}

bool cuda_fc_qs4cx_prewarm(const unsigned char *plain_w, unsigned int N,
                           unsigned int K, const char *cache_name) {
  if (plain_w == nullptr || N == 0 || K == 0)
    return true;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  if (g_dp4a_plain_cache.count(plain_w))
    return true; // already cached
  const size_t Kh = (K + 1u) / 2u;
  auto &tm = nntrainer::ThreadManager::Global();
  const bool pack_cache = cuda_pack::enabled() && cache_name != nullptr;
  // Record names are "<weight name>#<kind>"; the weight name comes from the
  // graph, so it is stable across launches -- unlike the plain pointer, which
  // is exactly what must never key a persistent cache.
  const std::string rec_dp4a =
    pack_cache ? std::string(cache_name) + "#dp4a" : std::string();
  const std::string rec_i8 =
    pack_cache ? std::string(cache_name) + "#i8" : std::string();

  // Build + upload in bounded chunks: a full host mirror of the untied
  // lm_head (N=262144) is ~350MB packed + ~700MB int8 and those transients
  // WERE the process peak RSS once the Section-A copy was gone (RSS timeline:
  // a +1GB step right at the peak, late in load). ~64MB chunks keep the
  // prewarm off the peak entirely; results are byte-identical (same values,
  // same device offsets).
  static constexpr size_t PREWARM_CHUNK_BYTES = 64u << 20;

  DevWeightQ dw;
  if (cudaMalloc(&dw.plain, (size_t)N * Kh) != cudaSuccess)
    return false;
  if (cudaMalloc(&dw.rowsum, sizeof(int) * (size_t)N) != cudaSuccess) {
    cudaFree(dw.plain);
    return false;
  }
  // [pack-cache HIT] the permute + row-sum fold are a deterministic pure
  // function of the plain nibbles, so a validated pack record can be uploaded
  // straight from the pack mmap: no host permute, no staging vector, and the
  // consumed file pages are dropped per chunk so residency stays at one chunk.
  bool dp4a_from_cache = false;
  if (pack_cache) {
    cuda_pack::Hit hit;
    if (cuda_pack::lookup(rec_dp4a.c_str(), N, K, Kh, (size_t)N * Kh, hit)) {
      const double t0 = g_stat_clock();
      bool ok = true;
      for (size_t off = 0; off < hit.payload_len && ok;
           off += PREWARM_CHUNK_BYTES) {
        const size_t len = std::min(PREWARM_CHUNK_BYTES, hit.payload_len - off);
        ok = cudaMemcpy(dw.plain + off, hit.payload + off, len,
                        cudaMemcpyHostToDevice) == cudaSuccess;
        cuda_pack::payload_consumed(hit.payload + off, len);
      }
      ok = ok && cudaMemcpy(dw.rowsum, hit.rowsum, sizeof(int) * (size_t)N,
                            cudaMemcpyHostToDevice) == cudaSuccess;
      if (ok) {
        dp4a_from_cache = true;
        g_prewarm_hit_ms += g_stat_clock() - t0;
        g_prewarm_hit_bytes += hit.payload_len;
      }
      // upload error: fall through to the derive (silent fallback)
    }
  }
  if (!dp4a_from_cache) {
    // packed int4 [N][Kh] in row chunks (rows are contiguous on both sides).
    const size_t chunk_rows =
      std::max<size_t>(1, std::min<size_t>(N, PREWARM_CHUNK_BYTES / Kh));
    std::vector<signed char> packed(chunk_rows * Kh);
    std::vector<int> rowsum(N, 0);
    cuda_pack::RecordWriter *rw =
      pack_cache
        ? cuda_pack::begin_record(rec_dp4a.c_str(), N, K, Kh, (size_t)N * Kh)
        : nullptr;
    for (size_t n0 = 0; n0 < N; n0 += chunk_rows) {
      const size_t rows = std::min(chunk_rows, (size_t)N - n0);
      const double t0 = g_stat_clock();
      tm.parallel_for(0, rows, [&](size_t r) {
        const unsigned char *src = plain_w + (n0 + r) * Kh;
        signed char *prow = packed.data() + r * Kh;
        long acc = 0;
        for (size_t kb = 0; kb < Kh; ++kb) {
          const unsigned char b = src[kb];
          prow[kb] = (signed char)(b ^ 0x88);
          // odd-K pad nibble is stored 8 (= int4 0), so it adds 0 here --
          // same rowsum the old k1<K guard produced.
          acc += ((int)(b & 0xF) - 8) + ((int)((b >> 4) & 0xF) - 8);
        }
        rowsum[n0 + r] = (int)acc;
      });
      const double t1 = g_stat_clock();
      // tee the derived chunk to the pack temp file (page-cache speed, no
      // extra staging: the bytes are already in `packed`)
      if (rw)
        cuda_pack::record_write(rw, n0 * Kh, packed.data(), rows * Kh);
      const double t2 = g_stat_clock();
      cudaMemcpy(dw.plain + n0 * Kh, packed.data(), rows * Kh,
                 cudaMemcpyHostToDevice);
      g_prewarm_derive_ms += t1 - t0;
      g_prewarm_tee_ms += t2 - t1;
      g_prewarm_upload_ms += g_stat_clock() - t2;
      g_prewarm_derive_bytes += rows * Kh;
    }
    cudaMemcpy(dw.rowsum, rowsum.data(), sizeof(int) * (size_t)N,
               cudaMemcpyHostToDevice);
    if (rw)
      cuda_pack::commit_record(rw, rowsum.data(), N);
  }
  g_dp4a_plain_cache.emplace(plain_w, dw);
  // The payload was just read successfully, so any drop mark on this address
  // belongs to a previous model generation (see forget_dropped_plain).
  forget_dropped_plain(plain_w);

  // Also prewarm the cuBLAS int8 [K,N] weight cache when the cuBLAS prefill FC
  // path is on: otherwise its one-time GPU repack (repack_plain_i8_kn, ~32% of
  // cold prefill GPU time) runs on the first prefill instead of at load.
  // Mirrors repack_plain_i8_kn (w8[k*N+n]=int4(n,k)) + weight_rowsum_kn
  // bit-exactly. Chunked along K ([k0,k1) rows of the [K,N] buffer are
  // contiguous on both sides); the per-channel rowsum accumulates across
  // chunks.
  // Only build eagerly when the lever is explicitly set: with it unset the
  // runtime path still uses cuBLAS but builds this [K,N] cache lazily, and an
  // FC that never reaches the M>=32 gate would otherwise pay the VRAM for
  // nothing (see g_i8_exempt). cublas_i8_on() supplies the =0 opt-out so the
  // two consumers cannot disagree about what =0 means.
  static const bool _cb_set = std::getenv("NNTR_FC_CUDA_CUBLAS") != nullptr;
  if (_cb_set && cublas_i8_on() && !i8_jit_on() &&
      !g_i8_weight_cache.count(plain_w) && !g_i8_exempt.count(plain_w)) {
    const size_t chunk_k =
      std::max<size_t>(1, std::min<size_t>(K, PREWARM_CHUNK_BYTES / N));
    std::vector<signed char> w8(chunk_k * (size_t)N);
    std::vector<long> rs8(N, 0);
    DevWeightI8 dw8;
    if (alloc_i8_weight(&dw8.w8, N, K) == cudaSuccess &&
        cudaMalloc(&dw8.rowsum, sizeof(int) * (size_t)N) == cudaSuccess) {
      // [pack-cache HIT] the [K,N] transpose-unpack is the most expensive
      // derive on this path (column-strided writes) and is likewise a pure
      // function of the plain nibbles.
      bool i8_from_cache = false;
      if (pack_cache) {
        cuda_pack::Hit hit;
        if (cuda_pack::lookup(rec_i8.c_str(), N, K, N, (size_t)K * N, hit)) {
          const double t0 = g_stat_clock();
          bool ok = true;
          for (size_t off = 0; off < hit.payload_len && ok;
               off += PREWARM_CHUNK_BYTES) {
            const size_t len =
              std::min(PREWARM_CHUNK_BYTES, hit.payload_len - off);
            ok = cudaMemcpy(dw8.w8 + off, hit.payload + off, len,
                            cudaMemcpyHostToDevice) == cudaSuccess;
            cuda_pack::payload_consumed(hit.payload + off, len);
          }
          ok = ok && cudaMemcpy(dw8.rowsum, hit.rowsum, sizeof(int) * (size_t)N,
                                cudaMemcpyHostToDevice) == cudaSuccess;
          if (ok) {
            i8_from_cache = true;
            g_prewarm_hit_ms += g_stat_clock() - t0;
            g_prewarm_hit_bytes += hit.payload_len;
          }
        }
      }
      cuda_pack::RecordWriter *rw =
        (pack_cache && !i8_from_cache)
          ? cuda_pack::begin_record(rec_i8.c_str(), N, K, N, (size_t)K * N)
          : nullptr;
      for (size_t k0 = 0; !i8_from_cache && k0 < K; k0 += chunk_k) {
        const size_t ks = std::min(chunk_k, (size_t)K - k0);
        const double t0 = g_stat_clock();
        tm.parallel_for(0, (size_t)N, [&](size_t n) {
          const unsigned char *src = plain_w + n * Kh;
          long acc = 0;
          for (size_t kk = k0; kk < k0 + ks; ++kk) {
            const unsigned char b = src[kk >> 1];
            const int v = (int)((kk & 1) ? ((b >> 4) & 0xF) : (b & 0xF)) - 8;
            w8[(kk - k0) * N + n] = (signed char)v;
            acc += v;
          }
          rs8[n] += acc;
        });
        const double t1 = g_stat_clock();
        if (rw)
          cuda_pack::record_write(rw, k0 * (size_t)N, w8.data(),
                                  ks * (size_t)N);
        const double t2 = g_stat_clock();
        cudaMemcpy(dw8.w8 + k0 * N, w8.data(), ks * (size_t)N,
                   cudaMemcpyHostToDevice);
        g_prewarm_derive_ms += t1 - t0;
        g_prewarm_tee_ms += t2 - t1;
        g_prewarm_upload_ms += g_stat_clock() - t2;
        g_prewarm_derive_bytes += ks * (size_t)N;
      }
      if (!i8_from_cache) {
        std::vector<int> rs8i(N);
        for (size_t n = 0; n < N; ++n)
          rs8i[n] = (int)rs8[n];
        cudaMemcpy(dw8.rowsum, rs8i.data(), sizeof(int) * (size_t)N,
                   cudaMemcpyHostToDevice);
        if (rw)
          cuda_pack::commit_record(rw, rs8i.data(), N);
      }
      g_i8_weight_cache.emplace(plain_w, dw8);
    } else {
      // A failed EAGER build used to be silent and to leave the lazy in-path
      // build as the ONLY builder for this weight -- and the lazy build runs
      // after the load-time walk may have discarded the plain payload's pages,
      // at which point plain_bindable() refuses it and the FC quietly loses the
      // Tensor-Core prefill path. Say so, and mark the weight exempt so a
      // later prewarm does not silently retry an allocation that just failed.
      // Clear the sticky cudaMalloc error too: leaving it set makes the NEXT
      // layer's cudaPointerGetAttributes() fail (see the dev_ok gates), which
      // is how an allocation failure here turns into a host-path crash there.
      if (dw8.w8)
        cudaFree(dw8.w8);
      cudaGetLastError();
      g_i8_exempt.insert(plain_w);
      ml_logw("[CUDA] fc_qint4: eager cuBLAS-i8 [K,N] weight cache alloc "
              "failed for N=%u K=%u (%zu MiB); this FC keeps the dp4a int-ALU "
              "GEMM at prefill. Exempted from further eager builds.",
              N, K, ((size_t)N * K + FC_I8_TAIL_PAD) >> 20);
    }
  }
  return true;
}

void cuda_fc_qs4cx_prewarm_stats(double *derive_ms, double *upload_ms,
                                 double *tee_ms, double *hit_ms,
                                 size_t *derive_bytes, size_t *hit_bytes) {
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  if (derive_ms)
    *derive_ms = g_prewarm_derive_ms;
  if (upload_ms)
    *upload_ms = g_prewarm_upload_ms;
  if (tee_ms)
    *tee_ms = g_prewarm_tee_ms;
  if (hit_ms)
    *hit_ms = g_prewarm_hit_ms;
  if (derive_bytes)
    *derive_bytes = g_prewarm_derive_bytes;
  if (hit_bytes)
    *hit_bytes = g_prewarm_hit_bytes;
}

bool cuda_fc_qint4_dp4a_prewarm(unsigned int maxM, unsigned int maxK,
                                unsigned int maxN) {
  // Pre-grow the activation-quant scratch (q8 + per-row scale/zero-point) to
  // the given decode bounds so the M=1 decode FC never cudaMallocs inside a
  // CUDA-graph capture (an in-capture malloc aborts the capture). maxN is
  // accepted for signature stability with the prefill-side scratch but the
  // decode dp4a path writes into the caller's output tensor -- there is no
  // N-sized scratch to grow here; the in-path isCapturing() guards remain the
  // safety net for anything outside these bounds.
  (void)maxN;
  if (maxM == 0 || maxK == 0)
    return true;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  return dp4a_stage_scratch(maxM, maxK);
}

void cuda_fc_qs4cx_release_weight_caches() {
  // Teardown for a model reload lifecycle: free every pointer-keyed derived
  // weight cache (dp4a packed int4 + cuBLAS int8) so a reloaded model does not
  // leak the previous generation's VRAM. The fp16-scale UVM side buffers stay
  // by design -- their cache is documented process-lifetime (keyed by the fp32
  // scale pointer, no erase).
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  for (auto &kv : g_dp4a_plain_cache) {
    cudaFree(kv.second.plain);
    cudaFree(kv.second.rowsum);
  }
  g_dp4a_plain_cache.clear();
  for (auto &kv : g_i8_weight_cache) {
    cudaFree(kv.second.w8);
    cudaFree(kv.second.rowsum);
  }
  g_i8_weight_cache.clear();
  // Drop marks describe pages of the generation being torn down. Keeping them
  // would make cuda_fc_qs4cx_plain_dropped() answer true for a RELOADED
  // model's fresh payload that happens to reuse the same heap address, and the
  // host-dot() refusal in CudaComputeOps::fc would then abort a run holding
  // valid bytes. Same lock order as everywhere else: g_dp4a_mtx (held) then
  // g_dropped_mtx.
  {
    std::lock_guard<std::mutex> dlk(g_dropped_mtx);
    g_dropped_plain.clear();
  }
  // The eager-build exemptions are per-generation too: a reloaded model gets
  // fresh pointers, and a stale entry keyed on a recycled address would skip
  // an eager build the new model wants.
  g_i8_exempt.clear();
}

} // namespace nntrainer::cuda

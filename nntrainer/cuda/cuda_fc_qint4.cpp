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
#include <cuda_elementwise.h> // cuda_add_pending_take / cuda_add_fp16
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

// Hardware half<->float conversion. The former ~20-op software bodies were
// verified BIT-IDENTICAL to these instructions over all 65536 half patterns
// and 4M random floats (see vq_h2f/vq_f2h below, same swap); every scalar
// kernel in this source (rmsnorm_quant_i8_h's three K-wide passes, the
// act_quant fallbacks, the gemv arms) inherits the swap with no value change.
__device__ __forceinline__ float dp4a_h2f(unsigned short h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}

// float -> fp16 (IEEE half), round to nearest even.
__device__ __forceinline__ unsigned short dp4a_f2h(float f) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
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

// Prefill-shaped variant of act_quant_i8_h_v4: same vector loads, hardware
// cvt, and warp-shuffle reduction, but NO per-thread register carry -- the
// carry is a decode win and a measured ~5% prefill LOSS (32 extra registers
// cut blocks/SM at large M), which is why the v4 kernel is gated to
// rows<=32. The quant pass re-reads the row instead; a K-wide row is
// L1-hot by then. BIT-IDENTICAL to act_quant_i8_h for the same reason v4
// is: min/max are order-independent, the conversions produce the same
// values, and the rint/clamp is unchanged.
__global__ void act_quant_i8_h_v4p(const unsigned short *Xh, signed char *q8,
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
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 f = vq_load4(xv[i]);
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
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 f = vq_load4(xv[i]);
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

// Prefill-shaped variant of rmsnorm_quant_i8_h_v4: same vector loads,
// hardware cvt, and warp-shuffle reductions, but NO per-thread register
// carry -- the same trade as act_quant_i8_h_v4p (the carry is a decode win
// and a measured prefill LOSS: 32 extra registers cut blocks/SM at large M).
// Passes 2 and 3 re-read rows that are L1-hot from the pass before them.
// Numerics match v4, not the scalar kernel: the sum of squares is reduced in
// vector-of-4 order, so `inv` can differ from the scalar kernel by an ulp;
// everything downstream of `inv` (including the quant of the ROUNDED fp16
// stores) is the scalar arithmetic unchanged.
__global__ void rmsnorm_quant_i8_h_v4p(const unsigned short *x,
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
  float p = 0.f;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 f = vq_load4(xv[i]);
    p += f.x * f.x + f.y * f.y + f.z * f.z + f.w * f.w;
  }
  VQ_REDUCE(ssq, p, vq_add, 0.f);
  const float inv = rsqrtf(ssq[0] / (float)K + eps);

  float lmn = 0.f, lmx = 0.f;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 f = vq_load4(xv[i]);
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
    float4 r = make_float4(vq_h2f(h0), vq_h2f(h1), vq_h2f(h2), vq_h2f(h3));
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
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 r = vq_load4(yv[i]);
    int q0 = max(-128, min(127, (int)rintf(r.x * scale_q) + zp));
    int q1 = max(-128, min(127, (int)rintf(r.y * scale_q) + zp));
    int q2 = max(-128, min(127, (int)rintf(r.z * scale_q) + zp));
    int q3 = max(-128, min(127, (int)rintf(r.w * scale_q) + zp));
    q32[i] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) |
             ((q3 & 0xFF) << 24);
  }
}

// Residual add fused into the norm+quant pair (the graph's add->norm pairs:
// decoder_add->ffn_norm and decoder_output->next attention_norm). Pass 1
// performs the add with cuda_add_fp16's exact arithmetic -- r = f2h(h2f(a)+
// h2f(b)) -- writes the residual plane R (its later consumers read it
// unchanged), and accumulates the sum of squares over the ROUNDED r, which
// is byte-for-byte what the split flow's norm reads back from R. Passes 2/3
// are rmsnorm_quant_i8_h_v4p's, re-reading the L1-hot R instead of a cold
// DRAM plane. BIT-IDENTICAL to add_fp16_v8 + rmsnorm_quant_i8_h_v4p.
__global__ void add_rmsnorm_quant_i8_h_v4p(
  const unsigned short *xa, const unsigned short *xb,
  const unsigned short *gamma, unsigned short *r, unsigned short *y,
  signed char *q8, float *ascale, int *azp, int M, int K, float eps,
  int has_gamma) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  const uint2 *av = (const uint2 *)(xa + (long)m * K);
  const uint2 *bv = (const uint2 *)(xb + (long)m * K);
  const uint2 *gv = (const uint2 *)gamma;
  uint2 *rv = (uint2 *)(r + (long)m * K);
  uint2 *yv = (uint2 *)(y + (long)m * K);
  int *q32 = (int *)(q8 + (long)m * K);
  const int nv = K >> 2;
  __shared__ float ssq[32];
  __shared__ float smn[32];
  __shared__ float smx[32];
  float p = 0.f;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    const uint2 ra = av[i];
    const uint2 rb = bv[i];
    const unsigned short h0 =
      vq_f2h(vq_h2f((unsigned short)(ra.x & 0xFFFFu)) +
             vq_h2f((unsigned short)(rb.x & 0xFFFFu)));
    const unsigned short h1 = vq_f2h(vq_h2f((unsigned short)(ra.x >> 16)) +
                                     vq_h2f((unsigned short)(rb.x >> 16)));
    const unsigned short h2 =
      vq_f2h(vq_h2f((unsigned short)(ra.y & 0xFFFFu)) +
             vq_h2f((unsigned short)(rb.y & 0xFFFFu)));
    const unsigned short h3 = vq_f2h(vq_h2f((unsigned short)(ra.y >> 16)) +
                                     vq_h2f((unsigned short)(rb.y >> 16)));
    uint2 o;
    o.x = (unsigned int)h0 | ((unsigned int)h1 << 16);
    o.y = (unsigned int)h2 | ((unsigned int)h3 << 16);
    rv[i] = o;
    const float f0 = vq_h2f(h0), f1 = vq_h2f(h1);
    const float f2 = vq_h2f(h2), f3 = vq_h2f(h3);
    p += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
  }
  VQ_REDUCE(ssq, p, vq_add, 0.f);
  const float inv = rsqrtf(ssq[0] / (float)K + eps);

  float lmn = 0.f, lmx = 0.f;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 f = vq_load4(rv[i]);
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
    float4 rr = make_float4(vq_h2f(h0), vq_h2f(h1), vq_h2f(h2), vq_h2f(h3));
    lmn = fminf(lmn, fminf(fminf(rr.x, rr.y), fminf(rr.z, rr.w)));
    lmx = fmaxf(lmx, fmaxf(fmaxf(rr.x, rr.y), fmaxf(rr.z, rr.w)));
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
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 rr = vq_load4(yv[i]);
    int q0 = max(-128, min(127, (int)rintf(rr.x * scale_q) + zp));
    int q1 = max(-128, min(127, (int)rintf(rr.y * scale_q) + zp));
    int q2 = max(-128, min(127, (int)rintf(rr.z * scale_q) + zp));
    int q3 = max(-128, min(127, (int)rintf(rr.w * scale_q) + zp));
    q32[i] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) |
             ((q3 & 0xFF) << 24);
  }
}

// Decode-shaped fused twin (the rows<=32 v4 form with the register carry;
// merges the decode add+norm+quant three-launch chain into one launch).
// BIT-IDENTICAL to add_fp16_v8 + rmsnorm_quant_i8_h_v4 for the same reasons
// as the v4p form above.
__global__ void add_rmsnorm_quant_i8_h_v4(
  const unsigned short *xa, const unsigned short *xb,
  const unsigned short *gamma, unsigned short *r, unsigned short *y,
  signed char *q8, float *ascale, int *azp, int M, int K, float eps,
  int has_gamma) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  const uint2 *av = (const uint2 *)(xa + (long)m * K);
  const uint2 *bv = (const uint2 *)(xb + (long)m * K);
  const uint2 *gv = (const uint2 *)gamma;
  uint2 *rv = (uint2 *)(r + (long)m * K);
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
    const uint2 ra = av[i];
    const uint2 rb = bv[i];
    const unsigned short h0 =
      vq_f2h(vq_h2f((unsigned short)(ra.x & 0xFFFFu)) +
             vq_h2f((unsigned short)(rb.x & 0xFFFFu)));
    const unsigned short h1 = vq_f2h(vq_h2f((unsigned short)(ra.x >> 16)) +
                                     vq_h2f((unsigned short)(rb.x >> 16)));
    const unsigned short h2 =
      vq_f2h(vq_h2f((unsigned short)(ra.y & 0xFFFFu)) +
             vq_h2f((unsigned short)(rb.y & 0xFFFFu)));
    const unsigned short h3 = vq_f2h(vq_h2f((unsigned short)(ra.y >> 16)) +
                                     vq_h2f((unsigned short)(rb.y >> 16)));
    uint2 o;
    o.x = (unsigned int)h0 | ((unsigned int)h1 << 16);
    o.y = (unsigned int)h2 | ((unsigned int)h3 << 16);
    rv[i] = o;
    const float4 f =
      make_float4(vq_h2f(h0), vq_h2f(h1), vq_h2f(h2), vq_h2f(h3));
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
    float4 f = (slot >= 0) ? carry[slot] : vq_load4(rv[i]);
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
    float4 rr = make_float4(vq_h2f(h0), vq_h2f(h1), vq_h2f(h2), vq_h2f(h3));
    if (slot >= 0)
      carry[slot] = rr;
    lmn = fminf(lmn, fminf(fminf(rr.x, rr.y), fminf(rr.z, rr.w)));
    lmx = fmaxf(lmx, fmaxf(fmaxf(rr.x, rr.y), fmaxf(rr.z, rr.w)));
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
    float4 rr = (nc < VQ_NCARRY) ? carry[nc++] : vq_load4(yv[i]);
    int q0 = max(-128, min(127, (int)rintf(rr.x * scale_q) + zp));
    int q1 = max(-128, min(127, (int)rintf(rr.y * scale_q) + zp));
    int q2 = max(-128, min(127, (int)rintf(rr.z * scale_q) + zp));
    int q3 = max(-128, min(127, (int)rintf(rr.w * scale_q) + zp));
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
// generic -> shared-window u32 address for ldmatrix (no headers under NVRTC)
__device__ __forceinline__ unsigned ip_sh(const void *p) {
  unsigned r;
  asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
      : "=r"(r) : "l"(p));
  return r;
}
// fp16 <-> fp32 through the hardware cvt unit (no headers under NVRTC).
// h2f is exact; f2h is cvt.rn = RNE, bit-identical to dp4a_f2h on every
// finite value, so an epilogue may use either without changing output bytes.
__device__ __forceinline__ float hw_h2f(unsigned short h) {
  float f;
  asm("{ .reg .f16 t; mov.b16 t, %1; cvt.f32.f16 %0, t; }"
      : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short hw_f2h(float f) {
  unsigned short h;
  asm("{ .reg .f16 t; cvt.rn.f16.f32 t, %1; mov.b16 %0, t; }"
      : "=h"(h) : "f"(f));
  return h;
}
struct alignas(8) IPf2 {
  float x, y;
};
__global__ __launch_bounds__(256, 2)
void imma_gemm_pipe(const signed char *q8, const unsigned char *plain,
                    const float *ascale, const int *azp, const int *wrowsum,
                    const unsigned short *wscale, float *Y, int M, int N, int K,
                    int out_fp16, int raw) {
  // [d2] TRIPLE-buffered smem + depth-2 register prefetch: the single-step
  // prefetch left <1 compute step of latency cover, measured as the binding
  // term (ip_wide_bench 2026-08-13: depth-2 +4.3..+7% across shapes/runs;
  // tile WIDENING and occupancy both flat -- this tile is NOT DRAM-traffic
  // bound, x1.5 intensity moved nothing). Same store order, same integer
  // accumulation: bit-identical (validated bit-exact in the bench).
  __shared__ IPv16 Asv[3][IP_BM * IP_LD / 16];
  __shared__ IPv16 Wsv[3][IP_BN * IP_LD / 16];
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

  const int ip_nsteps = K >> 6; // K/IP_BK; K%64==0 is a caller invariant
  IP_LOAD(0);
  IP_STORE(0);
  if (ip_nsteps > 1)
    IP_LOAD(IP_BK); // k=1 tiles ride in registers until step 0's head store
  __syncthreads();

  // ldmatrix lane addressing (the fragment-load compression that lifts the
  // ~25% mma-issue ceiling of 12 lds.32 per 4 mma):
  //  - A x4: lanes 0-7 -> rows wm*16+0..7 @k0, 8-15 -> +8 @k0, 16-23 ->
  //    +0..7 @k16, 24-31 -> +8 @k16; result d0..d3 IS {a0,a1,a2,a3}.
  //  - B x4: lane group s = lane>>3, row wn*32 + s*8 + (lane&7); the result
  //    hands SUB-TILE s's fragment back in register s -- one x4 covers b0 of
  //    all four s, a second (+16B) covers b1.
  //  m8n8.b16 distribution (lane l: row l>>2, b16 cols 2*(l&3)) equals the
  //  s8 mma fragment mapping exactly; values and mma order are unchanged, so
  //  the result stays BIT-IDENTICAL (the binary validation contract).
  const int lrow_a = wm * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
  const int lkb_a = (lane & 16) ? 16 : 0;
  const int lrow_b = wn * 32 + (lane >> 3) * 8 + (lane & 7);
  unsigned pa_buf[3], pb_buf[3];
#pragma unroll
  for (int b3 = 0; b3 < 3; ++b3) {
    pa_buf[b3] = ip_sh((const signed char *)Asv[b3] + lrow_a * IP_LD + lkb_a);
    pb_buf[b3] = ip_sh((const signed char *)Wsv[b3] + lrow_b * IP_LD);
  }

  for (int ks = 0; ks < ip_nsteps; ++ks) {
    // [d2] Step head: park step ks+1's tiles (loaded LAST step, so their
    // global latency spanned a full compute step) into the third buffer,
    // then issue ks+2's loads -- in flight through THIS step's mma too.
    // One barrier per step, same as before: it separates this store from
    // its readers next step; buffer (ks+1)%3 was last read at step ks-2,
    // already separated by that step's barrier.
    if (ks + 1 < ip_nsteps) {
      IP_STORE((ks + 1) % 3);
      if (ks + 2 < ip_nsteps)
        IP_LOAD((ks + 2) * IP_BK);
      __syncthreads();
    }
    const unsigned pa = pa_buf[ks % 3], pb = pb_buf[ks % 3];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int a0, a1, a2, a3, c0, c1, c2, c3, d0, d1, d2, d3;
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(pa + h * 32));
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
        : "r"(pb + h * 32));
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(pb + h * 32 + 16));
#define IP_MMA(S, B0, B1)                                                      \
  asm volatile(                                                                \
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "               \
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"              \
    : "=r"(acc[S][0]), "=r"(acc[S][1]), "=r"(acc[S][2]), "=r"(acc[S][3])       \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(B0), "r"(B1),                    \
      "r"(acc[S][0]), "r"(acc[S][1]), "r"(acc[S][2]), "r"(acc[S][3]))
      IP_MMA(0, c0, d0);
      IP_MMA(1, c1, d1);
      IP_MMA(2, c2, d2);
      IP_MMA(3, c3, d3);
#undef IP_MMA
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
  if ((N & 1) == 0) {
    // epilogue v2 (g3tax ladder): row-side loads hoist to the thread's two
    // fixed rows, fp16<->fp32 use the hardware cvt (bit-identical to the
    // software pair on finite values), and the (r&1) column pair -- same
    // row, adjacent columns -- stores as one 4B/8B word. Even N makes the
    // pair edge-safe (c0 is even, so c0 < N implies c0+1 < N); the odd-N
    // caller, if one ever appears, takes the scalar path below unchanged.
    const int row0 = blockM + wm * 16 + g, row1 = row0 + 8;
    const int az0 = row0 < M ? azp[row0] : 0;
    const int az1 = row1 < M ? azp[row1] : 0;
    const float as0 = row0 < M ? ascale[row0] : 0.0f;
    const float as1 = row1 < M ? ascale[row1] : 0.0f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int cb0 = wn * 32 + s * 8 + 2 * t;
      const int c0 = blockN + cb0;
      if (c0 < N) {
        const int rs0 = raw ? rsh[cb0] : wrowsum[c0];
        const int rs1 = raw ? rsh[cb0 + 1] : wrowsum[c0 + 1];
        const float ws0 = hw_h2f(wscale[c0]), ws1 = hw_h2f(wscale[c0 + 1]);
        if (row0 < M) {
          const float v00 = (float)(acc[s][0] - az0 * rs0) * as0 * ws0;
          const float v01 = (float)(acc[s][1] - az0 * rs1) * as0 * ws1;
          if (out_fp16)
            *(unsigned int *)((unsigned short *)Y + (long)row0 * N + c0) =
              (unsigned int)hw_f2h(v00) | ((unsigned int)hw_f2h(v01) << 16);
          else {
            IPf2 p;
            p.x = v00;
            p.y = v01;
            *(IPf2 *)(Y + (long)row0 * N + c0) = p;
          }
        }
        if (row1 < M) {
          const float v10 = (float)(acc[s][2] - az1 * rs0) * as1 * ws0;
          const float v11 = (float)(acc[s][3] - az1 * rs1) * as1 * ws1;
          if (out_fp16)
            *(unsigned int *)((unsigned short *)Y + (long)row1 * N + c0) =
              (unsigned int)hw_f2h(v10) | ((unsigned int)hw_f2h(v11) << 16);
          else {
            IPf2 p;
            p.x = v10;
            p.y = v11;
            *(IPf2 *)(Y + (long)row1 * N + c0) = p;
          }
        }
      }
    }
    return;
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

  // Same ldmatrix lane addressing as imma_gemm_pipe (see the notes there).
  const int lrow_a = wm * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
  const int lkb_a = (lane & 16) ? 16 : 0;
  const int lrow_b = wn * 32 + (lane >> 3) * 8 + (lane & 7);
  const unsigned pa_buf[2] = {
    ip_sh((const signed char *)Asv[0] + lrow_a * IP_LD + lkb_a),
    ip_sh((const signed char *)Asv[1] + lrow_a * IP_LD + lkb_a)};
  const unsigned pb_buf[2] = {
    ip_sh((const signed char *)Wsv[0] + lrow_b * IP_LD),
    ip_sh((const signed char *)Wsv[1] + lrow_b * IP_LD)};

  int cur = 0;
  for (int k0 = 0; k0 < K; k0 += IP_BK) {
    const int knext = k0 + IP_BK;
    if (knext < K)
      IPG_LOAD(knext);

    const unsigned pa = pa_buf[cur], pb = pb_buf[cur];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int a0, a1, a2, a3, c0, c1, c2, c3, d0, d1, d2, d3;
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(pa + h * 32));
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
        : "r"(pb + h * 32));
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(pb + h * 32 + 16));
#define IPG_MMA(S, B0, B1)                                                     \
  asm volatile(                                                                \
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "               \
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"              \
    : "=r"(acc[S][0]), "=r"(acc[S][1]), "=r"(acc[S][2]), "=r"(acc[S][3])       \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(B0), "r"(B1),                    \
      "r"(acc[S][0]), "r"(acc[S][1]), "r"(acc[S][2]), "r"(acc[S][3]))
      IPG_MMA(0, c0, d0);
      IPG_MMA(1, c1, d1);
      IPG_MMA(2, c2, d2);
      IPG_MMA(3, c3, d3);
#undef IPG_MMA
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

// clock64 phase-bracket twin of imma_moe_grouped (NNTR_IMMA_CK=1, debug only).
// Identical math and identical output; thread 0 of each block samples the
// per-k-step phase boundaries (the single barrier locksteps all 8 warps, so
// one thread's timeline prices the block's rhythm) and atomicAdds cycle sums
// into ck[4]: {mma+issue, store, barrier, full-step count}.
__global__ __launch_bounds__(256, 2)
void imma_moe_grouped_ck(const signed char *q8, const int *tokid,
                         const unsigned long long *wp_tab,
                         const unsigned long long *ws_tab,
                         const int *block_expert, const float *ascale,
                         const int *azp, float *Y, int N, int K, int out_fp16,
                         unsigned long long *ck) {
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
  const int wm = warp >> 1, wn = warp & 1;
  const int g = lane >> 2, t = lane & 3;
  const int blockM = blockIdx.y * IP_BM, blockN = blockIdx.x * IP_BN;
  const int Kh = K >> 1;
  const unsigned int cx = 0u;

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

#define IPGK_LOAD(K0)                                                          \
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

#define IPGK_STORE(BUF)                                                        \
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

  IPGK_LOAD(0);
  IPGK_STORE(0);
  __syncthreads();

  const int lrow_a = wm * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
  const int lkb_a = (lane & 16) ? 16 : 0;
  const int lrow_b = wn * 32 + (lane >> 3) * 8 + (lane & 7);
  const unsigned pa_buf[2] = {
    ip_sh((const signed char *)Asv[0] + lrow_a * IP_LD + lkb_a),
    ip_sh((const signed char *)Asv[1] + lrow_a * IP_LD + lkb_a)};
  const unsigned pb_buf[2] = {
    ip_sh((const signed char *)Wsv[0] + lrow_b * IP_LD),
    ip_sh((const signed char *)Wsv[1] + lrow_b * IP_LD)};

#define IPGK_MMA(S, B0, B1)                                                    \
  asm volatile(                                                                \
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "               \
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"              \
    : "=r"(acc[S][0]), "=r"(acc[S][1]), "=r"(acc[S][2]), "=r"(acc[S][3])       \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(B0), "r"(B1),                    \
      "r"(acc[S][0]), "r"(acc[S][1]), "r"(acc[S][2]), "r"(acc[S][3]))
#define IPGK_HALF(H)                                                           \
  do {                                                                         \
    int a0, a1, a2, a3, c_0, c_1, c_2, c_3, d0, d1, d2, d3;                    \
    asm volatile(                                                              \
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"        \
      : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)                                 \
      : "r"(pa + (H) * 32));                                                   \
    asm volatile(                                                              \
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"        \
      : "=r"(c_0), "=r"(c_1), "=r"(c_2), "=r"(c_3)                             \
      : "r"(pb + (H) * 32));                                                   \
    asm volatile(                                                              \
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"        \
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)                                 \
      : "r"(pb + (H) * 32 + 16));                                              \
    IPGK_MMA(0, c_0, d0);                                                      \
    IPGK_MMA(1, c_1, d1);                                                      \
    IPGK_MMA(2, c_2, d2);                                                      \
    IPGK_MMA(3, c_3, d3);                                                      \
  } while (0)

  unsigned long long t_ld = 0ull, t_h0 = 0ull, t_h1 = 0ull, t_st = 0ull,
                     t_bar = 0ull, ns = 0ull;
  long long c0 = 0, cl = 0, ch = 0, c1 = 0, c2 = 0, c3 = 0;
  int cur = 0;
  for (int k0 = 0; k0 < K; k0 += IP_BK) {
    const int knext = k0 + IP_BK;
    if (tid == 0)
      c0 = clock64();
    if (knext < K)
      IPGK_LOAD(knext);
    if (tid == 0)
      cl = clock64();

    const unsigned pa = pa_buf[cur], pb = pb_buf[cur];
    IPGK_HALF(0);
    if (tid == 0)
      ch = clock64();
    IPGK_HALF(1);

    if (knext < K) {
      if (tid == 0)
        c1 = clock64();
      IPGK_STORE(cur ^ 1);
      if (tid == 0)
        c2 = clock64();
      __syncthreads();
      if (tid == 0) {
        c3 = clock64();
        t_ld += (unsigned long long)(cl - c0);
        t_h0 += (unsigned long long)(ch - cl);
        t_h1 += (unsigned long long)(c1 - ch);
        t_st += (unsigned long long)(c2 - c1);
        t_bar += (unsigned long long)(c3 - c2);
        ++ns;
      }
      cur ^= 1;
    }
  }
#undef IPGK_HALF
#undef IPGK_MMA
#undef IPGK_LOAD
#undef IPGK_STORE

  {
    int v = rsacc;
    v += __shfl_down_sync(0xffffffffu, v, 1);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    if (ssub == 0)
      rsh[srow] = v;
    __syncthreads();
  }
  if (tid == 0 && ck) {
    atomicAdd(&ck[0], t_ld);
    atomicAdd(&ck[1], t_h0);
    atomicAdd(&ck[2], t_h1);
    atomicAdd(&ck[3], t_st);
    atomicAdd(&ck[4], t_bar);
    atomicAdd(&ck[5], ns);
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

// gate+up FUSED grouped GEMM (NNTR_MOE_G2=1): one A staging serves TWO W
// tiles, so per k-step the block runs 16 mma against ONE A load/store/barrier
// instead of 8 -- a direct attack on the measured per-k-step overhead
// (~1,550 of ~1,700 cycles are not tensor-core time). Same math as two
// imma_moe_grouped launches; outputs Yg/Yu are written identically.
__global__ __launch_bounds__(256, 2)
void imma_moe_grouped_g2(const signed char *q8, const int *tokid,
                         const unsigned long long *wpg_tab,
                         const unsigned long long *wsg_tab,
                         const unsigned long long *wpu_tab,
                         const unsigned long long *wsu_tab,
                         const int *block_expert, const float *ascale,
                         const int *azp, float *Yg, float *Yu, int N, int K,
                         int out_fp16) {
  const int e = block_expert[blockIdx.y];
  if (e < 0)
    return;
  const unsigned char *plg =
    (const unsigned char *)(unsigned long long)wpg_tab[e];
  const unsigned short *wscg =
    (const unsigned short *)(unsigned long long)wsg_tab[e];
  const unsigned char *plu =
    (const unsigned char *)(unsigned long long)wpu_tab[e];
  const unsigned short *wscu =
    (const unsigned short *)(unsigned long long)wsu_tab[e];
  __shared__ IPv16 Asv[2][IP_BM * IP_LD / 16];
  __shared__ IPv16 Wgv[2][IP_BN * IP_LD / 16];
  __shared__ IPv16 Wuv[2][IP_BN * IP_LD / 16];
  __shared__ int rshg[IP_BN];
  __shared__ int rshu[IP_BN];
  __shared__ int toks[IP_BM];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1;
  const int g = lane >> 2, t = lane & 3;
  const int blockM = blockIdx.y * IP_BM, blockN = blockIdx.x * IP_BN;
  const int Kh = K >> 1;
  const unsigned int cx = 0u;

  if (tid < IP_BM)
    toks[tid] = tokid ? tokid[blockM + tid] : (blockM + tid);
  __syncthreads();

  const int srow = tid >> 2, ssub = tid & 3;
  const int atok = toks[srow];
  const int wrow = blockN + srow;
  const bool arow_ok = atok >= 0, wrow_ok = wrow < N;
  const signed char *aptr = q8 + (long)atok * K + (ssub << 4);
  const unsigned char *wgptr = plg + (long)wrow * Kh + (ssub << 3);
  const unsigned char *wuptr = plu + (long)wrow * Kh + (ssub << 3);
  const int sdst = srow * IP_LD + (ssub << 4);

  int accg[4][4], accu[4][4];
#pragma unroll
  for (int s = 0; s < 4; ++s)
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      accg[s][r] = 0;
      accu[s][r] = 0;
    }
  int rsg = 0, rsu = 0;

  IPv16 ra;
  IPv8 rwg, rwu;
  IPv16 wv;

#define IPG2_LOAD(K0)                                                          \
  do {                                                                         \
    if (arow_ok) {                                                             \
      ra = *(const IPv16 *)(aptr + (K0));                                      \
    } else {                                                                   \
      ra.a = ra.b = ra.c = ra.d = 0u;                                          \
    }                                                                          \
    if (wrow_ok) {                                                             \
      rwg = *(const IPv8 *)(wgptr + ((K0) >> 1));                              \
      rwu = *(const IPv8 *)(wuptr + ((K0) >> 1));                              \
    } else {                                                                   \
      rwg.a = rwg.b = 0u;                                                      \
      rwu.a = rwu.b = 0u;                                                      \
    }                                                                          \
  } while (0)

#define IPG2_UNPACK(RW, RS, BUFV)                                              \
  do {                                                                         \
    unsigned int lo = (RW).a & 0x0F0F0F0Fu;                                    \
    unsigned int hi = ((RW).a >> 4) & 0x0F0F0F0Fu;                             \
    wv.a = ip_nib2i8(__byte_perm(lo, hi, 0x5140), cx);                         \
    wv.b = ip_nib2i8(__byte_perm(lo, hi, 0x7362), cx);                         \
    lo = (RW).b & 0x0F0F0F0Fu;                                                 \
    hi = ((RW).b >> 4) & 0x0F0F0F0Fu;                                          \
    wv.c = ip_nib2i8(__byte_perm(lo, hi, 0x5140), cx);                         \
    wv.d = ip_nib2i8(__byte_perm(lo, hi, 0x7362), cx);                         \
    RS = __dp4a(0x01010101, (int)wv.a, RS);                                    \
    RS = __dp4a(0x01010101, (int)wv.b, RS);                                    \
    RS = __dp4a(0x01010101, (int)wv.c, RS);                                    \
    RS = __dp4a(0x01010101, (int)wv.d, RS);                                    \
    *(IPv16 *)((signed char *)(BUFV) + sdst) = wv;                             \
  } while (0)

#define IPG2_STORE(BUF)                                                        \
  do {                                                                         \
    IPG2_UNPACK(rwg, rsg, Wgv[BUF]);                                           \
    IPG2_UNPACK(rwu, rsu, Wuv[BUF]);                                           \
    *(IPv16 *)((signed char *)Asv[BUF] + sdst) = ra;                           \
  } while (0)

  IPG2_LOAD(0);
  IPG2_STORE(0);
  __syncthreads();

  const int lrow_a = wm * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
  const int lkb_a = (lane & 16) ? 16 : 0;
  const int lrow_b = wn * 32 + (lane >> 3) * 8 + (lane & 7);
  const unsigned pa_buf[2] = {
    ip_sh((const signed char *)Asv[0] + lrow_a * IP_LD + lkb_a),
    ip_sh((const signed char *)Asv[1] + lrow_a * IP_LD + lkb_a)};
  const unsigned pbg_buf[2] = {
    ip_sh((const signed char *)Wgv[0] + lrow_b * IP_LD),
    ip_sh((const signed char *)Wgv[1] + lrow_b * IP_LD)};
  const unsigned pbu_buf[2] = {
    ip_sh((const signed char *)Wuv[0] + lrow_b * IP_LD),
    ip_sh((const signed char *)Wuv[1] + lrow_b * IP_LD)};

#define IPG2_MMA(ACC, S, B0, B1)                                               \
  asm volatile(                                                                \
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "               \
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"              \
    : "=r"(ACC[S][0]), "=r"(ACC[S][1]), "=r"(ACC[S][2]), "=r"(ACC[S][3])       \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(B0), "r"(B1),                    \
      "r"(ACC[S][0]), "r"(ACC[S][1]), "r"(ACC[S][2]), "r"(ACC[S][3]))
#define IPG2_HALF(H)                                                           \
  do {                                                                         \
    int a0, a1, a2, a3, c0, c1, c2, c3, d0, d1, d2, d3;                        \
    asm volatile(                                                              \
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"        \
      : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)                                 \
      : "r"(pa + (H) * 32));                                                   \
    asm volatile(                                                              \
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"        \
      : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)                                 \
      : "r"(pbg + (H) * 32));                                                  \
    asm volatile(                                                              \
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"        \
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)                                 \
      : "r"(pbg + (H) * 32 + 16));                                             \
    IPG2_MMA(accg, 0, c0, d0);                                                 \
    IPG2_MMA(accg, 1, c1, d1);                                                 \
    IPG2_MMA(accg, 2, c2, d2);                                                 \
    IPG2_MMA(accg, 3, c3, d3);                                                 \
    asm volatile(                                                              \
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"        \
      : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)                                 \
      : "r"(pbu + (H) * 32));                                                  \
    asm volatile(                                                              \
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"        \
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)                                 \
      : "r"(pbu + (H) * 32 + 16));                                             \
    IPG2_MMA(accu, 0, c0, d0);                                                 \
    IPG2_MMA(accu, 1, c1, d1);                                                 \
    IPG2_MMA(accu, 2, c2, d2);                                                 \
    IPG2_MMA(accu, 3, c3, d3);                                                 \
  } while (0)

  int cur = 0;
  for (int k0 = 0; k0 < K; k0 += IP_BK) {
    const int knext = k0 + IP_BK;
    if (knext < K)
      IPG2_LOAD(knext);

    const unsigned pa = pa_buf[cur], pbg = pbg_buf[cur], pbu = pbu_buf[cur];
    IPG2_HALF(0);
    IPG2_HALF(1);

    if (knext < K) {
      IPG2_STORE(cur ^ 1);
      __syncthreads();
      cur ^= 1;
    }
  }
#undef IPG2_HALF
#undef IPG2_MMA
#undef IPG2_STORE
#undef IPG2_UNPACK
#undef IPG2_LOAD

  {
    int vg = rsg, vu = rsu;
    vg += __shfl_down_sync(0xffffffffu, vg, 1);
    vg += __shfl_down_sync(0xffffffffu, vg, 2);
    vu += __shfl_down_sync(0xffffffffu, vu, 1);
    vu += __shfl_down_sync(0xffffffffu, vu, 2);
    if (ssub == 0) {
      rshg[srow] = vg;
      rshu[srow] = vu;
    }
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
        float vg = 0.0f, vu = 0.0f;
        if (tok >= 0) {
          vg = (float)(accg[s][r] - azp[tok] * rshg[cb]) * ascale[tok] *
               dp4a_h2f(wscg[col]);
          vu = (float)(accu[s][r] - azp[tok] * rshu[cb]) * ascale[tok] *
               dp4a_h2f(wscu[col]);
        }
        if (out_fp16) {
          ((unsigned short *)Yg)[(long)(blockM + lrow) * N + col] =
            dp4a_f2h(vg);
          ((unsigned short *)Yu)[(long)(blockM + lrow) * N + col] =
            dp4a_f2h(vu);
        } else {
          Yg[(long)(blockM + lrow) * N + col] = vg;
          Yu[(long)(blockM + lrow) * N + col] = vu;
        }
      }
    }
  }
}

// WIDE-N grouped GEMM (NNTR_MOE_WT=1): 64x128 block tile, 8 warps in a 2x4
// grid of 32x32 warp tiles. Per half-k-step a warp loads 2 A + 2 B fragments
// and runs 8 mma -- B-ldmatrix per mma HALVES vs the 64x64 tile (2 ld / 8 mma
// vs 2 / 4). That is the surviving suspect from the gate+up-fusion negative
// result: halving A-staging+barriers per mma moved nothing, so the binder is
// the B-fragment ldmatrix/LSU path, which this variant is built to relieve.
// Same math and output as imma_moe_grouped; N must be a multiple of 128.
__global__ __launch_bounds__(256, 2)
void imma_moe_grouped_w(const signed char *q8, const int *tokid,
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
  __shared__ IPv16 Wsv[2][2 * IP_BN * IP_LD / 16]; // 128 rows
  __shared__ int rsh[2 * IP_BN];
  __shared__ int toks[IP_BM];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3; // 2 x 4 grid of 32x32 tiles
  const int g = lane >> 2, t = lane & 3;
  const int blockM = blockIdx.y * IP_BM, blockN = blockIdx.x * (2 * IP_BN);
  const int Kh = K >> 1;
  const unsigned int cx = 0u;

  if (tid < IP_BM)
    toks[tid] = tokid ? tokid[blockM + tid] : (blockM + tid);
  __syncthreads();

  const int srow = tid >> 2, ssub = tid & 3;
  const int atok = toks[srow];
  const int wrow0 = blockN + srow, wrow1 = blockN + IP_BN + srow;
  const bool arow_ok = atok >= 0;
  const bool w0_ok = wrow0 < N, w1_ok = wrow1 < N;
  const signed char *aptr = q8 + (long)atok * K + (ssub << 4);
  const unsigned char *w0ptr = plain + (long)wrow0 * Kh + (ssub << 3);
  const unsigned char *w1ptr = plain + (long)wrow1 * Kh + (ssub << 3);
  const int sdst = srow * IP_LD + (ssub << 4);
  const int sdst1 = (IP_BN + srow) * IP_LD + (ssub << 4);

  int acc[2][4][4];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int s = 0; s < 4; ++s)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        acc[m][s][r] = 0;
  int rs0 = 0, rs1 = 0;

  IPv16 ra;
  IPv8 rw0, rw1;
  IPv16 wv;

#define IPW_LOAD(K0)                                                           \
  do {                                                                         \
    if (arow_ok) {                                                             \
      ra = *(const IPv16 *)(aptr + (K0));                                      \
    } else {                                                                   \
      ra.a = ra.b = ra.c = ra.d = 0u;                                          \
    }                                                                          \
    if (w0_ok) {                                                               \
      rw0 = *(const IPv8 *)(w0ptr + ((K0) >> 1));                              \
    } else {                                                                   \
      rw0.a = rw0.b = 0u;                                                      \
    }                                                                          \
    if (w1_ok) {                                                               \
      rw1 = *(const IPv8 *)(w1ptr + ((K0) >> 1));                              \
    } else {                                                                   \
      rw1.a = rw1.b = 0u;                                                      \
    }                                                                          \
  } while (0)

#define IPW_UNPACK(RW, RS, DST)                                                \
  do {                                                                         \
    unsigned int lo = (RW).a & 0x0F0F0F0Fu;                                    \
    unsigned int hi = ((RW).a >> 4) & 0x0F0F0F0Fu;                             \
    wv.a = ip_nib2i8(__byte_perm(lo, hi, 0x5140), cx);                         \
    wv.b = ip_nib2i8(__byte_perm(lo, hi, 0x7362), cx);                         \
    lo = (RW).b & 0x0F0F0F0Fu;                                                 \
    hi = ((RW).b >> 4) & 0x0F0F0F0Fu;                                          \
    wv.c = ip_nib2i8(__byte_perm(lo, hi, 0x5140), cx);                         \
    wv.d = ip_nib2i8(__byte_perm(lo, hi, 0x7362), cx);                         \
    RS = __dp4a(0x01010101, (int)wv.a, RS);                                    \
    RS = __dp4a(0x01010101, (int)wv.b, RS);                                    \
    RS = __dp4a(0x01010101, (int)wv.c, RS);                                    \
    RS = __dp4a(0x01010101, (int)wv.d, RS);                                    \
    *(IPv16 *)((signed char *)(DST)) = wv;                                     \
  } while (0)

#define IPW_STORE(BUF)                                                         \
  do {                                                                         \
    IPW_UNPACK(rw0, rs0, (signed char *)Wsv[BUF] + sdst);                      \
    IPW_UNPACK(rw1, rs1, (signed char *)Wsv[BUF] + sdst1);                     \
    *(IPv16 *)((signed char *)Asv[BUF] + sdst) = ra;                           \
  } while (0)

  IPW_LOAD(0);
  IPW_STORE(0);
  __syncthreads();

  // A: two m16 tiles per warp (rows wm*32 .. +31); B: one 32-col strip.
  const int lrow_a0 = wm * 32 + (lane & 7) + ((lane & 8) ? 8 : 0);
  const int lkb_a = (lane & 16) ? 16 : 0;
  const int lrow_b = wn * 32 + (lane >> 3) * 8 + (lane & 7);
  const unsigned pa0_buf[2] = {
    ip_sh((const signed char *)Asv[0] + lrow_a0 * IP_LD + lkb_a),
    ip_sh((const signed char *)Asv[1] + lrow_a0 * IP_LD + lkb_a)};
  const unsigned pa1_buf[2] = {
    ip_sh((const signed char *)Asv[0] + (lrow_a0 + 16) * IP_LD + lkb_a),
    ip_sh((const signed char *)Asv[1] + (lrow_a0 + 16) * IP_LD + lkb_a)};
  const unsigned pb_buf[2] = {
    ip_sh((const signed char *)Wsv[0] + lrow_b * IP_LD),
    ip_sh((const signed char *)Wsv[1] + lrow_b * IP_LD)};

#define IPW_MMA(M, S, B0, B1)                                                  \
  asm volatile(                                                                \
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "               \
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"              \
    : "=r"(acc[M][S][0]), "=r"(acc[M][S][1]), "=r"(acc[M][S][2]),              \
      "=r"(acc[M][S][3])                                                       \
    : "r"(a##M##0), "r"(a##M##1), "r"(a##M##2), "r"(a##M##3), "r"(B0),         \
      "r"(B1), "r"(acc[M][S][0]), "r"(acc[M][S][1]), "r"(acc[M][S][2]),        \
      "r"(acc[M][S][3]))
#define IPW_HALF(H)                                                            \
  do {                                                                         \
    int a00, a01, a02, a03, a10, a11, a12, a13;                                \
    int c0, c1, c2, c3, d0, d1, d2, d3;                                        \
    asm volatile(                                                              \
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"        \
      : "=r"(a00), "=r"(a01), "=r"(a02), "=r"(a03)                             \
      : "r"(pa0 + (H) * 32));                                                  \
    asm volatile(                                                              \
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"        \
      : "=r"(a10), "=r"(a11), "=r"(a12), "=r"(a13)                             \
      : "r"(pa1 + (H) * 32));                                                  \
    asm volatile(                                                              \
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"        \
      : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)                                 \
      : "r"(pb + (H) * 32));                                                   \
    asm volatile(                                                              \
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"        \
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)                                 \
      : "r"(pb + (H) * 32 + 16));                                              \
    IPW_MMA(0, 0, c0, d0);                                                     \
    IPW_MMA(0, 1, c1, d1);                                                     \
    IPW_MMA(0, 2, c2, d2);                                                     \
    IPW_MMA(0, 3, c3, d3);                                                     \
    IPW_MMA(1, 0, c0, d0);                                                     \
    IPW_MMA(1, 1, c1, d1);                                                     \
    IPW_MMA(1, 2, c2, d2);                                                     \
    IPW_MMA(1, 3, c3, d3);                                                     \
  } while (0)

  int cur = 0;
  for (int k0 = 0; k0 < K; k0 += IP_BK) {
    const int knext = k0 + IP_BK;
    if (knext < K)
      IPW_LOAD(knext);

    const unsigned pa0 = pa0_buf[cur], pa1 = pa1_buf[cur], pb = pb_buf[cur];
    IPW_HALF(0);
    IPW_HALF(1);

    if (knext < K) {
      IPW_STORE(cur ^ 1);
      __syncthreads();
      cur ^= 1;
    }
  }
#undef IPW_HALF
#undef IPW_MMA
#undef IPW_STORE
#undef IPW_UNPACK
#undef IPW_LOAD

  {
    int v0 = rs0, v1 = rs1;
    v0 += __shfl_down_sync(0xffffffffu, v0, 1);
    v0 += __shfl_down_sync(0xffffffffu, v0, 2);
    v1 += __shfl_down_sync(0xffffffffu, v1, 1);
    v1 += __shfl_down_sync(0xffffffffu, v1, 2);
    if (ssub == 0) {
      rsh[srow] = v0;
      rsh[IP_BN + srow] = v1;
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int lrow = wm * 32 + m * 16 + g + ((r >> 1) ? 8 : 0);
        const int cb = wn * 32 + s * 8 + 2 * t + (r & 1);
        const int col = blockN + cb;
        if (col < N) {
          const int tok = toks[lrow];
          float v = 0.0f;
          if (tok >= 0)
            v = (float)(acc[m][s][r] - azp[tok] * rsh[cb]) * ascale[tok] *
                dp4a_h2f(wscale[col]);
          if (out_fp16)
            ((unsigned short *)Y)[(long)(blockM + lrow) * N + col] =
              dp4a_f2h(v);
          else
            Y[(long)(blockM + lrow) * N + col] = v;
        }
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

// ---- NNTR_MOE_G3: grouped tile, cp.async ring + PACKED fragment-order W ----
// The tile_bench E1/E2 transplant (25.4-25.7 TOPS standalone vs 23.7 for the
// pre-STS-unpack pipeline above; see tile_bench/MARLIN_DISSECTION_20260812.md):
//  - the weight payload is REPACKED ONCE at load into per-thread fragment
//    order (moe_repack_frag_g3), so staging is a verbatim cp.async copy: no
//    register round-trip, no unpack ALU in the store phase, and W stays
//    PACKED in shared memory (half the bytes);
//  - dequant is two ip_nib2i8 in registers between ldmatrix and mma;
//  - ONE __syncthreads per 128-K stage (fetch -> wait_group(ST-2) -> sync),
//    and the fragments for step k+1 load BEFORE the barrier, so the first
//    post-barrier instruction is an mma on resident registers;
//  - the A-side zero-point rowsum comes from the PRECOMPUTED per-expert
//    table (packed staging cannot re-derive it in-loop). The table holds the
//    same integer sum the in-loop __dp4a produced, and int32 accumulation is
//    exact, so the epilogue floats -- and the output bytes -- are unchanged
//    from imma_moe_grouped.
// A rows with tok < 0 stage as ZEROS via the cp.async src-size-0 form (16B
// window, 0 source bytes = zero-fill, no dereference), preserving the
// IPG_LOAD zero semantics for padding rows.
#define IPX_ST 3
#define IPX_ALD 144
#define IPX_WLD 80
__global__ __launch_bounds__(256, 2)
void imma_moe_grouped_g3(const signed char *q8, const int *tokid,
                         const unsigned long long *wp_tab,
                         const unsigned long long *ws_tab,
                         const unsigned long long *wr_tab,
                         const int *block_expert, const float *ascale,
                         const int *azp, float *Y, int N, int K,
                         int out_fp16, const int *wl_n) {
  const int e = block_expert[blockIdx.y];
  if (e < 0)
    return;
  const unsigned char *plain =
    (const unsigned char *)(unsigned long long)wp_tab[e];
  const unsigned short *wscale =
    (const unsigned short *)(unsigned long long)ws_tab[e];
  const int *wrsum = (const int *)(unsigned long long)wr_tab[e];
  __shared__ signed char As[IPX_ST][64 * IPX_ALD];
  __shared__ unsigned char Ws[IPX_ST][64 * IPX_WLD];
  __shared__ int toks[64];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1; // 4 x 2 warp grid over 64x64
  const int g = lane >> 2, t = lane & 3;
  const int blockM = blockIdx.y * 64, blockN = blockIdx.x * 64;
  const int Kh = K >> 1;
  if (tid < 64)
    toks[tid] = tokid ? tokid[blockM + tid] : (blockM + tid);
  __syncthreads();
  // dead-subtile skip: a warp whose 16-row m-subtile is entirely padding
  // (row >= the window's live count) skips its fragment loads, mma and
  // epilogue. Barriers and staging stay warp-uniform. g3tax R8: -8.1%.
  const int live = wl_n ? wl_n[blockIdx.y] : 64;
  const bool mdead = (wm * 16 >= live);

  // staging: 4 threads per row; A carries 2x16B per 128-K stage, W 1x16B of
  // the row's 64 packed bytes.
  const int srow = tid >> 2, ssub = tid & 3;
  const int atok = toks[srow];
  const signed char *aptr = q8 + (long)atok * K + (ssub << 4);
  const int a_sz = atok >= 0 ? 16 : 0; // src-size 0 = zero-fill
  const unsigned char *wptr = plain + (long)(blockN + srow) * Kh + (ssub << 4);
  const int a_dst = srow * IPX_ALD + (ssub << 4);
  const int w_dst = srow * IPX_WLD + (ssub << 4);

  const unsigned A_ST = 64 * IPX_ALD, W_ST = 64 * IPX_WLD;
  unsigned wa = ip_sh((const signed char *)As[0] + a_dst);
  unsigned ww = ip_sh((const unsigned char *)Ws[0] + w_dst);
  const unsigned wa_end = wa + IPX_ST * A_ST, ww_end = ww + IPX_ST * W_ST;

#define IPX_IN(WA, WW, K0)                                                     \
  do {                                                                         \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(WA),  \
                 "l"(aptr + (K0)), "r"(a_sz));                                 \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(      \
                   (WA) + 64),                                                 \
                 "l"(aptr + (K0) + 64), "r"(a_sz));                            \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(WW),      \
                 "l"(wptr + ((K0) >> 1)));                                     \
    asm volatile("cp.async.commit_group;\n");                                  \
  } while (0)

  int kt = 0;
  const int ksteps = K >> 7;
#pragma unroll
  for (int s = 0; s < IPX_ST - 1; ++s) {
    IPX_IN(wa, ww, kt << 7);
    ++kt;
    wa += A_ST;
    ww += W_ST;
  }

  int acc[4][4];
#pragma unroll
  for (int s = 0; s < 4; ++s)
#pragma unroll
    for (int r = 0; r < 4; ++r)
      acc[s][r] = 0;

  const int lrow_a = wm * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
  const int lkb_a = (lane & 16) ? 16 : 0;
  const int lrow_b = wn * 32 + (lane >> 3) * 8 + (lane & 7);
  unsigned pa = ip_sh((const signed char *)As[0] + lrow_a * IPX_ALD + lkb_a);
  unsigned pw = ip_sh((const unsigned char *)Ws[0] + lrow_b * IPX_WLD);
  const unsigned pa_end = pa + IPX_ST * A_ST, pw_end = pw + IPX_ST * W_ST;

  asm volatile("cp.async.wait_group %0;\n" ::"n"(IPX_ST - 2));
  __syncthreads();

#define IPX_FRAG(P, R0, R1, R2, R3)                                            \
  asm volatile(                                                                \
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"          \
    : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(P))

  int na0, na1, na2, na3, np0, np1, np2, np3;
  na0 = na1 = na2 = na3 = np0 = np1 = np2 = np3 = 0;
  if (!mdead) {
    IPX_FRAG(pa, na0, na1, na2, na3);
    IPX_FRAG(pw, np0, np1, np2, np3);
  }

  for (int c = 0; c < ksteps; ++c) {
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      const int a0 = na0, a1 = na1, a2 = na2, a3 = na3;
      const int p0 = np0, p1 = np1, p2 = np2, p3 = np3;
      if (h == 2) {
        // h3 fragments BEFORE the barrier (they read the current stage)
        if (!mdead) {
          IPX_FRAG(pa + 3 * 32, na0, na1, na2, na3);
          IPX_FRAG(pw + 3 * 16, np0, np1, np2, np3);
        }
        if (kt < ksteps) {
          IPX_IN(wa, ww, kt << 7);
          ++kt;
          wa += A_ST;
          wa = (wa >= wa_end) ? wa - IPX_ST * A_ST : wa;
          ww += W_ST;
          ww = (ww >= ww_end) ? ww - IPX_ST * W_ST : ww;
        } else {
          asm volatile("cp.async.commit_group;\n");
        }
        asm volatile("cp.async.wait_group %0;\n" ::"n"(IPX_ST - 2));
        __syncthreads();
      } else if (h == 3) {
        // advance to the just-published stage, prefetch its h0
        pa += A_ST;
        pa = (pa >= pa_end) ? pa - IPX_ST * A_ST : pa;
        pw += W_ST;
        pw = (pw >= pw_end) ? pw - IPX_ST * W_ST : pw;
        if (c + 1 < ksteps && !mdead) {
          IPX_FRAG(pa, na0, na1, na2, na3);
          IPX_FRAG(pw, np0, np1, np2, np3);
        }
      } else if (!mdead) {
        IPX_FRAG(pa + (h + 1) * 32, na0, na1, na2, na3);
        IPX_FRAG(pw + (h + 1) * 16, np0, np1, np2, np3);
      }
#define IPX_MMA(S, PK)                                                         \
  do {                                                                         \
    const unsigned b0 = ip_nib2i8((PK) & 0x0F0F0F0Fu, 0u);                     \
    const unsigned b1 = ip_nib2i8(((unsigned)(PK) >> 4) & 0x0F0F0F0Fu, 0u);    \
    asm volatile(                                                              \
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "             \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"            \
      : "=r"(acc[S][0]), "=r"(acc[S][1]), "=r"(acc[S][2]), "=r"(acc[S][3])     \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),                  \
        "r"(acc[S][0]), "r"(acc[S][1]), "r"(acc[S][2]), "r"(acc[S][3]));       \
  } while (0)
      if (!mdead) {
        IPX_MMA(0, p0);
        IPX_MMA(1, p1);
        IPX_MMA(2, p2);
        IPX_MMA(3, p3);
      }
#undef IPX_MMA
    }
  }
#undef IPX_FRAG
#undef IPX_IN

  // epilogue v2: same scalar math as imma_moe_grouped (rowsum from the
  // table), but the row-side token loads hoist to the thread's two fixed
  // rows, fp16<->fp32 use the hardware cvt (bit-identical, see hw_f2h), and
  // adjacent columns store as one 4B/8B word (the wrapper guarantees
  // N % 64 == 0 so a pair never crosses the edge). g3tax ladder: the old
  // per-element software-cvt epilogue was 0.8 ms of the 4.2 ms launch.
  if (mdead)
    return; // all barriers are behind us; padding rows are never read
  const int lr0 = wm * 16 + g, lr1 = lr0 + 8;
  const int tk0 = toks[lr0], tk1 = toks[lr1];
  const int az0 = tk0 >= 0 ? azp[tk0] : 0;
  const int az1 = tk1 >= 0 ? azp[tk1] : 0;
  const float as0 = tk0 >= 0 ? ascale[tk0] : 0.0f;
  const float as1 = tk1 >= 0 ? ascale[tk1] : 0.0f;
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int c0 = blockN + wn * 32 + s * 8 + 2 * t;
    const int rs0 = wrsum[c0], rs1 = wrsum[c0 + 1];
    const float ws0 = hw_h2f(wscale[c0]), ws1 = hw_h2f(wscale[c0 + 1]);
    const float v00 =
      tk0 >= 0 ? (float)(acc[s][0] - az0 * rs0) * as0 * ws0 : 0.0f;
    const float v01 =
      tk0 >= 0 ? (float)(acc[s][1] - az0 * rs1) * as0 * ws1 : 0.0f;
    const float v10 =
      tk1 >= 0 ? (float)(acc[s][2] - az1 * rs0) * as1 * ws0 : 0.0f;
    const float v11 =
      tk1 >= 0 ? (float)(acc[s][3] - az1 * rs1) * as1 * ws1 : 0.0f;
    if (out_fp16) {
      *(unsigned int *)((unsigned short *)Y + (long)(blockM + lr0) * N + c0) =
        (unsigned int)hw_f2h(v00) | ((unsigned int)hw_f2h(v01) << 16);
      *(unsigned int *)((unsigned short *)Y + (long)(blockM + lr1) * N + c0) =
        (unsigned int)hw_f2h(v10) | ((unsigned int)hw_f2h(v11) << 16);
    } else {
      IPf2 lo, hi;
      lo.x = v00; lo.y = v01; hi.x = v10; hi.y = v11;
      *(IPf2 *)(Y + (long)(blockM + lr0) * N + c0) = lo;
      *(IPf2 *)(Y + (long)(blockM + lr1) * N + c0) = hi;
    }
  }
}

// down-projection variant (K <= 512, K % 128 == 0): persistent-N. One CTA
// owns an m-block and LOOPS over all N/64 tiles: A (64 x K int8 <= 32 KB)
// loads ONCE with no ring, W streams through the 3-stage ring CONTINUOUSLY
// across tiles, one prologue per CTA instead of one per (m,n) tile, and the
// grid drops from (N/64, W) to (1, W). Down is DIRECT mode only (the SwiGLU
// output already lives in gathered space), so there is no tokid path; the
// epilogue math is identical to imma_moe_grouped_g3's tok>=0 branch.
// Bench: DOWN shape 15.9 -> 20.4 TOPS single / 19.6 multi (tile_g6).
#define IPD_ALD 528
__global__ __launch_bounds__(256, 2)
void imma_moe_grouped_g3d(const signed char *q8,
                          const unsigned long long *wp_tab,
                          const unsigned long long *ws_tab,
                          const unsigned long long *wr_tab,
                          const int *block_expert, const float *ascale,
                          const int *azp, float *Y, int N, int K,
                          int out_fp16, const int *wl_n) {
  const int e = block_expert[blockIdx.y];
  if (e < 0)
    return;
  const unsigned char *plain =
    (const unsigned char *)(unsigned long long)wp_tab[e];
  const unsigned short *wscale =
    (const unsigned short *)(unsigned long long)ws_tab[e];
  const int *wrsum = (const int *)(unsigned long long)wr_tab[e];
  __shared__ signed char As[64 * IPD_ALD];
  __shared__ unsigned char Ws[IPX_ST][64 * IPX_WLD];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1;
  const int g = lane >> 2, t = lane & 3;
  const int blockM = blockIdx.y * 64;
  const int Kh = K >> 1;
  const int NT = N >> 6;
  const int live = wl_n ? wl_n[blockIdx.y] : 64; // dead-subtile skip (R8)
  const bool mdead = (wm * 16 >= live);

  const int srow = tid >> 2, ssub = tid & 3;
  const long a_src = (long)(blockM + srow) * K + (ssub << 4);
  const int a_dst = srow * IPD_ALD + (ssub << 4);
  const long w_src = (long)srow * Kh + (ssub << 4);
  const int w_dst = srow * IPX_WLD + (ssub << 4);

  const unsigned W_ST = 64 * IPX_WLD;
  unsigned ww = ip_sh((const unsigned char *)Ws[0] + w_dst);
  const unsigned ww_end = ww + IPX_ST * W_ST;

#define IPD_WIN(WW, KT)                                                        \
  do {                                                                         \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(WW),      \
                 "l"(plain + ((long)((KT) >> 2) << 6) * Kh + w_src +           \
                     (((KT) & 3) << 6)));                                      \
    asm volatile("cp.async.commit_group;\n");                                  \
  } while (0)

  {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if ((ssub << 4) + i * 64 < K)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                       ip_sh((const signed char *)As + a_dst + i * 64)),
                     "l"(q8 + a_src + i * 64));
    }
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(ww),
                 "l"(plain + w_src));
    asm volatile("cp.async.commit_group;\n");
  }
  int kt = 1;
  const int kmax = NT << 2;
  IPD_WIN(ww + W_ST, 1);
  ++kt;
  unsigned wwf = ww + 2 * W_ST;

  int acc[4][4];
#pragma unroll
  for (int s = 0; s < 4; ++s)
#pragma unroll
    for (int r = 0; r < 4; ++r)
      acc[s][r] = 0;

  const int lrow_a = wm * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
  const int lkb_a = (lane & 16) ? 16 : 0;
  const int lrow_b = wn * 32 + (lane >> 3) * 8 + (lane & 7);
  const unsigned pa0 =
    ip_sh((const signed char *)As + lrow_a * IPD_ALD + lkb_a);
  unsigned pw = ip_sh((const unsigned char *)Ws[0] + lrow_b * IPX_WLD);
  const unsigned pw_end = pw + IPX_ST * W_ST;
  int ka = 0;

  asm volatile("cp.async.wait_group %0;\n" ::"n"(IPX_ST - 2));
  __syncthreads();

#define IPD_FRAG(P, R0, R1, R2, R3)                                            \
  asm volatile(                                                                \
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"          \
    : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(P))

  int na0, na1, na2, na3, np0, np1, np2, np3;
  na0 = na1 = na2 = na3 = np0 = np1 = np2 = np3 = 0;
  if (!mdead) {
    IPD_FRAG(pa0, na0, na1, na2, na3);
    IPD_FRAG(pw, np0, np1, np2, np3);
  }

  // epilogue v2 row-side hoist: each thread's two output rows are fixed for
  // the whole CTA, so their zero-point/scale load ONCE (was per n-tile).
  const int epr0 = blockM + wm * 16 + g, epr1 = epr0 + 8;
  const int eaz0 = azp[epr0], eaz1 = azp[epr1];
  const float eas0 = ascale[epr0], eas1 = ascale[epr1];

  for (int j = 0; j < NT; ++j) {
    // col-side prefetch: issues before this tile's mma burst and is resident
    // by the time the per-tile epilogue runs (the load latency used to sit
    // exposed inside the epilogue, 32 times per CTA -- g3tax ladder: that
    // plus the software cvt was 3.1 ms of the 7.2 ms launch).
    int cf_rs[8];
    float cf_ws[8];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      if (mdead)
        break;
      const int pc0 = (j << 6) + wn * 32 + s * 8 + 2 * t;
      cf_rs[2 * s] = wrsum[pc0];
      cf_rs[2 * s + 1] = wrsum[pc0 + 1];
      cf_ws[2 * s] = hw_h2f(wscale[pc0]);
      cf_ws[2 * s + 1] = hw_h2f(wscale[pc0 + 1]);
    }
    for (int cs = 0; cs < 4; ++cs) {
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const int a0 = na0, a1 = na1, a2 = na2, a3 = na3;
        const int p0 = np0, p1 = np1, p2 = np2, p3 = np3;
        if (h == 2) {
          if (!mdead) {
            IPD_FRAG(pa0 + ka + 3 * 32, na0, na1, na2, na3);
            IPD_FRAG(pw + 3 * 16, np0, np1, np2, np3);
          }
          if (kt < kmax) {
            IPD_WIN(wwf, kt);
            ++kt;
            wwf += W_ST;
            wwf = (wwf >= ww_end) ? wwf - IPX_ST * W_ST : wwf;
          } else {
            asm volatile("cp.async.commit_group;\n");
          }
          asm volatile("cp.async.wait_group %0;\n" ::"n"(IPX_ST - 2));
          __syncthreads();
        } else if (h == 3) {
          pw += W_ST;
          pw = (pw >= pw_end) ? pw - IPX_ST * W_ST : pw;
          ka += 128;
          ka = (ka >= K) ? 0 : ka;
          if ((j * 4 + cs) + 1 < kmax && !mdead) {
            IPD_FRAG(pa0 + ka, na0, na1, na2, na3);
            IPD_FRAG(pw, np0, np1, np2, np3);
          }
        } else if (!mdead) {
          IPD_FRAG(pa0 + ka + (h + 1) * 32, na0, na1, na2, na3);
          IPD_FRAG(pw + (h + 1) * 16, np0, np1, np2, np3);
        }
#define IPD_MMA(S, PK)                                                         \
  do {                                                                         \
    const unsigned b0 = ip_nib2i8((PK) & 0x0F0F0F0Fu, 0u);                     \
    const unsigned b1 = ip_nib2i8(((unsigned)(PK) >> 4) & 0x0F0F0F0Fu, 0u);    \
    asm volatile(                                                              \
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "             \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"            \
      : "=r"(acc[S][0]), "=r"(acc[S][1]), "=r"(acc[S][2]), "=r"(acc[S][3])     \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),                  \
        "r"(acc[S][0]), "r"(acc[S][1]), "r"(acc[S][2]), "r"(acc[S][3]));       \
  } while (0)
        if (!mdead) {
          IPD_MMA(0, p0);
          IPD_MMA(1, p1);
          IPD_MMA(2, p2);
          IPD_MMA(3, p3);
        }
#undef IPD_MMA
      }
    }
    const int blockN = j << 6;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      if (mdead)
        break; // padding rows: outputs never read
      const int c0 = blockN + wn * 32 + s * 8 + 2 * t;
      const float v00 =
        (float)(acc[s][0] - eaz0 * cf_rs[2 * s]) * eas0 * cf_ws[2 * s];
      const float v01 =
        (float)(acc[s][1] - eaz0 * cf_rs[2 * s + 1]) * eas0 *
        cf_ws[2 * s + 1];
      const float v10 =
        (float)(acc[s][2] - eaz1 * cf_rs[2 * s]) * eas1 * cf_ws[2 * s];
      const float v11 =
        (float)(acc[s][3] - eaz1 * cf_rs[2 * s + 1]) * eas1 *
        cf_ws[2 * s + 1];
      if (out_fp16) {
        *(unsigned int *)((unsigned short *)Y + (long)epr0 * N + c0) =
          (unsigned int)hw_f2h(v00) | ((unsigned int)hw_f2h(v01) << 16);
        *(unsigned int *)((unsigned short *)Y + (long)epr1 * N + c0) =
          (unsigned int)hw_f2h(v10) | ((unsigned int)hw_f2h(v11) << 16);
      } else {
        IPf2 lo, hi;
        lo.x = v00; lo.y = v01; hi.x = v10; hi.y = v11;
        *(IPf2 *)(Y + (long)epr0 * N + c0) = lo;
        *(IPf2 *)(Y + (long)epr1 * N + c0) = hi;
      }
      acc[s][0] = acc[s][1] = acc[s][2] = acc[s][3] = 0;
    }
  }
}
#undef IPD_FRAG
#undef IPD_WIN

// Batched, IN-PLACE fragment repack: the permutation is 16B-slot-local, so a
// thread loads its slot, permutes in registers and stores back -- no scratch,
// no copy. One launch covers EVERY expert of a projection via the pointer
// table (the per-expert launch+memcpy version of this cost 9.3 s of first-
// prefill wall; this one is ~bandwidth: 15 GB r+w over all layers).
// QS4CX byte order (byte b: lo = k[2b], hi = k[2b+1]) -> fragment order
// (byte p, q = p>>2, j = p&3: lo = k[4q+j], hi = k[16+4q+j]).
__global__ void moe_repack_frag_g3(const unsigned long long *wp_tab, int E,
                                   long nslots) {
  unsigned char *pl = (unsigned char *)(unsigned long long)wp_tab[blockIdx.y];
  for (long s = (long)blockIdx.x * blockDim.x + threadIdx.x; s < nslots;
       s += (long)gridDim.x * blockDim.x) {
    IPv16 *slot = (IPv16 *)(pl + s * 16);
    const IPv16 v = *slot; // ONE 16B load (byte loads ran at ~9 GB/s)
    unsigned int w[4];
    w[0] = v.a; w[1] = v.b; w[2] = v.c; w[3] = v.d;
    unsigned char nib[32];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const unsigned int byi = (w[i >> 2] >> ((i & 3) * 8)) & 0xFFu;
      nib[2 * i] = (unsigned char)(byi & 0x0F);
      nib[2 * i + 1] = (unsigned char)(byi >> 4);
    }
    unsigned int o[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int p = 0; p < 16; ++p) {
      const int q = p >> 2, j = p & 3;
      const unsigned int byo =
        (unsigned int)(nib[4 * q + j] | (nib[16 + 4 * q + j] << 4));
      o[p >> 2] |= byo << ((p & 3) * 8);
    }
    IPv16 ov;
    ov.a = o[0]; ov.b = o[1]; ov.c = o[2]; ov.d = o[3];
    *slot = ov; // ONE 16B store
  }
}

// Batched per-output-channel nibble sum minus the offset, one launch per
// projection (rs[e*N + n]). Permutation-invariant: valid on raw OR repacked
// payload, and integer-equal to the unpacked arms' in-loop __dp4a rowsum.
__global__ void moe_rowsum_g3(const unsigned long long *wp_tab, int *rs,
                              int E, int N, int Kh) {
  // one WARP per row, lane-strided 4B reads (the thread-per-row byte walk
  // was ~3% coalescing-efficient and cost seconds over 15 GB of payload)
  const int e = blockIdx.y;
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * 8 + wid;
  if (n >= N)
    return;
  const unsigned int *row = (const unsigned int *)(
    (const unsigned char *)(unsigned long long)wp_tab[e] + (long)n * Kh);
  const int nw = Kh >> 2;
  int s = 0;
  for (int i = lane; i < nw; i += 32) {
    const unsigned int u = row[i];
    s = __dp4a((int)0x01010101, (int)(u & 0x0F0F0F0Fu), s);
    s = __dp4a((int)0x01010101, (int)((u >> 4) & 0x0F0F0F0Fu), s);
  }
  for (int o = 16; o > 0; o >>= 1)
    s += __shfl_down_sync(0xffffffffu, s, o);
  if (lane == 0)
    rs[(long)e * N + n] = s - 16 * Kh;
}

// -------------------------------------------------- Marlin-form m4 arm ----
// ip_wide_bench 2026-08-13: P5 warp geometry (warp owns all 64 M rows, 4
// n-warps x 2 k-warps) at BN=128 on a 4-stage cp.async ring = 30.2-30.9 TOPS
// L2-defeating vs 20.2 for this file's imma_gemm_pipe twin (+52%), bit-exact
// at every 35B projection shape. The two shapes that were REFUTED on the way
// (banked): register-fragment double-buffering (-17%: our 4-mma batches are
// too small to hide the same-slot WAR that Marlin's 16-32-mma batches bury),
// and BN=256 at 1 CTA/SM (24-26 TOPS: Orin cannot hide latency at 12.5%
// occupancy). BN widening pays ONLY on the P5 geometry -- the classic-tile
// n128 arm stays flat.
//
// The weight is consumed in OFFLINE FRAGMENT ORDER (repack_marlin_m4): per
// (128-col block bn, 64-k step ks) a 4096-B chunk of 256 16-B lane slots
// [(nw*2+kw)*32+lane]; a slot's packed nibbles are stored in exactly the
// order the in-kernel byte_perm/nib2i8 expand emits, so B needs ONE
// ld.shared.v4 per lane per step (no ldmatrix) and one B fragment feeds all
// four m-blocks: 16 mma per fetch, 3.2 mma/smem-instruction.
// The repack BAKES the raw/cached nibble plane in (cxnib = 0 raw payload,
// 8 for the ^0x88 DevWeightQ copy), so this kernel is cx-free; expanded
// values equal nib_offset-8 either way and the epilogue math is the exact
// arithmetic of imma_gemm_pipe's -- the result is BIT-IDENTICAL.
__global__ void repack_marlin_m4(const unsigned char *src, unsigned char *dst,
                                 int N, int K, int cxnib) {
  // grid: (N/128, K/64); 256 threads = one 16-B slot each
  const int bn = blockIdx.x, ks = blockIdx.y, nsteps = K >> 6;
  const int t = threadIdx.x;
  const int nw = t >> 6, kw = (t >> 5) & 1, lane = t & 31;
  const int Kh = K >> 1;
  unsigned char out[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    unsigned nib[2];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const int i = 2 * j + jj;
      const int f = i >> 3, w = (i >> 2) & 1, m = i & 3;
      const int row = bn * 128 + nw * 32 + f * 8 + (lane >> 2);
      const int k = ks * 64 + kw * 32 + w * 16 + 4 * (lane & 3) + m;
      const unsigned char b = src[(long)row * Kh + (k >> 1)];
      nib[jj] = ((k & 1) ? (b >> 4) : (b & 0xFu)) ^ (unsigned)cxnib;
    }
    out[j] = (unsigned char)(nib[0] | (nib[1] << 4));
  }
  long off = (((long)bn * nsteps + ks) * 256 + t) * 16;
#pragma unroll
  for (int j = 0; j < 16; ++j)
    dst[off + j] = out[j];
}

// Per-channel rowsum for the m4 arm when the source is the RAW offset-binary
// payload (the cached DevWeightQ arm reuses its existing rowsum table). Same
// integer semantic as weight_rowsum / the pipe's in-kernel rsacc:
// rowsum[n] = sum_k (nib(n,k) - 8).
__global__ void m4_rowsum(const unsigned char *src, int *rs, int N, int Kh) {
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * 8 + wid;
  if (n >= N)
    return;
  const unsigned int *row = (const unsigned int *)(src + (long)n * Kh);
  const int nw = Kh >> 2;
  int s = 0;
  for (int i = lane; i < nw; i += 32) {
    const unsigned int u = row[i];
    s = __dp4a((int)0x01010101, (int)(u & 0x0F0F0F0Fu), s);
    s = __dp4a((int)0x01010101, (int)((u >> 4) & 0x0F0F0F0Fu), s);
  }
  for (int o = 16; o > 0; o >>= 1)
    s += __shfl_down_sync(0xffffffffu, s, o);
  if (lane == 0)
    rs[n] = s - 16 * Kh;
}

#define M4_EXPAND(V0, V1, V2, V3, P0, P1)                                      \
  do {                                                                         \
    unsigned int lo_ = (P0)&0x0F0F0F0Fu;                                       \
    unsigned int hi_ = ((P0) >> 4) & 0x0F0F0F0Fu;                              \
    V0 = ip_nib2i8(__byte_perm(lo_, hi_, 0x5140), 0u);                         \
    V1 = ip_nib2i8(__byte_perm(lo_, hi_, 0x7362), 0u);                         \
    lo_ = (P1)&0x0F0F0F0Fu;                                                    \
    hi_ = ((P1) >> 4) & 0x0F0F0F0Fu;                                           \
    V2 = ip_nib2i8(__byte_perm(lo_, hi_, 0x5140), 0u);                         \
    V3 = ip_nib2i8(__byte_perm(lo_, hi_, 0x7362), 0u);                         \
  } while (0)
#define M4_MMA(ACC, AR, B0, B1)                                                \
  asm volatile(                                                                \
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "               \
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"              \
    : "=r"(ACC[0]), "=r"(ACC[1]), "=r"(ACC[2]), "=r"(ACC[3])                   \
    : "r"(AR[0]), "r"(AR[1]), "r"(AR[2]), "r"(AR[3]), "r"(B0), "r"(B1),        \
      "r"(ACC[0]), "r"(ACC[1]), "r"(ACC[2]), "r"(ACC[3]))
#define M4_LDM(D, P)                                                           \
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, "     \
               "[%4];"                                                         \
               : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]) : "r"(P))
__global__ __launch_bounds__(256, 2)
void imma_gemm_m4(const signed char *q8, const unsigned char *wm,
                  const float *ascale, const int *azp, const int *wrowsum,
                  const unsigned short *wscale, float *Y, int M, int N, int K,
                  int out_fp16) {
  __shared__ IPv16 A4sv[4][64 * IP_LD / 16];
  __shared__ IPv16 W4sv[4][256]; // 4096-B fragment-order packed stage
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int nw = warp & 3, kw = warp >> 2;
  const int blockM = blockIdx.y * 64, blockN = blockIdx.x * 128;

  const int srow = tid >> 2, ssub = tid & 3;
  const signed char *aptr = q8 + (long)(blockM + srow) * K + (ssub << 4);
  // M tail: an out-of-range row cp.asyncs with src-size 0, which zero-fills
  // the 16 smem bytes -- zero rows add nothing to acc, the epilogue guards.
  const int a_src_sz = (blockM + srow) < M ? 16 : 0;
  const int sdst = srow * IP_LD + (ssub << 4);
  const int nsteps = K >> 6; // caller gates K % 256 == 0
  const unsigned char *wsrc = wm + (long)blockIdx.x * nsteps * 4096 + tid * 16;
  unsigned adst[4], wdst[4];
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    adst[b] = ip_sh((signed char *)A4sv[b] + sdst);
    wdst[b] = ip_sh((signed char *)W4sv[b] + tid * 16);
  }

  int acc[4][4][4]; // [m-block][n8-frag][reg]
#pragma unroll
  for (int mb = 0; mb < 4; ++mb)
#pragma unroll
    for (int f = 0; f < 4; ++f)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        acc[mb][f][r] = 0;

#define M4_CPAW(BUF, KS)                                                       \
  do {                                                                         \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(     \
                   adst[BUF]),                                                 \
                 "l"(aptr + (KS)*64), "r"(a_src_sz));                          \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(         \
                   wdst[BUF]),                                                 \
                 "l"(wsrc + (long)(KS)*4096));                                 \
  } while (0)
#define M4_CPC() asm volatile("cp.async.commit_group;\n" ::)
#define M4_CPW1() asm volatile("cp.async.wait_group 1;\n" ::)

  M4_CPAW(0, 0);
  M4_CPC();
  M4_CPAW(1, 1);
  M4_CPC();
  M4_CPAW(2, 2);
  M4_CPC();
  M4_CPW1();
  __syncthreads();

  const int lrow_a = (lane & 7) + ((lane & 8) ? 8 : 0);
  const int lkb_a = kw * 32 + ((lane & 16) ? 16 : 0);
  unsigned pa_buf[4], pw_buf[4];
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    pa_buf[b] = ip_sh((const signed char *)A4sv[b] + lrow_a * IP_LD + lkb_a);
    pw_buf[b] =
      ip_sh((const signed char *)W4sv[b] + ((nw * 2 + kw) * 32 + lane) * 16);
  }

  for (int kb = 0; kb < nsteps; kb += 4) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int k = kb + u;
      const int u2 = (u + 2) & 3;
      if (k > 0) {
        if (k + 2 < nsteps)
          M4_CPAW(u2, k + 2);
        M4_CPC(); // unconditional (possibly empty): uniform group accounting
        M4_CPW1();
        __syncthreads();
      }
      const unsigned pa = pa_buf[u], pw = pw_buf[u];
      unsigned p0, p1, p2, p3;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3) : "r"(pw));
      int a0[4], a1[4], a2[4], a3[4];
      M4_LDM(a0, pa);
      M4_LDM(a1, pa + 16 * IP_LD);
      M4_LDM(a2, pa + 32 * IP_LD);
      M4_LDM(a3, pa + 48 * IP_LD);
      int c0, c1, c2, c3, c4, c5, c6, c7;
      M4_EXPAND(c0, c1, c2, c3, p0, p1); // f0.b0 f0.b1 f1.b0 f1.b1
      M4_EXPAND(c4, c5, c6, c7, p2, p3); // f2.b0 f2.b1 f3.b0 f3.b1
      M4_MMA(acc[0][0], a0, c0, c1);
      M4_MMA(acc[1][0], a1, c0, c1);
      M4_MMA(acc[2][0], a2, c0, c1);
      M4_MMA(acc[3][0], a3, c0, c1);
      M4_MMA(acc[0][1], a0, c2, c3);
      M4_MMA(acc[1][1], a1, c2, c3);
      M4_MMA(acc[2][1], a2, c2, c3);
      M4_MMA(acc[3][1], a3, c2, c3);
      M4_MMA(acc[0][2], a0, c4, c5);
      M4_MMA(acc[1][2], a1, c4, c5);
      M4_MMA(acc[2][2], a2, c4, c5);
      M4_MMA(acc[3][2], a3, c4, c5);
      M4_MMA(acc[0][3], a0, c6, c7);
      M4_MMA(acc[1][3], a1, c6, c7);
      M4_MMA(acc[2][3], a2, c6, c7);
      M4_MMA(acc[3][3], a3, c6, c7);
    }
  }
#undef M4_CPAW
#undef M4_CPC
#undef M4_CPW1

  // Cross-k-warp reduction in two 16-KB column halves through the (now dead)
  // A stage buffers, with the pipe's dequant epilogue fused into the kw1
  // pass. TOT = scr + acc is the full int32 accumulator -- int32 adds are
  // exact and satfinite cannot fire at |acc| <= K*127*8 -- so
  // (TOT - azp*rowsum) * ascale * wscale through the hardware cvt is the
  // same arithmetic on the same integers as imma_gemm_pipe: bit-identical.
  __syncthreads();
  int *scr = (int *)A4sv;
  const int g = lane >> 2, t2 = lane & 3;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    if (kw == 0) {
#pragma unroll
      for (int mb = 0; mb < 4; ++mb)
#pragma unroll
        for (int ff = 0; ff < 2; ++ff)
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const int f = half * 2 + ff;
            const int row = mb * 16 + g + ((r >> 1) ? 8 : 0);
            const int cidx = nw * 16 + ff * 8 + 2 * t2 + (r & 1);
            scr[row * 64 + cidx] = acc[mb][f][r];
          }
    }
    __syncthreads();
    if (kw == 1) {
#pragma unroll
      for (int mb = 0; mb < 4; ++mb) {
        const int row0 = blockM + mb * 16 + g, row1 = row0 + 8;
        const int az0 = row0 < M ? azp[row0] : 0;
        const int az1 = row1 < M ? azp[row1] : 0;
        const float as0 = row0 < M ? ascale[row0] : 0.0f;
        const float as1 = row1 < M ? ascale[row1] : 0.0f;
#pragma unroll
        for (int ff = 0; ff < 2; ++ff) {
          const int f = half * 2 + ff;
          const int c0 = blockN + nw * 32 + f * 8 + 2 * t2;
          const int i0 = (mb * 16 + g) * 64 + nw * 16 + ff * 8 + 2 * t2;
          const int rs0 = wrowsum[c0], rs1 = wrowsum[c0 + 1];
          const float ws0 = hw_h2f(wscale[c0]), ws1 = hw_h2f(wscale[c0 + 1]);
          if (row0 < M) {
            const float v00 =
              (float)(scr[i0] + acc[mb][f][0] - az0 * rs0) * as0 * ws0;
            const float v01 =
              (float)(scr[i0 + 1] + acc[mb][f][1] - az0 * rs1) * as0 * ws1;
            if (out_fp16)
              *(unsigned int *)((unsigned short *)Y + (long)row0 * N + c0) =
                (unsigned int)hw_f2h(v00) | ((unsigned int)hw_f2h(v01) << 16);
            else {
              IPf2 p;
              p.x = v00;
              p.y = v01;
              *(IPf2 *)(Y + (long)row0 * N + c0) = p;
            }
          }
          if (row1 < M) {
            const float v10 =
              (float)(scr[i0 + 512] + acc[mb][f][2] - az1 * rs0) * as1 * ws0;
            const float v11 =
              (float)(scr[i0 + 513] + acc[mb][f][3] - az1 * rs1) * as1 * ws1;
            if (out_fp16)
              *(unsigned int *)((unsigned short *)Y + (long)row1 * N + c0) =
                (unsigned int)hw_f2h(v10) | ((unsigned int)hw_f2h(v11) << 16);
            else {
              IPf2 p;
              p.x = v10;
              p.y = v11;
              *(IPf2 *)(Y + (long)row1 * N + c0) = p;
            }
          }
        }
      }
    }
    __syncthreads();
  }
}
// Slab-to-slab m4-order repack for a MoE projection: every expert's payload
// moves from QS4CX row-major byte order into imma_moe_g4's fragment-chunk
// order (same 4096-B chunk layout as repack_marlin_m4, per expert). Global
// permutation -- needs distinct src/dst (the g3 repack is 16B-slot-local and
// runs in place; this one is not). Raw nibble VALUES are preserved
// (cx stays with the consumer, as for g3).
__global__ void moe_repack_m4(const unsigned char *src, unsigned char *dst,
                              int E, int N, int K) {
  // grid: (N/128, K/64, E); 256 threads = one 16-B slot each
  const int bn = blockIdx.x, ks = blockIdx.y, e = blockIdx.z;
  const int nsteps = K >> 6;
  const int t = threadIdx.x;
  const int nw = t >> 6, kw = (t >> 5) & 1, lane = t & 31;
  const int Kh = K >> 1;
  const size_t esz = (size_t)N * Kh;
  const unsigned char *s = src + (size_t)e * esz;
  unsigned char *d = dst + (size_t)e * esz;
  unsigned char out[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    unsigned nib[2];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const int i = 2 * j + jj;
      const int f = i >> 3, w = (i >> 2) & 1, m = i & 3;
      const int row = bn * 128 + nw * 32 + f * 8 + (lane >> 2);
      const int k = ks * 64 + kw * 32 + w * 16 + 4 * (lane & 3) + m;
      const unsigned char b = s[(long)row * Kh + (k >> 1)];
      nib[jj] = (k & 1) ? (b >> 4) : (b & 0xFu);
    }
    out[j] = (unsigned char)(nib[0] | (nib[1] << 4));
  }
  long off = (((long)bn * nsteps + ks) * 256 + t) * 16;
#pragma unroll
  for (int j = 0; j < 16; ++j)
    d[off + j] = out[j];
}

// MoE grouped gate/up on the m4 geometry: imma_gemm_m4's warp form (warp
// owns all 64 window rows, 4 n-warps x 2 k-warps, one ld.shared.v4 of
// fragment-order packed W per lane per k-step feeding 16 mma) driven by the
// g3 steering (block_expert window table, tokid gather with src-size-0
// zero-fill for padding rows, per-expert scale/rowsum tables, wl_n live
// counts). The g3 dead-SUBTILE skip becomes a dead-M-BLOCK skip: every warp
// guards each 16-row m-block on (mb*16 < live) uniformly, so a decode
// window (live=1) skips 3/4 of its mma on all 8 warps. Epilogue = the m4
// cross-k-warp int32 reduction with g3's per-token azp/ascale and
// per-expert rowsum/scale -- same integers, same arithmetic, bit-identical
// to g3 (dead m-blocks store nothing, exactly like g3's mdead warps).
// Requires K % 256 == 0 and N % 128 == 0 (the launcher gates).
__global__ __launch_bounds__(256, 2)
void imma_moe_g4(const signed char *q8, const int *tokid,
                 const unsigned long long *wp_tab,
                 const unsigned long long *ws_tab,
                 const unsigned long long *wr_tab, const int *block_expert,
                 const float *ascale, const int *azp, float *Y, int N, int K,
                 int out_fp16, const int *wl_n) {
  const int e = block_expert[blockIdx.y];
  if (e < 0)
    return;
  const unsigned char *plain =
    (const unsigned char *)(unsigned long long)wp_tab[e];
  const unsigned short *wscale =
    (const unsigned short *)(unsigned long long)ws_tab[e];
  const int *wrsum = (const int *)(unsigned long long)wr_tab[e];
  __shared__ IPv16 A4sv[4][64 * IP_LD / 16];
  __shared__ IPv16 W4sv[4][256];
  __shared__ int toks[64];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int nw = warp & 3, kw = warp >> 2;
  const int blockM = blockIdx.y * 64, blockN = blockIdx.x * 128;
  if (tid < 64)
    toks[tid] = tokid ? tokid[blockM + tid] : (blockM + tid);
  __syncthreads();
  const int live = wl_n ? wl_n[blockIdx.y] : 64;

  const int srow = tid >> 2, ssub = tid & 3;
  const int atok = toks[srow];
  const signed char *aptr = q8 + (long)atok * K + (ssub << 4);
  const int a_src_sz = atok >= 0 ? 16 : 0; // padding rows zero-fill
  const int sdst = srow * IP_LD + (ssub << 4);
  const int nsteps = K >> 6;
  const unsigned char *wsrc =
    plain + (long)blockIdx.x * nsteps * 4096 + tid * 16;
  unsigned adst[4], wdst[4];
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    adst[b] = ip_sh((signed char *)A4sv[b] + sdst);
    wdst[b] = ip_sh((signed char *)W4sv[b] + tid * 16);
  }

  int acc[4][4][4];
#pragma unroll
  for (int mb = 0; mb < 4; ++mb)
#pragma unroll
    for (int f = 0; f < 4; ++f)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        acc[mb][f][r] = 0;

#define G4_CPAW(BUF, KS)                                                       \
  do {                                                                         \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(     \
                   adst[BUF]),                                                 \
                 "l"(aptr + (KS)*64), "r"(a_src_sz));                          \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(         \
                   wdst[BUF]),                                                 \
                 "l"(wsrc + (long)(KS)*4096));                                 \
  } while (0)
#define G4_CPC() asm volatile("cp.async.commit_group;\n" ::)
#define G4_CPW1() asm volatile("cp.async.wait_group 1;\n" ::)

  G4_CPAW(0, 0);
  G4_CPC();
  G4_CPAW(1, 1);
  G4_CPC();
  G4_CPAW(2, 2);
  G4_CPC();
  G4_CPW1();
  __syncthreads();

  const int lrow_a = (lane & 7) + ((lane & 8) ? 8 : 0);
  const int lkb_a = kw * 32 + ((lane & 16) ? 16 : 0);
  unsigned pa_buf[4], pw_buf[4];
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    pa_buf[b] = ip_sh((const signed char *)A4sv[b] + lrow_a * IP_LD + lkb_a);
    pw_buf[b] =
      ip_sh((const signed char *)W4sv[b] + ((nw * 2 + kw) * 32 + lane) * 16);
  }
  const bool ml0 = 0 < live, ml1 = 16 < live, ml2 = 32 < live, ml3 = 48 < live;

  for (int kb = 0; kb < nsteps; kb += 4) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int k = kb + u;
      const int u2 = (u + 2) & 3;
      if (k > 0) {
        if (k + 2 < nsteps)
          G4_CPAW(u2, k + 2);
        G4_CPC();
        G4_CPW1();
        __syncthreads();
      }
      const unsigned pa = pa_buf[u], pw = pw_buf[u];
      unsigned p0, p1, p2, p3;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3) : "r"(pw));
      int a0[4], a1[4], a2[4], a3[4];
      if (ml0)
        M4_LDM(a0, pa);
      if (ml1)
        M4_LDM(a1, pa + 16 * IP_LD);
      if (ml2)
        M4_LDM(a2, pa + 32 * IP_LD);
      if (ml3)
        M4_LDM(a3, pa + 48 * IP_LD);
      int c0, c1, c2, c3, c4, c5, c6, c7;
      M4_EXPAND(c0, c1, c2, c3, p0, p1);
      M4_EXPAND(c4, c5, c6, c7, p2, p3);
      if (ml0) {
        M4_MMA(acc[0][0], a0, c0, c1);
        M4_MMA(acc[0][1], a0, c2, c3);
        M4_MMA(acc[0][2], a0, c4, c5);
        M4_MMA(acc[0][3], a0, c6, c7);
      }
      if (ml1) {
        M4_MMA(acc[1][0], a1, c0, c1);
        M4_MMA(acc[1][1], a1, c2, c3);
        M4_MMA(acc[1][2], a1, c4, c5);
        M4_MMA(acc[1][3], a1, c6, c7);
      }
      if (ml2) {
        M4_MMA(acc[2][0], a2, c0, c1);
        M4_MMA(acc[2][1], a2, c2, c3);
        M4_MMA(acc[2][2], a2, c4, c5);
        M4_MMA(acc[2][3], a2, c6, c7);
      }
      if (ml3) {
        M4_MMA(acc[3][0], a3, c0, c1);
        M4_MMA(acc[3][1], a3, c2, c3);
        M4_MMA(acc[3][2], a3, c4, c5);
        M4_MMA(acc[3][3], a3, c6, c7);
      }
    }
  }
#undef G4_CPAW
#undef G4_CPC
#undef G4_CPW1

  __syncthreads();
  int *scr = (int *)A4sv;
  const int g = lane >> 2, t2 = lane & 3;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    if (kw == 0) {
#pragma unroll
      for (int mb = 0; mb < 4; ++mb) {
        if (mb * 16 >= live)
          continue;
#pragma unroll
        for (int ff = 0; ff < 2; ++ff)
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const int f = half * 2 + ff;
            const int row = mb * 16 + g + ((r >> 1) ? 8 : 0);
            const int cidx = nw * 16 + ff * 8 + 2 * t2 + (r & 1);
            scr[row * 64 + cidx] = acc[mb][f][r];
          }
      }
    }
    __syncthreads();
    if (kw == 1) {
#pragma unroll
      for (int mb = 0; mb < 4; ++mb) {
        if (mb * 16 >= live)
          continue; // dead m-block: store nothing (g3 mdead parity)
        const int lr0 = mb * 16 + g, lr1 = lr0 + 8;
        const int tk0 = toks[lr0], tk1 = toks[lr1];
        const int az0 = tk0 >= 0 ? azp[tk0] : 0;
        const int az1 = tk1 >= 0 ? azp[tk1] : 0;
        const float as0 = tk0 >= 0 ? ascale[tk0] : 0.0f;
        const float as1 = tk1 >= 0 ? ascale[tk1] : 0.0f;
#pragma unroll
        for (int ff = 0; ff < 2; ++ff) {
          const int f = half * 2 + ff;
          const int c0 = blockN + nw * 32 + f * 8 + 2 * t2;
          const int i0 = lr0 * 64 + nw * 16 + ff * 8 + 2 * t2;
          const int rs0 = wrsum[c0], rs1 = wrsum[c0 + 1];
          const float ws0 = hw_h2f(wscale[c0]), ws1 = hw_h2f(wscale[c0 + 1]);
          const float v00 =
            tk0 >= 0 ? (float)(scr[i0] + acc[mb][f][0] - az0 * rs0) * as0 * ws0
                     : 0.0f;
          const float v01 = tk0 >= 0
                              ? (float)(scr[i0 + 1] + acc[mb][f][1] - az0 * rs1)
                                  * as0 * ws1
                              : 0.0f;
          const float v10 = tk1 >= 0
                              ? (float)(scr[i0 + 512] + acc[mb][f][2]
                                        - az1 * rs0) * as1 * ws0
                              : 0.0f;
          const float v11 = tk1 >= 0
                              ? (float)(scr[i0 + 513] + acc[mb][f][3]
                                        - az1 * rs1) * as1 * ws1
                              : 0.0f;
          if (out_fp16) {
            *(unsigned int *)((unsigned short *)Y + (long)(blockM + lr0) * N +
                              c0) =
              (unsigned int)hw_f2h(v00) | ((unsigned int)hw_f2h(v01) << 16);
            *(unsigned int *)((unsigned short *)Y + (long)(blockM + lr1) * N +
                              c0) =
              (unsigned int)hw_f2h(v10) | ((unsigned int)hw_f2h(v11) << 16);
          } else {
            IPf2 lo, hi;
            lo.x = v00;
            lo.y = v01;
            hi.x = v10;
            hi.y = v11;
            *(IPf2 *)(Y + (long)(blockM + lr0) * N + c0) = lo;
            *(IPf2 *)(Y + (long)(blockM + lr1) * N + c0) = hi;
          }
        }
      }
    }
    __syncthreads();
  }
}
#undef M4_EXPAND
#undef M4_MMA
#undef M4_LDM

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

// --- Marlin-form m4 arm (opt-in NNTR_FC_MARLIN=1) --------------------------
// Fragment-order device weight + rowsum table for imma_gemm_m4, keyed by the
// pointer the pipe arm would have read (raw payload or DevWeightQ copy). The
// repack bakes that arm's nibble plane in (cxnib), so the kernel is cx-free
// and its output stays bit-identical to imma_gemm_pipe's. Building a device
// copy also fixes the operand kind for house-pool payloads.
struct M4WeightQ {
  unsigned char *wm;
  int *rowsum; // owned only when built here; the cached arm reuses DevWeightQ's
  bool owns_rowsum;
};
static std::unordered_map<const void *, M4WeightQ> g_m4_cache;

// DEFAULT ON (NNTR_FC_MARLIN=0 restores the pipe arm): 20K prefill
// 1,968.9/1,963.8 -> 2,060.9/2,073.5 TPS (+5.1%), text byte-identical,
// decode untouched (M >= 256 gate keeps the GEMV arms).
static bool marlin_on() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_FC_MARLIN");
    return !(e && e[0] == '0');
  }();
  return on;
}

// Caller holds g_dp4a_mtx.
static M4WeightQ *ensure_m4_cache_locked(const void *src, unsigned int N,
                                         unsigned int K, int cxnib,
                                         int *rowsum_reuse) {
  auto it = g_m4_cache.find(src);
  if (it != g_m4_cache.end())
    return &it->second;
  auto kr = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "repack_marlin_m4");
  auto ks = rowsum_reuse ? nullptr
                         : CudaContext::Global().registerCudaKernel(
                             FC_QINT4_DP4A_SRC, "m4_rowsum");
  if (!kr || (!rowsum_reuse && !ks))
    return nullptr;
  M4WeightQ mw{};
  const size_t bytes = (size_t)N * K / 2u;
  if (cudaMalloc(&mw.wm, bytes) != cudaSuccess)
    return nullptr;
  const int n = (int)N, k = (int)K;
  kr->SetKernelArguments(0, &src, sizeof(src));
  kr->SetKernelArguments(1, &mw.wm, sizeof(mw.wm));
  kr->SetKernelArguments(2, &n, sizeof(n));
  kr->SetKernelArguments(3, &k, sizeof(k));
  kr->SetKernelArguments(4, &cxnib, sizeof(cxnib));
  const int rb[3] = {256, 1, 1};
  const int rg[3] = {(int)(N / 128u), (int)(K / 64u), 1};
  if (!StreamManager::Global().DispatchCommand(*kr, rg, rb)) {
    cudaFree(mw.wm);
    return nullptr;
  }
  if (rowsum_reuse) {
    mw.rowsum = rowsum_reuse;
    mw.owns_rowsum = false;
  } else {
    if (cudaMalloc(&mw.rowsum, sizeof(int) * (size_t)N) != cudaSuccess) {
      cudaFree(mw.wm);
      return nullptr;
    }
    mw.owns_rowsum = true;
    const int kh = (int)(K / 2u);
    ks->SetKernelArguments(0, &src, sizeof(src));
    ks->SetKernelArguments(1, &mw.rowsum, sizeof(mw.rowsum));
    ks->SetKernelArguments(2, &n, sizeof(n));
    ks->SetKernelArguments(3, &kh, sizeof(kh));
    const int sb[3] = {256, 1, 1};
    const int sg[3] = {((int)N + 7) / 8, 1, 1};
    if (!StreamManager::Global().DispatchCommand(*ks, sg, sb)) {
      cudaFree(mw.wm);
      cudaFree(mw.rowsum);
      return nullptr;
    }
  }
  it = g_m4_cache.emplace(src, mw).first;
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

  // Marlin-form m4 arm: P5 warp geometry at BN=128 (bench +52% over the pipe
  // at every 35B projection shape, bit-identical). Prefill-scale GEMMs only;
  // any cache/registration failure falls through to the pipe arm untouched.
  const bool use_m4 = use_pipe && marlin_on() && (N % 128u) == 0u &&
                      (K % 256u) == 0u && M >= 256u;
  if (use_m4) {
    M4WeightQ *mw = ensure_m4_cache_locked(w_arg, N, K, raw_i ? 0 : 8,
                                           raw_i ? nullptr : wrowsum);
    auto k4 = mw ? CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                            "imma_gemm_m4")
                 : nullptr;
    if (k4) {
      const int mm4 = (int)M;
      k4->SetKernelArguments(0, &g_dp4a_q8, sizeof(g_dp4a_q8));
      k4->SetKernelArguments(1, &mw->wm, sizeof(mw->wm));
      k4->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
      k4->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
      k4->SetKernelArguments(4, &mw->rowsum, sizeof(mw->rowsum));
      k4->SetKernelArguments(5, &scales_fp16, sizeof(scales_fp16));
      k4->SetKernelArguments(6, &Yf, sizeof(Yf));
      k4->SetKernelArguments(7, &mm4, sizeof(mm4));
      k4->SetKernelArguments(8, &n, sizeof(n));
      k4->SetKernelArguments(9, &k, sizeof(k));
      k4->SetKernelArguments(10, &out_fp16, sizeof(out_fp16));
      const int ib[3] = {256, 1, 1};
      const int ig[3] = {(int)(N / 128u), ((int)M + 63) / 64, 1};
      return StreamManager::Global().DispatchCommand(*k4, ig, ib);
    }
  }

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

extern bool g_last_quant_valid; // defined below with the staging record

static bool dp4a_stage_scratch(unsigned int M, unsigned int K) {
  const size_t cap0 = g_dp4a_q8_cap;
  const bool ok = ensure_buf((void **)&g_dp4a_q8, &g_dp4a_q8_cap,
                             (size_t)M * K + FC_I8_TAIL_PAD) &&
                  ensure_buf((void **)&g_dp4a_ascale, &g_dp4a_ascale_cap,
                             sizeof(float) * (size_t)M) &&
                  ensure_buf((void **)&g_dp4a_azp, &g_dp4a_azp_cap,
                             sizeof(int) * (size_t)M);
  // A grow reallocates g_dp4a_q8 — any staged quant died with the old buffer,
  // and a pointer+seq match must not be allowed to resurrect it.
  if (g_dp4a_q8_cap != cap0)
    g_last_quant_valid = false;
  return ok;
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

// NNTR_NORM_V4P: vectorized prefill norms (rmsnorm_quant_i8_h_v4p here,
// rmsnorm_fp16_w4p in cuda_rmsnorm.cpp). NOT bit-identical to the scalar
// kernels -- the sum-of-squares reduction order changes, so `inv` can move
// by an ulp (same deviation class as the decode v4 arm, NLL-gated). =0
// restores the scalar prefill kernels; decode (rows<=32) is unaffected.
bool norm_v4p_on() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_NORM_V4P");
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
  const bool ok = g_last_quant_valid && xh == g_last_quant_xh &&
                  k == g_last_quant_k &&
                  StreamManager::Global().dispatchSeq() == g_last_quant_seq;
  // NNTR_QUANT_DBG=1: print the first probes with the failing term (pointer
  // vs K vs sequence drift) so a staging miss can be attributed, not guessed.
  static int dbg = -1;
  if (dbg < 0) {
    const char *e = std::getenv("NNTR_QUANT_DBG");
    dbg = (e != nullptr && e[0] == '1') ? 0 : 1 << 30;
    if (dbg == 0)
      StreamManager::Global().enableLaunchTrace();
  }
  if (dbg < 48) {
    const long long dseq = (long long)(StreamManager::Global().dispatchSeq() -
                                       g_last_quant_seq);
    fprintf(stderr, "[qstage] xh=%p k=%d ok=%d ptr=%d kk=%d dseq=%lld\n", xh, k,
            (int)ok, (int)(xh == g_last_quant_xh), (int)(k == g_last_quant_k),
            dseq);
    for (long long b = 0; !ok && b < dseq && b < 8; ++b)
      fprintf(stderr, "[qstage]   -%lld: %s\n", b + 1,
              StreamManager::Global().lastLaunch((unsigned)b));
    ++dbg;
  }
  return ok;
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

// See cuda_common.h. Extends only a stage that was valid up to EXACTLY this
// one dispatch (seq == staged+1): a chain of benign dispatches each extends
// by one, while a stage that already lapsed stays lapsed — reviving it would
// reopen the recycled-pointer hazard, because the re-purposing writer's own
// dispatch is exactly what lapsed it. (Defined OUTSIDE the anonymous
// namespace: the callers are other translation units; the staging record it
// reads stays internal.)
void quant_stage_survive(const void *w0, const void *w1, const void *w2) {
  if (!g_last_quant_valid)
    return;
  if (w0 == g_last_quant_xh || (w1 != nullptr && w1 == g_last_quant_xh) ||
      (w2 != nullptr && w2 == g_last_quant_xh))
    return;
  if (StreamManager::Global().dispatchSeq() == g_last_quant_seq + 1)
    g_last_quant_seq = StreamManager::Global().dispatchSeq();
}

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
  // NNTR_IMMA_CK=1: run the clock64 phase-bracket twin for the first few
  // launches (identical output; prints per-k-step cycle split, then falls
  // back to the production kernel for the rest of the run).
  static const bool g_ck_dbg = []() {
    const char *e = std::getenv("NNTR_IMMA_CK");
    return e != nullptr && e[0] == '1';
  }();
  static int g_ck_n = 0;
  if (g_ck_dbg && g_ck_n < 6) {
    auto kc = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                       "imma_moe_grouped_ck");
    static unsigned long long *d_ck = nullptr;
    if (kc && (d_ck || cudaMalloc(&d_ck, 8 * sizeof(unsigned long long)) ==
                         cudaSuccess)) {
      cudaStream_t st = StreamManager::Global().GetStream();
      cudaMemsetAsync(d_ck, 0, 8 * sizeof(unsigned long long), st);
      const int n2 = (int)N, k2 = (int)K;
      kc->SetKernelArguments(0, &q8, sizeof(q8));
      kc->SetKernelArguments(1, &tokid, sizeof(tokid));
      kc->SetKernelArguments(2, &wp_tab, sizeof(wp_tab));
      kc->SetKernelArguments(3, &ws_tab, sizeof(ws_tab));
      kc->SetKernelArguments(4, &block_expert, sizeof(block_expert));
      kc->SetKernelArguments(5, &ascale, sizeof(ascale));
      kc->SetKernelArguments(6, &azp, sizeof(azp));
      kc->SetKernelArguments(7, &Y, sizeof(Y));
      kc->SetKernelArguments(8, &n2, sizeof(n2));
      kc->SetKernelArguments(9, &k2, sizeof(k2));
      kc->SetKernelArguments(10, &out_fp16, sizeof(out_fp16));
      kc->SetKernelArguments(11, &d_ck, sizeof(d_ck));
      const int cb[3] = {256, 1, 1};
      const int cg[3] = {(int)(N / 64u), (int)n_mblocks, 1};
      if (StreamManager::Global().DispatchCommand(*kc, cg, cb)) {
        unsigned long long h_ck[6] = {0, 0, 0, 0, 0, 0};
        cudaStreamSynchronize(st);
        cudaMemcpy(h_ck, d_ck, sizeof(h_ck), cudaMemcpyDeviceToHost);
        const double s = h_ck[5] ? (double)h_ck[5] : 1.0;
        const double ld = h_ck[0] / s, h0 = h_ck[1] / s, h1 = h_ck[2] / s;
        const double w = h_ck[3] / s, b = h_ck[4] / s;
        const double tot = ld + h0 + h1 + w + b;
        fprintf(stderr,
                "[imma_ck] N=%u K=%u mblocks=%u steps=%llu cyc/kstep: "
                "ld_issue=%.0f h0=%.0f h1=%.0f store=%.0f bar=%.0f "
                "total=%.0f\n",
                N, K, n_mblocks, h_ck[5], ld, h0, h1, w, b, tot);
        ++g_ck_n;
        return true;
      }
    }
    // twin unavailable: fall through to the production kernel
  }
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

bool cuda_fc_qs4cx_moe_grouped_gemm2(
  const signed char *q8, const int *tokid, const unsigned long long *wpg_tab,
  const unsigned long long *wsg_tab, const unsigned long long *wpu_tab,
  const unsigned long long *wsu_tab, const int *block_expert,
  const float *ascale, const int *azp, void *Yg, void *Yu,
  unsigned int n_mblocks, unsigned int N, unsigned int K, int out_fp16) {
  if (q8 == nullptr || wpg_tab == nullptr || wsg_tab == nullptr ||
      wpu_tab == nullptr || wsu_tab == nullptr || block_expert == nullptr ||
      Yg == nullptr || Yu == nullptr || n_mblocks == 0 || (N & 63u) != 0u ||
      (K & 63u) != 0u)
    return false;
  auto kg = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "imma_moe_grouped_g2");
  if (!kg)
    return false;
  const int n = (int)N, k = (int)K;
  kg->SetKernelArguments(0, &q8, sizeof(q8));
  kg->SetKernelArguments(1, &tokid, sizeof(tokid));
  kg->SetKernelArguments(2, &wpg_tab, sizeof(wpg_tab));
  kg->SetKernelArguments(3, &wsg_tab, sizeof(wsg_tab));
  kg->SetKernelArguments(4, &wpu_tab, sizeof(wpu_tab));
  kg->SetKernelArguments(5, &wsu_tab, sizeof(wsu_tab));
  kg->SetKernelArguments(6, &block_expert, sizeof(block_expert));
  kg->SetKernelArguments(7, &ascale, sizeof(ascale));
  kg->SetKernelArguments(8, &azp, sizeof(azp));
  kg->SetKernelArguments(9, &Yg, sizeof(Yg));
  kg->SetKernelArguments(10, &Yu, sizeof(Yu));
  kg->SetKernelArguments(11, &n, sizeof(n));
  kg->SetKernelArguments(12, &k, sizeof(k));
  kg->SetKernelArguments(13, &out_fp16, sizeof(out_fp16));
  const int ib[3] = {256, 1, 1};
  const int ig[3] = {(int)(N / 64u), (int)n_mblocks, 1};
  return StreamManager::Global().DispatchCommand(*kg, ig, ib);
}

bool cuda_fc_qs4cx_moe_grouped_gemm_w(
  const signed char *q8, const int *tokid, const unsigned long long *wp_tab,
  const unsigned long long *ws_tab, const int *block_expert,
  const float *ascale, const int *azp, void *Y, unsigned int n_mblocks,
  unsigned int N, unsigned int K, int out_fp16) {
  if (q8 == nullptr || wp_tab == nullptr || ws_tab == nullptr ||
      block_expert == nullptr || Y == nullptr || n_mblocks == 0 ||
      (N & 127u) != 0u || (K & 63u) != 0u)
    return false;
  auto kg = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "imma_moe_grouped_w");
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
  const int ig[3] = {(int)(N / 128u), (int)n_mblocks, 1};
  return StreamManager::Global().DispatchCommand(*kg, ig, ib);
}

// NNTR_MOE_G3 launcher: same shape/steering as the _gemm entry plus the
// per-expert rowsum pointer table. Requires the payload to have been through
// cuda_fc_qs4cx_moe_repack_g3 (fragment order) -- the caller owns that
// invariant; nothing here can verify it.
bool cuda_fc_qs4cx_moe_grouped_gemm_g3(
  const signed char *q8, const int *tokid, const unsigned long long *wp_tab,
  const unsigned long long *ws_tab, const unsigned long long *wr_tab,
  const int *block_expert, const float *ascale, const int *azp, void *Y,
  unsigned int n_mblocks, unsigned int N, unsigned int K, int out_fp16,
  const int *wl_n) {
  if (q8 == nullptr || wp_tab == nullptr || ws_tab == nullptr ||
      wr_tab == nullptr || block_expert == nullptr || Y == nullptr ||
      n_mblocks == 0 || (N & 63u) != 0u || (K & 127u) != 0u || K < 256u)
    return false;
  auto kg = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "imma_moe_grouped_g3");
  if (!kg)
    return false;
  static bool attr_once = []() {
    const char *e = std::getenv("NNTR_MOE_G_DBG");
    return e != nullptr && e[0] != '0';
  }();
  if (attr_once) {
    attr_once = false;
    int nregs = 0, lmem = 0, smem = 0, maxthr = 0;
    cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, kg->GetFunction());
    cuFuncGetAttribute(&lmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                       kg->GetFunction());
    cuFuncGetAttribute(&smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                       kg->GetFunction());
    cuFuncGetAttribute(&maxthr, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                       kg->GetFunction());
    fprintf(stderr,
            "[g3_attr] regs=%d local=%dB smem_static=%dB maxthreads=%d\n",
            nregs, lmem, smem, maxthr);
  }
  const int n = (int)N, k = (int)K;
  kg->SetKernelArguments(0, &q8, sizeof(q8));
  kg->SetKernelArguments(1, &tokid, sizeof(tokid));
  kg->SetKernelArguments(2, &wp_tab, sizeof(wp_tab));
  kg->SetKernelArguments(3, &ws_tab, sizeof(ws_tab));
  kg->SetKernelArguments(4, &wr_tab, sizeof(wr_tab));
  kg->SetKernelArguments(5, &block_expert, sizeof(block_expert));
  kg->SetKernelArguments(6, &ascale, sizeof(ascale));
  kg->SetKernelArguments(7, &azp, sizeof(azp));
  kg->SetKernelArguments(8, &Y, sizeof(Y));
  kg->SetKernelArguments(9, &n, sizeof(n));
  kg->SetKernelArguments(10, &k, sizeof(k));
  kg->SetKernelArguments(11, &out_fp16, sizeof(out_fp16));
  kg->SetKernelArguments(12, &wl_n, sizeof(wl_n));
  const int ib[3] = {256, 1, 1};
  const int ig[3] = {(int)(N / 64u), (int)n_mblocks, 1};
  return StreamManager::Global().DispatchCommand(*kg, ig, ib);
}

// down (K <= 512) persistent-N launcher: grid (1, W) -- see the kernel note.
bool cuda_fc_qs4cx_moe_grouped_gemm_g3d(
  const signed char *q8, const unsigned long long *wp_tab,
  const unsigned long long *ws_tab, const unsigned long long *wr_tab,
  const int *block_expert, const float *ascale, const int *azp, void *Y,
  unsigned int n_mblocks, unsigned int N, unsigned int K, int out_fp16,
  const int *wl_n) {
  if (q8 == nullptr || wp_tab == nullptr || ws_tab == nullptr ||
      wr_tab == nullptr || block_expert == nullptr || Y == nullptr ||
      n_mblocks == 0 || (N & 63u) != 0u || (K & 127u) != 0u || K < 256u ||
      K > 512u)
    return false;
  auto kg = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "imma_moe_grouped_g3d");
  if (!kg)
    return false;
  const int n = (int)N, k = (int)K;
  kg->SetKernelArguments(0, &q8, sizeof(q8));
  kg->SetKernelArguments(1, &wp_tab, sizeof(wp_tab));
  kg->SetKernelArguments(2, &ws_tab, sizeof(ws_tab));
  kg->SetKernelArguments(3, &wr_tab, sizeof(wr_tab));
  kg->SetKernelArguments(4, &block_expert, sizeof(block_expert));
  kg->SetKernelArguments(5, &ascale, sizeof(ascale));
  kg->SetKernelArguments(6, &azp, sizeof(azp));
  kg->SetKernelArguments(7, &Y, sizeof(Y));
  kg->SetKernelArguments(8, &n, sizeof(n));
  kg->SetKernelArguments(9, &k, sizeof(k));
  kg->SetKernelArguments(10, &out_fp16, sizeof(out_fp16));
  kg->SetKernelArguments(11, &wl_n, sizeof(wl_n));
  const int ib[3] = {256, 1, 1};
  const int ig[3] = {1, (int)n_mblocks, 1};
  return StreamManager::Global().DispatchCommand(*kg, ig, ib);
}

// m4-order slab-to-slab repack (see header note). The permutation is global,
// so it bounces through a fresh slab; the old slab is freed on success.
bool cuda_fc_qs4cx_moe_repack_m4(unsigned long long *wp_tab, unsigned int E,
                                 unsigned int N, unsigned int K) {
  const size_t esz = (size_t)N * (K >> 1);
  if (wp_tab == nullptr || E == 0 || (N & 127u) != 0u || (K & 255u) != 0u ||
      esz == 0)
    return false;
  // contiguity invariant: the slab builder laid experts out at stride esz
  unsigned char *src = (unsigned char *)(unsigned long long)wp_tab[0];
  for (unsigned int e = 0; e < E; ++e)
    if ((unsigned char *)(unsigned long long)wp_tab[e] != src + (size_t)e * esz)
      return false;
  auto kr = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "moe_repack_m4");
  if (!kr)
    return false;
  unsigned char *dst = nullptr;
  if (cudaMalloc(&dst, esz * E) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  const int e_ = (int)E, n = (int)N, k = (int)K;
  kr->SetKernelArguments(0, &src, sizeof(src));
  kr->SetKernelArguments(1, &dst, sizeof(dst));
  kr->SetKernelArguments(2, &e_, sizeof(e_));
  kr->SetKernelArguments(3, &n, sizeof(n));
  kr->SetKernelArguments(4, &k, sizeof(k));
  const int b[3] = {256, 1, 1};
  const int g[3] = {(int)(N / 128u), (int)(K / 64u), (int)E};
  if (!StreamManager::Global().DispatchCommand(*kr, g, b)) {
    cudaFree(dst);
    return false;
  }
  // The repack must complete before the source slab is freed and before any
  // caller-side rowsum reads the new order through the repointed table.
  if (cudaDeviceSynchronize() != cudaSuccess) {
    cudaFree(dst);
    return false;
  }
  for (unsigned int e = 0; e < E; ++e)
    wp_tab[e] = (unsigned long long)(dst + (size_t)e * esz);
  cudaFree(src);
  return true;
}

// m4 grouped gate/up launcher: g3's contract on the m4-order payload.
bool cuda_fc_qs4cx_moe_grouped_gemm_g4(
  const signed char *q8, const int *tokid, const unsigned long long *wp_tab,
  const unsigned long long *ws_tab, const unsigned long long *wr_tab,
  const int *block_expert, const float *ascale, const int *azp, void *Y,
  unsigned int n_mblocks, unsigned int N, unsigned int K, int out_fp16,
  const int *wl_n) {
  if (q8 == nullptr || wp_tab == nullptr || ws_tab == nullptr ||
      wr_tab == nullptr || block_expert == nullptr || Y == nullptr ||
      n_mblocks == 0 || (N & 127u) != 0u || (K & 255u) != 0u)
    return false;
  auto kg = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "imma_moe_g4");
  if (!kg)
    return false;
  static bool attr_once = []() {
    const char *e = std::getenv("NNTR_MOE_G_DBG");
    return e != nullptr && e[0] != '0';
  }();
  if (attr_once) {
    attr_once = false;
    int nregs = 0, lmem = 0;
    cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, kg->GetFunction());
    cuFuncGetAttribute(&lmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                       kg->GetFunction());
    fprintf(stderr, "[g4_attr] regs=%d local=%dB\n", nregs, lmem);
  }
  const int n = (int)N, k = (int)K;
  kg->SetKernelArguments(0, &q8, sizeof(q8));
  kg->SetKernelArguments(1, &tokid, sizeof(tokid));
  kg->SetKernelArguments(2, &wp_tab, sizeof(wp_tab));
  kg->SetKernelArguments(3, &ws_tab, sizeof(ws_tab));
  kg->SetKernelArguments(4, &wr_tab, sizeof(wr_tab));
  kg->SetKernelArguments(5, &block_expert, sizeof(block_expert));
  kg->SetKernelArguments(6, &ascale, sizeof(ascale));
  kg->SetKernelArguments(7, &azp, sizeof(azp));
  kg->SetKernelArguments(8, &Y, sizeof(Y));
  kg->SetKernelArguments(9, &n, sizeof(n));
  kg->SetKernelArguments(10, &k, sizeof(k));
  kg->SetKernelArguments(11, &out_fp16, sizeof(out_fp16));
  kg->SetKernelArguments(12, &wl_n, sizeof(wl_n));
  const int ib[3] = {256, 1, 1};
  const int ig[3] = {(int)(N / 128u), (int)n_mblocks, 1};
  return StreamManager::Global().DispatchCommand(*kg, ig, ib);
}

// One launch per projection: in-place fragment repack of ALL E expert
// payloads through the (mapped) pointer table. See the kernel note; the
// per-expert scratch/memcpy version measured 9.3 s of first-prefill wall.
bool cuda_fc_qs4cx_moe_repack_g3(const unsigned long long *wp_tab,
                                 unsigned int E, unsigned int N,
                                 unsigned int K) {
  const size_t bytes = (size_t)N * (K >> 1);
  if (wp_tab == nullptr || E == 0 || (K & 127u) != 0u || bytes == 0 ||
      (bytes & 15u) != 0u)
    return false;
  auto k = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                    "moe_repack_frag_g3");
  if (!k)
    return false;
  const int e = (int)E;
  const long nslots = (long)(bytes >> 4);
  k->SetKernelArguments(0, &wp_tab, sizeof(void *));
  k->SetKernelArguments(1, &e, sizeof(int));
  k->SetKernelArguments(2, &nslots, sizeof(long));
  const int b[3] = {256, 1, 1};
  const int gx = (int)((nslots + 255) / 256) < 128
                   ? (int)((nslots + 255) / 256)
                   : 128;
  const int g[3] = {gx, (int)E, 1};
  return StreamManager::Global().DispatchCommand(*k, g, b);
}

bool cuda_fc_qs4cx_moe_rowsum_g3(const unsigned long long *wp_tab,
                                 unsigned int E, unsigned int N,
                                 unsigned int K, int *rs) {
  if (wp_tab == nullptr || rs == nullptr || E == 0)
    return false;
  auto k = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                    "moe_rowsum_g3");
  if (!k)
    return false;
  const int e = (int)E, n = (int)N, kh = (int)(K >> 1);
  k->SetKernelArguments(0, &wp_tab, sizeof(void *));
  k->SetKernelArguments(1, &rs, sizeof(void *));
  k->SetKernelArguments(2, &e, sizeof(int));
  k->SetKernelArguments(3, &n, sizeof(int));
  k->SetKernelArguments(4, &kh, sizeof(int));
  const int b[3] = {256, 1, 1}; // 8 warps = 8 rows per block
  const int g[3] = {(int)((N + 7u) / 8u), (int)E, 1};
  return StreamManager::Global().DispatchCommand(*k, g, b);
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
  const bool okv = cuda_vec4_rows_ok(width, x, y);
  const bool vec4 = okv && cuda_vec4_rows_small(rows);
  const bool v4p = okv && !vec4 && norm_v4p_on();
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // Sizing the quant scratch is a cudaMalloc, which is illegal mid-capture --
  // ensure_buf refuses there and we hand the row back to the plain norm. In
  // practice prefill has already grown the scratch past a single decode row by
  // the time the decode graph is captured, so this is a cold-start guard, not
  // the steady state.
  if (!dp4a_stage_scratch(rows, width))
    return false;
  // Pending residual-add fusion: if the plane we are about to norm is a
  // deferred add's output, pass 1 performs the add itself (bit-identical --
  // see add_rmsnorm_quant_i8_h_v4p). Taken only AFTER the early-return
  // guards above so a false return cannot orphan the record. On any
  // non-eligible corner the add is simply re-deferred: the dispatch below
  // flushes it ahead of the norm kernel, restoring the split flow.
  const unsigned short *fa = nullptr, *fb = nullptr;
  bool fuse =
    (vec4 || v4p) &&
    cuda_add_pending_take(x, (unsigned long long)rows * width, &fa, &fb);
  if (fuse && !cuda_vec4_rows_ok(width, fa, fb)) {
    cuda_add_fp16(fa, fb, const_cast<unsigned short *>(x), rows * width);
    fuse = false;
  }
  auto k = CudaContext::Global().registerCudaKernel(
    FC_QINT4_DP4A_SRC, fuse   ? (vec4 ? "add_rmsnorm_quant_i8_h_v4"
                                      : "add_rmsnorm_quant_i8_h_v4p")
                       : vec4 ? "rmsnorm_quant_i8_h_v4"
                       : v4p  ? "rmsnorm_quant_i8_h_v4p"
                              : "rmsnorm_quant_i8_h");
  if (!k) {
    if (fuse)
      cuda_add_fp16(fa, fb, const_cast<unsigned short *>(x), rows * width);
    return false;
  }
  int m = (int)rows, kk = (int)width;
  int has_gamma = (gamma == nullptr) ? 0
                  : (!(vec4 || v4p) || cuda_vec4_rows_ok(4, gamma)) ? 1
                                                                    : 2;
  int ai = 0;
  if (fuse) {
    k->SetKernelArguments(ai++, &fa, sizeof(fa));
    k->SetKernelArguments(ai++, &fb, sizeof(fb));
    k->SetKernelArguments(ai++, &gamma, sizeof(gamma));
    k->SetKernelArguments(ai++, &x, sizeof(x)); // r: the residual plane
  } else {
    k->SetKernelArguments(ai++, &x, sizeof(x));
    k->SetKernelArguments(ai++, &gamma, sizeof(gamma));
  }
  k->SetKernelArguments(ai++, &y, sizeof(y));
  k->SetKernelArguments(ai++, &g_dp4a_q8, sizeof(g_dp4a_q8));
  k->SetKernelArguments(ai++, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  k->SetKernelArguments(ai++, &g_dp4a_azp, sizeof(g_dp4a_azp));
  k->SetKernelArguments(ai++, &m, sizeof(m));
  k->SetKernelArguments(ai++, &kk, sizeof(kk));
  k->SetKernelArguments(ai++, &eps, sizeof(eps));
  k->SetKernelArguments(ai++, &has_gamma, sizeof(has_gamma));
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
  const bool q_okv = fused_normq_on() && cuda_vec4_rows_ok(K, Xh);
  const bool q_vec4 = q_okv && cuda_vec4_rows_small(M);
  auto kqh = CudaContext::Global().registerCudaKernel(
    FC_QINT4_DP4A_SRC, q_vec4  ? "act_quant_i8_h_v4"
                       : q_okv ? "act_quant_i8_h_v4p"
                               : "act_quant_i8_h");
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

// fp16 activation in, FP32 out: the GDN projection variant. The gated
// delta net consumes its projections in fp32 (conv/scan are fp32 end to
// end), so when in_proj_qkv is QS4CX this entry runs the same act-quant +
// w4a8 ladder as the fp16 entry but writes the float Y directly
// (out_fp16=0) -- no extra convert kernel, no fp16 round-trip on a tensor
// the fp32 reference path never rounded.
bool cuda_fc_qs4cx_dp4a_gemm_fp16in_f32out(const unsigned short *Xh,
                                           const unsigned char *plain_w,
                                           const unsigned short *scales_fp16,
                                           float *Yf, unsigned int M,
                                           unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  const bool q_okv = fused_normq_on() && cuda_vec4_rows_ok(K, Xh);
  const bool q_vec4 = q_okv && cuda_vec4_rows_small(M);
  auto kqh = CudaContext::Global().registerCudaKernel(
    FC_QINT4_DP4A_SRC, q_vec4  ? "act_quant_i8_h_v4"
                       : q_okv ? "act_quant_i8_h_v4p"
                               : "act_quant_i8_h");
  if (!kqh) {
    ml_loge("[CUDA] fc_qint4 dp4a f32out: kernel registration failed");
    return false;
  }
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  if (!dp4a_stage_scratch(M, K))
    return false;
  int m = (int)M, k = (int)K;
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
  if (!dp4a_repack_and_gemm(plain_w, scales_fp16, Yf, M, N, K,
                            /** out_fp16= */ 0))
    return false;
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
  const bool q_okv = fused_normq_on() && cuda_vec4_rows_ok(K, Xh);
  const bool q_vec4 = q_okv && cuda_vec4_rows_small(M);
  auto kqh = CudaContext::Global().registerCudaKernel(
    FC_QINT4_DP4A_SRC, q_vec4  ? "act_quant_i8_h_v4"
                       : q_okv ? "act_quant_i8_h_v4p"
                               : "act_quant_i8_h");
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

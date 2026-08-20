// SPDX-License-Identifier: Apache-2.0
/**
 * @file    cuda_moe.cpp
 * @brief   Grouped MoE expert FFN on the device (see header).
 */
#include "cuda_moe.h"

#include <cuda_common.h> // quant_stage_survive
#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_fc_dense.h>  // cuda_fc_dense_gemm_fp16_f32out_acc (router h2)
#include <cuda_fc_qint4.h> // cuda_fc_qs4cx_moe_grouped_gemm (imma tile)
#include <map>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#include <cstdio>
#include <mutex>
#include <nntrainer_log.h>
#include <vector>

namespace nntrainer::cuda {

// Hardware fp16 conversion (cuda_fc_qint4.cpp:253 / cuda_rmsnorm.cpp:132), not
// the software bit-twiddling variants: those hold a data-dependent `while` for
// denormals, which stops the compiler unrolling any loop that calls them --
// the defect that made the GDN GEMVs 68% of GPU time.
static const char *MOE_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float moe_h2f(unsigned short h){
  float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ unsigned short moe_f2h(float f){
  unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}

// dst[i,:] = src[rows[i],:]
__global__ void moe_gather_h(const unsigned short *src, unsigned short *dst,
                             const int *rows, int m, int width){
  const long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= (long)m*width) return;
  const int i = (int)(idx / width), w = (int)(idx - (long)i*width);
  dst[idx] = src[(long)rows[i]*width + w];
}

// out = silu(gate)*up. fp32 math, fp16 storage, matching the host path's
// acti_func.run_fn(gate) followed by multiply_i(up).
__global__ void moe_swiglu_h(const unsigned short *gate,
                             const unsigned short *up,
                             unsigned short *out, int n){
  const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = moe_h2f(gate[i]);
  out[i] = moe_f2h((g / (1.0f + expf(-g))) * moe_h2f(up[i]));
}

// Per-row asymmetric int8 activation quant. Byte-for-byte the convention of
// act_quant_i8_h / asym_qparams in cuda_fc_qint4.cpp -- range forced to
// include zero, 255 levels, nudged zero point -- so the grouped path dequants
// identically to the per-expert one it replaces.
__device__ __forceinline__ void moe_asym(float fmn, float fmx, float &scale_q,
                                         float &recip, int &zp){
  float rmin = fminf(0.f, fmn), rmax = fmaxf(0.f, fmx);
  float range = rmax - rmin;
  scale_q = range > 0.f ? 255.f / range : 1.f;
  recip   = range > 0.f ? range / 255.f : 1.f;
  float dmin = rmin*scale_q, dmax = rmax*scale_q;
  float zp_lo = -128.f - dmin, zp_hi = 127.f - dmax;
  float zp_f = ((-128.f + dmin) + (127.f + dmax) > 0.f) ? zp_lo : zp_hi;
  zp_f = fmaxf(-128.f, fminf(127.f, zp_f));
  zp = (int)rintf(zp_f);
}
__global__ void moe_actq_h(const unsigned short *Xh, signed char *q8,
                           float *ascale, int *azp, int M, int K){
  int m = blockIdx.x;
  if (m >= M) return;
  __shared__ float smn[256];
  __shared__ float smx[256];
  const unsigned short *xr = Xh + (long)m*K;
  float lmn = 0.f, lmx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = moe_h2f(xr[k]);
    lmn = fminf(lmn, v); lmx = fmaxf(lmx, v);
  }
  smn[threadIdx.x] = lmn; smx[threadIdx.x] = lmx;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x+s]);
      smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x+s]);
    }
    __syncthreads();
  }
  float scale_q, recip; int zp;
  moe_asym(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) { ascale[m] = recip; azp[m] = zp; }
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = (int)rintf(moe_h2f(xr[k]) * scale_q) + zp;
    q = max(-128, min(127, q));
    q8[(long)m*K + k] = (signed char)q;
  }
}
// Vectorized twin of moe_actq_h (uint2 loads, warp-shuffle reduction, packed
// int stores, quant pass re-reads the L1-hot row -- the act_quant_i8_h_v4p
// recipe). BIT-IDENTICAL: min/max are order-independent, the conversions
// produce the same values, and the rint/clamp is unchanged. Requires
// K % 4 == 0 and 8-byte-aligned rows (the launcher checks).
__global__ void moe_actq_h_v4p(const unsigned short *Xh, signed char *q8,
                               float *ascale, int *azp, int M, int K){
  int m = blockIdx.x;
  if (m >= M) return;
  const uint2 *xv = (const uint2 *)(Xh + (long)m*K);
  int *q32 = (int *)(q8 + (long)m*K);
  const int nv = K >> 2;
  __shared__ float smn[32];
  __shared__ float smx[32];
  float lmn = 0.f, lmx = 0.f;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    const uint2 r = xv[i];
    const float f0 = moe_h2f((unsigned short)(r.x & 0xFFFFu));
    const float f1 = moe_h2f((unsigned short)(r.x >> 16));
    const float f2 = moe_h2f((unsigned short)(r.y & 0xFFFFu));
    const float f3 = moe_h2f((unsigned short)(r.y >> 16));
    lmn = fminf(lmn, fminf(fminf(f0, f1), fminf(f2, f3)));
    lmx = fmaxf(lmx, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
  }
  for (int o = 16; o > 0; o >>= 1) {
    lmn = fminf(lmn, __shfl_down_sync(0xffffffffu, lmn, o));
    lmx = fmaxf(lmx, __shfl_down_sync(0xffffffffu, lmx, o));
  }
  if ((threadIdx.x & 31) == 0) {
    smn[threadIdx.x >> 5] = lmn;
    smx[threadIdx.x >> 5] = lmx;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    const int nw = blockDim.x >> 5;
    float a = (threadIdx.x < nw) ? smn[threadIdx.x] : __int_as_float(0x7F800000);
    float b = (threadIdx.x < nw) ? smx[threadIdx.x] : __int_as_float((int)0xFF800000);
    for (int o = 16; o > 0; o >>= 1) {
      a = fminf(a, __shfl_down_sync(0xffffffffu, a, o));
      b = fmaxf(b, __shfl_down_sync(0xffffffffu, b, o));
    }
    if (threadIdx.x == 0) { smn[0] = a; smx[0] = b; }
  }
  __syncthreads();
  float scale_q, recip; int zp;
  moe_asym(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) { ascale[m] = recip; azp[m] = zp; }
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    const uint2 r = xv[i];
    const int q0 = max(-128, min(127, (int)rintf(moe_h2f((unsigned short)(r.x & 0xFFFFu)) * scale_q) + zp));
    const int q1 = max(-128, min(127, (int)rintf(moe_h2f((unsigned short)(r.x >> 16)) * scale_q) + zp));
    const int q2 = max(-128, min(127, (int)rintf(moe_h2f((unsigned short)(r.y & 0xFFFFu)) * scale_q) + zp));
    const int q3 = max(-128, min(127, (int)rintf(moe_h2f((unsigned short)(r.y >> 16)) * scale_q) + zp));
    q32[i] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) |
             ((q3 & 0xFF) << 24);
  }
}

// GROUPED dp4a GEMM. Same 64x64 register-blocked tile as dp4a_gemm_reg -- that
// shape is already right, and was measured 1.4x FASTER than a 16x16 tile with
// 12x the block count, because it stages coalesced tiles into shared memory and
// yields 16 outputs per thread. What it lacked was work per launch: one expert
// at M~42, N=512 gives ceil(N/64)*ceil(M/64) = EIGHT blocks on a 16-SM part.
//
// Here blockIdx.y indexes a WORK LIST of (expert, row range) instead of a row
// tile of one matrix, so a single launch covers every routed expert: 2,048
// blocks at prefill and 64 at decode. The weight pointer is read from a device
// array indexed by that expert -- the same discipline as d_pos, and the
// precondition for ever capturing this in a CUDA graph.
//
// Reads the QS4CX payload directly: the offset-binary bias comes off with the
// XOR on the staging load and the per-channel rowsum is accumulated off the
// tile already in registers, so no DevWeightQ copy is needed (int32 sums over
// exactly k in [0,K) -> bit-identical to the cached path).
#define MG_BM 64
#define MG_BN 64
#define MG_BK 32
#define MG_TM 4
#define MG_TN 4
__global__ void moe_gemm_grouped(const signed char *q8, const float *ascale,
                                 const int *azp,
                                 const unsigned char * const *wptr,
                                 const unsigned short * const *wsc,
                                 const int *wl_e, const int *wl_r0,
                                 const int *wl_n, unsigned short *Y,
                                 int N, int K){
  __shared__ signed char As[MG_BM][MG_BK];
  __shared__ signed char Ws[MG_BN][MG_BK];
  const int wid = blockIdx.y;
  const int ex = wl_e[wid], r0 = wl_r0[wid], nrow = wl_n[wid];
  const unsigned char *plain = wptr[ex];
  const unsigned short *wscale = wsc[ex];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty*16 + tx;
  const int blockN = blockIdx.x * MG_BN;
  const int Kh = (K + 1) >> 1;
  int acc[MG_TM][MG_TN], rs[MG_TN];
#pragma unroll
  for (int j = 0; j < MG_TN; j++) rs[j] = 0;
#pragma unroll
  for (int i = 0; i < MG_TM; i++)
#pragma unroll
    for (int j = 0; j < MG_TN; j++) acc[i][j] = 0;

  for (int k0 = 0; k0 < K; k0 += MG_BK) {
    for (int q = tid; q < MG_BM*MG_BK; q += 256) {
      int i = q / MG_BK, j = q % MG_BK;
      int kk = k0 + j;
      As[i][j] = (i < nrow && kk < K)
                   ? q8[(long)(r0 + i)*K + kk] : (signed char)0;
    }
    for (int q = tid; q < MG_BN*MG_BK; q += 256) {
      int i = q / MG_BK, j = q % MG_BK;
      int nn = blockN + i, kk = k0 + j;
      signed char wv = 0;
      if (nn < N && kk < K) {
        unsigned char b =
          (unsigned char)(((unsigned int)plain[(long)nn*Kh + (kk>>1)]) ^ 0x88u);
        wv = (kk & 1) ? (((signed char)b) >> 4) : (((signed char)(b << 4)) >> 4);
      }
      Ws[i][j] = wv;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < MG_BK; kk += 4) {
      int af[MG_TM], wf[MG_TN];
#pragma unroll
      for (int i = 0; i < MG_TM; i++) af[i] = *(const int*)&As[ty*MG_TM+i][kk];
#pragma unroll
      for (int j = 0; j < MG_TN; j++) wf[j] = *(const int*)&Ws[tx*MG_TN+j][kk];
#pragma unroll
      for (int j = 0; j < MG_TN; j++) rs[j] = __dp4a(0x01010101, wf[j], rs[j]);
#pragma unroll
      for (int i = 0; i < MG_TM; i++)
#pragma unroll
        for (int j = 0; j < MG_TN; j++)
          acc[i][j] = __dp4a(af[i], wf[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < MG_TM; i++) {
    int lr = ty*MG_TM + i;
    if (lr >= nrow) continue;
    const int arow = r0 + lr;
    const float as = ascale[arow];
    const int zp = azp[arow];
#pragma unroll
    for (int j = 0; j < MG_TN; j++) {
      int col = blockN + tx*MG_TN + j;
      if (col < N)
        Y[(long)arow*N + col] =
          moe_f2h((float)(acc[i][j] - zp*rs[j]) * as * moe_h2f(wscale[col]));
    }
  }
}

// Per-expert scatter (the shipping path). Race-free without atomics only
// because one expert never sees the same token twice.
__global__ void moe_scatter_add_h(const unsigned short *src,
                                  unsigned short *dst, const int *rows,
                                  const float *wts, int m, int width){
  const long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= (long)m*width) return;
  const int i = (int)(idx / width), w = (int)(idx - (long)i*width);
  const long d = (long)rows[i]*width + w;
  dst[d] = moe_f2h(moe_h2f(dst[d]) + moe_h2f(src[idx]) * wts[i]);
}

// Token-major combine. One thread per OUTPUT element, summing that token's own
// top-k contributions in slot order.
//
// Not a scatter with atomics: every token is written by topk different
// assignments, so an assignment-major scatter races, and an fp16 atomicAdd
// would also make the accumulation order non-deterministic -- which is exactly
// the class of defect this backend was just debugged out of. Inverting it also
// improves accuracy: the host path rounded to fp16 after EVERY expert, this
// keeps the whole sum in fp32 and rounds once.
__global__ void moe_combine_h(const unsigned short *Y, unsigned short *out,
                              const int *slots, const float *wts,
                              int T, int topk, int width){
  const long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= (long)T*width) return;
  const int t = (int)(idx / width), w = (int)(idx - (long)t*width);
  float acc = 0.0f;
  for (int k = 0; k < topk; ++k) {
    const int a = slots[t*topk + k];
    if (a >= 0) acc += moe_h2f(Y[(long)a*width + w]) * wts[a];
  }
  out[idx] = moe_f2h(acc);
}
// ------------------------------------------------------------- ROUTING ----
// logits[t,e] = sum_h X[t,h] * Wg[h,e], fp16 activation widened to fp32,
// fp32 weight, fp32 accumulate.
//
// This is the single largest item in a 35B prefill profile and it was on the
// HOST: `input.clone(FP32)` (a 33 MB materialisation per layer per chunk)
// followed by an OpenBLAS sgemm of [4096,2048]x[2048,256] = 2.1 GMAC. The
// whole host routing block measured 13,637 ms of a 35 s prefill -- 39% of it,
// and the leading explanation for the GPU sitting ~41% idle.
//
// Deliberately NOT fp16 tensor cores. Rounding the GATE weight to fp16 would
// flip the top-k selection on near-ties, and the top-k choice is discrete: a
// flipped expert is not a small numeric error, it is a different model. The
// host widens fp16 -> fp32 and accumulates in fp32, so widening per element
// here reproduces its arithmetic up to summation order.
//
// 64x64 tile, BK=16, 256 threads, 4x4 per thread. Shared rows are padded by 1
// float so the 4-wide column reads do not all land in one bank.
// [rgemm-h2] One-time split of the fp32 gate weight into hi=(half)w and
// lo=(half)(w - (float)hi) planes: X*hi + X*lo with fp32 accumulate carries
// ~22-24 mantissa bits of the fp32 product -- the dtype swap plain-fp16
// cannot make (top-k is discrete; a flipped near-tie is a different model).
__global__ void moe_rsplit(const float *w, unsigned short *hi,
                           unsigned short *lo, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float f = w[i];
  unsigned short h;
  float fh;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  asm("cvt.f32.f16 %0, %1;" : "=f"(fh) : "h"(h));
  unsigned short l;
  float r = f - fh;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(l) : "f"(r));
  hi[i] = h;
  lo[i] = l;
}
// Transposed split for the decode GEMV: w is [H,E] h-major (the dense-gemm
// B layout); hiT/loT come out [E,H] e-major so a warp streams its expert's
// row with coalesced vec4 loads. Same split arithmetic as moe_rsplit.
__global__ void moe_rsplit_t(const float *w, unsigned short *hiT,
                             unsigned short *loT, int H, int E) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= H * E) return;
  const int h = i / E, e = i - h * E;
  float f = w[i];
  unsigned short hh;
  float fh;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hh) : "f"(f));
  asm("cvt.f32.f16 %0, %1;" : "=f"(fh) : "h"(hh));
  unsigned short ll;
  float r = f - fh;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(ll) : "f"(r));
  hiT[(long)e * H + h] = hh;
  loT[(long)e * H + h] = ll;
}
// M=1 decode router on the transposed split planes: warp per expert, BOTH
// planes in one pass (one launch, ~2 MB device reads), each plane dot
// accumulated in its own fp32 register then summed once at the end -- the
// dual-pass hi+lo structure. Replaces the fp32 SIMT tile whose 64x64 M-tile
// is 63/64 idle at M=1 (measured 240 us/call on decode).
__global__ void moe_router_gemv_h2(const unsigned short *X,
                                   const unsigned short *hiT,
                                   const unsigned short *loT, float *L,
                                   int H, int E) {
  const int e = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (e >= E) return;
  const int lane = threadIdx.x & 31;
  const uint2 *xv = (const uint2 *)X;
  const uint2 *hv = (const uint2 *)(hiT + (long)e * H);
  const uint2 *lv = (const uint2 *)(loT + (long)e * H);
  const int nv = H >> 2;
  float ah = 0.f, al = 0.f;
  for (int i = lane; i < nv; i += 32) {
    const uint2 xr = xv[i];
    const uint2 hr = hv[i];
    const uint2 lr = lv[i];
    const float x0 = moe_h2f((unsigned short)(xr.x & 0xFFFFu));
    const float x1 = moe_h2f((unsigned short)(xr.x >> 16));
    const float x2 = moe_h2f((unsigned short)(xr.y & 0xFFFFu));
    const float x3 = moe_h2f((unsigned short)(xr.y >> 16));
    ah += x0 * moe_h2f((unsigned short)(hr.x & 0xFFFFu)) +
          x1 * moe_h2f((unsigned short)(hr.x >> 16)) +
          x2 * moe_h2f((unsigned short)(hr.y & 0xFFFFu)) +
          x3 * moe_h2f((unsigned short)(hr.y >> 16));
    al += x0 * moe_h2f((unsigned short)(lr.x & 0xFFFFu)) +
          x1 * moe_h2f((unsigned short)(lr.x >> 16)) +
          x2 * moe_h2f((unsigned short)(lr.y & 0xFFFFu)) +
          x3 * moe_h2f((unsigned short)(lr.y >> 16));
  }
  for (int o = 16; o > 0; o >>= 1) {
    ah += __shfl_down_sync(0xffffffffu, ah, o);
    al += __shfl_down_sync(0xffffffffu, al, o);
  }
  if (lane == 0)
    L[e] = ah + al;
}
__global__ void moe_router_gemm(const unsigned short *X, const float *Wg,
                                float *L, int T, int H, int E){
  // 68-float row stride: 16B-aligned rows so the inner-loop fragment loads
  // are two lds.128 instead of eight lds.32. The fma ORDER is untouched, so
  // the fp32 logits match the previous tile bit-for-bit given the same
  // input (2.28 -> 2.01 ms measured; NOT load-bound beyond that -- see the
  // calibration bank). NOTE the top-k census still wobbles RUN TO RUN by
  // layer ~2 -- that is upstream nondeterminism (present on the old tile
  // too, same binary back-to-back), not this kernel's doing.
  struct alignas(16) RV4 { float x, y, z, w; };
  __shared__ float As[16][68];   // [k][m]
  __shared__ float Bs[16][68];   // [k][n]
  const int tid = threadIdx.x;                 // 256
  const int tm = (tid >> 4) * 4, tn = (tid & 15) * 4;
  const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  for (int k0 = 0; k0 < H; k0 += 16) {
    // 64x16 of A and 16x64 of B, 4 elements per thread each.
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int idx = tid + r * 256;           // 0..1023
      const int mm = idx >> 4, kk = idx & 15;  // A: 64 rows x 16 k
      const int m = m0 + mm, k = k0 + kk;
      As[kk][mm] = (m < T && k < H) ? moe_h2f(X[(long)m * H + k]) : 0.f;
      const int kb = idx >> 6, nb = idx & 63;  // B: 16 k x 64 cols
      const int k2 = k0 + kb, n = n0 + nb;
      Bs[kb][nb] = (k2 < H && n < E) ? Wg[(long)k2 * E + n] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
      const RV4 av = *(const RV4 *)&As[kk][tm];
      const RV4 bv = *(const RV4 *)&Bs[kk][tn];
      const float a[4] = {av.x, av.y, av.z, av.w};
      const float b[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int m = m0 + tm + i;
    if (m >= T) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int n = n0 + tn + j;
      if (n < E) L[(long)m * E + n] = acc[i][j];
    }
  }
}
// --------------------------------------------------- ROUTING: topk + bucket
// One block per token: softmax over the E router logits, take the top-k,
// renormalise, and histogram the picks. Mirrors the host order exactly --
// softmax, THEN topK, THEN divide by the sum of the k kept values -- because
// dividing before the selection would not be the same number.
//
// E <= 1024 (256 here) so one block covers a row and the reductions are
// block-local. The top-k is k passes of argmax-then-mask: with k=8 over 256
// that is 8 tree reductions, far cheaper than a sort.
__global__ void moe_route_topk(const float *logits, int *tk_idx, float *tk_wt,
                               int *counts, int T, int E, int K){
  extern __shared__ float sh[];             // E floats
  __shared__ float red[32];
  __shared__ int   redi[32];
  const int t = blockIdx.x;
  const int tid = threadIdx.x, nt = blockDim.x;
  if (t >= T) return;
  const float *row = logits + (long)t * E;
  for (int e = tid; e < E; e += nt) sh[e] = row[e];
  __syncthreads();

  // softmax
  float m = -1e30f;
  for (int e = tid; e < E; e += nt) m = fmaxf(m, sh[e]);
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
  if ((tid & 31) == 0) red[tid >> 5] = m;
  __syncthreads();
  if (tid == 0) { float v = red[0];
    for (int i = 1; i < (nt + 31) / 32; ++i) v = fmaxf(v, red[i]);
    red[0] = v; }
  __syncthreads();
  m = red[0];
  float s = 0.f;
  for (int e = tid; e < E; e += nt) { float x = __expf(sh[e] - m); sh[e] = x; s += x; }
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  if ((tid & 31) == 0) red[tid >> 5] = s;
  __syncthreads();
  if (tid == 0) { float v = 0.f;
    for (int i = 0; i < (nt + 31) / 32; ++i) v += red[i];
    red[0] = v; }
  __syncthreads();
  const float inv = 1.0f / red[0];
  for (int e = tid; e < E; e += nt) sh[e] *= inv;
  __syncthreads();

  // top-k. Fast path (E == nt, K <= 16): each warp takes a warp-local top-K
  // over its own 32 experts with shfl-only masked argmax passes (no block
  // barriers), then thread 0 merges the 8 descending lists. Tie-break is
  // IDENTICAL to the serial 8-pass argmax below: strict >, so the lower lane
  // (= lower expert) wins inside a warp and the lower warp (= lower expert
  // range) wins in the merge. Same selected experts, same order, same bits.
  if (E == nt && nt <= 256 && K <= 16) {
    __shared__ float wlv[8][16];
    __shared__ int wli[8][16];
    const int lane = tid & 31, w = tid >> 5;
    float cv = sh[tid];
    const int ci = tid;
    for (int j = 0; j < K; ++j) {
      float bv = cv; int bi = ci;
      for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, o);
        int   oi = __shfl_down_sync(0xffffffffu, bi, o);
        if (ov > bv) { bv = ov; bi = oi; }
      }
      bv = __shfl_sync(0xffffffffu, bv, 0);
      bi = __shfl_sync(0xffffffffu, bi, 0);
      if (lane == 0) { wlv[w][j] = bv; wli[w][j] = bi; }
      if (ci == bi) cv = -1.f; // winner self-masks for the next pass
    }
    __syncthreads();
    if (tid == 0) {
      const int nw = nt >> 5;
      int hp[8];
      for (int i = 0; i < 8; ++i) hp[i] = 0;
      float wsum = 0.f;
      for (int j = 0; j < K; ++j) {
        float bv = -1.f; int bw = 0;
        for (int i = 0; i < nw; ++i)
          if (hp[i] < K && wlv[i][hp[i]] > bv) { bv = wlv[i][hp[i]]; bw = i; }
        tk_idx[(long)t * K + j] = wli[bw][hp[bw]];
        tk_wt[(long)t * K + j] = bv;
        ++hp[bw];
        wsum += bv;
      }
      const float n = (wsum > 0.f) ? (1.0f / wsum) : 1.0f;
      for (int j = 0; j < K; ++j) {
        tk_wt[(long)t * K + j] *= n;
        atomicAdd(&counts[tk_idx[(long)t * K + j]], 1);
      }
    }
    return;
  }
  // Generic path: top-k by repeated argmax; the winner is masked to -1 so it
  // cannot repeat.
  float wsum = 0.f;
  for (int j = 0; j < K; ++j) {
    float bv = -1.f; int bi = -1;
    for (int e = tid; e < E; e += nt)
      if (sh[e] > bv) { bv = sh[e]; bi = e; }
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_down_sync(0xffffffffu, bv, o);
      int   oi = __shfl_down_sync(0xffffffffu, bi, o);
      if (ov > bv) { bv = ov; bi = oi; }
    }
    if ((tid & 31) == 0) { red[tid >> 5] = bv; redi[tid >> 5] = bi; }
    __syncthreads();
    if (tid == 0) {
      float v = red[0]; int i2 = redi[0];
      for (int i = 1; i < (nt + 31) / 32; ++i)
        if (red[i] > v) { v = red[i]; i2 = redi[i]; }
      red[0] = v; redi[0] = i2;
      tk_idx[(long)t * K + j] = i2;
      tk_wt[(long)t * K + j] = v;
      sh[i2] = -1.f;                    // mask so the next pass skips it
    }
    __syncthreads();
    wsum += red[0];
  }
  // norm_topk_prob, then the per-expert histogram
  if (tid == 0) {
    const float n = (wsum > 0.f) ? (1.0f / wsum) : 1.0f;
    for (int j = 0; j < K; ++j) {
      tk_wt[(long)t * K + j] *= n;
      atomicAdd(&counts[tk_idx[(long)t * K + j]], 1);
    }
  }
}
// Warp-per-token routing (E==256, K<=8): one warp owns a token -- 8 experts
// per lane as float4 pairs, shuffle-only reductions, no shared memory and no
// block barriers (the block form pays 6 __syncthreads rounds per token).
// SELECTION IS IDENTICAL to moe_route_topk: the row max is order-independent
// (same m), exp inputs are per-element identical, comparisons scan lane-local
// experts in ascending order with strict > and the shuffle merge keeps the
// lower lane on ties -- the global winner is the lowest maximal expert,
// exactly the serial tie-break. Only the softmax DENOMINATOR's reduction
// order changes, so tk_wt can move by an ulp while tk_idx and counts (and
// therefore the whole routing/work-list build) are bit-identical.
__global__ void moe_route_topk_w(const float *logits, int *tk_idx,
                                 float *tk_wt, int *counts, int T, int E,
                                 int K){
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t = blockIdx.x * (blockDim.x >> 5) + w;
  if (t >= T) return;
  const float4 *rv = (const float4 *)(logits + (long)t * E);
  float p[8];
  {
    const float4 a = rv[lane * 2];
    const float4 b = rv[lane * 2 + 1];
    p[0] = a.x; p[1] = a.y; p[2] = a.z; p[3] = a.w;
    p[4] = b.x; p[5] = b.y; p[6] = b.z; p[7] = b.w;
  }
  float m = p[0];
  #pragma unroll
  for (int j = 1; j < 8; ++j) m = fmaxf(m, p[j]);
  for (int o = 16; o > 0; o >>= 1)
    m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
  m = __shfl_sync(0xffffffffu, m, 0);
  float s = 0.f;
  #pragma unroll
  for (int j = 0; j < 8; ++j) { p[j] = __expf(p[j] - m); s += p[j]; }
  for (int o = 16; o > 0; o >>= 1)
    s += __shfl_down_sync(0xffffffffu, s, o);
  s = __shfl_sync(0xffffffffu, s, 0);
  const float inv = 1.0f / s;
  float wsum = 0.f;
  for (int j = 0; j < K; ++j) {
    float bv = -1.f; int bi = -1;
    #pragma unroll
    for (int q = 0; q < 8; ++q)
      if (p[q] > bv) { bv = p[q]; bi = (lane << 3) + q; }
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, bv, o);
      const int oi = __shfl_down_sync(0xffffffffu, bi, o);
      if (ov > bv) { bv = ov; bi = oi; }
    }
    bv = __shfl_sync(0xffffffffu, bv, 0);
    bi = __shfl_sync(0xffffffffu, bi, 0);
    if ((bi >> 3) == lane) p[bi & 7] = -1.f; // winner self-masks
    if (lane == 0) {
      const float nw = bv * inv;
      tk_idx[(long)t * K + j] = bi;
      tk_wt[(long)t * K + j] = nw;
      wsum += nw;
    }
  }
  if (lane == 0) {
    const float n = (wsum > 0.f) ? (1.0f / wsum) : 1.0f;
    for (int j = 0; j < K; ++j) {
      tk_wt[(long)t * K + j] *= n;
      atomicAdd(&counts[tk_idx[(long)t * K + j]], 1);
    }
  }
}

// exclusive scan of counts[E] -> offs[E], and seed the scatter cursors.
// One block; E is 256, so a serial scan in thread 0 is a few hundred ns.
__global__ void moe_route_scan(const int *counts, int *offs, int *cursor, int E){
  if (threadIdx.x != 0) return;
  int acc = 0;
  for (int e = 0; e < E; ++e) { offs[e] = acc; cursor[e] = acc; acc += counts[e]; }
}

// Scatter every (token, k) assignment into the expert-major rows/wts arrays.
//
// The slot comes from an atomicAdd, so the ORDER inside one expert varies from
// run to run -- and that is provably harmless here, which is why the cheap
// form is the right one: a token appears at most once per expert, so the
// downstream scatter-add writes each dst row exactly once per expert, and the
// FC treats rows independently. Different order, identical bits.
__global__ void moe_route_bucket(const int *tk_idx, const float *tk_wt,
                                 int *cursor, int *rows, float *wts,
                                 int T, int K){
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)T * K) return;
  const int e = tk_idx[i];
  const int slot = atomicAdd(&cursor[e], 1);
  rows[slot] = (int)(i / K);
  wts[slot] = tk_wt[i];
}

// ---- grouped (padded work-list) routing tail --------------------------------
// Sort each token's K (expert, weight) pairs by ASCENDING EXPERT ID, after the
// normalization already happened in topk order. Values are untouched, only
// their storage order changes -- which makes the j index of slots[t*K+j] an
// e-ascending walk, so the sequential combine reproduces the per-expert host
// loop's e-ascending fp16 accumulation order EXACTLY (bit-exactness contract).
__global__ void moe_tk_esort(int *tk_idx, float *tk_wt, int T, int K){
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  int   *ei = tk_idx + (long)t * K;
  float *ew = tk_wt + (long)t * K;
  for (int a = 1; a < K; ++a) {
    int e = ei[a]; float w = ew[a]; int b = a - 1;
    for (; b >= 0 && ei[b] > e; --b) { ei[b+1] = ei[b]; ew[b+1] = ew[b]; }
    ei[b+1] = e; ew[b+1] = w;
  }
}

// Padded scan (vLLM moe_align_block_size shape): every expert's bucket starts
// at a multiple of BM in the gathered row space, and the block work list maps
// grid block b -> its expert (or -1 = padding block, self-discard). The HOST
// never reads any of this: Wcap is the data-independent worst case
// ceil(T*K/BM) + E, computed from shapes alone.
__global__ void moe_route_scan_pad(const int *counts, int *cursor, int *wl_e,
                                   int *wl_n, int E, int BM, int Wcap){
  if (threadIdx.x != 0) return;
  int accp = 0, w = 0;
  for (int e = 0; e < E; ++e) {
    cursor[e] = accp;
    const int nb = (counts[e] + BM - 1) / BM;
    for (int i = 0; i < nb && w < Wcap; ++i) {
      const int rem = counts[e] - i * BM;
      wl_n[w] = rem < BM ? rem : BM; // live rows of this window
      wl_e[w++] = e;
    }
    accp += nb * BM;
  }
  for (; w < Wcap; ++w) { wl_n[w] = 0; wl_e[w] = -1; }
}
// Parallel form of moe_route_scan_pad: ONE block; a Hillis-Steele scan over
// the per-expert window counts replaces the single-thread serial walk, and
// every window resolves its owning expert by binary search on the scanned
// bases. Outputs are integer and deterministic — identical to the serial
// kernel's, including the Wcap-overflow truncation (windows past Wcap are
// simply never written, and the tail fill stops at Wcap). Requires
// E <= blockDim (the launcher passes 256/256).
__global__ void moe_route_scan_pad_p(const int *counts, int *cursor, int *wl_e,
                                     int *wl_n, int E, int BM, int Wcap){
  extern __shared__ int swb[]; // E+1 ints: exclusive scan of window counts
  const int tid = threadIdx.x, nt = blockDim.x;
  const int nb = (tid < E) ? (counts[tid] + BM - 1) / BM : 0;
  // exclusive prefix over nb (Hillis-Steele on an inclusive scan, shifted)
  swb[tid + 1] = nb;
  if (tid == 0)
    swb[0] = 0;
  __syncthreads();
  for (int off = 1; off < E; off <<= 1) {
    int add = (tid >= off && tid < E) ? swb[tid + 1 - off] : 0;
    __syncthreads();
    if (tid < E)
      swb[tid + 1] += add;
    __syncthreads();
  }
  if (tid < E)
    cursor[tid] = swb[tid] * BM;
  const int wtot = swb[E];
  for (int w = tid; w < Wcap; w += nt) {
    if (w >= wtot) { wl_n[w] = 0; wl_e[w] = -1; continue; }
    // find e with swb[e] <= w < swb[e+1]
    int lo = 0, hi = E - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (swb[mid] <= w) lo = mid; else hi = mid - 1;
    }
    const int rem = counts[lo] - (w - swb[lo]) * BM;
    wl_n[w] = rem < BM ? rem : BM;
    wl_e[w] = lo;
  }
}

// bucket + reverse map: slots[t*K+j] = the gathered row of token t's j-th
// assignment. With moe_tk_esort applied first, j order IS e-ascending order.
// rows[] must be pre-filled with -1 so per-expert padding tails read as
// "no source token" in the grouped GEMM's A staging.
__global__ void moe_route_bucket_rev(const int *tk_idx, const float *tk_wt,
                                     int *cursor, int *rows, float *wts,
                                     int *slots, int T, int K){
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)T * K) return;
  const int e = tk_idx[i];
  const int slot = atomicAdd(&cursor[e], 1);
  rows[slot] = (int)(i / K);
  wts[slot] = tk_wt[i];
  slots[i] = slot;
}

// Work-list-indexed SwiGLU + act-quant for the grouped path: blockIdx.y walks
// the SAME padded block work list as the GEMMs, so padding blocks self-discard
// on wl_e == -1 instead of processing Pcap-worth of dead rows (2.6x of the
// real rows at a 1.3K chunk). One block = one 64-row tile of the gathered
// space; grid.x covers the feature dim.
__global__ void moe_swiglu_wl(const unsigned short *gate,
                              const unsigned short *up, unsigned short *out,
                              const int *wl_e, int I){
  if (wl_e[blockIdx.y] < 0) return;
  const long base = (long)blockIdx.y*64*I;
  const long n = 64L*I;
  for (long i = (long)blockIdx.x*blockDim.x + threadIdx.x; i < n;
       i += (long)gridDim.x*blockDim.x) {
    const float g = moe_h2f(gate[base + i]);
    out[base + i] = moe_f2h((g / (1.0f + expf(-g))) * moe_h2f(up[base + i]));
  }
}
__global__ void moe_actq_wl(const unsigned short *Xh, signed char *q8,
                            float *ascale, int *azp, const int *wl_e, int K){
  if (wl_e[blockIdx.y] < 0) return;
  const int m = blockIdx.y*64 + blockIdx.x; // gathered row
  __shared__ float smn[256];
  __shared__ float smx[256];
  const unsigned short *xr = Xh + (long)m*K;
  float lmn = 0.f, lmx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = moe_h2f(xr[k]);
    lmn = fminf(lmn, v); lmx = fmaxf(lmx, v);
  }
  smn[threadIdx.x] = lmn; smx[threadIdx.x] = lmx;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x+s]);
      smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x+s]);
    }
    __syncthreads();
  }
  float scale_q, recip; int zp;
  moe_asym(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) { ascale[m] = recip; azp[m] = zp; }
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = (int)rintf(moe_h2f(xr[k]) * scale_q) + zp;
    q = max(-128, min(127, q));
    q8[(long)m*K + k] = (signed char)q;
  }
}

// SwiGLU and the int8 re-quant in ONE pass over the gathered row. The
// two-kernel sequence rounds silu(g)*u to fp16 (the g_S plane) and quantizes
// what it reads back; here that round happens in-register (moe_f2h then
// moe_h2f), so per-thread min/max, the reduction, scale/zp and every q8 byte
// are identical -- g_S never exists. v[] carries the row's fp16 values across
// the reduction: capacity 8*blockDim elements, driver guards K <= 2048.
__global__ void moe_swiglu_actq_wl(const unsigned short *gate,
                                   const unsigned short *up, signed char *q8,
                                   float *ascale, int *azp, const int *wl_e,
                                   int K){
  if (wl_e[blockIdx.y] < 0) return;
  const int m = blockIdx.y*64 + blockIdx.x; // gathered row
  const unsigned short *gr = gate + (long)m*K;
  const unsigned short *ur = up + (long)m*K;
  __shared__ float smn[256];
  __shared__ float smx[256];
  unsigned short v[8];
  float lmn = 0.f, lmx = 0.f;
  int c = 0;
  for (int k = threadIdx.x; k < K; k += blockDim.x, ++c) {
    const float g = moe_h2f(gr[k]);
    const unsigned short h =
      moe_f2h((g / (1.0f + expf(-g))) * moe_h2f(ur[k]));
    v[c] = h;
    const float f = moe_h2f(h);
    lmn = fminf(lmn, f); lmx = fmaxf(lmx, f);
  }
  smn[threadIdx.x] = lmn; smx[threadIdx.x] = lmx;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x+s]);
      smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x+s]);
    }
    __syncthreads();
  }
  float scale_q, recip; int zp;
  moe_asym(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) { ascale[m] = recip; azp[m] = zp; }
  c = 0;
  for (int k = threadIdx.x; k < K; k += blockDim.x, ++c) {
    int q = (int)rintf(moe_h2f(v[c]) * scale_q) + zp;
    q = max(-128, min(127, q));
    q8[(long)m*K + k] = (signed char)q;
  }
}

// Warp-per-row variant of the fused kernel: 8 warps = 8 rows per block, the
// min/max reduced by shuffles -- no shared memory, no __syncthreads (the
// 256-thread tree above pays 8 barriers for a 512-element row). min/max is
// order-insensitive (fminf/fmaxf drop NaN identically in any order, and a
// signed-zero min feeds arithmetic where -0 == +0), so the different lane
// mapping still produces the two-kernel arm's exact scale/zp and q8 bytes.
// v[16] caps K at 512; the driver falls back to the block variant above.
__global__ void moe_swiglu_actq_w32(const unsigned short *gate,
                                    const unsigned short *up, signed char *q8,
                                    float *ascale, int *azp, const int *wl_e,
                                    const int *wl_n, int K){
  if (wl_e[blockIdx.y] < 0) return;
  const int wid = threadIdx.x >> 5;
  // Padding rows are dead outputs: their q8/scale feed only the down GEMM's
  // pad output rows, which the combine's slots never reference. Skipping is
  // therefore output-identical (real pad share ~25% of Wcap*64 rows;
  // decode windows are live=1 -> 7 of 8 warps exit).
  if (blockIdx.x*8 + wid >= wl_n[blockIdx.y]) return;
  const int lane = threadIdx.x & 31;
  const int m = blockIdx.y*64 + blockIdx.x*8 + wid; // gathered row
  const unsigned short *gr = gate + (long)m*K;
  const unsigned short *ur = up + (long)m*K;
  unsigned short v[16];
  float lmn = 0.f, lmx = 0.f;
  int c = 0;
  for (int k = lane; k < K; k += 32, ++c) {
    const float g = moe_h2f(gr[k]);
    const unsigned short h =
      moe_f2h((g / (1.0f + expf(-g))) * moe_h2f(ur[k]));
    v[c] = h;
    const float f = moe_h2f(h);
    lmn = fminf(lmn, f); lmx = fmaxf(lmx, f);
  }
  for (int o = 16; o > 0; o >>= 1) {
    lmn = fminf(lmn, __shfl_down_sync(0xffffffffu, lmn, o));
    lmx = fmaxf(lmx, __shfl_down_sync(0xffffffffu, lmx, o));
  }
  lmn = __shfl_sync(0xffffffffu, lmn, 0);
  lmx = __shfl_sync(0xffffffffu, lmx, 0);
  float scale_q, recip; int zp;
  moe_asym(lmn, lmx, scale_q, recip, zp);
  if (lane == 0) { ascale[m] = recip; azp[m] = zp; }
  c = 0;
  for (int k = lane; k < K; k += 32, ++c) {
    int q = (int)rintf(moe_h2f(v[c]) * scale_q) + zp;
    q = max(-128, min(127, q));
    q8[(long)m*K + k] = (signed char)q;
  }
}
// Vectorized twin of moe_swiglu_actq_w32: uint2 gate/up loads, the fp16
// SwiGLU carried in registers as uint2 (same v[16] register budget), packed
// int q8 stores. BIT-IDENTICAL: only the lane->element assignment changes
// (vector-of-4 instead of stride-32); per-element arithmetic is unchanged
// and the min/max reduction is order-independent. Requires K % 128 == 0
// (so nv = K/4 splits evenly across 32 lanes) and 8-byte-aligned rows; the
// launcher checks both and falls back to the scalar form.
__global__ void moe_swiglu_actq_w32_v4(const unsigned short *gate,
                                       const unsigned short *up,
                                       signed char *q8, float *ascale,
                                       int *azp, const int *wl_e,
                                       const int *wl_n, int K){
  if (wl_e[blockIdx.y] < 0) return;
  const int wid = threadIdx.x >> 5;
  if (blockIdx.x*8 + wid >= wl_n[blockIdx.y]) return; // dead pad row
  const int lane = threadIdx.x & 31;
  const int m = blockIdx.y*64 + blockIdx.x*8 + wid; // gathered row
  const uint2 *gv = (const uint2 *)(gate + (long)m*K);
  const uint2 *uv = (const uint2 *)(up + (long)m*K);
  int *q32 = (int *)(q8 + (long)m*K);
  const int nv = K >> 2;
  uint2 v[4]; // fp16 SwiGLU carry, 4 halves per slot (K<=512 at 32 lanes)
  float lmn = 0.f, lmx = 0.f;
  int c = 0;
  for (int i = lane; i < nv; i += 32, ++c) {
    const uint2 gr = gv[i];
    const uint2 ur = uv[i];
    const float g0 = moe_h2f((unsigned short)(gr.x & 0xFFFFu));
    const float g1 = moe_h2f((unsigned short)(gr.x >> 16));
    const float g2 = moe_h2f((unsigned short)(gr.y & 0xFFFFu));
    const float g3 = moe_h2f((unsigned short)(gr.y >> 16));
    const unsigned short h0 = moe_f2h((g0 / (1.0f + expf(-g0))) *
                                      moe_h2f((unsigned short)(ur.x & 0xFFFFu)));
    const unsigned short h1 = moe_f2h((g1 / (1.0f + expf(-g1))) *
                                      moe_h2f((unsigned short)(ur.x >> 16)));
    const unsigned short h2 = moe_f2h((g2 / (1.0f + expf(-g2))) *
                                      moe_h2f((unsigned short)(ur.y & 0xFFFFu)));
    const unsigned short h3 = moe_f2h((g3 / (1.0f + expf(-g3))) *
                                      moe_h2f((unsigned short)(ur.y >> 16)));
    v[c].x = (unsigned int)h0 | ((unsigned int)h1 << 16);
    v[c].y = (unsigned int)h2 | ((unsigned int)h3 << 16);
    const float f0 = moe_h2f(h0), f1 = moe_h2f(h1);
    const float f2 = moe_h2f(h2), f3 = moe_h2f(h3);
    lmn = fminf(lmn, fminf(fminf(f0, f1), fminf(f2, f3)));
    lmx = fmaxf(lmx, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
  }
  for (int o = 16; o > 0; o >>= 1) {
    lmn = fminf(lmn, __shfl_down_sync(0xffffffffu, lmn, o));
    lmx = fmaxf(lmx, __shfl_down_sync(0xffffffffu, lmx, o));
  }
  lmn = __shfl_sync(0xffffffffu, lmn, 0);
  lmx = __shfl_sync(0xffffffffu, lmx, 0);
  float scale_q, recip; int zp;
  moe_asym(lmn, lmx, scale_q, recip, zp);
  if (lane == 0) { ascale[m] = recip; azp[m] = zp; }
  c = 0;
  for (int i = lane; i < nv; i += 32, ++c) {
    const int q0 = max(-128, min(127, (int)rintf(moe_h2f((unsigned short)(v[c].x & 0xFFFFu)) * scale_q) + zp));
    const int q1 = max(-128, min(127, (int)rintf(moe_h2f((unsigned short)(v[c].x >> 16)) * scale_q) + zp));
    const int q2 = max(-128, min(127, (int)rintf(moe_h2f((unsigned short)(v[c].y & 0xFFFFu)) * scale_q) + zp));
    const int q3 = max(-128, min(127, (int)rintf(moe_h2f((unsigned short)(v[c].y >> 16)) * scale_q) + zp));
    q32[i] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) |
             ((q3 & 0xFF) << 24);
  }
}

// Sequential-rounding combine: bit-identical to the per-expert path's
// moe_scatter_add_h sequence (one fp16 round after EVERY expert, dst starting
// from zero), walked in slots' j order == e-ascending. moe_combine_h (fp32
// accumulate, one final round) is the better-precision variant but does NOT
// reproduce the per-expert bytes; this one exists so the grouped path can be
// gated as bit-identical before anything else is argued about.
__global__ void moe_combine_seq_h(const unsigned short *Y, unsigned short *out,
                                  const int *slots, const float *wts,
                                  int T, int topk, int width){
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)T * width) return;
  const int t = (int)(idx / width);
  const int w = (int)(idx - (long)t * width);
  unsigned short acc = (unsigned short)0; // fp16 +0
  for (int k = 0; k < topk; ++k) {
    const int a = slots[(long)t * topk + k];
    if (a >= 0)
      acc = moe_f2h(moe_h2f(acc) + moe_h2f(Y[(long)a * width + w]) * wts[a]);
  }
  out[(long)t * width + w] = acc;
}
// uint4 (8-half) form of the sequential-rounding combine. BIT-IDENTICAL to
// moe_combine_seq_h: the rounding chain runs along k per (t,w) element, and
// that chain is untouched -- only the w axis is vectorized, so each element
// sees the identical operation sequence. The 8x wider thread also gives the
// 8 dependent k-gathers of the scalar form 8 independent FMA chains of ILP.
// Requires width % 8 == 0 and 16-byte-aligned Y/out (the launcher checks).
__global__ void moe_combine_seq_h_v8(const unsigned short *Y,
                                     unsigned short *out, const int *slots,
                                     const float *wts, int T, int topk,
                                     int width){
  const int wv8 = width >> 3;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)T * wv8) return;
  const int t = (int)(idx / wv8);
  const int v = (int)(idx - (long)t * wv8);
  const uint4 *Yv = (const uint4 *)Y;
  uint4 *ov = (uint4 *)out;
  unsigned short acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int k = 0; k < topk; ++k) {
    const int a = slots[(long)t * topk + k];
    if (a < 0) continue;
    const float wt = wts[a];
    const uint4 y = Yv[(long)a * wv8 + v];
    const unsigned short *yh = (const unsigned short *)&y;
    #pragma unroll
    for (int j = 0; j < 8; ++j)
      acc[j] = moe_f2h(moe_h2f(acc[j]) + moe_h2f(yh[j]) * wt);
  }
  uint4 o;
  unsigned short *oh = (unsigned short *)&o;
  #pragma unroll
  for (int j = 0; j < 8; ++j)
    oh[j] = acc[j];
  ov[(long)t * wv8 + v] = o;
}
} // extern "C"
)CU";

namespace {
std::mutex g_moe_mtx;

// Host-written, device-read staging (mapped) ...
int *g_rows = nullptr, *g_slots = nullptr;
int *g_wl_e = nullptr, *g_wl_r0 = nullptr, *g_wl_n = nullptr;
float *g_wts = nullptr;
const unsigned char **g_wp = nullptr; // unused since ptr tables went per-layer
const unsigned short **g_ws = nullptr;
// One capacity per pointer. Sharing a single c_wl across the three work-list
// arrays silently left two of them null: the first grow set the capacity and
// the next two then saw need <= cap and returned true without allocating.
size_t c_rows = 0, c_slots = 0, c_wts = 0;
size_t c_wl_e = 0, c_wl_r0 = 0, c_wl_n = 0;
// ... and device-only scratch.
unsigned short *g_X = nullptr, *g_G = nullptr, *g_U = nullptr, *g_S = nullptr,
               *g_Y = nullptr;
signed char *g_qa = nullptr, *g_qb = nullptr;
float *g_sa = nullptr, *g_sb = nullptr;
int *g_za = nullptr, *g_zb = nullptr;
size_t c_X = 0, c_G = 0, c_U = 0, c_S = 0, c_Y = 0, c_qa = 0, c_qb = 0,
       c_sa = 0, c_sb = 0, c_za = 0, c_zb = 0;

bool grow_mapped(void **p, size_t *cap, size_t need) {
  if (need <= *cap)
    return true;
  if (StreamManager::Global().isCapturing())
    return false;
  if (*p)
    cudaFreeHost(*p);
  *p = nullptr;
  *cap = 0;
  if (cudaHostAlloc(p, need, cudaHostAllocMapped) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  *cap = need;
  return true;
}
bool grow_dev(void **p, size_t *cap, size_t need) {
  if (need <= *cap)
    return true;
  if (StreamManager::Global().isCapturing()) {
    // A cudaMalloc inside a capture invalidates the stream, so the guard is
    // correct -- but the REFUSAL then makes the caller fall back to a host
    // path, which invalidates the capture just as surely. Either way the
    // prefill graph is lost, and silently. Say so: the fix is to size this
    // scratch before the first captured forward, not to relax the guard.
    ml_logw("[CUDA] moe scratch grow refused during capture (need %zu, cap "
            "%zu) -- the prefill graph will fall back",
            need, *cap);
    return false;
  }
  if (*p)
    cudaFree(*p);
  *p = nullptr;
  *cap = 0;
  if (cudaMalloc(p, need) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  *cap = need;
  return true;
}
} // namespace

bool cuda_moe_stage(unsigned int m, int **rows_out, float **wts_out) {
  std::lock_guard<std::mutex> lk(g_moe_mtx);
  if (!grow_mapped((void **)&g_rows, &c_rows, (size_t)m * sizeof(int)) ||
      !grow_mapped((void **)&g_wts, &c_wts, (size_t)m * sizeof(float)))
    return false;
  *rows_out = g_rows;
  *wts_out = g_wts;
  return true;
}

// counts[E] and offs[E]. MAPPED, not device-only: the per-expert loop below
// is host-driven, so the host has to read these back. That single 2 KB read is
// what still stands between this and a capturable prefill -- removing it needs
// the grouped kernel, which consumes the plan without a host loop.
bool cuda_moe_route_stage(unsigned int E, int **counts_out, int **offs_out) {
  static int *g_counts = nullptr, *g_offs = nullptr;
  static size_t c_counts = 0, c_offs = 0;
  std::lock_guard<std::mutex> lk(g_moe_mtx);
  if (!grow_mapped((void **)&g_counts, &c_counts, (size_t)E * sizeof(int)) ||
      !grow_mapped((void **)&g_offs, &c_offs, (size_t)E * sizeof(int)))
    return false;
  *counts_out = g_counts;
  *offs_out = g_offs;
  return true;
}

bool cuda_moe_gather_fp16(const unsigned short *src, unsigned short *dst,
                          const int *rows, unsigned int m, unsigned int width) {
  if (m == 0 || width == 0)
    return true;
  auto k = CudaContext::Global().registerCudaKernel(MOE_SRC, "moe_gather_h");
  if (!k)
    return false;
  int im = (int)m, iw = (int)width;
  k->SetKernelArguments(0, &src, sizeof(src));
  k->SetKernelArguments(1, &dst, sizeof(dst));
  k->SetKernelArguments(2, &rows, sizeof(rows));
  k->SetKernelArguments(3, &im, sizeof(im));
  k->SetKernelArguments(4, &iw, sizeof(iw));
  const long total = (long)m * width;
  const int B = 256;
  const int g[3] = {(int)((total + B - 1) / B), 1, 1}, b[3] = {B, 1, 1};
  return StreamManager::Global().DispatchCommand(*k, g, b);
}

bool cuda_moe_router_gemm_fp16(const unsigned short *X, const float *Wg,
                               float *L, unsigned int T, unsigned int H,
                               unsigned int E) {
  if (T == 0 || H == 0 || E == 0)
    return true;
  // [rgemm-h2] Split-fp16 dual-pass on cuBLAS fp16 tensor cores. The fp32
  // SIMT tile runs 2.1 TFLOPS = 39% of SIMT peak (its structural ceiling);
  // cuBLAS fp16 measures 22.5 TFLOPS at this very shape (dense_mchunk_probe
  // (4096,256,2048) = 0.191 ms), so hi+lo at twice the FLOPs still lands
  // ~4-5x. hi=(half)w, lo=(half)(w-hi): fp32-accumulated X*hi + X*lo carries
  // ~22-24 mantissa bits of the fp32 product -- top-k is discrete, so plain
  // fp16 W was never an option, and the gate is logits max|d| + top-1
  // agreement (NNTR_MOE_R_DBG=1 compares the first calls against the tile)
  // plus the e2e text/NLL gates. Prefill only (T>1); decode keeps the tile.
  // NNTR_MOE_RGEMM_H2=0 opts out; any failure falls through to the tile.
  static const bool h2_on = []() {
    const char *e = std::getenv("NNTR_MOE_RGEMM_H2");
    return !(e != nullptr && e[0] == '0');
  }();
  static const bool rg_cmp = []() {
    const char *e = std::getenv("NNTR_MOE_R_DBG");
    return e != nullptr && e[0] == '1';
  }();
  if (h2_on && T > 1) {
    static std::map<const float *, std::pair<unsigned short *, unsigned short *>>
      g_rsplit;
    auto it = g_rsplit.find(Wg);
    if (it == g_rsplit.end() && !StreamManager::Global().isCapturing()) {
      auto ks = CudaContext::Global().registerCudaKernel(MOE_SRC, "moe_rsplit");
      unsigned short *hi = nullptr, *lo = nullptr;
      const size_t n = (size_t)H * E;
      bool built = false;
      if (ks && cudaMalloc(&hi, n * 2) == cudaSuccess &&
          cudaMalloc(&lo, n * 2) == cudaSuccess) {
        int ni = (int)n;
        ks->SetKernelArguments(0, &Wg, sizeof(Wg));
        ks->SetKernelArguments(1, &hi, sizeof(hi));
        ks->SetKernelArguments(2, &lo, sizeof(lo));
        ks->SetKernelArguments(3, &ni, sizeof(ni));
        const int gs[3] = {(int)((n + 255) / 256), 1, 1}, bs[3] = {256, 1, 1};
        built = StreamManager::Global().DispatchCommand(*ks, gs, bs);
        if (built)
          quant_stage_survive(hi, lo); // one-time plane build: writes only
                                       // the freshly cudaMalloc'd hi/lo
      }
      if (!built) {
        if (hi)
          cudaFree(hi);
        if (lo)
          cudaFree(lo);
        hi = lo = nullptr; // negative-cache: do not retry every call
      }
      it = g_rsplit.emplace(Wg, std::make_pair(hi, lo)).first;
    }
    if (it != g_rsplit.end() && it->second.first != nullptr) {
      if (cuda_fc_dense_gemm_fp16_f32out_acc(X, it->second.first, L, T, E, H,
                                             false) &&
          cuda_fc_dense_gemm_fp16_f32out_acc(X, it->second.second, L, T, E, H,
                                             true)) {
        static int cmp_n = 0;
        if (rg_cmp && cmp_n < 3) {
          ++cmp_n;
          float *Lr = nullptr;
          const size_t ln = (size_t)T * E;
          if (cudaMalloc(&Lr, ln * 4) == cudaSuccess) {
            auto kr =
              CudaContext::Global().registerCudaKernel(MOE_SRC,
                                                       "moe_router_gemm");
            if (kr) {
              int t2 = (int)T, h2 = (int)H, e2 = (int)E;
              kr->SetKernelArguments(0, &X, sizeof(X));
              kr->SetKernelArguments(1, &Wg, sizeof(Wg));
              kr->SetKernelArguments(2, &Lr, sizeof(Lr));
              kr->SetKernelArguments(3, &t2, sizeof(t2));
              kr->SetKernelArguments(4, &h2, sizeof(h2));
              kr->SetKernelArguments(5, &e2, sizeof(e2));
              const int gg[3] = {(int)((E + 63) / 64), (int)((T + 63) / 64), 1};
              const int bb[3] = {256, 1, 1};
              if (StreamManager::Global().DispatchCommand(*kr, gg, bb)) {
                StreamManager::Global().finish();
                std::vector<float> a(ln), r(ln);
                cudaMemcpy(a.data(), L, ln * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(r.data(), Lr, ln * 4, cudaMemcpyDeviceToHost);
                double md = 0;
                unsigned t1d = 0;
                for (unsigned rr = 0; rr < T; ++rr) {
                  unsigned a1 = 0, b1 = 0;
                  for (unsigned c = 0; c < E; ++c) {
                    const size_t ix = (size_t)rr * E + c;
                    const double d = fabsf(a[ix] - r[ix]);
                    if (d > md)
                      md = d;
                    if (a[ix] > a[(size_t)rr * E + a1])
                      a1 = c;
                    if (r[ix] > r[(size_t)rr * E + b1])
                      b1 = c;
                  }
                  if (a1 != b1)
                    ++t1d;
                }
                fprintf(stderr,
                        "[moe_r_h2] T=%u cmp-vs-tile max|d|=%.3e top1_diff=%u/%u\n",
                        T, md, t1d, T);
              }
            }
            cudaFree(Lr);
          }
        }
        return true;
      }
    }
  }
  // [rgemv-h2] Decode arm of the split-fp16 router: the fp32 SIMT tile's
  // 64x64 M-tile is 63/64 idle at M=1 and measured 240 us/call; a warp-per-
  // expert GEMV over TRANSPOSED hi/lo planes (one-time moe_rsplit_t, cached
  // per Wg) reads ~2 MB device-resident fp16 in one launch. Same numeric
  // class as the prefill dual-pass (split-fp16, fp32 accumulate); the tile
  // stays the fallback and the NNTR_MOE_RGEMV_H2=0 opt-out.
  static const bool rgemv_on = []() {
    const char *e = std::getenv("NNTR_MOE_RGEMV_H2");
    return !(e != nullptr && e[0] == '0');
  }();
  if (h2_on && rgemv_on && T == 1 && (H % 4u) == 0u) {
    static std::map<const float *, std::pair<unsigned short *, unsigned short *>>
      g_rsplit_t;
    auto it = g_rsplit_t.find(Wg);
    if (it == g_rsplit_t.end() && !StreamManager::Global().isCapturing()) {
      auto ks =
        CudaContext::Global().registerCudaKernel(MOE_SRC, "moe_rsplit_t");
      unsigned short *hiT = nullptr, *loT = nullptr;
      const size_t n = (size_t)H * E;
      bool built = false;
      if (ks && cudaMalloc(&hiT, n * 2) == cudaSuccess &&
          cudaMalloc(&loT, n * 2) == cudaSuccess) {
        int hh = (int)H, ee = (int)E;
        ks->SetKernelArguments(0, &Wg, sizeof(Wg));
        ks->SetKernelArguments(1, &hiT, sizeof(hiT));
        ks->SetKernelArguments(2, &loT, sizeof(loT));
        ks->SetKernelArguments(3, &hh, sizeof(hh));
        ks->SetKernelArguments(4, &ee, sizeof(ee));
        const int gs[3] = {(int)((n + 255) / 256), 1, 1}, bs[3] = {256, 1, 1};
        built = StreamManager::Global().DispatchCommand(*ks, gs, bs);
        if (built)
          quant_stage_survive(hiT, loT); // one-time plane build: writes only
                                         // the freshly cudaMalloc'd planes
      }
      if (!built) {
        if (hiT)
          cudaFree(hiT);
        if (loT)
          cudaFree(loT);
        hiT = loT = nullptr; // negative-cache: do not retry every call
      }
      it = g_rsplit_t.emplace(Wg, std::make_pair(hiT, loT)).first;
    }
    if (it != g_rsplit_t.end() && it->second.first != nullptr) {
      auto kg =
        CudaContext::Global().registerCudaKernel(MOE_SRC, "moe_router_gemv_h2");
      if (kg) {
        int hh = (int)H, ee = (int)E;
        kg->SetKernelArguments(0, &X, sizeof(X));
        kg->SetKernelArguments(1, &it->second.first, sizeof(void *));
        kg->SetKernelArguments(2, &it->second.second, sizeof(void *));
        kg->SetKernelArguments(3, &L, sizeof(L));
        kg->SetKernelArguments(4, &hh, sizeof(hh));
        kg->SetKernelArguments(5, &ee, sizeof(ee));
        const int gg[3] = {(int)((E + 7) / 8), 1, 1}, bb[3] = {256, 1, 1};
        if (StreamManager::Global().DispatchCommand(*kg, gg, bb)) {
          StreamManager::Global().maybeFinish();
          return true;
        }
      }
    }
  }
  auto k = CudaContext::Global().registerCudaKernel(MOE_SRC, "moe_router_gemm");
  if (!k)
    return false;
  int t = (int)T, h = (int)H, e = (int)E;
  k->SetKernelArguments(0, &X, sizeof(X));
  k->SetKernelArguments(1, &Wg, sizeof(Wg));
  k->SetKernelArguments(2, &L, sizeof(L));
  k->SetKernelArguments(3, &t, sizeof(t));
  k->SetKernelArguments(4, &h, sizeof(h));
  k->SetKernelArguments(5, &e, sizeof(e));
  const int g[3] = {(int)((E + 63) / 64), (int)((T + 63) / 64), 1};
  const int b[3] = {256, 1, 1};
  // NNTR_MOE_R_DBG=1: router GEMM GPU ms via events (first few calls only).
  static const bool g_rgdbg = []() {
    const char *e = std::getenv("NNTR_MOE_R_DBG");
    return e != nullptr && e[0] == '1';
  }();
  static int g_rgdbg_n = 0;
  if (g_rgdbg && g_rgdbg_n < 3) {
    cudaStream_t st = StreamManager::Global().GetStream();
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, st);
    const bool ok = StreamManager::Global().DispatchCommand(*k, g, b);
    cudaEventRecord(ev1, st);
    cudaEventSynchronize(ev1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    fprintf(stderr, "[moe_r_dbg] T=%u router_gemm=%.3fms\n", T, ms);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    ++g_rgdbg_n;
    return ok;
  }
  return StreamManager::Global().DispatchCommand(*k, g, b);
}

bool cuda_moe_route_fp32(const float *logits, int *rows, float *wts,
                         int *counts, int *offs, unsigned int T, unsigned int E,
                         unsigned int K) {
  if (T == 0 || E == 0 || K == 0)
    return true;
  auto &ctx = CudaContext::Global();
  auto k1 = ctx.registerCudaKernel(MOE_SRC, "moe_route_topk");
  auto k2 = ctx.registerCudaKernel(MOE_SRC, "moe_route_scan");
  auto k3 = ctx.registerCudaKernel(MOE_SRC, "moe_route_bucket");
  if (!k1 || !k2 || !k3)
    return false;

  // scratch: [T*K] expert ids + [T*K] weights + [E] cursors, grown once
  static int *d_idx = nullptr, *d_cur = nullptr;
  static float *d_wt = nullptr;
  static size_t c_idx = 0, c_wt = 0, c_cur = 0;
  const size_t A = (size_t)T * K;
  if (!grow_dev((void **)&d_idx, &c_idx, sizeof(int) * A) ||
      !grow_dev((void **)&d_wt, &c_wt, sizeof(float) * A) ||
      !grow_dev((void **)&d_cur, &c_cur, sizeof(int) * E))
    return false;
  auto &sm = StreamManager::Global();
  if (cudaMemsetAsync(counts, 0, sizeof(int) * E, sm.GetStream()) != cudaSuccess)
    return false;

  int t = (int)T, e = (int)E, kk = (int)K;
  k1->SetKernelArguments(0, &logits, sizeof(logits));
  k1->SetKernelArguments(1, &d_idx, sizeof(d_idx));
  k1->SetKernelArguments(2, &d_wt, sizeof(d_wt));
  k1->SetKernelArguments(3, &counts, sizeof(counts));
  k1->SetKernelArguments(4, &t, sizeof(t));
  k1->SetKernelArguments(5, &e, sizeof(e));
  k1->SetKernelArguments(6, &kk, sizeof(kk));
  const int g1[3] = {(int)T, 1, 1}, b1[3] = {256, 1, 1};
  if (!sm.DispatchCommand(*k1, g1, b1, (unsigned int)(sizeof(float) * E)))
    return false;

  k2->SetKernelArguments(0, &counts, sizeof(counts));
  k2->SetKernelArguments(1, &offs, sizeof(offs));
  k2->SetKernelArguments(2, &d_cur, sizeof(d_cur));
  k2->SetKernelArguments(3, &e, sizeof(e));
  const int g2[3] = {1, 1, 1}, b2[3] = {32, 1, 1};
  if (!sm.DispatchCommand(*k2, g2, b2))
    return false;

  k3->SetKernelArguments(0, &d_idx, sizeof(d_idx));
  k3->SetKernelArguments(1, &d_wt, sizeof(d_wt));
  k3->SetKernelArguments(2, &d_cur, sizeof(d_cur));
  k3->SetKernelArguments(3, &rows, sizeof(rows));
  k3->SetKernelArguments(4, &wts, sizeof(wts));
  k3->SetKernelArguments(5, &t, sizeof(t));
  k3->SetKernelArguments(6, &kk, sizeof(kk));
  const int B = 256;
  const int g3[3] = {(int)((A + B - 1) / B), 1, 1}, b3[3] = {B, 1, 1};
  return sm.DispatchCommand(*k3, g3, b3);
}

bool cuda_moe_swiglu_fp16(const unsigned short *gate, const unsigned short *up,
                          unsigned short *out, unsigned int n) {
  if (n == 0)
    return true;
  auto k = CudaContext::Global().registerCudaKernel(MOE_SRC, "moe_swiglu_h");
  if (!k)
    return false;
  int in = (int)n;
  k->SetKernelArguments(0, &gate, sizeof(gate));
  k->SetKernelArguments(1, &up, sizeof(up));
  k->SetKernelArguments(2, &out, sizeof(out));
  k->SetKernelArguments(3, &in, sizeof(in));
  const int B = 256;
  const int g[3] = {(int)(((long)n + B - 1) / B), 1, 1}, b[3] = {B, 1, 1};
  return StreamManager::Global().DispatchCommand(*k, g, b);
}

bool cuda_moe_scatter_add_fp16(const unsigned short *src, unsigned short *dst,
                               const int *rows, const float *wts,
                               unsigned int m, unsigned int width) {
  if (m == 0 || width == 0)
    return true;
  auto k =
    CudaContext::Global().registerCudaKernel(MOE_SRC, "moe_scatter_add_h");
  if (!k)
    return false;
  int im = (int)m, iw = (int)width;
  k->SetKernelArguments(0, &src, sizeof(src));
  k->SetKernelArguments(1, &dst, sizeof(dst));
  k->SetKernelArguments(2, &rows, sizeof(rows));
  k->SetKernelArguments(3, &wts, sizeof(wts));
  k->SetKernelArguments(4, &im, sizeof(im));
  k->SetKernelArguments(5, &iw, sizeof(iw));
  const long total = (long)m * width;
  const int B = 256;
  const int g[3] = {(int)((total + B - 1) / B), 1, 1}, b[3] = {B, 1, 1};
  return StreamManager::Global().DispatchCommand(*k, g, b);
}

bool cuda_moe_plan_stage(unsigned int A, unsigned int T, unsigned int topk,
                         unsigned int E, unsigned int Wmax, MoePlan *out) {
  std::lock_guard<std::mutex> lk(g_moe_mtx);
  if (!grow_mapped((void **)&g_rows, &c_rows, (size_t)A * 4) ||
      !grow_mapped((void **)&g_wts, &c_wts, (size_t)A * 4) ||
      !grow_mapped((void **)&g_slots, &c_slots, (size_t)T * topk * 4) ||
      !grow_mapped((void **)&g_wl_e, &c_wl_e, (size_t)Wmax * 4) ||
      !grow_mapped((void **)&g_wl_r0, &c_wl_r0, (size_t)Wmax * 4) ||
      !grow_mapped((void **)&g_wl_n, &c_wl_n, (size_t)Wmax * 4))
    return false;
  (void)E;
  out->rows = g_rows;
  out->wts = g_wts;
  out->slots = g_slots;
  out->wl_e = g_wl_e;
  out->wl_r0 = g_wl_r0;
  out->wl_n = g_wl_n;
  return true; // wptr/wsc are the CALLER's, per layer
}

bool cuda_moe_plan_stage_dev(unsigned int A, unsigned int T, unsigned int topk,
                             unsigned int E, unsigned int Wmax, MoePlan *out) {
  // Separate statics from the mapped plan: the =1 arm's host loops write
  // g_rows/g_slots directly and MUST keep zero-copy staging; this path's
  // buffers are producer/consumer device-only.
  static int *d_rows = nullptr, *d_slots = nullptr, *d_wl_e = nullptr;
  static int *d_wl_r0 = nullptr, *d_wl_n = nullptr;
  static float *d_wts = nullptr;
  static size_t cr = 0, cs = 0, ce = 0, c0 = 0, cn = 0, cw = 0;
  std::lock_guard<std::mutex> lk(g_moe_mtx);
  if (!grow_dev((void **)&d_rows, &cr, (size_t)A * 4) ||
      !grow_dev((void **)&d_wts, &cw, (size_t)A * 4) ||
      !grow_dev((void **)&d_slots, &cs, (size_t)T * topk * 4) ||
      !grow_dev((void **)&d_wl_e, &ce, (size_t)Wmax * 4) ||
      !grow_dev((void **)&d_wl_r0, &c0, (size_t)Wmax * 4) ||
      !grow_dev((void **)&d_wl_n, &cn, (size_t)Wmax * 4))
    return false;
  (void)E;
  out->rows = d_rows;
  out->wts = d_wts;
  out->slots = d_slots;
  out->wl_e = d_wl_e;
  out->wl_r0 = d_wl_r0;
  out->wl_n = d_wl_n;
  return true; // wptr/wsc are the CALLER's, per layer
}

bool cuda_moe_new_ptr_table(unsigned int n, const unsigned char ***wp,
                            const unsigned short ***ws) {
  void *a = nullptr, *b = nullptr;
  const size_t sz = (size_t)n * sizeof(void *);
  if (cudaHostAlloc(&a, sz, cudaHostAllocMapped) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  if (cudaHostAlloc(&b, sz, cudaHostAllocMapped) != cudaSuccess) {
    cudaFreeHost(a);
    cudaGetLastError();
    return false;
  }
  *wp = (const unsigned char **)a;
  *ws = (const unsigned short **)b;
  return true;
}

bool moe_g3_enabled() {
  // DEFAULT ON (2026-08-12): gates passed -- byte-identical output, prefill
  // 1,123.2 -> 1,129.3 TPS (repack included; down 9.9 -> 7.52 ms via the
  // persistent-N g3d), decode 6.05 -> 7.04 TPS. =0 restores the classic
  // unpacked-payload arms (and skips the repack, so they stay valid).
  static const bool v = []() {
    const char *e = std::getenv("NNTR_MOE_G3");
    return e == nullptr || e[0] != '0';
  }();
  return v;
}

bool cuda_moe_new_wr_table(unsigned int n, const int ***wr) {
  void *a = nullptr;
  const size_t sz = (size_t)n * sizeof(void *);
  if (cudaHostAlloc(&a, sz, cudaHostAllocMapped) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  *wr = (const int **)a;
  return true;
}

namespace {
bool dispatch1d(const char *name, void **argp, const size_t *argsz, int nargs,
                long total, int block) {
  auto k = CudaContext::Global().registerCudaKernel(MOE_SRC, name);
  if (!k) {
    ml_loge("[CUDA] moe: kernel registration failed (%s)", name);
    return false;
  }
  for (int i = 0; i < nargs; ++i)
    k->SetKernelArguments(i, argp[i], argsz[i]);
  const int g[3] = {(int)((total + block - 1) / block), 1, 1},
            b[3] = {block, 1, 1};
  return StreamManager::Global().DispatchCommand(*k, g, b);
}
} // namespace

bool cuda_moe_expert_ffn_fp16(const unsigned short *input,
                              unsigned short *output, const MoePlan &p,
                              unsigned int A, unsigned int W, unsigned int T,
                              unsigned int topk, unsigned int H,
                              unsigned int I) {
  if (A == 0 || W == 0)
    return true;
  std::lock_guard<std::mutex> lk(g_moe_mtx);
  if (!grow_dev((void **)&g_X, &c_X, (size_t)A * H * 2) ||
      !grow_dev((void **)&g_G, &c_G, (size_t)A * I * 2) ||
      !grow_dev((void **)&g_U, &c_U, (size_t)A * I * 2) ||
      !grow_dev((void **)&g_S, &c_S, (size_t)A * I * 2) ||
      !grow_dev((void **)&g_Y, &c_Y, (size_t)A * H * 2) ||
      !grow_dev((void **)&g_qa, &c_qa, (size_t)A * H) ||
      !grow_dev((void **)&g_qb, &c_qb, (size_t)A * I) ||
      !grow_dev((void **)&g_sa, &c_sa, (size_t)A * 4) ||
      !grow_dev((void **)&g_sb, &c_sb, (size_t)A * 4) ||
      !grow_dev((void **)&g_za, &c_za, (size_t)A * 4) ||
      !grow_dev((void **)&g_zb, &c_zb, (size_t)A * 4)) {
    std::fprintf(stderr, "[cuda_moe] scratch alloc FAILED (A=%u): %s\n", A,
                 cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  auto &sm = StreamManager::Global();
  auto &ctx = CudaContext::Global();
  const size_t PS = sizeof(void *);
  int iA = (int)A, iH = (int)H, iI = (int)I, iT = (int)T, ik = (int)topk;

  { // gather every assignment's row in one launch
    void *a[] = {(void *)&input, (void *)&g_X, (void *)&p.rows, &iA, &iH};
    const size_t s[] = {PS, PS, PS, sizeof(int), sizeof(int)};
    if (!dispatch1d("moe_gather_h", a, s, 5, (long)A * H, 256))
      return false;
  }
  { // quantize once for BOTH gate and up (same activation)
    auto k = ctx.registerCudaKernel(MOE_SRC, "moe_actq_h");
    if (!k)
      return false;
    k->SetKernelArguments(0, &g_X, PS);
    k->SetKernelArguments(1, &g_qa, PS);
    k->SetKernelArguments(2, &g_sa, PS);
    k->SetKernelArguments(3, &g_za, PS);
    k->SetKernelArguments(4, &iA, sizeof(int));
    k->SetKernelArguments(5, &iH, sizeof(int));
    const int g[3] = {iA, 1, 1}, b[3] = {256, 1, 1};
    if (!sm.DispatchCommand(*k, g, b))
      return false;
  }
  auto grouped = [&](const unsigned char *const *wp,
                     const unsigned short *const *ws, const signed char *q8,
                     const float *sc, const int *zp, unsigned short *Y, int N,
                     int K) {
    auto k = ctx.registerCudaKernel(MOE_SRC, "moe_gemm_grouped");
    if (!k)
      return false;
    k->SetKernelArguments(0, &q8, PS);
    k->SetKernelArguments(1, &sc, PS);
    k->SetKernelArguments(2, &zp, PS);
    k->SetKernelArguments(3, &wp, PS);
    k->SetKernelArguments(4, &ws, PS);
    k->SetKernelArguments(5, &p.wl_e, PS);
    k->SetKernelArguments(6, &p.wl_r0, PS);
    k->SetKernelArguments(7, &p.wl_n, PS);
    k->SetKernelArguments(8, &Y, PS);
    k->SetKernelArguments(9, &N, sizeof(int));
    k->SetKernelArguments(10, &K, sizeof(int));
    const int g[3] = {(N + 63) / 64, (int)W, 1}, b[3] = {16, 16, 1};
    return sm.DispatchCommand(*k, g, b);
  };
  // The plan's pointer table is expert-major with three entries each; off_gate
  // / off_up / off_down pick the projection. The layer requests them in the
  // order up, gate, down, and applying gate/up the wrong way round yields
  // silu(up)*gate -- fluent-looking garbage, not an error.
  if (!grouped(p.wptr + p.off_gate, p.wsc + p.off_gate, g_qa, g_sa, g_za, g_G,
               (int)I, (int)H))
    return false;
  if (!grouped(p.wptr + p.off_up, p.wsc + p.off_up, g_qa, g_sa, g_za, g_U,
               (int)I, (int)H))
    return false;
  { // silu(gate) * up
    const long n = (long)A * I;
    int nn = (int)n;
    void *a[] = {(void *)&g_G, (void *)&g_U, (void *)&g_S, &nn};
    const size_t s[] = {PS, PS, PS, sizeof(int)};
    if (!dispatch1d("moe_swiglu_h", a, s, 4, n, 256))
      return false;
  }
  { // re-quantize the SwiGLU output for the down projection
    auto k = ctx.registerCudaKernel(MOE_SRC, "moe_actq_h");
    if (!k)
      return false;
    k->SetKernelArguments(0, &g_S, PS);
    k->SetKernelArguments(1, &g_qb, PS);
    k->SetKernelArguments(2, &g_sb, PS);
    k->SetKernelArguments(3, &g_zb, PS);
    k->SetKernelArguments(4, &iA, sizeof(int));
    k->SetKernelArguments(5, &iI, sizeof(int));
    const int g[3] = {iA, 1, 1}, b[3] = {256, 1, 1};
    if (!sm.DispatchCommand(*k, g, b))
      return false;
  }
  if (!grouped(p.wptr + p.off_down, p.wsc + p.off_down, g_qb, g_sb, g_zb, g_Y,
               (int)H, (int)I))
    return false;
  { // token-major weighted combine
    void *a[] = {(void *)&g_Y, (void *)&output, (void *)&p.slots,
                 (void *)&p.wts, &iT,           &ik,
                 &iH};
    const size_t s[] = {PS,          PS,          PS,         PS,
                        sizeof(int), sizeof(int), sizeof(int)};
    if (!dispatch1d("moe_combine_h", a, s, 7, (long)T * H, 256))
      return false;
  }
  return true;
}

// ---- padded grouped routing (vLLM moe_align_block_size shape) --------------
// Everything the per-expert loop needed the host to know (counts, offsets)
// stays on the device: the grid is sized from shapes alone (Wcap, Pcap) and
// padding blocks self-discard on wl_e == -1. No finish() anywhere.
bool cuda_moe_route_grouped_fp32(const float *logits, int *rows, float *wts,
                                 int *counts, int *wl_e, int *wl_n,
                                 int *slots, unsigned int T, unsigned int E,
                                 unsigned int K, unsigned int BM,
                                 unsigned int Wcap, unsigned int Pcap) {
  if (T == 0 || E == 0 || K == 0 || BM == 0 || Wcap == 0 || Pcap == 0)
    return false;
  // device-only intermediates (same lifetime pattern as cuda_moe_route_fp32).
  // d_cnt: the caller hands us the MAPPED zero-copy counts it shares with the
  // per-expert arm, but nothing on the host reads counts in the grouped path
  // -- and 32,768 atomicAdds per layer-chunk on zero-copy memory are the
  // measured cost of the 5 ms "topk" stage. Use a device-resident histogram
  // instead; the passed-in mapped buffer is deliberately ignored.
  static int *d_idx = nullptr;
  static float *d_wt = nullptr;
  static int *d_cur = nullptr;
  static int *d_cnt = nullptr;
  static size_t c_idx = 0, c_wt = 0, c_cur = 0, c_cnt = 0;
  std::lock_guard<std::mutex> lk(g_moe_mtx);
  if (!grow_dev((void **)&d_idx, &c_idx, (size_t)T * K * 4) ||
      !grow_dev((void **)&d_wt, &c_wt, (size_t)T * K * 4) ||
      !grow_dev((void **)&d_cur, &c_cur, (size_t)E * 4) ||
      !grow_dev((void **)&d_cnt, &c_cnt, (size_t)E * 4))
    return false;
  // NNTR_MOE_ROUTE_DEV=0 restores the caller's mapped counts (A/B isolation
  // switch; the device histogram is the default and the 12.4x win).
  static const bool g_route_dev = []() {
    const char *e = std::getenv("NNTR_MOE_ROUTE_DEV");
    return e == nullptr || e[0] != '0';
  }();
  if (g_route_dev)
    counts = d_cnt;
  auto &sm = StreamManager::Global();
  auto &ctx = CudaContext::Global();
  cudaStream_t st = sm.GetStream();
  // NNTR_MOE_R_DBG=1: per-kernel GPU ms via events (first few calls only).
  static const bool g_rdbg = []() {
    const char *e = std::getenv("NNTR_MOE_R_DBG");
    return e != nullptr && e[0] == '1';
  }();
  static int g_rdbg_n = 0;
  cudaEvent_t rev[5];
  const bool rdbg_this = g_rdbg && g_rdbg_n < 3;
  if (rdbg_this)
    for (int i = 0; i < 5; ++i)
      cudaEventCreate(&rev[i]);
  auto rstamp = [&](int i) {
    if (rdbg_this)
      cudaEventRecord(rev[i], st);
  };
  rstamp(0);
  if (cudaMemsetAsync(counts, 0, (size_t)E * 4, st) != cudaSuccess)
    return false;
  // rows pre-filled with -1: per-expert padding tails must read as "no source
  // token" in the grouped GEMM's A staging.
  if (cudaMemsetAsync(rows, 0xFF, (size_t)Pcap * 4, st) != cudaSuccess)
    return false;
  const size_t PS = sizeof(void *);
  int iT = (int)T, iE = (int)E, iK = (int)K, iBM = (int)BM, iW = (int)Wcap;
  { // softmax + top-k + normalize. _w = warp-per-token form: identical
    // selection/counts, tk_wt can move an ulp (denominator reduce order) --
    // NNTR_MOE_TOPK_W=0 restores the block form.
    static const bool tkw_on = []() {
      const char *e = std::getenv("NNTR_MOE_TOPK_W");
      return !(e != nullptr && e[0] == '0');
    }();
    const bool tkw = tkw_on && E == 256u && K <= 8u &&
                     (((uintptr_t)logits) & 15u) == 0u;
    auto k = ctx.registerCudaKernel(MOE_SRC,
                                    tkw ? "moe_route_topk_w" : "moe_route_topk");
    if (!k)
      return false;
    k->SetKernelArguments(0, &logits, PS);
    k->SetKernelArguments(1, &d_idx, PS);
    k->SetKernelArguments(2, &d_wt, PS);
    k->SetKernelArguments(3, &counts, PS);
    k->SetKernelArguments(4, &iT, sizeof(int));
    k->SetKernelArguments(5, &iE, sizeof(int));
    k->SetKernelArguments(6, &iK, sizeof(int));
    const int g[3] = {tkw ? (int)((T + 7) / 8) : iT, 1, 1}, b[3] = {256, 1, 1};
    if (!sm.DispatchCommand(*k, g, b,
                            tkw ? 0u : (unsigned int)(E * sizeof(float))))
      return false;
    quant_stage_survive(d_idx, d_wt, counts); // routing scratch only
  }
  rstamp(1);
  { // e-ascending sort of each token's pairs (order only, values untouched)
    void *a[] = {(void *)&d_idx, (void *)&d_wt, &iT, &iK};
    const size_t s[] = {PS, PS, sizeof(int), sizeof(int)};
    if (!dispatch1d("moe_tk_esort", a, s, 4, (long)T, 256))
      return false;
    quant_stage_survive(d_idx, d_wt); // in-place pair reorder
  }
  rstamp(2);
  { // padded offsets + block work list, single-threaded over E like the scan
    void *a[] = {(void *)&counts, (void *)&d_cur, (void *)&wl_e,
                 (void *)&wl_n,   &iE,            &iBM,          &iW};
    const size_t s[] = {PS,          PS,          PS,          PS,
                        sizeof(int), sizeof(int), sizeof(int)};
    // _p = identical-output parallel form (block scan + binary search);
    // needs E <= blockDim. The serial one-thread kernel stays for odd E.
    const bool par = E <= 256u;
    auto k = ctx.registerCudaKernel(
      MOE_SRC, par ? "moe_route_scan_pad_p" : "moe_route_scan_pad");
    if (!k)
      return false;
    for (int i = 0; i < 7; ++i)
      k->SetKernelArguments(i, a[i], s[i]);
    const int g[3] = {1, 1, 1}, b[3] = {par ? 256 : 32, 1, 1};
    if (!sm.DispatchCommand(*k, g, b, par ? (E + 1) * 4 : 0))
      return false;
    quant_stage_survive(d_cur, wl_e, wl_n); // routing scratch only
  }
  rstamp(3);
  { // bucket into padded slots + reverse map
    void *a[] = {(void *)&d_idx, (void *)&d_wt, (void *)&d_cur, (void *)&rows,
                 (void *)&wts,   (void *)&slots, &iT,           &iK};
    const size_t s[] = {PS, PS, PS, PS, PS, PS, sizeof(int), sizeof(int)};
    if (!dispatch1d("moe_route_bucket_rev", a, s, 8, (long)T * K, 256))
      return false;
    quant_stage_survive(rows, wts, slots, d_cur); // routing planes only
  }
  rstamp(4);
  if (rdbg_this) {
    cudaEventSynchronize(rev[4]);
    const char *nm[4] = {"topk+memset", "esort", "scan_pad", "bucket_rev"};
    fprintf(stderr, "[moe_r_dbg] T=%u ", T);
    for (int i = 0; i < 4; ++i) {
      float ms = 0.f;
      cudaEventElapsedTime(&ms, rev[i], rev[i + 1]);
      fprintf(stderr, "%s=%.3fms ", nm[i], ms);
    }
    fprintf(stderr, "\n");
    { // real padding census (the sync above already drained the stream)
      std::vector<int> hc(E);
      if (cudaMemcpy(hc.data(), counts, (size_t)E * 4,
                     cudaMemcpyDeviceToHost) == cudaSuccess) {
        long p = 0;
        int wu = 0, mx = 0, ze = 0;
        for (unsigned int e = 0; e < E; ++e) {
          p += hc[e];
          wu += (hc[e] + (int)BM - 1) / (int)BM;
          if (hc[e] > mx)
            mx = hc[e];
          if (hc[e] == 0)
            ++ze;
        }
        fprintf(stderr,
                "[moe_r_dbg] pairs=%ld rows_padded=%d (pad %.1f%%) Wcap=%u "
                "max_c=%d zero_e=%d\n",
                p, wu * (int)BM,
                p ? 100.0 * (wu * (int)BM - p) / (double)p : 0.0,
                (unsigned int)Wcap, mx, ze);
      }
    }
    for (int i = 0; i < 5; ++i)
      cudaEventDestroy(rev[i]);
    ++g_rdbg_n;
  }
  return true;
}

// ---- grouped expert FFN on the int4 Tensor-Core tile -----------------------
// The imma_gemm_pipe variant (cuda_fc_qint4.cpp: imma_moe_grouped) with the M
// axis on the padded work list. Differences from cuda_moe_expert_ffn_fp16:
// no gather and no per-assignment quant -- the layer input is quantized ONCE
// ([T,H] rows; per-row params are identical to quantizing gathered copies, so
// this is bit-exact) and the gate/up GEMMs read rows through p.rows; only the
// SwiGLU output lives (and quantizes) in gathered space.
bool cuda_moe_grouped_ffn_imma(const unsigned short *input,
                               unsigned short *output, const MoePlan &p,
                               unsigned int T, unsigned int topk,
                               unsigned int H, unsigned int I,
                               unsigned int Pcap, unsigned int Wcap) {
  if (T == 0 || Pcap == 0 || Wcap == 0)
    return true;
  if ((H & 63u) != 0u || (I & 63u) != 0u)
    return false; // the tile has no k tail and needs N%64==0 too (H,I serve
                  // as both N and K across the three projections)
  std::lock_guard<std::mutex> lk(g_moe_mtx);
  // MoE glue: SwiGLU and the re-quant run as ONE work-list kernel, with the
  // inter-kernel fp16 round reproduced in-register -- bytes match the
  // two-kernel arm exactly and the g_S plane is never allocated.
  // NNTR_MOE_GLUE=0 restores the two-kernel arm. The v[] register carry caps
  // K at 8*blockDim (2048); I beyond that falls back too.
  static const bool g_glue = []() {
    const char *e = std::getenv("NNTR_MOE_GLUE");
    return e == nullptr || e[0] != '0';
  }();
  const bool glue = g_glue && I <= 2048u;
  if (!grow_dev((void **)&g_G, &c_G, (size_t)Pcap * I * 2) ||
      !grow_dev((void **)&g_U, &c_U, (size_t)Pcap * I * 2) ||
      (!glue && !grow_dev((void **)&g_S, &c_S, (size_t)Pcap * I * 2)) ||
      !grow_dev((void **)&g_Y, &c_Y, (size_t)Pcap * H * 2) ||
      !grow_dev((void **)&g_qa, &c_qa, (size_t)T * H) ||
      !grow_dev((void **)&g_qb, &c_qb, (size_t)Pcap * I) ||
      !grow_dev((void **)&g_sa, &c_sa, (size_t)T * 4) ||
      !grow_dev((void **)&g_sb, &c_sb, (size_t)Pcap * 4) ||
      !grow_dev((void **)&g_za, &c_za, (size_t)T * 4) ||
      !grow_dev((void **)&g_zb, &c_zb, (size_t)Pcap * 4)) {
    std::fprintf(stderr, "[cuda_moe] grouped-imma scratch alloc FAILED "
                         "(T=%u Pcap=%u): %s\n",
                 T, Pcap, cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  auto &sm = StreamManager::Global();
  auto &ctx = CudaContext::Global();
  const size_t PS = sizeof(void *);
  int iT = (int)T, iH = (int)H, iI = (int)I, iP = (int)Pcap, ik = (int)topk;

  // NNTR_MOE_G_DBG=1: per-launch GPU ms via events (first few calls only).
  static const int g_dbg = []() {
    const char *e = std::getenv("NNTR_MOE_G_DBG");
    return e ? std::atoi(e) : 0; // N = print the first N calls (1 -> 3)
  }();
  static int g_dbg_n = 0;
  cudaEvent_t ev[8];
  const bool dbg_this = g_dbg && g_dbg_n < (g_dbg == 1 ? 3 : g_dbg);
  if (dbg_this)
    for (int i = 0; i < 8; ++i)
      cudaEventCreate(&ev[i]);
  auto stamp = [&](int i) {
    if (dbg_this)
      cudaEventRecord(ev[i], sm.GetStream());
  };
  stamp(0);

  { // quantize the LAYER input once (shared by gate and up, all experts).
    // v4p = bit-identical vectorized twin (min/max order-independent).
    const bool aq4 = (H % 4u) == 0u &&
                     ((((uintptr_t)input) | ((uintptr_t)g_qa)) & 7u) == 0u;
    auto k =
      ctx.registerCudaKernel(MOE_SRC, aq4 ? "moe_actq_h_v4p" : "moe_actq_h");
    if (!k)
      return false;
    k->SetKernelArguments(0, &input, PS);
    k->SetKernelArguments(1, &g_qa, PS);
    k->SetKernelArguments(2, &g_sa, PS);
    k->SetKernelArguments(3, &g_za, PS);
    k->SetKernelArguments(4, &iT, sizeof(int));
    k->SetKernelArguments(5, &iH, sizeof(int));
    const int g[3] = {iT, 1, 1}, b[3] = {256, 1, 1};
    if (!sm.DispatchCommand(*k, g, b))
      return false;
    quant_stage_survive(g_qa, g_sa, g_za); // MoE's OWN quant planes, not
                                           // the FC staging scratch
  }
  stamp(1);
  const auto *wp64 = reinterpret_cast<const unsigned long long *>(p.wptr);
  const auto *ws64 = reinterpret_cast<const unsigned long long *>(p.wsc);
  // gate, then up (the table order is up/gate/down -- off_* pick, see the
  // trap note in cuda_moe_expert_ffn_fp16). NNTR_MOE_G2=1 runs both as ONE
  // fused kernel (one A staging per two W tiles); output identical.
  static const bool g_g2 = []() {
    const char *e = std::getenv("NNTR_MOE_G2");
    return e != nullptr && e[0] == '1';
  }();
  // Wide-N 64x128 tile (32x32 warp tiles, half the B-ldmatrix per mma) for
  // the K=2048 gate/up shapes: measured -5% each, byte-identical (integer
  // accumulation is order-exact). Default ON; NNTR_MOE_WT=0 keeps the 64x64
  // tile for A/B. down is NOT wide (see the stamp(5) note).
  static const bool g_wt = []() {
    const char *e = std::getenv("NNTR_MOE_WT");
    return e == nullptr || e[0] != '0';
  }();
  const auto *wr64 = reinterpret_cast<const unsigned long long *>(p.wrs);
  const bool g3 = moe_g3_enabled() && p.wrs != nullptr;
  if (g3) {
    // packed fragment-order tile; wide/_g2 arms stand aside (they read the
    // raw nibble order, which no longer exists once the repack ran)
    if (p.m4_gateup) {
      // gate/up payloads are in m4 fragment-chunk order: _g4 only
      if (!cuda_fc_qs4cx_moe_grouped_gemm_g4(g_qa, p.rows, wp64 + p.off_gate,
                                             ws64 + p.off_gate,
                                             wr64 + p.off_gate, p.wl_e, g_sa,
                                             g_za, g_G, Wcap, I, H, 1, p.wl_n))
        return false;
      stamp(2);
      if (!cuda_fc_qs4cx_moe_grouped_gemm_g4(g_qa, p.rows, wp64 + p.off_up,
                                             ws64 + p.off_up, wr64 + p.off_up,
                                             p.wl_e, g_sa, g_za, g_U, Wcap, I,
                                             H, 1, p.wl_n))
        return false;
      stamp(3);
    } else {
      if (!cuda_fc_qs4cx_moe_grouped_gemm_g3(g_qa, p.rows, wp64 + p.off_gate,
                                             ws64 + p.off_gate,
                                             wr64 + p.off_gate, p.wl_e, g_sa,
                                             g_za, g_G, Wcap, I, H, 1, p.wl_n))
        return false;
      stamp(2);
      if (!cuda_fc_qs4cx_moe_grouped_gemm_g3(g_qa, p.rows, wp64 + p.off_up,
                                             ws64 + p.off_up, wr64 + p.off_up,
                                             p.wl_e, g_sa, g_za, g_U, Wcap, I,
                                             H, 1, p.wl_n))
        return false;
      stamp(3);
    }
  } else if (g_wt) {
    if (!cuda_fc_qs4cx_moe_grouped_gemm_w(g_qa, p.rows, wp64 + p.off_gate,
                                          ws64 + p.off_gate, p.wl_e, g_sa,
                                          g_za, g_G, Wcap, I, H, 1))
      return false;
    stamp(2);
    if (!cuda_fc_qs4cx_moe_grouped_gemm_w(g_qa, p.rows, wp64 + p.off_up,
                                          ws64 + p.off_up, p.wl_e, g_sa, g_za,
                                          g_U, Wcap, I, H, 1))
      return false;
    stamp(3);
  } else if (g_g2) {
    if (!cuda_fc_qs4cx_moe_grouped_gemm2(
          g_qa, p.rows, wp64 + p.off_gate, ws64 + p.off_gate,
          wp64 + p.off_up, ws64 + p.off_up, p.wl_e, g_sa, g_za, g_G, g_U,
          Wcap, I, H, 1))
      return false;
    stamp(2);
    stamp(3);
  } else {
    if (!cuda_fc_qs4cx_moe_grouped_gemm(g_qa, p.rows, wp64 + p.off_gate,
                                        ws64 + p.off_gate, p.wl_e, g_sa, g_za,
                                        g_G, Wcap, I, H, 1))
      return false;
    stamp(2);
    if (!cuda_fc_qs4cx_moe_grouped_gemm(g_qa, p.rows, wp64 + p.off_up,
                                        ws64 + p.off_up, p.wl_e, g_sa, g_za,
                                        g_U, Wcap, I, H, 1))
      return false;
    stamp(3);
  }
  if (glue) { // fused silu(gate)*up + re-quant; warp-per-row when it fits
    // (dbg labels unchanged: the fused time prints as "swiglu", "actq2"~0)
    const bool w32 = I <= 512u;
    // _v4 = bit-identical uint2 twin; its 4-slot register carry only covers
    // K <= 512 at 32 lanes, and K % 128 keeps the vec split lane-even.
    const bool w32v4 =
      w32 && (I % 128u) == 0u &&
      ((((uintptr_t)g_G) | ((uintptr_t)g_U) | ((uintptr_t)g_qb)) & 7u) == 0u;
    auto k = ctx.registerCudaKernel(MOE_SRC, w32v4 ? "moe_swiglu_actq_w32_v4"
                                             : w32 ? "moe_swiglu_actq_w32"
                                                   : "moe_swiglu_actq_wl");
    if (!k)
      return false;
    k->SetKernelArguments(0, &g_G, PS);
    k->SetKernelArguments(1, &g_U, PS);
    k->SetKernelArguments(2, &g_qb, PS);
    k->SetKernelArguments(3, &g_sb, PS);
    k->SetKernelArguments(4, &g_zb, PS);
    k->SetKernelArguments(5, &p.wl_e, PS);
    if (w32) { // both w32 arms take wl_n for the dead-pad-row skip
      k->SetKernelArguments(6, &p.wl_n, PS);
      k->SetKernelArguments(7, &iI, sizeof(int));
    } else {
      k->SetKernelArguments(6, &iI, sizeof(int));
    }
    const int g[3] = {w32 ? 8 : 64, (int)Wcap, 1}, b[3] = {256, 1, 1};
    if (!sm.DispatchCommand(*k, g, b))
      return false;
    quant_stage_survive(g_qb, g_sb, g_zb);
    stamp(4);
    stamp(5);
  } else {
  { // silu(gate) * up, work-list-indexed: padding blocks self-discard
    auto k = ctx.registerCudaKernel(MOE_SRC, "moe_swiglu_wl");
    if (!k)
      return false;
    k->SetKernelArguments(0, &g_G, PS);
    k->SetKernelArguments(1, &g_U, PS);
    k->SetKernelArguments(2, &g_S, PS);
    k->SetKernelArguments(3, &p.wl_e, PS);
    k->SetKernelArguments(4, &iI, sizeof(int));
    const int g[3] = {32, (int)Wcap, 1}, b[3] = {256, 1, 1};
    if (!sm.DispatchCommand(*k, g, b))
      return false;
    quant_stage_survive(g_S);
  }
  stamp(4);
  { // re-quantize the SwiGLU output, work-list-indexed
    auto k = ctx.registerCudaKernel(MOE_SRC, "moe_actq_wl");
    if (!k)
      return false;
    k->SetKernelArguments(0, &g_S, PS);
    k->SetKernelArguments(1, &g_qb, PS);
    k->SetKernelArguments(2, &g_sb, PS);
    k->SetKernelArguments(3, &g_zb, PS);
    k->SetKernelArguments(4, &p.wl_e, PS);
    k->SetKernelArguments(5, &iI, sizeof(int));
    const int g[3] = {64, (int)Wcap, 1}, b[3] = {256, 1, 1};
    if (!sm.DispatchCommand(*k, g, b))
      return false;
    quant_stage_survive(g_qb, g_sb, g_zb);
  }
  stamp(5);
  }
  // down stays on the 64x64 tile: with K=512 (8 k-steps) the wide tile's
  // doubled W staging outweighs its fragment savings -- measured 8.7 -> 9.5
  // ms. The wide tile pays only on the K=2048 gate/up shapes (-5% each).
  if (g3 && I <= 512u) {
    // persistent-N down: A loads once, W ring never drains between n-tiles
    if (!cuda_fc_qs4cx_moe_grouped_gemm_g3d(g_qb, wp64 + p.off_down,
                                            ws64 + p.off_down,
                                            wr64 + p.off_down, p.wl_e, g_sb,
                                            g_zb, g_Y, Wcap, H, I, 1,
                                            p.wl_n))
      return false;
  } else if (g3) {
    if (!cuda_fc_qs4cx_moe_grouped_gemm_g3(g_qb, /*tokid=*/nullptr,
                                           wp64 + p.off_down,
                                           ws64 + p.off_down,
                                           wr64 + p.off_down, p.wl_e, g_sb,
                                           g_zb, g_Y, Wcap, H, I, 1,
                                           p.wl_n))
      return false;
  } else if (!cuda_fc_qs4cx_moe_grouped_gemm(g_qb, /*tokid=*/nullptr,
                                             wp64 + p.off_down,
                                             ws64 + p.off_down, p.wl_e, g_sb,
                                             g_zb, g_Y, Wcap, H, I, 1))
    return false;
  stamp(6);
  { // sequential-rounding combine: bit-identical to the per-expert scatter.
    // v8 = the same chain 8 halves per thread (bit-identical, see kernel);
    // scalar stays as the odd-width / misaligned fallback.
    const bool v8 = (H % 8u) == 0u &&
                    ((((uintptr_t)g_Y) | ((uintptr_t)output)) & 15u) == 0u;
    void *a[] = {(void *)&g_Y, (void *)&output, (void *)&p.slots,
                 (void *)&p.wts, &iT,           &ik,
                 &iH};
    const size_t s[] = {PS,          PS,          PS,         PS,
                        sizeof(int), sizeof(int), sizeof(int)};
    if (!dispatch1d(v8 ? "moe_combine_seq_h_v8" : "moe_combine_seq_h", a, s, 7,
                    v8 ? (long)T * (H / 8) : (long)T * H, 256))
      return false;
    quant_stage_survive(output); // the ffn output tensor; never the staged X
  }
  stamp(7);
  if (dbg_this) {
    cudaEventSynchronize(ev[7]);
    float ms[7];
    const char *nm[7] = {"actq1", "gate", "up", "swiglu", "actq2", "down",
                         "combine"};
    fprintf(stderr, "[moe_g_dbg] T=%u ", T);
    for (int i = 0; i < 7; ++i) {
      cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]);
      fprintf(stderr, "%s=%.2fms ", nm[i], ms[i]);
    }
    fprintf(stderr, "\n");
    for (int i = 0; i < 8; ++i)
      cudaEventDestroy(ev[i]);
    ++g_dbg_n;
  }
  return true;
}

} // namespace nntrainer::cuda

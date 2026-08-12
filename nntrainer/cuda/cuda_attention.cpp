// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_attention.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA flash-style attention core (NVRTC).
 */

#include "cuda_attention.h"

#include <cuda_blas_manager.h>
#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_stream_manager.h>
#include <env_compat.h>

#include <cublas_v2.h>

#include <nntrainer_log.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace nntrainer::cuda {

// One block per (query head h, query row i). Online (flash) softmax in FP32.
// Each thread owns the head dims d = tid, tid+B, ... (<=4 for head_dim<=512,
// B>=128). Shared = Q row [head_dim] + reduction scratch [B].
static const char *ATTN_CORE_SRC = R"CU(
extern "C" __global__ void attn_core(const float *Q, const float *K,
                                     const float *V, float *O, int num_heads,
                                     int num_kv_heads, int q_rows, int kv_len,
                                     int q_pos0, int head_dim, int window,
                                     float softcap) {
  int i = blockIdx.x;
  int h = blockIdx.y;
  if (i >= q_rows || h >= num_heads)
    return;
  int gqa = num_heads / num_kv_heads;
  int hkv = h / gqa;
  const float *Qhi = Q + ((long)h * q_rows + i) * head_dim;
  const float *Kh = K + (long)hkv * kv_len * head_dim;
  const float *Vh = V + (long)hkv * kv_len * head_dim;
  float *Ohi = O + ((long)h * q_rows + i) * head_dim;

  int tid = threadIdx.x;
  int B = blockDim.x;
  extern __shared__ float sh[];
  float *Qsh = sh;        // [head_dim]
  float *red = sh + head_dim; // [B]
  for (int d = tid; d < head_dim; d += B)
    Qsh[d] = Qhi[d];
  __syncthreads();

  const float scale = rsqrtf((float)head_dim);
  float acc[4];
#pragma unroll
  for (int r = 0; r < 4; r++)
    acc[r] = 0.f;

  int i_abs = q_pos0 + i;
  int j_lo = i_abs - window + 1;
  if (j_lo < 0)
    j_lo = 0;
  int j_hi = i_abs;
  if (j_hi >= kv_len)
    j_hi = kv_len - 1;

  float m = -1e30f, l = 0.f;
  for (int j = j_lo; j <= j_hi; ++j) {
    const float *Kj = Kh + (long)j * head_dim;
    float pd = 0.f;
    for (int d = tid; d < head_dim; d += B)
      pd += Qsh[d] * Kj[d];
    red[tid] = pd;
    __syncthreads();
    for (int s = B >> 1; s > 0; s >>= 1) {
      if (tid < s)
        red[tid] += red[tid + s];
      __syncthreads();
    }
    float score = red[0] * scale;
    __syncthreads();
    if (softcap > 0.f)
      score = softcap * tanhf(score / softcap);
    float m_new = fmaxf(m, score);
    float corr = __expf(m - m_new);
    float p = __expf(score - m_new);
    l = l * corr + p;
    m = m_new;
    const float *Vj = Vh + (long)j * head_dim;
    int r = 0;
    for (int d = tid; d < head_dim; d += B, ++r)
      acc[r] = acc[r] * corr + p * Vj[d];
  }

  float inv = l > 0.f ? 1.f / l : 0.f;
  int r = 0;
  for (int d = tid; d < head_dim; d += B, ++r)
    Ohi[d] = acc[r] * inv;
}
)CU";

// Interleaved fp16 variant: reads the head-interleaved fp16 query step + fp16
// KV cache directly (de-interleave + half->float inline), flash core in FP32,
// writes interleaved fp16 output. Drop-in for the host gemm_attention.
static const char *ATTN_IL_FP16_SRC = R"CU(
extern "C" {

__device__ __forceinline__ float a_h2f(unsigned short h) {
  // native fp16->fp32 (exact); replaces the software bit-twiddle -- this runs
  // per Q/K/V element in the hot loop
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short a_f2h(float f) {
  // native fp32->fp16 round-to-nearest-even (same rounding as the software
  // path -> bit-identical outputs)
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}

__global__ void attn_core_il_fp16(const unsigned short *q, const unsigned short *k,
                                  const unsigned short *v, unsigned short *o,
                                  int HQ, int HKV, int N_q, int N_kv,
                                  int cache_from, int d, int window, float softcap,
                                  const int *d_pos, int ring_cap) {
  int i = blockIdx.x, h = blockIdx.y;
  // M2-B: when d_pos is bound, read the live position/key-count from the device
  // buffer so a captured graph reads the new token's state on replay (else use
  // the baked int args = original non-graph behaviour).
  int cf = d_pos ? d_pos[0] : cache_from;
  int nkv = d_pos ? d_pos[1] : N_kv;
  if (i >= N_q || h >= HQ) return;
  int gqa = HQ / HKV, hkv = h / gqa;
  int HD_Q = HQ * d, HD_KV = HKV * d;
  const unsigned short *qrow = q + (long)i * HD_Q + (long)h * d;
  unsigned short *orow = o + (long)i * HD_Q + (long)h * d;
  int tid = threadIdx.x, B = blockDim.x;
  extern __shared__ float sh[];
  float *Qsh = sh; float *red = sh + d;
  for (int dd = tid; dd < d; dd += B) Qsh[dd] = a_h2f(qrow[dd]);
  __syncthreads();
  float scale = rsqrtf((float)d);
  float acc[4];
#pragma unroll
  for (int r=0;r<4;r++) acc[r]=0.f;
  int i_abs = cf + i;
  int j_lo = i_abs - window + 1; if (j_lo<0) j_lo=0;
  int j_hi = i_abs; if (j_hi>=nkv) j_hi=nkv-1;
  float mmax=-1e30f, l=0.f;
  for (int j=j_lo;j<=j_hi;++j) {
    // [kv-window-ring] physical cache row = j % ring_cap (ring_cap<=0: linear).
    long pj = (ring_cap > 0) ? (long)(j % ring_cap) : (long)j;
    const unsigned short *kr = k + pj*HD_KV + (long)hkv*d;
    float pd=0.f;
    for (int dd=tid;dd<d;dd+=B) pd += Qsh[dd]*a_h2f(kr[dd]);
    red[tid]=pd; __syncthreads();
    for (int s=B>>1;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
    float score=red[0]*scale; __syncthreads();
    if (softcap>0.f) score=softcap*tanhf(score/softcap);
    float mn=fmaxf(mmax,score), corr=__expf(mmax-mn), p=__expf(score-mn);
    l=l*corr+p; mmax=mn;
    const unsigned short *vr = v + pj*HD_KV + (long)hkv*d;
    int r=0; for (int dd=tid;dd<d;dd+=B,++r) acc[r]=acc[r]*corr + p*a_h2f(vr[dd]);
  }
  float inv = l>0.f?1.f/l:0.f;
  int r=0; for (int dd=tid;dd<d;dd+=B,++r) orow[dd]=a_f2h(acc[r]*inv);
}

}
)CU";

namespace {
// device mirror of a host-resident KV cache (keyed by host pointer). The cache
// (cache_key/cache_value) is a MAX_LIFESPAN tensor that is NOT UVM-resident on
// engine=cuda, so a device kernel can't read it directly; mirror it (small:
// num_kv_heads=1). Re-copied each call (the cache grows) -- correct, and cheap
// for the per-layer cache size.
struct DevKV {
  unsigned short *buf = nullptr;
  size_t cap = 0;
};
std::unordered_map<const void *, DevKV> g_kv_mirror;
std::mutex g_kv_mtx;

const unsigned short *mirror_kv(const unsigned short *host, size_t elems) {
  // Integrated GPU (Tegra/Jetson Orin): the iGPU reads ordinary host memory
  // directly (one shared physical pool), so NO device mirror is needed. Passing
  // through also avoids a latent stale-KV bug -- the mirror snapshots the host
  // cache, and a host-side V-copy that updates the slot AFTER the snapshot
  // would be served stale keys/values. Discrete GPUs keep the mirror.
  static const bool integrated = ContextManager::Global().isIntegrated();
  if (integrated)
    return host;
  const bool dev = nntrainer::cuda::dev_accessible(host);
  if (dev)
    return host; // already device-accessible
  std::lock_guard<std::mutex> lk(g_kv_mtx);
  auto &e = g_kv_mirror[host];
  size_t bytes = elems * sizeof(unsigned short);
  if (bytes > e.cap) {
    if (e.buf)
      cudaFree(e.buf);
    if (cudaMalloc(&e.buf, bytes) != cudaSuccess) {
      e.buf = nullptr;
      e.cap = 0;
      return nullptr;
    }
    e.cap = bytes;
  }
  cudaMemcpy(e.buf, host, bytes, cudaMemcpyHostToDevice);
  return e.buf;
}
} // namespace

// Flash-decoding (split-KV) for M=1 decode: the single-pass kernel launches
// only num_heads blocks (8 for gemma4) -- it underutilizes the SMs and
// serializes the long KV loop. Split the KV axis into chunks so
// num_heads*n_chunks blocks run a partial online-softmax in parallel, then a
// small reduce combines the chunks.
static const char *ATTN_SPLITKV_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float s_h2f(unsigned short h) {
  // native fp16->fp32 (exact); replaces the software bit-twiddle -- this runs
  // per Q/K/V element in the hot loop
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short s_f2h(float f) {
  // native fp32->fp16 round-to-nearest-even (same rounding as the software
  // path -> bit-identical outputs)
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
// One block per (head h, chunk c); query row is 0 (decode M=1). Online softmax
// over the chunk's keys; writes (m, l, acc[d]) to scratch[h*n_chunks + c].
// [splitkv-clip] chunk-grid anchor. Without clipping chunk c always covers the
// absolute keys [c*chunk_kv, ...), so a sliding layer launches a chunk for every
// key ever cached and all but the last few are empty (29K context, 1K window:
// 457 chunks, 8 of them live) -- the empty ones still cost a block and, worse,
// a pass through the reduce loop. Anchoring the grid at the chunk that contains
// the window's low bound drops them. The anchor stays a multiple of chunk_kv so
// every surviving chunk covers EXACTLY the key set it covered before: the
// per-chunk (m, l, acc) triples are unchanged and the dropped ones are the
// softmax identity (m=-inf, l=0, acc=0), which contributes exactly 0.0f to the
// merge -- the output is bit-identical.
__device__ __forceinline__ int sk_jbase(int cf, int window, int chunk_kv,
                                        int clip) {
  if (!clip) return 0;
  int j_lo_g = cf - window + 1; if (j_lo_g < 0) j_lo_g = 0;
  return (j_lo_g / chunk_kv) * chunk_kv;
}
__global__ void attn_partial(const unsigned short *q, const unsigned short *k,
                             const unsigned short *v, float *pm, float *pl,
                             float *pacc, int HQ, int HKV, int N_kv,
                             int cache_from, int d, int window, float softcap,
                             int chunk_kv, int n_chunks,
                             const int *d_pos, int max_n_chunks, int ring_cap,
                             int clip) {
  int h = blockIdx.x, c = blockIdx.y;
  // M2-B: read the live query position / key-count from the device d_pos buffer
  // (mirrors attn_core_il_fp16) so ONE captured graph is valid for every token.
  // The grid is captured at gridDim.y=max_n_chunks; blocks past the live chunk
  // count early-return without writing, and the partial->reduce stride uses the
  // FIXED max_n_chunks so writer and reader always agree.
  int cf = d_pos ? d_pos[0] : cache_from;
  int nkv = d_pos ? d_pos[1] : N_kv;
  int i_abs = cf; // i=0
  int j_lo_g = i_abs - window + 1; if (j_lo_g < 0) j_lo_g = 0;
  int j_hi_g = i_abs; if (j_hi_g >= nkv) j_hi_g = nkv - 1;
  int j_base = sk_jbase(cf, window, chunk_kv, clip);
  int live_nchunks = clip ? ((j_hi_g >= j_base)
                               ? ((j_hi_g - j_base) / chunk_kv + 1) : 0)
                          : ((nkv + chunk_kv - 1) / chunk_kv);
  if (c >= live_nchunks) return;
  int gqa = HQ / HKV, hkv = h / gqa;
  int HD_KV = HKV * d;
  const unsigned short *qrow = q + (long)h * d; // i=0
  int tid = threadIdx.x, B = blockDim.x;
  extern __shared__ float sh[];
  float *Qsh = sh; float *red = sh + d;
  for (int dd = tid; dd < d; dd += B) Qsh[dd] = s_h2f(qrow[dd]);
  __syncthreads();
  float scale = rsqrtf((float)d);
  int j_lo = j_base + c * chunk_kv; if (j_lo < j_lo_g) j_lo = j_lo_g;
  int j_hi = j_base + (c + 1) * chunk_kv - 1; if (j_hi > j_hi_g) j_hi = j_hi_g;
  float acc[4];
#pragma unroll
  for (int r = 0; r < 4; r++) acc[r] = 0.f;
  float mmax = -1e30f, l = 0.f;
  for (int j = j_lo; j <= j_hi; ++j) {
    // [kv-window-ring] physical cache row = j % ring_cap (ring_cap<=0: linear).
    long pj = (ring_cap > 0) ? (long)(j % ring_cap) : (long)j;
    const unsigned short *kr = k + pj * HD_KV + (long)hkv * d;
    float pd = 0.f;
    for (int dd = tid; dd < d; dd += B) pd += Qsh[dd] * s_h2f(kr[dd]);
    red[tid] = pd; __syncthreads();
    for (int s = B >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
    float score = red[0] * scale; __syncthreads();
    if (softcap > 0.f) score = softcap * tanhf(score / softcap);
    float mn = fmaxf(mmax, score), corr = __expf(mmax - mn), p = __expf(score - mn);
    l = l * corr + p; mmax = mn;
    const unsigned short *vr = v + pj * HD_KV + (long)hkv * d;
    int r = 0; for (int dd = tid; dd < d; dd += B, ++r) acc[r] = acc[r]*corr + p*s_h2f(vr[dd]);
  }
  if (j_lo > j_hi) { mmax = -1e30f; l = 0.f; }
  int base = h * max_n_chunks + c;
  if (tid == 0) { pm[base] = mmax; pl[base] = l; }
  int r = 0; for (int dd = tid; dd < d; dd += B, ++r) pacc[(long)base * d + dd] = acc[r];
}
// One block per head; combine the n_chunks partials into the fp16 output row.
__global__ void attn_reduce(const float *pm, const float *pl, const float *pacc,
                            unsigned short *o, int HQ, int d, int n_chunks,
                            const int *d_pos, int chunk_kv, int max_n_chunks,
                            int window, int cache_from, int N_kv, int clip) {
  int h = blockIdx.x;
  int tid = threadIdx.x, B = blockDim.x;
  // M2-B: live chunk count from d_pos; FIXED max_n_chunks stride to match the
  // partial writer (must be the SAME constant the scratch was sized with).
  // [splitkv-clip] when clipping, re-derive the live count from the SAME
  // anchored grid the partial writer used (identical expression).
  int cf = d_pos ? d_pos[0] : cache_from;
  int nkv = d_pos ? d_pos[1] : (clip ? N_kv : (n_chunks * chunk_kv));
  int live_nchunks;
  if (clip) {
    int j_lo_g = cf - window + 1; if (j_lo_g < 0) j_lo_g = 0;
    int j_hi_g = cf; if (j_hi_g >= nkv) j_hi_g = nkv - 1;
    int j_base = sk_jbase(cf, window, chunk_kv, clip);
    live_nchunks = (j_hi_g >= j_base) ? ((j_hi_g - j_base) / chunk_kv + 1) : 0;
  } else {
    live_nchunks = d_pos ? (nkv + chunk_kv - 1) / chunk_kv : n_chunks;
  }
  int base = h * max_n_chunks;
  __shared__ float M, L;
  if (tid == 0) {
    float mx = -1e30f;
    for (int c = 0; c < live_nchunks; ++c) mx = fmaxf(mx, pm[base + c]);
    float l = 0.f;
    for (int c = 0; c < live_nchunks; ++c) l += pl[base + c] * __expf(pm[base + c] - mx);
    M = mx; L = l;
  }
  __syncthreads();
  float inv = L > 0.f ? 1.f / L : 0.f;
  unsigned short *orow = o + (long)h * d; // i=0
  for (int dd = tid; dd < d; dd += B) {
    float a = 0.f;
    for (int c = 0; c < live_nchunks; ++c)
      a += pacc[((long)(base + c)) * d + dd] * __expf(pm[base + c] - M);
    orow[dd] = s_f2h(a * inv);
  }
}
// [reduce-tile] Wide-grid form of attn_reduce. The kernel above runs ONE block
// per head and does the (m, l) merge in thread 0, so at 29K context 8 blocks --
// 8 of the device's SMs -- walk 454 chunks, the first ~900 of those steps being
// a single thread's dependent chain. Measured 35 GB/s on a 384 GB/s part.
//
// Three changes, none of which touches a summation order:
//  - the grid gains a dim-tile axis, so every output dim gets its own thread
//    and the block count scales with head_dim (each dim's chunk sum is
//    independent, so which thread owns it is irrelevant);
//  - the running max becomes a block tree-reduction (fmaxf is associative and
//    commutative, so the result is the same float);
//  - the per-chunk softmax weights are computed ONCE into shared memory instead
//    of once per (dim, chunk); the serial l-sum then reads them, which takes
//    the exp out of its dependent chain while keeping its chunk order.
// The l-sum and the per-dim accumulation still run in the original chunk order,
// so the output is bit-identical. Requires (B + max_n_chunks) floats of shared
// memory; the host falls back to attn_reduce when that does not fit.
__global__ void attn_reduce_t(const float *pm, const float *pl,
                              const float *pacc, unsigned short *o, int HQ,
                              int d, int n_chunks, const int *d_pos,
                              int chunk_kv, int max_n_chunks, int window,
                              int cache_from, int N_kv, int clip) {
  int h = blockIdx.x;
  int tid = threadIdx.x, B = blockDim.x;
  int dd = blockIdx.y * B + tid;
  // live chunk count: identical expression to attn_reduce (see there).
  int cf = d_pos ? d_pos[0] : cache_from;
  int nkv = d_pos ? d_pos[1] : (clip ? N_kv : (n_chunks * chunk_kv));
  int live_nchunks;
  if (clip) {
    int j_lo_g = cf - window + 1; if (j_lo_g < 0) j_lo_g = 0;
    int j_hi_g = cf; if (j_hi_g >= nkv) j_hi_g = nkv - 1;
    int j_base = sk_jbase(cf, window, chunk_kv, clip);
    live_nchunks = (j_hi_g >= j_base) ? ((j_hi_g - j_base) / chunk_kv + 1) : 0;
  } else {
    live_nchunks = d_pos ? (nkv + chunk_kv - 1) / chunk_kv : n_chunks;
  }
  int base = h * max_n_chunks;
  extern __shared__ float sh[];
  float *red = sh;      // [B] tree-reduction scratch
  float *ew = sh + B;   // [live_nchunks] staged softmax weights
  float loc = -1e30f;
  for (int c = tid; c < live_nchunks; c += B) loc = fmaxf(loc, pm[base + c]);
  red[tid] = loc; __syncthreads();
  for (int s = B >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  const float M = red[0];
  for (int c = tid; c < live_nchunks; c += B) ew[c] = __expf(pm[base + c] - M);
  __syncthreads();
  __shared__ float L;
  // one thread, ORIGINAL chunk order; the other warps run the accumulation
  // below meanwhile and only need L at the final scale.
  if (tid == 0) {
    float l = 0.f;
    for (int c = 0; c < live_nchunks; ++c) l += pl[base + c] * ew[c];
    L = l;
  }
  float a = 0.f;
  if (dd < d)
    for (int c = 0; c < live_nchunks; ++c)
      a += pacc[((long)(base + c)) * d + dd] * ew[c];
  __syncthreads();
  const float inv = L > 0.f ? 1.f / L : 0.f;
  if (dd < d) o[(long)h * d + dd] = s_f2h(a * inv);
}
}
)CU";

// Warp-level split-KV decode partial (replaces attn_partial above).
//
// attn_partial reduces EVERY key's d-dot through a 128-way shared-memory tree:
// 7 __syncthreads for the tree plus 2 more around the scratch reuse, i.e. ~9
// block barriers per key, with only ONE key in flight per block. nsys measured
// it at ~7 GB/s effective on a 384 GB/s part (1/40 of peak) -- the kernel is
// barrier/latency bound, not bandwidth bound, which is also why shrinking the
// chunk (NNTR_CUDA_FLASH_DECODE=16) sped decode up: it only shortened the
// serial dependent chain.
//
// Here the block's 4 warps each own a stride-NW subset of the chunk's keys and
// run an INDEPENDENT register online-softmax. The per-key d-dot is one
// __shfl_xor butterfly (lane owns VPL = head_dim/32 CONTIGUOUS dims, loaded
// with uint2/uint4 vector loads like the split-prefill kernel), so the key loop
// contains ZERO barriers and NW keys are in flight per block. A single
// __syncthreads at the end merges the NW warp partials, in fixed warp order, to
// exactly the (m, l, acc) triple attn_partial wrote -- so the scratch layout
// (h * max_n_chunks + c), the M2-B d_pos/live-chunk contract and attn_reduce
// are all untouched, and the launch geometry (grid HQ x max_n_chunks, block
// 128) is bit-for-bit the same under graph capture.
//
// Numerics: same online-softmax, same fp32 accumulators, same key order per
// chunk. Only the summation ORDER of the d-dot (32 lane partials of VPL
// contiguous dims vs 128 thread partials of stride-128 dims) and the extra
// per-warp renormalisation differ, so fp16 outputs may move by a rounding ulp.
static const char *ATTN_SPLITKV_WARP_SRC = R"CU(
__device__ __forceinline__ float wk_h2f(unsigned short h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ float wk_wreduce(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}
// [splitkv-clip] chunk-grid anchor, identical to attn_partial's sk_jbase.
__device__ __forceinline__ int wk_jbase(int cf, int window, int chunk_kv,
                                        int clip) {
  if (!clip) return 0;
  int j_lo_g = cf - window + 1; if (j_lo_g < 0) j_lo_g = 0;
  return (j_lo_g / chunk_kv) * chunk_kv;
}
// Vectorized fp16 row-slice load: uint2 (8B) for VPL=4, uint4 (16B) above.
// The host dispatch verifies the base alignment and falls back to attn_partial
// otherwise.
template <int VPL>
__device__ __forceinline__ void wk_ldrow(const unsigned short *p,
                                         unsigned short *h) {
  if (VPL == 4) {
    uint2 w = *(const uint2 *)p;
    h[0] = (unsigned short)(w.x); h[1] = (unsigned short)(w.x >> 16);
    h[2] = (unsigned short)(w.y); h[3] = (unsigned short)(w.y >> 16);
  } else {
#pragma unroll
    for (int p4 = 0; p4 < VPL / 8; p4++) {
      uint4 w = *(const uint4 *)(p + p4 * 8);
      h[p4*8+0]=(unsigned short)(w.x); h[p4*8+1]=(unsigned short)(w.x>>16);
      h[p4*8+2]=(unsigned short)(w.y); h[p4*8+3]=(unsigned short)(w.y>>16);
      h[p4*8+4]=(unsigned short)(w.z); h[p4*8+5]=(unsigned short)(w.z>>16);
      h[p4*8+6]=(unsigned short)(w.w); h[p4*8+7]=(unsigned short)(w.w>>16);
    }
  }
}
// [gqa-fuse] HPW = query heads handled by ONE warp, KPI = keys staged per loop
// trip. With grouped-query attention every query head of a group reads the SAME
// K/V rows, so the HPW=1 grid (one block per query head) fetches each row gqa
// times: at 29K context that is 8 x 396 MB/token of L2 traffic for gemma4's
// full layers, and the fp16->fp32 converts are paid gqa times too. Folding HPW
// heads into one warp loads and converts the row ONCE and runs HPW independent
// register online-softmaxes over it, so both the re-reads and the converts drop
// by HPW. KPI stages KPI keys' loads before any of them is consumed, which is
// what gets the memory pipe and the ALU to overlap (measured: the HPW=1 kernel
// costs almost exactly stream-time PLUS alu-time, i.e. no overlap at all).
//
// Numerics: each (head, key) still runs the same per-lane dot over the same
// contiguous VPL dims, the same butterfly, the same online-softmax over the
// same per-warp key subsequence, and the same fixed-warp-order merge. Fusing
// only changes WHICH warp does the arithmetic, never the arithmetic or its
// order -- outputs are BIT-IDENTICAL to HPW=1 (verified standalone over the
// d=512/gqa=8 and d=128/gqa=6 shapes). The host still keeps a
// kill switch (NNTR_CUDA_ATTN_FUSE=0) that restores the HPW=1 launch.
//
// Constraint: the fused heads must share one KV head, so HPW must divide gqa
// (the host checks this); hkv is then h0/gqa for the whole group.
template <int VPL, int NW, int HPW, int KPI>
__device__ __forceinline__ void
splitkv_warp_body(const unsigned short *q, const unsigned short *k,
                  const unsigned short *v, float *pm, float *pl, float *pacc,
                  int HQ, int HKV, int N_kv, int cache_from, int d, int window,
                  float softcap, int chunk_kv, int n_chunks, const int *d_pos,
                  int max_n_chunks, int ring_cap, int clip) {
  const int h = blockIdx.x * HPW, c = blockIdx.y;
  // M2-B: live query position / key count from the device pos buffer; blocks
  // past the live chunk count early-return without writing (identical gate to
  // attn_partial, and the partial->reduce stride stays the FIXED max_n_chunks).
  const int cf = d_pos ? d_pos[0] : cache_from;
  const int nkv = d_pos ? d_pos[1] : N_kv;
  const int i_abs = cf; // i = 0 (decode M=1)
  int j_lo_g = i_abs - window + 1; if (j_lo_g < 0) j_lo_g = 0;
  int j_hi_g = i_abs; if (j_hi_g >= nkv) j_hi_g = nkv - 1;
  // [splitkv-clip] anchor the chunk grid at the window's chunk-aligned low
  // bound (see attn_partial): same key set per surviving chunk, so the merged
  // output is bit-identical -- only the empty chunks disappear.
  const int j_base = wk_jbase(cf, window, chunk_kv, clip);
  const int live_nchunks =
    clip ? ((j_hi_g >= j_base) ? ((j_hi_g - j_base) / chunk_kv + 1) : 0)
         : ((nkv + chunk_kv - 1) / chunk_kv);
  if (c >= live_nchunks) return;
  const int gqa = HQ / HKV, hkv = h / gqa;
  const int HD_KV = HKV * d;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int lane0 = lane * VPL;
  const float scale = rsqrtf((float)d);
  int j_lo = j_base + c * chunk_kv; if (j_lo < j_lo_g) j_lo = j_lo_g;
  int j_hi = j_base + (c + 1) * chunk_kv - 1; if (j_hi > j_hi_g) j_hi = j_hi_g;

  float q_reg[HPW][VPL], acc[HPW][VPL], mmax[HPW], l[HPW];
#pragma unroll
  for (int e = 0; e < HPW; ++e) {
    unsigned short qh[VPL];
    wk_ldrow<VPL>(q + (long)(h + e) * d + lane0, qh);
#pragma unroll
    for (int vv = 0; vv < VPL; vv++) {
      q_reg[e][vv] = wk_h2f(qh[vv]); acc[e][vv] = 0.f;
    }
    mmax[e] = -1e30f; l[e] = 0.f;
  }
  // warp w walks keys j_lo+w, j_lo+w+NW, ... : NW*KPI keys in flight, no
  // barriers. The per-warp key subsequence (and therefore the online-softmax
  // order) is identical to the KPI=1 form.
  for (int j = j_lo + w; j <= j_hi; j += NW * KPI) {
    unsigned short kh[KPI][VPL], vh[KPI][VPL];
    int nstaged = 0;
#pragma unroll
    for (int t = 0; t < KPI; ++t) {
      const int jj = j + t * NW;
      if (jj <= j_hi) {
        // [kv-window-ring] physical row = j % ring_cap (ring_cap<=0: linear).
        long pj = (ring_cap > 0) ? (long)(jj % ring_cap) : (long)jj;
        long base = pj * HD_KV + (long)hkv * d + lane0;
        wk_ldrow<VPL>(k + base, kh[t]);
        wk_ldrow<VPL>(v + base, vh[t]);
        ++nstaged;
      }
    }
#pragma unroll
    for (int t = 0; t < KPI; ++t) {
      if (t >= nstaged) break;
      float kf[VPL], vf[VPL];
#pragma unroll
      for (int vv = 0; vv < VPL; vv++) {
        kf[vv] = wk_h2f(kh[t][vv]); vf[vv] = wk_h2f(vh[t][vv]);
      }
#pragma unroll
      for (int e = 0; e < HPW; ++e) {
        float p = 0.f;
#pragma unroll
        for (int vv = 0; vv < VPL; vv++) p += q_reg[e][vv] * kf[vv];
        float score = wk_wreduce(p) * scale;
        if (softcap > 0.f) score = softcap * tanhf(score / softcap);
        float mn = fmaxf(mmax[e], score), corr = __expf(mmax[e] - mn),
              pp = __expf(score - mn);
        l[e] = l[e] * corr + pp; mmax[e] = mn;
#pragma unroll
        for (int vv = 0; vv < VPL; vv++)
          acc[e][vv] = acc[e][vv] * corr + pp * vf[vv];
      }
    }
  }

  // merge the NW warp partials in FIXED warp order (deterministic). Empty
  // warps carry (-1e30, 0, 0) and weight to exactly zero.
  extern __shared__ float sh[];
  float *sacc = sh;                          // [HPW][NW][d]
  float *sm = sh + (long)HPW * NW * d;       // [HPW][NW]
  float *sl = sm + HPW * NW;                 // [HPW][NW]
#pragma unroll
  for (int e = 0; e < HPW; ++e) {
    if (lane == 0) { sm[e * NW + w] = mmax[e]; sl[e * NW + w] = l[e]; }
#pragma unroll
    for (int vv = 0; vv < VPL; vv++)
      sacc[((long)e * NW + w) * d + lane0 + vv] = acc[e][vv];
  }
  __syncthreads();
  const int B = blockDim.x;
#pragma unroll
  for (int e = 0; e < HPW; ++e) {
    float M = -1e30f;
#pragma unroll
    for (int t = 0; t < NW; ++t) M = fmaxf(M, sm[e * NW + t]);
    float ew[NW];
#pragma unroll
    for (int t = 0; t < NW; ++t) ew[t] = __expf(sm[e * NW + t] - M);
    const int obase = (h + e) * max_n_chunks + c;
    if (tid == 0) {
      float L = 0.f;
#pragma unroll
      for (int t = 0; t < NW; ++t) L += sl[e * NW + t] * ew[t];
      pm[obase] = M; pl[obase] = L;
    }
    for (int dd = tid; dd < d; dd += B) {
      float a = 0.f;
#pragma unroll
      for (int t = 0; t < NW; ++t) a += sacc[((long)e * NW + t) * d + dd] * ew[t];
      pacc[(long)obase * d + dd] = a;
    }
  }
}
#define SPLITKV_WARP_ENTRY(NAME, VPL, HPW, KPI)                                \
  extern "C" __global__ void NAME(                                             \
    const unsigned short *q, const unsigned short *k, const unsigned short *v, \
    float *pm, float *pl, float *pacc, int HQ, int HKV, int N_kv,              \
    int cache_from, int d, int window, float softcap, int chunk_kv,            \
    int n_chunks, const int *d_pos, int max_n_chunks, int ring_cap, int clip) {\
    splitkv_warp_body<VPL, 4, HPW, KPI>(q, k, v, pm, pl, pacc, HQ, HKV, N_kv,  \
                                        cache_from, d, window, softcap,        \
                                        chunk_kv, n_chunks, d_pos,             \
                                        max_n_chunks, ring_cap, clip);         \
  }
// HPW=1: the unfused launch, kept as the kill-switch / narrow-grid path.
SPLITKV_WARP_ENTRY(attn_partial_w128, 4, 1, 1)
SPLITKV_WARP_ENTRY(attn_partial_w256, 8, 1, 1)
SPLITKV_WARP_ENTRY(attn_partial_w512, 16, 1, 1)
// [gqa-fuse] fused launches. KPI=4 throughout (measured best or within noise of
// best on every shape); HPW is capped per head_dim by the register budget --
// d=512 needs VPL=16 registers per head, so HPW>4 there spills.
SPLITKV_WARP_ENTRY(attn_partial_w128_h2, 4, 2, 4)
SPLITKV_WARP_ENTRY(attn_partial_w128_h3, 4, 3, 4)
SPLITKV_WARP_ENTRY(attn_partial_w128_h4, 4, 4, 4)
SPLITKV_WARP_ENTRY(attn_partial_w128_h6, 4, 6, 4)
SPLITKV_WARP_ENTRY(attn_partial_w256_h2, 8, 2, 4)
SPLITKV_WARP_ENTRY(attn_partial_w256_h3, 8, 3, 4)
SPLITKV_WARP_ENTRY(attn_partial_w256_h4, 8, 4, 4)
SPLITKV_WARP_ENTRY(attn_partial_w256_h6, 8, 6, 4)
SPLITKV_WARP_ENTRY(attn_partial_w512_h2, 16, 2, 4)
SPLITKV_WARP_ENTRY(attn_partial_w512_h4, 16, 4, 4)
)CU";

// Block-Q multi-row prefill attention (CUDA mirror of the Intel OpenCL
// flash_attention_prefill_f16_blockq + FBQ_SG kernel). One WARP (32 lanes) owns
// a tile of TM query rows of one head; lane owns VPL = head_dim/32 CONTIGUOUS
// head dims so the K/V/Q loads are coalesced. Per key: the full d-dot is a
// single warp butterfly all-reduce (__shfl_xor, NO __syncthreads / shared mem),
// and K[n]/V[n] are loaded ONCE and reused across all TM rows (register
// online-softmax). Replaces attn_core_il_fp16's per-key 128-way LDS tree-reduce
// (7 __syncthreads x #keys, only 1 key in flight). Measured 3-4x faster on
// gemma4 shapes, fp16-identical output. GQA via hkv = head_q / (HQ/HKV).
static const char *ATTN_BLOCKQ_SRC = R"CU(
__device__ __forceinline__ float bq_h2f(unsigned short h) {
  // native fp16->fp32 (exact); replaces the software bit-twiddle -- this runs
  // per Q/K/V element in the hot loop
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short bq_f2h(float f) {
  // native fp32->fp16 round-to-nearest-even (same rounding as the software
  // path -> bit-identical outputs)
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
__device__ __forceinline__ float bq_wreduce(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}
template <int TM, int VPL>
__device__ __forceinline__ void
blockq_body(const unsigned short *q, const unsigned short *k,
            const unsigned short *v, unsigned short *o, int HQ, int HKV, int N_q,
            int N_kv, int cache_from, int d, int window, float softcap,
            int ring_cap) { // [kv-window-ring] >0: K/V physical row = n%ring_cap
  const int lane = threadIdx.x;             // 0..31
  const int grp = blockIdx.x;
  const int n_row_tiles = (N_q + TM - 1) / TM;
  const int head_q = grp / n_row_tiles;
  const int tile = grp % n_row_tiles;
  const int m0 = tile * TM;
  if (head_q >= HQ || m0 >= N_q) return;
  const int gqa = HQ / HKV, hkv = head_q / gqa;
  const int HD_Q = HQ * d, HD_KV = HKV * d;
  const float scale = rsqrtf((float)d);
  const int lane0 = lane * VPL;
  float q_reg[TM][VPL], acc_reg[TM][VPL], m_i[TM], l_i[TM];
  int valid[TM];
#pragma unroll
  for (int r = 0; r < TM; r++) {
    int m = m0 + r; valid[r] = (m < N_q) ? 1 : 0; m_i[r] = -1e30f; l_i[r] = 0.f;
    long q_base = (long)(valid[r] ? m : 0) * HD_Q + (long)head_q * d;
#pragma unroll
    for (int vv = 0; vv < VPL; vv++) {
      q_reg[r][vv] = bq_h2f(q[q_base + lane0 + vv]); acc_reg[r][vv] = 0.f;
    }
  }
  const int q_pos_off = cache_from;          // absolute query pos = m0+r+cache_from
  int last_row = ((m0 + TM - 1 < N_q) ? (m0 + TM - 1) : (N_q - 1)) + q_pos_off;
  int n_last = (N_kv - 1 < last_row) ? (N_kv - 1) : last_row;   // causal
  // Sliding-window tile-union lower bound (mirror of the CL blockq n_lo fix):
  // keys below row 0's window floor (n + window <= m0 + q_pos_off) are masked
  // for EVERY row in the tile (the floor only grows with r), so start the
  // walk there instead of paying K-load + dot + warp-reduce + V-load per
  // skipped key. Bit-identical by construction; without this the 16 sliding
  // prefill layers do O(M*N) key-visits instead of O(M*W).
  int n_lo = (window > 0) ? (m0 + q_pos_off - window + 1) : 0;
  if (n_lo < 0) n_lo = 0;
  for (int n = n_lo; n <= n_last; ++n) {
    // [kv-window-ring] physical cache row = n % ring_cap (ring_cap<=0: linear).
    long pn = (ring_cap > 0) ? (long)(n % ring_cap) : (long)n;
    long k_base = pn * HD_KV + (long)hkv * d;
    float k_reg[VPL];
#pragma unroll
    for (int vv = 0; vv < VPL; vv++) k_reg[vv] = bq_h2f(k[k_base + lane0 + vv]);
    float sdot[TM];
#pragma unroll
    for (int r = 0; r < TM; r++) {
      float p = 0.f;
#pragma unroll
      for (int vv = 0; vv < VPL; vv++) p += q_reg[r][vv] * k_reg[vv];
      sdot[r] = bq_wreduce(p);
    }
    long v_base = pn * HD_KV + (long)hkv * d;
    float v_reg[VPL];
#pragma unroll
    for (int vv = 0; vv < VPL; vv++) v_reg[vv] = bq_h2f(v[v_base + lane0 + vv]);
#pragma unroll
    for (int r = 0; r < TM; r++) {
      int m = m0 + r + q_pos_off;
      if (!valid[r] || n > m || (window > 0 && n + window <= m)) continue;
      float s = scale * sdot[r];
      if (softcap > 0.f) s = softcap * tanhf(s / softcap);
      float m_new = fmaxf(m_i[r], s), alpha = __expf(m_i[r] - m_new),
            pp = __expf(s - m_new);
#pragma unroll
      for (int vv = 0; vv < VPL; vv++)
        acc_reg[r][vv] = alpha * acc_reg[r][vv] + pp * v_reg[vv];
      l_i[r] = alpha * l_i[r] + pp; m_i[r] = m_new;
    }
  }
#pragma unroll
  for (int r = 0; r < TM; r++) {
    if (!valid[r]) continue;
    float inv = l_i[r] > 0.f ? 1.f / l_i[r] : 0.f;
    long o_base = (long)(m0 + r) * HD_Q + (long)head_q * d;
#pragma unroll
    for (int vv = 0; vv < VPL; vv++)
      o[o_base + lane0 + vv] = bq_f2h(acc_reg[r][vv] * inv);
  }
}
extern "C" __global__ void
attn_blockq_d256(const unsigned short *q, const unsigned short *k,
                 const unsigned short *v, unsigned short *o, int HQ, int HKV,
                 int N_q, int N_kv, int cache_from, int d, int window,
                 float softcap, int ring_cap) {
  blockq_body<4, 8>(q, k, v, o, HQ, HKV, N_q, N_kv, cache_from, d, window,
                    softcap, ring_cap);
}
extern "C" __global__ void
attn_blockq_d512(const unsigned short *q, const unsigned short *k,
                 const unsigned short *v, unsigned short *o, int HQ, int HKV,
                 int N_q, int N_kv, int cache_from, int d, int window,
                 float softcap, int ring_cap) {
  blockq_body<4, 16>(q, k, v, o, HQ, HKV, N_q, N_kv, cache_from, d, window,
                     softcap, ring_cap);
}
extern "C" __global__ void
attn_blockq_d128(const unsigned short *q, const unsigned short *k,
                 const unsigned short *v, unsigned short *o, int HQ, int HKV,
                 int N_q, int N_kv, int cache_from, int d, int window,
                 float softcap, int ring_cap) {
  blockq_body<4, 4>(q, k, v, o, HQ, HKV, N_q, N_kv, cache_from, d, window,
                    softcap, ring_cap);
}
)CU";

// Split-KV chunked-prefill attention (global/full-attention layers, long K).
// The serial blockq_body launches ONE fixed 32-lane warp per (head, TM-row
// tile) that walks the ENTIRE causal K range; once the growing KV exceeds L2
// (ctx ~9K on gemma4) every warp streams K/V from DRAM independently and the
// per-key cost plateaus ~2x higher. Split the key axis into a FIXED,
// deterministic partition (ceil(N_kv/split_len), capped at 32): each block
// owns (tile, split) and runs the identical register online-softmax over its
// sub-range, writing partial (m, l, acc[d]) to scratch; a second kernel
// combines the partials per row in FIXED split order (serial loop, no
// atomics -> byte-stable run to run). Same-split blocks are adjacent on the
// fast grid axis so they walk the same K window together (L2-aligned
// frontier). Numerics: exact-arithmetic-equal to the serial kernel; fp32
// renormalization ORDER differs (per-split then combine vs one running
// pass), so fp16 outputs may differ by rounding -- gated by the golden runs.
static const char *ATTN_BLOCKQ_SPLIT_SRC = R"CU(
__device__ __forceinline__ float bs_h2f(unsigned short h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short bs_f2h(float f) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
__device__ __forceinline__ float bs_wreduce(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}
// Vectorized K/V row-slice load (uint2 for VPL=4, uint4 pairs above). The
// serial kernel's 16 scalar u16 loads per slice make the per-key dependent
// chain latency-dominated (probe: the whole kernel is latency-bound at ~8
// warps/SM, NOT bandwidth-bound -- a 4x shared-memory traffic cut measured
// flat). Wide loads + the launch-bounds occupancy floor bought 1.9x on d512
// in the standalone probe, with bit-identical partials (loads only; the FP
// arithmetic order is untouched). Requires an 8B (VPL=4) / 16B (VPL>=8)
// aligned K/V base -- verified by the host dispatch, which otherwise falls
// back to the serial kernel.
template <int VPL>
__device__ __forceinline__ void bs_ldrow(const unsigned short *p,
                                         unsigned short *h) {
  if (VPL == 4) {
    uint2 w = *(const uint2 *)p;
    h[0] = (unsigned short)(w.x); h[1] = (unsigned short)(w.x >> 16);
    h[2] = (unsigned short)(w.y); h[3] = (unsigned short)(w.y >> 16);
  } else {
#pragma unroll
    for (int p4 = 0; p4 < VPL / 8; p4++) {
      uint4 w = *(const uint4 *)(p + p4 * 8);
      h[p4*8+0]=(unsigned short)(w.x); h[p4*8+1]=(unsigned short)(w.x>>16);
      h[p4*8+2]=(unsigned short)(w.y); h[p4*8+3]=(unsigned short)(w.y>>16);
      h[p4*8+4]=(unsigned short)(w.z); h[p4*8+5]=(unsigned short)(w.z>>16);
      h[p4*8+6]=(unsigned short)(w.w); h[p4*8+7]=(unsigned short)(w.w>>16);
    }
  }
}
// Partial pass: block (blockIdx.x = slab-local tile, blockIdx.y = split sp).
// Body identical to blockq_body except the key walk is clipped to split sp's
// sub-range [sp*split_len, (sp+1)*split_len), K/V loads are vectorized (see
// bs_ldrow), and the result is written as partial (m, l, acc) instead of the
// normalized output row. Empty sub-ranges (fully masked / beyond the tile's
// causal end) write (-1e30, 0, 0) which the reduce weights to zero. grp0 =
// first tile of this slab (scratch is sized for a slab of tiles, not the
// whole grid -- a pure memory knob, slab boundaries never touch numerics).
template <int TM, int VPL>
__device__ __forceinline__ void
blockq_split_body(const unsigned short *q, const unsigned short *k,
                  const unsigned short *v, float *pm, float *pl, float *pacc,
                  int HQ, int HKV, int N_q, int N_kv, int cache_from, int d,
                  int window, float softcap, int ring_cap, int split_len,
                  int n_splits, int grp0) {
  const int lane = threadIdx.x;             // 0..31
  const int grp = grp0 + blockIdx.x;
  const int sp = blockIdx.y;
  const int n_row_tiles = (N_q + TM - 1) / TM;
  const int head_q = grp / n_row_tiles;
  const int tile = grp % n_row_tiles;
  const int m0 = tile * TM;
  if (head_q >= HQ || m0 >= N_q) return;
  const int gqa = HQ / HKV, hkv = head_q / gqa;
  const int HD_Q = HQ * d, HD_KV = HKV * d;
  const float scale = rsqrtf((float)d);
  const int lane0 = lane * VPL;
  float q_reg[TM][VPL], acc_reg[TM][VPL], m_i[TM], l_i[TM];
  int valid[TM];
#pragma unroll
  for (int r = 0; r < TM; r++) {
    int m = m0 + r; valid[r] = (m < N_q) ? 1 : 0; m_i[r] = -1e30f; l_i[r] = 0.f;
    long q_base = (long)(valid[r] ? m : 0) * HD_Q + (long)head_q * d;
#pragma unroll
    for (int vv = 0; vv < VPL; vv++) {
      q_reg[r][vv] = bs_h2f(q[q_base + lane0 + vv]); acc_reg[r][vv] = 0.f;
    }
  }
  const int q_pos_off = cache_from;          // absolute query pos = m0+r+cache_from
  int last_row = ((m0 + TM - 1 < N_q) ? (m0 + TM - 1) : (N_q - 1)) + q_pos_off;
  int n_last = (N_kv - 1 < last_row) ? (N_kv - 1) : last_row;   // causal
  int n_lo = (window > 0) ? (m0 + q_pos_off - window + 1) : 0;
  if (n_lo < 0) n_lo = 0;
  // clip to split sp's sub-range of the uniform key partition
  int s_lo = sp * split_len; if (s_lo < n_lo) s_lo = n_lo;
  long s_hi_l = (long)(sp + 1) * split_len - 1;
  int s_hi = (s_hi_l > (long)n_last) ? n_last : (int)s_hi_l;
  for (int n = s_lo; n <= s_hi; ++n) {
    // [kv-window-ring] physical cache row = n % ring_cap (ring_cap<=0: linear).
    long pn = (ring_cap > 0) ? (long)(n % ring_cap) : (long)n;
    long kv_base = pn * HD_KV + (long)hkv * d + lane0;
    unsigned short kh[VPL], vh[VPL];
    bs_ldrow<VPL>(k + kv_base, kh);
    bs_ldrow<VPL>(v + kv_base, vh);
    float k_reg[VPL];
#pragma unroll
    for (int vv = 0; vv < VPL; vv++) k_reg[vv] = bs_h2f(kh[vv]);
    float sdot[TM];
#pragma unroll
    for (int r = 0; r < TM; r++) {
      float p = 0.f;
#pragma unroll
      for (int vv = 0; vv < VPL; vv++) p += q_reg[r][vv] * k_reg[vv];
      sdot[r] = bs_wreduce(p);
    }
    float v_reg[VPL];
#pragma unroll
    for (int vv = 0; vv < VPL; vv++) v_reg[vv] = bs_h2f(vh[vv]);
#pragma unroll
    for (int r = 0; r < TM; r++) {
      int m = m0 + r + q_pos_off;
      if (!valid[r] || n > m || (window > 0 && n + window <= m)) continue;
      float s = scale * sdot[r];
      if (softcap > 0.f) s = softcap * tanhf(s / softcap);
      float m_new = fmaxf(m_i[r], s), alpha = __expf(m_i[r] - m_new),
            pp = __expf(s - m_new);
#pragma unroll
      for (int vv = 0; vv < VPL; vv++)
        acc_reg[r][vv] = alpha * acc_reg[r][vv] + pp * v_reg[vv];
      l_i[r] = alpha * l_i[r] + pp; m_i[r] = m_new;
    }
  }
  // partial write; scratch rows are slab-local: (tile-in-slab, sp, r)
  long sbase = ((long)blockIdx.x * n_splits + sp) * TM;
#pragma unroll
  for (int r = 0; r < TM; r++) {
    if (lane == 0) { pm[sbase + r] = m_i[r]; pl[sbase + r] = l_i[r]; }
#pragma unroll
    for (int vv = 0; vv < VPL; vv++)
      pacc[(sbase + r) * (long)d + lane0 + vv] = acc_reg[r][vv];
  }
}
extern "C" __global__ void __launch_bounds__(32, 24)
attn_blockq_split_d128(const unsigned short *q, const unsigned short *k,
                       const unsigned short *v, float *pm, float *pl,
                       float *pacc, int HQ, int HKV, int N_q, int N_kv,
                       int cache_from, int d, int window, float softcap,
                       int ring_cap, int split_len, int n_splits, int grp0) {
  blockq_split_body<4, 4>(q, k, v, pm, pl, pacc, HQ, HKV, N_q, N_kv,
                          cache_from, d, window, softcap, ring_cap, split_len,
                          n_splits, grp0);
}
extern "C" __global__ void __launch_bounds__(32, 16)
attn_blockq_split_d256(const unsigned short *q, const unsigned short *k,
                       const unsigned short *v, float *pm, float *pl,
                       float *pacc, int HQ, int HKV, int N_q, int N_kv,
                       int cache_from, int d, int window, float softcap,
                       int ring_cap, int split_len, int n_splits, int grp0) {
  blockq_split_body<4, 8>(q, k, v, pm, pl, pacc, HQ, HKV, N_q, N_kv,
                          cache_from, d, window, softcap, ring_cap, split_len,
                          n_splits, grp0);
}
extern "C" __global__ void __launch_bounds__(32, 12)
attn_blockq_split_d512(const unsigned short *q, const unsigned short *k,
                       const unsigned short *v, float *pm, float *pl,
                       float *pacc, int HQ, int HKV, int N_q, int N_kv,
                       int cache_from, int d, int window, float softcap,
                       int ring_cap, int split_len, int n_splits, int grp0) {
  blockq_split_body<4, 16>(q, k, v, pm, pl, pacc, HQ, HKV, N_q, N_kv,
                           cache_from, d, window, softcap, ring_cap, split_len,
                           n_splits, grp0);
}
// Reduce pass: one 32-lane block per slab-local tile. Combines the n_splits
// partials of each of the tile's TM rows with the standard flash merge
// (M = max_s m_s; out = sum_s exp(m_s - M) acc_s / sum_s exp(m_s - M) l_s),
// looping the splits in FIXED order in every lane -> deterministic, no
// atomics. n_splits <= 32 is enforced by the host dispatch (w[] bound).
extern "C" __global__ void
attn_blockq_split_reduce(const float *pm, const float *pl, const float *pacc,
                         unsigned short *o, int HQ, int N_q, int d,
                         int n_splits, int grp0) {
  const int TM = 4;
  const int lane = threadIdx.x;             // 0..31
  const int grp = grp0 + blockIdx.x;
  const int n_row_tiles = (N_q + TM - 1) / TM;
  const int head_q = grp / n_row_tiles;
  const int tile = grp % n_row_tiles;
  const int m0 = tile * TM;
  if (head_q >= HQ || m0 >= N_q) return;
  const int HD_Q = HQ * d;
  const long tbase = (long)blockIdx.x * n_splits * TM;
  for (int r = 0; r < TM; r++) {
    const int m = m0 + r;
    if (m >= N_q) continue;
    float M = -1e30f;
    for (int s = 0; s < n_splits; ++s)
      M = fmaxf(M, pm[tbase + (long)s * TM + r]);
    float w[32];
    float L = 0.f;
    for (int s = 0; s < n_splits; ++s) {
      float ws = __expf(pm[tbase + (long)s * TM + r] - M);
      w[s] = ws;
      L += pl[tbase + (long)s * TM + r] * ws;
    }
    float inv = L > 0.f ? 1.f / L : 0.f;
    unsigned short *orow = o + (long)m * HD_Q + (long)head_q * d;
    for (int dd = lane; dd < d; dd += 32) {
      float a = 0.f;
      for (int s = 0; s < n_splits; ++s)
        a += pacc[(tbase + (long)s * TM + r) * d + dd] * w[s];
      orow[dd] = bs_f2h(a * inv);
    }
  }
}
)CU";

// Row-wise causal+window softmax over a per-head scores matrix [N_q, N_kv]
// (row-major: scores[i*N_kv + j] = dot(Q_i, K_j) already scaled). Masks
// j>i_abs (causal) and j<i_abs-window+1 (sliding) to 0, softmax in FP32 over
// the valid range, writes fp16 probabilities in place. One block per query row.
static const char *ATTN_SOFTMAX_SRC = R"CU(
extern "C" {
// Native fp16<->fp32 conversion. These were a hand-rolled software bit
// twiddle -- including a data-dependent do/while for denormals -- and they run
// PER ELEMENT of the score matrix, three times on the read side plus once on
// the write. bq_h2f/bq_f2h 380 lines above already made exactly this swap for
// the Q/K/V hot loop and record that the rounding is identical, so this is
// bit-for-bit the same output. Ampere issues INT32 on half its lanes, so the
// ~58 integer ops per element these cost were roughly 116 lane-cycles each.
__device__ __forceinline__ float sm_h2f(unsigned short h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short sm_f2h(float f) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
__global__ void attn_softmax_fp16(unsigned short *scores, int N_q, int N_kv,
                                  int ld_s, int cache_from, int window,
                                  float softcap) {
  int i = blockIdx.x;
  if (i >= N_q) return;
  unsigned short *row = scores + (long)i * ld_s;
  int i_abs = cache_from + i;
  int j_hi = i_abs < N_kv - 1 ? i_abs : N_kv - 1;
  int j_lo = (window > 0) ? i_abs - window + 1 : 0;
  if (j_lo < 0) j_lo = 0;
  int tid = threadIdx.x, B = blockDim.x;
  extern __shared__ float sh[];
  float lm = -1e30f;
  for (int j = j_lo + tid; j <= j_hi; j += B) {
    float v = sm_h2f(row[j]);
    if (softcap > 0.f) v = softcap * tanhf(v / softcap);
    lm = fmaxf(lm, v);
  }
  sh[tid] = lm; __syncthreads();
  for (int s = B >> 1; s > 0; s >>= 1) { if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]); __syncthreads(); }
  float mx = sh[0]; __syncthreads();
  float ls = 0.f;
  for (int j = j_lo + tid; j <= j_hi; j += B) {
    float v = sm_h2f(row[j]);
    if (softcap > 0.f) v = softcap * tanhf(v / softcap);
    ls += __expf(v - mx);
  }
  sh[tid] = ls; __syncthreads();
  for (int s = B >> 1; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
  float inv = sh[0] > 0.f ? 1.f / sh[0] : 0.f; __syncthreads();
  for (int j = tid; j < N_kv; j += B) {
    float p = 0.f;
    if (j >= j_lo && j <= j_hi) {
      float v = sm_h2f(row[j]);
      if (softcap > 0.f) v = softcap * tanhf(v / softcap);
      p = __expf(v - mx) * inv;
    }
    row[j] = sm_f2h(p);
  }
}
}
)CU";

namespace {
float *g_pm = nullptr, *g_pl = nullptr, *g_pacc = nullptr;
size_t g_pm_cap = 0, g_pacc_cap = 0;
std::mutex g_sk_mtx;
// M2-B: the FIXED chunk-count stride that the split-KV scratch
// (g_pm/g_pl/g_pacc) is pre-sized to at prewarm. The captured graph launches
// gridDim.y and strides partial<->reduce by THIS value (not the per-token live
// n_chunks) so one capture is valid for every token. 0 until prewarm runs (then
// M2-B is unavailable -> M1).
int g_sk_max_nchunks = 0;
bool ensure_sk(size_t mn, size_t acc) {
  if (mn > g_pm_cap) {
    // cudaMalloc/cudaFree inside a CUDA-graph stream capture invalidates the
    // capture. The decode split-KV scratch is pre-grown at load by
    // cuda_attention_splitkv_prewarm() so this branch must not run under
    // capture; if it ever would (an under-sized prewarm), bail so the caller
    // falls back rather than corrupting the graph.
    if (StreamManager::Global().isCapturing()) {
      StreamManager::Global().markCaptureDoomed(
        "the split-KV attention scratch was under-sized by the prewarm");
      return false;
    }
    if (g_pm)
      cudaFree(g_pm);
    if (g_pl)
      cudaFree(g_pl);
    if (cudaMalloc(&g_pm, mn * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&g_pl, mn * sizeof(float)) != cudaSuccess)
      return false;
    g_pm_cap = mn;
  }
  if (acc > g_pacc_cap) {
    if (StreamManager::Global().isCapturing()) {
      StreamManager::Global().markCaptureDoomed(
        "the split-KV attention accumulator was under-sized by the prewarm");
      return false;
    }
    if (g_pacc)
      cudaFree(g_pacc);
    if (cudaMalloc(&g_pacc, acc * sizeof(float)) != cudaSuccess)
      return false;
    g_pacc_cap = acc;
  }
  return true;
}

bool attention_splitkv_decode(const unsigned short *q, const unsigned short *k,
                              const unsigned short *v, unsigned short *o,
                              int HQ, int HKV, int N_kv, int cache_from, int d,
                              int window, float softcap, int chunk_kv,
                              int ring_cap) {
  const int n_chunks = (N_kv + chunk_kv - 1) / chunk_kv;
  // M2-B: when the device pos buffer is bound, the captured graph uses the
  // FIXED max-chunk stride/grid (g_sk_max_nchunks, published at prewarm) so one
  // capture serves every token. M1/non-graph (dpos=nullptr) uses the live
  // n_chunks -> bit-identical to the original. Mirror the dense path's m2b gate
  // (line ~839).
  static const bool m2b = nntr_env_on("NNTR_CUDA_M2B");
  const int *dpos = (m2b && g_sk_max_nchunks > 0) ? cuda_pos_buffer() : nullptr;
  int max_nc = dpos ? g_sk_max_nchunks : n_chunks;
  // [splitkv-clip] On a sliding layer only the keys inside the window are ever
  // read, so anchoring the chunk grid at the window's chunk-aligned low bound
  // (see attn_partial) turns the whole-context grid into a window-sized one:
  // 29K context / 1K window / chunk 64 -> 457 chunks become 8. Bit-neutral,
  // the dropped chunks are the softmax identity. NNTR_CUDA_SPLITKV_CLIP=0 is
  // the kill switch (restores the whole-context grid verbatim).
  //
  // M2-B contract: the captured graph freezes the grid and the partial<->reduce
  // stride, so the clipped stride must be a pure function of the LAYER (window,
  // chunk) and never of the live key count -- ceil(window/chunk)+1 is the
  // maximum a window anchored at a chunk boundary can span, for every token.
  // The scratch is sized by the prewarm at the unclipped maximum, so a smaller
  // stride is always in bounds.
  static const bool clip_on = []() {
    const char *e = std::getenv("NNTR_CUDA_SPLITKV_CLIP");
    return !(e != nullptr && e[0] == '0');
  }();
  const int clip = (clip_on && window > 0 && window < N_kv) ? 1 : 0;
  if (clip) {
    const long wc = (long)window / chunk_kv + 2;
    if (wc < (long)max_nc)
      max_nc = (int)wc;
  }
  std::lock_guard<std::mutex> lk(g_sk_mtx);
  if (!ensure_sk((size_t)HQ * max_nc, (size_t)HQ * max_nc * d))
    return false;
  // [warp-decode] Single selection point for the partial kernel. The warp
  // kernel (barrier-free key loop, see ATTN_SPLITKV_WARP_SRC) is the default;
  // NNTR_CUDA_ATTN_LEGACY=1 restores the shared-memory tree-reduce
  // attn_partial verbatim (kill switch). It needs head_dim in {128,256,512}
  // (VPL = d/32 in {4,8,16}) and 8B/16B-aligned Q/K/V bases for the vector
  // loads; anything else falls back to the legacy kernel automatically.
  static const bool attn_legacy = nntr_env_on("NNTR_CUDA_ATTN_LEGACY");
  const char *wname = (d == 128)   ? "attn_partial_w128"
                      : (d == 256) ? "attn_partial_w256"
                      : (d == 512) ? "attn_partial_w512"
                                   : nullptr;
  const size_t valign = (d == 128) ? 8u : 16u;
  const bool warp_ok =
    !attn_legacy && wname != nullptr &&
    (((uintptr_t)q | (uintptr_t)k | (uintptr_t)v) % valign) == 0;
  const int NW = 4; // warps per block in the warp kernel
  // [gqa-fuse] Fold HPW query heads of one GQA group into a single warp so the
  // shared K/V row is fetched and converted once instead of gqa times (see
  // splitkv_warp_body). HPW must divide gqa (the group must sit inside one KV
  // head) and the resulting grid must still be wide enough to fill the device,
  // which is why short contexts and window-clipped sliding layers stay at
  // HPW=1: fusing there would trade re-reads for idle SMs.
  //
  // M2-B: HPW only scales gridDim.x, and it is derived from per-layer
  // constants plus the chunk count at capture time -- never from the live key
  // count -- so the captured geometry stays valid for every later token (the
  // per-token growth is absorbed by the live-chunk gate inside the kernel, as
  // before). Bit-identical to HPW=1; NNTR_CUDA_ATTN_FUSE=0 is the kill switch.
  static const bool fuse_on = []() {
    const char *e = std::getenv("NNTR_CUDA_ATTN_FUSE");
    return !(e != nullptr && e[0] == '0');
  }();
  static const int min_blocks = []() {
    const char *e = std::getenv("NNTR_CUDA_ATTN_FUSE_MINBLK");
    int v = e ? atoi(e) : 64;
    return v > 0 ? v : 64;
  }();
  const int gqa_all = (HKV > 0) ? (HQ / HKV) : 1;
  // work actually launched for this layer: clipped layers already fold their
  // whole-context grid down to the window, so use the same bound the kernel
  // does rather than the (max-context) scratch stride.
  const int eff_nc = clip ? max_nc : ((n_chunks < max_nc) ? n_chunks : max_nc);
  int hpw = 1;
  if (fuse_on && warp_ok && gqa_all > 1) {
    const int hpw_cap = (d == 512) ? 4 : 6; // register budget per head_dim
    static const int cand[] = {6, 4, 3, 2};
    for (int i = 0; i < 4; ++i) {
      const int hp = cand[i];
      if (hp > hpw_cap || (gqa_all % hp) != 0 || (HQ % hp) != 0)
        continue;
      if ((long)(HQ / hp) * eff_nc < (long)min_blocks)
        continue;
      hpw = hp;
      break;
    }
  }
  char fname[40];
  if (hpw > 1) {
    snprintf(fname, sizeof(fname), "attn_partial_w%d_h%d", d, hpw);
    wname = fname;
  }
  auto kp = warp_ok ? CudaContext::Global().registerCudaKernel(
                        ATTN_SPLITKV_WARP_SRC, wname)
                    : CudaContext::Global().registerCudaKernel(ATTN_SPLITKV_SRC,
                                                               "attn_partial");
  const int B = 128;
  // [reduce-tile] wide-grid merge (see attn_reduce_t). Needs (B + max_nc)
  // floats of shared memory; anything larger falls back to the one-block-per-
  // head attn_reduce, which NNTR_CUDA_ATTN_REDUCE_T=0 also restores.
  static const bool reduce_tiled = []() {
    const char *e = std::getenv("NNTR_CUDA_ATTN_REDUCE_T");
    return !(e != nullptr && e[0] == '0');
  }();
  const size_t rshmem = sizeof(float) * ((size_t)B + (size_t)max_nc);
  const bool rt_ok = reduce_tiled && rshmem <= 32768u;
  auto kr = CudaContext::Global().registerCudaKernel(
    ATTN_SPLITKV_SRC, rt_ok ? "attn_reduce_t" : "attn_reduce");
  if (!kp || !kr)
    return false;
  kp->SetKernelArguments(0, &q, sizeof(q));
  kp->SetKernelArguments(1, &k, sizeof(k));
  kp->SetKernelArguments(2, &v, sizeof(v));
  kp->SetKernelArguments(3, &g_pm, sizeof(g_pm));
  kp->SetKernelArguments(4, &g_pl, sizeof(g_pl));
  kp->SetKernelArguments(5, &g_pacc, sizeof(g_pacc));
  kp->SetKernelArguments(6, &HQ, sizeof(HQ));
  kp->SetKernelArguments(7, &HKV, sizeof(HKV));
  kp->SetKernelArguments(8, &N_kv, sizeof(N_kv));
  kp->SetKernelArguments(9, &cache_from, sizeof(cache_from));
  kp->SetKernelArguments(10, &d, sizeof(d));
  kp->SetKernelArguments(11, &window, sizeof(window));
  kp->SetKernelArguments(12, &softcap, sizeof(softcap));
  kp->SetKernelArguments(13, &chunk_kv, sizeof(chunk_kv));
  kp->SetKernelArguments(14, &n_chunks, sizeof(n_chunks));
  kp->SetKernelArguments(15, &dpos, sizeof(dpos));
  kp->SetKernelArguments(16, &max_nc, sizeof(max_nc));
  kp->SetKernelArguments(17, &ring_cap, sizeof(ring_cap));
  kp->SetKernelArguments(18, &clip, sizeof(clip));
  const int pg[3] = {HQ / hpw, max_nc, 1};
  const int pb[3] = {B, 1, 1};
  // legacy: Q row [d] + reduction scratch [B]; warp: HPW*NW acc rows [d] +
  // (m, l) per (head, warp).
  const unsigned int shmem =
    (unsigned int)(sizeof(float) * (warp_ok
                                      ? ((size_t)hpw * NW * d + 2 * hpw * NW)
                                      : ((size_t)d + B)));
  if (!StreamManager::Global().DispatchCommand(*kp, pg, pb, shmem))
    return false;
  kr->SetKernelArguments(0, &g_pm, sizeof(g_pm));
  kr->SetKernelArguments(1, &g_pl, sizeof(g_pl));
  kr->SetKernelArguments(2, &g_pacc, sizeof(g_pacc));
  kr->SetKernelArguments(3, &o, sizeof(o));
  kr->SetKernelArguments(4, &HQ, sizeof(HQ));
  kr->SetKernelArguments(5, &d, sizeof(d));
  kr->SetKernelArguments(6, &n_chunks, sizeof(n_chunks));
  kr->SetKernelArguments(7, &dpos, sizeof(dpos));
  kr->SetKernelArguments(8, &chunk_kv, sizeof(chunk_kv));
  kr->SetKernelArguments(9, &max_nc, sizeof(max_nc));
  kr->SetKernelArguments(10, &window, sizeof(window));
  kr->SetKernelArguments(11, &cache_from, sizeof(cache_from));
  kr->SetKernelArguments(12, &N_kv, sizeof(N_kv));
  kr->SetKernelArguments(13, &clip, sizeof(clip));
  const int rg[3] = {HQ, rt_ok ? ((d + B - 1) / B) : 1, 1};
  const int rb[3] = {B, 1, 1};
  if (!StreamManager::Global().DispatchCommand(
        *kr, rg, rb, rt_ok ? (unsigned int)rshmem : 0u))
    return false;
  return true;
}


// ---- T2 flash prefill (head_dim 256): online softmax, no score plane -----
// The tile_bench/attention_bench.cu kernel in NVRTC dialect. One CTA =
// (q-head, 128-row Q tile); 8 warps x 16 rows; KV tiles of 64 rows staged
// once (single buffer; a 2-ring exceeds the 163KB cap); S via f16-acc
// m16n8k16 off proven fragment recipes; P stays in C-fragment registers and
// feeds P@V as the A operand with zero relayout; O in fp32 C-frags.
// Semantic contract (NOT byte): bench-validated max|d| <= 1.3e-4 vs fp64.
static const char *ATTN_FLASH_SRC = R"CU(
#define AFD 256
#define AFBR 128
#define AFBC 64
#define AFLD (AFD + 8)
__device__ __forceinline__ float af_h2f(unsigned short h){
  float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ unsigned short af_f2h(float f){
  unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
__device__ __forceinline__ unsigned af_sh(const void *p){
  unsigned r;
  asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
      : "=r"(r) : "l"(p));
  return r;
}
extern "C" __global__ void __launch_bounds__(256, 1)
attn_flash_d256(const unsigned short *Q, const unsigned short *K,
                const unsigned short *V, unsigned short *O, int Nq, int Nkv,
                int qpos0, int nqh, int nkvh, float scale) {
  extern __shared__ unsigned short smem[];
  unsigned short *Qs = smem;
  unsigned short *Ks = Qs + AFBR * AFLD;
  unsigned short *Vs = Ks + AFBC * AFLD;
  const int qh = blockIdx.y, kvh = qh / (nqh / nkvh);
  const int r0 = blockIdx.x * AFBR;
  if (r0 >= Nq) return;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int qstride = nqh * AFD, kstride = nkvh * AFD;
  const unsigned short *Qg = Q + (long)qh * AFD;
  const unsigned short *Kg = K + (long)kvh * AFD;
  const unsigned short *Vg = V + (long)kvh * AFD;
  {
    const int row = tid >> 1, h8 = (tid & 1) * (AFD / 2);
    const unsigned short *srcq = Qg + (long)(r0 + row) * qstride + h8;
    const unsigned dst = af_sh(Qs + row * AFLD + h8);
    const int ok = (r0 + row) < Nq ? 16 : 0;
#pragma unroll
    for (int i = 0; i < AFD / 2 / 8; ++i)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(
                     dst + i * 16),
                   "l"(srcq + i * 8), "r"(ok));
    asm volatile("cp.async.commit_group;\n");
  }
  float o[32][4];
#pragma unroll
  for (int i = 0; i < 32; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
      o[i][j] = 0.f;
  float m_lo = -1e30f, m_hi = -1e30f, l_lo = 0.f, l_hi = 0.f;
  const int g = lane >> 2, t = lane & 3;
  const int wrow = warp * 16;
  const int q_hi_pos = qpos0 + r0 + AFBR - 1;
  const int kv_stop = Nkv < q_hi_pos + 1 ? Nkv : q_hi_pos + 1;
  asm volatile("cp.async.wait_group 0;\n");
  __syncthreads();
  for (int kbase = 0; kbase < kv_stop; kbase += AFBC) {
    {
      const int row = tid >> 1, h8 = (tid & 1) * (AFD / 2);
      const int kr = kbase + (row & (AFBC - 1));
      const int isv = row >= AFBC;
      const unsigned short *srckv =
        (isv ? Vg : Kg) + (long)kr * kstride + h8;
      const unsigned dst =
        af_sh((isv ? Vs : Ks) + (row & (AFBC - 1)) * AFLD + h8);
      const int ok = kr < Nkv ? 16 : 0;
#pragma unroll
      for (int i = 0; i < AFD / 2 / 8; ++i)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(
                       dst + i * 16),
                     "l"(srckv + i * 8), "r"(ok));
      asm volatile("cp.async.commit_group;\n");
      asm volatile("cp.async.wait_group 0;\n");
    }
    __syncthreads();
    unsigned sh2[8][2];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      sh2[i][0] = sh2[i][1] = 0u;
#pragma unroll
    for (int k0 = 0; k0 < AFD; k0 += 16) {
      int a0, a1, a2, a3;
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(af_sh(Qs + (wrow + (lane & 15)) * AFLD + k0 +
                    ((lane >> 4) & 1) * 8)));
      int bl0, bl1, bl2, bl3, bh0, bh1, bh2, bh3;
#pragma unroll
      for (int half32 = 0; half32 < 2; ++half32) {
        const int nb = half32 * 32;
        asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
          : "=r"(bl0), "=r"(bl1), "=r"(bl2), "=r"(bl3)
          : "r"(af_sh(Ks + (nb + (lane >> 3) * 8 + (lane & 7)) * AFLD + k0)));
        asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
          : "=r"(bh0), "=r"(bh1), "=r"(bh2), "=r"(bh3)
          : "r"(af_sh(Ks + (nb + (lane >> 3) * 8 + (lane & 7)) * AFLD + k0 +
                      8)));
#define AF_SMMA(SS, BL, BH)                                                    \
  asm volatile(                                                                \
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "                       \
    "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"                              \
    : "+r"(sh2[SS][0]), "+r"(sh2[SS][1])                                       \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(BL), "r"(BH))
        AF_SMMA(half32 * 4 + 0, bl0, bh0);
        AF_SMMA(half32 * 4 + 1, bl1, bh1);
        AF_SMMA(half32 * 4 + 2, bl2, bh2);
        AF_SMMA(half32 * 4 + 3, bl3, bh3);
#undef AF_SMMA
      }
    }
    float s[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s[i][0] = af_h2f((unsigned short)(sh2[i][0] & 0xFFFFu));
      s[i][1] = af_h2f((unsigned short)(sh2[i][0] >> 16));
      s[i][2] = af_h2f((unsigned short)(sh2[i][1] & 0xFFFFu));
      s[i][3] = af_h2f((unsigned short)(sh2[i][1] >> 16));
    }
    const int qlo = qpos0 + r0 + wrow + g, qhi = qlo + 8;
#pragma unroll
    for (int n8 = 0; n8 < 8; ++n8) {
      const int c = kbase + n8 * 8 + 2 * t;
      s[n8][0] = (c <= qlo && c < Nkv) ? s[n8][0] * scale : -1e30f;
      s[n8][1] = (c + 1 <= qlo && c + 1 < Nkv) ? s[n8][1] * scale : -1e30f;
      s[n8][2] = (c <= qhi && c < Nkv) ? s[n8][2] * scale : -1e30f;
      s[n8][3] = (c + 1 <= qhi && c + 1 < Nkv) ? s[n8][3] * scale : -1e30f;
    }
    float tm0 = -1e30f, tm1 = -1e30f;
#pragma unroll
    for (int n8 = 0; n8 < 8; ++n8) {
      tm0 = fmaxf(tm0, fmaxf(s[n8][0], s[n8][1]));
      tm1 = fmaxf(tm1, fmaxf(s[n8][2], s[n8][3]));
    }
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1));
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));
    const float mn0 = fmaxf(m_lo, tm0), mn1 = fmaxf(m_hi, tm1);
    const float rs0 = expf(m_lo - mn0), rs1 = expf(m_hi - mn1);
    unsigned pf[8][2];
    float tl0 = 0.f, tl1 = 0.f;
#pragma unroll
    for (int n8 = 0; n8 < 8; ++n8) {
      const float p00 = expf(s[n8][0] - mn0), p01 = expf(s[n8][1] - mn0);
      const float p10 = expf(s[n8][2] - mn1), p11 = expf(s[n8][3] - mn1);
      tl0 += p00 + p01;
      tl1 += p10 + p11;
      pf[n8][0] = (unsigned)af_f2h(p00) | ((unsigned)af_f2h(p01) << 16);
      pf[n8][1] = (unsigned)af_f2h(p10) | ((unsigned)af_f2h(p11) << 16);
    }
    tl0 += __shfl_xor_sync(0xffffffffu, tl0, 1);
    tl0 += __shfl_xor_sync(0xffffffffu, tl0, 2);
    tl1 += __shfl_xor_sync(0xffffffffu, tl1, 1);
    tl1 += __shfl_xor_sync(0xffffffffu, tl1, 2);
    l_lo = l_lo * rs0 + tl0;
    l_hi = l_hi * rs1 + tl1;
    m_lo = mn0;
    m_hi = mn1;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      o[i][0] *= rs0;
      o[i][1] *= rs0;
      o[i][2] *= rs1;
      o[i][3] *= rs1;
    }
#pragma unroll
    for (int k16 = 0; k16 < AFBC / 16; ++k16) {
      const unsigned pa0 = pf[k16 * 2][0], pa1 = pf[k16 * 2][1];
      const unsigned pa2 = pf[k16 * 2 + 1][0], pa3 = pf[k16 * 2 + 1][1];
#pragma unroll
      for (int q4 = 0; q4 < 8; ++q4) {
        const int nb = q4 * 32;
        int bl0, bl1, bl2, bl3, bh0, bh1, bh2, bh3;
        asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, "
          "[%4];\n"
          : "=r"(bl0), "=r"(bl1), "=r"(bl2), "=r"(bl3)
          : "r"(af_sh(Vs + (k16 * 16 + (lane & 7)) * AFLD + nb +
                      (lane >> 3) * 8)));
        asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, "
          "[%4];\n"
          : "=r"(bh0), "=r"(bh1), "=r"(bh2), "=r"(bh3)
          : "r"(af_sh(Vs + (k16 * 16 + 8 + (lane & 7)) * AFLD + nb +
                      (lane >> 3) * 8)));
#define AF_PMMA(NS, BL, BH)                                                    \
  asm volatile(                                                                \
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                       \
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                  \
    : "+f"(o[NS][0]), "+f"(o[NS][1]), "+f"(o[NS][2]), "+f"(o[NS][3])           \
    : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3), "r"(BL), "r"(BH))
        AF_PMMA(q4 * 4 + 0, bl0, bh0);
        AF_PMMA(q4 * 4 + 1, bl1, bh1);
        AF_PMMA(q4 * 4 + 2, bl2, bh2);
        AF_PMMA(q4 * 4 + 3, bl3, bh3);
#undef AF_PMMA
      }
    }
    __syncthreads();
  }
  const float il0 = l_lo > 0.f ? 1.f / l_lo : 0.f;
  const float il1 = l_hi > 0.f ? 1.f / l_hi : 0.f;
#pragma unroll
  for (int n8 = 0; n8 < 32; ++n8) {
    const int c = n8 * 8 + 2 * t;
    const int rlo = r0 + wrow + g, rhi = rlo + 8;
    if (rlo < Nq) {
      unsigned short *dp = O + (long)rlo * qstride + qh * AFD + c;
      dp[0] = af_f2h(o[n8][0] * il0);
      dp[1] = af_f2h(o[n8][1] * il0);
    }
    if (rhi < Nq) {
      unsigned short *dp = O + (long)rhi * qstride + qh * AFD + c;
      dp[0] = af_f2h(o[n8][2] * il1);
      dp[1] = af_f2h(o[n8][3] * il1);
    }
  }
}
)CU";

static bool attn_flash_prefill_d256(const unsigned short *q,
                                    const unsigned short *k,
                                    const unsigned short *v, unsigned short *o,
                                    int HQ, int HKV, int N_q, int N_kv,
                                    int cache_from, int d) {
  auto kf = CudaContext::Global().registerCudaKernel(ATTN_FLASH_SRC,
                                                     "attn_flash_d256");
  if (!kf)
    return false;
  constexpr int SMEM = (128 + 2 * 64) * (256 + 8) * 2; // 135,168 B
  static bool attr_ok = []() { return true; }();
  static bool attr_done = false;
  if (!attr_done) {
    attr_done = true;
    if (cuFuncSetAttribute(kf->GetFunction(),
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           SMEM) != CUDA_SUCCESS)
      attr_ok = false;
  }
  if (!attr_ok)
    return false;
  static bool rdbg = []() {
    const char *e = std::getenv("NNTR_ATTN_DBG");
    if (e == nullptr || e[0] == '0')
      return false;
    return true;
  }();
  if (rdbg) {
    rdbg = false;
    int nr = 0, lm = 0;
    cuFuncGetAttribute(&nr, CU_FUNC_ATTRIBUTE_NUM_REGS, kf->GetFunction());
    cuFuncGetAttribute(&lm, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                       kf->GetFunction());
    fprintf(stderr, "[attn_flash] regs=%d local=%dB smem=%d\n", nr, lm, SMEM);
  }
  const float scale = 1.0f / sqrtf((float)d);
  kf->SetKernelArguments(0, &q, sizeof(q));
  kf->SetKernelArguments(1, &k, sizeof(k));
  kf->SetKernelArguments(2, &v, sizeof(v));
  kf->SetKernelArguments(3, &o, sizeof(o));
  kf->SetKernelArguments(4, &N_q, sizeof(N_q));
  kf->SetKernelArguments(5, &N_kv, sizeof(N_kv));
  kf->SetKernelArguments(6, &cache_from, sizeof(cache_from));
  kf->SetKernelArguments(7, &HQ, sizeof(HQ));
  kf->SetKernelArguments(8, &HKV, sizeof(HKV));
  kf->SetKernelArguments(9, &scale, sizeof(scale));
  const int g[3] = {(N_q + 127) / 128, HQ, 1}, b[3] = {256, 1, 1};
  return StreamManager::Global().DispatchCommand(*kf, g, b, (unsigned)SMEM);
}

// GEMM-based multi-row prefill attention. The per-key flash kernel
// (attn_core_il) is fetch/sync-bound (~0.4% of peak on d=128) because it
// serial- reduces every key; for prefill (N_q>1) materialising scores via
// cuBLAS fp16 GEMMs (QK^T -> softmax -> PV) is far faster and
// head_dim-agnostic, so it also helps the head_dim=128 (qwen3/llama) case that
// block-Q does not. Layout (interleaved fp16, column-major cuBLAS): per
// query-head h (kv-head hkv=h/gqa) scores_cm[N_kv,N_q] = K_h^T@Q_h reads back
// as row-major scores[N_q,N_kv]; then O_cm[d,N_q] = V_h@scores_cm.
float *g_scores = nullptr;
size_t g_scores_cap = 0;
std::mutex g_ga_mtx;
static bool attn_gemm_prefill_block(const unsigned short *q,
                                    const unsigned short *k,
                                    const unsigned short *v, unsigned short *o,
                                    int HQ, int HKV, int N_q, int N_kv,
                                    int cache_from, int d, int window,
                                    float softcap) {
  std::lock_guard<std::mutex> lk(g_ga_mtx);
  cublasHandle_t bh = BlasManager::Global().handle();
  if (!bh)
    return false;
  auto sm = CudaContext::Global().registerCudaKernel(ATTN_SOFTMAX_SRC,
                                                     "attn_softmax_fp16");
  if (!sm)
    return false;
  // scratch scores [N_q, ld_s] fp16 reused across heads (heads run serially).
  // ld_s pads the leading dimension to a multiple of 16 elements (32 B): an
  // odd N_kv (the last chunk's 20,463) misaligns every score column, which
  // measured 28-39% slower on both GEMMs. Padding columns are written by
  // nothing and read by nothing (QK^T writes N_kv rows; PV's K loop stops at
  // N_kv; the softmax row pointer strides by ld_s) -- output is byte-exact.
  const int ld_s = (N_kv + 15) & ~15;
  const size_t need = (size_t)N_q * ld_s;
  if (need > g_scores_cap) {
    // This runs on the PREFILL path, which IS graph-captured on an integrated
    // GPU. A cudaFree/cudaMalloc there does not merely fail -- it INVALIDATES
    // the capture, and every later cuLaunchKernel on the stream then fails with
    // CUDA_ERROR_STREAM_CAPTURE_INVALIDATED, so the rest of the forward falls
    // off its device paths one op at a time (field: the attention output
    // projection then took the host dot() and threw "pack before run model").
    // The load-time warmup prefill grows this before any capture; if a later,
    // longer prefill still needs more, refuse and abandon the capture instead.
    if (StreamManager::Global().isCapturing()) {
      StreamManager::Global().markCaptureDoomed(
        "the GEMM-attention score scratch has to grow under capture");
      return false;
    }
    if (g_scores)
      cudaFree(g_scores);
    if (cudaMalloc(&g_scores, need * sizeof(unsigned short)) != cudaSuccess) {
      g_scores = nullptr;
      g_scores_cap = 0;
      return false;
    }
    g_scores_cap = need;
  }
  auto *scores = reinterpret_cast<unsigned short *>(g_scores);
  const int gqa = HQ / HKV;
  const int HD_Q = HQ * d, HD_KV = HKV * d;
  const float scale = 1.0f / sqrtf((float)d);
  const float one = 1.0f, zero = 0.0f;
  const int win = (window <= 0 || window >= N_kv) ? 0 : window;
  const int B = 256;
  const size_t shmem = (size_t)B * sizeof(float);
  // QK^T accumulate type: 32F by DEFAULT, and the 16F arm is a MEASURED
  // TRAP on this cuBLAS/GA10B: numerics held (answer correct) but kernel
  // selection collapsed on two of the five chunk shapes (Nkv=4096: 14.6 ->
  // 177.9 ms; Nkv=20463: 96.5 -> 181.2 ms) for a -3.7% e2e regression. The
  // '16F doubles the mma rate' arithmetic does NOT survive contact with
  // cublasGemmEx here. NNTR_ATTN_QKT_16F=1 keeps the arm reproducible.
  static const cublasComputeType_t g_qkt_compute = []() {
    const char *e = std::getenv("NNTR_ATTN_QKT_16F");
    return (e != nullptr && e[0] == '1') ? CUBLAS_COMPUTE_16F
                                         : CUBLAS_COMPUTE_32F;
  }();
  // NNTR_ATTN_DBG=1: per-stage GPU ms via events, printed for the FIRST call
  // at each distinct N_kv (so every prefill chunk shape gets one sample).
  static const bool g_adbg = []() {
    const char *e = std::getenv("NNTR_ATTN_DBG");
    return e != nullptr && e[0] == '1';
  }();
  static int g_adbg_seen[8];
  static int g_adbg_n = 0;
  bool adbg_this = false;
  if (g_adbg && HQ <= 16) {
    bool seen = false;
    for (int i = 0; i < g_adbg_n; ++i)
      if (g_adbg_seen[i] == N_kv)
        seen = true;
    if (!seen && g_adbg_n < 8) {
      g_adbg_seen[g_adbg_n++] = N_kv;
      adbg_this = true;
    }
  }
  cudaEvent_t aev[1 + 3 * 16];
  cudaStream_t astream = StreamManager::Global().GetStream();
  if (adbg_this) {
    for (int i = 0; i < 1 + 3 * HQ; ++i)
      cudaEventCreate(&aev[i]);
    cudaEventRecord(aev[0], astream);
  }
  for (int h = 0; h < HQ; ++h) {
    const int hkv = h / gqa;
    const unsigned short *Qh = q + (long)h * d;   // [N_q,d] ld=HD_Q
    const unsigned short *Kh = k + (long)hkv * d; // [N_kv,d] ld=HD_KV
    const unsigned short *Vh = v + (long)hkv * d; // [N_kv,d] ld=HD_KV
    unsigned short *Oh = o + (long)h * d;         // [N_q,d] ld=HD_Q
    // scores_cm[N_kv,N_q] = (K_h^T)[N_kv,d] @ Q_h[d,N_q] * scale -> row-major
    // scores[i*N_kv+j] = scale*dot(Q_i,K_j).
    cublasStatus_t s1;
    if (g_qkt_compute == CUBLAS_COMPUTE_16F) {
      // 16F compute wants half alpha/beta. scale = 1/16 is exact in fp16.
      const __half sh = __float2half(scale), zh = __float2half(0.0f);
      s1 = cublasGemmEx(bh, CUBLAS_OP_T, CUBLAS_OP_N, N_kv, N_q, d, &sh, Kh,
                        CUDA_R_16F, HD_KV, Qh, CUDA_R_16F, HD_Q, &zh, scores,
                        CUDA_R_16F, ld_s, CUBLAS_COMPUTE_16F,
                        CUBLAS_GEMM_DEFAULT);
    } else {
      s1 = cublasGemmEx(bh, CUBLAS_OP_T, CUBLAS_OP_N, N_kv, N_q, d, &scale,
                        Kh, CUDA_R_16F, HD_KV, Qh, CUDA_R_16F, HD_Q, &zero,
                        scores, CUDA_R_16F, ld_s, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT);
    }
    if (s1 != CUBLAS_STATUS_SUCCESS)
      return false;
    if (adbg_this)
      cudaEventRecord(aev[1 + 3 * h], astream);
    sm->SetKernelArguments(0, &scores, sizeof(scores));
    sm->SetKernelArguments(1, &N_q, sizeof(N_q));
    sm->SetKernelArguments(2, &N_kv, sizeof(N_kv));
    sm->SetKernelArguments(3, &ld_s, sizeof(ld_s));
    sm->SetKernelArguments(4, &cache_from, sizeof(cache_from));
    sm->SetKernelArguments(5, &win, sizeof(win));
    sm->SetKernelArguments(6, &softcap, sizeof(softcap));
    const int sg[3] = {N_q, 1, 1};
    const int sb[3] = {B, 1, 1};
    if (!StreamManager::Global().DispatchCommand(*sm, sg, sb, shmem))
      return false;
    if (adbg_this)
      cudaEventRecord(aev[2 + 3 * h], astream);
    // O_cm[d,N_q] = V_h[d,N_kv] @ scores_cm[N_kv,N_q] -> row-major O[i,e].
    cublasStatus_t s2 =
      cublasGemmEx(bh, CUBLAS_OP_N, CUBLAS_OP_N, d, N_q, N_kv, &one, Vh,
                   CUDA_R_16F, HD_KV, scores, CUDA_R_16F, ld_s, &zero, Oh,
                   CUDA_R_16F, HD_Q, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s2 != CUBLAS_STATUS_SUCCESS)
      return false;
    if (adbg_this)
      cudaEventRecord(aev[3 + 3 * h], astream);
  }
  if (adbg_this) {
    cudaEventSynchronize(aev[3 * HQ]);
    float t_qkt = 0.f, t_sm = 0.f, t_pv = 0.f, ms = 0.f;
    for (int h = 0; h < HQ; ++h) {
      cudaEventElapsedTime(&ms, aev[3 * h], aev[1 + 3 * h]);
      t_qkt += ms;
      cudaEventElapsedTime(&ms, aev[1 + 3 * h], aev[2 + 3 * h]);
      t_sm += ms;
      cudaEventElapsedTime(&ms, aev[2 + 3 * h], aev[3 + 3 * h]);
      t_pv += ms;
    }
    fprintf(stderr,
            "[attn_dbg] Nq=%d Nkv=%d from=%d qkt=%.2fms softmax=%.2fms "
            "pv=%.2fms total=%.2fms\n",
            N_q, N_kv, cache_from, t_qkt, t_sm, t_pv, t_qkt + t_sm + t_pv);
    for (int i = 0; i < 1 + 3 * HQ; ++i)
      cudaEventDestroy(aev[i]);
  }
  return true;
}

bool attention_gemm_prefill_fp16(const unsigned short *q,
                                 const unsigned short *k,
                                 const unsigned short *v, unsigned short *o,
                                 int HQ, int HKV, int N_q, int N_kv,
                                 int cache_from, int d, int window,
                                 float softcap) {
  // Q sub-tiling: a 4096-row Q chunk computes its causal diagonal block as a
  // full square, so 16.7% of QK^T/softmax/PV touches masked elements. Sub-
  // blocks of 1024 rows shrink each sub-block's kv ceiling to its own causal
  // limit (total score elements -12.5% at 20K). Numerics: each row's softmax
  // span [j_lo..j_hi] is untouched, so probabilities are identical; only the
  // exactly-zero masked tail of the PV reduction is dropped.
  // NNTR_ATTN_QSUB overrides the sub-block rows (0 disables).
  static const int g_qsub = []() {
    const char *e = std::getenv("NNTR_ATTN_QSUB");
    return e != nullptr ? atoi(e) : 1024;
  }();
  // Flash DEFAULT ON (2026-08-12, gates: long-context continuation EXACT
  // match; 16-prompt NLL 0.2371 vs baseline 0.2416 -- flash marginally
  // BETTER; every greedy divergence eyeballed benign; 20K prefill
  // 17,833 -> 16,308 ms interleaved). NNTR_ATTN_FLASH=0 restores the
  // GEMM/materialized-scores path. The online-softmax kernel replaces the
  // whole QK^T/softmax/PV sequence for the shapes it supports (d=256, no
  // window, no softcap); everything else falls through to the GEMM path.
  static const bool g_flash = []() {
    const char *e = std::getenv("NNTR_ATTN_FLASH");
    return e == nullptr || e[0] != '0';
  }();
  if (g_flash && d == 256 && softcap == 0.0f && HKV > 0 && HQ % HKV == 0 &&
      N_q > 1 && (window <= 0 || window >= N_kv)) {
    if (attn_flash_prefill_d256(q, k, v, o, HQ, HKV, N_q, N_kv, cache_from, d))
      return true; // launch failure falls through to the GEMM path
  }
  if (g_qsub <= 0 || N_q <= g_qsub)
    return attn_gemm_prefill_block(q, k, v, o, HQ, HKV, N_q, N_kv, cache_from,
                                   d, window, softcap);
  const long ldq = (long)HQ * d;
  for (int q0 = 0; q0 < N_q; q0 += g_qsub) {
    const int nsub = (N_q - q0 < g_qsub) ? (N_q - q0) : g_qsub;
    const int cf = cache_from + q0;
    int nkv = cf + nsub; // this sub-block's causal ceiling
    if (nkv > N_kv)
      nkv = N_kv;
    if (!attn_gemm_prefill_block(q + q0 * ldq, k, v, o + q0 * ldq, HQ, HKV,
                                 nsub, nkv, cf, d, window, softcap))
      return false;
  }
  return true;
}

// --- split-KV chunked-prefill scratch (partial m/l/acc, slab-sized) ---
// Separate from the decode scratch (g_pm/...): that one is pre-sized by
// cuda_attention_splitkv_prewarm under the M2-B fixed-stride contract. This one
// grows lazily -- which is fine only OUTSIDE a capture, and the prefill IS
// captured on an integrated GPU (NNTR_CUDA_PREFILL_GRAPH, default on there).
// The first prefill after the load-time warmup pass has already grown it; the
// guards below cover the case where a later, longer prefill needs more.
float *g_bs_pm = nullptr, *g_bs_pl = nullptr, *g_bs_pacc = nullptr;
size_t g_bs_pm_cap = 0, g_bs_pacc_cap = 0;
std::mutex g_bs_mtx;
bool ensure_bs(size_t mn, size_t acc) {
  if (mn > g_bs_pm_cap) {
    if (StreamManager::Global().isCapturing()) {
      StreamManager::Global().markCaptureDoomed(
        "the chunked-prefill attention scratch has to grow under capture");
      return false;
    }
    if (g_bs_pm)
      cudaFree(g_bs_pm);
    if (g_bs_pl)
      cudaFree(g_bs_pl);
    if (cudaMalloc(&g_bs_pm, mn * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&g_bs_pl, mn * sizeof(float)) != cudaSuccess) {
      g_bs_pm = g_bs_pl = nullptr;
      g_bs_pm_cap = 0;
      return false;
    }
    g_bs_pm_cap = mn;
  }
  if (acc > g_bs_pacc_cap) {
    if (StreamManager::Global().isCapturing()) {
      StreamManager::Global().markCaptureDoomed(
        "the chunked-prefill attention accumulator has to grow under capture");
      return false;
    }
    if (g_bs_pacc)
      cudaFree(g_bs_pacc);
    if (cudaMalloc(&g_bs_pacc, acc * sizeof(float)) != cudaSuccess) {
      g_bs_pacc = nullptr;
      g_bs_pacc_cap = 0;
      return false;
    }
    g_bs_pacc_cap = acc;
  }
  return true;
}

// Split-KV chunked prefill (see ATTN_BLOCKQ_SPLIT_SRC). The K axis is cut
// into n_splits = ceil(N_kv/split_len) fixed uniform partitions (capped at
// 32; the cap re-derives split_len so the partition stays a pure function of
// N_kv -> deterministic). Tiles are dispatched in slabs bounded by a fixed
// scratch budget; slab boundaries never affect numerics (each row reduces
// over its own splits only). On ANY failure the caller falls through to the
// serial blockq_body which recomputes every row -- output is never mixed.
bool attention_blockq_splitkv_prefill(
  const unsigned short *q, const unsigned short *k, const unsigned short *v,
  unsigned short *o, int HQ, int HKV, int N_q, int N_kv, int cache_from, int d,
  int window, float softcap, int ring_cap, int split_len) {
  const int TM = 4; // must match blockq_body / the _split kernels
  int n_splits = (N_kv + split_len - 1) / split_len;
  if (n_splits >
      32) { // grid.y + reduce w[32] bound; re-derive deterministically
    split_len = (N_kv + 31) / 32;
    n_splits = (N_kv + split_len - 1) / split_len;
  }
  if (n_splits < 2)
    return false;
  // The split kernels read K/V through uint2/uint4 slices (bs_ldrow). All
  // in-row offsets are provably aligned (d and lane0*2B are multiples of the
  // width) but the pool packs tensors by cumulative byte size with NO
  // alignment padding, so the cache BASE can land misaligned -- verify here
  // and let the caller fall back to the serial kernel otherwise.
  const size_t amask = (d == 128) ? (size_t)7 : (size_t)15;
  if ((((size_t)k | (size_t)v) & amask) != 0) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      fprintf(stderr,
              "[cuda-splitkv] WARNING: K/V cache base not %zu-byte "
              "aligned; keeping the serial prefill kernel\n",
              amask + 1);
    }
    return false;
  }
  const char *fn = (d == 256)   ? "attn_blockq_split_d256"
                   : (d == 512) ? "attn_blockq_split_d512"
                                : "attn_blockq_split_d128";
  auto kp = CudaContext::Global().registerCudaKernel(ATTN_BLOCKQ_SPLIT_SRC, fn);
  auto kr = CudaContext::Global().registerCudaKernel(
    ATTN_BLOCKQ_SPLIT_SRC, "attn_blockq_split_reduce");
  if (!kp || !kr)
    return false;
  const int n_row_tiles = (N_q + TM - 1) / TM;
  const int n_grp = HQ * n_row_tiles;
  // scratch budget (pacc dominates): default 64 MiB, env-tunable. A pure
  // memory/duty-cycle knob -- numerics are slab-invariant.
  static const long budget_floats = []() {
    const char *e = std::getenv("NNTR_CUDA_SPLITKV_PREFILL_MB");
    long mb = e ? atol(e) : 0;
    if (mb <= 0)
      mb = 64;
    return mb * (long)(1 << 20) / (long)sizeof(float);
  }();
  const long per_tile_acc = (long)n_splits * TM * d;
  int tiles_per_slab = (int)(budget_floats / per_tile_acc);
  if (tiles_per_slab < 1)
    tiles_per_slab = 1;
  if (tiles_per_slab > n_grp)
    tiles_per_slab = n_grp;
  std::lock_guard<std::mutex> lk(g_bs_mtx);
  if (!ensure_bs((size_t)tiles_per_slab * n_splits * TM,
                 (size_t)tiles_per_slab * per_tile_acc))
    return false;
  const int pb[3] = {32, 1, 1};
  for (int grp0 = 0; grp0 < n_grp; grp0 += tiles_per_slab) {
    const int slab =
      (n_grp - grp0 < tiles_per_slab) ? (n_grp - grp0) : tiles_per_slab;
    kp->SetKernelArguments(0, &q, sizeof(q));
    kp->SetKernelArguments(1, &k, sizeof(k));
    kp->SetKernelArguments(2, &v, sizeof(v));
    kp->SetKernelArguments(3, &g_bs_pm, sizeof(g_bs_pm));
    kp->SetKernelArguments(4, &g_bs_pl, sizeof(g_bs_pl));
    kp->SetKernelArguments(5, &g_bs_pacc, sizeof(g_bs_pacc));
    kp->SetKernelArguments(6, &HQ, sizeof(HQ));
    kp->SetKernelArguments(7, &HKV, sizeof(HKV));
    kp->SetKernelArguments(8, &N_q, sizeof(N_q));
    kp->SetKernelArguments(9, &N_kv, sizeof(N_kv));
    kp->SetKernelArguments(10, &cache_from, sizeof(cache_from));
    kp->SetKernelArguments(11, &d, sizeof(d));
    kp->SetKernelArguments(12, &window, sizeof(window));
    kp->SetKernelArguments(13, &softcap, sizeof(softcap));
    kp->SetKernelArguments(14, &ring_cap, sizeof(ring_cap));
    kp->SetKernelArguments(15, &split_len, sizeof(split_len));
    kp->SetKernelArguments(16, &n_splits, sizeof(n_splits));
    kp->SetKernelArguments(17, &grp0, sizeof(grp0));
    // grid: x = tiles (fast axis -> same-split blocks run together, keeping
    // their K frontier L2-aligned), y = splits.
    const int pg[3] = {slab, n_splits, 1};
    if (!StreamManager::Global().DispatchCommand(*kp, pg, pb, 0))
      return false;
    kr->SetKernelArguments(0, &g_bs_pm, sizeof(g_bs_pm));
    kr->SetKernelArguments(1, &g_bs_pl, sizeof(g_bs_pl));
    kr->SetKernelArguments(2, &g_bs_pacc, sizeof(g_bs_pacc));
    kr->SetKernelArguments(3, &o, sizeof(o));
    kr->SetKernelArguments(4, &HQ, sizeof(HQ));
    kr->SetKernelArguments(5, &N_q, sizeof(N_q));
    kr->SetKernelArguments(6, &d, sizeof(d));
    kr->SetKernelArguments(7, &n_splits, sizeof(n_splits));
    kr->SetKernelArguments(8, &grp0, sizeof(grp0));
    const int rg[3] = {slab, 1, 1};
    if (!StreamManager::Global().DispatchCommand(*kr, rg, pb))
      return false;
  }
  return true;
}
} // namespace

// Pre-grow the split-KV decode scratch (g_pm/g_pl/g_pacc) to the model's max
// decode capacity at load. The M=1 split-KV path is only reached under graph
// capture once NNTR_CUDA_GRAPH is on; a cudaMalloc/Free inside
// cudaStreamBeginCapture..EndCapture invalidates the capture. Warming here
// (before any capture) makes every captured ensure_sk a pure cap-hit, so the
// fast flash-decode path stays usable under the graph. Idempotent (cap check).
bool cuda_attention_splitkv_prewarm(int max_seq_len, int max_hq,
                                    int max_head_dim) {
  const char *e = std::getenv("NNTR_CUDA_FLASH_DECODE");
  if (!e)
    return true; // split-KV off -> no scratch needed (mirror of interleaved)
  int chunk = atoi(e);
  if (chunk <= 0)
    chunk = 64;
  if (max_seq_len <= 0 || max_hq <= 0 || max_head_dim <= 0)
    return true;
  const int max_nchunks = (max_seq_len + chunk - 1) / chunk;
  const size_t mn = (size_t)max_hq * (size_t)max_nchunks;
  std::lock_guard<std::mutex> lk(g_sk_mtx);
  if (!ensure_sk(mn, mn * (size_t)max_head_dim))
    return false;
  // Publish the fixed stride so attention_splitkv_decode + the kernels all use
  // the SAME value the scratch was just sized with (M2-B capture correctness).
  g_sk_max_nchunks = max_nchunks;
  return true;
}

// =============================================================================
// [cuda-kvi8] int8 KV-cache path (NNTR_KV_INT8 on engine=cuda).
// Mirrors the OpenCL kvi8 design: cache format is the CPU-verified concat
// [row][HKV*d] signed int8 (K roped BEFORE quantization, symmetric amax/127)
// with per-(row,head_kv) fp16 scales at scale[row*HKV + h]; rows are
// ring-indexed when ring_cap > 0 (scales in lockstep). Both scales fold OUT
// of the inner loops: the K scale multiplies the finished dot, the V scale
// pre-multiplies p (decode) / v_reg (prefill).
// =============================================================================
static const char *ATTN_I8_SRC = R"CU(
__device__ __forceinline__ float i8_h2f(unsigned short h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short i8_f2h(float f) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
__device__ __forceinline__ float i8_wreduce(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}
__device__ __forceinline__ float i8_wreduce_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}
// dp4a: 4-way int8 dot with int32 accumulate. Inline PTX like the cvt
// helpers above -- NVRTC needs no header for it. Below sm_61 (no dp4a
// instruction; cubin compile validates the asm against the real arch) fall
// back to exact byte math so the shared ATTN_I8_SRC module still compiles
// and the _dp4a kernels stay correct everywhere.
__device__ __forceinline__ int i8_dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
  int r;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
#else
  int r = c;
#pragma unroll
  for (int bb = 0; bb < 4; bb++)
    r += (int)((signed char)(a >> (8 * bb))) * (int)((signed char)(b >> (8 * bb)));
  return r;
#endif
}
// Quantize one fp16 step row (token s, head_kv h) into the int8 cache +
// fp16 amax/127 scale, at the ring-aware physical row. Matches the host
// quantize_kv_fp16_to_int8_per_row exactly: symmetric, round-half-away-
// from-zero (roundf), clamp [-127,127], scale = amax/127 (0 when amax==0).
extern "C" __global__ void kv_quant_i8(const unsigned short *src, signed char *dst,
                            unsigned short *scale, int step_n, int HKV, int d,
                            int cache_from, int ring_cap,
                            const int *d_pos) {
  int s = blockIdx.x, h = blockIdx.y;
  if (s >= step_n || h >= HKV) return;
  // [d_pos] M2-B graph replay: read the live write position from the device
  // pos buffer so ONE captured decode graph serves every token.
  int cf = d_pos ? d_pos[0] : cache_from;
  const unsigned short *row = src + ((long)s * HKV + h) * d;
  long abs_row = (long)cf + s;
  long prow = (ring_cap > 0) ? (abs_row % ring_cap) : abs_row;
  signed char *drow = dst + (prow * HKV + h) * (long)d;
  int tid = threadIdx.x, B = blockDim.x;
  __shared__ float red[256];
  float am = 0.f;
  for (int x = tid; x < d; x += B) {
    float v = fabsf(i8_h2f(row[x]));
    am = fmaxf(am, v);
  }
  red[tid] = am;
  __syncthreads();
  for (int st = B >> 1; st > 0; st >>= 1) {
    if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
    __syncthreads();
  }
  float amax = red[0];
  float inv = amax > 0.f ? 127.f / amax : 0.f;
  for (int x = tid; x < d; x += B) {
    int qv = (int)roundf(i8_h2f(row[x]) * inv);
    qv = max(-127, min(127, qv));
    drow[x] = (signed char)qv;
  }
  if (tid == 0)
    scale[prow * HKV + h] = i8_f2h(amax / 127.f);
}
// int8 block-Q prefill: identical structure to blockq_body (warp per
// (head, TM=4 row tile), warp-shuffle d-dot, ring modulo, n_lo window skip)
// with int8 K/V loads and the scales folded outside the dot loops.
// DP4A=1 additionally quantizes Q per (row, head) with the same symmetric
// amax/127 convention and runs the QK dot as packed int8 dp4a words; the
// int32 lane partials stay exact through the float shuffle-reduce
// (|dot| <= 127*127*512 < 2^24). The Q scale folds onto the finished score
// next to the K scale, so softmax/PV are untouched.
template <int TM, int VPL, int DP4A>
__device__ __forceinline__ void
blockq_body_i8(const unsigned short *q, const signed char *k,
               const signed char *v, const unsigned short *kscale,
               const unsigned short *vscale, unsigned short *o, int HQ,
               int HKV, int N_q, int N_kv, int cache_from, int d, int window,
               float softcap, int ring_cap) {
  const int lane = threadIdx.x;
  const int grp = blockIdx.x;
  const int n_row_tiles = (N_q + TM - 1) / TM;
  const int head_q = grp / n_row_tiles;
  const int tile = grp % n_row_tiles;
  const int m0 = tile * TM;
  if (head_q >= HQ || m0 >= N_q) return;
  const int gqa = HQ / HKV, hkv = head_q / gqa;
  const int HD_Q = HQ * d, HD_KV = HKV * d;
  const float scale = rsqrtf((float)d);
  const int lane0 = lane * VPL;
  float q_reg[TM][VPL], acc_reg[TM][VPL], m_i[TM], l_i[TM];
  int valid[TM];
#pragma unroll
  for (int r = 0; r < TM; r++) {
    int m = m0 + r; valid[r] = (m < N_q) ? 1 : 0; m_i[r] = -1e30f; l_i[r] = 0.f;
    long q_base = (long)(valid[r] ? m : 0) * HD_Q + (long)head_q * d;
#pragma unroll
    for (int vv = 0; vv < VPL; vv++) {
      q_reg[r][vv] = i8_h2f(q[q_base + lane0 + vv]); acc_reg[r][vv] = 0.f;
    }
  }
  // [dp4a] per-(row,head) symmetric Q quantization: amax over the full d via
  // warp max-reduce, then each lane packs its VPL int8 into VPL/4 dp4a words.
  float q_scl[TM];
  int q_pk[TM][VPL / 4];
  if (DP4A) {
#pragma unroll
    for (int r = 0; r < TM; r++) {
      float am = 0.f;
#pragma unroll
      for (int vv = 0; vv < VPL; vv++) am = fmaxf(am, fabsf(q_reg[r][vv]));
      am = i8_wreduce_max(am);
      float inv = am > 0.f ? 127.f / am : 0.f;
      q_scl[r] = am / 127.f;
#pragma unroll
      for (int p = 0; p < VPL / 4; p++) {
        unsigned int w = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) {
          int qv = (int)roundf(q_reg[r][p * 4 + b] * inv);
          qv = max(-127, min(127, qv));
          w |= ((unsigned int)(qv & 0xff)) << (8 * b);
        }
        q_pk[r][p] = (int)w;
      }
    }
  }
  const int q_pos_off = cache_from;
  int last_row = ((m0 + TM - 1 < N_q) ? (m0 + TM - 1) : (N_q - 1)) + q_pos_off;
  int n_last = (N_kv - 1 < last_row) ? (N_kv - 1) : last_row;
  int n_lo = (window > 0) ? (m0 + q_pos_off - window + 1) : 0;
  if (n_lo < 0) n_lo = 0;
  for (int n = n_lo; n <= n_last; ++n) {
    long pn = (ring_cap > 0) ? (long)(n % ring_cap) : (long)n;
    long kv_base = pn * HD_KV + (long)hkv * d;
    // per-key scales (ring-lockstep row index)
    const float ks = scale * i8_h2f(kscale[pn * HKV + hkv]);
    const float vs = i8_h2f(vscale[pn * HKV + hkv]);
    float sdot[TM];
    if (DP4A) {
      // K lane slice as packed dp4a words. kv_base/lane0 are multiples of 4;
      // the K/V BASE pointers are verified 4-aligned by the host dispatch
      // (the tensor pool packs by byte size with no alignment guarantee).
      const int *kp = (const int *)(k + kv_base + lane0);
      int k_pk[VPL / 4];
#pragma unroll
      for (int p = 0; p < VPL / 4; p++) k_pk[p] = kp[p];
#pragma unroll
      for (int r = 0; r < TM; r++) {
        int di = 0;
#pragma unroll
        for (int p = 0; p < VPL / 4; p++) di = i8_dp4a(q_pk[r][p], k_pk[p], di);
        sdot[r] = i8_wreduce((float)di);
      }
    } else {
      // byte loads: ld.global.s8 has no alignment requirement, and the pool
      // gives the cache base none -- keep the always-on path unconditional.
      float k_reg[VPL];
#pragma unroll
      for (int vv = 0; vv < VPL; vv++)
        k_reg[vv] = (float)k[kv_base + lane0 + vv];
#pragma unroll
      for (int r = 0; r < TM; r++) {
        float p = 0.f;
#pragma unroll
        for (int vv = 0; vv < VPL; vv++) p += q_reg[r][vv] * k_reg[vv];
        sdot[r] = i8_wreduce(p);
      }
    }
    float v_reg[VPL];
    if (DP4A) {
      // packed words: base alignment verified by the host dispatch.
      const int *vp = (const int *)(v + kv_base + lane0);
#pragma unroll
      for (int p2 = 0; p2 < VPL / 4; p2++) {
        int w = vp[p2];
#pragma unroll
        for (int b = 0; b < 4; b++)
          v_reg[p2 * 4 + b] = vs * (float)((signed char)(w >> (8 * b)));
      }
    } else {
#pragma unroll
      for (int vv = 0; vv < VPL; vv++)
        v_reg[vv] = vs * (float)v[kv_base + lane0 + vv];
    }
#pragma unroll
    for (int r = 0; r < TM; r++) {
      int m = m0 + r + q_pos_off;
      if (!valid[r] || n > m || (window > 0 && n + window <= m)) continue;
      float s = ks * sdot[r];
      if (DP4A) s *= q_scl[r];
      if (softcap > 0.f) s = softcap * tanhf(s / softcap);
      float m_new = fmaxf(m_i[r], s), alpha = __expf(m_i[r] - m_new),
            pp = __expf(s - m_new);
#pragma unroll
      for (int vv = 0; vv < VPL; vv++)
        acc_reg[r][vv] = alpha * acc_reg[r][vv] + pp * v_reg[vv];
      l_i[r] = alpha * l_i[r] + pp; m_i[r] = m_new;
    }
  }
#pragma unroll
  for (int r = 0; r < TM; r++) {
    if (!valid[r]) continue;
    float inv = l_i[r] > 0.f ? 1.f / l_i[r] : 0.f;
    long o_base = (long)(m0 + r) * HD_Q + (long)head_q * d;
#pragma unroll
    for (int vv = 0; vv < VPL; vv++)
      o[o_base + lane0 + vv] = i8_f2h(acc_reg[r][vv] * inv);
  }
}
extern "C" __global__ void attn_blockq_i8_d128(const unsigned short *q,
    const signed char *k, const signed char *v, const unsigned short *kscale,
    const unsigned short *vscale, unsigned short *o, int HQ, int HKV, int N_q,
    int N_kv, int cache_from, int d, int window, float softcap, int ring_cap) {
  blockq_body_i8<4, 4, 0>(q, k, v, kscale, vscale, o, HQ, HKV, N_q, N_kv,
                          cache_from, d, window, softcap, ring_cap);
}
extern "C" __global__ void attn_blockq_i8_d256(const unsigned short *q,
    const signed char *k, const signed char *v, const unsigned short *kscale,
    const unsigned short *vscale, unsigned short *o, int HQ, int HKV, int N_q,
    int N_kv, int cache_from, int d, int window, float softcap, int ring_cap) {
  blockq_body_i8<4, 8, 0>(q, k, v, kscale, vscale, o, HQ, HKV, N_q, N_kv,
                          cache_from, d, window, softcap, ring_cap);
}
extern "C" __global__ void attn_blockq_i8_d512(const unsigned short *q,
    const signed char *k, const signed char *v, const unsigned short *kscale,
    const unsigned short *vscale, unsigned short *o, int HQ, int HKV, int N_q,
    int N_kv, int cache_from, int d, int window, float softcap, int ring_cap) {
  blockq_body_i8<4, 16, 0>(q, k, v, kscale, vscale, o, HQ, HKV, N_q, N_kv,
                           cache_from, d, window, softcap, ring_cap);
}
// dp4a variants (NNTR_CUDA_KVI8_DP4A=1): int8 arithmetic QK^T, same layout.
extern "C" __global__ void attn_blockq_i8_d128_dp4a(const unsigned short *q,
    const signed char *k, const signed char *v, const unsigned short *kscale,
    const unsigned short *vscale, unsigned short *o, int HQ, int HKV, int N_q,
    int N_kv, int cache_from, int d, int window, float softcap, int ring_cap) {
  blockq_body_i8<4, 4, 1>(q, k, v, kscale, vscale, o, HQ, HKV, N_q, N_kv,
                          cache_from, d, window, softcap, ring_cap);
}
extern "C" __global__ void attn_blockq_i8_d256_dp4a(const unsigned short *q,
    const signed char *k, const signed char *v, const unsigned short *kscale,
    const unsigned short *vscale, unsigned short *o, int HQ, int HKV, int N_q,
    int N_kv, int cache_from, int d, int window, float softcap, int ring_cap) {
  blockq_body_i8<4, 8, 1>(q, k, v, kscale, vscale, o, HQ, HKV, N_q, N_kv,
                          cache_from, d, window, softcap, ring_cap);
}
extern "C" __global__ void attn_blockq_i8_d512_dp4a(const unsigned short *q,
    const signed char *k, const signed char *v, const unsigned short *kscale,
    const unsigned short *vscale, unsigned short *o, int HQ, int HKV, int N_q,
    int N_kv, int cache_from, int d, int window, float softcap, int ring_cap) {
  blockq_body_i8<4, 16, 1>(q, k, v, kscale, vscale, o, HQ, HKV, N_q, N_kv,
                           cache_from, d, window, softcap, ring_cap);
}
// int8 split-KV decode partial: identical to attn_partial with int8 K/V and
// the scales folded (K onto the score, V onto p). Reduce stage is the shared
// fp32 attn_reduce from ATTN_SPLITKV_SRC. d_pos/M2-B is supported since the
// d_pos indirection (57e8a7b82); see the [d_pos] note inside attn_partial_i8.
extern "C" __global__ void attn_partial_i8(const unsigned short *q, const signed char *k,
                                const signed char *v, float *pm, float *pl,
                                float *pacc, int HQ, int HKV, int N_kv,
                                int cache_from, int d, int window,
                                float softcap, int chunk_kv, int n_chunks,
                                int ring_cap, const unsigned short *kscale,
                                const unsigned short *vscale,
                                const int *d_pos, int max_n_chunks) {
  int h = blockIdx.x, c = blockIdx.y;
  // [d_pos] M2-B: live position/key-count from the device buffer; the grid is
  // captured at gridDim.y = max_n_chunks, blocks past the live chunk count
  // early-return, and the partial stride uses the FIXED max_n_chunks so the
  // writer and attn_reduce always agree (mirrors the fp16 attn_partial).
  int cf = d_pos ? d_pos[0] : cache_from;
  int nkv = d_pos ? d_pos[1] : N_kv;
  int live_nchunks = (nkv + chunk_kv - 1) / chunk_kv;
  if (c >= live_nchunks) return;
  int gqa = HQ / HKV, hkv = h / gqa;
  int HD_KV = HKV * d;
  const unsigned short *qrow = q + (long)h * d;
  int tid = threadIdx.x, B = blockDim.x;
  extern __shared__ float sh[];
  float *Qsh = sh; float *red = sh + d;
  for (int dd = tid; dd < d; dd += B) Qsh[dd] = i8_h2f(qrow[dd]);
  __syncthreads();
  float scale = rsqrtf((float)d);
  int i_abs = cf;
  int j_lo_g = i_abs - window + 1; if (j_lo_g < 0) j_lo_g = 0;
  int j_hi_g = i_abs; if (j_hi_g >= nkv) j_hi_g = nkv - 1;
  int j_lo = c * chunk_kv; if (j_lo < j_lo_g) j_lo = j_lo_g;
  int j_hi = (c + 1) * chunk_kv - 1; if (j_hi > j_hi_g) j_hi = j_hi_g;
  float acc[4];
#pragma unroll
  for (int r = 0; r < 4; r++) acc[r] = 0.f;
  float mmax = -1e30f, l = 0.f;
  for (int j = j_lo; j <= j_hi; ++j) {
    long pj = (ring_cap > 0) ? (long)(j % ring_cap) : (long)j;
    const signed char *kr = k + pj * HD_KV + (long)hkv * d;
    const float ksj = scale * i8_h2f(kscale[pj * HKV + hkv]);
    const float vsj = i8_h2f(vscale[pj * HKV + hkv]);
    float pd = 0.f;
    for (int dd = tid; dd < d; dd += B) pd += Qsh[dd] * (float)kr[dd];
    red[tid] = pd; __syncthreads();
    for (int s = B >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
    float score = red[0] * ksj; __syncthreads();
    if (softcap > 0.f) score = softcap * tanhf(score / softcap);
    float mn = fmaxf(mmax, score), corr = __expf(mmax - mn), p = __expf(score - mn);
    l = l * corr + p; mmax = mn;
    const signed char *vr = v + pj * HD_KV + (long)hkv * d;
    float pv = p * vsj;
    int r = 0; for (int dd = tid; dd < d; dd += B, ++r) acc[r] = acc[r]*corr + pv*(float)vr[dd];
  }
  if (j_lo > j_hi) { mmax = -1e30f; l = 0.f; }
  int base = h * max_n_chunks + c;
  if (tid == 0) { pm[base] = mmax; pl[base] = l; }
  int r = 0; for (int dd = tid; dd < d; dd += B, ++r) pacc[(long)base * d + dd] = acc[r];
}
)CU";

bool cuda_kv_int8_quantize(const unsigned short *src_fp16, signed char *dst_i8,
                           unsigned short *scale_fp16, int step_n, int HKV,
                           int d, int cache_from, int ring_cap) {
  if (step_n <= 0 || HKV <= 0 || d <= 0 || d > 4096)
    return false;
  auto kq =
    CudaContext::Global().registerCudaKernel(ATTN_I8_SRC, "kv_quant_i8");
  if (!kq)
    return false;
  const int B = 128;
  kq->SetKernelArguments(0, &src_fp16, sizeof(src_fp16));
  kq->SetKernelArguments(1, &dst_i8, sizeof(dst_i8));
  kq->SetKernelArguments(2, &scale_fp16, sizeof(scale_fp16));
  kq->SetKernelArguments(3, &step_n, sizeof(step_n));
  kq->SetKernelArguments(4, &HKV, sizeof(HKV));
  kq->SetKernelArguments(5, &d, sizeof(d));
  kq->SetKernelArguments(6, &cache_from, sizeof(cache_from));
  kq->SetKernelArguments(7, &ring_cap, sizeof(ring_cap));
  // decode + M2-B: live write position via the device pos buffer (prefill
  // stays baked -- prefill is never graph-captured).
  static const bool _m2b_q = nntr_env_on("NNTR_CUDA_M2B");
  const int *dpos_q = (_m2b_q && step_n == 1) ? cuda_pos_buffer() : nullptr;
  kq->SetKernelArguments(8, &dpos_q, sizeof(dpos_q));
  const int grid[3] = {step_n, HKV, 1};
  const int block[3] = {B, 1, 1};
  return StreamManager::Global().DispatchCommand(*kq, grid, block, 0);
}

static bool attention_splitkv_decode_i8(
  const unsigned short *q, const signed char *k, const signed char *v,
  const unsigned short *kscale, const unsigned short *vscale, unsigned short *o,
  int HQ, int HKV, int N_kv, int cache_from, int d, int window, float softcap,
  int chunk_kv, int ring_cap) {
  const int n_chunks = (N_kv + chunk_kv - 1) / chunk_kv;
  // M2-B: fixed max-chunk grid/stride from the prewarm-published
  // g_sk_max_nchunks so one captured graph serves every token (fp16 idiom).
  static const bool _m2b_i8 = nntr_env_on("NNTR_CUDA_M2B");
  const int *dpos =
    (_m2b_i8 && g_sk_max_nchunks > 0) ? cuda_pos_buffer() : nullptr;
  const int max_nc = dpos ? g_sk_max_nchunks : n_chunks;
  std::lock_guard<std::mutex> lk(g_sk_mtx);
  if (!ensure_sk((size_t)HQ * max_nc, (size_t)HQ * max_nc * d))
    return false;
  auto kp =
    CudaContext::Global().registerCudaKernel(ATTN_I8_SRC, "attn_partial_i8");
  auto kr =
    CudaContext::Global().registerCudaKernel(ATTN_SPLITKV_SRC, "attn_reduce");
  if (!kp || !kr)
    return false;
  const int B = 128;
  kp->SetKernelArguments(0, &q, sizeof(q));
  kp->SetKernelArguments(1, &k, sizeof(k));
  kp->SetKernelArguments(2, &v, sizeof(v));
  kp->SetKernelArguments(3, &g_pm, sizeof(g_pm));
  kp->SetKernelArguments(4, &g_pl, sizeof(g_pl));
  kp->SetKernelArguments(5, &g_pacc, sizeof(g_pacc));
  kp->SetKernelArguments(6, &HQ, sizeof(HQ));
  kp->SetKernelArguments(7, &HKV, sizeof(HKV));
  kp->SetKernelArguments(8, &N_kv, sizeof(N_kv));
  kp->SetKernelArguments(9, &cache_from, sizeof(cache_from));
  kp->SetKernelArguments(10, &d, sizeof(d));
  kp->SetKernelArguments(11, &window, sizeof(window));
  kp->SetKernelArguments(12, &softcap, sizeof(softcap));
  kp->SetKernelArguments(13, &chunk_kv, sizeof(chunk_kv));
  kp->SetKernelArguments(14, &n_chunks, sizeof(n_chunks));
  kp->SetKernelArguments(15, &ring_cap, sizeof(ring_cap));
  kp->SetKernelArguments(16, &kscale, sizeof(kscale));
  kp->SetKernelArguments(17, &vscale, sizeof(vscale));
  kp->SetKernelArguments(18, &dpos, sizeof(dpos));
  kp->SetKernelArguments(19, &max_nc, sizeof(max_nc));
  const int pg[3] = {HQ, max_nc, 1};
  const int pb[3] = {B, 1, 1};
  const unsigned int shmem = (unsigned int)(sizeof(float) * ((size_t)d + B));
  if (!StreamManager::Global().DispatchCommand(*kp, pg, pb, shmem))
    return false;
  kr->SetKernelArguments(0, &g_pm, sizeof(g_pm));
  kr->SetKernelArguments(1, &g_pl, sizeof(g_pl));
  kr->SetKernelArguments(2, &g_pacc, sizeof(g_pacc));
  kr->SetKernelArguments(3, &o, sizeof(o));
  kr->SetKernelArguments(4, &HQ, sizeof(HQ));
  kr->SetKernelArguments(5, &d, sizeof(d));
  kr->SetKernelArguments(6, &n_chunks, sizeof(n_chunks));
  kr->SetKernelArguments(7, &dpos, sizeof(dpos));
  kr->SetKernelArguments(8, &chunk_kv, sizeof(chunk_kv));
  kr->SetKernelArguments(9, &max_nc, sizeof(max_nc));
  // [splitkv-clip] the int8 partial kernel still writes the whole-context chunk
  // grid, so the reduce must read it unclipped (clip = 0 -> previous formula).
  const int no_clip = 0;
  kr->SetKernelArguments(10, &window, sizeof(window));
  kr->SetKernelArguments(11, &cache_from, sizeof(cache_from));
  kr->SetKernelArguments(12, &N_kv, sizeof(N_kv));
  kr->SetKernelArguments(13, &no_clip, sizeof(no_clip));
  const int rg[3] = {HQ, 1, 1};
  const int rb[3] = {B, 1, 1};
  return StreamManager::Global().DispatchCommand(*kr, rg, rb);
}

bool cuda_attention_interleaved_kvi8(
  const unsigned short *q_fp16, const signed char *k_i8,
  const signed char *v_i8, const unsigned short *k_scale,
  const unsigned short *v_scale, unsigned short *o_fp16, int num_heads_Q,
  int num_heads_KV, int N_q, int N_kv, int cache_from, int head_dim, int window,
  float softcap, int ring_cap) {
  if (num_heads_Q == 0 || N_q == 0 || N_kv == 0 || head_dim == 0)
    return true;
  if (!dev_accessible(q_fp16) || !dev_accessible(k_i8) ||
      !dev_accessible(v_i8) || !dev_accessible(k_scale) ||
      !dev_accessible(v_scale) || !dev_accessible(o_fp16))
    return false;
  // decode: split-KV int8 partials + shared fp32 reduce
  static const int sk_chunk = []() {
    const char *e = std::getenv("NNTR_CUDA_FLASH_DECODE");
    if (!e)
      return 64; // kvi8 decode has no dense fallback -- default the chunk on
    int c = atoi(e);
    return c > 0 ? c : 64;
  }();
  if (N_q == 1) {
    const int ck = (N_kv > sk_chunk) ? sk_chunk : N_kv;
    return attention_splitkv_decode_i8(q_fp16, k_i8, v_i8, k_scale, v_scale,
                                       o_fp16, num_heads_Q, num_heads_KV, N_kv,
                                       cache_from, head_dim, window, softcap,
                                       ck, ring_cap) &&
           (StreamManager::Global().maybeFinish(), true);
  }
  // prefill: int8 block-Q. NNTR_CUDA_KVI8_DP4A=1 routes to the dp4a variant
  // (int8-arithmetic QK^T with per-row Q quantization) -- experimental,
  // default off: it adds Q quantization noise on top of the K/V int8 loss.
  if (head_dim != 128 && head_dim != 256 && head_dim != 512)
    return false;
  static const bool _dp4a_on = nntr_env_on("NNTR_CUDA_KVI8_DP4A");
  // The dp4a kernel reads K/V as packed int words. Offsets inside the cache
  // are provably 4-aligned (d and VPL are multiples of 4) but the memory
  // planners pack pool tensors by cumulative byte size with NO alignment
  // padding, so the cache BASE itself can land misaligned (e.g. after an
  // odd-element fp16 scale tensor). Verify it here; fall back to the scalar
  // int8 kernel otherwise.
  const bool dp4a_ok =
    _dp4a_on && (((size_t)k_i8 | (size_t)v_i8) & (size_t)3) == 0;
  if (_dp4a_on && !dp4a_ok) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      fprintf(stderr,
              "[cuda-kvi8] WARNING: KVI8_DP4A requested but the int8 K/V "
              "cache base is not 4-byte aligned; using the scalar kernel\n");
    }
  }
  const char *fn = dp4a_ok ? ((head_dim == 256)   ? "attn_blockq_i8_d256_dp4a"
                              : (head_dim == 512) ? "attn_blockq_i8_d512_dp4a"
                                                  : "attn_blockq_i8_d128_dp4a")
                           : ((head_dim == 256)   ? "attn_blockq_i8_d256"
                              : (head_dim == 512) ? "attn_blockq_i8_d512"
                                                  : "attn_blockq_i8_d128");
  auto kb = CudaContext::Global().registerCudaKernel(ATTN_I8_SRC, fn);
  if (!kb)
    return false;
  int win_bq = (window <= 0 || window >= N_kv) ? 0 : window;
  const int TM = 4;
  const int n_row_tiles = (N_q + TM - 1) / TM;
  kb->SetKernelArguments(0, &q_fp16, sizeof(q_fp16));
  kb->SetKernelArguments(1, &k_i8, sizeof(k_i8));
  kb->SetKernelArguments(2, &v_i8, sizeof(v_i8));
  kb->SetKernelArguments(3, &k_scale, sizeof(k_scale));
  kb->SetKernelArguments(4, &v_scale, sizeof(v_scale));
  kb->SetKernelArguments(5, &o_fp16, sizeof(o_fp16));
  kb->SetKernelArguments(6, &num_heads_Q, sizeof(num_heads_Q));
  kb->SetKernelArguments(7, &num_heads_KV, sizeof(num_heads_KV));
  kb->SetKernelArguments(8, &N_q, sizeof(N_q));
  kb->SetKernelArguments(9, &N_kv, sizeof(N_kv));
  kb->SetKernelArguments(10, &cache_from, sizeof(cache_from));
  kb->SetKernelArguments(11, &head_dim, sizeof(head_dim));
  kb->SetKernelArguments(12, &win_bq, sizeof(win_bq));
  kb->SetKernelArguments(13, &softcap, sizeof(softcap));
  kb->SetKernelArguments(14, &ring_cap, sizeof(ring_cap));
  const int grid[3] = {num_heads_Q * n_row_tiles, 1, 1};
  const int block[3] = {32, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kb, grid, block, 0))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

bool cuda_attention_interleaved_fp16(const unsigned short *q_fp16,
                                     const unsigned short *k_fp16,
                                     const unsigned short *v_fp16,
                                     unsigned short *o_fp16, int num_heads_Q,
                                     int num_heads_KV, int N_q, int N_kv,
                                     int cache_from, int head_dim, int window,
                                     float softcap, int ring_cap) {
  if (num_heads_Q == 0 || N_q == 0 || N_kv == 0 || head_dim == 0)
    return true;

  // mirror the KV cache to the device if it is host-resident (engine=cuda KV
  // cache is not UVM). K/V are [N_kv, num_heads_KV*head_dim] interleaved.
  //
  // [kv-window-ring] N_kv is the LOGICAL key count. With the ring active the
  // host cache holds only ring_cap rows and every kernel here modulo-maps
  // (row = n % ring_cap), so the mirror must copy the PHYSICAL row count:
  // copying N_kv rows reads past the end of the cache as soon as the context
  // outgrows the window. Same rule the OpenCL path applies to its read view
  // (min(cache_to, kv_ring_cap) in MHACore).
  const int kv_rows = (ring_cap > 0) ? std::min(N_kv, ring_cap) : N_kv;
  const size_t kv_elems = (size_t)kv_rows * num_heads_KV * head_dim;
  k_fp16 = mirror_kv(k_fp16, kv_elems);
  v_fp16 = mirror_kv(v_fp16, kv_elems);
  if (!k_fp16 || !v_fp16)
    return false;

  // Flash-decoding (split-KV) for M=1 decode with enough keys to fill the SMs.
  static const int sk_chunk = []() {
    const char *e = std::getenv("NNTR_CUDA_FLASH_DECODE");
    if (!e)
      return 0; // off
    int c = atoi(e);
    return c > 0 ? c : 64; // =1 -> default chunk 64; or an explicit chunk size
  }();
  if (sk_chunk > 0 && N_q == 1 && N_kv > sk_chunk) {
    if (attention_splitkv_decode(q_fp16, k_fp16, v_fp16, o_fp16, num_heads_Q,
                                 num_heads_KV, N_kv, cache_from, head_dim,
                                 window, softcap, sk_chunk, ring_cap)) {
      StreamManager::Global().maybeFinish();
      return true;
    }
  }

  // GEMM prefill attention (cuBLAS fp16 QK^T -> softmax -> PV): materialises
  // scores instead of the per-key flash reduce, so it is far faster for prefill
  // and head_dim-agnostic -- the lever for head_dim=128 (qwen3/llama) where
  // block-Q underperforms. Opt-in (NNTR_CUDA_GEMM_ATTN); falls through on any
  // cuBLAS/registration failure.
  // head_dim 256/512 are faster on block-Q (warp-shuffle, K/V reuse); GEMM wins
  // for the smaller head dims (128 = qwen3/llama) where block-Q underutilises.
  // NNTR_CUDA_GEMM_ATTN: unset -> caps/shape-derived default (below); =0 ->
  // kill switch (block-Q everywhere, the pre-lever behaviour byte-for-byte);
  // =1 -> force ON for every layer and every shape (the historical opt-in);
  // =N>1 -> default gating with the long-context threshold set to N keys.
  static const int gemm_attn_mode = []() {
    const char *e = std::getenv("NNTR_CUDA_GEMM_ATTN");
    if (e == nullptr || e[0] == '\0')
      return -1; // default
    const int v = atoi(e);
    return (v <= 0) ? 0 : v; // 0 = off, 1 = force, N>1 = threshold
  }();
  static const bool integrated_gpu = ContextManager::Global().isIntegrated();
  // window<=0 or window>=N_kv -> no sliding mask (full causal); mha passes
  // INT_MAX for global layers. Hoisted here because the GEMM default keys off
  // it; the block-Q path below reuses the same value (identical semantic).
  const int win_bq = (window <= 0 || window >= N_kv) ? 0 : window;
  // Default: integrated GPUs (Orin sm_87) need the cuBLAS GEMM attention for
  // every layer because block-Q runs ~0.2 TFLOP/s there. On discrete GPUs the
  // long-context FULL-attention layers (win_bq==0) also want it: their key
  // walk is unbounded, so block-Q's cost grows with N_kv^2 while the cuBLAS
  // Tensor-Core QK/PV does not. Measured on RTX 5060, 29K prefill: 2742 ->
  // 6924 TPS. The sliding layers keep block-Q (their walk is already bounded
  // by the window, and past the ring wrap the guard below excludes them
  // anyway), and so do short contexts: at N_kv<=1K block-Q wins outright
  // (1K cell measured 8019 -> 4148 TPS when the GEMM path is taken), so the
  // path only engages from GEMM_ATTN_MIN_KV keys up. That threshold matches
  // the block-Q split-prefill engage point, so 1K runs stay byte-identical.
  // Safe: any cuBLAS/registration failure falls through to block-Q, never
  // wrong output.
  constexpr int GEMM_ATTN_MIN_KV = 4096;
  const int gemm_min_kv =
    (gemm_attn_mode > 1) ? gemm_attn_mode : GEMM_ATTN_MIN_KV;
  const bool gemm_attn_on =
    (gemm_attn_mode == 1) ||
    (gemm_attn_mode != 0 &&
     (integrated_gpu || (win_bq == 0 && N_kv >= gemm_min_kv)));
  // head_dim 256/512 (gemma4 sliding/global) were historically excluded because
  // block-Q beat the cuBLAS path on RTX/Adreno. On Orin (sm_87) block-Q runs at
  // only ~0.2 TFLOP/s, so the cuBLAS int8/fp16 Tensor-Core QK/PV is worth
  // trying here -- opt-in via NNTR_CUDA_GEMM_ATTN, so other arches keep
  // block-Q.
  // [kv-window-ring] attention_gemm_prefill_fp16 hands K/V straight to cuBLAS
  // as dense [N_kv, d] matrices, so it can only be used while the ring has not
  // wrapped (logical rows <= ring_cap). Past that the physical cache holds only
  // ring_cap rows and a linear walk both reads past the mirror and pairs the
  // wrong key with each query; fall through to the ring-mapping kernels below.
  const bool ring_linear = (ring_cap <= 0) || (N_kv <= ring_cap);
  if (gemm_attn_on && N_q > 1 && ring_linear) {
    if (attention_gemm_prefill_fp16(q_fp16, k_fp16, v_fp16, o_fp16, num_heads_Q,
                                    num_heads_KV, N_q, N_kv, cache_from,
                                    head_dim, window, softcap)) {
      StreamManager::Global().maybeFinish();
      return true;
    }
  }

  // Block-Q multi-row prefill: one warp per (head, TM=4 row tile), warp-shuffle
  // d-dot, K/V reused across rows. 3-4x faster than the per-key LDS-reduce
  // attn_core_il_fp16 below, fp16-identical. Opt-in (NNTR_CUDA_BLOCKQ) until
  // folded; only the multi-row (prefill) path with head_dim in {256, 512}
  // (gemma4 sliding/global) -- decode (N_q==1) keeps split-KV above.
  static const bool blockq_on = nntr_env_on("NNTR_CUDA_BLOCKQ");
  if (blockq_on && N_q > 1 &&
      (head_dim == 128 || head_dim == 256 || head_dim == 512)) {
    // [splitkv-prefill] NNTR_CUDA_SPLITKV_PREFILL: unset/=1 -> ON with the
    // default 4096-key split (engage threshold == split_len, so any context
    // at or below it takes the serial kernel VERBATIM -- 1K runs are
    // bit-unchanged); =N>1 -> custom split length; =0 -> OFF (kill-switch:
    // reproduces the pre-lever serial streams byte-for-byte). Global/full
    // layers only (win_bq==0): the sliding layers' key walk is already
    // bounded by the n_lo window skip.
    // DEFAULT ON (owner decision + golden re-baseline, qwen3 precedent):
    // deterministic (byte-stable run to run), numerics fp64-probe-equal to
    // the serial kernel (<=1 fp16 ulp on 0.67% of outputs, equal rms vs an
    // fp64 reference); the d512 32K stream is byte-identical to serial; the
    // d128 32K cell flips ONE near-tie argmax in the generated continuation
    // -- that stream is re-baselined as the new canonical golden. Measured
    // (RTX 5060, 32K/chunk1024/ring profile cells): d512-global model
    // 1760.3 -> 2850.3 prefill TPS (+62%), d128-global model 2489.6 ->
    // 2678.7 (+7.6%), decode untouched, 1K cells byte-identical (below
    // threshold), +64 MiB scratch.
    static const int sp_len = []() {
      const char *e = std::getenv("NNTR_CUDA_SPLITKV_PREFILL");
      if (!e || e[0] == '\0')
        return 4096; // default ON (owner decision, see above)
      int v_ = atoi(e);
      if (v_ <= 0)
        return 0;
      return (v_ == 1) ? 4096 : v_;
    }();
    // win_bq (sliding mask disabled when window<=0 or window>=N_kv) is hoisted
    // above the GEMM gate; the split path shares the exact same semantic.
    if (sp_len > 0 && win_bq == 0 && N_kv > sp_len) {
      if (attention_blockq_splitkv_prefill(
            q_fp16, k_fp16, v_fp16, o_fp16, num_heads_Q, num_heads_KV, N_q,
            N_kv, cache_from, head_dim, win_bq, softcap, ring_cap, sp_len)) {
        StreamManager::Global().maybeFinish();
        return true;
      }
      // any failure: fall through to the serial blockq (full recompute)
    }
    const char *fn = (head_dim == 256)   ? "attn_blockq_d256"
                     : (head_dim == 512) ? "attn_blockq_d512"
                                         : "attn_blockq_d128";
    auto kb = CudaContext::Global().registerCudaKernel(ATTN_BLOCKQ_SRC, fn);
    if (kb) {
      const int TM = 4;
      const int n_row_tiles = (N_q + TM - 1) / TM;
      kb->SetKernelArguments(0, &q_fp16, sizeof(q_fp16));
      kb->SetKernelArguments(1, &k_fp16, sizeof(k_fp16));
      kb->SetKernelArguments(2, &v_fp16, sizeof(v_fp16));
      kb->SetKernelArguments(3, &o_fp16, sizeof(o_fp16));
      kb->SetKernelArguments(4, &num_heads_Q, sizeof(num_heads_Q));
      kb->SetKernelArguments(5, &num_heads_KV, sizeof(num_heads_KV));
      kb->SetKernelArguments(6, &N_q, sizeof(N_q));
      kb->SetKernelArguments(7, &N_kv, sizeof(N_kv));
      kb->SetKernelArguments(8, &cache_from, sizeof(cache_from));
      kb->SetKernelArguments(9, &head_dim, sizeof(head_dim));
      kb->SetKernelArguments(10, &win_bq, sizeof(win_bq));
      kb->SetKernelArguments(11, &softcap, sizeof(softcap));
      kb->SetKernelArguments(12, &ring_cap, sizeof(ring_cap));
      const int grid[3] = {num_heads_Q * n_row_tiles, 1, 1};
      const int block[3] = {32, 1, 1};
      if (StreamManager::Global().DispatchCommand(*kb, grid, block, 0)) {
        StreamManager::Global().maybeFinish();
        return true;
      }
    }
  }

  auto kernel = CudaContext::Global().registerCudaKernel(ATTN_IL_FP16_SRC,
                                                         "attn_core_il_fp16");
  if (!kernel) {
    ml_loge("[CUDA] attn_core_il_fp16: registration failed");
    return false;
  }
  const int B = 128;
  kernel->SetKernelArguments(0, &q_fp16, sizeof(q_fp16));
  kernel->SetKernelArguments(1, &k_fp16, sizeof(k_fp16));
  kernel->SetKernelArguments(2, &v_fp16, sizeof(v_fp16));
  kernel->SetKernelArguments(3, &o_fp16, sizeof(o_fp16));
  kernel->SetKernelArguments(4, &num_heads_Q, sizeof(num_heads_Q));
  kernel->SetKernelArguments(5, &num_heads_KV, sizeof(num_heads_KV));
  kernel->SetKernelArguments(6, &N_q, sizeof(N_q));
  kernel->SetKernelArguments(7, &N_kv, sizeof(N_kv));
  kernel->SetKernelArguments(8, &cache_from, sizeof(cache_from));
  kernel->SetKernelArguments(9, &head_dim, sizeof(head_dim));
  kernel->SetKernelArguments(10, &window, sizeof(window));
  kernel->SetKernelArguments(11, &softcap, sizeof(softcap));
  // M2-B: bind the device position buffer so the captured graph reads the live
  // cache_from/N_kv on replay; nullptr keeps the baked-arg (non-graph) path.
  static const bool m2b_attn = nntr_env_on("NNTR_CUDA_M2B");
  const int *attn_dpos = m2b_attn ? cuda_pos_buffer() : nullptr;
  kernel->SetKernelArguments(12, &attn_dpos, sizeof(attn_dpos));
  kernel->SetKernelArguments(13, &ring_cap, sizeof(ring_cap));
  const int grid[3] = {N_q, num_heads_Q, 1};
  const int block[3] = {B, 1, 1};
  const unsigned int shmem =
    (unsigned int)(sizeof(float) * ((size_t)head_dim + B));
  static const bool dbg = std::getenv("NNTR_CUDA_ATTN_DBG") != nullptr;
  if (dbg)
    fprintf(
      stderr,
      "[ATTNDBG] HQ=%d HKV=%d N_q=%d N_kv=%d from=%d d=%d win=%d cap=%.0f "
      "shmem=%u\n",
      num_heads_Q, num_heads_KV, N_q, N_kv, cache_from, head_dim, window,
      softcap, shmem);
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block, shmem))
    return false;
  StreamManager::Global().maybeFinish();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    ml_loge("[CUDA] attn_core_il_fp16 runtime error: %s",
            cudaGetErrorString(e));
    return false;
  }
  return true;
}

bool cuda_attention_core_fp32(const float *Q, const float *K, const float *V,
                              float *O, int num_heads, int num_kv_heads,
                              int q_rows, int kv_len, int q_pos0, int head_dim,
                              int window, float softcap) {
  if (num_heads == 0 || q_rows == 0 || kv_len == 0 || head_dim == 0)
    return true;

  auto kernel =
    CudaContext::Global().registerCudaKernel(ATTN_CORE_SRC, "attn_core");
  if (!kernel) {
    ml_loge("[CUDA] attn_core: kernel registration failed");
    return false;
  }

  const int B = 128; // head_dim/B <= 4 for head_dim<=512
  kernel->SetKernelArguments(0, &Q, sizeof(Q));
  kernel->SetKernelArguments(1, &K, sizeof(K));
  kernel->SetKernelArguments(2, &V, sizeof(V));
  kernel->SetKernelArguments(3, &O, sizeof(O));
  kernel->SetKernelArguments(4, &num_heads, sizeof(num_heads));
  kernel->SetKernelArguments(5, &num_kv_heads, sizeof(num_kv_heads));
  kernel->SetKernelArguments(6, &q_rows, sizeof(q_rows));
  kernel->SetKernelArguments(7, &kv_len, sizeof(kv_len));
  kernel->SetKernelArguments(8, &q_pos0, sizeof(q_pos0));
  kernel->SetKernelArguments(9, &head_dim, sizeof(head_dim));
  kernel->SetKernelArguments(10, &window, sizeof(window));
  kernel->SetKernelArguments(11, &softcap, sizeof(softcap));

  const int grid[3] = {q_rows, num_heads, 1};
  const int block[3] = {B, 1, 1};
  const unsigned int shmem =
    (unsigned int)(sizeof(float) * ((size_t)head_dim + B));
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block, shmem))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

} // namespace nntrainer::cuda

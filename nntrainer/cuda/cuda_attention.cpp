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

#include <cuda_context.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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
  unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, o;
  if (e == 0u) {
    if (m == 0u) o = s;
    else { int x=-1; do{m<<=1;x++;}while((m&0x400u)==0u); m&=0x3FFu;
           o = s | ((unsigned int)(127-15-x)<<23) | (m<<13); }
  } else if (e == 0x1Fu) o = s | 0x7F800000u | (m<<13);
  else o = s | ((e + (127u-15u))<<23) | (m<<13);
  return __int_as_float((int)o);
}
__device__ __forceinline__ unsigned short a_f2h(float f) {
  unsigned int x=(unsigned int)__float_as_int(f), s=(x>>16)&0x8000u, mant=x&0x7FFFFFu;
  int e=(int)((x>>23)&0xFFu);
  if (e==0xFF) return (unsigned short)(s|0x7C00u|(mant?0x200u:0u));
  int exp=e-127+15;
  if (exp>=0x1F) return (unsigned short)(s|0x7C00u);
  if (exp<=0){ if(exp<-10) return (unsigned short)s; mant|=0x800000u; int sh=14-exp;
    unsigned int hh=mant>>sh, rem=mant&((1u<<sh)-1u), half=1u<<(sh-1);
    if(rem>half||(rem==half&&(hh&1u))) hh++; return (unsigned short)(s|hh); }
  unsigned int hh=((unsigned int)exp<<10)|(mant>>13), rem=mant&0x1FFFu;
  if(rem>0x1000u||(rem==0x1000u&&(hh&1u))) hh++;
  return (unsigned short)(s|hh);
}

__global__ void attn_core_il_fp16(const unsigned short *q, const unsigned short *k,
                                  const unsigned short *v, unsigned short *o,
                                  int HQ, int HKV, int N_q, int N_kv,
                                  int cache_from, int d, int window, float softcap,
                                  const int *d_pos) {
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
    const unsigned short *kr = k + (long)j*HD_KV + (long)hkv*d;
    float pd=0.f;
    for (int dd=tid;dd<d;dd+=B) pd += Qsh[dd]*a_h2f(kr[dd]);
    red[tid]=pd; __syncthreads();
    for (int s=B>>1;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
    float score=red[0]*scale; __syncthreads();
    if (softcap>0.f) score=softcap*tanhf(score/softcap);
    float mn=fmaxf(mmax,score), corr=__expf(mmax-mn), p=__expf(score-mn);
    l=l*corr+p; mmax=mn;
    const unsigned short *vr = v + (long)j*HD_KV + (long)hkv*d;
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
  cudaPointerAttributes a{};
  bool dev = cudaPointerGetAttributes(&a, host) == cudaSuccess &&
             (a.type == cudaMemoryTypeManaged || a.type == cudaMemoryTypeDevice);
  cudaGetLastError();
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

// Flash-decoding (split-KV) for M=1 decode: the single-pass kernel launches only
// num_heads blocks (8 for gemma4) -- it underutilizes the SMs and serializes the
// long KV loop. Split the KV axis into chunks so num_heads*n_chunks blocks run a
// partial online-softmax in parallel, then a small reduce combines the chunks.
static const char *ATTN_SPLITKV_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float s_h2f(unsigned short h) {
  unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, o;
  if (e == 0u) {
    if (m == 0u) o = s;
    else { int x=-1; do{m<<=1;x++;}while((m&0x400u)==0u); m&=0x3FFu;
           o = s | ((unsigned int)(127-15-x)<<23) | (m<<13); }
  } else if (e == 0x1Fu) o = s | 0x7F800000u | (m<<13);
  else o = s | ((e + (127u-15u))<<23) | (m<<13);
  return __int_as_float((int)o);
}
__device__ __forceinline__ unsigned short s_f2h(float f) {
  unsigned int x=(unsigned int)__float_as_int(f), s=(x>>16)&0x8000u, mant=x&0x7FFFFFu;
  int e=(int)((x>>23)&0xFFu);
  if (e==0xFF) return (unsigned short)(s|0x7C00u|(mant?0x200u:0u));
  int exp=e-127+15;
  if (exp>=0x1F) return (unsigned short)(s|0x7C00u);
  if (exp<=0){ if(exp<-10) return (unsigned short)s; mant|=0x800000u; int sh=14-exp;
    unsigned int hh=mant>>sh, rem=mant&((1u<<sh)-1u), half=1u<<(sh-1);
    if(rem>half||(rem==half&&(hh&1u))) hh++; return (unsigned short)(s|hh); }
  unsigned int hh=((unsigned int)exp<<10)|(mant>>13), rem=mant&0x1FFFu;
  if(rem>0x1000u||(rem==0x1000u&&(hh&1u))) hh++;
  return (unsigned short)(s|hh);
}
// One block per (head h, chunk c); query row is 0 (decode M=1). Online softmax
// over the chunk's keys; writes (m, l, acc[d]) to scratch[h*n_chunks + c].
__global__ void attn_partial(const unsigned short *q, const unsigned short *k,
                             const unsigned short *v, float *pm, float *pl,
                             float *pacc, int HQ, int HKV, int N_kv,
                             int cache_from, int d, int window, float softcap,
                             int chunk_kv, int n_chunks) {
  int h = blockIdx.x, c = blockIdx.y;
  int gqa = HQ / HKV, hkv = h / gqa;
  int HD_KV = HKV * d;
  const unsigned short *qrow = q + (long)h * d; // i=0
  int tid = threadIdx.x, B = blockDim.x;
  extern __shared__ float sh[];
  float *Qsh = sh; float *red = sh + d;
  for (int dd = tid; dd < d; dd += B) Qsh[dd] = s_h2f(qrow[dd]);
  __syncthreads();
  float scale = rsqrtf((float)d);
  int i_abs = cache_from; // i=0
  int j_lo_g = i_abs - window + 1; if (j_lo_g < 0) j_lo_g = 0;
  int j_hi_g = i_abs; if (j_hi_g >= N_kv) j_hi_g = N_kv - 1;
  int j_lo = c * chunk_kv; if (j_lo < j_lo_g) j_lo = j_lo_g;
  int j_hi = (c + 1) * chunk_kv - 1; if (j_hi > j_hi_g) j_hi = j_hi_g;
  float acc[4];
#pragma unroll
  for (int r = 0; r < 4; r++) acc[r] = 0.f;
  float mmax = -1e30f, l = 0.f;
  for (int j = j_lo; j <= j_hi; ++j) {
    const unsigned short *kr = k + (long)j * HD_KV + (long)hkv * d;
    float pd = 0.f;
    for (int dd = tid; dd < d; dd += B) pd += Qsh[dd] * s_h2f(kr[dd]);
    red[tid] = pd; __syncthreads();
    for (int s = B >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
    float score = red[0] * scale; __syncthreads();
    if (softcap > 0.f) score = softcap * tanhf(score / softcap);
    float mn = fmaxf(mmax, score), corr = __expf(mmax - mn), p = __expf(score - mn);
    l = l * corr + p; mmax = mn;
    const unsigned short *vr = v + (long)j * HD_KV + (long)hkv * d;
    int r = 0; for (int dd = tid; dd < d; dd += B, ++r) acc[r] = acc[r]*corr + p*s_h2f(vr[dd]);
  }
  if (j_lo > j_hi) { mmax = -1e30f; l = 0.f; }
  int base = h * n_chunks + c;
  if (tid == 0) { pm[base] = mmax; pl[base] = l; }
  int r = 0; for (int dd = tid; dd < d; dd += B, ++r) pacc[(long)base * d + dd] = acc[r];
}
// One block per head; combine the n_chunks partials into the fp16 output row.
__global__ void attn_reduce(const float *pm, const float *pl, const float *pacc,
                            unsigned short *o, int HQ, int d, int n_chunks) {
  int h = blockIdx.x;
  int tid = threadIdx.x, B = blockDim.x;
  int base = h * n_chunks;
  __shared__ float M, L;
  if (tid == 0) {
    float mx = -1e30f;
    for (int c = 0; c < n_chunks; ++c) mx = fmaxf(mx, pm[base + c]);
    float l = 0.f;
    for (int c = 0; c < n_chunks; ++c) l += pl[base + c] * __expf(pm[base + c] - mx);
    M = mx; L = l;
  }
  __syncthreads();
  float inv = L > 0.f ? 1.f / L : 0.f;
  unsigned short *orow = o + (long)h * d; // i=0
  for (int dd = tid; dd < d; dd += B) {
    float a = 0.f;
    for (int c = 0; c < n_chunks; ++c)
      a += pacc[((long)(base + c)) * d + dd] * __expf(pm[base + c] - M);
    orow[dd] = s_f2h(a * inv);
  }
}
}
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
  unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, o;
  if (e == 0u) {
    if (m == 0u) o = s;
    else { int x=-1; do{m<<=1;x++;}while((m&0x400u)==0u); m&=0x3FFu;
           o = s | ((unsigned int)(127-15-x)<<23) | (m<<13); }
  } else if (e == 0x1Fu) o = s | 0x7F800000u | (m<<13);
  else o = s | ((e + (127u-15u))<<23) | (m<<13);
  return __int_as_float((int)o);
}
__device__ __forceinline__ unsigned short bq_f2h(float f) {
  unsigned int x=(unsigned int)__float_as_int(f), s=(x>>16)&0x8000u, mant=x&0x7FFFFFu;
  int e=(int)((x>>23)&0xFFu);
  if (e==0xFF) return (unsigned short)(s|0x7C00u|(mant?0x200u:0u));
  int exp=e-127+15;
  if (exp>=0x1F) return (unsigned short)(s|0x7C00u);
  if (exp<=0){ if(exp<-10) return (unsigned short)s; mant|=0x800000u; int sh=14-exp;
    unsigned int hh=mant>>sh, rem=mant&((1u<<sh)-1u), half=1u<<(sh-1);
    if(rem>half||(rem==half&&(hh&1u))) hh++; return (unsigned short)(s|hh); }
  unsigned int hh=((unsigned int)exp<<10)|(mant>>13), rem=mant&0x1FFFu;
  if(rem>0x1000u||(rem==0x1000u&&(hh&1u))) hh++;
  return (unsigned short)(s|hh);
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
            int N_kv, int cache_from, int d, int window, float softcap) {
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
  for (int n = 0; n <= n_last; ++n) {
    long k_base = (long)n * HD_KV + (long)hkv * d;
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
    long v_base = (long)n * HD_KV + (long)hkv * d;
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
                 float softcap) {
  blockq_body<4, 8>(q, k, v, o, HQ, HKV, N_q, N_kv, cache_from, d, window, softcap);
}
extern "C" __global__ void
attn_blockq_d512(const unsigned short *q, const unsigned short *k,
                 const unsigned short *v, unsigned short *o, int HQ, int HKV,
                 int N_q, int N_kv, int cache_from, int d, int window,
                 float softcap) {
  blockq_body<4, 16>(q, k, v, o, HQ, HKV, N_q, N_kv, cache_from, d, window, softcap);
}
)CU";

namespace {
float *g_pm = nullptr, *g_pl = nullptr, *g_pacc = nullptr;
size_t g_pm_cap = 0, g_pacc_cap = 0;
std::mutex g_sk_mtx;
bool ensure_sk(size_t mn, size_t acc) {
  if (mn > g_pm_cap) {
    // cudaMalloc/cudaFree inside a CUDA-graph stream capture invalidates the
    // capture. The decode split-KV scratch is pre-grown at load by
    // cuda_attention_splitkv_prewarm() so this branch must not run under
    // capture; if it ever would (an under-sized prewarm), bail so the caller
    // falls back rather than corrupting the graph.
    if (StreamManager::Global().isCapturing())
      return false;
    if (g_pm) cudaFree(g_pm);
    if (g_pl) cudaFree(g_pl);
    if (cudaMalloc(&g_pm, mn * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&g_pl, mn * sizeof(float)) != cudaSuccess)
      return false;
    g_pm_cap = mn;
  }
  if (acc > g_pacc_cap) {
    if (StreamManager::Global().isCapturing())
      return false;
    if (g_pacc) cudaFree(g_pacc);
    if (cudaMalloc(&g_pacc, acc * sizeof(float)) != cudaSuccess)
      return false;
    g_pacc_cap = acc;
  }
  return true;
}

bool attention_splitkv_decode(const unsigned short *q, const unsigned short *k,
                              const unsigned short *v, unsigned short *o, int HQ,
                              int HKV, int N_kv, int cache_from, int d,
                              int window, float softcap, int chunk_kv) {
  const int n_chunks = (N_kv + chunk_kv - 1) / chunk_kv;
  std::lock_guard<std::mutex> lk(g_sk_mtx);
  if (!ensure_sk((size_t)HQ * n_chunks, (size_t)HQ * n_chunks * d))
    return false;
  auto kp = CudaContext::Global().registerCudaKernel(ATTN_SPLITKV_SRC, "attn_partial");
  auto kr = CudaContext::Global().registerCudaKernel(ATTN_SPLITKV_SRC, "attn_reduce");
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
  const int pg[3] = {HQ, n_chunks, 1};
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
  const int rg[3] = {HQ, 1, 1};
  const int rb[3] = {B, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kr, rg, rb))
    return false;
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
  return ensure_sk(mn, mn * (size_t)max_head_dim);
}

bool cuda_attention_interleaved_fp16(const unsigned short *q_fp16,
                                     const unsigned short *k_fp16,
                                     const unsigned short *v_fp16,
                                     unsigned short *o_fp16, int num_heads_Q,
                                     int num_heads_KV, int N_q, int N_kv,
                                     int cache_from, int head_dim, int window,
                                     float softcap) {
  if (num_heads_Q == 0 || N_q == 0 || N_kv == 0 || head_dim == 0)
    return true;

  // mirror the KV cache to the device if it is host-resident (engine=cuda KV
  // cache is not UVM). K/V are [N_kv, num_heads_KV*head_dim] interleaved.
  const size_t kv_elems = (size_t)N_kv * num_heads_KV * head_dim;
  k_fp16 = mirror_kv(k_fp16, kv_elems);
  v_fp16 = mirror_kv(v_fp16, kv_elems);
  if (!k_fp16 || !v_fp16)
    return false;

  // Flash-decoding (split-KV) for M=1 decode with enough keys to fill the SMs.
  static const int sk_chunk = []() {
    const char *e = std::getenv("NNTR_CUDA_FLASH_DECODE");
    if (!e)
      return 0;            // off
    int c = atoi(e);
    return c > 0 ? c : 64; // =1 -> default chunk 64; or an explicit chunk size
  }();
  if (sk_chunk > 0 && N_q == 1 && N_kv > sk_chunk) {
    if (attention_splitkv_decode(q_fp16, k_fp16, v_fp16, o_fp16, num_heads_Q,
                                 num_heads_KV, N_kv, cache_from, head_dim, window,
                                 softcap, sk_chunk)) {
      StreamManager::Global().maybeFinish();
      return true;
    }
  }

  // Block-Q multi-row prefill: one warp per (head, TM=4 row tile), warp-shuffle
  // d-dot, K/V reused across rows. 3-4x faster than the per-key LDS-reduce
  // attn_core_il_fp16 below, fp16-identical. Opt-in (NNTR_CUDA_BLOCKQ) until
  // folded; only the multi-row (prefill) path with head_dim in {256, 512}
  // (gemma4 sliding/global) -- decode (N_q==1) keeps split-KV above.
  static const bool blockq_on = std::getenv("NNTR_CUDA_BLOCKQ") != nullptr;
  if (blockq_on && N_q > 1 && (head_dim == 256 || head_dim == 512)) {
    const char *fn = (head_dim == 256) ? "attn_blockq_d256" : "attn_blockq_d512";
    auto kb = CudaContext::Global().registerCudaKernel(ATTN_BLOCKQ_SRC, fn);
    if (kb) {
      // window<=0 or window>=N_kv -> disable the sliding mask (full causal);
      // avoids n+window overflow when mha passes INT_MAX for global layers.
      int win_bq = (window <= 0 || window >= N_kv) ? 0 : window;
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
  static const bool m2b_attn = std::getenv("NNTR_CUDA_M2B") != nullptr;
  const int *attn_dpos = m2b_attn ? cuda_pos_buffer() : nullptr;
  kernel->SetKernelArguments(12, &attn_dpos, sizeof(attn_dpos));
  const int grid[3] = {N_q, num_heads_Q, 1};
  const int block[3] = {B, 1, 1};
  const unsigned int shmem =
    (unsigned int)(sizeof(float) * ((size_t)head_dim + B));
  static const bool dbg = std::getenv("NNTR_CUDA_ATTN_DBG") != nullptr;
  if (dbg)
    fprintf(stderr,
            "[ATTNDBG] HQ=%d HKV=%d N_q=%d N_kv=%d from=%d d=%d win=%d cap=%.0f "
            "shmem=%u\n",
            num_heads_Q, num_heads_KV, N_q, N_kv, cache_from, head_dim, window,
            softcap, shmem);
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block, shmem))
    return false;
  StreamManager::Global().maybeFinish();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    ml_loge("[CUDA] attn_core_il_fp16 runtime error: %s", cudaGetErrorString(e));
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

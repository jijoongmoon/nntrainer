// SPDX-License-Identifier: Apache-2.0
/**
 * @file    cuda_gdn.cpp
 * @brief   Eager GPU GatedDeltaNet single-token decode (see cuda_gdn.h).
 */
#include "cuda_gdn.h"

#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_fc_dense.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <nntrainer_log.h>

namespace nntrainer::cuda {

// Math order/accumulation mirrors gated_delta_net_layer.cpp runDecode exactly
// (per fixed output index, inner sums ascend over the same axis), so GPU and
// host outputs differ only by transcendental ulps.
static const char *GDN_SRC = R"CU(
extern "C" {
// HARDWARE fp16<->fp32, as cuda_rmsnorm.cpp:132 and cuda_fc_qint4.cpp:253 do.
// These replaced software bit-twiddling versions, and that was not a
// micro-optimisation: the old h2f contained a data-dependent `while` loop for
// denormals, which stopped the compiler unrolling ANY loop that called it. The
// GEMV's k-loop therefore ran one unhidden global-memory round trip per
// iteration (~890 cycles), which is why its runtime was INDEPENDENT of N --
// 1.40 ms at N=32 and 2.03 ms at N=8192, a 256x work range in the same time.
// fp16->fp32 is exact, and cvt.rn matches the old RNE rounding, so the numbers
// are unchanged (verify with NNTR_CUDA_GDN=2).
__device__ __forceinline__ float h2f(unsigned short h){
  float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ unsigned short f2h(float f){
  unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
__device__ __forceinline__ float4 gdn_load4(uint2 r){
  return make_float4(h2f((unsigned short)(r.x & 0xFFFFu)),
                     h2f((unsigned short)(r.x >> 16)),
                     h2f((unsigned short)(r.y & 0xFFFFu)),
                     h2f((unsigned short)(r.y >> 16)));
}
__device__ __forceinline__ float gdn_silu(float x){ return x/(1.0f+expf(-x)); }

// out[n] = sum_k x[k]*W[k*N+n]  (x fp16 [K<=4096], W fp16 [K,N] row-major)
__global__ void gdn_gemv_h_f(const unsigned short *x, const unsigned short *W,
                             float *out, int K, int N){
  __shared__ float xs[4096];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = h2f(x[i]);
  __syncthreads();
  int n = blockIdx.x*blockDim.x + threadIdx.x;
  if (n >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += xs[k] * h2f(W[(long)k*N + n]);
  out[n] = acc;
}
// out[n] = f2h(sum_k x[k]*W[k*N+n])  (x fp32 [K<=4096], W fp16, out fp16)
__global__ void gdn_gemv_f_h(const float *x, const unsigned short *W,
                             unsigned short *out, int K, int N){
  __shared__ float xs[4096];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  int n = blockIdx.x*blockDim.x + threadIdx.x;
  if (n >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += xs[k] * h2f(W[(long)k*N + n]);
  out[n] = f2h(acc);
}
// Vector-4 forms of the two GEMVs above, used whenever N%4==0 and W is
// 8-byte aligned (true for every real 35B shape: N is 8192/4096/2048/32).
// Each thread owns FOUR consecutive n and issues one 64-bit load per k, with
// the k loop unrolled 8x -- 8 independent loads in flight per thread against
// the scalar version's one. Consecutive threads still read consecutive n, so a
// warp fetches 256 contiguous bytes per iteration.
// Accumulation stays strictly ascending in k per output index, matching
// gated_delta_net_layer.cpp runDecode exactly, so this is a pure scheduling
// change and NNTR_CUDA_GDN=2 must still report fp16-rounding-level agreement.
__global__ void gdn_gemv_h_f4(const unsigned short *x, const unsigned short *W,
                              float *out, int K, int N){
  __shared__ float xs[4096];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = h2f(x[i]);
  __syncthreads();
  const int n0 = (blockIdx.x*blockDim.x + threadIdx.x)*4;
  if (n0 >= N) return;
  const uint2 *Wv = (const uint2 *)(W + n0);
  const int st = N >> 2;                       // row stride in uint2 units
  float a0=0.0f, a1=0.0f, a2=0.0f, a3=0.0f;
  #pragma unroll 8
  for (int k = 0; k < K; ++k) {
    const float4 w = gdn_load4(Wv[(long)k*st]);
    const float xv = xs[k];
    a0 += xv*w.x; a1 += xv*w.y; a2 += xv*w.z; a3 += xv*w.w;
  }
  out[n0]=a0; out[n0+1]=a1; out[n0+2]=a2; out[n0+3]=a3;
}
__global__ void gdn_gemv_f_h4(const float *x, const unsigned short *W,
                              unsigned short *out, int K, int N){
  __shared__ float xs[4096];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const int n0 = (blockIdx.x*blockDim.x + threadIdx.x)*4;
  if (n0 >= N) return;
  const uint2 *Wv = (const uint2 *)(W + n0);
  const int st = N >> 2;
  float a0=0.0f, a1=0.0f, a2=0.0f, a3=0.0f;
  #pragma unroll 8
  for (int k = 0; k < K; ++k) {
    const float4 w = gdn_load4(Wv[(long)k*st]);
    const float xv = xs[k];
    a0 += xv*w.x; a1 += xv*w.y; a2 += xv*w.z; a3 += xv*w.w;
  }
  out[n0]=f2h(a0); out[n0+1]=f2h(a1); out[n0+2]=f2h(a2); out[n0+3]=f2h(a3);
}
// causal depthwise conv1d (persistent ring) + SiLU; advances the ring.
__global__ void gdn_conv_ring(const float *qkv, const float *wconv,
                              float *ring, float *conv, int CONV, int KS){
  int c = blockIdx.x*blockDim.x + threadIdx.x;
  if (c >= CONV) return;
  float acc = 0.0f;
  for (int j = 0; j < KS-1; ++j) acc += wconv[c*KS+j] * ring[c*(KS-1)+j];
  acc += wconv[c*KS+(KS-1)] * qkv[c];
  conv[c] = gdn_silu(acc);
  for (int j = 0; j < KS-2; ++j) ring[c*(KS-1)+j] = ring[c*(KS-1)+j+1];
  ring[c*(KS-1)+(KS-2)] = qkv[c];
}
// one decay-first delta step + gated RMSNorm; one block per v-head,
// blockDim.x == HVD (power of two), HKD <= 128.
__global__ void gdn_delta_head(const float *conv, const float *z,
                               const float *pb, const float *pa,
                               const float *alog, const float *dtb,
                               const float *wnorm, float *state, float *normed,
                               int NVH, int NKH, int HKD, int HVD,
                               float scale, float eps){
  __shared__ float sq[128];
  __shared__ float sk[128];
  __shared__ float red[1024];
  const int vh = blockIdx.x, b = threadIdx.x;
  const int GQA = NVH / NKH, kh = vh / GQA;
  const int KEY = NKH * HKD;
  for (int d = b; d < HKD; d += blockDim.x) {
    sq[d] = conv[kh*HKD + d];
    sk[d] = conv[KEY + kh*HKD + d];
  }
  __syncthreads();
  // l2norm(q), l2norm(k): fp32 block reductions, 1/sqrt(sum+eps) as on host
  float psq = 0.0f, psk = 0.0f;
  for (int d = b; d < HKD; d += blockDim.x) { psq += sq[d]*sq[d]; psk += sk[d]*sk[d]; }
  red[b] = psq; __syncthreads();
  for (int s = blockDim.x>>1; s > 0; s >>= 1) { if (b < s) red[b] += red[b+s]; __syncthreads(); }
  const float iq = 1.0f/sqrtf(red[0] + eps); __syncthreads();
  red[b] = psk; __syncthreads();
  for (int s = blockDim.x>>1; s > 0; s >>= 1) { if (b < s) red[b] += red[b+s]; __syncthreads(); }
  const float ik = 1.0f/sqrtf(red[0] + eps); __syncthreads();
  for (int d = b; d < HKD; d += blockDim.x) { sq[d] *= iq; sk[d] *= ik; }
  __syncthreads();
  // decay/beta scalars (softplus guard as on host)
  const float aa = pa[vh] + dtb[vh];
  const float sp = aa > 20.0f ? aa : log1pf(expf(aa));
  const float gt = expf(-expf(alog[vh]) * sp);
  const float bt = 1.0f/(1.0f+expf(-pb[vh]));
  float *S = state + (long)vh*HKD*HVD;
  const float vb = conv[2*KEY + vh*HVD + b];
  // pass 1: S *= gt (decay first), kv[b] = sum_a S[a,b]*k[a]
  float kvb = 0.0f;
  for (int a = 0; a < HKD; ++a) {
    float s = S[(long)a*HVD + b] * gt;
    S[(long)a*HVD + b] = s;
    kvb += s * sk[a];
  }
  const float db = (vb - kvb) * bt;
  // pass 2: S += k (outer) delta, o[b] = sum_a S[a,b]*q[a]*scale (updated S)
  float ob = 0.0f;
  for (int a = 0; a < HKD; ++a) {
    float s = S[(long)a*HVD + b] + sk[a]*db;
    S[(long)a*HVD + b] = s;
    ob += s * sq[a] * scale;
  }
  // gated RMSNorm over HVD: o*rsqrt(mean(o^2)+eps)*wnorm*silu(z)
  red[b] = ob*ob; __syncthreads();
  for (int s = blockDim.x>>1; s > 0; s >>= 1) { if (b < s) red[b] += red[b+s]; __syncthreads(); }
  const float inv = 1.0f/sqrtf(red[0]/(float)HVD + eps);
  normed[vh*HVD + b] = ob*inv*wnorm[b]*gdn_silu(z[vh*HVD + b]);
}
// ---------------------------------------------------------------- prefill --
// causal depthwise conv1d + SiLU over the whole [T,CONV] plane. The left pad
// comes from the persistent ring when this is a resumed chunk, zeros when it
// is the first -- ring[c][j] holds the input at position t-(KS-1)+j, the same
// convention gdn_conv_ring advances during decode.
__global__ void gdn_conv_prefill(const float *qkv, const float *wconv,
                                 const float *ring, float *conv, int T,
                                 int CONV, int KS, int has_ring){
  const long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= (long)T*CONV) return;
  const int t = (int)(idx / CONV), c = (int)(idx - (long)t*CONV);
  float acc = 0.0f;
  for (int j = 0; j < KS; ++j) {
    const int ti = t - (KS-1) + j;
    float xv;
    if (ti >= 0)         xv = qkv[(long)ti*CONV + c];
    else if (has_ring)   xv = ring[(long)c*(KS-1) + (KS-1+ti)];
    else                 xv = 0.0f;
    acc += wconv[c*KS + j] * xv;
  }
  conv[idx] = gdn_silu(acc);
}

// In-place l2norm of the q and k slices of conv, per (token, k-head), and the
// q.k dot the scan needs. Doing the dot here rather than inside the scan is
// what keeps the scan's inner loop at ~96 FMAs: q.k is per (t, K-HEAD) and is
// shared by the GQA group, so the scan would otherwise recompute a 128-long
// reduction per v-head per token for a value it could have been handed.
// blockDim.x == HKD, which the dispatch pins to 128.
__global__ void gdn_l2norm_prefill(float *conv, float *qkdot, int CONV,
                                   int KEY, int NKH, int HKD, float eps){
  const int kh = blockIdx.x % NKH, t = blockIdx.x / NKH;
  const int d = threadIdx.x;
  __shared__ float red[128];
  float *qr = conv + (long)t*CONV + kh*HKD;
  float *kr = conv + (long)t*CONV + KEY + kh*HKD;
  float qv = qr[d], kv = kr[d];
  red[d] = qv*qv; __syncthreads();
  for (int s = blockDim.x>>1; s > 0; s >>= 1){ if (d<s) red[d]+=red[d+s]; __syncthreads(); }
  const float iq = 1.0f/sqrtf(red[0] + eps); __syncthreads();
  red[d] = kv*kv; __syncthreads();
  for (int s = blockDim.x>>1; s > 0; s >>= 1){ if (d<s) red[d]+=red[d+s]; __syncthreads(); }
  const float ik = 1.0f/sqrtf(red[0] + eps); __syncthreads();
  qv *= iq; kv *= ik;
  qr[d] = qv; kr[d] = kv;
  red[d] = qv*kv; __syncthreads();
  for (int s = blockDim.x>>1; s > 0; s >>= 1){ if (d<s) red[d]+=red[d+s]; __syncthreads(); }
  if (d == 0) qkdot[(long)t*NKH + kh] = red[0];
}

// The scan. ONE BLOCK PER V-HEAD, 512 threads, S[128][128] held entirely in
// registers as 32 floats per thread -- thread (ag,b) owns S[ag*32+j][b].
//
// The recurrence is algebraically restructured so S is touched ONCE per token
// instead of twice. With S'' = gt*S + k (x) dl:
//     o = sum_a S''[a,b] q[a] = gt*(sum_a S[a,b] q[a]) + dl[b]*(sum_a k[a] q[a])
// so one pass computing BOTH sum_a S[a,b]k[a] and sum_a S[a,b]q[a] off the old
// state, plus the precomputed q.k, gives o without ever reading the updated S.
// This is the same arithmetic as the host reference, regrouped; the summation
// order per output stays ascending in a.
#define GDN_APT 32   /* HKD*HVD / blockDim == 128*128/512 */
__global__ __launch_bounds__(512, 2)
void gdn_scan_prefill(const float *conv, const float *z, const float *pb,
                      const float *pa, const float *qkdot, const float *alog,
                      const float *dtb, const float *wnorm, float *state,
                      unsigned short *normed, int T, int NVH, int NKH,
                      int HKD, int HVD, int KEY, int CONV, int VAL,
                      float scale, float eps, int seed_state, int save_state){
  const int vh = blockIdx.x;
  const int GQA = NVH/NKH, kh = vh/GQA;
  const int tid = threadIdx.x;
  const int b = tid & 127;      // HVD == 128
  const int ag = tid >> 7;      // 0..3
  const int a0 = ag*GDN_APT;

  float S[GDN_APT];
  #pragma unroll
  for (int j = 0; j < GDN_APT; ++j)
    S[j] = seed_state ? state[((long)vh*HKD + (a0+j))*HVD + b] : 0.0f;

  __shared__ float qs[128], ks[128], vs[128], dls[128], os[128];
  __shared__ float rk[4][128], rq[4][128], wred[4];

  const float al = alog[vh], dbias = dtb[vh], wn = wnorm[b];

  for (int t = 0; t < T; ++t) {
    if (ag == 0) {
      qs[b] = conv[(long)t*CONV + kh*HKD + b];
      ks[b] = conv[(long)t*CONV + KEY + kh*HKD + b];
      vs[b] = conv[(long)t*CONV + 2*KEY + vh*HVD + b];
    }
    __syncthreads();

    const float aa = pa[(long)t*NVH + vh] + dbias;
    const float sp = aa > 20.0f ? aa : log1pf(expf(aa));
    const float gt = expf(-expf(al) * sp);
    const float bt = 1.0f/(1.0f + expf(-pb[(long)t*NVH + vh]));
    const float kq = qkdot[(long)t*NKH + kh];

    float pk = 0.0f, pq = 0.0f;
    #pragma unroll
    for (int j = 0; j < GDN_APT; ++j) {
      const float s = S[j];
      pk += s*ks[a0+j];
      pq += s*qs[a0+j];
    }
    rk[ag][b] = pk; rq[ag][b] = pq;
    __syncthreads();

    if (ag == 0) {
      const float PK = rk[0][b]+rk[1][b]+rk[2][b]+rk[3][b];
      const float PQ = rq[0][b]+rq[1][b]+rq[2][b]+rq[3][b];
      const float dv = (vs[b] - gt*PK) * bt;
      dls[b] = dv;
      os[b] = scale * (gt*PQ + dv*kq);
    }
    __syncthreads();

    const float dv_b = dls[b];
    #pragma unroll
    for (int j = 0; j < GDN_APT; ++j) S[j] = gt*S[j] + ks[a0+j]*dv_b;

    // gated RMSNorm over HVD, warp-shuffled: one barrier instead of the seven
    // a shared-memory tree would need, and this loop runs T times.
    float sq = (ag == 0) ? os[b]*os[b] : 0.0f;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_down_sync(0xffffffffu, sq, o);
    if (ag == 0 && (b & 31) == 0) wred[b >> 5] = sq;
    __syncthreads();
    if (ag == 0) {
      const float ss = wred[0]+wred[1]+wred[2]+wred[3];
      const float inv = 1.0f/sqrtf(ss/(float)HVD + eps);
      const float zv = z[(long)t*VAL + vh*HVD + b];
      normed[(long)t*VAL + vh*HVD + b] =
        f2h(os[b]*inv*wn*gdn_silu(zv));
    }
    __syncthreads();
  }

  if (save_state) {
    #pragma unroll
    for (int j = 0; j < GDN_APT; ++j)
      state[((long)vh*HKD + (a0+j))*HVD + b] = S[j];
  }
}

// ==================== chunked (WY / UT-transform) prefill ====================
// The delta-rule recurrence re-expressed as chunk-of-64 matrix work (vLLM's
// fla decomposition, read from source 2026-08-10). Consumes the SAME buffers
// the sequential scan does (conv output with l2normed q/k, pb/pa, alog/dtb),
// preserves the SAME algebraic contract:
//   lg_t = -exp(A_log[vh]) * softplus(pa + dt_bias)   (log decay, <= 0)
//   b_t  = sigmoid(pb)
//   S_t  = e^{lg_t} (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T
//   o_t  = scale * S_t^T q_t          (UPDATED state; the inclusive causal
//                                      mask in the output kernel carries the
//                                      current token's own contribution)
// and hands the result to the unchanged gated-RMSNorm + out_proj tail. All
// fp32 (the sequential scan is fp32 end to end; the only difference is
// summation ORDER: exp-of-cumsum-differences instead of per-token products).
// State orientation is OURS: S[a=key][b=value], so v_new = u - w @ S and
// S += k^T @ (v_new decayed) with no transposes.
//
// Per chunk i (C = 64):
//   gc   = inclusive cumsum of lg within the chunk (fp32)
//   A[c,d] = b_c (k_c . k_d) e^{gc_c - gc_d}, strictly lower, else 0
//   Tv   = (I + A)^-1 by forward substitution (fp32)
//   w    = Tv @ (b * e^{gc} * k)    u = Tv @ (b * v)
//   h_i  = S  (chunk-INITIAL state, stored for the output kernel)
//   v_new = u - w @ h_i             (stored un-decayed)
//   S    = e^{gl} S + k^T @ (v_new * e^{gl - gc})     gl = gc at chunk end
//   o[c] = scale * ( e^{gc_c} (q_c @ h_i)
//                    + sum_{d<=c} e^{gc_c-gc_d} (q_c . k_d) v_new[d] )

// per-token log-decay + beta, then the in-chunk inclusive cumsum.
// grid (NT, NVH), block 64 (= C). gc[t,vh], beta[t,vh]; glast[chunk,vh].
__global__ void gdn_ck_gcum(const float *pa, const float *pb,
                            const float *alog, const float *dtb, float *gc,
                            float *beta, float *glast, int T, int NVH){
  const int ck = blockIdx.x, vh = blockIdx.y;
  const int c = threadIdx.x;
  const int t = ck*64 + c;
  __shared__ float lg[64];
  float l = 0.0f;
  if (t < T) {
    const float aa = pa[(long)t*NVH + vh] + dtb[vh];
    const float sp = aa > 20.0f ? aa : log1pf(expf(aa));
    l = -expf(alog[vh]) * sp;
    beta[(long)t*NVH + vh] = 1.0f/(1.0f + expf(-pb[(long)t*NVH + vh]));
  }
  lg[c] = l;
  __syncthreads();
  if (c == 0) {                      // 64-wide serial cumsum: trivial next to
    float acc = 0.0f;                // the GEMM work, and bit-stable
    for (int j = 0; j < 64; ++j) { acc += lg[j]; lg[j] = acc; }
  }
  __syncthreads();
  if (t < T) gc[(long)t*NVH + vh] = lg[c];
  const int cn = (T - ck*64 < 64) ? (T - ck*64) : 64;
  if (c == 0) glast[(long)ck*NVH + vh] = lg[cn-1];
}

// A[c,d] = beta_c (k_c . k_d) e^{gc_c - gc_d}, strictly lower triangular.
// grid (NT, NVH), block 256. k tile staged once (fp32, 32 KB).
__global__ void gdn_ck_kkt(const float *conv, const float *gc,
                           const float *beta, float *A, int T, int CONV,
                           int KEY, int NVH, int NKH, int HKD){
  const int ck = blockIdx.x, vh = blockIdx.y;
  const int kh = vh / (NVH / NKH);
  const int cn = (T - ck*64 < 64) ? (T - ck*64) : 64;
  __shared__ float ks[64][128];
  __shared__ float gs[64], bs[64];
  const int tid = threadIdx.x;
  for (int i = tid; i < 64*128; i += 256) {
    const int c = i >> 7, d = i & 127;
    ks[c][d] = (c < cn) ? conv[(long)(ck*64 + c)*CONV + KEY + kh*HKD + d]
                        : 0.0f;
  }
  if (tid < 64) {
    const int t = ck*64 + tid;
    gs[tid] = (tid < cn) ? gc[(long)t*NVH + vh] : 0.0f;
    bs[tid] = (tid < cn) ? beta[(long)t*NVH + vh] : 0.0f;
  }
  __syncthreads();
  float *Ab = A + ((long)ck*NVH + vh)*4096;
  for (int p = tid; p < 4096; p += 256) {
    const int c = p >> 6, d = p & 63;
    float v = 0.0f;
    if (c > d && c < cn) {
      float acc = 0.0f;
      #pragma unroll 8
      for (int e = 0; e < 128; ++e) acc += ks[c][e]*ks[d][e];
      v = bs[c]*acc*expf(gs[c] - gs[d]);
    }
    Ab[p] = v;
  }
}

// Tensor-core kkt (NNTR_GDN_CK_TC=1): same A[c][d] = beta_c (k_c.k_d)
// e^{gs_c-gs_d} for c>d, via the proven fp16 recipes (A-normal + B-non-trans
// from k rows). 256 threads, 4x2 warp grid of 16x32 tiles over [64x64].
__global__ __launch_bounds__(256, 2)
void gdn_ck_kkt_tc(const float *conv, const float *gc, const float *beta,
                   float *A, int T, int CONV, int KEY, int NVH, int NKH,
                   int HKD){
  const int ck = blockIdx.x, vh = blockIdx.y;
  const int kh = vh / (NVH / NKH);
  const int cn = (T - ck*64 < 64) ? (T - ck*64) : 64;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1;
  const int g = lane >> 2, tq = lane & 3;
  __shared__ __align__(16) unsigned short ksh[64 * 136];
  __shared__ float gs[64], bs[64];
  for (int i = tid; i < 64*128; i += 256) {
    const int c = i >> 7, e = i & 127;
    float kv = (c < cn) ? conv[(long)(ck*64 + c)*CONV + KEY + kh*HKD + e]
                        : 0.0f;
    unsigned short hv;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(kv));
    ksh[c*136 + e] = hv;
  }
  if (tid < 64) {
    const int t = ck*64 + tid;
    gs[tid] = (tid < cn) ? gc[(long)t*NVH + vh] : 0.0f;
    bs[tid] = (tid < cn) ? beta[(long)t*NVH + vh] : 0.0f;
  }
  __syncthreads();
  float acc2[4][4];
#pragma unroll
  for (int s = 0; s < 4; ++s)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc2[s][r] = 0.0f;
  const int a_row = wm*16 + (lane & 15);
  const int a_k8 = (lane >> 4) & 1;
  const int b_row = wn*32 + (lane >> 3)*8 + (lane & 7);
#pragma unroll
  for (int k0 = 0; k0 < 128; k0 += 16) {
    int a0,a1,a2,a3, bl0,bl1,bl2,bl3, bh0,bh1,bh2,bh3;
    asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3)
      : "r"((unsigned)__cvta_generic_to_shared(
          &ksh[a_row*136 + k0 + a_k8*8])));
    asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(bl0),"=r"(bl1),"=r"(bl2),"=r"(bl3)
      : "r"((unsigned)__cvta_generic_to_shared(&ksh[b_row*136 + k0])));
    asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(bh0),"=r"(bh1),"=r"(bh2),"=r"(bh3)
      : "r"((unsigned)__cvta_generic_to_shared(&ksh[b_row*136 + k0 + 8])));
#define KKMMA(S_,BL,BH)                                                        \
    asm volatile(                                                              \
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                     \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                \
      : "+f"(acc2[S_][0]),"+f"(acc2[S_][1]),"+f"(acc2[S_][2]),                 \
        "+f"(acc2[S_][3])                                                      \
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(BL), "r"(BH))
    KKMMA(0,bl0,bh0); KKMMA(1,bl1,bh1); KKMMA(2,bl2,bh2); KKMMA(3,bl3,bh3);
#undef KKMMA
  }
  float *Ab = A + ((long)ck*NVH + vh)*4096;
#pragma unroll
  for (int s = 0; s < 4; ++s)
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int c = wm*16 + g + ((r >> 1) ? 8 : 0);
      const int d = wn*32 + s*8 + 2*tq + (r & 1);
      float v = 0.0f;
      if (c > d && c < cn)
        v = bs[c]*acc2[s][r]*expf(gs[c] - gs[d]);
      Ab[c*64 + d] = v;
    }
}

// Tensor-core wu (NNTR_GDN_CK_TC=1): w = T@xk, u = T@xv with SHARED T
// A-fragments (one load, two mma chains). xk = beta e^{gc} k, xv = beta v,
// both built fp16 in smem; T rounded to fp16 (a new fp16 surface -- the =2
// gate arbitrates, and out's coef precedent passed with 16x margin).
__global__ __launch_bounds__(256, 2)
void gdn_ck_wu_tc(const float *conv, const float *A, const float *gc,
                  const float *beta, float *w, float *u, int T, int CONV,
                  int KEY, int NVH, int NKH, int HKD, int HVD){
  const int ck = blockIdx.x, vh = blockIdx.y;
  const int kh = vh / (NVH / NKH);
  const int cn = (T - ck*64 < 64) ? (T - ck*64) : 64;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1;
  const int g = lane >> 2, tq = lane & 3;
  __shared__ __align__(16) unsigned short Th[64 * 72];
  __shared__ __align__(16) unsigned short xk[64 * 136];
  __shared__ __align__(16) unsigned short xv[64 * 136];
  const float *Ab = A + ((long)ck*NVH + vh)*4096;
  for (int i = tid; i < 4096; i += 256) {
    unsigned short hv;
    float tv = Ab[i];
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(tv));
    Th[(i >> 6)*72 + (i & 63)] = hv;
  }
  for (int i = tid; i < 64*128; i += 256) {
    const int c = i >> 7, dim = i & 127;
    float k_ = 0.0f, v_ = 0.0f;
    if (c < cn) {
      const long t = (long)ck*64 + c;
      const float be = beta[t*NVH + vh];
      k_ = be*expf(gc[t*NVH + vh])*conv[t*CONV + KEY + kh*HKD + dim];
      v_ = be*conv[t*CONV + 2*KEY + vh*HVD + dim];
    }
    unsigned short hk, hvv;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hk) : "f"(k_));
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hvv) : "f"(v_));
    xk[c*136 + dim] = hk;
    xv[c*136 + dim] = hvv;
  }
  __syncthreads();
  float aw[8][4], au[8][4];
#pragma unroll
  for (int s = 0; s < 8; ++s)
#pragma unroll
    for (int r = 0; r < 4; ++r) { aw[s][r] = 0.0f; au[s][r] = 0.0f; }
  const int a_row = wm*16 + (lane & 15);
  const int a_k8 = (lane >> 4) & 1;
#pragma unroll
  for (int k0 = 0; k0 < 64; k0 += 16) {
    int a0,a1,a2,a3;
    asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3)
      : "r"((unsigned)__cvta_generic_to_shared(
          &Th[a_row*72 + k0 + a_k8*8])));
#define WUMMA(ACC,S_,BL,BH)                                                    \
    asm volatile(                                                              \
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                     \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                \
      : "+f"(ACC[S_][0]),"+f"(ACC[S_][1]),"+f"(ACC[S_][2]),"+f"(ACC[S_][3])    \
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(BL), "r"(BH))
#pragma unroll
    for (int s4 = 0; s4 < 2; ++s4) {
      const int nb = wn*64 + s4*32;
      int bl0,bl1,bl2,bl3, bh0,bh1,bh2,bh3;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(bl0),"=r"(bl1),"=r"(bl2),"=r"(bl3)
        : "r"((unsigned)__cvta_generic_to_shared(
            &xk[(k0 + (lane & 7))*136 + nb + (lane >> 3)*8])));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(bh0),"=r"(bh1),"=r"(bh2),"=r"(bh3)
        : "r"((unsigned)__cvta_generic_to_shared(
            &xk[(k0 + 8 + (lane & 7))*136 + nb + (lane >> 3)*8])));
      WUMMA(aw,s4*4+0,bl0,bh0); WUMMA(aw,s4*4+1,bl1,bh1);
      WUMMA(aw,s4*4+2,bl2,bh2); WUMMA(aw,s4*4+3,bl3,bh3);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(bl0),"=r"(bl1),"=r"(bl2),"=r"(bl3)
        : "r"((unsigned)__cvta_generic_to_shared(
            &xv[(k0 + (lane & 7))*136 + nb + (lane >> 3)*8])));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(bh0),"=r"(bh1),"=r"(bh2),"=r"(bh3)
        : "r"((unsigned)__cvta_generic_to_shared(
            &xv[(k0 + 8 + (lane & 7))*136 + nb + (lane >> 3)*8])));
      WUMMA(au,s4*4+0,bl0,bh0); WUMMA(au,s4*4+1,bl1,bh1);
      WUMMA(au,s4*4+2,bl2,bh2); WUMMA(au,s4*4+3,bl3,bh3);
    }
#undef WUMMA
  }
#pragma unroll
  for (int s = 0; s < 8; ++s)
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int c = wm*16 + g + ((r >> 1) ? 8 : 0);
      const int dim = wn*64 + s*8 + 2*tq + (r & 1);
      if (c < cn) {
        const long t = (long)ck*64 + c;
        w[(t*NVH + vh)*HKD + dim] = aw[s][r];
        u[(t*NVH + vh)*HVD + dim] = au[s][r];
      }
    }
}

// Tv = (I + A)^-1 by forward substitution, in place over A.
// grid (NT, NVH), block 64 (thread = column). fp32 throughout.
__global__ void gdn_ck_tril(float *A, int NVH){
  const long base = ((long)blockIdx.x*NVH + blockIdx.y)*4096;
  const int d = threadIdx.x;
  __shared__ float M[64][64];
  __shared__ float Tv[64][64];
  for (int r = 0; r < 64; ++r) M[r][d] = A[base + r*64 + d];
  __syncthreads();
  for (int r = 0; r < 64; ++r) {
    float s = (r == d) ? 1.0f : 0.0f;
    for (int j = 0; j < r; ++j) s -= M[r][j]*Tv[j][d];
    Tv[r][d] = s;
    __syncthreads();
  }
  for (int r = 0; r < 64; ++r) A[base + r*64 + d] = Tv[r][d];
}

// w = Tv @ (beta e^{gc} k), u = Tv @ (beta v). grid (NT, NVH), block 128
// (thread = feature dim). One staging buffer reused for both passes:
// Tv 16 KB + xb 32 KB = 48 KB static shared, the sm_87 per-block limit.
__global__ void gdn_ck_wu(const float *conv, const float *A, const float *gc,
                          const float *beta, float *w, float *u, int T,
                          int CONV, int KEY, int NVH, int NKH, int HKD,
                          int HVD){
  const int ck = blockIdx.x, vh = blockIdx.y;
  const int kh = vh / (NVH / NKH);
  const int cn = (T - ck*64 < 64) ? (T - ck*64) : 64;
  const int dim = threadIdx.x;
  __shared__ float Tv[64][64];
  __shared__ float xb[64][128];
  const float *Ab = A + ((long)ck*NVH + vh)*4096;
  for (int i = dim; i < 4096; i += 128) Tv[i >> 6][i & 63] = Ab[i];
  for (int c = 0; c < 64; ++c) {
    float x = 0.0f;
    if (c < cn) {
      const long t = (long)ck*64 + c;
      x = beta[t*NVH + vh]*expf(gc[t*NVH + vh])*
          conv[t*CONV + KEY + kh*HKD + dim];
    }
    xb[c][dim] = x;
  }
  __syncthreads();
  for (int c = 0; c < cn; ++c) {
    float acc = 0.0f;
    #pragma unroll 8
    for (int d = 0; d < 64; ++d) acc += Tv[c][d]*xb[d][dim];
    w[(((long)ck*64 + c)*NVH + vh)*HKD + dim] = acc;
  }
  __syncthreads();
  for (int c = 0; c < 64; ++c) {
    float x = 0.0f;
    if (c < cn) {
      const long t = (long)ck*64 + c;
      x = beta[t*NVH + vh]*conv[t*CONV + 2*KEY + vh*HVD + dim];
    }
    xb[c][dim] = x;
  }
  __syncthreads();
  for (int c = 0; c < cn; ++c) {
    float acc = 0.0f;
    #pragma unroll 8
    for (int d = 0; d < 64; ++d) acc += Tv[c][d]*xb[d][dim];
    u[(((long)ck*64 + c)*NVH + vh)*HVD + dim] = acc;
  }
}

// The only sequential piece: state propagation across chunks. grid NVH,
// block 512 -- the scan's exact register geometry (thread (ag,b) owns
// S[a0+j][b], j<32) but per CHUNK instead of per token. Per chunk:
// store h_i, v_new = u - w @ S (whole chunk vs the chunk-INITIAL S),
// then S = e^{gl} S + k^T @ (v_new e^{gl - gc}).
__global__ void __launch_bounds__(512, 2)
gdn_ck_state(const float *conv, const float *w, const float *u,
             const float *gc, const float *glast, float *h, float *vnew,
             float *state, int T, int CONV, int KEY, int NVH, int NKH,
             int HKD, int HVD, int seed_state, int save_state){
  const int vh = blockIdx.x;
  const int kh = vh / (NVH / NKH);
  const int NT = (T + 63) >> 6;
  const int tid = threadIdx.x;
  const int b = tid & 127, ag = tid >> 7, a0 = ag*32;
  __shared__ float dv[64][128];
  __shared__ float rk[4][128];
  float S[32];
  #pragma unroll
  for (int j = 0; j < 32; ++j)
    S[j] = seed_state ? state[((long)vh*HKD + (a0+j))*HVD + b] : 0.0f;

  for (int ck = 0; ck < NT; ++ck) {
    const int cn = (T - ck*64 < 64) ? (T - ck*64) : 64;
    const float gl = glast[(long)ck*NVH + vh];
    // h_i = chunk-initial state
    float *hb = h + ((long)ck*NVH + vh)*16384;
    #pragma unroll
    for (int j = 0; j < 32; ++j) hb[(a0+j)*128 + b] = S[j];
    // v_new for the whole chunk against h_i
    for (int c = 0; c < cn; ++c) {
      const long t = (long)ck*64 + c;
      const float *wr = w + (t*NVH + vh)*HKD;
      float pk = 0.0f;
      #pragma unroll
      for (int j = 0; j < 32; ++j) pk += wr[a0+j]*S[j];
      rk[ag][b] = pk;
      __syncthreads();
      if (ag == 0) {
        const float vn = u[(t*NVH + vh)*HVD + b] -
                         (rk[0][b]+rk[1][b]+rk[2][b]+rk[3][b]);
        vnew[(t*NVH + vh)*HVD + b] = vn;
        dv[c][b] = vn*expf(gl - gc[t*NVH + vh]);
      }
      __syncthreads();
    }
    // S = e^{gl} S + k^T @ dv
    const float egl = expf(gl);
    #pragma unroll
    for (int j = 0; j < 32; ++j) S[j] *= egl;
    for (int c = 0; c < cn; ++c) {
      const float *kr = conv + ((long)ck*64 + c)*CONV + KEY + kh*HKD;
      const float d_b = dv[c][b];
      #pragma unroll
      for (int j = 0; j < 32; ++j) S[j] += kr[a0+j]*d_b;
    }
    __syncthreads();
  }
  if (save_state) {
    #pragma unroll
    for (int j = 0; j < 32; ++j)
      state[((long)vh*HKD + (a0+j))*HVD + b] = S[j];
  }
}

// o[c] = scale * ( e^{gc_c} (q_c @ h_i) + sum_{d<=c} coef[c][d] v_new[d] ),
// coef[c][d] = (q_c . k_d) e^{gc_c - gc_d}, INCLUSIVE mask (the diagonal is
// the current token's own contribution -- the sequential scan reads the
// UPDATED state). grid (NT, NVH), block 128 (thread = b).
__global__ void gdn_ck_out(const float *conv, const float *gc,
                           const float *h, const float *vnew, float *o,
                           float scale, int T, int CONV, int KEY, int NVH,
                           int NKH, int HKD, int HVD){
  const int ck = blockIdx.x, vh = blockIdx.y;
  const int kh = vh / (NVH / NKH);
  const int cn = (T - ck*64 < 64) ? (T - ck*64) : 64;
  const int b = threadIdx.x;
  // 48 KB static-shared budget exactly: ks 32 KB + coef 16 KB. The 64 gc
  // values are read from global (L2-resident) -- a gs[64] array here was 256
  // bytes over the sm_87 per-block limit and failed ptxas.
  __shared__ float ks[64][128];
  __shared__ float coef[64][64];
  for (int i = b; i < 64*128; i += 128) {
    const int c = i >> 7, d = i & 127;
    ks[c][d] = (c < cn) ? conv[(long)(ck*64 + c)*CONV + KEY + kh*HKD + d]
                        : 0.0f;
  }
  __syncthreads();
  const float *gcb = gc;
  for (int p = b; p < 4096; p += 128) {
    const int c = p >> 6, d = p & 63;
    float v = 0.0f;
    if (d <= c && c < cn) {
      const float *qr = conv + (long)(ck*64 + c)*CONV + kh*HKD;
      float acc = 0.0f;
      #pragma unroll 8
      for (int e = 0; e < 128; ++e) acc += qr[e]*ks[d][e];
      v = acc*expf(gcb[((long)ck*64 + c)*NVH + vh] -
                   gcb[((long)ck*64 + d)*NVH + vh]);
    }
    coef[c][d] = v;
  }
  __syncthreads();
  // Phase 2: intra term entirely from shared memory -- reuse ks[] as the
  // v_new tile (its k data is consumed; coef[] holds the masked products).
  for (int i = b; i < 64*128; i += 128) {
    const int c = i >> 7, d = i & 127;
    ks[c][d] = (c < cn)
                 ? vnew[(((long)ck*64 + c)*NVH + vh)*HVD + d]
                 : 0.0f;
  }
  __syncthreads();
  const long obase = ((long)ck*64)*(long)(NVH*HVD) + vh*HVD + b;
  for (int c = 0; c < cn; ++c) {
    float intra = 0.0f;
    #pragma unroll 8
    for (int d = 0; d < 64; ++d) intra += coef[c][d]*ks[d][b];
    o[obase + (long)c*(NVH*HVD)] = intra; // unscaled; phase 3 completes it
  }
  __syncthreads();
  // Phase 3: inter term with h staged in 32-row a-tiles (16 KB each, reusing
  // ks[]) and 16-wide c-blocks accumulated in registers -- one global RMW per
  // output instead of the 64x re-read of h that made v1 16 ms per block.
  const float *hb = h + ((long)ck*NVH + vh)*16384;
  for (int cb = 0; cb < 4; ++cb) {
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (int at = 0; at < 4; ++at) {
      __syncthreads();
      for (int i = b; i < 32*128; i += 128)
        ks[i >> 7][i & 127] = hb[(at*32 + (i >> 7))*128 + (i & 127)];
      __syncthreads();
      for (int ci = 0; ci < 16; ++ci) {
        const int c = cb*16 + ci;
        if (c >= cn) break;
        const float *qr = conv + ((long)ck*64 + c)*CONV + kh*HKD + at*32;
        float p = 0.0f;
        #pragma unroll 8
        for (int j = 0; j < 32; ++j) p += qr[j]*ks[j][b];
        acc[ci] += p;
      }
    }
    for (int ci = 0; ci < 16; ++ci) {
      const int c = cb*16 + ci;
      if (c >= cn) break;
      const long t = (long)ck*64 + c;
      const long oi = t*(long)(NVH*HVD) + vh*HVD + b;
      o[oi] = scale*(acc[ci]*expf(gcb[t*NVH + vh]) + o[oi]);
    }
  }
}

// Tensor-core rewrite of gdn_ck_out (NNTR_GDN_CK_TC=1): the same
//   o = scale * ( e^{gc_c} (q_c @ h) + sum_{d<=c} (q_c.k_d) e^{gc_c-gc_d} vn_d )
// with the three products as fp16 m16n8k16 mma (fp32 accumulate). Inputs are
// rounded to fp16 at staging -- exactly the surfaces the measured acceptance
// gate covers (out <= 0.0625 vs the scan; the fp16-stub envelope was
// 0.03125). 256 threads (vs the SIMT kernel's 128), smem phase-union 44 KB.
// Fragment recipe proven standalone in tile_bench/fp16_frag_bench.cu:
//   A (row-major m16k16): x4 quadrants, row wm*16+(lane&15), k8 half lane>>4
//   B from [k][n] row-major: ldmatrix.trans.x4 -> 4 n8-subtile k8-halves
//   B from [n][k] row-major: ldmatrix.x4 (non-trans), reg s = subtile s
__global__ __launch_bounds__(256, 2)
void gdn_ck_out_tc(const float *conv, const float *gc, const float *h,
                   const float *vnew, float *o, float scale, int T, int CONV,
                   int KEY, int NVH, int NKH, int HKD, int HVD){
  const int ck = blockIdx.x, vh = blockIdx.y;
  const int kh = vh / (NVH / NKH);
  const int cn = (T - ck*64 < 64) ? (T - ck*64) : 64;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1; // 4x2 warp grid
  const int g = lane >> 2, tq = lane & 3;
  // phase-union shared memory. LD = 136 halves (128 + 8 pad = 16B-aligned
  // rows, the bench-proven stride). Layout:
  //   [0)      q   64x136 fp16 (17,408 B)  -- alive through T1
  //   [17408)  kx  64x136 fp16 (17,408 B)  -- k in phase A; h-half in T1;
  //                                            vn in T2
  //   [34816)  coef 64x72 fp16  (9,216 B)  -- written end of phase A
  __shared__ __align__(16) unsigned char sb[44032];
  unsigned short *qs = (unsigned short *)sb;
  unsigned short *kx = (unsigned short *)(sb + 17408);
  unsigned short *cf = (unsigned short *)(sb + 34816);
  const int LD = 136, LDC = 72;

  // ---- stage q and k (fp32 -> fp16 round at load) ----
  for (int i = tid; i < 64*128; i += 256) {
    const int c = i >> 7, e = i & 127;
    float qv = 0.f, kv = 0.f;
    if (c < cn) {
      const long tbase = (long)(ck*64 + c)*CONV + kh*HKD;
      qv = conv[tbase + e];
      kv = conv[tbase + KEY + e];
    }
    unsigned short hq, hk;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hq) : "f"(qv));
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hk) : "f"(kv));
    qs[c*LD + e] = hq;
    kx[c*LD + e] = hk;
  }
  __syncthreads();

  // ---- phase A: S_qk[64][64] = q @ k^T; coef = masked gate(S_qk) ----
  // warp tile 16(m) x 32(n): 4 n8-subtiles, k walked 16 at a time.
  {
    float acc2[4][4];
    #pragma unroll
    for (int s = 0; s < 4; ++s)
      #pragma unroll
      for (int r = 0; r < 4; ++r) acc2[s][r] = 0.f;
    const int a_row = wm*16 + (lane & 15);
    const int a_kh8 = (lane >> 4) & 1;
    const int b_row = wn*32 + (lane >> 3)*8 + (lane & 7); // k rows = n dim
    #pragma unroll
    for (int k0 = 0; k0 < 128; k0 += 16) {
      int a0,a1,a2,a3, c0,c1,c2,c3;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3)
        : "r"((unsigned)__cvta_generic_to_shared(&qs[a_row*LD + k0 + a_kh8*8])));
      // B = k^T from k stored [n(=d)][k(=e)] row-major: non-trans x4, two
      // 16B k-halves live in regs (c0,c1)= subtiles k-lo? Proven int8-style
      // mapping: reg s covers subtile s at THIS 16-elem k slice; a second
      // load at +8 halves... For fp16 n8k16 B needs 2 regs per subtile:
      // load k-lo (8 halves) and k-hi separately.
      int bl0,bl1,bl2,bl3, bh0,bh1,bh2,bh3;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(bl0),"=r"(bl1),"=r"(bl2),"=r"(bl3)
        : "r"((unsigned)__cvta_generic_to_shared(&kx[b_row*LD + k0])));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(bh0),"=r"(bh1),"=r"(bh2),"=r"(bh3)
        : "r"((unsigned)__cvta_generic_to_shared(&kx[b_row*LD + k0 + 8])));
#define GOMMA(ACC,S,BL,BH)                                                     \
      asm volatile(                                                            \
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                   \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"              \
        : "+f"(ACC[S][0]), "+f"(ACC[S][1]), "+f"(ACC[S][2]), "+f"(ACC[S][3])   \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(BL), "r"(BH))
      GOMMA(acc2,0,bl0,bh0); GOMMA(acc2,1,bl1,bh1);
      GOMMA(acc2,2,bl2,bh2); GOMMA(acc2,3,bl3,bh3);
    }
    // mask + gate -> coef (fp16). C layout: row wm*16+g(+8), col wn*32+s*8+2tq(+1)
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
      #pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int c = wm*16 + g + ((r >> 1) ? 8 : 0);
        const int d = wn*32 + s*8 + 2*tq + (r & 1);
        float v = 0.f;
        if (d <= c && c < cn)
          v = acc2[s][r]*expf(gc[((long)ck*64 + c)*NVH + vh] -
                              gc[((long)ck*64 + d)*NVH + vh]);
        unsigned short hv;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(v));
        cf[c*LDC + d] = hv;
      }
    }
  }
  __syncthreads();

  // ---- T1: acc = q @ h (h [e][b] row-major, staged in two k64 halves) ----
  float acc[8][4];
  #pragma unroll
  for (int s = 0; s < 8; ++s)
    #pragma unroll
    for (int r = 0; r < 4; ++r) acc[s][r] = 0.f;
  const float *hb = h + ((long)ck*NVH + vh)*16384;
  const int a_row = wm*16 + (lane & 15);
  const int a_kh8 = (lane >> 4) & 1;
  for (int eh = 0; eh < 2; ++eh) {
    // restage kx <- h rows [eh*64, eh*64+64) as fp16
    for (int i = tid; i < 64*128; i += 256) {
      const int e = i >> 7, bcol = i & 127;
      unsigned short hh;
      float hv = hb[(eh*64 + e)*128 + bcol];
      asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hh) : "f"(hv));
      kx[e*LD + bcol] = hh;
    }
    __syncthreads();
    #pragma unroll
    for (int k0 = 0; k0 < 64; k0 += 16) {
      int a0,a1,a2,a3;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3)
        : "r"((unsigned)__cvta_generic_to_shared(
            &qs[a_row*LD + eh*64 + k0 + a_kh8*8])));
      #pragma unroll
      for (int s4 = 0; s4 < 2; ++s4) {
        const int nb = wn*64 + s4*32;
        int bl0,bl1,bl2,bl3, bh0,bh1,bh2,bh3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
          : "=r"(bl0),"=r"(bl1),"=r"(bl2),"=r"(bl3)
          : "r"((unsigned)__cvta_generic_to_shared(
              &kx[(k0 + (lane & 7))*LD + nb + (lane >> 3)*8])));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
          : "=r"(bh0),"=r"(bh1),"=r"(bh2),"=r"(bh3)
          : "r"((unsigned)__cvta_generic_to_shared(
              &kx[(k0 + 8 + (lane & 7))*LD + nb + (lane >> 3)*8])));
        GOMMA(acc,s4*4+0,bl0,bh0); GOMMA(acc,s4*4+1,bl1,bh1);
        GOMMA(acc,s4*4+2,bl2,bh2); GOMMA(acc,s4*4+3,bl3,bh3);
      }
    }
    __syncthreads(); // kx restaged next iteration
  }
  // scale the inter term by e^{gc_c} (row-dependent) before adding intra
  {
    const int c0 = wm*16 + g, c1 = c0 + 8;
    const float e0 = (c0 < cn) ? expf(gc[((long)ck*64 + c0)*NVH + vh]) : 0.f;
    const float e1 = (c1 < cn) ? expf(gc[((long)ck*64 + c1)*NVH + vh]) : 0.f;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
      acc[s][0] *= e0; acc[s][1] *= e0;
      acc[s][2] *= e1; acc[s][3] *= e1;
    }
  }

  // ---- T2: acc += coef @ vn (A = coef [c][d], B = vn [d][b] row-major) ----
  for (int i = tid; i < 64*128; i += 256) { // restage kx <- vn as fp16
    const int d = i >> 7, bcol = i & 127;
    float vv = (d < cn) ? vnew[(((long)ck*64 + d)*NVH + vh)*HVD + bcol] : 0.f;
    unsigned short hv;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(vv));
    kx[d*LD + bcol] = hv;
  }
  __syncthreads();
  #pragma unroll
  for (int k0 = 0; k0 < 64; k0 += 16) {
    int a0,a1,a2,a3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3)
      : "r"((unsigned)__cvta_generic_to_shared(
          &cf[a_row*LDC + k0 + a_kh8*8])));
    #pragma unroll
    for (int s4 = 0; s4 < 2; ++s4) {
      const int nb = wn*64 + s4*32;
      int bl0,bl1,bl2,bl3, bh0,bh1,bh2,bh3;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(bl0),"=r"(bl1),"=r"(bl2),"=r"(bl3)
        : "r"((unsigned)__cvta_generic_to_shared(
            &kx[(k0 + (lane & 7))*LD + nb + (lane >> 3)*8])));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(bh0),"=r"(bh1),"=r"(bh2),"=r"(bh3)
        : "r"((unsigned)__cvta_generic_to_shared(
            &kx[(k0 + 8 + (lane & 7))*LD + nb + (lane >> 3)*8])));
      GOMMA(acc,s4*4+0,bl0,bh0); GOMMA(acc,s4*4+1,bl1,bh1);
      GOMMA(acc,s4*4+2,bl2,bh2); GOMMA(acc,s4*4+3,bl3,bh3);
    }
  }
#undef GOMMA

  // ---- epilogue: o = scale * acc ----
  #pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int bcol = wn*64 + s*8 + 2*tq;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int c = wm*16 + g + ((r >> 1) ? 8 : 0);
      if (c < cn) {
        const long t = (long)ck*64 + c;
        o[t*(long)(NVH*HVD) + vh*HVD + bcol + (r & 1)] = scale*acc[s][r];
      }
    }
  }
}

// Tensor-core rewrite of the sequential state sweep (NNTR_GDN_CK_TC=1).
// Same math: per chunk c -- h_c = S; vpre = w_c @ S; vn = u_c - vpre;
// dv = vn*e^{gl_c - gc}; S = S*e^{gl_c} + k_c^T @ dv. S[128x128] lives in
// mma-C-fragment fp32 registers of 16 warps (8x2 grid of 16x64 tiles, 32
// floats/thread), so the k^T@dv update lands on S with NO relayout. w@S gets
// S via a per-chunk fp16 smem dump (doubles as the h write); k^T uses the
// A-TRANS quadrant recipe derived in tile_bench/fp16_state_bench.cu:
//   addr = (k0 + (lane&7) + ((lane&16)?8:0))*LD + m0 + ((lane&8)?8:0)
// Bench-validated rel<=1e-3; 20.05 -> 4.13 ms at the T=4096 shape.
__global__ __launch_bounds__(512, 1)
void gdn_ck_state_tc(const float *conv, const float *w, const float *u,
                     const float *gc, const float *gl, float *h, float *vnew,
                     float *state, int T, int CONV, int KEY, int NVH, int NKH,
                     int HKD, int HVD, int seed_state, int save_state){
  const int vh = blockIdx.x;
  const int kh = vh / (NVH / NKH);
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1; // 8x2 grid of 16x64 tiles
  const int g = lane >> 2, tq = lane & 3;
  const int NCH = (T + 63) >> 6;
  const int SLD = 136;
  __shared__ __align__(16) unsigned short Sh[64 * 136]; // S j-half / k tile
  __shared__ __align__(16) unsigned short Tb[64 * 136]; // w tile / dv tile

  float S[8][4];
#pragma unroll
  for (int s = 0; s < 8; ++s)
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int j = wm*16 + g + ((r >> 1) ? 8 : 0);
      const int b = wn*64 + s*8 + 2*tq + (r & 1);
      S[s][r] = seed_state ? state[((long)vh*HKD + j)*HVD + b] : 0.0f;
    }

  for (int c = 0; c < NCH; ++c) {
    const int cn = (T - c*64 < 64) ? (T - c*64) : 64;
    const float eglc = expf(gl[(long)c*NVH + vh]);
    float vp[8][4];
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
      for (int r = 0; r < 4; ++r) vp[s][r] = 0.0f;
    for (int jh = 0; jh < 2; ++jh) {
      __syncthreads();
#pragma unroll
      for (int s = 0; s < 8; ++s)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int j = wm*16 + g + ((r >> 1) ? 8 : 0);
          const int b = wn*64 + s*8 + 2*tq + (r & 1);
          if (j >= jh*64 && j < jh*64 + 64) {
            unsigned short hv;
            asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(S[s][r]));
            Sh[(j - jh*64)*SLD + b] = hv;
            h[((long)c*NVH + vh)*16384 + j*128 + b] = S[s][r];
          }
        }
      __syncthreads();
      for (int i = tid; i < 64*64; i += 512) {
        const int cc = i >> 6, jj = i & 63;
        float wv = (cc < cn)
          ? w[(((long)c*64 + cc)*NVH + vh)*HKD + jh*64 + jj] : 0.0f;
        unsigned short hv;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(wv));
        Tb[cc*SLD + jj] = hv;
      }
      __syncthreads();
      if (wm < 4) {
        const int a_row = wm*16 + (lane & 15);
        const int a_k8 = (lane >> 4) & 1;
#pragma unroll
        for (int k0 = 0; k0 < 64; k0 += 16) {
          int a0,a1,a2,a3;
          asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3)
            : "r"((unsigned)__cvta_generic_to_shared(
                &Tb[a_row*SLD + k0 + a_k8*8])));
#pragma unroll
          for (int s4 = 0; s4 < 2; ++s4) {
            const int nb = wn*64 + s4*32;
            int bl0,bl1,bl2,bl3, bh0,bh1,bh2,bh3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
              "{%0,%1,%2,%3}, [%4];\n"
              : "=r"(bl0),"=r"(bl1),"=r"(bl2),"=r"(bl3)
              : "r"((unsigned)__cvta_generic_to_shared(
                  &Sh[(k0 + (lane & 7))*SLD + nb + (lane >> 3)*8])));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
              "{%0,%1,%2,%3}, [%4];\n"
              : "=r"(bh0),"=r"(bh1),"=r"(bh2),"=r"(bh3)
              : "r"((unsigned)__cvta_generic_to_shared(
                  &Sh[(k0 + 8 + (lane & 7))*SLD + nb + (lane >> 3)*8])));
#define STMMA(ACC,S_,BL,BH)                                                    \
            asm volatile(                                                      \
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "             \
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"        \
              : "+f"(ACC[S_][0]),"+f"(ACC[S_][1]),"+f"(ACC[S_][2]),            \
                "+f"(ACC[S_][3])                                               \
              : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(BL), "r"(BH))
            STMMA(vp,s4*4+0,bl0,bh0); STMMA(vp,s4*4+1,bl1,bh1);
            STMMA(vp,s4*4+2,bl2,bh2); STMMA(vp,s4*4+3,bl3,bh3);
          }
        }
      }
    }
    __syncthreads();
    if (wm < 4) {
#pragma unroll
      for (int s = 0; s < 8; ++s)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int cc = wm*16 + g + ((r >> 1) ? 8 : 0);
          const int b = wn*64 + s*8 + 2*tq + (r & 1);
          float dvv = 0.0f;
          if (cc < cn) {
            const long t = (long)c*64 + cc;
            const float vn = u[(t*NVH + vh)*HVD + b] - vp[s][r];
            vnew[(t*NVH + vh)*HVD + b] = vn;
            dvv = vn*expf(gl[(long)c*NVH + vh] - gc[t*NVH + vh]);
          }
          unsigned short hv;
          asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(dvv));
          Tb[cc*SLD + b] = hv;
        }
    }
    for (int i = tid; i < 64*128; i += 512) {
      const int cc = i >> 7, jj = i & 127;
      float kv = (cc < cn)
        ? conv[((long)c*64 + cc)*CONV + KEY + kh*HKD + jj] : 0.0f;
      unsigned short hv;
      asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(kv));
      Sh[cc*SLD + jj] = hv;
    }
    __syncthreads();
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
      for (int r = 0; r < 4; ++r) S[s][r] *= eglc;
#pragma unroll
    for (int k0 = 0; k0 < 64; k0 += 16) {
      int a0,a1,a2,a3;
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3)
        : "r"((unsigned)__cvta_generic_to_shared(
            &Sh[(k0 + (lane & 7) + ((lane & 16) ? 8 : 0))*SLD +
                wm*16 + ((lane & 8) ? 8 : 0)])));
#pragma unroll
      for (int s4 = 0; s4 < 2; ++s4) {
        const int nb = wn*64 + s4*32;
        int bl0,bl1,bl2,bl3, bh0,bh1,bh2,bh3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
          "{%0,%1,%2,%3}, [%4];\n"
          : "=r"(bl0),"=r"(bl1),"=r"(bl2),"=r"(bl3)
          : "r"((unsigned)__cvta_generic_to_shared(
              &Tb[(k0 + (lane & 7))*SLD + nb + (lane >> 3)*8])));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
          "{%0,%1,%2,%3}, [%4];\n"
          : "=r"(bh0),"=r"(bh1),"=r"(bh2),"=r"(bh3)
          : "r"((unsigned)__cvta_generic_to_shared(
              &Tb[(k0 + 8 + (lane & 7))*SLD + nb + (lane >> 3)*8])));
        STMMA(S,s4*4+0,bl0,bh0); STMMA(S,s4*4+1,bl1,bh1);
        STMMA(S,s4*4+2,bl2,bh2); STMMA(S,s4*4+3,bl3,bh3);
      }
    }
#undef STMMA
    __syncthreads();
  }
  if (save_state) {
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int j = wm*16 + g + ((r >> 1) ? 8 : 0);
        const int b = wn*64 + s*8 + 2*tq + (r & 1);
        state[((long)vh*HKD + j)*HVD + b] = S[s][r];
      }
  }
}

// gated RMSNorm + silu(z) gate -> fp16, VERBATIM the scan's fused tail.
// grid T*NVH, block 128 (thread = b).
// fp16 round-trip stub (NNTR_GDN_CK_F16STUB=1, =2 harness only): rounds a
// buffer through fp16 in place, emulating what a tensor-core rewrite would
// feed as fp16 mma operands. Used on the conv plane (q/k/v), w/u (before the
// state sweep reads them) and h/v_new (before the out kernel) to measure the
// INTRINSIC fp16-input error envelope that sets the acceptance gate.
__global__ void gdn_ck_f16rt(float *x, long n){
  long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short hh; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hh) : "f"(x[i]));
  float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(hh));
  x[i] = f;
}

__global__ void gdn_ck_gate(const float *o, const float *z,
                            const float *wnorm, unsigned short *normed,
                            float eps, int NVH, int HVD){
  const int vh = blockIdx.x % NVH;
  const long t = blockIdx.x / NVH;
  const int b = threadIdx.x;
  __shared__ float red[128];
  const long off = t*(long)(NVH*HVD) + vh*HVD + b;
  const float ov = o[off];
  red[b] = ov*ov;
  __syncthreads();
  for (int s = 64; s > 0; s >>= 1) {
    if (b < s) red[b] += red[b + s];
    __syncthreads();
  }
  const float inv = 1.0f/sqrtf(red[0]/(float)HVD + eps);
  const float zv = z[off];
  normed[off] = f2h(ov*inv*wnorm[b]*gdn_silu(zv));
}

// Device conv-ring rebuild: the byte-exact replacement for the layer's host
// save_ring loop (전부 GPU). One thread per conv channel; the old ring is
// read into registers FIRST, so the in-place update is race-free even when
// the chunk is shorter than the ring.
__global__ void gdn_save_ring(const float *qkv, float *ring, int T, int CONV,
                              int KS, int has_prev){
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  if (c >= CONV) return;
  float prev[8]; // KS <= 9 in practice (KS-1 slots); 4 here
  for (int j = 0; j < KS-1; ++j)
    prev[j] = has_prev ? ring[(long)c*(KS-1) + j] : 0.0f;
  for (int j = 0; j < KS-1; ++j) {
    const int ti = T - (KS-1) + j;
    float v;
    if (ti >= 0)            v = qkv[(long)ti*CONV + c];
    else if (has_prev)      v = prev[(KS-1) + ti];
    else                    v = 0.0f;
    ring[(long)c*(KS-1) + j] = v;
  }
}
} // extern "C"
)CU";

namespace {
std::mutex g_gdn_mtx;
// device copies of the small per-layer fp32 params, keyed by the layer's
// stable heap wconv pointer (uploaded once per layer)
struct DevParams {
  float *wconv, *alog, *dtb, *wnorm;
};
std::map<const float *, DevParams> g_gdn_params;
// shared decode scratch (layers run sequentially on one stream)
float *g_qkv = nullptr, *g_z = nullptr, *g_pb = nullptr, *g_pa = nullptr;
float *g_conv = nullptr, *g_normed = nullptr;
size_t g_qkv_cap = 0, g_z_cap = 0, g_pb_cap = 0, g_pa_cap = 0;
size_t g_conv_cap = 0, g_normed_cap = 0;
// prefill scratch: sized by T, so kept apart from the decode buffers above
float *g_pf_conv = nullptr, *g_pf_qkdot = nullptr;
unsigned short *g_pf_normed = nullptr;
size_t g_pf_conv_cap = 0, g_pf_qkdot_cap = 0, g_pf_normed_cap = 0;

bool grow(void **p, size_t *cap, size_t need) {
  if (need <= *cap)
    return true;
  if (StreamManager::Global().isCapturing())
    return false;
  if (*p)
    cudaFree(*p);
  *p = nullptr;
  *cap = 0;
  if (cudaMalloc(p, need) != cudaSuccess)
    return false;
  *cap = need;
  return true;
}

bool ensure_scratch(unsigned int CONV, unsigned int VAL, unsigned int NVH) {
  return grow((void **)&g_qkv, &g_qkv_cap, (size_t)CONV * 4) &&
         grow((void **)&g_z, &g_z_cap, (size_t)VAL * 4) &&
         grow((void **)&g_pb, &g_pb_cap, (size_t)NVH * 4) &&
         grow((void **)&g_pa, &g_pa_cap, (size_t)NVH * 4) &&
         grow((void **)&g_conv, &g_conv_cap, (size_t)CONV * 4) &&
         grow((void **)&g_normed, &g_normed_cap, (size_t)VAL * 4);
}

const DevParams *ensure_params(const float *h_wconv, const float *h_alog,
                               const float *h_dtb, const float *h_wnorm,
                               unsigned int CONV, unsigned int KS,
                               unsigned int NVH, unsigned int HVD,
                               cudaStream_t stream) {
  auto it = g_gdn_params.find(h_wconv);
  if (it != g_gdn_params.end())
    return &it->second;
  if (StreamManager::Global().isCapturing())
    return nullptr;
  DevParams p{};
  const size_t sz[4] = {(size_t)CONV * KS * 4, (size_t)NVH * 4,
                        (size_t)NVH * 4, (size_t)HVD * 4};
  float **dst[4] = {&p.wconv, &p.alog, &p.dtb, &p.wnorm};
  const float *src[4] = {h_wconv, h_alog, h_dtb, h_wnorm};
  for (int i = 0; i < 4; ++i) {
    if (cudaMalloc((void **)dst[i], sz[i]) != cudaSuccess ||
        cudaMemcpyAsync(*dst[i], src[i], sz[i], cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      cudaGetLastError();
      for (int j = 0; j <= i; ++j)
        if (*dst[j])
          cudaFree(*dst[j]);
      return nullptr;
    }
  }
  return &g_gdn_params.emplace(h_wconv, p).first->second;
}
} // namespace

bool cuda_gdn_prewarm(unsigned int H, unsigned int NVH, unsigned int NKH,
                      unsigned int HKD, unsigned int HVD) {
  if (H == 0 || NVH == 0 || NKH == 0)
    return true;
  std::lock_guard<std::mutex> lk(g_gdn_mtx);
  return ensure_scratch(2 * NKH * HKD + NVH * HVD, NVH * HVD, NVH);
}

bool cuda_gdn_prefill_fp16(const float *p_qkv, const float *p_z,
                           const float *p_b, const float *p_a,
                           const unsigned short *wout, const float *h_wconv,
                           const float *h_alog, const float *h_dtb,
                           const float *h_wnorm, float *state,
                           const float *ring, unsigned short *out,
                           unsigned int T, unsigned int H, unsigned int NVH,
                           unsigned int NKH, unsigned int HKD, unsigned int HVD,
                           unsigned int KS, float eps, bool seed_state,
                           bool save_state) {
  const unsigned int KEY = NKH * HKD, VAL = NVH * HVD;
  const unsigned int CONV = 2 * KEY + VAL;
  // The scan holds S in registers at exactly 512 threads x 32 floats, which
  // covers 128x128 and nothing else. A general version would have to spill S
  // to shared and give up the whole point (see the header).
  if (T == 0 || HKD != 128 || HVD != 128 || NVH == 0 || NKH == 0 ||
      NVH % NKH != 0 || KS < 2 || p_qkv == nullptr || out == nullptr)
    return false;
  std::lock_guard<std::mutex> lk(g_gdn_mtx);
  if (!grow((void **)&g_pf_conv, &g_pf_conv_cap, (size_t)T * CONV * 4) ||
      !grow((void **)&g_pf_qkdot, &g_pf_qkdot_cap, (size_t)T * NKH * 4) ||
      !grow((void **)&g_pf_normed, &g_pf_normed_cap, (size_t)T * VAL * 2)) {
    fprintf(stderr, "[cuda_gdn] prefill scratch alloc FAILED (T=%u): %s\n", T,
            cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  auto &sm = StreamManager::Global();
  cudaStream_t stream = sm.GetStream();
  // NNTR_GDN_P_DBG=1: per-stage GPU ms via events (first few calls only).
  static const bool g_pdbg = []() {
    const char *e = std::getenv("NNTR_GDN_P_DBG");
    return e != nullptr && e[0] == '1';
  }();
  static int g_pdbg_n = 0;
  cudaEvent_t pev[5];
  const bool pdbg_this = g_pdbg && g_pdbg_n < 3;
  if (pdbg_this)
    for (int i = 0; i < 5; ++i)
      cudaEventCreate(&pev[i]);
  auto pstamp = [&](int i) {
    if (pdbg_this)
      cudaEventRecord(pev[i], stream);
  };
  pstamp(0);
  const DevParams *dp =
    ensure_params(h_wconv, h_alog, h_dtb, h_wnorm, CONV, KS, NVH, HVD, stream);
  if (!dp) {
    fprintf(stderr, "[cuda_gdn] prefill param upload FAILED\n");
    return false;
  }
  auto &ctx = CudaContext::Global();
  auto kcv = ctx.registerCudaKernel(GDN_SRC, "gdn_conv_prefill");
  auto kln = ctx.registerCudaKernel(GDN_SRC, "gdn_l2norm_prefill");
  auto ksc = ctx.registerCudaKernel(GDN_SRC, "gdn_scan_prefill");
  if (!kcv || !kln || !ksc) {
    ml_loge("[CUDA] gdn: prefill kernel registration failed");
    return false;
  }
  int iT = (int)T, iCONV = (int)CONV, iKS = (int)KS, iKEY = (int)KEY;
  int iNVH = (int)NVH, iNKH = (int)NKH, iHKD = (int)HKD, iHVD = (int)HVD;
  int iVAL = (int)VAL;
  int has_ring = (seed_state && ring != nullptr) ? 1 : 0;
  int i_seed = seed_state ? 1 : 0, i_save = save_state ? 1 : 0;
  float scale = 1.0f / std::sqrt((float)HKD);

  { // conv1d + SiLU over the whole plane
    kcv->SetKernelArguments(0, &p_qkv, sizeof(p_qkv));
    kcv->SetKernelArguments(1, &dp->wconv, sizeof(dp->wconv));
    kcv->SetKernelArguments(2, &ring, sizeof(ring));
    kcv->SetKernelArguments(3, &g_pf_conv, sizeof(g_pf_conv));
    kcv->SetKernelArguments(4, &iT, sizeof(iT));
    kcv->SetKernelArguments(5, &iCONV, sizeof(iCONV));
    kcv->SetKernelArguments(6, &iKS, sizeof(iKS));
    kcv->SetKernelArguments(7, &has_ring, sizeof(has_ring));
    const long total = (long)T * CONV;
    const int B2 = 256;
    const int g[3] = {(int)((total + B2 - 1) / B2), 1, 1}, b3[3] = {B2, 1, 1};
    if (!sm.DispatchCommand(*kcv, g, b3)) {
      fprintf(stderr, "[cuda_gdn] prefill conv dispatch FAILED: %s\n",
              cudaGetErrorString(cudaGetLastError()));
      return false;
    }
  }
  pstamp(1);
  { // in-place l2norm(q,k) + the per-(token,k-head) q.k
    kln->SetKernelArguments(0, &g_pf_conv, sizeof(g_pf_conv));
    kln->SetKernelArguments(1, &g_pf_qkdot, sizeof(g_pf_qkdot));
    kln->SetKernelArguments(2, &iCONV, sizeof(iCONV));
    kln->SetKernelArguments(3, &iKEY, sizeof(iKEY));
    kln->SetKernelArguments(4, &iNKH, sizeof(iNKH));
    kln->SetKernelArguments(5, &iHKD, sizeof(iHKD));
    kln->SetKernelArguments(6, &eps, sizeof(eps));
    const int g[3] = {(int)(T * NKH), 1, 1}, b3[3] = {iHKD, 1, 1};
    if (!sm.DispatchCommand(*kln, g, b3)) {
      fprintf(stderr, "[cuda_gdn] prefill l2norm dispatch FAILED: %s\n",
              cudaGetErrorString(cudaGetLastError()));
      return false;
    }
  }
  pstamp(2);
  { // the sequential scan + fused gated RMSNorm
    ksc->SetKernelArguments(0, &g_pf_conv, sizeof(g_pf_conv));
    ksc->SetKernelArguments(1, &p_z, sizeof(p_z));
    ksc->SetKernelArguments(2, &p_b, sizeof(p_b));
    ksc->SetKernelArguments(3, &p_a, sizeof(p_a));
    ksc->SetKernelArguments(4, &g_pf_qkdot, sizeof(g_pf_qkdot));
    ksc->SetKernelArguments(5, &dp->alog, sizeof(dp->alog));
    ksc->SetKernelArguments(6, &dp->dtb, sizeof(dp->dtb));
    ksc->SetKernelArguments(7, &dp->wnorm, sizeof(dp->wnorm));
    ksc->SetKernelArguments(8, &state, sizeof(state));
    ksc->SetKernelArguments(9, &g_pf_normed, sizeof(g_pf_normed));
    ksc->SetKernelArguments(10, &iT, sizeof(iT));
    ksc->SetKernelArguments(11, &iNVH, sizeof(iNVH));
    ksc->SetKernelArguments(12, &iNKH, sizeof(iNKH));
    ksc->SetKernelArguments(13, &iHKD, sizeof(iHKD));
    ksc->SetKernelArguments(14, &iHVD, sizeof(iHVD));
    ksc->SetKernelArguments(15, &iKEY, sizeof(iKEY));
    ksc->SetKernelArguments(16, &iCONV, sizeof(iCONV));
    ksc->SetKernelArguments(17, &iVAL, sizeof(iVAL));
    ksc->SetKernelArguments(18, &scale, sizeof(scale));
    ksc->SetKernelArguments(19, &eps, sizeof(eps));
    ksc->SetKernelArguments(20, &i_seed, sizeof(i_seed));
    ksc->SetKernelArguments(21, &i_save, sizeof(i_save));
    const int g[3] = {iNVH, 1, 1}, b3[3] = {512, 1, 1};
    if (!sm.DispatchCommand(*ksc, g, b3)) {
      fprintf(stderr, "[cuda_gdn] prefill scan dispatch FAILED: %s\n",
              cudaGetErrorString(cudaGetLastError()));
      return false;
    }
  }
  pstamp(3);
  // out_proj: [T,VAL] fp16 x [VAL,H] fp16 -> [T,H] fp16, fp32 accumulate.
  // cuBLAS drains, so the host may read `out` immediately after this returns.
  if (!cuda_fc_dense_gemm_fp16(g_pf_normed, wout, out, T, H, VAL)) {
    fprintf(stderr, "[cuda_gdn] prefill out_proj FAILED\n");
    return false;
  }
  pstamp(4);
  if (pdbg_this) {
    cudaEventSynchronize(pev[4]);
    const char *nm[4] = {"conv", "l2norm", "scan", "out_proj"};
    fprintf(stderr, "[gdn_p_dbg] T=%u ", T);
    for (int i = 0; i < 4; ++i) {
      float ms = 0.f;
      cudaEventElapsedTime(&ms, pev[i], pev[i + 1]);
      fprintf(stderr, "%s=%.2fms ", nm[i], ms);
    }
    fprintf(stderr, "\n");
    for (int i = 0; i < 5; ++i)
      cudaEventDestroy(pev[i]);
    ++g_pdbg_n;
  }
  return true;
}

// chunked-prefill scratch (fp32, grow-once, shared by all 30 layers)
namespace {
float *g_ck_gc = nullptr, *g_ck_beta = nullptr, *g_ck_gl = nullptr;
float *g_ck_A = nullptr, *g_ck_w = nullptr, *g_ck_u = nullptr;
float *g_ck_vn = nullptr, *g_ck_h = nullptr, *g_ck_o = nullptr;
size_t c_ck_gc = 0, c_ck_beta = 0, c_ck_gl = 0, c_ck_A = 0, c_ck_w = 0,
       c_ck_u = 0, c_ck_vn = 0, c_ck_h = 0, c_ck_o = 0;
} // namespace

bool cuda_gdn_prefill_chunked_fp16(
  const float *p_qkv, const float *p_z, const float *p_b, const float *p_a,
  const unsigned short *wout, const float *h_wconv, const float *h_alog,
  const float *h_dtb, const float *h_wnorm, float *state, const float *ring,
  unsigned short *out, unsigned int T, unsigned int H, unsigned int NVH,
  unsigned int NKH, unsigned int HKD, unsigned int HVD, unsigned int KS,
  float eps, bool seed_state, bool save_state) {
  const unsigned int KEY = NKH * HKD, VAL = NVH * HVD;
  const unsigned int CONV = 2 * KEY + VAL;
  // Same geometry pins as the sequential scan (S in registers, C=64 tiles).
  if (T == 0 || HKD != 128 || HVD != 128 || NVH == 0 || NKH == 0 ||
      NVH % NKH != 0 || KS < 2 || KS > 8 || p_qkv == nullptr || out == nullptr)
    return false;
  const unsigned int NT = (T + 63) / 64;
  std::lock_guard<std::mutex> lk(g_gdn_mtx);
  if (!grow((void **)&g_pf_conv, &g_pf_conv_cap, (size_t)T * CONV * 4) ||
      !grow((void **)&g_pf_normed, &g_pf_normed_cap, (size_t)T * VAL * 2) ||
      !grow((void **)&g_pf_qkdot, &g_pf_qkdot_cap, (size_t)T * NKH * 4) ||
      !grow((void **)&g_ck_gc, &c_ck_gc, (size_t)T * NVH * 4) ||
      !grow((void **)&g_ck_beta, &c_ck_beta, (size_t)T * NVH * 4) ||
      !grow((void **)&g_ck_gl, &c_ck_gl, (size_t)NT * NVH * 4) ||
      !grow((void **)&g_ck_A, &c_ck_A, (size_t)NT * NVH * 4096 * 4) ||
      !grow((void **)&g_ck_w, &c_ck_w, (size_t)T * NVH * HKD * 4) ||
      !grow((void **)&g_ck_u, &c_ck_u, (size_t)T * NVH * HVD * 4) ||
      !grow((void **)&g_ck_vn, &c_ck_vn, (size_t)T * NVH * HVD * 4) ||
      !grow((void **)&g_ck_h, &c_ck_h, (size_t)NT * NVH * 16384 * 4) ||
      !grow((void **)&g_ck_o, &c_ck_o, (size_t)T * VAL * 4)) {
    fprintf(stderr, "[cuda_gdn] chunked scratch alloc FAILED (T=%u): %s\n", T,
            cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  auto &sm = StreamManager::Global();
  cudaStream_t stream = sm.GetStream();
  const DevParams *dp =
    ensure_params(h_wconv, h_alog, h_dtb, h_wnorm, CONV, KS, NVH, HVD, stream);
  if (!dp)
    return false;
  auto &ctx = CudaContext::Global();
  auto kcv = ctx.registerCudaKernel(GDN_SRC, "gdn_conv_prefill");
  auto kln = ctx.registerCudaKernel(GDN_SRC, "gdn_l2norm_prefill");
  // NNTR_GDN_CK_TC=1: the fp16 m16n8k16 tensor-core chunked kernels
  // (kkt/wu/state/out). Opt-in; gated by the =2 harness (out <= 0.0625,
  // state <= 0.021) + text identity.
  static const bool g_ck_tc = []() {
    const char *e = std::getenv("NNTR_GDN_CK_TC");
    return e == nullptr || e[0] != '0'; // default ON; =0 restores fp32 SIMT
  }();
  auto kgc = ctx.registerCudaKernel(GDN_SRC, "gdn_ck_gcum");
  auto kkt = ctx.registerCudaKernel(GDN_SRC, g_ck_tc ? "gdn_ck_kkt_tc"
                                                     : "gdn_ck_kkt");
  auto ktr = ctx.registerCudaKernel(GDN_SRC, "gdn_ck_tril");
  auto kwu = ctx.registerCudaKernel(GDN_SRC, g_ck_tc ? "gdn_ck_wu_tc"
                                                     : "gdn_ck_wu");
  auto kst = ctx.registerCudaKernel(GDN_SRC, g_ck_tc ? "gdn_ck_state_tc"
                                                     : "gdn_ck_state");
  auto kou =
    ctx.registerCudaKernel(GDN_SRC, g_ck_tc ? "gdn_ck_out_tc" : "gdn_ck_out");
  auto kga = ctx.registerCudaKernel(GDN_SRC, "gdn_ck_gate");
  // fp16-input stub for the acceptance-gate measurement (task #11).
  static const bool g_f16stub = []() {
    const char *e = std::getenv("NNTR_GDN_CK_F16STUB");
    return e != nullptr && e[0] == '1';
  }();
  auto kf16 = g_f16stub ? ctx.registerCudaKernel(GDN_SRC, "gdn_ck_f16rt")
                        : nullptr;
  auto f16rt = [&](float *p, long n) -> bool {
    if (!g_f16stub)
      return true;
    if (!kf16)
      return false;
    kf16->SetKernelArguments(0, &p, sizeof(p));
    kf16->SetKernelArguments(1, &n, sizeof(n));
    const int gg[3] = {(int)((n + 255) / 256), 1, 1}, bb[3] = {256, 1, 1};
    return sm.DispatchCommand(*kf16, gg, bb);
  };
  if (!kcv || !kln || !kgc || !kkt || !ktr || !kwu || !kst || !kou || !kga) {
    ml_loge("[CUDA] gdn: chunked kernel registration failed");
    return false;
  }
  int iT = (int)T, iCONV = (int)CONV, iKS = (int)KS, iKEY = (int)KEY;
  int iNVH = (int)NVH, iNKH = (int)NKH, iHKD = (int)HKD, iHVD = (int)HVD;
  int has_ring = (seed_state && ring != nullptr) ? 1 : 0;
  int i_seed = seed_state ? 1 : 0, i_save = save_state ? 1 : 0;
  float scale = 1.0f / std::sqrt((float)HKD);

  // NNTR_GDN_CK_DBG=1: per-kernel GPU ms via events (first few calls only).
  static const bool ck_dbg = []() {
    const char *e = std::getenv("NNTR_GDN_CK_DBG");
    return e != nullptr && e[0] == '1';
  }();
  static int ck_dbg_n = 0;
  cudaEvent_t ev[10];
  const bool dbg_this = ck_dbg && ck_dbg_n < 3;
  if (dbg_this)
    for (int i = 0; i < 10; ++i)
      cudaEventCreate(&ev[i]);
  auto stamp = [&](int i) {
    if (dbg_this)
      cudaEventRecord(ev[i], stream);
  };
  stamp(0);

  { // conv1d + SiLU (identical to the sequential path)
    kcv->SetKernelArguments(0, &p_qkv, sizeof(p_qkv));
    kcv->SetKernelArguments(1, &dp->wconv, sizeof(dp->wconv));
    kcv->SetKernelArguments(2, &ring, sizeof(ring));
    kcv->SetKernelArguments(3, &g_pf_conv, sizeof(g_pf_conv));
    kcv->SetKernelArguments(4, &iT, sizeof(iT));
    kcv->SetKernelArguments(5, &iCONV, sizeof(iCONV));
    kcv->SetKernelArguments(6, &iKS, sizeof(iKS));
    kcv->SetKernelArguments(7, &has_ring, sizeof(has_ring));
    const long total = (long)T * CONV;
    const int g[3] = {(int)((total + 255) / 256), 1, 1}, b3[3] = {256, 1, 1};
    if (!sm.DispatchCommand(*kcv, g, b3))
      return false;
  }
  stamp(1);
  { // l2norm(q,k) in place (qkdot is computed but unused by the chunked form)
    kln->SetKernelArguments(0, &g_pf_conv, sizeof(g_pf_conv));
    kln->SetKernelArguments(1, &g_pf_qkdot, sizeof(g_pf_qkdot));
    kln->SetKernelArguments(2, &iCONV, sizeof(iCONV));
    kln->SetKernelArguments(3, &iKEY, sizeof(iKEY));
    kln->SetKernelArguments(4, &iNKH, sizeof(iNKH));
    kln->SetKernelArguments(5, &iHKD, sizeof(iHKD));
    kln->SetKernelArguments(6, &eps, sizeof(eps));
    const int g[3] = {(int)(T * NKH), 1, 1}, b3[3] = {iHKD, 1, 1};
    if (!sm.DispatchCommand(*kln, g, b3))
      return false;
  }
  stamp(2);
  { // per-token log-decay + beta + in-chunk cumsum
    kgc->SetKernelArguments(0, &p_a, sizeof(p_a));
    kgc->SetKernelArguments(1, &p_b, sizeof(p_b));
    kgc->SetKernelArguments(2, &dp->alog, sizeof(dp->alog));
    kgc->SetKernelArguments(3, &dp->dtb, sizeof(dp->dtb));
    kgc->SetKernelArguments(4, &g_ck_gc, sizeof(g_ck_gc));
    kgc->SetKernelArguments(5, &g_ck_beta, sizeof(g_ck_beta));
    kgc->SetKernelArguments(6, &g_ck_gl, sizeof(g_ck_gl));
    kgc->SetKernelArguments(7, &iT, sizeof(iT));
    kgc->SetKernelArguments(8, &iNVH, sizeof(iNVH));
    const int g[3] = {(int)NT, iNVH, 1}, b3[3] = {64, 1, 1};
    if (!sm.DispatchCommand(*kgc, g, b3))
      return false;
  }
  // stub point 1: q/k/v as the TC tiles would see them (conv plane).
  if (!f16rt(g_pf_conv, (long)T * CONV))
    return false;
  {
  }
  stamp(3);
  { // A = beta (k k^T) e^{gdiff}, strictly lower
    kkt->SetKernelArguments(0, &g_pf_conv, sizeof(g_pf_conv));
    kkt->SetKernelArguments(1, &g_ck_gc, sizeof(g_ck_gc));
    kkt->SetKernelArguments(2, &g_ck_beta, sizeof(g_ck_beta));
    kkt->SetKernelArguments(3, &g_ck_A, sizeof(g_ck_A));
    kkt->SetKernelArguments(4, &iT, sizeof(iT));
    kkt->SetKernelArguments(5, &iCONV, sizeof(iCONV));
    kkt->SetKernelArguments(6, &iKEY, sizeof(iKEY));
    kkt->SetKernelArguments(7, &iNVH, sizeof(iNVH));
    kkt->SetKernelArguments(8, &iNKH, sizeof(iNKH));
    kkt->SetKernelArguments(9, &iHKD, sizeof(iHKD));
    const int g[3] = {(int)NT, iNVH, 1}, b3[3] = {256, 1, 1};
    if (!sm.DispatchCommand(*kkt, g, b3))
      return false;
  }
  stamp(4);
  { // (I+A)^-1 in place
    ktr->SetKernelArguments(0, &g_ck_A, sizeof(g_ck_A));
    ktr->SetKernelArguments(1, &iNVH, sizeof(iNVH));
    const int g[3] = {(int)NT, iNVH, 1}, b3[3] = {64, 1, 1};
    if (!sm.DispatchCommand(*ktr, g, b3))
      return false;
  }
  stamp(5);
  { // w, u
    kwu->SetKernelArguments(0, &g_pf_conv, sizeof(g_pf_conv));
    kwu->SetKernelArguments(1, &g_ck_A, sizeof(g_ck_A));
    kwu->SetKernelArguments(2, &g_ck_gc, sizeof(g_ck_gc));
    kwu->SetKernelArguments(3, &g_ck_beta, sizeof(g_ck_beta));
    kwu->SetKernelArguments(4, &g_ck_w, sizeof(g_ck_w));
    kwu->SetKernelArguments(5, &g_ck_u, sizeof(g_ck_u));
    kwu->SetKernelArguments(6, &iT, sizeof(iT));
    kwu->SetKernelArguments(7, &iCONV, sizeof(iCONV));
    kwu->SetKernelArguments(8, &iKEY, sizeof(iKEY));
    kwu->SetKernelArguments(9, &iNVH, sizeof(iNVH));
    kwu->SetKernelArguments(10, &iNKH, sizeof(iNKH));
    kwu->SetKernelArguments(11, &iHKD, sizeof(iHKD));
    kwu->SetKernelArguments(12, &iHVD, sizeof(iHVD));
    const int g[3] = {(int)NT, iNVH, 1}, b3[3] = {g_ck_tc ? 256 : 128, 1, 1};
    if (!sm.DispatchCommand(*kwu, g, b3))
      return false;
  }
  // stub point 2: w/u before the state sweep consumes them.
  if (!f16rt(g_ck_w, (long)T * NVH * HKD) ||
      !f16rt(g_ck_u, (long)T * NVH * HVD))
    return false;
  stamp(6);
  { // state propagation (the only sequential piece)
    kst->SetKernelArguments(0, &g_pf_conv, sizeof(g_pf_conv));
    kst->SetKernelArguments(1, &g_ck_w, sizeof(g_ck_w));
    kst->SetKernelArguments(2, &g_ck_u, sizeof(g_ck_u));
    kst->SetKernelArguments(3, &g_ck_gc, sizeof(g_ck_gc));
    kst->SetKernelArguments(4, &g_ck_gl, sizeof(g_ck_gl));
    kst->SetKernelArguments(5, &g_ck_h, sizeof(g_ck_h));
    kst->SetKernelArguments(6, &g_ck_vn, sizeof(g_ck_vn));
    kst->SetKernelArguments(7, &state, sizeof(state));
    kst->SetKernelArguments(8, &iT, sizeof(iT));
    kst->SetKernelArguments(9, &iCONV, sizeof(iCONV));
    kst->SetKernelArguments(10, &iKEY, sizeof(iKEY));
    kst->SetKernelArguments(11, &iNVH, sizeof(iNVH));
    kst->SetKernelArguments(12, &iNKH, sizeof(iNKH));
    kst->SetKernelArguments(13, &iHKD, sizeof(iHKD));
    kst->SetKernelArguments(14, &iHVD, sizeof(iHVD));
    kst->SetKernelArguments(15, &i_seed, sizeof(i_seed));
    kst->SetKernelArguments(16, &i_save, sizeof(i_save));
    const int g[3] = {iNVH, 1, 1}, b3[3] = {512, 1, 1};
    if (!sm.DispatchCommand(*kst, g, b3))
      return false;
  }
  // stub point 3: h and v_new before the out kernel consumes them.
  if (!f16rt(g_ck_h, (long)NT * NVH * 16384) ||
      !f16rt(g_ck_vn, (long)T * NVH * HVD))
    return false;
  stamp(7);
  { // outputs (fully parallel over chunks)
    kou->SetKernelArguments(0, &g_pf_conv, sizeof(g_pf_conv));
    kou->SetKernelArguments(1, &g_ck_gc, sizeof(g_ck_gc));
    kou->SetKernelArguments(2, &g_ck_h, sizeof(g_ck_h));
    kou->SetKernelArguments(3, &g_ck_vn, sizeof(g_ck_vn));
    kou->SetKernelArguments(4, &g_ck_o, sizeof(g_ck_o));
    kou->SetKernelArguments(5, &scale, sizeof(scale));
    kou->SetKernelArguments(6, &iT, sizeof(iT));
    kou->SetKernelArguments(7, &iCONV, sizeof(iCONV));
    kou->SetKernelArguments(8, &iKEY, sizeof(iKEY));
    kou->SetKernelArguments(9, &iNVH, sizeof(iNVH));
    kou->SetKernelArguments(10, &iNKH, sizeof(iNKH));
    kou->SetKernelArguments(11, &iHKD, sizeof(iHKD));
    kou->SetKernelArguments(12, &iHVD, sizeof(iHVD));
    const int g[3] = {(int)NT, iNVH, 1}, b3[3] = {g_ck_tc ? 256 : 128, 1, 1};
    if (!sm.DispatchCommand(*kou, g, b3))
      return false;
  }
  stamp(8);
  { // gated RMSNorm + silu(z) -> fp16 (the scan's fused tail, un-fused)
    kga->SetKernelArguments(0, &g_ck_o, sizeof(g_ck_o));
    kga->SetKernelArguments(1, &p_z, sizeof(p_z));
    kga->SetKernelArguments(2, &dp->wnorm, sizeof(dp->wnorm));
    kga->SetKernelArguments(3, &g_pf_normed, sizeof(g_pf_normed));
    kga->SetKernelArguments(4, &eps, sizeof(eps));
    kga->SetKernelArguments(5, &iNVH, sizeof(iNVH));
    kga->SetKernelArguments(6, &iHVD, sizeof(iHVD));
    const int g[3] = {(int)(T * NVH), 1, 1}, b3[3] = {iHVD, 1, 1};
    if (!sm.DispatchCommand(*kga, g, b3))
      return false;
  }
  stamp(9);
  if (dbg_this) {
    cudaEventSynchronize(ev[9]);
    float ms[9];
    const char *nm[9] = {"conv", "l2norm", "gcum", "kkt", "tril",
                         "wu",   "state",  "out",  "gate"};
    fprintf(stderr, "[gdn_ck_dbg] T=%u ", T);
    for (int i = 0; i < 9; ++i) {
      cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]);
      fprintf(stderr, "%s=%.2fms ", nm[i], ms[i]);
    }
    fprintf(stderr, "\n");
    for (int i = 0; i < 10; ++i)
      cudaEventDestroy(ev[i]);
    ++ck_dbg_n;
  }
  if (!cuda_fc_dense_gemm_fp16(g_pf_normed, wout, out, T, H, VAL)) {
    fprintf(stderr, "[cuda_gdn] chunked out_proj FAILED\n");
    return false;
  }
  return true;
}

bool cuda_gdn_save_ring_dev(const float *p_qkv, float *ring, unsigned int T,
                            unsigned int CONV, unsigned int KS,
                            bool has_prev) {
  if (p_qkv == nullptr || ring == nullptr || CONV == 0 || KS < 2 || KS > 8)
    return false;
  auto k = CudaContext::Global().registerCudaKernel(GDN_SRC, "gdn_save_ring");
  if (!k)
    return false;
  int iT = (int)T, iCONV = (int)CONV, iKS = (int)KS;
  int ihp = has_prev ? 1 : 0;
  k->SetKernelArguments(0, &p_qkv, sizeof(p_qkv));
  k->SetKernelArguments(1, &ring, sizeof(ring));
  k->SetKernelArguments(2, &iT, sizeof(iT));
  k->SetKernelArguments(3, &iCONV, sizeof(iCONV));
  k->SetKernelArguments(4, &iKS, sizeof(iKS));
  k->SetKernelArguments(5, &ihp, sizeof(ihp));
  const int g[3] = {(int)((CONV + 255) / 256), 1, 1}, b3[3] = {256, 1, 1};
  return StreamManager::Global().DispatchCommand(*k, g, b3);
}

bool cuda_gdn_decode_fp16(const unsigned short *x, const unsigned short *wqkv,
                          const unsigned short *wz, const unsigned short *wb,
                          const unsigned short *wa, const unsigned short *wout,
                          const float *h_wconv, const float *h_alog,
                          const float *h_dtb, const float *h_wnorm,
                          float *state, float *ring, unsigned short *out,
                          unsigned int H, unsigned int NVH, unsigned int NKH,
                          unsigned int HKD, unsigned int HVD, unsigned int KS,
                          float eps) {
  const unsigned int KEY = NKH * HKD, VAL = NVH * HVD;
  const unsigned int CONV = 2 * KEY + VAL;
  // static shared/block limits of the kernels above
  if (HVD == 0 || (HVD & (HVD - 1)) || HVD > 1024 || HKD > 128 || H > 4096 ||
      VAL > 4096 || NVH == 0 || NKH == 0 || NVH % NKH != 0 || KS < 2)
    return false;
  std::lock_guard<std::mutex> lk(g_gdn_mtx);
  if (!ensure_scratch(CONV, VAL, NVH)) {
    fprintf(stderr, "[cuda_gdn] scratch alloc FAILED: %s\n",
            cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  auto &sm = StreamManager::Global();
  cudaStream_t stream = sm.GetStream();
  const DevParams *dp =
    ensure_params(h_wconv, h_alog, h_dtb, h_wnorm, CONV, KS, NVH, HVD, stream);
  if (!dp) {
    fprintf(stderr, "[cuda_gdn] param upload FAILED\n");
    return false;
  }
  auto &ctx = CudaContext::Global();
  auto kg = ctx.registerCudaKernel(GDN_SRC, "gdn_gemv_h_f");
  auto ko = ctx.registerCudaKernel(GDN_SRC, "gdn_gemv_f_h");
  auto kg4 = ctx.registerCudaKernel(GDN_SRC, "gdn_gemv_h_f4");
  auto ko4 = ctx.registerCudaKernel(GDN_SRC, "gdn_gemv_f_h4");
  auto kc = ctx.registerCudaKernel(GDN_SRC, "gdn_conv_ring");
  auto kd = ctx.registerCudaKernel(GDN_SRC, "gdn_delta_head");
  if (!kg || !ko || !kg4 || !ko4 || !kc || !kd) {
    ml_loge("[CUDA] gdn: kernel registration failed");
    return false;
  }
  const int B = 256;
  // Take the vector-4 kernel whenever the row is 4-wide and 8-byte aligned.
  // Every real 35B shape qualifies (N = 8192/4096/2048/32); the scalar
  // kernels remain as the general fallback rather than being deleted.
  auto vec_ok = [](const unsigned short *W, int N) {
    return (N % 4 == 0) && ((reinterpret_cast<uintptr_t>(W) & 7u) == 0);
  };
  auto gemv_h = [&](const unsigned short *W, float *dst, int K, int N) {
    const bool v4 = vec_ok(W, N);
    auto &kk = v4 ? *kg4 : *kg;
    kk.SetKernelArguments(0, &x, sizeof(x));
    kk.SetKernelArguments(1, &W, sizeof(W));
    kk.SetKernelArguments(2, &dst, sizeof(dst));
    kk.SetKernelArguments(3, &K, sizeof(K));
    kk.SetKernelArguments(4, &N, sizeof(N));
    const int lanes = v4 ? (N / 4) : N;
    const int g[3] = {(lanes + B - 1) / B, 1, 1}, b[3] = {B, 1, 1};
    return sm.DispatchCommand(kk, g, b);
  };
  // 1) projections
  if (!gemv_h(wqkv, g_qkv, (int)H, (int)CONV) ||
      !gemv_h(wz, g_z, (int)H, (int)VAL) ||
      !gemv_h(wb, g_pb, (int)H, (int)NVH) ||
      !gemv_h(wa, g_pa, (int)H, (int)NVH)) {
    fprintf(stderr, "[cuda_gdn] gemv dispatch FAILED: %s\n",
            cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  // 2) conv + ring advance
  {
    int cv = (int)CONV, ks = (int)KS;
    kc->SetKernelArguments(0, &g_qkv, sizeof(g_qkv));
    kc->SetKernelArguments(1, &dp->wconv, sizeof(dp->wconv));
    kc->SetKernelArguments(2, &ring, sizeof(ring));
    kc->SetKernelArguments(3, &g_conv, sizeof(g_conv));
    kc->SetKernelArguments(4, &cv, sizeof(cv));
    kc->SetKernelArguments(5, &ks, sizeof(ks));
    const int g[3] = {((int)CONV + B - 1) / B, 1, 1}, b[3] = {B, 1, 1};
    if (!sm.DispatchCommand(*kc, g, b)) {
      fprintf(stderr, "[cuda_gdn] conv dispatch FAILED: %s\n",
              cudaGetErrorString(cudaGetLastError()));
      return false;
    }
  }
  // 3) fused delta recurrence + gated RMSNorm (one block per v-head)
  {
    int nvh = (int)NVH, nkh = (int)NKH, hkd = (int)HKD, hvd = (int)HVD;
    float scale = 1.0f / std::sqrt((float)HKD);
    kd->SetKernelArguments(0, &g_conv, sizeof(g_conv));
    kd->SetKernelArguments(1, &g_z, sizeof(g_z));
    kd->SetKernelArguments(2, &g_pb, sizeof(g_pb));
    kd->SetKernelArguments(3, &g_pa, sizeof(g_pa));
    kd->SetKernelArguments(4, &dp->alog, sizeof(dp->alog));
    kd->SetKernelArguments(5, &dp->dtb, sizeof(dp->dtb));
    kd->SetKernelArguments(6, &dp->wnorm, sizeof(dp->wnorm));
    kd->SetKernelArguments(7, &state, sizeof(state));
    kd->SetKernelArguments(8, &g_normed, sizeof(g_normed));
    kd->SetKernelArguments(9, &nvh, sizeof(nvh));
    kd->SetKernelArguments(10, &nkh, sizeof(nkh));
    kd->SetKernelArguments(11, &hkd, sizeof(hkd));
    kd->SetKernelArguments(12, &hvd, sizeof(hvd));
    kd->SetKernelArguments(13, &scale, sizeof(scale));
    kd->SetKernelArguments(14, &eps, sizeof(eps));
    const int g[3] = {(int)NVH, 1, 1}, b[3] = {(int)HVD, 1, 1};
    if (!sm.DispatchCommand(*kd, g, b)) {
      fprintf(stderr, "[cuda_gdn] delta dispatch FAILED: %s\n",
              cudaGetErrorString(cudaGetLastError()));
      return false;
    }
  }
  // 4) out_proj
  {
    int K = (int)VAL, N = (int)H;
    ko->SetKernelArguments(0, &g_normed, sizeof(g_normed));
    ko->SetKernelArguments(1, &wout, sizeof(wout));
    ko->SetKernelArguments(2, &out, sizeof(out));
    ko->SetKernelArguments(3, &K, sizeof(K));
    ko->SetKernelArguments(4, &N, sizeof(N));
    const int g[3] = {((int)H + B - 1) / B, 1, 1}, b[3] = {B, 1, 1};
    if (!sm.DispatchCommand(*ko, g, b)) {
      fprintf(stderr, "[cuda_gdn] out_proj dispatch FAILED: %s\n",
              cudaGetErrorString(cudaGetLastError()));
      return false;
    }
  }
  // single drain for the whole GDN step (host reads the fp16 output next)
  sm.maybeFinish();
  return true;
}

} // namespace nntrainer::cuda

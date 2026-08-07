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
  // out_proj: [T,VAL] fp16 x [VAL,H] fp16 -> [T,H] fp16, fp32 accumulate.
  // cuBLAS drains, so the host may read `out` immediately after this returns.
  if (!cuda_fc_dense_gemm_fp16(g_pf_normed, wout, out, T, H, VAL)) {
    fprintf(stderr, "[cuda_gdn] prefill out_proj FAILED\n");
    return false;
  }
  return true;
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

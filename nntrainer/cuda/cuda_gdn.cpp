// SPDX-License-Identifier: Apache-2.0
/**
 * @file    cuda_gdn.cpp
 * @brief   Eager GPU GatedDeltaNet single-token decode (see cuda_gdn.h).
 */
#include "cuda_gdn.h"

#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#include <cmath>
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
__device__ __forceinline__ float h2f(unsigned short h){
  unsigned int s=(h&0x8000u)<<16, e=(h>>10)&0x1Fu, m=h&0x3FFu, b;
  if(e==0){ if(m==0){b=s;} else { e=113u; while(!(m&0x400u)){m<<=1;e--;} m&=0x3FFu; b=s|(e<<23)|(m<<13);} }
  else if(e==31){ b=s|0x7F800000u|(m<<13); }
  else { b=s|((e+112u)<<23)|(m<<13); }
  return __uint_as_float(b);
}
__device__ __forceinline__ unsigned short f2h(float f){
  unsigned int x=__float_as_uint(f);
  unsigned int s=(x>>16)&0x8000u; int e=(int)((x>>23)&0xFFu)-127+15; unsigned int m=x&0x7FFFFFu;
  if(e<=0){ if(e<-10) return (unsigned short)s; m|=0x800000u; int sh=14-e; unsigned int r=m>>sh;
            if((m>>(sh-1))&1u) r++; return (unsigned short)(s|r); }
  if(e>=31) return (unsigned short)(s|0x7C00u);
  unsigned int hh=((unsigned int)e<<10)|(m>>13), rem=m&0x1FFFu;
  if(rem>0x1000u||(rem==0x1000u&&(hh&1u))) hh++;
  return (unsigned short)(s|hh);
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
  auto kc = ctx.registerCudaKernel(GDN_SRC, "gdn_conv_ring");
  auto kd = ctx.registerCudaKernel(GDN_SRC, "gdn_delta_head");
  if (!kg || !ko || !kc || !kd) {
    ml_loge("[CUDA] gdn: kernel registration failed");
    return false;
  }
  const int B = 256;
  auto gemv_h = [&](const unsigned short *W, float *dst, int K, int N) {
    kg->SetKernelArguments(0, &x, sizeof(x));
    kg->SetKernelArguments(1, &W, sizeof(W));
    kg->SetKernelArguments(2, &dst, sizeof(dst));
    kg->SetKernelArguments(3, &K, sizeof(K));
    kg->SetKernelArguments(4, &N, sizeof(N));
    const int g[3] = {(N + B - 1) / B, 1, 1}, b[3] = {B, 1, 1};
    return sm.DispatchCommand(*kg, g, b);
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

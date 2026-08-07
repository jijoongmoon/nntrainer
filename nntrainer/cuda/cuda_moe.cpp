// SPDX-License-Identifier: Apache-2.0
/**
 * @file    cuda_moe.cpp
 * @brief   Grouped MoE expert FFN on the device (see header).
 */
#include "cuda_moe.h"

#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#include <cstdio>
#include <mutex>
#include <nntrainer_log.h>

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
  if (StreamManager::Global().isCapturing())
    return false;
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

} // namespace nntrainer::cuda

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
__global__ void moe_router_gemm(const unsigned short *X, const float *Wg,
                                float *L, int T, int H, int E){
  __shared__ float As[16][65];   // [k][m]
  __shared__ float Bs[16][65];   // [k][n]
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
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[kk][tm + i];
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tn + j];
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

  // top-k by repeated argmax; the winner is masked to -1 so it cannot repeat
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

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_elementwise.cpp
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device element-wise ops (NVRTC kernels) -- geglu/add/scalar/slice.
 */

#include "cuda_elementwise.h"

#include <cuda_common.h>
#include <cuda_context.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

#include <cstdlib>
#include <cuda_runtime.h>

namespace nntrainer::cuda {

static const char *ELTWISE_SRC = R"CU(
extern "C" {
// Hardware half<->float conversion. The previous software routines here were
// ~20 integer ops each (with a data-dependent denormal loop); the hardware
// instruction is one, and it was verified BIT-IDENTICAL to the software pair
// over all 65536 half patterns and 4M random floats when the same swap
// shipped in the fc_qint4 epilogues (commit c6b318763). Every kernel in this
// source inherits the swap with no value change.
__device__ __forceinline__ float ew_h2f(unsigned short h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short ew_f2h(float f) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
__global__ void geglu_fp16(const unsigned short *gate, const unsigned short *up,
                           unsigned short *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = ew_h2f(gate[i]);
  const float k = 0.7978845608028654f;
  float g = 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
  out[i] = ew_f2h(g * ew_h2f(up[i]));
}
// SwiGLU: out[i] = silu(gate[i]) * up[i], silu(x) = x / (1 + exp(-x)) (qwen3/
// llama FFN). Same shape as geglu_fp16, SiLU gate instead of gelu_tanh.
__global__ void swiglu_fp16(const unsigned short *gate, const unsigned short *up,
                            unsigned short *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = ew_h2f(gate[i]);
  float s = x / (1.0f + expf(-x));
  out[i] = ew_f2h(s * ew_h2f(up[i]));
}
// Fused sigmoid gates. sigmoid_glu: out[i] = sigmoid(gate[i]) * x[i]
// (attention output gate). sigmoid_add: out[i] = sigmoid(gate[i]) + emb[i]
// (PLE mix, method 1). FP32 math like the host CpuComputeOps loops.
__global__ void sigmoid_glu_fp16(const unsigned short *gate,
                                 const unsigned short *x, unsigned short *out,
                                 int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = 1.0f / (1.0f + expf(-ew_h2f(gate[i])));
  out[i] = ew_f2h(g * ew_h2f(x[i]));
}
// Row-broadcast multiply: out[r,w] = a[r,w] * g[r] (the shared-expert gate).
// FP32 math with one fp16 rounding -- identical arithmetic to the
// BroadcastMulLayer host loop it replaces.
__global__ void bcast_mul_fp16(const unsigned short *a,
                               const unsigned short *g, unsigned short *out,
                               int n, int W) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = ew_f2h(ew_h2f(a[i]) * ew_h2f(g[i / W]));
}
// Same-shape eltwise multiply (the attention output gate's `multiply` node).
// An fp16 x fp16 product is exact in fp32, so one rn round here is
// bit-identical to the host Tensor::multiply loop.
__global__ void mul_fp16(const unsigned short *a, const unsigned short *b,
                         unsigned short *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = ew_f2h(ew_h2f(a[i]) * ew_h2f(b[i]));
}
// Plain in-place sigmoid, the whole-tensor activation. Same fp32 math as
// sigmoid_glu's gate half and as the host ActiFunc, so the two agree.
//
// This exists because CudaComputeOps::apply_activation had NO device path at
// all -- it host_math_gate'd straight into CpuComputeOps, and CpuComputeOps
// runs ActiFunc::run_fn, i.e. Tensor::apply with a std::function indirect per
// element. On a 20,463-token prefill the model's two sigmoid `activation`
// nodes measured 8,199 ms at 39.0 ns/element. (Fixing exp_util's double
// promotion first moved that only 42.7 -> 39.0 ns, which is what proved the
// cost is the per-element dispatch and not the precision.)
__global__ void act_sigmoid_fp16(unsigned short *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] = ew_f2h(1.0f / (1.0f + expf(-ew_h2f(x[i]))));
}
__global__ void act_sigmoid_fp32(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] = 1.0f / (1.0f + expf(-x[i]));
}
__global__ void sigmoid_add_fp16(const unsigned short *gate,
                                 const unsigned short *emb, unsigned short *out,
                                 int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = 1.0f / (1.0f + expf(-ew_h2f(gate[i])));
  out[i] = ew_f2h(g + ew_h2f(emb[i]));
}
__global__ void add_fp16(const unsigned short *a, const unsigned short *b,
                         unsigned short *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = ew_f2h(ew_h2f(a[i]) + ew_h2f(b[i]));
}
__global__ void scalar_mul_fp16(const unsigned short *in, unsigned short *out,
                                int n, float scalar) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = ew_f2h(ew_h2f(in[i]) * scalar);
}
// M2-B V-copy: write into the KV cache at the live slot d_pos[0] computed
// on-device (out_base is the cache BASE, width = per-row element count), so a
// captured graph writes V to the correct (new-token) slot on every replay.
// [kv-window-ring] ring_cap > 0 maps each ABSOLUTE row (d_pos[0] + i/width) to
// its physical ring row (% ring_cap) -- the sliding-layer cache only has
// ring_cap physical rows, so the absolute row was an OOB write under the ring
// (the 2026-07-24 M2B x ring garbage). Per-row mapping (not just the base) so
// multi-row prefill-chunk writes stay safe without relying on the Wcap
// seam-alignment invariant.
__global__ void scalar_mul_fp16_slot(const unsigned short *in,
                                     unsigned short *out_base, int n, float scalar,
                                     const int *d_pos, int width, int ring_cap) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long row_abs = (long)d_pos[0] + i / width;
  long row = (ring_cap > 0) ? (row_abs % ring_cap) : row_abs;
  out_base[row * width + (i % width)] = ew_f2h(ew_h2f(in[i]) * scalar);
}
__global__ void slice_copy_fp16(const unsigned short *in, unsigned short *out,
                                int rows, int in_width, int layer_off, int fs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * fs) return;
  int r = idx / fs, f = idx % fs;
  out[(size_t)r * fs + f] = in[(size_t)r * in_width + layer_off + f];
}
__global__ void softcap_fp16(const unsigned short *in, unsigned short *out,
                             int n, float cap) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = ew_f2h(cap * tanhf(ew_h2f(in[i]) / cap));
}
// ---- uint4 (8-half) vectorized variants of the volume kernels ----
// Selection is host-side (n%8==0 and 16B-aligned pointers, else the scalar
// kernel runs). Same per-element fp32 math and single RNE round; element
// order never enters these ops, so the vector forms are bit-identical.
__device__ __forceinline__ void ew_ld8(const uint4 *p, int i, float *f) {
  uint4 v = p[i];
  f[0] = ew_h2f((unsigned short)(v.x & 0xFFFFu));
  f[1] = ew_h2f((unsigned short)(v.x >> 16));
  f[2] = ew_h2f((unsigned short)(v.y & 0xFFFFu));
  f[3] = ew_h2f((unsigned short)(v.y >> 16));
  f[4] = ew_h2f((unsigned short)(v.z & 0xFFFFu));
  f[5] = ew_h2f((unsigned short)(v.z >> 16));
  f[6] = ew_h2f((unsigned short)(v.w & 0xFFFFu));
  f[7] = ew_h2f((unsigned short)(v.w >> 16));
}
__device__ __forceinline__ uint4 ew_st8(const float *f) {
  uint4 o;
  o.x = (unsigned int)ew_f2h(f[0]) | ((unsigned int)ew_f2h(f[1]) << 16);
  o.y = (unsigned int)ew_f2h(f[2]) | ((unsigned int)ew_f2h(f[3]) << 16);
  o.z = (unsigned int)ew_f2h(f[4]) | ((unsigned int)ew_f2h(f[5]) << 16);
  o.w = (unsigned int)ew_f2h(f[6]) | ((unsigned int)ew_f2h(f[7]) << 16);
  return o;
}
__global__ void add_fp16_v8(const uint4 *a, const uint4 *b, uint4 *out,
                            int nv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nv) return;
  float fa[8], fb[8], fo[8];
  ew_ld8(a, i, fa);
  ew_ld8(b, i, fb);
  #pragma unroll
  for (int j = 0; j < 8; ++j) fo[j] = fa[j] + fb[j];
  out[i] = ew_st8(fo);
}
__global__ void mul_fp16_v8(const uint4 *a, const uint4 *b, uint4 *out,
                            int nv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nv) return;
  float fa[8], fb[8], fo[8];
  ew_ld8(a, i, fa);
  ew_ld8(b, i, fb);
  #pragma unroll
  for (int j = 0; j < 8; ++j) fo[j] = fa[j] * fb[j];
  out[i] = ew_st8(fo);
}
__global__ void act_sigmoid_fp16_v8(uint4 *x, int nv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nv) return;
  float f[8];
  ew_ld8(x, i, f);
  #pragma unroll
  for (int j = 0; j < 8; ++j) f[j] = 1.0f / (1.0f + expf(-f[j]));
  x[i] = ew_st8(f);
}
__global__ void swiglu_fp16_v8(const uint4 *gate, const uint4 *up, uint4 *out,
                               int nv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nv) return;
  float fg[8], fu[8], fo[8];
  ew_ld8(gate, i, fg);
  ew_ld8(up, i, fu);
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    float s = fg[j] / (1.0f + expf(-fg[j]));
    fo[j] = s * fu[j];
  }
  out[i] = ew_st8(fo);
}
// Row-per-block broadcast multiply: one block per row removes the per-element
// i/W integer division entirely, and the row gate g[r] is one scalar load.
__global__ void bcast_mul_fp16_v8(const uint4 *a, const unsigned short *g,
                                  uint4 *out, int rows, int Wv) {
  int r = blockIdx.x;
  if (r >= rows) return;
  float gv = ew_h2f(g[r]);
  const uint4 *ar = a + (long)r * Wv;
  uint4 *orow = out + (long)r * Wv;
  for (int i = threadIdx.x; i < Wv; i += blockDim.x) {
    float f[8];
    ew_ld8(ar, i, f);
    #pragma unroll
    for (int j = 0; j < 8; ++j) f[j] *= gv;
    orow[i] = ew_st8(f);
  }
}
}
)CU";

// Two-pass on-GPU greedy argmax over the vocab logits. Pass 1: each of GRIDDIM
// blocks reduces a grid-strided slice of [N] to one (max, idx) pair, written to
// the per-block scratch (pmax[b], pidx[b]). Pass 2: a single block reduces the
// GRIDDIM partials to the final (max, idx) and writes the 4-byte index to
// oidx[0]. Ties resolve to the LOWEST index (matches std::max_element, which
// keeps the first of equal maxima). fp32 and fp16 variants (fp16 decoded inline
// with the same half->float as the other elementwise kernels).
static const char *ARGMAX_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float am_h2f(unsigned short h) {
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
// Block-reduce shared (val,idx), tie -> lowest idx. blockDim.x must be 256.
__device__ __forceinline__ void am_block_reduce(float *sv, int *si) {
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      int j = threadIdx.x + s;
      if (sv[j] > sv[threadIdx.x] ||
          (sv[j] == sv[threadIdx.x] && si[j] < si[threadIdx.x])) {
        sv[threadIdx.x] = sv[j];
        si[threadIdx.x] = si[j];
      }
    }
    __syncthreads();
  }
}
__global__ void argmax_p1_f32(const float *logits, int n, float *pmax,
                              int *pidx) {
  __shared__ float sv[256];
  __shared__ int si[256];
  float bv = -3.402823466e+38f; // -FLT_MAX
  int bi = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    float v = logits[i];
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  am_block_reduce(sv, si);
  if (threadIdx.x == 0) { pmax[blockIdx.x] = sv[0]; pidx[blockIdx.x] = si[0]; }
}
__global__ void argmax_p1_f16(const unsigned short *logits, int n, float *pmax,
                              int *pidx) {
  __shared__ float sv[256];
  __shared__ int si[256];
  float bv = -3.402823466e+38f;
  int bi = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    float v = am_h2f(logits[i]);
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  am_block_reduce(sv, si);
  if (threadIdx.x == 0) { pmax[blockIdx.x] = sv[0]; pidx[blockIdx.x] = si[0]; }
}
__global__ void argmax_p2(const float *pmax, const int *pidx, int g,
                          unsigned int *oidx) {
  __shared__ float sv[256];
  __shared__ int si[256];
  float bv = -3.402823466e+38f;
  int bi = 0;
  for (int i = threadIdx.x; i < g; i += blockDim.x) {
    float v = pmax[i];
    int idx = pidx[i];
    if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
  }
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  am_block_reduce(sv, si);
  if (threadIdx.x == 0) oidx[0] = (unsigned int)si[0];
}
}
)CU";

namespace {
constexpr int ARGMAX_GRID = 256;   // pass-1 blocks (== pass-2 reduction width)
float *g_am_pmax = nullptr;        // [ARGMAX_GRID] per-block partial max
int *g_am_pidx = nullptr;          // [ARGMAX_GRID] per-block partial idx
unsigned int *g_am_oidx = nullptr; // [1] device final index
unsigned int *g_am_oidx_host =
  nullptr; // pinned host staging for the 4-byte D2H

// One-time allocation of the small fixed-size argmax scratch (partials + the
// 1-int device/host result). Capture-safe: a cudaMalloc inside stream capture
// invalidates the graph, so bail under capture (the buffers are tiny and are
// allocated on the first non-captured call -- the gating env makes this
// opt-in).
bool ensure_argmax_scratch() {
  if (g_am_pmax && g_am_pidx && g_am_oidx && g_am_oidx_host)
    return true;
  if (StreamManager::Global().isCapturing())
    return false;
  if (!g_am_pmax &&
      cudaMalloc(&g_am_pmax, sizeof(float) * ARGMAX_GRID) != cudaSuccess)
    return false;
  if (!g_am_pidx &&
      cudaMalloc(&g_am_pidx, sizeof(int) * ARGMAX_GRID) != cudaSuccess)
    return false;
  if (!g_am_oidx && cudaMalloc(&g_am_oidx, sizeof(unsigned int)) != cudaSuccess)
    return false;
  if (!g_am_oidx_host && cudaHostAlloc(&g_am_oidx_host, sizeof(unsigned int),
                                       cudaHostAllocDefault) != cudaSuccess)
    return false;
  return true;
}

// Run the two-pass reduction over a device-resident logits pointer (fp32 or
// fp16) and copy the 4-byte token id back to the host. Shared by both dtypes.
bool argmax_dispatch(const void *logits_dev, bool is_fp16, unsigned int vocab,
                     unsigned int *token_out_host) {
  if (logits_dev == nullptr || vocab == 0 || token_out_host == nullptr)
    return false;
  // Capture-safe scratch (no cudaMalloc under graph capture).
  if (!ensure_argmax_scratch())
    return false;

  auto kp1 = CudaContext::Global().registerCudaKernel(
    ARGMAX_SRC, is_fp16 ? "argmax_p1_f16" : "argmax_p1_f32");
  auto kp2 = CudaContext::Global().registerCudaKernel(ARGMAX_SRC, "argmax_p2");
  if (!kp1 || !kp2) {
    ml_loge("[CUDA] argmax: kernel registration failed");
    return false;
  }

  int n = (int)vocab, g = ARGMAX_GRID;
  kp1->SetKernelArguments(0, &logits_dev, sizeof(logits_dev));
  kp1->SetKernelArguments(1, &n, sizeof(n));
  kp1->SetKernelArguments(2, &g_am_pmax, sizeof(g_am_pmax));
  kp1->SetKernelArguments(3, &g_am_pidx, sizeof(g_am_pidx));
  const int b1[3] = {256, 1, 1};
  const int g1[3] = {ARGMAX_GRID, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kp1, g1, b1))
    return false;

  kp2->SetKernelArguments(0, &g_am_pmax, sizeof(g_am_pmax));
  kp2->SetKernelArguments(1, &g_am_pidx, sizeof(g_am_pidx));
  kp2->SetKernelArguments(2, &g, sizeof(g));
  kp2->SetKernelArguments(3, &g_am_oidx, sizeof(g_am_oidx));
  const int b2[3] = {256, 1, 1};
  const int g2[3] = {1, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kp2, g2, b2))
    return false;

  // Drain so the 4-byte D2H sees the final write, then copy the token id.
  StreamManager::Global().finish();
  if (cudaMemcpy(g_am_oidx_host, g_am_oidx, sizeof(unsigned int),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return false;
  *token_out_host = *g_am_oidx_host;
  return true;
}
} // namespace

bool cuda_argmax_fp32(const float *logits_dev, unsigned int vocab,
                      unsigned int *token_out_host) {
  return argmax_dispatch(logits_dev, /*is_fp16=*/false, vocab, token_out_host);
}

bool cuda_argmax_fp16(const unsigned short *logits_dev, unsigned int vocab,
                      unsigned int *token_out_host) {
  return argmax_dispatch(logits_dev, /*is_fp16=*/true, vocab, token_out_host);
}

template <typename K> static bool dispatch1d(K &kernel, unsigned int n) {
  const int block[3] = {256, 1, 1};
  const int grid[3] = {(int)((n + 255) / 256), 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

// The uint4 (8-half) kernels need n divisible by 8 and 16B-aligned row bases;
// anything else stays on the scalar kernel. nullptr entries are ignored.
static inline bool ew_v8_ok(unsigned int n, const void *a, const void *b,
                            const void *c) {
  if ((n & 7u) != 0u)
    return false;
  auto aligned = [](const void *p) {
    return p == nullptr || ((reinterpret_cast<uintptr_t>(p) & 15u) == 0u);
  };
  return aligned(a) && aligned(b) && aligned(c);
}

bool cuda_geglu_fp16(const unsigned short *gate, const unsigned short *up,
                     unsigned short *out, unsigned int n) {
  if (n == 0)
    return true;
  auto k = CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "geglu_fp16");
  if (!k) {
    ml_loge("[CUDA] geglu_fp16: registration failed");
    return false;
  }
  int ni = (int)n;
  k->SetKernelArguments(0, &gate, sizeof(gate));
  k->SetKernelArguments(1, &up, sizeof(up));
  k->SetKernelArguments(2, &out, sizeof(out));
  k->SetKernelArguments(3, &ni, sizeof(ni));
  return dispatch1d(k, n);
}

bool cuda_swiglu_fp16(const unsigned short *gate, const unsigned short *up,
                      unsigned short *out, unsigned int n) {
  if (n == 0)
    return true;
  const bool v8 = ew_v8_ok(n, gate, up, out);
  auto k = CudaContext::Global().registerCudaKernel(
    ELTWISE_SRC, v8 ? "swiglu_fp16_v8" : "swiglu_fp16");
  if (!k) {
    ml_loge("[CUDA] swiglu_fp16: registration failed");
    return false;
  }
  int ni = (int)(v8 ? n / 8 : n);
  k->SetKernelArguments(0, &gate, sizeof(gate));
  k->SetKernelArguments(1, &up, sizeof(up));
  k->SetKernelArguments(2, &out, sizeof(out));
  k->SetKernelArguments(3, &ni, sizeof(ni));
  return dispatch1d(k, (unsigned int)ni);
}

bool cuda_act_sigmoid_fp16(unsigned short *x, unsigned int n) {
  if (n == 0)
    return true;
  const bool v8 = ew_v8_ok(n, x, nullptr, nullptr);
  auto k = CudaContext::Global().registerCudaKernel(
    ELTWISE_SRC, v8 ? "act_sigmoid_fp16_v8" : "act_sigmoid_fp16");
  if (!k) {
    ml_loge("[CUDA] act_sigmoid_fp16: registration failed");
    return false;
  }
  int ni = (int)(v8 ? n / 8 : n);
  k->SetKernelArguments(0, &x, sizeof(x));
  k->SetKernelArguments(1, &ni, sizeof(ni));
  return dispatch1d(k, (unsigned int)ni);
}

bool cuda_act_sigmoid_fp32(float *x, unsigned int n) {
  if (n == 0)
    return true;
  auto k =
    CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "act_sigmoid_fp32");
  if (!k) {
    ml_loge("[CUDA] act_sigmoid_fp32: registration failed");
    return false;
  }
  int ni = (int)n;
  k->SetKernelArguments(0, &x, sizeof(x));
  k->SetKernelArguments(1, &ni, sizeof(ni));
  return dispatch1d(k, n);
}

bool cuda_sigmoid_glu_fp16(const unsigned short *gate, const unsigned short *x,
                           unsigned short *out, unsigned int n) {
  if (n == 0)
    return true;
  auto k =
    CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "sigmoid_glu_fp16");
  if (!k) {
    ml_loge("[CUDA] sigmoid_glu_fp16: registration failed");
    return false;
  }
  int ni = (int)n;
  k->SetKernelArguments(0, &gate, sizeof(gate));
  k->SetKernelArguments(1, &x, sizeof(x));
  k->SetKernelArguments(2, &out, sizeof(out));
  k->SetKernelArguments(3, &ni, sizeof(ni));
  return dispatch1d(k, n);
}

bool cuda_sigmoid_add_fp16(const unsigned short *gate,
                           const unsigned short *emb, unsigned short *out,
                           unsigned int n) {
  if (n == 0)
    return true;
  auto k =
    CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "sigmoid_add_fp16");
  if (!k) {
    ml_loge("[CUDA] sigmoid_add_fp16: registration failed");
    return false;
  }
  int ni = (int)n;
  k->SetKernelArguments(0, &gate, sizeof(gate));
  k->SetKernelArguments(1, &emb, sizeof(emb));
  k->SetKernelArguments(2, &out, sizeof(out));
  k->SetKernelArguments(3, &ni, sizeof(ni));
  return dispatch1d(k, n);
}

// The one-deep pending-add record (see the header). Single-slot on purpose:
// pending is created at cuda_add_fp16 and resolved by the very next backend
// entry, so two live records cannot exist. The dispatch path is single-
// threaded (one stream, one graph walker); no lock, same as the staged-quant
// record in cuda_fc_qint4.cpp.
namespace {
struct PendingAdd {
  const unsigned short *a = nullptr;
  const unsigned short *b = nullptr;
  unsigned short *out = nullptr;
  unsigned int n = 0;
  bool valid = false;
};
PendingAdd g_pend_add;

bool add_fuse_on() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_ADD_FUSE");
    return !(e != nullptr && e[0] == '0');
  }();
  return v;
}

// The immediate launch (also the flush body). Never touches g_pend_add.
bool add_launch_now(const unsigned short *a, const unsigned short *b,
                    unsigned short *out, unsigned int n) {
  const bool v8 = ew_v8_ok(n, a, b, out);
  auto k = CudaContext::Global().registerCudaKernel(
    ELTWISE_SRC, v8 ? "add_fp16_v8" : "add_fp16");
  if (!k) {
    ml_loge("[CUDA] add_fp16: registration failed");
    return false;
  }
  int ni = (int)(v8 ? n / 8 : n);
  k->SetKernelArguments(0, &a, sizeof(a));
  k->SetKernelArguments(1, &b, sizeof(b));
  k->SetKernelArguments(2, &out, sizeof(out));
  k->SetKernelArguments(3, &ni, sizeof(ni));
  return dispatch1d(k, (unsigned int)ni);
}
} // namespace

void cuda_add_flush_pending() {
  if (!g_pend_add.valid)
    return;
  // Clear BEFORE launching: the launch re-enters DispatchCommand, whose
  // entry hook calls back into this function.
  PendingAdd p = g_pend_add;
  g_pend_add.valid = false;
  if (!add_launch_now(p.a, p.b, p.out, p.n))
    ml_loge("[CUDA] pending add flush FAILED (out=%p n=%u)", (void *)p.out,
            p.n);
}

bool cuda_add_pending_take(const void *out, unsigned long long n,
                           const unsigned short **a,
                           const unsigned short **b) {
  if (!g_pend_add.valid || g_pend_add.out != out ||
      (unsigned long long)g_pend_add.n != n)
    return false;
  *a = g_pend_add.a;
  *b = g_pend_add.b;
  g_pend_add.valid = false;
  return true;
}

bool cuda_add_fp16(const unsigned short *a, const unsigned short *b,
                   unsigned short *out, unsigned int n) {
  if (n == 0)
    return true;
  cuda_add_flush_pending(); // never stack two records
  if (add_fuse_on() && ew_v8_ok(n, a, b, out)) {
    g_pend_add = {a, b, out, n, true};
    return true;
  }
  return add_launch_now(a, b, out, n);
}

bool cuda_mul_fp16(const unsigned short *a, const unsigned short *b,
                   unsigned short *out, unsigned int n) {
  if (n == 0)
    return true;
  const bool v8 = ew_v8_ok(n, a, b, out);
  auto k = CudaContext::Global().registerCudaKernel(
    ELTWISE_SRC, v8 ? "mul_fp16_v8" : "mul_fp16");
  if (!k) {
    ml_loge("[CUDA] mul_fp16: registration failed");
    return false;
  }
  int ni = (int)(v8 ? n / 8 : n);
  k->SetKernelArguments(0, &a, sizeof(a));
  k->SetKernelArguments(1, &b, sizeof(b));
  k->SetKernelArguments(2, &out, sizeof(out));
  k->SetKernelArguments(3, &ni, sizeof(ni));
  return dispatch1d(k, (unsigned int)ni);
}

bool cuda_bcast_mul_fp16(const unsigned short *a, const unsigned short *g,
                         unsigned short *out, unsigned int n, unsigned int W) {
  if (n == 0)
    return true;
  // Row-per-block vector form: needs whole rows of uint4 (W%8), row-aligned
  // bases, and n an exact rows*W. The gate g is read as scalar halves.
  const bool v8 =
    W != 0 && (W & 7u) == 0u && (n % W) == 0u && ew_v8_ok(n, a, out, nullptr);
  auto k = CudaContext::Global().registerCudaKernel(
    ELTWISE_SRC, v8 ? "bcast_mul_fp16_v8" : "bcast_mul_fp16");
  if (!k) {
    ml_loge("[CUDA] bcast_mul_fp16: registration failed");
    return false;
  }
  if (v8) {
    int rows = (int)(n / W), wv = (int)(W / 8);
    k->SetKernelArguments(0, &a, sizeof(a));
    k->SetKernelArguments(1, &g, sizeof(g));
    k->SetKernelArguments(2, &out, sizeof(out));
    k->SetKernelArguments(3, &rows, sizeof(rows));
    k->SetKernelArguments(4, &wv, sizeof(wv));
    const int block[3] = {256, 1, 1};
    const int grid[3] = {rows, 1, 1};
    if (!StreamManager::Global().DispatchCommand(*k, grid, block))
      return false;
    StreamManager::Global().maybeFinish();
    return true;
  }
  int ni = (int)n, wi = (int)W;
  k->SetKernelArguments(0, &a, sizeof(a));
  k->SetKernelArguments(1, &g, sizeof(g));
  k->SetKernelArguments(2, &out, sizeof(out));
  k->SetKernelArguments(3, &ni, sizeof(ni));
  k->SetKernelArguments(4, &wi, sizeof(wi));
  return dispatch1d(k, n);
}

bool cuda_scalar_mul_fp16(const unsigned short *in, unsigned short *out,
                          unsigned int n, float scalar) {
  if (n == 0)
    return true;
  auto k =
    CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "scalar_mul_fp16");
  if (!k) {
    ml_loge("[CUDA] scalar_mul_fp16: registration failed");
    return false;
  }
  int ni = (int)n;
  k->SetKernelArguments(0, &in, sizeof(in));
  k->SetKernelArguments(1, &out, sizeof(out));
  k->SetKernelArguments(2, &ni, sizeof(ni));
  k->SetKernelArguments(3, &scalar, sizeof(scalar));
  if (!dispatch1d(k, n))
    return false;
  quant_stage_survive(out); // writes only `out` (the mha V-cache copy path)
  return true;
}

bool cuda_scalar_mul_fp16_slot(const unsigned short *in,
                               unsigned short *out_base, unsigned int n,
                               float scalar, int width, int ring_cap) {
  if (n == 0)
    return true;
  auto k = CudaContext::Global().registerCudaKernel(ELTWISE_SRC,
                                                    "scalar_mul_fp16_slot");
  if (!k) {
    ml_loge("[CUDA] scalar_mul_fp16_slot: registration failed");
    return false;
  }
  int ni = (int)n;
  const int *d_pos = cuda_pos_buffer();
  k->SetKernelArguments(0, &in, sizeof(in));
  k->SetKernelArguments(1, &out_base, sizeof(out_base));
  k->SetKernelArguments(2, &ni, sizeof(ni));
  k->SetKernelArguments(3, &scalar, sizeof(scalar));
  k->SetKernelArguments(4, &d_pos, sizeof(d_pos));
  k->SetKernelArguments(5, &width, sizeof(width));
  k->SetKernelArguments(6, &ring_cap, sizeof(ring_cap));
  return dispatch1d(k, n);
}

bool cuda_softcap_fp16(const unsigned short *in, unsigned short *out,
                       unsigned int n, float cap) {
  if (n == 0)
    return true;
  auto k =
    CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "softcap_fp16");
  if (!k) {
    ml_loge("[CUDA] softcap_fp16: registration failed");
    return false;
  }
  int ni = (int)n;
  k->SetKernelArguments(0, &in, sizeof(in));
  k->SetKernelArguments(1, &out, sizeof(out));
  k->SetKernelArguments(2, &ni, sizeof(ni));
  k->SetKernelArguments(3, &cap, sizeof(cap));
  return dispatch1d(k, n);
}

bool cuda_slice_copy_fp16(const unsigned short *in, unsigned short *out,
                          unsigned int rows, unsigned int in_width,
                          unsigned int layer_off, unsigned int fs) {
  if (rows == 0 || fs == 0)
    return true;
  auto k =
    CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "slice_copy_fp16");
  if (!k) {
    ml_loge("[CUDA] slice_copy_fp16: registration failed");
    return false;
  }
  int ri = (int)rows, iw = (int)in_width, lo = (int)layer_off, fsi = (int)fs;
  k->SetKernelArguments(0, &in, sizeof(in));
  k->SetKernelArguments(1, &out, sizeof(out));
  k->SetKernelArguments(2, &ri, sizeof(ri));
  k->SetKernelArguments(3, &iw, sizeof(iw));
  k->SetKernelArguments(4, &lo, sizeof(lo));
  k->SetKernelArguments(5, &fsi, sizeof(fsi));
  return dispatch1d(k, rows * fs);
}

} // namespace nntrainer::cuda

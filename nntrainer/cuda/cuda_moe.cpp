// SPDX-License-Identifier: Apache-2.0
/**
 * @file    cuda_moe.cpp
 * @brief   Device-side gather / SwiGLU / weighted scatter for MoE (see header).
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

// Hardware fp16 conversion, as cuda_fc_qint4.cpp:253 / cuda_rmsnorm.cpp:132.
// Not the software bit-twiddling variants: those contain a data-dependent
// `while` for denormals, which stops the compiler unrolling any loop calling
// them -- the defect that made the GDN GEMVs 68% of GPU time.
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

// out = silu(gate) * up, fp32 math, fp16 storage. Matches the host path's
// acti_func.run_fn(gate) followed by multiply_i(up).
__global__ void moe_swiglu_h(const unsigned short *gate,
                             const unsigned short *up,
                             unsigned short *out, int n){
  const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = moe_h2f(gate[i]);
  out[i] = moe_f2h((g / (1.0f + expf(-g))) * moe_h2f(up[i]));
}

// dst[rows[i],:] += wts[i]*src[i,:]
__global__ void moe_scatter_add_h(const unsigned short *src,
                                  unsigned short *dst, const int *rows,
                                  const float *wts, int m, int width){
  const long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= (long)m*width) return;
  const int i = (int)(idx / width), w = (int)(idx - (long)i*width);
  const long d = (long)rows[i]*width + w;
  dst[d] = moe_f2h(moe_h2f(dst[d]) + moe_h2f(src[idx]) * wts[i]);
}
} // extern "C"
)CU";

namespace {
std::mutex g_moe_mtx;
// Host-mapped staging for the per-expert row list and routing weights.
int *g_rows = nullptr;
float *g_wts = nullptr;
size_t g_rows_cap = 0, g_wts_cap = 0;

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
} // namespace

bool cuda_moe_stage(unsigned int m, int **rows_out, float **wts_out) {
  std::lock_guard<std::mutex> lk(g_moe_mtx);
  if (!grow_mapped((void **)&g_rows, &g_rows_cap, (size_t)m * sizeof(int)) ||
      !grow_mapped((void **)&g_wts, &g_wts_cap, (size_t)m * sizeof(float)))
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

} // namespace nntrainer::cuda

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_rope.cpp
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device RoPE op (NVRTC kernel) -- split-half, matches host math.
 */

#include "cuda_rope.h"

#include <cuda_context.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

#include <cuda_runtime.h>

#include <mutex>
#include <unordered_map>

namespace nntrainer::cuda {

// One block per head; threads sweep the rotated-pair index k in [0, half).
// Reuses the fp16<->fp32 codec from cuda_rmsnorm.
static const char *ROPE_FP16_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float rp_h2f(unsigned short h) {
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
__device__ __forceinline__ unsigned short rp_f2h(float f) {
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
__global__ void rope_fp16(const unsigned short *in, unsigned short *out,
                          const unsigned short *cosr, const unsigned short *sinr,
                          int head_dim, int half) {
  int head = blockIdx.x;
  const unsigned short *xr = in + (size_t)head * head_dim;
  unsigned short *yr = out + (size_t)head * head_dim;
  for (int k = threadIdx.x; k < half; k += blockDim.x) {
    float a = rp_h2f(xr[k]);
    float b = rp_h2f(xr[k + half]);
    float c = rp_h2f(cosr[k]);
    float s = rp_h2f(sinr[k]);
    yr[k]        = rp_f2h(a * c - b * s);
    yr[k + half] = rp_f2h(a * s + b * c);
  }
}
}
)CU";

namespace {
// Mirror a host LUT row to the device (keyed by host ptr; rows are tiny and
// immutable per position, so the cached copy can be reused for the same row).
struct DevRow {
  unsigned short *buf = nullptr;
  size_t cap = 0;
};
std::unordered_map<const void *, DevRow> g_lut_mirror;
std::mutex g_lut_mtx;

const unsigned short *mirror_row(const unsigned short *host, size_t elems) {
  cudaPointerAttributes a{};
  bool dev = cudaPointerGetAttributes(&a, host) == cudaSuccess &&
             (a.type == cudaMemoryTypeManaged || a.type == cudaMemoryTypeDevice);
  cudaGetLastError();
  if (dev)
    return host;
  std::lock_guard<std::mutex> lk(g_lut_mtx);
  auto &e = g_lut_mirror[host];
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

bool cuda_rope_fp16(const unsigned short *in, unsigned short *out,
                    const unsigned short *cos_row, const unsigned short *sin_row,
                    int num_heads, int head_dim) {
  if (num_heads == 0 || head_dim == 0)
    return true;
  const int half = head_dim / 2;
  cos_row = mirror_row(cos_row, half);
  sin_row = mirror_row(sin_row, half);
  if (!cos_row || !sin_row)
    return false;

  auto kernel =
    CudaContext::Global().registerCudaKernel(ROPE_FP16_SRC, "rope_fp16");
  if (!kernel) {
    ml_loge("[CUDA] rope_fp16: kernel registration failed");
    return false;
  }
  kernel->SetKernelArguments(0, &in, sizeof(in));
  kernel->SetKernelArguments(1, &out, sizeof(out));
  kernel->SetKernelArguments(2, &cos_row, sizeof(cos_row));
  kernel->SetKernelArguments(3, &sin_row, sizeof(sin_row));
  kernel->SetKernelArguments(4, &head_dim, sizeof(head_dim));
  kernel->SetKernelArguments(5, &half, sizeof(half));
  const int block[3] = {half < 256 ? half : 256, 1, 1};
  const int grid[3] = {num_heads, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().finish();
  return true;
}

} // namespace nntrainer::cuda

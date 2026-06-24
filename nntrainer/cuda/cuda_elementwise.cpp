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

#include <cuda_context.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

namespace nntrainer::cuda {

static const char *ELTWISE_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float ew_h2f(unsigned short h) {
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
__device__ __forceinline__ unsigned short ew_f2h(float f) {
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
__global__ void scalar_mul_fp16_slot(const unsigned short *in,
                                     unsigned short *out_base, int n, float scalar,
                                     const int *d_pos, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out_base[(long)d_pos[0] * width + i] = ew_f2h(ew_h2f(in[i]) * scalar);
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
}
)CU";

template <typename K> static bool dispatch1d(K &kernel, unsigned int n) {
  const int block[3] = {256, 1, 1};
  const int grid[3] = {(int)((n + 255) / 256), 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
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
  auto k = CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "swiglu_fp16");
  if (!k) {
    ml_loge("[CUDA] swiglu_fp16: registration failed");
    return false;
  }
  int ni = (int)n;
  k->SetKernelArguments(0, &gate, sizeof(gate));
  k->SetKernelArguments(1, &up, sizeof(up));
  k->SetKernelArguments(2, &out, sizeof(out));
  k->SetKernelArguments(3, &ni, sizeof(ni));
  return dispatch1d(k, n);
}

bool cuda_add_fp16(const unsigned short *a, const unsigned short *b,
                   unsigned short *out, unsigned int n) {
  if (n == 0)
    return true;
  auto k = CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "add_fp16");
  if (!k) {
    ml_loge("[CUDA] add_fp16: registration failed");
    return false;
  }
  int ni = (int)n;
  k->SetKernelArguments(0, &a, sizeof(a));
  k->SetKernelArguments(1, &b, sizeof(b));
  k->SetKernelArguments(2, &out, sizeof(out));
  k->SetKernelArguments(3, &ni, sizeof(ni));
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
  return dispatch1d(k, n);
}

bool cuda_scalar_mul_fp16_slot(const unsigned short *in,
                               unsigned short *out_base, unsigned int n,
                               float scalar, int width) {
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

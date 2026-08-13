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

namespace nntrainer::cuda {

// One block per (row, head); threads sweep the rotated-pair index k in
// [0,half). Per-row position is from + blockIdx.x, indexing the flat device
// cos/sin LUTs.
static const char *ROPE_FP16_SRC = R"CU(
extern "C" {
// Hardware cvt (bit-identical to the former software pair on every finite
// and non-finite value -- same exhaustive verification as rms_h2f_hw /
// dp4a_h2f). This was the one kernel source the scalar hw-cvt sweep missed:
// the old software rp_f2h carried a data-dependent denormal `while` loop,
// the exact defect the MoE header documents as blocking loop unrolling.
__device__ __forceinline__ float rp_h2f(unsigned short h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short rp_f2h(float f) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
__global__ void rope_fp16(const unsigned short *in, unsigned short *out,
                          const unsigned short *cos_lut,
                          const unsigned short *sin_lut, int num_heads,
                          int head_dim, int half, int from) {
  int row = blockIdx.x, head = blockIdx.y;
  long HD = (long)num_heads * head_dim;
  const unsigned short *xr = in + (long)row * HD + (long)head * head_dim;
  unsigned short *yr = out + (long)row * HD + (long)head * head_dim;
  const unsigned short *cosr = cos_lut + (long)(from + row) * half;
  const unsigned short *sinr = sin_lut + (long)(from + row) * half;
  for (int k = threadIdx.x; k < half; k += blockDim.x) {
    float a = rp_h2f(xr[k]);
    float b = rp_h2f(xr[k + half]);
    float c = rp_h2f(cosr[k]);
    float s = rp_h2f(sinr[k]);
    yr[k]        = rp_f2h(a * c - b * s);
    yr[k + half] = rp_f2h(a * s + b * c);
  }
}
// M2-B device-pos variant: reads the RoPE position `from` from d_pos[0] (so the
// captured graph never bakes a stale int), and -- when out_slot_dpos != 0 --
// writes each input row to OUTPUT row (from+row) so the K-into-cache write lands
// at the live cache slot computed on-device (out is then the cache BASE pointer,
// not a host pre-offset slice). out_slot_dpos==0 keeps the row-relative output
// (Q, in-place). Math identical to rope_fp16.
// [kv-window-ring] ring_cap > 0 maps the slot-mode OUTPUT row to the physical
// ring row ((from+row) % ring_cap) -- the sliding-layer cache only has ring_cap
// physical rows, so the absolute row was an OOB write under the ring (the
// 2026-07-24 M2B x ring garbage). The RoPE LUT index stays ABSOLUTE (from+row):
// only physical storage wraps, positions do not.
__global__ void rope_fp16_dpos(const unsigned short *in, unsigned short *out,
                               const unsigned short *cos_lut,
                               const unsigned short *sin_lut, int num_heads,
                               int head_dim, int half, const int *d_pos,
                               int out_slot_dpos, int ring_cap) {
  int from = d_pos[0];
  int row = blockIdx.x, head = blockIdx.y;
  long HD = (long)num_heads * head_dim;
  long out_row;
  if (out_slot_dpos) {
    long abs_row = (long)from + row;
    out_row = (ring_cap > 0) ? (abs_row % ring_cap) : abs_row;
  } else {
    out_row = (long)row;
  }
  const unsigned short *xr = in + (long)row * HD + (long)head * head_dim;
  unsigned short *yr = out + out_row * HD + (long)head * head_dim;
  const unsigned short *cosr = cos_lut + (long)(from + row) * half;
  const unsigned short *sinr = sin_lut + (long)(from + row) * half;
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

bool cuda_rope_fp16(const unsigned short *in, unsigned short *out,
                    const unsigned short *cos_lut,
                    const unsigned short *sin_lut, int num_heads, int head_dim,
                    int num_rows, int from) {
  if (num_heads == 0 || head_dim == 0 || num_rows == 0)
    return true;
  const int half = head_dim / 2;
  auto kernel =
    CudaContext::Global().registerCudaKernel(ROPE_FP16_SRC, "rope_fp16");
  if (!kernel) {
    ml_loge("[CUDA] rope_fp16: kernel registration failed");
    return false;
  }
  kernel->SetKernelArguments(0, &in, sizeof(in));
  kernel->SetKernelArguments(1, &out, sizeof(out));
  kernel->SetKernelArguments(2, &cos_lut, sizeof(cos_lut));
  kernel->SetKernelArguments(3, &sin_lut, sizeof(sin_lut));
  kernel->SetKernelArguments(4, &num_heads, sizeof(num_heads));
  kernel->SetKernelArguments(5, &head_dim, sizeof(head_dim));
  kernel->SetKernelArguments(6, &half, sizeof(half));
  kernel->SetKernelArguments(7, &from, sizeof(from));
  const int block[3] = {half < 256 ? half : 256, 1, 1};
  const int grid[3] = {num_rows, num_heads, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

bool cuda_rope_fp16_dpos(const unsigned short *in, unsigned short *out,
                         const unsigned short *cos_lut,
                         const unsigned short *sin_lut, int num_heads,
                         int head_dim, int num_rows, int out_slot_dpos,
                         int ring_cap) {
  if (num_heads == 0 || head_dim == 0 || num_rows == 0)
    return true;
  const int half = head_dim / 2;
  auto kernel =
    CudaContext::Global().registerCudaKernel(ROPE_FP16_SRC, "rope_fp16_dpos");
  if (!kernel) {
    ml_loge("[CUDA] rope_fp16_dpos: kernel registration failed");
    return false;
  }
  const int *d_pos = cuda_pos_buffer();
  kernel->SetKernelArguments(0, &in, sizeof(in));
  kernel->SetKernelArguments(1, &out, sizeof(out));
  kernel->SetKernelArguments(2, &cos_lut, sizeof(cos_lut));
  kernel->SetKernelArguments(3, &sin_lut, sizeof(sin_lut));
  kernel->SetKernelArguments(4, &num_heads, sizeof(num_heads));
  kernel->SetKernelArguments(5, &head_dim, sizeof(head_dim));
  kernel->SetKernelArguments(6, &half, sizeof(half));
  kernel->SetKernelArguments(7, &d_pos, sizeof(d_pos));
  kernel->SetKernelArguments(8, &out_slot_dpos, sizeof(out_slot_dpos));
  kernel->SetKernelArguments(9, &ring_cap, sizeof(ring_cap));
  const int block[3] = {half < 256 ? half : 256, 1, 1};
  const int grid[3] = {num_rows, num_heads, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

} // namespace nntrainer::cuda

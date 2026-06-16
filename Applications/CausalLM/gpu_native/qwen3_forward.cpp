// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen3_forward.cpp
 * @date   29 May 2026
 * @brief  Paper-aligned GPU-native Qwen3 forward (skeleton commit).
 */

#include "qwen3_forward.h"

#include <attention_kernels.h>
#include <blas_kernels.h>
#include <cl_context.h>
#include <cl_tensor_view.h>
#include <cpu_backend.h>
#include <engine.h>
#include <fp16.h>
#include <int4_utils.h>
#include <rmsnorm.h>
#include <rmsnorm_fp16.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace causallm_gpu {

namespace {
// Q6_K block layout: 256 elements per block, 210 bytes per block
// (see nntrainer/tensor/q6_k_tensor.h:32 — Q6_K_SIZE = 210).
constexpr size_t Q6_K_BLOCK_ELTS = 256;
constexpr size_t Q6_K_BLOCK_BYTES = 210;
} // namespace

// =============================================================================
// Inline OpenCL kernels for the GPU-native runtime. Strings are passed to
// ClContext::registerClKernel which caches by string identity, so each one
// only compiles once per process. Kept at file scope so any dispatch method
// can reference them regardless of declaration order.
// =============================================================================

// fp16 -> fp32 element-wise convert. One WI per element; gws = (N).
static const std::string kConvertFp16ToFp32Kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void cvt_h2f(__global const half *in, __global float *out,
                      const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = (float)in[i];
}
)CL";

// fp32 element-wise add. out[i] = a[i] + b[i]. One WI per element.
static const std::string kAddFp32Kernel = R"CL(
__kernel void add_fp32(__global const float *a, __global const float *b,
                       __global float *out, const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = a[i] + b[i];
}
)CL";

// §3.8 V OHWI-reversed scatter: copy fp16 V from concat layout
// [M, hKV*d] to OHWI-reversed [hKV, d, max_S] at position offset.
// One WI per (t, h, x) destination element. Reads are strided in
// source (hKV*d apart per t), writes are scattered in dest — but
// the destination is what the sv_matmul_f16_ohwi kernel reads
// sequentially, so this write-side cost is amortized by attention.
// GWS = (M, hKV, d). For typical sizes 1024 * 8 * 128 = 1M WIs.
// K scatter to OHWI [hKV, S_max, d_h] layout. Same shape as the
// existing K SVM scatter but on GPU side and writing to cl_mem.
// Lets us skip the CPU sync-map dual-write when NNTR_OHWI_IMG=1.
// WI: (t, h, x). x is stride-1 in both src and dst → coalesced.
static const std::string kKScatterOhwiKernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void k_scatter_ohwi(__global const half *src,
                             __global half *dst,
                             const int M, const int hKV,
                             const int d, const int max_S,
                             const int position) {
  int t = get_global_id(0);
  int h = get_global_id(1);
  int x = get_global_id(2);
  if (t >= M || h >= hKV || x >= d) return;
  dst[(long)h * max_S * d + (long)(position + t) * d + x] =
    src[(long)t * hKV * d + (long)h * d + x];
}
)CL";

static const std::string kVScatterOhwiTKernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void v_scatter_ohwi_t(__global const half *src,
                               __global half *dst,
                               const int M, const int hKV,
                               const int d, const int max_S,
                               const int position) {
  int t = get_global_id(0);
  int h = get_global_id(1);
  int x = get_global_id(2);
  if (t >= M || h >= hKV || x >= d) return;
  dst[(long)h * d * max_S + (long)x * max_S + position + t] =
    src[(long)t * hKV * d + (long)h * d + x];
}
)CL";

// ML Drift reaudit #1 (decode): Q6_K GEMV for the tied lm_head. The CPU path
// (run_lm_head_and_argmax_cpu) streams the 484 MB Q6_K table through one core
// at 149-155 ms/token = 66% of decode. This kernel replicates
// dequantize_row_q6_K_impl exactly (same q1..q4 nibble/2-bit expansion, same
// d*sc*q products) and dots against the post-norm hidden row; only the fp32
// accumulation ORDER differs (per-WI partials + LDS tree), so logits drift in
// the lsb — verification gate is token-ID equality over the greedy sequence.
// One 64-WI workgroup per vocab row: iteration t+64*s maps to (block, n-half,
// l) so adjacent WIs touch adjacent ql/qh bytes (coalesced); hidden lives in
// __constant (H*4 = 9.2 KB, broadcast-cached).
static const std::string kQ6kGemvKernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void q6k_gemv_lmhead(__global const uchar *W,   // [V rows][H/256 blocks][210 B]
                              __constant float *x,        // [H] post-norm hidden
                              __global float *logits,     // [V]
                              const int V, const int H) {
  // 64 WIs cover 4 Q6_K blocks per iteration; each WI owns one (block,
  // n-half, l-quad) unit and expands 16 elements via uchar4 loads (vs the
  // scalar per-byte variant that was ALU/issue-bound at ~10 GB/s).
  // One 64-WI workgroup per vocab row; each WI owns one (block-lane, n-half,
  // l-quad) unit with direct per-field uchar4 loads. Two LDS-staging
  // variants (1-row coalesced 16 B chunks, 4-rows + 16 WIs/row) both
  // measured SLOWER on Adreno 840 (24.4 -> 26.4 / 28.5 ms) — the limiter is
  // not intra-row coalescing or barrier count, so the simplest mapping
  // stays. Host guards V % 4 == 0 is not needed here (1 row/WG).
  const int row = get_group_id(0);
  const int t = get_local_id(0); // 0..63
  const int nb = H >> 8;         // Q6_K blocks per row
  __global const uchar *rb = W + (size_t)row * (size_t)(nb * 210);
  const int bl = t >> 4;         // block lane 0..3
  const int u = t & 15;          // unit within block
  const int nh = u >> 3;         // n-half: 0 -> elems [0,128), 1 -> [128,256)
  const int q = u & 7;           // l-quad: l = 4q .. 4q+3
  float sum = 0.0f;
  for (int s = 0; s < nb; s += 4) {
    const int bi = s + bl;
    if (bi < nb) {
      __global const uchar *blk = rb + bi * 210;
      const uchar4 qlo = vload4(0, blk + (nh << 6) + (q << 2));
      const uchar4 qhi = vload4(0, blk + (nh << 6) + 32 + (q << 2));
      const uchar4 qh4 = vload4(0, blk + 128 + (nh << 5) + (q << 2));
      const float d = vload_half(0, (__global const half *)(blk + 208));
      const int is = q >> 2;     // (4q)/16 == (4q+3)/16 for q in 0..7
      const int sbase = 192 + (nh << 3);
      const float s0 = d * (float)((__global const char *)blk)[sbase + is];
      const float s2 = d * (float)((__global const char *)blk)[sbase + is + 2];
      const float s4 = d * (float)((__global const char *)blk)[sbase + is + 4];
      const float s6 = d * (float)((__global const char *)blk)[sbase + is + 6];
    const int yb = (bi << 8) + (nh << 7) + (q << 2);
    float4 a1, a2, a3, a4;
    a1.x = (float)((int)((qlo.x & 0xF) | (((qh4.x >> 0) & 3) << 4)) - 32);
    a1.y = (float)((int)((qlo.y & 0xF) | (((qh4.y >> 0) & 3) << 4)) - 32);
    a1.z = (float)((int)((qlo.z & 0xF) | (((qh4.z >> 0) & 3) << 4)) - 32);
    a1.w = (float)((int)((qlo.w & 0xF) | (((qh4.w >> 0) & 3) << 4)) - 32);
    a2.x = (float)((int)((qhi.x & 0xF) | (((qh4.x >> 2) & 3) << 4)) - 32);
    a2.y = (float)((int)((qhi.y & 0xF) | (((qh4.y >> 2) & 3) << 4)) - 32);
    a2.z = (float)((int)((qhi.z & 0xF) | (((qh4.z >> 2) & 3) << 4)) - 32);
    a2.w = (float)((int)((qhi.w & 0xF) | (((qh4.w >> 2) & 3) << 4)) - 32);
    a3.x = (float)((int)((qlo.x >> 4) | (((qh4.x >> 4) & 3) << 4)) - 32);
    a3.y = (float)((int)((qlo.y >> 4) | (((qh4.y >> 4) & 3) << 4)) - 32);
    a3.z = (float)((int)((qlo.z >> 4) | (((qh4.z >> 4) & 3) << 4)) - 32);
    a3.w = (float)((int)((qlo.w >> 4) | (((qh4.w >> 4) & 3) << 4)) - 32);
    a4.x = (float)((int)((qhi.x >> 4) | (((qh4.x >> 6) & 3) << 4)) - 32);
    a4.y = (float)((int)((qhi.y >> 4) | (((qh4.y >> 6) & 3) << 4)) - 32);
    a4.z = (float)((int)((qhi.z >> 4) | (((qh4.z >> 6) & 3) << 4)) - 32);
    a4.w = (float)((int)((qhi.w >> 4) | (((qh4.w >> 6) & 3) << 4)) - 32);
    const float4 x1 = (float4)(x[yb], x[yb + 1], x[yb + 2], x[yb + 3]);
    const float4 x2 = (float4)(x[yb + 32], x[yb + 33], x[yb + 34], x[yb + 35]);
    const float4 x3 = (float4)(x[yb + 64], x[yb + 65], x[yb + 66], x[yb + 67]);
    const float4 x4 = (float4)(x[yb + 96], x[yb + 97], x[yb + 98], x[yb + 99]);
    sum += s0 * dot(a1, x1) + s2 * dot(a2, x2) + s4 * dot(a3, x3) +
           s6 * dot(a4, x4);
    }
  }
  __local float red[64];
  red[t] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = 32; off > 0; off >>= 1) {
    if (t < off) red[t] += red[t + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) logits[row] = red[0];
}

// Stage-1 argmax over the fp32 logits: 64 workgroups x 64 WIs scan strided
// slices and LDS-reduce to one (value, index) candidate per workgroup. The
// host reads 64 pairs (512 B instead of the 1 MB logits) and picks the
// global winner. Tie rule matches the CPU scan (strict >, ties keep the
// smaller index) so the selected token is identical.
__kernel void argmax_f32_stage1(__global const float *logits, const int V,
                                __global float *best_val,
                                __global int *best_idx) {
  const int g = get_group_id(0);
  const int t = get_local_id(0);
  const int gid = g * 64 + t; // 0..4095
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int v = gid; v < V; v += 4096) {
    const float x = logits[v];
    if (x > bv || (x == bv && v < bi)) {
      bv = x;
      bi = v;
    }
  }
  __local float lv[64];
  __local int li[64];
  lv[t] = bv;
  li[t] = bi;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = 32; off > 0; off >>= 1) {
    if (t < off) {
      const float ov = lv[t + off];
      const int oi = li[t + off];
      if (ov > lv[t] || (ov == lv[t] && oi < li[t])) {
        lv[t] = ov;
        li[t] = oi;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) {
    best_val[g] = lv[0];
    best_idx[g] = li[0];
  }
}
)CL";

// gemm_int8_v8c_cl grew a trailing M_valid store-guard parameter (direct-out
// mode in the layer-graph path; 0 = legacy store-every-row). The 8/4/4
// function-pointer ternaries below pair it with the still-11-arg
// gemm_int8_int8_v8c_cl, so adapt it back to the shared 11-arg signature.
static void gemm_int8_v8c_cl_legacy(cl_mem act, cl_mem wgt, cl_mem scale_act,
                                    cl_mem scale_wgt, cl_mem row_sum_act,
                                    cl_mem zp_act, cl_mem row_sum_w,
                                    cl_mem output_fp16, unsigned int M,
                                    unsigned int N, unsigned int K) {
  nntrainer::gemm_int8_v8c_cl(act, wgt, scale_act, scale_wgt, row_sum_act,
                              zp_act, row_sum_w, output_fp16, M, N, K,
                              /*M_valid=*/0);
}

// SwiGLU element-wise: out[i] = silu(gate[i]) * up[i]
//   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// fp32 throughout (matches the residual stream dtype). One WI per element.
// #46m: fp16 helpers for residual stream refactor.
static const std::string kAddFp16Kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void add_fp16(__global const half *a,
                       __global const half *b,
                       __global       half *out,
                       const int n) {
  int i = get_global_id(0);
  if (i >= n) return;
  out[i] = a[i] + b[i];
}
)CL";

static const std::string kCvtF2hKernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void cvt_f2h(__global const float *in, __global half *out,
                      const int n) {
  int i = get_global_id(0);
  if (i >= n) return;
  out[i] = (half)in[i];
}
)CL";

static const std::string kAddFp16ToFp32Kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void add_fp16_to_fp32(__global const half *a,
                               __global const half *b,
                               __global       float *out,
                               const int n) {
  int i = get_global_id(0);
  if (i >= n) return;
  out[i] = (float)a[i] + (float)b[i];
}
)CL";

static const std::string kFusedSwigluFp16Kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void fused_swiglu_fp16(__global const half *gate,
                                __global const half *up,
                                __global       half *out,
                                const int n) {
  int i = get_global_id(0);
  if (i >= n) return;
  float g = (float)gate[i];
  float u = (float)up[i];
  float s = g / (1.0f + native_exp(-g));
  out[i] = (half)(s * u);
}
)CL";

// #46m: SVM fp16 → cl_mem fp16 GPU copy. Used to bridge attention
// output (SVM) into quantize_act_v8c_fp16_cl input (cl_mem).
static const std::string kCopySvmFp16Kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void copy_svm_to_clmem_fp16(__global const half *src,
                                     __global       half *dst,
                                     const int n) {
  int i = get_global_id(0);
  if (i >= n) return;
  dst[i] = src[i];
}
)CL";

// Fused cvt h2f + add fp32 (#46j). out_fp32[i] = a_fp32[i] + (float)b_fp16[i].
// Replaces (cvt b_fp16 → b_fp32) + (add a_fp32 + b_fp32 → out_fp32) used
// at the end of wo and ffn-down. Saves 1 dispatch per occurrence × 2 ×
// 28 layers ≈ 50-100 ms at M=1024.
static const std::string kFusedAddH2fFp32Kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void fused_add_h2f_fp32(__global const float *a,
                                 __global const half *b,
                                 __global       float *out,
                                 const int n) {
  int i = get_global_id(0);
  if (i >= n) return;
  out[i] = a[i] + (float)b[i];
}
)CL";

// Fused cvt+swiglu (#46j). Replaces 3 dispatches (cvt up_fp16->up_fp32,
// cvt gate_fp16->gate_fp32, swiglu(gate,up)->out_fp32) with one. Saves
// ~2 dispatch overheads per layer × 28 layers ≈ 100-150 ms at M=1024.
//   out[i] = silu(gate[i]) * up[i],  silu(x) = x / (1 + exp(-x))
// Inputs are read as fp16 directly; output is fp32. 1 WI per element.
static const std::string kFusedSwigluH2fFp32Kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void fused_swiglu_h2f_fp32(__global const half *gate,
                                    __global const half *up,
                                    __global       float *out,
                                    const int n) {
  int i = get_global_id(0);
  if (i >= n) return;
  float g = (float)gate[i];
  float u = (float)up[i];
  float s = g / (1.0f + native_exp(-g));
  out[i] = s * u;
}
)CL";

// #63 Gemma2 GeGLU: out[i] = gelu_tanh(gate[i]) * up[i], where
//   gelu_tanh(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
// Same fp16-in / fp32-out contract as fused_swiglu_h2f_fp32.
static const std::string kFusedGegluH2fFp32Kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void fused_geglu_h2f_fp32(__global const half *gate,
                                   __global const half *up,
                                   __global       float *out,
                                   const int n) {
  int i = get_global_id(0);
  if (i >= n) return;
  float g = (float)gate[i];
  float u = (float)up[i];
  const float k = 0.7978845608028654f; // sqrt(2/pi)
  float t = tanh(k * (g + 0.044715f * g * g * g));
  out[i] = (0.5f * g * (1.0f + t)) * u;
}
)CL";

// #81 fuse GeGLU + int8 act-quant (FFN down-proj input). Replaces
// fused_geglu_h2f_fp32 (writes [M,I] fp32 swiglu_out) + v8c_act_quant_f32_par
// (reads it back TWICE) with one cooperative pass per row: compute
// geglu = gelu_tanh(gate)*up in fp32 registers, reduce per-row min/max, then
// RECOMPUTE geglu + quantize to int8 (recompute avoids a global RAW hazard and
// is cheaper than the saved [M,9216] fp32 round-trip — 157MB->81MB DRAM at
// M=1024). The quant math is byte-for-byte v8c_act_quant_f32_par applied to the
// geglu fp32 value, so dn_i8/dn_sc/dn_zp/dn_rs are bit-identical to the 2-kernel
// path. NOTE arg order mirrors the de-swapped geglu dispatch: 'gate' = up_fp16
// (=gate_proj output, the activated operand), 'up' = gate_fp16 (=up_proj).
static const std::string kFusedGegluQuantKernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define GGQ_LWS 64
__attribute__((reqd_work_group_size(GGQ_LWS, 1, 1)))
__kernel void fused_geglu_quant(__global const half *gate,
                                __global const half *up,
                                __global       char  *act_int8,
                                __global       float *scale_per_row,
                                __global       int   *zp_per_row,
                                __global       int   *row_sum_act,
                                const int M, const int K) {
  const int row = get_group_id(0);
  if (row >= M) return;
  const int tid = get_local_id(0);
  __local float lmin[GGQ_LWS];
  __local float lmax[GGQ_LWS];
  __local int   lsum[GGQ_LWS];
  __local float l_scale_q;
  __local int   l_zp;
  const float kc = 0.7978845608028654f; // sqrt(2/pi)

  // pass 1: per-WI partial min/max of geglu(gate,up)
  float pmin = 0.0f, pmax = 0.0f;
  for (int k = tid; k < K; k += GGQ_LWS) {
    float g = (float)gate[(long)row * K + k];
    float u = (float)up[(long)row * K + k];
    float t = tanh(kc * (g + 0.044715f * g * g * g));
    float v = (0.5f * g * (1.0f + t)) * u;
    pmin = fmin(pmin, v);
    pmax = fmax(pmax, v);
  }
  lmin[tid] = pmin;
  lmax[tid] = pmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = GGQ_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) {
      lmin[tid] = fmin(lmin[tid], lmin[tid + s]);
      lmax[tid] = fmax(lmax[tid], lmax[tid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    const float fmn = lmin[0], fmx = lmax[0];
    const float rmin = fmn < 0.0f ? fmn : 0.0f;
    const float rmax = fmx > 0.0f ? fmx : 0.0f;
    const float qmin = -128.0f, qmax = 127.0f;
    const float range = rmax - rmin;
    const float scale_q = range > 0.0f ? 255.0f / range : 1.0f;
    const float recip = range > 0.0f ? range / 255.0f : 1.0f;
    const float dmin = rmin * scale_q, dmax = rmax * scale_q;
    const float zp_lo = qmin - dmin, zp_hi = qmax - dmax;
    float zp_f = (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;
    if (zp_f < qmin) zp_f = qmin;
    if (zp_f > qmax) zp_f = qmax;
    l_scale_q = scale_q;
    l_zp = (int)rint(zp_f);
    scale_per_row[row] = recip;
    zp_per_row[row] = l_zp;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // pass 2: recompute geglu, quantize + partial row_sum
  const float scale_q = l_scale_q;
  const int zp = l_zp;
  int psum = 0;
  for (int k = tid; k < K; k += GGQ_LWS) {
    float g = (float)gate[(long)row * K + k];
    float u = (float)up[(long)row * K + k];
    float t = tanh(kc * (g + 0.044715f * g * g * g));
    float v = (0.5f * g * (1.0f + t)) * u;
    int q = (int)rint(v * scale_q) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    act_int8[(long)row * K + k] = (char)q;
    psum += q;
  }
  lsum[tid] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = GGQ_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) lsum[tid] += lsum[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) row_sum_act[row] = lsum[0];
}
)CL";

// #71 fuse residual-add + RMSNorm (MLC fuse_add_norm). One workgroup (LWS=64)
// per row: resid = a(fp32) + b(fp16) [written for the later residual add], then
// normed = resid * rsqrt(mean(resid^2)+eps) * gamma [fp16, gamma has Gemma2
// (1+w) baked]. Replaces the standalone add + pad-copy + rmsnorm (3 dispatches
// -> 1). Recompute a+b in pass 2 to avoid a global read-after-write hazard.
static const std::string kFusedAddRmsnormKernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void fused_add_rmsnorm(__global const float *a,
                                __global const half  *b,
                                __global       float *resid,
                                __global       half  *normed,
                                __global const half  *gamma,
                                const half eps, const int W) {
  const int row = get_group_id(0);
  const int tid = get_local_id(0);
  const int L = get_local_size(0);
  __local float lss[64];
  float psum = 0.0f;
  for (int k = tid; k < W; k += L) {
    float v = a[(long)row * W + k] + (float)b[(long)row * W + k];
    resid[(long)row * W + k] = v;
    psum += v * v;
  }
  lss[tid] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = L / 2; s > 0; s >>= 1) {
    if (tid < s) lss[tid] += lss[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float inv = rsqrt(lss[0] / (float)W + (float)eps);
  for (int k = tid; k < W; k += L) {
    float v = a[(long)row * W + k] + (float)b[(long)row * W + k];
    normed[(long)row * W + k] = (half)(v * inv * (float)gamma[k]);
  }
}
)CL";

// #80 rmsnorm + int8 act-quant fused into one cooperative pass. Folds
// rmsnorm_f32in_f16out_coop (float8 sum-of-squares -> normed=(half)(x*scale)*gamma)
// and v8c_act_quant_f16_par (row min/max -> recip-scale/zp -> int8 round + rowsum)
// so the fp16 normed buffer is never round-tripped to DRAM and one dispatch is
// removed. Bit-identical to the split path: the SS reduction matches
// rmsnorm_f32in_f16out_coop exactly (float8 dot), and the quant math operates on
// the SAME (half)-cast normed value the split path stored/reloaded. One WG(64)/row.
static const std::string kFusedNormQuantKernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define FNQ_LWS 64
__kernel void rmsnorm_f32in_quant_fused(
    __global const float *input,         // [M, W] fp32 residual
    __global const half  *gamma,         // [W] norm weight (fp16)
    __global       char  *act_int8,      // [M, W] int8 out
    __global       float *scale_per_row, // [M] recip-scale
    __global       int   *zp_per_row,    // [M]
    __global       int   *row_sum_act,   // [M]
    const half epsilon, const int M, const int W) {
  const int row = get_group_id(0);
  if (row >= M) return;
  const int tid = get_local_id(0);
  const long base = (long)row * (long)W;
  const int W8 = W >> 3;
  __global const float8 *in8 = (__global const float8 *)(input + base);

  // pass 1: sum of squares (float8 dot == rmsnorm_f32in_f16out_coop)
  float partial = 0.0f;
  for (int i = tid; i < W8; i += FNQ_LWS) {
    const float8 v = in8[i];
    partial += dot(v.lo, v.lo) + dot(v.hi, v.hi);
  }
  __local float lss[FNQ_LWS];
  lss[tid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = FNQ_LWS >> 1; s > 0; s >>= 1) {
    if (tid < s) lss[tid] += lss[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float nscale = rsqrt(lss[0] / (float)W + (float)epsilon);

  // pass 2: normed = (half)(x*nscale)*gamma; reduce row min/max
  __local float lmin[FNQ_LWS];
  __local float lmax[FNQ_LWS];
  __local float l_scale_q;
  __local int   l_zp;
  float pmin = 0.0f, pmax = 0.0f;
  for (int k = tid; k < W; k += FNQ_LWS) {
    half nk = (half)((float)input[base + k] * nscale) * gamma[k];
    float fv = (float)nk;
    pmin = fmin(pmin, fv);
    pmax = fmax(pmax, fv);
  }
  lmin[tid] = pmin;
  lmax[tid] = pmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = FNQ_LWS >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      lmin[tid] = fmin(lmin[tid], lmin[tid + s]);
      lmax[tid] = fmax(lmax[tid], lmax[tid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    const float fmn = lmin[0], fmx = lmax[0];
    const float rmin = fmn < 0.0f ? fmn : 0.0f;
    const float rmax = fmx > 0.0f ? fmx : 0.0f;
    const float qmin = -128.0f, qmax = 127.0f;
    const float range = rmax - rmin;
    const float scale_q = range > 0.0f ? 255.0f / range : 1.0f;
    const float recip = range > 0.0f ? range / 255.0f : 1.0f;
    const float dmin = rmin * scale_q, dmax = rmax * scale_q;
    const float zp_lo = qmin - dmin, zp_hi = qmax - dmax;
    float zp_f = (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;
    if (zp_f < qmin) zp_f = qmin;
    if (zp_f > qmax) zp_f = qmax;
    l_scale_q = scale_q;
    l_zp = (int)rint(zp_f);
    scale_per_row[row] = recip;
    zp_per_row[row] = l_zp;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float scale_q = l_scale_q;
  const int zp = l_zp;

  // pass 3: quantize (recompute normed) + row sum
  __local int lsum[FNQ_LWS];
  int psum = 0;
  for (int k = tid; k < W; k += FNQ_LWS) {
    half nk = (half)((float)input[base + k] * nscale) * gamma[k];
    int q = (int)rint((float)nk * scale_q) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    act_int8[base + k] = (char)q;
    psum += q;
  }
  lsum[tid] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = FNQ_LWS >> 1; s > 0; s >>= 1) {
    if (tid < s) lsum[tid] += lsum[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) row_sum_act[row] = lsum[0];
}
)CL";

// #80b fused residual-add + rmsnorm + int8 act-quant (FFN path). Extends #71
// fused_add_rmsnorm to also emit int8 + recip-scale/zp/rowsum, skipping the
// fp16 ffn_normed round-trip + the standalone quant dispatch. Bit-identical to
// (#71 -> quantize_act) for rows<M: same scalar SS reduction, same
// normed=(half)(v*inv*(float)gamma), same quant math on the (half)-normed.
static const std::string kFusedAddRmsnormQuantKernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define FARQ_LWS 64
__kernel void fused_add_rmsnorm_quant(
    __global const float *a,             // [M,W] in_padded fp32
    __global const half  *b,             // [M,W] wo_y fp16
    __global       float *resid,         // [M,W] residual_1 fp32 out
    __global const half  *gamma,         // [W]
    __global       char  *act_int8,      // [M,W] int8 out
    __global       float *scale_per_row, // [M]
    __global       int   *zp_per_row,    // [M]
    __global       int   *row_sum_act,   // [M]
    const half eps, const int M, const int W) {
  const int row = get_group_id(0);
  if (row >= M) return;
  const int tid = get_local_id(0);
  const long base = (long)row * (long)W;
  // pass 1: v = a + b ; write resid ; sum of squares (scalar, == #71)
  __local float lss[FARQ_LWS];
  float psum = 0.0f;
  for (int k = tid; k < W; k += FARQ_LWS) {
    float v = a[base + k] + (float)b[base + k];
    resid[base + k] = v;
    psum += v * v;
  }
  lss[tid] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = FARQ_LWS >> 1; s > 0; s >>= 1) {
    if (tid < s) lss[tid] += lss[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = rsqrt(lss[0] / (float)W + (float)eps);
  // pass 2: normed = (half)(v*inv*(float)gamma); reduce row min/max
  __local float lmin[FARQ_LWS];
  __local float lmax[FARQ_LWS];
  __local float l_scale_q;
  __local int   l_zp;
  float pmin = 0.0f, pmax = 0.0f;
  for (int k = tid; k < W; k += FARQ_LWS) {
    float v = a[base + k] + (float)b[base + k];
    half nk = (half)(v * inv * (float)gamma[k]);
    float fv = (float)nk;
    pmin = fmin(pmin, fv);
    pmax = fmax(pmax, fv);
  }
  lmin[tid] = pmin;
  lmax[tid] = pmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = FARQ_LWS >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      lmin[tid] = fmin(lmin[tid], lmin[tid + s]);
      lmax[tid] = fmax(lmax[tid], lmax[tid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    const float fmn = lmin[0], fmx = lmax[0];
    const float rmin = fmn < 0.0f ? fmn : 0.0f;
    const float rmax = fmx > 0.0f ? fmx : 0.0f;
    const float qmin = -128.0f, qmax = 127.0f;
    const float range = rmax - rmin;
    const float scale_q = range > 0.0f ? 255.0f / range : 1.0f;
    const float recip = range > 0.0f ? range / 255.0f : 1.0f;
    const float dmin = rmin * scale_q, dmax = rmax * scale_q;
    const float zp_lo = qmin - dmin, zp_hi = qmax - dmax;
    float zp_f = (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;
    if (zp_f < qmin) zp_f = qmin;
    if (zp_f > qmax) zp_f = qmax;
    l_scale_q = scale_q;
    l_zp = (int)rint(zp_f);
    scale_per_row[row] = recip;
    zp_per_row[row] = l_zp;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float scale_q = l_scale_q;
  const int zp = l_zp;
  // pass 3: quantize (recompute) + row sum
  __local int lsum[FARQ_LWS];
  int isum = 0;
  for (int k = tid; k < W; k += FARQ_LWS) {
    float v = a[base + k] + (float)b[base + k];
    half nk = (half)(v * inv * (float)gamma[k]);
    int q = (int)rint((float)nk * scale_q) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    act_int8[base + k] = (char)q;
    isum += q;
  }
  lsum[tid] = isum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = FARQ_LWS >> 1; s > 0; s >>= 1) {
    if (tid < s) lsum[tid] += lsum[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) row_sum_act[row] = lsum[0];
}
)CL";

static const std::string kSwigluFp32Kernel = R"CL(
__kernel void swiglu_fp32(__global const float *gate,
                          __global const float *up,
                          __global float *out, const int n) {
  int i = get_global_id(0);
  if (i < n) {
    float g = gate[i];
    float s = g / (1.0f + exp(-g));
    out[i] = s * up[i];
  }
}
)CL";

// fp16 RoPE: in-place rotation of [num_heads, head_dim] fp16 buffer.
// cos/sin tables are doubled-half (cos[k+half] = cos[k], sin[k+half] =
// sin[k]) to match the CPU mha_core convention. GWS = (num_heads,
// head_dim/2); one WI per rotation pair (writes positions k and k+half).
static const std::string kRopeFp16Kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void rope_fp16(__global half *xy,
                        __global const half *cos_tbl,
                        __global const half *sin_tbl,
                        const int num_heads,
                        const int half_d) {
  int h = get_global_id(0);
  int k = get_global_id(1);
  if (h >= num_heads || k >= half_d) return;
  int base = h * (half_d * 2);
  half c = cos_tbl[k];
  half s = sin_tbl[k];
  half x_lo = xy[base + k];
  half x_hi = xy[base + k + half_d];
  xy[base + k]          = x_lo * c - x_hi * s;
  xy[base + k + half_d] = x_hi * c + x_lo * s;
}
)CL";

// Batched RoPE (Path 4 / #45b): single dispatch covers M tokens
// across all heads. cos_full / sin_full are session-wide precomputed
// LUTs of shape [max_positions, half_d] (the +half copy that the
// single-token kernel duplicates is unused here; the kernel indexes
// cos/sin only on the first-half index k).
//
//   xy layout: [M, num_heads * (2 * half_d)] fp16 row-major
//              (head_dim = 2*half_d).
//   gws = (M, num_heads, half_d). Each WI handles one (k, k+half_d)
//   pair on one (token, head). 1 dispatch per Q and per K vs the
//   M loop the per-position kernel needed.
static const std::string kRopeFp16BatchedKernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void rope_fp16_batched(
    __global       half *xy,
    __global const half *cos_full,    // [max_positions, half_d]
    __global const half *sin_full,    // [max_positions, half_d]
    const int M,
    const int num_heads,
    const int half_d,
    const int start_pos) {
  int t = get_global_id(0);
  int h = get_global_id(1);
  int k = get_global_id(2);
  if (t >= M || h >= num_heads || k >= half_d) return;
  long row_base = (long)t * num_heads * (2 * half_d) + (long)h * (2 * half_d);
  long lut_off = (long)(start_pos + t) * half_d + k;
  half c = cos_full[lut_off];
  half s = sin_full[lut_off];
  half x_lo = xy[row_base + k];
  half x_hi = xy[row_base + k + half_d];
  xy[row_base + k]          = x_lo * c - x_hi * s;
  xy[row_base + k + half_d] = x_hi * c + x_lo * s;
}
)CL";

Qwen3Forward::Qwen3Forward() = default;

Qwen3Forward::~Qwen3Forward() {
  // ForwardScratch persistent cl_mems / SVM
  {
    auto rel = [&](cl_mem &m) { if (m) { clReleaseMemObject(m); m = nullptr; } };
    auto svm = [&](void *&p) { if (p && cl_ctx_) { clSVMFree(cl_ctx_, p); p = nullptr; } };
    rel(scratch_.in_padded);    rel(scratch_.attn_normed);
    rel(scratch_.qkv_act_i8);   rel(scratch_.qkv_act_scale);
    rel(scratch_.qkv_act_zp);   rel(scratch_.qkv_act_rs);
    rel(scratch_.y_q);          rel(scratch_.y_k);          rel(scratch_.y_v);
    svm(scratch_.q_svm);        svm(scratch_.o_svm);
    rel(scratch_.o_fp32);
    rel(scratch_.wo_act_i8);    rel(scratch_.wo_act_scale);
    rel(scratch_.wo_act_zp);    rel(scratch_.wo_act_rs);
    rel(scratch_.wo_y_fp16);    rel(scratch_.wo_fp32);
    rel(scratch_.residual_1);
    rel(scratch_.ffn_in_padded);rel(scratch_.ffn_normed);
    rel(scratch_.fa_i8);        rel(scratch_.fa_sc);
    rel(scratch_.fa_zp);        rel(scratch_.fa_rs);
    rel(scratch_.up_fp16);      rel(scratch_.gate_fp16);
    rel(scratch_.up_fp32);      rel(scratch_.gate_fp32);
    rel(scratch_.swiglu_out);
    rel(scratch_.dn_i8);        rel(scratch_.dn_sc);
    rel(scratch_.dn_zp);        rel(scratch_.dn_rs);
    rel(scratch_.dn_fp16);      rel(scratch_.dn_fp32);
  }
  if (output_norm_gamma_svm_ && cl_ctx_)
    clSVMFree(cl_ctx_, output_norm_gamma_svm_);
  if (output_norm_gamma_svm_fp16_ && cl_ctx_)
    clSVMFree(cl_ctx_, output_norm_gamma_svm_fp16_);
  // Generic per-layer state (loaded via load_layer).
  for (auto &lw : layers_) {
    if (lw.attn_norm_gamma_svm && cl_ctx_)
      clSVMFree(cl_ctx_, lw.attn_norm_gamma_svm);
    if (lw.attn_norm_gamma_svm_fp16 && cl_ctx_)
      clSVMFree(cl_ctx_, lw.attn_norm_gamma_svm_fp16);
    if (lw.q_norm_gamma_svm_fp16 && cl_ctx_)
      clSVMFree(cl_ctx_, lw.q_norm_gamma_svm_fp16);
    if (lw.k_norm_gamma_svm_fp16 && cl_ctx_)
      clSVMFree(cl_ctx_, lw.k_norm_gamma_svm_fp16);
    if (lw.ffn_norm_gamma_svm && cl_ctx_)
      clSVMFree(cl_ctx_, lw.ffn_norm_gamma_svm);
    if (lw.ffn_norm_gamma_svm_fp16 && cl_ctx_)
      clSVMFree(cl_ctx_, lw.ffn_norm_gamma_svm_fp16);
    if (lw.post_attn_norm_gamma_svm && cl_ctx_)
      clSVMFree(cl_ctx_, lw.post_attn_norm_gamma_svm);
    if (lw.post_attn_norm_gamma_svm_fp16 && cl_ctx_)
      clSVMFree(cl_ctx_, lw.post_attn_norm_gamma_svm_fp16);
    if (lw.post_ffn_norm_gamma_svm && cl_ctx_)
      clSVMFree(cl_ctx_, lw.post_ffn_norm_gamma_svm);
    if (lw.post_ffn_norm_gamma_svm_fp16 && cl_ctx_)
      clSVMFree(cl_ctx_, lw.post_ffn_norm_gamma_svm_fp16);
    if (lw.cache_k_svm && cl_ctx_)
      clSVMFree(cl_ctx_, lw.cache_k_svm);
    if (lw.cache_v_svm && cl_ctx_)
      clSVMFree(cl_ctx_, lw.cache_v_svm);
    if (lw.cache_v_image_ohwi) {
      clReleaseMemObject(lw.cache_v_image_ohwi);
      lw.cache_v_image_ohwi = nullptr;
    }
    if (lw.cache_v_buf_ohwi) {
      clReleaseMemObject(lw.cache_v_buf_ohwi);
      lw.cache_v_buf_ohwi = nullptr;
    }
    if (lw.cache_k_image_ohwi) {
      clReleaseMemObject(lw.cache_k_image_ohwi);
      lw.cache_k_image_ohwi = nullptr;
    }
    if (lw.cache_k_buf_ohwi) {
      clReleaseMemObject(lw.cache_k_buf_ohwi);
      lw.cache_k_buf_ohwi = nullptr;
    }
    release_v8c_weight(&lw.wq);
    release_v8c_weight(&lw.wk);
    release_v8c_weight(&lw.wv);
    release_v8c_weight(&lw.wo);
    release_v8c_weight(&lw.ffn_up);
    release_v8c_weight(&lw.ffn_gate);
    release_v8c_weight(&lw.ffn_down);
  }
  if (layer0_output_fp32_ != nullptr)
    clReleaseMemObject(layer0_output_fp32_);
  if (layer0_residual1_fp32_ != nullptr)
    clReleaseMemObject(layer0_residual1_fp32_);
  if (layer0_ffn_norm_gamma_svm_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_ffn_norm_gamma_svm_);
  release_v8c_weight(&layer0_wq_);
  release_v8c_weight(&layer0_wk_);
  release_v8c_weight(&layer0_wv_);
  release_v8c_weight(&layer0_wo_);
  release_v8c_weight(&layer0_ffn_up_);
  release_v8c_weight(&layer0_ffn_gate_);
  release_v8c_weight(&layer0_ffn_down_);
  if (layer0_cache_k_svm_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_cache_k_svm_);
  if (layer0_cache_v_svm_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_cache_v_svm_);
  if (layer0_rope_cos_svm_fp16_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_rope_cos_svm_fp16_);
  if (layer0_rope_sin_svm_fp16_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_rope_sin_svm_fp16_);
  if (rope_cos_full_svm_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, rope_cos_full_svm_);
  if (rope_sin_full_svm_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, rope_sin_full_svm_);
  if (layer0_q_norm_gamma_svm_fp16_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_q_norm_gamma_svm_fp16_);
  if (layer0_k_norm_gamma_svm_fp16_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_k_norm_gamma_svm_fp16_);
  if (layer0_attn_norm_gamma_svm_ != nullptr && cl_ctx_ != nullptr) {
    clSVMFree(cl_ctx_, layer0_attn_norm_gamma_svm_);
  }
  if (weight_mmap_ != nullptr && weight_bytes_ > 0) {
    munmap(const_cast<uint8_t *>(weight_mmap_), weight_bytes_);
  }
  if (weight_fd_ >= 0) {
    close(weight_fd_);
  }
}

void Qwen3Forward::release_v8c_weight(V8cFcWeight *w) {
  if (w->scale_buf != nullptr) clReleaseMemObject(w->scale_buf);
  if (w->row_sum_w_int4 != nullptr) clReleaseMemObject(w->row_sum_w_int4);
  // weight_image is owned by the backing's image cache; the backing's
  // destructor releases it. We don't ReleaseMemObject it ourselves.
  if (w->backing != nullptr) {
    delete static_cast<nntrainer::tv::TensorBacking *>(w->backing);
  }
  *w = V8cFcWeight{};
}

size_t Qwen3Forward::layers_start_offset() const {
  return embed_table_bytes();
}

size_t Qwen3Forward::embed_table_bytes() const {
  // Q6_K: vocab * hidden / 256 blocks * 210 bytes/block.
  const size_t total_elts =
    static_cast<size_t>(cfg_.vocab_size) * cfg_.hidden_size;
  if ((total_elts % Q6_K_BLOCK_ELTS) != 0) {
    throw std::runtime_error("Q6_K requires vocab*hidden multiple of 256");
  }
  return (total_elts / Q6_K_BLOCK_ELTS) * Q6_K_BLOCK_BYTES;
}

bool Qwen3Forward::init(const Qwen3Config &cfg, const std::string &weight_path) {
  cfg_ = cfg;
  // #90 Resolve architecture feature flags from model identity once, centrally,
  // so every dispatch site below reads explicit features instead of is_gemma2.
  cfg_.derive_features();
  weight_path_ = weight_path;

  weight_fd_ = open(weight_path.c_str(), O_RDONLY);
  if (weight_fd_ < 0) {
    std::fprintf(stderr, "[qwen3-gpu] open(%s) failed: %s\n",
                 weight_path.c_str(), std::strerror(errno));
    return false;
  }
  struct stat st;
  if (fstat(weight_fd_, &st) != 0) {
    std::fprintf(stderr, "[qwen3-gpu] fstat failed: %s\n", std::strerror(errno));
    return false;
  }
  weight_bytes_ = static_cast<size_t>(st.st_size);
  void *m = mmap(nullptr, weight_bytes_, PROT_READ, MAP_PRIVATE, weight_fd_, 0);
  if (m == MAP_FAILED) {
    std::fprintf(stderr, "[qwen3-gpu] mmap failed: %s\n", std::strerror(errno));
    weight_bytes_ = 0;
    return false;
  }
  weight_mmap_ = static_cast<const uint8_t *>(m);

  auto *cl =
    static_cast<nntrainer::ClContext *>(
      nntrainer::Engine::Global().getRegisteredContext("gpu"));
  if (cl == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] no gpu context registered\n");
    return false;
  }
  cl_ctx_ = cl->context_inst_.GetContext();
  cl_q_ = cl->command_queue_inst_.GetCommandQueue();
  cl_dev_ = cl->context_inst_.GetDeviceId();
  if (cl_ctx_ == nullptr || cl_q_ == nullptr || cl_dev_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] ClContext handles null: ctx=%p q=%p dev=%p\n",
                 cl_ctx_, cl_q_, cl_dev_);
    return false;
  }

  // Device specialization (paper §3.4 / increment 3): query image2d caps once
  // so packed activation views are validated against real device limits.
  img_caps_ = nntrainer::tv::queryDeviceImageCaps(cl_dev_);
  std::fprintf(stderr,
               "[qwen3-gpu] device image caps: support=%d max=%zux%zu "
               "pitch_align=%u\n",
               (int)img_caps_.image_support, img_caps_.max_width,
               img_caps_.max_height, img_caps_.pitch_align);

  std::fprintf(stderr,
               "[qwen3-gpu] init OK: weights=%s size=%zu MB cl_ctx=%p\n",
               weight_path.c_str(), weight_bytes_ / (1024 * 1024), cl_ctx_);
  std::fprintf(stderr,
               "[qwen3-gpu] cfg: hidden=%u inter=%u d=%u hQ=%u hKV=%u "
               "L=%u vocab=%u S_max=%u\n",
               cfg_.hidden_size, cfg_.intermediate_size, cfg_.head_dim,
               cfg_.num_heads_Q, cfg_.num_heads_KV, cfg_.num_layers,
               cfg_.vocab_size, cfg_.max_seq_len);
  return true;
}

void Qwen3Forward::dump_weight_header(size_t n) {
  if (weight_mmap_ == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] dump_weight_header: not mmap'd\n");
    return;
  }
  const size_t lim = (n < weight_bytes_) ? n : weight_bytes_;
  std::fprintf(stderr, "[qwen3-gpu] first %zu bytes of %s:\n", lim,
               weight_path_.c_str());
  for (size_t i = 0; i < lim; ++i) {
    std::fprintf(stderr, "%02x ", weight_mmap_[i]);
    if ((i + 1) % 16 == 0) std::fprintf(stderr, "\n");
  }
  if (lim % 16 != 0) std::fprintf(stderr, "\n");
}

bool Qwen3Forward::svm_smoke_test(size_t bytes) {
  if (cl_ctx_ == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] svm_smoke_test: no cl_ctx\n");
    return false;
  }
  void *svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, bytes, /*alignment*/ 0);
  if (svm == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] clSVMAlloc(%zu) returned null — SVM may be "
                 "unsupported on this device\n", bytes);
    return false;
  }
  // Map for host write (CL_MAP_WRITE) — coarse-grained SVM requires
  // explicit map/unmap; fine-grained also accepts it as a no-op.
  cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, svm, bytes, 0,
                               nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] clEnqueueSVMMap(WRITE) err=%d\n", err);
    clSVMFree(cl_ctx_, svm);
    return false;
  }
  uint8_t *p = static_cast<uint8_t *>(svm);
  for (size_t i = 0; i < bytes; ++i) p[i] = static_cast<uint8_t>(i & 0xFF);
  err = clEnqueueSVMUnmap(cl_q_, svm, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] clEnqueueSVMUnmap(write) err=%d\n", err);
    clSVMFree(cl_ctx_, svm);
    return false;
  }
  clFinish(cl_q_);

  err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_READ, svm, bytes, 0, nullptr,
                        nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] clEnqueueSVMMap(READ) err=%d\n", err);
    clSVMFree(cl_ctx_, svm);
    return false;
  }
  bool ok = true;
  for (size_t i = 0; i < bytes; ++i) {
    if (p[i] != static_cast<uint8_t>(i & 0xFF)) {
      std::fprintf(stderr,
                   "[qwen3-gpu] svm round-trip mismatch at %zu: got 0x%02x\n",
                   i, p[i]);
      ok = false;
      break;
    }
  }
  err = clEnqueueSVMUnmap(cl_q_, svm, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] clEnqueueSVMUnmap(read) err=%d\n", err);
    ok = false;
  }
  clFinish(cl_q_);
  clSVMFree(cl_ctx_, svm);
  if (ok) {
    std::fprintf(stderr,
                 "[qwen3-gpu] SVM smoke test PASS: %zu bytes round-trip\n",
                 bytes);
  }
  return ok;
}

bool Qwen3Forward::load_layer0_attention_norm_to_svm() {
  if (weight_mmap_ == nullptr || cl_ctx_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] load_layer0_attention_norm: not initialized\n");
    return false;
  }
  if (layer0_attn_norm_gamma_svm_ != nullptr) {
    return true; // already loaded
  }

  const size_t embed_bytes = embed_table_bytes();
  const size_t gamma_bytes =
    static_cast<size_t>(cfg_.hidden_size) * sizeof(float);
  const size_t gamma_offset = embed_bytes;

  if (gamma_offset + gamma_bytes > weight_bytes_) {
    std::fprintf(stderr,
                 "[qwen3-gpu] computed gamma offset %zu + %zu > file %zu\n",
                 gamma_offset, gamma_bytes, weight_bytes_);
    return false;
  }
  const float *gamma_host =
    reinterpret_cast<const float *>(weight_mmap_ + gamma_offset);

  // Sanity log: dump first 8 + last 4 gamma values. For typical LLM RMSNorm
  // the loaded values cluster near 1.0 (initialized as ones, learned to
  // small deviations). Wildly different values strongly suggest the offset
  // is wrong (we landed mid-Q6_K-block or skipped past a wrong tensor).
  std::fprintf(stderr,
               "[qwen3-gpu] layer 0 attention_norm gamma "
               "(host fp32, offset=%zu MB, %u floats):\n  first 8:",
               gamma_offset / (1024 * 1024), cfg_.hidden_size);
  for (int i = 0; i < 8; ++i)
    std::fprintf(stderr, " %f", gamma_host[i]);
  std::fprintf(stderr, "\n  last  4:");
  for (int i = 0; i < 4; ++i)
    std::fprintf(stderr, " %f",
                 gamma_host[cfg_.hidden_size - 4 + i]);
  std::fprintf(stderr, "\n");

  layer0_attn_norm_gamma_svm_ =
    clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, gamma_bytes, /*alignment*/ 0);
  if (layer0_attn_norm_gamma_svm_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] clSVMAlloc(%zu) for gamma failed\n",
                 gamma_bytes);
    return false;
  }
  cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE,
                               layer0_attn_norm_gamma_svm_, gamma_bytes, 0,
                               nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] gamma SVMMap WRITE err=%d\n", err);
    return false;
  }
  std::memcpy(layer0_attn_norm_gamma_svm_, gamma_host, gamma_bytes);
  err = clEnqueueSVMUnmap(cl_q_, layer0_attn_norm_gamma_svm_, 0, nullptr,
                          nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] gamma SVMUnmap err=%d\n", err);
    return false;
  }
  clFinish(cl_q_);
  std::fprintf(stderr,
               "[qwen3-gpu] layer 0 attention_norm gamma -> SVM ok "
               "(%zu bytes)\n", gamma_bytes);
  return true;
}

bool Qwen3Forward::run_rmsnorm_layer0() {
  if (layer0_attn_norm_gamma_svm_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] run_rmsnorm_layer0: gamma not loaded\n");
    return false;
  }
  const unsigned int W = cfg_.hidden_size;
  const unsigned int H = 1;
  if (W % 4 != 0) {
    std::fprintf(stderr,
                 "[qwen3-gpu] rmsnorm.cl requires hidden %% 4 == 0\n");
    return false;
  }
  const size_t io_bytes = static_cast<size_t>(W) * sizeof(float);

  // Allocate input + output SVM. Input is a known deterministic pattern
  // so we can spot-check the rmsnorm math by hand.
  void *in_svm =
    clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, io_bytes, 0);
  void *out_svm =
    clSVMAlloc(cl_ctx_, CL_MEM_WRITE_ONLY, io_bytes, 0);
  if (in_svm == nullptr || out_svm == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] rmsnorm SVMAlloc failed\n");
    if (in_svm) clSVMFree(cl_ctx_, in_svm);
    if (out_svm) clSVMFree(cl_ctx_, out_svm);
    return false;
  }

  cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, in_svm,
                               io_bytes, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] rmsnorm in SVMMap WRITE err=%d\n", err);
    clSVMFree(cl_ctx_, in_svm); clSVMFree(cl_ctx_, out_svm);
    return false;
  }
  float *in_ptr = static_cast<float *>(in_svm);
  // Pattern: gentle ramp so RMS = sqrt(mean of squares) is computable.
  // Values 0.001 * (i + 1), i in [0, W). RMS = 0.001 * sqrt(sum/W) where
  // sum = W*(W+1)*(2W+1)/6. For W=2560: sum=5604687360, mean=2189330,
  // sqrt(mean)=1479.6, scale = 1/1479.6 ≈ 6.758e-4. After scale, then
  // multiplied by gamma (~1.0), output[0] ≈ 0.001 * 6.758e-4 ≈ 6.758e-7.
  // (Tiny because pattern range >> gamma range; this is just to verify
  // the kernel runs and produces finite numbers.)
  for (unsigned int i = 0; i < W; ++i)
    in_ptr[i] = 0.001f * static_cast<float>(i + 1);
  err = clEnqueueSVMUnmap(cl_q_, in_svm, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] rmsnorm in SVMUnmap err=%d\n", err);
    clSVMFree(cl_ctx_, in_svm); clSVMFree(cl_ctx_, out_svm);
    return false;
  }
  clFinish(cl_q_);

  // Register + dispatch rmsnorm_cl.
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  auto kp = cl->registerClKernel(nntrainer::rmsnorm_kernel, "rmsnorm_cl");
  if (!kp) {
    std::fprintf(stderr, "[qwen3-gpu] registerClKernel(rmsnorm_cl) failed\n");
    clSVMFree(cl_ctx_, in_svm); clSVMFree(cl_ctx_, out_svm);
    return false;
  }
  if (!kp->SetKernelSVMArguments(0, in_svm) ||
      !kp->SetKernelSVMArguments(1, out_svm) ||
      !kp->SetKernelSVMArguments(2, layer0_attn_norm_gamma_svm_)) {
    std::fprintf(stderr, "[qwen3-gpu] rmsnorm SVM args failed\n");
    clSVMFree(cl_ctx_, in_svm); clSVMFree(cl_ctx_, out_svm);
    return false;
  }
  float eps = cfg_.rms_norm_eps;
  int H_i = static_cast<int>(H), W_i = static_cast<int>(W);
  if (!kp->SetKernelArguments(3, &eps, sizeof(float)) ||
      !kp->SetKernelArguments(4, &H_i, sizeof(int)) ||
      !kp->SetKernelArguments(5, &W_i, sizeof(int))) {
    std::fprintf(stderr, "[qwen3-gpu] rmsnorm scalar args failed\n");
    clSVMFree(cl_ctx_, in_svm); clSVMFree(cl_ctx_, out_svm);
    return false;
  }
  // 1 workgroup per row (h = get_group_id(0)); LWS=64 to match
  // qcom_reqd_sub_group_size("half") on Adreno.
  std::array<size_t, 1> gws = {static_cast<size_t>(H) * 64};
  std::array<size_t, 1> lws = {64};
  cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                        lws.data(), 0, nullptr, nullptr);
  clFinish(cl_q_);

  // Read back + sanity check.
  err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_READ, out_svm, io_bytes, 0,
                        nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] rmsnorm out SVMMap READ err=%d\n", err);
    clSVMFree(cl_ctx_, in_svm); clSVMFree(cl_ctx_, out_svm);
    return false;
  }
  const float *out_ptr = static_cast<const float *>(out_svm);
  bool all_finite = true;
  for (unsigned int i = 0; i < W; ++i) {
    if (!std::isfinite(out_ptr[i])) { all_finite = false; break; }
  }
  // Quick host-side reference compute for the first 4 values.
  // input[i] = 0.001*(i+1); RMS = sqrt(mean(input^2)); scale = 1/sqrt(RMS^2+eps)
  // expected[i] = input[i] * scale * gamma[i]
  err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_READ,
                        layer0_attn_norm_gamma_svm_, W * sizeof(float), 0,
                        nullptr, nullptr);
  const float *gamma_ptr =
    static_cast<const float *>(layer0_attn_norm_gamma_svm_);
  double ss = 0.0;
  for (unsigned int i = 0; i < W; ++i) {
    const double v = 0.001 * (i + 1);
    ss += v * v;
  }
  const double mean = ss / W;
  const double scale = 1.0 / std::sqrt(mean + eps);
  std::fprintf(stderr,
               "[qwen3-gpu] rmsnorm dispatch: H=%u W=%u eps=%g\n", H, W, eps);
  std::fprintf(stderr,
               "  host-ref mean=%g scale=%g, expected first 4:\n   ", mean,
               scale);
  for (int i = 0; i < 4; ++i) {
    const double expected =
      0.001 * (i + 1) * scale * static_cast<double>(gamma_ptr[i]);
    std::fprintf(stderr, " %g", expected);
  }
  std::fprintf(stderr, "\n  gpu  out first 4:\n   ");
  for (int i = 0; i < 4; ++i)
    std::fprintf(stderr, " %g", static_cast<double>(out_ptr[i]));
  std::fprintf(stderr, "\n  gpu  out last  4:\n   ");
  for (int i = 0; i < 4; ++i)
    std::fprintf(stderr, " %g", static_cast<double>(out_ptr[W - 4 + i]));
  std::fprintf(stderr, "\n  all_finite=%d\n", all_finite ? 1 : 0);

  err = clEnqueueSVMUnmap(cl_q_, layer0_attn_norm_gamma_svm_, 0, nullptr,
                          nullptr);
  err = clEnqueueSVMUnmap(cl_q_, out_svm, 0, nullptr, nullptr);
  clFinish(cl_q_);
  clSVMFree(cl_ctx_, in_svm);
  clSVMFree(cl_ctx_, out_svm);
  return all_finite;
}

size_t Qwen3Forward::qint4_record_bytes(size_t file_offset, unsigned int K,
                                        unsigned int N) const {
  // FC record size is container-dependent — peek the u16 tag at the record
  // start: 8 = int8 (paper 8/4/4, K*N bytes + fp16 scales), 1 = shared plain
  // container (PR#3978 form: plain nibbles + fp32 scales + KAI nr=8 pad),
  // anything else = legacy KAI Section A (K*N/2 + fp16 scales).
  const uint16_t qs =
    *reinterpret_cast<const uint16_t *>(weight_mmap_ + file_offset);
  if (qs == 1) {
    return sizeof(uint16_t) +
           nntrainer::Int4Utils::plainRecordPayloadBytes(N, K);
  }
  const size_t packed = (qs == 8) ? (size_t)K * N : (size_t)K * N / 2;
  return sizeof(uint16_t) + packed + (size_t)N * 2;
}

bool Qwen3Forward::load_qint4_weight_at(size_t file_offset, unsigned int K,
                                        unsigned int N, V8cFcWeight *out,
                                        const char *tag) {
  if (weight_mmap_ == nullptr || cl_ctx_ == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] %s: not initialized\n", tag);
    return false;
  }
  if (out->backing != nullptr) return true; // already loaded

  // [qscheme u16][packed bytes][scales]. Packed = K*N/2 for int4 KAI
  // Section A (tag 6), K*N for int8 (tag 8 = paper 8/4/4 attention weight),
  // or the PR#3978 plain payload (tag 1: plain nibbles + fp32 scales + pad
  // — repacked to Section A below). Any other tag would silently decode as
  // garbage, so reject it.
  const uint16_t qscheme =
    *reinterpret_cast<const uint16_t *>(weight_mmap_ + file_offset);
  const bool is_int8 = (qscheme == 8);
  const bool is_plain = (qscheme == 1);
  if (!is_int8 && !is_plain && qscheme != 6) {
    std::fprintf(stderr,
                 "[qwen3-gpu] %s off=%zu: unsupported qscheme tag %u — "
                 "refusing to decode (would be silent garbage)\n",
                 tag, file_offset, qscheme);
    return false;
  }
  const size_t packed_bytes = is_int8 ? (size_t)K * N : (size_t)K * N / 2;
  const size_t total_bytes = qint4_record_bytes(file_offset, K, N);
  if (file_offset + total_bytes > weight_bytes_) {
    std::fprintf(stderr,
                 "[qwen3-gpu] %s offset %zu + size %zu > file %zu\n", tag,
                 file_offset, total_bytes, weight_bytes_);
    return false;
  }
  const uint8_t *payload = weight_mmap_ + file_offset + sizeof(uint16_t);
  // Plain container: permute the row-major nibbles into Section A and widen
  // the fp32 scales to the fp16 form the v8c packer consumes — bit-identical
  // to loading the equivalent Section A record (lossless fp32->fp16: the
  // writer stored fp16-rounded values).
  std::vector<uint8_t> sa_repacked;
  std::vector<uint16_t> sc_repacked;
  const uint8_t *section_a = payload;
  const uint16_t *scales_fp16 =
    reinterpret_cast<const uint16_t *>(payload + packed_bytes);
  if (is_plain) {
    sa_repacked.resize(nntrainer::Int4Utils::kaiNibblePayloadBytes(N, K));
    nntrainer::Int4Utils::packPlainToSectionA(payload, N, K,
                                              sa_repacked.data());
    const uint8_t *sc_src =
      payload + nntrainer::Int4Utils::plainScalesOffsetBytes(N, K);
    sc_repacked.resize(N);
    for (unsigned int n = 0; n < N; ++n) {
      float s;
      std::memcpy(&s, sc_src + (size_t)n * sizeof(float), sizeof(float));
      sc_repacked[n] = nntrainer::compute_fp32_to_fp16(s);
    }
    section_a = sa_repacked.data();
    scales_fp16 = sc_repacked.data();
  }

  std::fprintf(stderr,
               "[qwen3-gpu] %s off=%zu (~%zu MB) qscheme=%u%s K=%u N=%u "
               "packed=%zu KB\n", tag, file_offset,
               file_offset / (1024 * 1024), qscheme,
               is_int8 ? "(int8)" : (is_plain ? "(plain)" : ""), K, N,
               packed_bytes / 1024);

  cl_mem scale_buf = nullptr;
  cl_mem rsw_buf = nullptr;
  std::unique_ptr<nntrainer::tv::TensorBacking> backing;
  try {
    backing = is_int8
                ? nntrainer::make_v8c_int8_weight_backing(
                    reinterpret_cast<const int8_t *>(section_a), scales_fp16, N,
                    K, &scale_buf, &rsw_buf)
                : nntrainer::make_v8c_weight_backing_from_kai_section_a(
                    section_a, scales_fp16, N, K, &scale_buf, &rsw_buf);
  } catch (const std::exception &e) {
    std::fprintf(stderr,
                 "[qwen3-gpu] %s make_v8c_weight_backing threw: %s\n",
                 tag, e.what());
    if (scale_buf) clReleaseMemObject(scale_buf);
    if (rsw_buf) clReleaseMemObject(rsw_buf);
    return false;
  }
  // #v8c-buf: skip the image2d-from-buffer view when the buffer-load FC path
  // is active (NNTR_V8C_BUF=1). On runtimes like Intel NEO the buffer kernel
  // indexes the raw cl_mem directly; the image view is unused. Default builds
  // (Adreno) still create the view, keeping that path bit-identical.
  const bool buf_path = []() {
    const char *e = std::getenv("NNTR_V8C_BUF");
    return e && std::atoi(e) != 0;
  }();
  if (!buf_path) {
    nntrainer::tv::ViewSpec ws;
    ws.kind = nntrainer::tv::ViewKind::IMAGE_2D;
    ws.image_channel_order = CL_RGBA;
    ws.image_channel_type = CL_UNSIGNED_INT32;
    // int8: 16 int8/texel -> width K/16, pitch K. int4: 32 int4/texel -> K/32, K/2.
    ws.width = is_int8 ? K / 16 : K / 32;
    ws.height = N;
    ws.row_pitch_bytes = is_int8 ? K : K / 2;
    try {
      out->weight_image = backing->imageView(ws);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "[qwen3-gpu] %s imageView threw: %s\n", tag,
                   e.what());
      clReleaseMemObject(scale_buf);
      clReleaseMemObject(rsw_buf);
      return false;
    }
  }
  // Raw backing buffer for the NNTR_V8C_BUF (buffer-load) device path.
  // image2d-from-buffer reads the same bytes; the buffer variant indexes
  // this cl_mem as uint4 texels directly.
  out->weight_buf = backing->buffer();
  out->backing = backing.release();
  out->scale_buf = scale_buf;
  out->row_sum_w_int4 = rsw_buf;
  out->K = K;
  out->N = N;
  out->is_int8 = is_int8;
  return true;
}

// Convert N fp32 values to fp16-bits (uint16). Round-to-nearest-even.
// Minimal correct converter — for one-time small gamma loads, perf isn't
// a concern. Returns 0x7E00 (qNaN) on NaN, ±inf on overflow, denormal
// on underflow.
static uint16_t f2h(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  uint32_t s = (u >> 16) & 0x8000u;
  int32_t e = ((u >> 23) & 0xff) - 127 + 15;
  uint32_t m = u & 0x7fffff;
  if (((u >> 23) & 0xff) == 0xff) {
    // inf / nan
    return (uint16_t)(s | 0x7c00 | (m ? 0x200 : 0));
  }
  if (e >= 31) return (uint16_t)(s | 0x7c00);             // overflow -> inf
  if (e <= 0) {
    // subnormal
    if (e < -10) return (uint16_t)s;
    m |= 0x800000;
    uint32_t shift = (uint32_t)(14 - e);
    uint32_t half = m >> shift;
    if ((m >> (shift - 1)) & 1u) half += 1; // round to nearest
    return (uint16_t)(s | half);
  }
  uint16_t r = (uint16_t)(s | (uint16_t)(e << 10) | (uint16_t)(m >> 13));
  if (m & 0x1000) r += 1; // round-to-nearest-even (simplified)
  return r;
}

// Build a v8c int8 activation image2d view of a row-major [rows][row_bytes]
// scratch buffer, routing the packing decision through the tensor-virtualization
// layer (paper §3.1: tv::make_image2d_rgba_uint32 = 16 int8 per RGBA UINT32
// texel). Centralizes what used to be 4 hand-rolled clCreateImage blocks so the
// layout choice lives in one place. Increment 3 (paper §3.4 device
// specialization): the spec is pre-validated against the device's image2d caps,
// so an oversized/unsupported view is reported clearly instead of failing
// opaquely inside clCreateImage. Returns null on failure.
static cl_mem make_act_image2d(cl_context ctx, cl_mem buf, unsigned int row_bytes,
                               unsigned int rows,
                               const nntrainer::tv::DeviceImageCaps &caps) {
  const nntrainer::tv::ViewSpec s =
    nntrainer::tv::make_image2d_rgba_uint32(row_bytes, rows);
  if (!nntrainer::tv::image2dViewFits(s, caps)) {
    std::fprintf(stderr,
                 "[qwen3-gpu] make_act_image2d: %zux%zu texel view exceeds "
                 "device image caps (max %zux%zu, support=%d)\n",
                 s.width, s.height, caps.max_width, caps.max_height,
                 (int)caps.image_support);
    return nullptr;
  }
  cl_image_format fmt{s.image_channel_order, s.image_channel_type};
  cl_image_desc desc{};
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = s.width;
  desc.image_height = s.height;
  desc.image_row_pitch = s.row_pitch_bytes;
  desc.buffer = buf;
  cl_int err = CL_SUCCESS;
  cl_mem img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
  if (err != CL_SUCCESS || img == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] make_act_image2d err=%d\n", err);
    return nullptr;
  }
  return img;
}

bool Qwen3Forward::load_layer0_qk_norm_gammas() {
  if (weight_mmap_ == nullptr || cl_ctx_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] load_layer0_qk_norm_gammas: not initialized\n");
    return false;
  }
  if (layer0_q_norm_gamma_svm_fp16_ != nullptr &&
      layer0_k_norm_gamma_svm_fp16_ != nullptr)
    return true;

  // Per the layer save layout (qkv weights commit), q_norm/k_norm gammas
  // live right after wq/wk respectively. Recompute offsets here so the
  // loader is independent of the wq load step.
  const size_t embed_bytes = embed_table_bytes();
  const size_t attn_norm_bytes =
    static_cast<size_t>(cfg_.hidden_size) * sizeof(float);
  const size_t head_dim_bytes =
    static_cast<size_t>(cfg_.head_dim) * sizeof(float);

  const unsigned int K_hidden = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;

  const size_t wq_off = embed_bytes + attn_norm_bytes;
  const size_t wq_bytes = qint4_record_bytes(wq_off, K_hidden, N_q);
  const size_t q_norm_off = wq_off + wq_bytes;
  const size_t wk_off = q_norm_off + head_dim_bytes;
  const size_t wk_bytes = qint4_record_bytes(wk_off, K_hidden, N_kv);
  const size_t k_norm_off = wk_off + wk_bytes;

  if (k_norm_off + head_dim_bytes > weight_bytes_) {
    std::fprintf(stderr,
                 "[qwen3-gpu] qk_norm offsets out of range\n");
    return false;
  }

  const float *q_gamma_fp32 =
    reinterpret_cast<const float *>(weight_mmap_ + q_norm_off);
  const float *k_gamma_fp32 =
    reinterpret_cast<const float *>(weight_mmap_ + k_norm_off);

  std::fprintf(stderr,
               "[qwen3-gpu] q_norm off=%zu (~%zu KB) first 4: %g %g %g %g\n"
               "[qwen3-gpu] k_norm off=%zu (~%zu KB) first 4: %g %g %g %g\n",
               q_norm_off, q_norm_off / 1024,
               q_gamma_fp32[0], q_gamma_fp32[1], q_gamma_fp32[2],
               q_gamma_fp32[3],
               k_norm_off, k_norm_off / 1024,
               k_gamma_fp32[0], k_gamma_fp32[1], k_gamma_fp32[2],
               k_gamma_fp32[3]);

  // Convert fp32 -> fp16 and push to SVM. head_dim values × 2 bytes.
  const size_t gamma_fp16_bytes =
    (size_t)cfg_.head_dim * sizeof(uint16_t);
  auto load_one = [this, gamma_fp16_bytes](
                    const float *src_fp32, void **dst_svm,
                    const char *tag) -> bool {
    *dst_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, gamma_fp16_bytes, 0);
    if (*dst_svm == nullptr) {
      std::fprintf(stderr, "[qwen3-gpu] %s SVMAlloc failed\n", tag);
      return false;
    }
    cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, *dst_svm,
                                 gamma_fp16_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::fprintf(stderr, "[qwen3-gpu] %s SVMMap WRITE err=%d\n", tag, err);
      return false;
    }
    uint16_t *p = static_cast<uint16_t *>(*dst_svm);
    for (unsigned int i = 0; i < cfg_.head_dim; ++i)
      p[i] = f2h(src_fp32[i]);
    err = clEnqueueSVMUnmap(cl_q_, *dst_svm, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;
    clFinish(cl_q_);
    return true;
  };
  if (!load_one(q_gamma_fp32, &layer0_q_norm_gamma_svm_fp16_, "q_norm")) return false;
  if (!load_one(k_gamma_fp32, &layer0_k_norm_gamma_svm_fp16_, "k_norm")) return false;
  std::fprintf(stderr,
               "[qwen3-gpu] q_norm + k_norm gammas -> SVM (fp16, %zu B each)\n",
               gamma_fp16_bytes);
  return true;
}

bool Qwen3Forward::load_layer0_qkv_weights() {
  // Layer save order in Qwen3 createTransformerDecoderBlock +
  // Qwen3Transformer::createAttention:
  //   attn_norm -> wq -> q_norm -> wk -> k_norm -> wv -> wo -> ffn_norm -> ...
  // Per-tensor disk size:
  //   fp32 norm gamma:  width * 4
  //   QINT4 FC weight:  2 + (K*N)/2 + N*2
  const size_t embed_bytes = embed_table_bytes();
  const size_t attn_norm_bytes =
    static_cast<size_t>(cfg_.hidden_size) * sizeof(float);
  const size_t q_norm_bytes =
    static_cast<size_t>(cfg_.head_dim) * sizeof(float);
  const size_t k_norm_bytes = q_norm_bytes;

  const unsigned int K_hidden = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;

  const size_t wq_off = embed_bytes + attn_norm_bytes;
  const size_t wq_bytes = qint4_record_bytes(wq_off, K_hidden, N_q);
  const size_t wk_off = wq_off + wq_bytes + q_norm_bytes;
  const size_t wk_bytes = qint4_record_bytes(wk_off, K_hidden, N_kv);
  const size_t wv_off = wk_off + wk_bytes + k_norm_bytes;

  return load_qint4_weight_at(wq_off, K_hidden, N_q, &layer0_wq_, "wq") &&
         load_qint4_weight_at(wk_off, K_hidden, N_kv, &layer0_wk_, "wk") &&
         load_qint4_weight_at(wv_off, K_hidden, N_kv, &layer0_wv_, "wv");
}

bool Qwen3Forward::load_layer0_wo() {
  // wo immediately follows wv (no norm in between). See layer save
  // order in load_layer0_qkv_weights for the running offsets.
  const size_t embed_bytes = embed_table_bytes();
  const size_t attn_norm_bytes =
    static_cast<size_t>(cfg_.hidden_size) * sizeof(float);
  const size_t head_dim_bytes =
    static_cast<size_t>(cfg_.head_dim) * sizeof(float);
  const unsigned int K_hidden = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;
  const size_t wq_off = embed_bytes + attn_norm_bytes;
  const size_t wq_bytes = qint4_record_bytes(wq_off, K_hidden, N_q);
  const size_t wk_off = wq_off + wq_bytes + head_dim_bytes;
  const size_t wk_bytes = qint4_record_bytes(wk_off, K_hidden, N_kv);
  const size_t wv_off = wk_off + wk_bytes + head_dim_bytes;
  const size_t wv_bytes = qint4_record_bytes(wv_off, K_hidden, N_kv);
  const size_t wo_off = wv_off + wv_bytes;

  // wo: [K=hQ*d, N=hidden] in the saved tensor (createTransformerDecoder
  // sets the FC unit to DIM=hidden). After v8c packing, K and N match
  // the on-disk save dim.
  const unsigned int wo_K = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int wo_N = cfg_.hidden_size;
  return load_qint4_weight_at(wo_off, wo_K, wo_N, &layer0_wo_, "wo");
}

namespace {
// fp16-bits -> fp32 (host-side decode), used for printing GPU outputs.
inline float h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
  uint32_t o;
  if (e == 0) o = m ? (m << 13) : 0;
  else if (e == 31) o = (m ? 0x7fc00000u : 0x7f800000u);
  else { e += 112; o = (e << 23) | (m << 13); }
  o |= s;
  float f; std::memcpy(&f, &o, 4); return f;
}

// Summary print + finite check for a fp16 cl_mem of length N. Used as a
// quick sanity gate after each GPU FC dispatch.
bool summarize_fp16_buf(cl_command_queue q, cl_mem buf, unsigned int N,
                        const char *tag) {
  std::vector<uint16_t> host(N);
  cl_int err = clEnqueueReadBuffer(q, buf, CL_TRUE, 0,
                                   (size_t)N * sizeof(uint16_t),
                                   host.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] %s readback err=%d\n", tag, err);
    return false;
  }
  bool all_finite = true;
  float min_v = std::numeric_limits<float>::infinity();
  float max_v = -std::numeric_limits<float>::infinity();
  for (unsigned int n = 0; n < N; ++n) {
    float f = h2f(host[n]);
    if (!std::isfinite(f)) all_finite = false;
    if (f < min_v) min_v = f;
    if (f > max_v) max_v = f;
  }
  std::fprintf(stderr,
               "[qwen3-gpu] %s fp16 N=%u first 8:", tag, N);
  for (int i = 0; i < 8; ++i)
    std::fprintf(stderr, " %g", h2f(host[i]));
  std::fprintf(stderr, "\n  last 4:");
  for (int i = 0; i < 4; ++i)
    std::fprintf(stderr, " %g", h2f(host[N - 4 + i]));
  std::fprintf(stderr, "\n  min=%g max=%g all_finite=%d\n", min_v, max_v,
               all_finite ? 1 : 0);
  return all_finite;
}
} // namespace

bool Qwen3Forward::run_layer0_qkv_projection() {
  if (layer0_wq_.backing == nullptr || layer0_wk_.backing == nullptr ||
      layer0_wv_.backing == nullptr ||
      layer0_attn_norm_gamma_svm_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] qkv proj: weights or gamma not loaded\n");
    return false;
  }
  // M_pad to the v8c kernel tile (TM=4). Single-token forward → M=1.
  const unsigned int M = 1, M_pad = 4;
  const unsigned int K = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;

  cl_int err = CL_SUCCESS;
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  // (a) FC input: deterministic ramp pattern (same as step 2/3).
  const size_t in_bytes = (size_t)M_pad * K * sizeof(float);
  std::vector<float> in_host(M_pad * K, 0.0f);
  for (unsigned int k = 0; k < K; ++k)
    in_host[k] = 0.001f * static_cast<float>(k + 1);
  cl_mem in_buf = clCreateBuffer(cl_ctx_,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 in_bytes, in_host.data(), &err);
  cl_mem rmsnorm_out_buf =
    clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE, in_bytes, nullptr, &err);

  // (b) rmsnorm.cl on in_buf with SVM gamma. Single call; output shared
  //     by all three FCs (Q/K/V).
  {
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_kernel, "rmsnorm_cl");
    float eps = cfg_.rms_norm_eps;
    int H = static_cast<int>(M_pad), W = static_cast<int>(K);
    if (!kp ||
        !kp->SetKernelArguments(0, &in_buf, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &rmsnorm_out_buf, sizeof(cl_mem)) ||
        !kp->SetKernelSVMArguments(2, layer0_attn_norm_gamma_svm_) ||
        !kp->SetKernelArguments(3, &eps, sizeof(float)) ||
        !kp->SetKernelArguments(4, &H, sizeof(int)) ||
        !kp->SetKernelArguments(5, &W, sizeof(int))) {
      std::fprintf(stderr, "[qwen3-gpu] qkv proj: rmsnorm args failed\n");
      clReleaseMemObject(in_buf); clReleaseMemObject(rmsnorm_out_buf);
      return false;
    }
    std::array<size_t, 1> gws = {(size_t)M_pad * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // (c) Shared activation quantization (paper §3.6 fused-quant insight):
  //     quantize ONCE; reuse for all three FCs. act_image is also one-time.
  cl_mem act_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 (size_t)M_pad * K, nullptr, &err);
  cl_mem act_scale = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(float) * M_pad, nullptr, &err);
  cl_mem act_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 sizeof(int) * M_pad, nullptr, &err);
  cl_mem act_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 sizeof(int) * M_pad, nullptr, &err);
  if (err != CL_SUCCESS || !act_i8 || !act_scale || !act_zp || !act_rs) {
    std::fprintf(stderr, "[qwen3-gpu] qkv proj scratch alloc failed\n");
    return false;
  }
  nntrainer::quantize_act_v8c_fp32_cl(rmsnorm_out_buf, act_i8, act_scale,
                                      act_zp, act_rs, M_pad, K);

  cl_image_format afmt{CL_RGBA, CL_UNSIGNED_INT32};
  cl_image_desc adesc{};
  adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  adesc.image_width = K / 16;
  adesc.image_height = M_pad;
  adesc.image_row_pitch = K;
  adesc.buffer = act_i8;
  cl_mem act_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &adesc, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] qkv proj act image err=%d\n", err);
    return false;
  }

  // (d) Three GEMM dispatches. y_q [M_pad*N_q], y_k [M_pad*N_kv],
  //     y_v [M_pad*N_kv] — each will feed downstream attention.
  cl_mem y_q = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                              sizeof(uint16_t) * (size_t)M_pad * N_q,
                              nullptr, &err);
  cl_mem y_k = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                              sizeof(uint16_t) * (size_t)M_pad * N_kv,
                              nullptr, &err);
  cl_mem y_v = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                              sizeof(uint16_t) * (size_t)M_pad * N_kv,
                              nullptr, &err);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] qkv proj y_* alloc err=%d\n", err);
    return false;
  }

  nntrainer::gemm_int8_v8c_cl(act_image, layer0_wq_.weight_image, act_scale,
                              layer0_wq_.scale_buf, act_rs, act_zp,
                              layer0_wq_.row_sum_w_int4, y_q, M_pad, N_q, K);
  nntrainer::gemm_int8_v8c_cl(act_image, layer0_wk_.weight_image, act_scale,
                              layer0_wk_.scale_buf, act_rs, act_zp,
                              layer0_wk_.row_sum_w_int4, y_k, M_pad, N_kv, K);
  nntrainer::gemm_int8_v8c_cl(act_image, layer0_wv_.weight_image, act_scale,
                              layer0_wv_.scale_buf, act_rs, act_zp,
                              layer0_wv_.row_sum_w_int4, y_v, M_pad, N_kv, K);
  clFinish(cl_q_);

  // (e) Per-head q_norm / k_norm via rmsnorm_cl_fp16 in place. Q is
  //     reshaped [M=1, hQ*d] -> [1, 1, hQ, d] and normed per-head;
  //     same for K with hKV. V is unchanged (Qwen3 has no v_norm).
  //     Kernel signature: rmsnorm_cl_fp16(in, out, alpha, eps_half,
  //     B, C, H, W). For our case B=1, C=1, H=num_heads, W=head_dim.
  //     GWS = (B*C, H) = (1, num_heads); LWS = (1, 1) — no subgroup
  //     reqs in this kernel.
  if (layer0_q_norm_gamma_svm_fp16_ != nullptr &&
      layer0_k_norm_gamma_svm_fp16_ != nullptr) {
    auto dispatch_qk_norm =
      [&](cl_mem io_buf, void *gamma_svm, unsigned int num_heads,
          const char *tag) -> bool {
      auto kp = cl->registerClKernel(nntrainer::rmsnorm_fp16_kernel,
                                     "rmsnorm_cl_fp16");
      if (!kp) {
        std::fprintf(stderr,
                     "[qwen3-gpu] %s register rmsnorm_cl_fp16 failed\n", tag);
        return false;
      }
      uint16_t eps_h = f2h(cfg_.rms_norm_eps);
      int B = 1, C = 1;
      int H = static_cast<int>(num_heads),
          W = static_cast<int>(cfg_.head_dim);
      if (!kp->SetKernelArguments(0, &io_buf, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(1, &io_buf, sizeof(cl_mem)) || // in-place
          !kp->SetKernelSVMArguments(2, gamma_svm) ||
          !kp->SetKernelArguments(3, &eps_h, sizeof(uint16_t)) ||
          !kp->SetKernelArguments(4, &B, sizeof(int)) ||
          !kp->SetKernelArguments(5, &C, sizeof(int)) ||
          !kp->SetKernelArguments(6, &H, sizeof(int)) ||
          !kp->SetKernelArguments(7, &W, sizeof(int))) {
        std::fprintf(stderr, "[qwen3-gpu] %s rmsnorm_fp16 args failed\n", tag);
        return false;
      }
      std::array<size_t, 2> gws = {(size_t)B * C, (size_t)H};
      std::array<size_t, 2> lws = {1, 1};
      cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 2, gws.data(),
                                            lws.data(), 0, nullptr, nullptr);
      clFinish(cl_q_);
      return true;
    };
    if (!dispatch_qk_norm(y_q, layer0_q_norm_gamma_svm_fp16_,
                          cfg_.num_heads_Q, "q_norm") ||
        !dispatch_qk_norm(y_k, layer0_k_norm_gamma_svm_fp16_,
                          cfg_.num_heads_KV, "k_norm")) {
      // fall through to summarize so we can see partial state
    }
  }

  // (f) RoPE on Q and K (in place). Skipped when rope freqs haven't been
  //     precomputed for any position (precompute_rope_for_position
  //     wasn't called). V has no RoPE.
  if (layer0_rope_position_ >= 0) {
    if (!run_layer0_rope_on_qk(y_q, y_k)) {
      std::fprintf(stderr, "[qwen3-gpu] RoPE dispatch failed\n");
      // fall through to summarize
    } else {
      std::fprintf(stderr,
                   "[qwen3-gpu] RoPE applied at position=%d\n",
                   layer0_rope_position_);
    }
  }

  // (g) Sanity-check each output (M=0 valid row only). Q/K are
  //     post-q_norm/k_norm + optional RoPE; V is post-projection only.
  bool ok_q = summarize_fp16_buf(cl_q_, y_q, N_q,
                                 "Q (post q_norm + RoPE)");
  bool ok_k = summarize_fp16_buf(cl_q_, y_k, N_kv,
                                 "K (post k_norm + RoPE)");
  bool ok_v = summarize_fp16_buf(cl_q_, y_v, N_kv, "V");

  // (h) KV cache write + attention dispatch. Only runs if a position has
  //     been configured via precompute_rope_for_position AND the KV cache
  //     has been allocated. The cache write is row 0 of y_k / y_v
  //     (M_pad=4 but only row 0 is the valid token) into cache[position].
  //     Then dispatch the existing two_conv_attention_prefill_f16_cl
  //     kernel with svm_inputs=true — first time in the codebase this
  //     path runs in production because the existing CausalLM doesn't
  //     SVM-allocate its KV cache.
  bool ok_attn = true;
  if (layer0_cache_k_svm_ != nullptr && layer0_cache_v_svm_ != nullptr &&
      layer0_rope_position_ >= 0) {
    const unsigned int pos = (unsigned int)layer0_rope_position_;
    const size_t kv_row_elts = (size_t)N_kv;          // hKV * d
    const size_t kv_row_bytes = kv_row_elts * sizeof(uint16_t);
    const size_t q_row_bytes  = (size_t)N_q * sizeof(uint16_t);

    // Copy y_k row 0 -> cache_K[pos], y_v row 0 -> cache_V[pos] via
    // map-source + memcpy-into-SVM-region. Both src and dst are tiny
    // (~2 KB each for our 0.6B config) so the host round-trip is
    // negligible. A pure-GPU path (copy kernel) is a follow-up.
    auto map_src = [&](cl_mem src) -> void * {
      cl_int err;
      void *p = clEnqueueMapBuffer(cl_q_, src, CL_TRUE, CL_MAP_READ, 0,
                                   std::max(q_row_bytes, kv_row_bytes), 0,
                                   nullptr, nullptr, &err);
      return err == CL_SUCCESS ? p : nullptr;
    };
    auto write_into_svm_region = [&](void *svm_base, size_t offset_bytes,
                                     const void *src_host, size_t bytes,
                                     const char *tag) -> bool {
      void *dst = static_cast<uint8_t *>(svm_base) + offset_bytes;
      cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, dst, bytes,
                                   0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[qwen3-gpu] %s SVMMap WRITE err=%d\n", tag,
                     err);
        return false;
      }
      std::memcpy(dst, src_host, bytes);
      err = clEnqueueSVMUnmap(cl_q_, dst, 0, nullptr, nullptr);
      return err == CL_SUCCESS;
    };

    void *p_k = map_src(y_k);
    if (p_k) {
      write_into_svm_region(layer0_cache_k_svm_, pos * kv_row_bytes, p_k,
                            kv_row_bytes, "cache_K");
      clEnqueueUnmapMemObject(cl_q_, y_k, p_k, 0, nullptr, nullptr);
    }
    void *p_v = map_src(y_v);
    if (p_v) {
      write_into_svm_region(layer0_cache_v_svm_, pos * kv_row_bytes, p_v,
                            kv_row_bytes, "cache_V");
      clEnqueueUnmapMemObject(cl_q_, y_v, p_v, 0, nullptr, nullptr);
    }
    clFinish(cl_q_);
    std::fprintf(stderr,
                 "[qwen3-gpu] wrote K/V at cache position %u\n", pos);

    // Allocate Q SVM + O SVM (per-dispatch scratch) and copy Q row 0.
    void *q_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, q_row_bytes, 0);
    void *o_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, q_row_bytes, 0);
    void *p_q = map_src(y_q);
    if (q_svm && o_svm && p_q) {
      write_into_svm_region(q_svm, 0, p_q, q_row_bytes, "Q_svm");
      clEnqueueUnmapMemObject(cl_q_, y_q, p_q, 0, nullptr, nullptr);
      clFinish(cl_q_);

      // Dispatch attention. M=1 query, N_kv = pos + 1 (all positions 0..pos
      // in cache; with our setup pos=0 means N_kv=1 → degenerate single-
      // token attention where softmax of one element is 1.0, so output
      // per head_q is exactly V[head_q / gqa] for this token. Easy to
      // sanity-check vs the post-projection V values.
      const unsigned int N_kv_cache = pos + 1;
      bool ok = nntrainer::two_conv_attention_prefill_f16_cl(
        static_cast<const uint16_t *>(q_svm),
        static_cast<const uint16_t *>(layer0_cache_k_svm_),
        static_cast<const uint16_t *>(layer0_cache_v_svm_),
        static_cast<uint16_t *>(o_svm),
        /*M=*/1, /*N_kv=*/N_kv_cache,
        cfg_.num_heads_Q, cfg_.num_heads_KV, cfg_.head_dim,
        /*causal=*/true, /*svm_inputs=*/true);
      std::fprintf(stderr,
                   "[qwen3-gpu] two_conv_attention_prefill_f16_cl returned "
                   "%d (M=1, N_kv=%u)\n",
                   (int)ok, N_kv_cache);

      if (ok) {
        // Verify: for M=1, N_kv=1, output per head_q ≈ V[head_q / gqa].
        // Print attention output + sample expected V per head.
        cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_READ, o_svm,
                                     q_row_bytes, 0, nullptr, nullptr);
        if (err == CL_SUCCESS) {
          const uint16_t *o = static_cast<const uint16_t *>(o_svm);
          bool finite = true;
          float min_v = std::numeric_limits<float>::infinity();
          float max_v = -std::numeric_limits<float>::infinity();
          for (unsigned int i = 0; i < N_q; ++i) {
            float f = h2f(o[i]);
            if (!std::isfinite(f)) finite = false;
            if (f < min_v) min_v = f;
            if (f > max_v) max_v = f;
          }
          std::fprintf(stderr,
                       "[qwen3-gpu] Attention output (fp16, N=%u) first 8:",
                       N_q);
          for (int i = 0; i < 8; ++i)
            std::fprintf(stderr, " %g", h2f(o[i]));
          std::fprintf(stderr, "\n  last 4:");
          for (int i = 0; i < 4; ++i)
            std::fprintf(stderr, " %g", h2f(o[N_q - 4 + i]));
          std::fprintf(stderr,
                       "\n  min=%g max=%g all_finite=%d\n",
                       min_v, max_v, finite ? 1 : 0);

          // For N_kv=1 each output[hq, :d] ≈ V[hq/gqa, :d]. Spot-check
          // hq=0 (gqa group 0) against V[0, :d] — both should be the
          // same fp16 bit pattern.
          if (N_kv_cache == 1) {
            void *p_v_again = map_src(y_v);
            if (p_v_again) {
              const uint16_t *v_host =
                static_cast<const uint16_t *>(p_v_again);
              std::fprintf(stderr,
                           "  V[head_kv=0] first 8 ref:");
              for (int i = 0; i < 8; ++i)
                std::fprintf(stderr, " %g", h2f(v_host[i]));
              std::fprintf(stderr, "\n  O[head_q=0] first 8 actual:");
              for (int i = 0; i < 8; ++i)
                std::fprintf(stderr, " %g", h2f(o[i]));
              std::fprintf(stderr, "\n");
              clEnqueueUnmapMemObject(cl_q_, y_v, p_v_again, 0, nullptr,
                                      nullptr);
            }
          }
          clEnqueueSVMUnmap(cl_q_, o_svm, 0, nullptr, nullptr);
          ok_attn = finite;
        } else {
          ok_attn = false;
        }
      } else {
        ok_attn = false;
      }
    }

    // (i) wo (attention output projection): O_svm (fp16) -> wo_out fp16
    //     -> residual_1 = x + wo_out (fp32). Inline cvt_h2f + add_fp32
    //     bridge keeps the residual stream in fp32 (matches the existing
    //     rmsnorm.cl dtype). x is in_buf (fp32 ramp pattern).
    if (ok_attn && layer0_wo_.backing != nullptr && o_svm != nullptr) {
      const unsigned int wo_K = N_q;             // hQ * d = 2048
      const unsigned int wo_N = cfg_.hidden_size; // 1024
      // (i.1) Convert attention output O_svm (fp16, [M_pad*hQ*d]) into
      //       a fresh cl_mem fp32 buffer so quantize_act_v8c_fp32_cl
      //       can consume it. Only M=1 valid row is meaningful; we
      //       still allocate M_pad rows of padding (v8c kernel tile).
      const size_t o_fp32_bytes =
        (size_t)M_pad * wo_K * sizeof(float);
      cl_mem o_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                     o_fp32_bytes, nullptr, &err);
      // Pre-zero padded rows.
      float zero = 0.0f;
      clEnqueueFillBuffer(cl_q_, o_fp32, &zero, sizeof(float), 0,
                          o_fp32_bytes, 0, nullptr, nullptr);
      {
        auto kp = cl->registerClKernel(kConvertFp16ToFp32Kernel,
                                       "cvt_h2f");
        int n = (int)wo_K;
        if (!kp ||
            !kp->SetKernelSVMArguments(0, o_svm) ||
            !kp->SetKernelArguments(1, &o_fp32, sizeof(cl_mem)) ||
            !kp->SetKernelArguments(2, &n, sizeof(int))) {
          std::fprintf(stderr,
                       "[qwen3-gpu] wo cvt_h2f args failed\n");
          ok_attn = false;
        } else {
          std::array<size_t, 1> gws = {(size_t)wo_K};
          std::array<size_t, 1> lws = {64};
          // pad gws[0] up to multiple of lws[0]
          gws[0] = ((gws[0] + lws[0] - 1) / lws[0]) * lws[0];
          cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1,
                                                gws.data(), lws.data(), 0,
                                                nullptr, nullptr);
          clFinish(cl_q_);
        }
      }
      // (i.2) v8c FC on (M_pad=4, K=2048) input -> (M_pad, N=1024)
      //       output. Fresh scratch (separate from QKV's act_i8 etc
      //       because K is different).
      cl_mem wo_act_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                        (size_t)M_pad * wo_K, nullptr, &err);
      cl_mem wo_act_scale = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                           sizeof(float) * M_pad, nullptr,
                                           &err);
      cl_mem wo_act_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                        sizeof(int) * M_pad, nullptr, &err);
      cl_mem wo_act_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                        sizeof(int) * M_pad, nullptr, &err);
      cl_mem wo_y_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                        sizeof(uint16_t) * (size_t)M_pad *
                                          wo_N,
                                        nullptr, &err);
      nntrainer::quantize_act_v8c_fp32_cl(o_fp32, wo_act_i8, wo_act_scale,
                                          wo_act_zp, wo_act_rs, M_pad, wo_K);
      cl_image_format wo_afmt{CL_RGBA, CL_UNSIGNED_INT32};
      cl_image_desc wo_adesc{};
      wo_adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
      wo_adesc.image_width = wo_K / 16;
      wo_adesc.image_height = M_pad;
      wo_adesc.image_row_pitch = wo_K;
      wo_adesc.buffer = wo_act_i8;
      cl_mem wo_act_image = clCreateImage(cl_ctx_, CL_MEM_READ_ONLY,
                                          &wo_afmt, &wo_adesc, nullptr,
                                          &err);
      nntrainer::gemm_int8_v8c_cl(wo_act_image, layer0_wo_.weight_image,
                                  wo_act_scale, layer0_wo_.scale_buf,
                                  wo_act_rs, wo_act_zp,
                                  layer0_wo_.row_sum_w_int4, wo_y_fp16,
                                  M_pad, wo_N, wo_K);
      clFinish(cl_q_);
      bool ok_wo = summarize_fp16_buf(cl_q_, wo_y_fp16, wo_N, "wo (att->h)");
      // (i.3) Convert wo_out fp16 -> fp32, then add to x (in_buf) to form
      //       the post-attention residual. Allocate the persistent
      //       residual_1 cl_mem on first use.
      cl_mem wo_out_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                          (size_t)wo_N * sizeof(float),
                                          nullptr, &err);
      {
        auto kp = cl->registerClKernel(kConvertFp16ToFp32Kernel, "cvt_h2f");
        int n = (int)wo_N;
        kp->SetKernelArguments(0, &wo_y_fp16, sizeof(cl_mem));
        kp->SetKernelArguments(1, &wo_out_fp32, sizeof(cl_mem));
        kp->SetKernelArguments(2, &n, sizeof(int));
        std::array<size_t, 1> gws = {(size_t)wo_N};
        std::array<size_t, 1> lws = {64};
        gws[0] = ((gws[0] + lws[0] - 1) / lws[0]) * lws[0];
        cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                              lws.data(), 0, nullptr,
                                              nullptr);
        clFinish(cl_q_);
      }
      if (layer0_residual1_fp32_ == nullptr) {
        layer0_residual1_fp32_ =
          clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                         (size_t)wo_N * sizeof(float), nullptr, &err);
      }
      {
        auto kp = cl->registerClKernel(kAddFp32Kernel, "add_fp32");
        int n = (int)wo_N;
        kp->SetKernelArguments(0, &in_buf, sizeof(cl_mem));
        kp->SetKernelArguments(1, &wo_out_fp32, sizeof(cl_mem));
        kp->SetKernelArguments(2, &layer0_residual1_fp32_, sizeof(cl_mem));
        kp->SetKernelArguments(3, &n, sizeof(int));
        std::array<size_t, 1> gws = {(size_t)wo_N};
        std::array<size_t, 1> lws = {64};
        gws[0] = ((gws[0] + lws[0] - 1) / lws[0]) * lws[0];
        cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                              lws.data(), 0, nullptr,
                                              nullptr);
        clFinish(cl_q_);
      }
      // Custom fp32 summary for residual_1 (post wo + residual add).
      {
        std::vector<float> r(wo_N);
        clEnqueueReadBuffer(cl_q_, layer0_residual1_fp32_, CL_TRUE, 0,
                            (size_t)wo_N * sizeof(float), r.data(), 0,
                            nullptr, nullptr);
        bool finite = true;
        float mn = std::numeric_limits<float>::infinity();
        float mx = -mn;
        for (unsigned int i = 0; i < wo_N; ++i) {
          if (!std::isfinite(r[i])) finite = false;
          if (r[i] < mn) mn = r[i];
          if (r[i] > mx) mx = r[i];
        }
        std::fprintf(stderr,
                     "[qwen3-gpu] residual_1 fp32 N=%u first 8:", wo_N);
        for (int i = 0; i < 8; ++i)
          std::fprintf(stderr, " %g", r[i]);
        std::fprintf(stderr, "\n  last 4:");
        for (int i = 0; i < 4; ++i)
          std::fprintf(stderr, " %g", r[wo_N - 4 + i]);
        std::fprintf(stderr,
                     "\n  min=%g max=%g all_finite=%d\n", mn, mx, finite);
        ok_attn = ok_attn && ok_wo && finite;
      }
      clReleaseMemObject(wo_out_fp32);
      clReleaseMemObject(wo_act_image);
      clReleaseMemObject(wo_y_fp16);
      clReleaseMemObject(wo_act_rs);
      clReleaseMemObject(wo_act_zp);
      clReleaseMemObject(wo_act_scale);
      clReleaseMemObject(wo_act_i8);
      clReleaseMemObject(o_fp32);
    }

    if (q_svm) clSVMFree(cl_ctx_, q_svm);
    if (o_svm) clSVMFree(cl_ctx_, o_svm);
  }

  clReleaseMemObject(y_v);
  clReleaseMemObject(y_k);
  clReleaseMemObject(y_q);
  clReleaseMemObject(act_image);
  clReleaseMemObject(act_rs);
  clReleaseMemObject(act_zp);
  clReleaseMemObject(act_scale);
  clReleaseMemObject(act_i8);
  clReleaseMemObject(rmsnorm_out_buf);
  clReleaseMemObject(in_buf);
  return ok_q && ok_k && ok_v && ok_attn;
}

// Inline fp16 RoPE kernel for the GPU-native runtime. Operates in place
// on a [num_heads, head_dim] fp16 buffer. cos/sin tables are
// pre-doubled (cos[k+half] = cos[k], sin[k+half] = sin[k]) to match the
// CPU mha_core convention. Math: for k in [0, half):
//   x_lo = xy[h, k]; x_hi = xy[h, k + half]
//   xy[h, k]        = x_lo * cos[k] - x_hi * sin[k]
//   xy[h, k + half] = x_hi * cos[k] + x_lo * sin[k]
// (compute_rotary_emb_value's transformed_value sign convention).
// GWS = (num_heads, half); one WI per rotation pair.
bool Qwen3Forward::precompute_rope_for_position(unsigned int position) {
  if (cl_ctx_ == nullptr) return false;
  const unsigned int d = cfg_.head_dim;
  const unsigned int half = d / 2;
  const size_t tbl_bytes = (size_t)d * sizeof(uint16_t);

  // (Re)allocate SVM if first time.
  auto ensure_svm = [this, tbl_bytes](void **dst) -> bool {
    if (*dst != nullptr) return true;
    *dst = clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, tbl_bytes, 0);
    if (*dst == nullptr) {
      std::fprintf(stderr, "[qwen3-gpu] rope SVMAlloc failed\n");
      return false;
    }
    return true;
  };
  if (!ensure_svm(&layer0_rope_cos_svm_fp16_) ||
      !ensure_svm(&layer0_rope_sin_svm_fp16_))
    return false;

  // Compute thetas[j] = theta ^ (-2j/d) for j in [0, half), then for
  // this position, cos/sin per "doubled half" layout.
  std::vector<float> cos_tmp(d), sin_tmp(d);
  for (unsigned int j = 0; j < half; ++j) {
    float exponent = -2.0f * static_cast<float>(j) / static_cast<float>(d);
    float theta_j = std::pow(cfg_.rope_theta, exponent);
    float angle = static_cast<float>(position) * theta_j;
    float c = std::cos(angle);
    float s = std::sin(angle);
    cos_tmp[j]        = c;
    cos_tmp[j + half] = c;
    sin_tmp[j]        = s;
    sin_tmp[j + half] = s;
  }

  // Push (converted to fp16) into SVM.
  auto write_svm = [&](void *dst, const std::vector<float> &src,
                       const char *tag) -> bool {
    cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, dst,
                                 tbl_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::fprintf(stderr, "[qwen3-gpu] rope %s SVMMap WRITE err=%d\n",
                   tag, err);
      return false;
    }
    uint16_t *p = static_cast<uint16_t *>(dst);
    for (unsigned int i = 0; i < d; ++i) p[i] = f2h(src[i]);
    err = clEnqueueSVMUnmap(cl_q_, dst, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;
    clFinish(cl_q_);
    return true;
  };
  if (!write_svm(layer0_rope_cos_svm_fp16_, cos_tmp, "cos")) return false;
  if (!write_svm(layer0_rope_sin_svm_fp16_, sin_tmp, "sin")) return false;
  layer0_rope_position_ = static_cast<int>(position);
  std::fprintf(stderr,
               "[qwen3-gpu] RoPE freqs precomputed for position=%u: "
               "cos[0..3] = %g %g %g %g, sin[0..3] = %g %g %g %g\n",
               position, cos_tmp[0], cos_tmp[1], cos_tmp[2], cos_tmp[3],
               sin_tmp[0], sin_tmp[1], sin_tmp[2], sin_tmp[3]);
  return true;
}

bool Qwen3Forward::run_layer0_rope_on_qk(cl_mem y_q, cl_mem y_k) {
  if (layer0_rope_cos_svm_fp16_ == nullptr ||
      layer0_rope_sin_svm_fp16_ == nullptr || layer0_rope_position_ < 0) {
    std::fprintf(stderr,
                 "[qwen3-gpu] run_layer0_rope_on_qk: freqs not loaded\n");
    return false;
  }
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  auto kp = cl->registerClKernel(kRopeFp16Kernel, "rope_fp16");
  if (!kp) {
    std::fprintf(stderr, "[qwen3-gpu] rope_fp16 register failed\n");
    return false;
  }
  auto dispatch_one = [&](cl_mem io, unsigned int num_heads,
                          const char *tag) -> bool {
    int nh = static_cast<int>(num_heads);
    int half_d = static_cast<int>(cfg_.head_dim / 2);
    if (!kp->SetKernelArguments(0, &io, sizeof(cl_mem)) ||
        !kp->SetKernelSVMArguments(1, layer0_rope_cos_svm_fp16_) ||
        !kp->SetKernelSVMArguments(2, layer0_rope_sin_svm_fp16_) ||
        !kp->SetKernelArguments(3, &nh, sizeof(int)) ||
        !kp->SetKernelArguments(4, &half_d, sizeof(int))) {
      std::fprintf(stderr, "[qwen3-gpu] %s rope args failed\n", tag);
      return false;
    }
    std::array<size_t, 2> gws = {(size_t)num_heads,
                                 (size_t)(cfg_.head_dim / 2)};
    std::array<size_t, 2> lws = {1, 1};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 2, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    return true;
  };
  if (!dispatch_one(y_q, cfg_.num_heads_Q, "Q")) return false;
  if (!dispatch_one(y_k, cfg_.num_heads_KV, "K")) return false;
  clFinish(cl_q_);
  return true;
}

bool Qwen3Forward::precompute_rope_full_lut(unsigned int max_positions) {
  if (max_positions == 0 || cl_ctx_ == nullptr) return false;
  if (rope_cos_full_svm_ != nullptr &&
      rope_full_max_positions_ >= max_positions) {
    return true;
  }
  const unsigned int d = cfg_.head_dim;
  if (d % 2 != 0) return false;
  const unsigned int half = d / 2;
  const size_t bytes =
    (size_t)max_positions * (size_t)half * sizeof(uint16_t);

  if (rope_cos_full_svm_ != nullptr) {
    clSVMFree(cl_ctx_, rope_cos_full_svm_);
    rope_cos_full_svm_ = nullptr;
  }
  if (rope_sin_full_svm_ != nullptr) {
    clSVMFree(cl_ctx_, rope_sin_full_svm_);
    rope_sin_full_svm_ = nullptr;
  }
  rope_cos_full_svm_ = clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, bytes, 0);
  rope_sin_full_svm_ = clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, bytes, 0);
  if (rope_cos_full_svm_ == nullptr || rope_sin_full_svm_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] precompute_rope_full_lut: SVMAlloc(%zu) failed\n",
                 bytes);
    return false;
  }

  // Map both for write, fill, unmap.
  cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE,
                               rope_cos_full_svm_, bytes, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) return false;
  err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, rope_sin_full_svm_,
                        bytes, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) return false;
  uint16_t *pc = static_cast<uint16_t *>(rope_cos_full_svm_);
  uint16_t *ps = static_cast<uint16_t *>(rope_sin_full_svm_);
  for (unsigned int j = 0; j < half; ++j) {
    float exponent = -2.0f * (float)j / (float)d;
    float theta_j = std::pow(cfg_.rope_theta, exponent);
    for (unsigned int pos = 0; pos < max_positions; ++pos) {
      float angle = (float)pos * theta_j;
      pc[(size_t)pos * half + j] = f2h(std::cos(angle));
      ps[(size_t)pos * half + j] = f2h(std::sin(angle));
    }
  }
  clEnqueueSVMUnmap(cl_q_, rope_cos_full_svm_, 0, nullptr, nullptr);
  clEnqueueSVMUnmap(cl_q_, rope_sin_full_svm_, 0, nullptr, nullptr);
  clFinish(cl_q_);
  rope_full_max_positions_ = max_positions;
  std::fprintf(stderr,
               "[qwen3-gpu] RoPE full LUT built: %u positions × %u half_d "
               "(%.2f MB)\n",
               max_positions, half, (double)(2 * bytes) / (1024.0 * 1024.0));
  return true;
}

bool Qwen3Forward::dispatch_rope_batched(cl_mem io, unsigned int M,
                                         unsigned int num_heads,
                                         unsigned int start_pos) {
  if (io == nullptr || M == 0 || num_heads == 0) return true;
  if (rope_cos_full_svm_ == nullptr || rope_sin_full_svm_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] dispatch_rope_batched: LUT not built\n");
    return false;
  }
  if (start_pos + M > rope_full_max_positions_) {
    std::fprintf(stderr,
                 "[qwen3-gpu] dispatch_rope_batched: start+M=%u > LUT %u\n",
                 start_pos + M, rope_full_max_positions_);
    return false;
  }
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  auto kp =
    cl->registerClKernel(kRopeFp16BatchedKernel, "rope_fp16_batched");
  if (!kp) {
    std::fprintf(stderr,
                 "[qwen3-gpu] rope_fp16_batched register failed\n");
    return false;
  }
  int Mi = (int)M;
  int nh = (int)num_heads;
  int half_d = (int)(cfg_.head_dim / 2);
  int sp = (int)start_pos;
  if (!kp->SetKernelArguments(0, &io, sizeof(cl_mem)) ||
      !kp->SetKernelSVMArguments(1, rope_cos_full_svm_) ||
      !kp->SetKernelSVMArguments(2, rope_sin_full_svm_) ||
      !kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(4, &nh, sizeof(int)) ||
      !kp->SetKernelArguments(5, &half_d, sizeof(int)) ||
      !kp->SetKernelArguments(6, &sp, sizeof(int))) {
    std::fprintf(stderr,
                 "[qwen3-gpu] rope_fp16_batched arg setup failed\n");
    return false;
  }
  // Pick lws covering the half_d direction (small: 64 at d=128 / 2 = 64),
  // tile M and heads on the outer dims.
  constexpr size_t LWS_K = 64;
  constexpr size_t LWS_H = 1;
  constexpr size_t LWS_T = 1;
  const size_t gws_k = ((size_t)half_d + LWS_K - 1) / LWS_K * LWS_K;
  const size_t gws_h = (size_t)nh;
  const size_t gws_t = (size_t)Mi;
  std::array<size_t, 3> gws = {gws_t, gws_h, gws_k};
  std::array<size_t, 3> lws = {LWS_T, LWS_H, LWS_K};
  cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                        lws.data(), 0, nullptr, nullptr);
  return true;
}

bool Qwen3Forward::load_layer0_ffn_weights() {
  // After wo there are NO weights for decoder_add. ffn_norm gamma is
  // the next slab, then ffn_up, ffn_gate, ffn_down.
  const size_t embed_bytes = embed_table_bytes();
  const size_t attn_norm_bytes =
    static_cast<size_t>(cfg_.hidden_size) * sizeof(float);
  const size_t head_dim_bytes =
    static_cast<size_t>(cfg_.head_dim) * sizeof(float);
  const unsigned int K_h = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;
  const size_t wo_K = N_q;
  const size_t wo_N = K_h;

  const size_t wq_off = embed_bytes + attn_norm_bytes;
  const size_t wq_bytes = qint4_record_bytes(wq_off, K_h, N_q);
  const size_t wk_off = wq_off + wq_bytes + head_dim_bytes;
  const size_t wk_bytes = qint4_record_bytes(wk_off, K_h, N_kv);
  const size_t wv_off = wk_off + wk_bytes + head_dim_bytes;
  const size_t wv_bytes = qint4_record_bytes(wv_off, K_h, N_kv);
  const size_t wo_off = wv_off + wv_bytes;
  const size_t wo_bytes = qint4_record_bytes(wo_off, wo_K, wo_N);
  const size_t ffn_norm_off = wo_off + wo_bytes;
  // ffn_norm gamma fp32 [hidden] = hidden * 4 bytes
  const size_t ffn_norm_bytes =
    (size_t)cfg_.hidden_size * sizeof(float);
  // ffn_up: K=hidden, N=intermediate
  const unsigned int up_K = cfg_.hidden_size;
  const unsigned int up_N = cfg_.intermediate_size;
  // ffn_down: K=intermediate, N=hidden
  const unsigned int dn_K = cfg_.intermediate_size;
  const unsigned int dn_N = cfg_.hidden_size;

  const size_t ffn_up_off = ffn_norm_off + ffn_norm_bytes;
  const size_t up_bytes = qint4_record_bytes(ffn_up_off, up_K, up_N);
  const size_t ffn_gate_off = ffn_up_off + up_bytes;
  const size_t gate_bytes = qint4_record_bytes(ffn_gate_off, up_K, up_N);
  const size_t ffn_down_off = ffn_gate_off + gate_bytes;

  // (a) ffn_norm gamma load to SVM (fp32 path matches rmsnorm.cl).
  if (layer0_ffn_norm_gamma_svm_ == nullptr) {
    if (ffn_norm_off + ffn_norm_bytes > weight_bytes_) {
      std::fprintf(stderr,
                   "[qwen3-gpu] ffn_norm gamma offset out of range\n");
      return false;
    }
    const float *gp =
      reinterpret_cast<const float *>(weight_mmap_ + ffn_norm_off);
    std::fprintf(stderr,
                 "[qwen3-gpu] ffn_norm off=%zu (~%zu KB) first 8:",
                 ffn_norm_off, ffn_norm_off / 1024);
    for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %g", gp[i]);
    std::fprintf(stderr, "\n");
    layer0_ffn_norm_gamma_svm_ =
      clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, ffn_norm_bytes, 0);
    if (layer0_ffn_norm_gamma_svm_ == nullptr) {
      std::fprintf(stderr, "[qwen3-gpu] ffn_norm SVMAlloc failed\n");
      return false;
    }
    cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE,
                                 layer0_ffn_norm_gamma_svm_, ffn_norm_bytes,
                                 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;
    std::memcpy(layer0_ffn_norm_gamma_svm_, gp, ffn_norm_bytes);
    clEnqueueSVMUnmap(cl_q_, layer0_ffn_norm_gamma_svm_, 0, nullptr,
                      nullptr);
    clFinish(cl_q_);
  }

  // (b) Three QINT4 FCs: ffn_up, ffn_gate, ffn_down.
  return load_qint4_weight_at(ffn_up_off, up_K, up_N, &layer0_ffn_up_,
                              "ffn_up") &&
         load_qint4_weight_at(ffn_gate_off, up_K, up_N, &layer0_ffn_gate_,
                              "ffn_gate") &&
         load_qint4_weight_at(ffn_down_off, dn_K, dn_N, &layer0_ffn_down_,
                              "ffn_down");
}

bool Qwen3Forward::run_layer0_ffn() {
  if (layer0_residual1_fp32_ == nullptr ||
      layer0_ffn_norm_gamma_svm_ == nullptr ||
      layer0_ffn_up_.backing == nullptr ||
      layer0_ffn_gate_.backing == nullptr ||
      layer0_ffn_down_.backing == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] run_layer0_ffn: residual_1 or weights not "
                 "loaded\n");
    return false;
  }
  cl_int err = CL_SUCCESS;
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  const unsigned int M_pad = 4;
  const unsigned int K_h = cfg_.hidden_size;        // 1024
  const unsigned int I = cfg_.intermediate_size;    // 3072

  // (a) ffn_norm.cl on residual_1 (fp32 [K_h]) -> ffn_normed (fp32).
  //     Need M_pad rows for the v8c quantize/GEMM tile alignment, so
  //     allocate a [M_pad * K_h] buffer with residual_1 in row 0 and
  //     zeros elsewhere. residual_1_fp32_ is [K_h]; clEnqueueCopyBuffer
  //     into row 0.
  const size_t row_h_bytes = (size_t)K_h * sizeof(float);
  cl_mem ffn_in_padded =
    clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                   (size_t)M_pad * row_h_bytes, nullptr, &err);
  float zero = 0.0f;
  clEnqueueFillBuffer(cl_q_, ffn_in_padded, &zero, sizeof(float), 0,
                      (size_t)M_pad * row_h_bytes, 0, nullptr, nullptr);
  clEnqueueCopyBuffer(cl_q_, layer0_residual1_fp32_, ffn_in_padded, 0, 0,
                      row_h_bytes, 0, nullptr, nullptr);
  cl_mem ffn_normed = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                     (size_t)M_pad * row_h_bytes, nullptr,
                                     &err);
  {
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_kernel, "rmsnorm_cl");
    float eps = cfg_.rms_norm_eps;
    int H = (int)M_pad, W = (int)K_h;
    if (!kp ||
        !kp->SetKernelArguments(0, &ffn_in_padded, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &ffn_normed, sizeof(cl_mem)) ||
        !kp->SetKernelSVMArguments(2, layer0_ffn_norm_gamma_svm_) ||
        !kp->SetKernelArguments(3, &eps, sizeof(float)) ||
        !kp->SetKernelArguments(4, &H, sizeof(int)) ||
        !kp->SetKernelArguments(5, &W, sizeof(int))) {
      std::fprintf(stderr, "[qwen3-gpu] ffn rmsnorm args failed\n");
      return false;
    }
    std::array<size_t, 1> gws = {(size_t)M_pad * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // (b) Shared activation quant for ffn_up + ffn_gate.
  cl_mem act_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 (size_t)M_pad * K_h, nullptr, &err);
  cl_mem act_scale = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(float) * M_pad, nullptr, &err);
  cl_mem act_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 sizeof(int) * M_pad, nullptr, &err);
  cl_mem act_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 sizeof(int) * M_pad, nullptr, &err);
  nntrainer::quantize_act_v8c_fp32_cl(ffn_normed, act_i8, act_scale, act_zp,
                                      act_rs, M_pad, K_h);
  cl_image_format afmt{CL_RGBA, CL_UNSIGNED_INT32};
  cl_image_desc adesc{};
  adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  adesc.image_width = K_h / 16;
  adesc.image_height = M_pad;
  adesc.image_row_pitch = K_h;
  adesc.buffer = act_i8;
  cl_mem act_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &adesc, nullptr, &err);

  // (c) GEMM ffn_up -> up_fp16, GEMM ffn_gate -> gate_fp16.
  cl_mem up_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  sizeof(uint16_t) * (size_t)M_pad * I,
                                  nullptr, &err);
  cl_mem gate_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(uint16_t) * (size_t)M_pad * I,
                                    nullptr, &err);
  nntrainer::gemm_int8_v8c_cl(act_image, layer0_ffn_up_.weight_image,
                              act_scale, layer0_ffn_up_.scale_buf, act_rs,
                              act_zp, layer0_ffn_up_.row_sum_w_int4,
                              up_fp16, M_pad, I, K_h);
  nntrainer::gemm_int8_v8c_cl(act_image, layer0_ffn_gate_.weight_image,
                              act_scale, layer0_ffn_gate_.scale_buf, act_rs,
                              act_zp, layer0_ffn_gate_.row_sum_w_int4,
                              gate_fp16, M_pad, I, K_h);
  clFinish(cl_q_);

  // (d) Convert up/gate fp16 -> fp32 (for swiglu in fp32) and swiglu.
  cl_mem up_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  (size_t)I * sizeof(float), nullptr, &err);
  cl_mem gate_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    (size_t)I * sizeof(float), nullptr,
                                    &err);
  auto dispatch_cvt = [&](cl_mem in_fp16, cl_mem out_fp32, unsigned int n,
                          const char *tag) -> bool {
    auto kp = cl->registerClKernel(kConvertFp16ToFp32Kernel, "cvt_h2f");
    int ni = (int)n;
    if (!kp ||
        !kp->SetKernelArguments(0, &in_fp16, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &out_fp32, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(2, &ni, sizeof(int))) {
      std::fprintf(stderr, "[qwen3-gpu] ffn %s cvt args failed\n", tag);
      return false;
    }
    std::array<size_t, 1> gws = {((size_t)n + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    return true;
  };
  dispatch_cvt(up_fp16, up_fp32, I, "up");
  dispatch_cvt(gate_fp16, gate_fp32, I, "gate");
  clFinish(cl_q_);

  cl_mem swiglu_out = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                     (size_t)M_pad * I * sizeof(float),
                                     nullptr, &err);
  clEnqueueFillBuffer(cl_q_, swiglu_out, &zero, sizeof(float), 0,
                      (size_t)M_pad * I * sizeof(float), 0, nullptr, nullptr);
  {
    auto kp = cl->registerClKernel(kSwigluFp32Kernel, "swiglu_fp32");
    int n = (int)I;
    kp->SetKernelArguments(0, &gate_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(1, &up_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &swiglu_out, sizeof(cl_mem));
    kp->SetKernelArguments(3, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)I + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // (e) quantize swiglu_out + v8c(ffn_down) -> ffn_down_out fp16.
  cl_mem dn_act_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    (size_t)M_pad * I, nullptr, &err);
  cl_mem dn_act_scale = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                       sizeof(float) * M_pad, nullptr, &err);
  cl_mem dn_act_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(int) * M_pad, nullptr, &err);
  cl_mem dn_act_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(int) * M_pad, nullptr, &err);
  nntrainer::quantize_act_v8c_fp32_cl(swiglu_out, dn_act_i8, dn_act_scale,
                                      dn_act_zp, dn_act_rs, M_pad, I);
  cl_image_desc dn_adesc{};
  dn_adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  dn_adesc.image_width = I / 16;
  dn_adesc.image_height = M_pad;
  dn_adesc.image_row_pitch = I;
  dn_adesc.buffer = dn_act_i8;
  cl_mem dn_act_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &dn_adesc, nullptr,
                  &err);
  cl_mem dn_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  sizeof(uint16_t) * (size_t)M_pad * K_h,
                                  nullptr, &err);
  nntrainer::gemm_int8_v8c_cl(dn_act_image, layer0_ffn_down_.weight_image,
                              dn_act_scale, layer0_ffn_down_.scale_buf,
                              dn_act_rs, dn_act_zp,
                              layer0_ffn_down_.row_sum_w_int4, dn_fp16,
                              M_pad, K_h, I);
  clFinish(cl_q_);
  summarize_fp16_buf(cl_q_, dn_fp16, K_h, "ffn_down");

  // (f) cvt dn_fp16 -> fp32, then residual_2 = residual_1 + dn_fp32.
  cl_mem dn_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  (size_t)K_h * sizeof(float), nullptr,
                                  &err);
  dispatch_cvt(dn_fp16, dn_fp32, K_h, "dn");
  clFinish(cl_q_);

  if (layer0_output_fp32_ == nullptr) {
    layer0_output_fp32_ = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                         (size_t)K_h * sizeof(float),
                                         nullptr, &err);
  }
  {
    auto kp = cl->registerClKernel(kAddFp32Kernel, "add_fp32");
    int n = (int)K_h;
    kp->SetKernelArguments(0, &layer0_residual1_fp32_, sizeof(cl_mem));
    kp->SetKernelArguments(1, &dn_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &layer0_output_fp32_, sizeof(cl_mem));
    kp->SetKernelArguments(3, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)K_h + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // (g) Custom fp32 summary for layer 0 output.
  {
    std::vector<float> r(K_h);
    clEnqueueReadBuffer(cl_q_, layer0_output_fp32_, CL_TRUE, 0,
                        (size_t)K_h * sizeof(float), r.data(), 0, nullptr,
                        nullptr);
    bool finite = true;
    float mn = std::numeric_limits<float>::infinity();
    float mx = -mn;
    for (unsigned int i = 0; i < K_h; ++i) {
      if (!std::isfinite(r[i])) finite = false;
      if (r[i] < mn) mn = r[i];
      if (r[i] > mx) mx = r[i];
    }
    std::fprintf(stderr,
                 "[qwen3-gpu] layer0 output fp32 N=%u first 8:", K_h);
    for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %g", r[i]);
    std::fprintf(stderr, "\n  last 4:");
    for (int i = 0; i < 4; ++i) std::fprintf(stderr, " %g", r[K_h - 4 + i]);
    std::fprintf(stderr,
                 "\n  min=%g max=%g all_finite=%d\n", mn, mx, finite);
    if (!finite) return false;
  }

  clReleaseMemObject(dn_fp32);
  clReleaseMemObject(dn_fp16);
  clReleaseMemObject(dn_act_image);
  clReleaseMemObject(dn_act_rs);
  clReleaseMemObject(dn_act_zp);
  clReleaseMemObject(dn_act_scale);
  clReleaseMemObject(dn_act_i8);
  clReleaseMemObject(swiglu_out);
  clReleaseMemObject(gate_fp32);
  clReleaseMemObject(up_fp32);
  clReleaseMemObject(gate_fp16);
  clReleaseMemObject(up_fp16);
  clReleaseMemObject(act_image);
  clReleaseMemObject(act_rs);
  clReleaseMemObject(act_zp);
  clReleaseMemObject(act_scale);
  clReleaseMemObject(act_i8);
  clReleaseMemObject(ffn_normed);
  clReleaseMemObject(ffn_in_padded);
  return true;
}

// Load a [head_dim] fp32 norm gamma at file_offset into SVM, converted
// to fp16 (for the rmsnorm_cl_fp16 kernel). Used by load_layer for
// q_norm / k_norm. Bytes consumed: head_dim * sizeof(float).
static bool load_qk_norm_to_svm_fp16(cl_context cl_ctx, cl_command_queue q,
                                     const float *src_fp32,
                                     unsigned int head_dim,
                                     void **dst_svm, const char *tag) {
  const size_t bytes = (size_t)head_dim * sizeof(uint16_t);
  *dst_svm = clSVMAlloc(cl_ctx, CL_MEM_READ_ONLY, bytes, 0);
  if (*dst_svm == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] %s SVMAlloc failed\n", tag);
    return false;
  }
  if (clEnqueueSVMMap(q, CL_TRUE, CL_MAP_WRITE, *dst_svm, bytes, 0,
                      nullptr, nullptr) != CL_SUCCESS)
    return false;
  uint16_t *p = static_cast<uint16_t *>(*dst_svm);
  for (unsigned int i = 0; i < head_dim; ++i) p[i] = f2h(src_fp32[i]);
  clEnqueueSVMUnmap(q, *dst_svm, 0, nullptr, nullptr);
  clFinish(q);
  return true;
}

// Load a [hidden] fp32 norm gamma at file_offset into SVM as fp32
// (matches the rmsnorm.cl kernel). Used by load_layer for attn_norm
// and ffn_norm.
static bool load_norm_to_svm_fp32(cl_context cl_ctx, cl_command_queue q,
                                  const float *src_fp32,
                                  unsigned int hidden, void **dst_svm,
                                  const char *tag) {
  const size_t bytes = (size_t)hidden * sizeof(float);
  *dst_svm = clSVMAlloc(cl_ctx, CL_MEM_READ_ONLY, bytes, 0);
  if (*dst_svm == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] %s SVMAlloc failed\n", tag);
    return false;
  }
  if (clEnqueueSVMMap(q, CL_TRUE, CL_MAP_WRITE, *dst_svm, bytes, 0,
                      nullptr, nullptr) != CL_SUCCESS)
    return false;
  std::memcpy(*dst_svm, src_fp32, bytes);
  clEnqueueSVMUnmap(q, *dst_svm, 0, nullptr, nullptr);
  clFinish(q);
  return true;
}

// #46m Phase 1: Same as load_norm_to_svm_fp32 but stores fp16 (for
// rmsnorm_cl_fp16 kernel). Used for attn_norm / ffn_norm / output_norm.
static bool load_hidden_norm_to_svm_fp16(cl_context cl_ctx, cl_command_queue q,
                                         const float *src_fp32,
                                         unsigned int hidden, void **dst_svm,
                                         const char *tag) {
  const size_t bytes = (size_t)hidden * sizeof(uint16_t);
  *dst_svm = clSVMAlloc(cl_ctx, CL_MEM_READ_ONLY, bytes, 0);
  if (*dst_svm == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] %s (fp16) SVMAlloc failed\n", tag);
    return false;
  }
  if (clEnqueueSVMMap(q, CL_TRUE, CL_MAP_WRITE, *dst_svm, bytes, 0,
                      nullptr, nullptr) != CL_SUCCESS)
    return false;
  uint16_t *p = static_cast<uint16_t *>(*dst_svm);
  for (unsigned int i = 0; i < hidden; ++i) p[i] = f2h(src_fp32[i]);
  clEnqueueSVMUnmap(q, *dst_svm, 0, nullptr, nullptr);
  clFinish(q);
  return true;
}

bool Qwen3Forward::load_layer(unsigned int layer_id, size_t *offset_inout,
                              unsigned int max_seq_len_used) {
  if (weight_mmap_ == nullptr || cl_ctx_ == nullptr) return false;
  if (layers_.size() <= layer_id) layers_.resize(layer_id + 1);
  LayerWeights &lw = layers_[layer_id];
  if (lw.wq.backing != nullptr) return true; // already loaded

  const unsigned int K_h = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;
  const unsigned int I = cfg_.intermediate_size;
  const size_t norm_bytes = (size_t)K_h * sizeof(float);
  const size_t head_norm_bytes = (size_t)cfg_.head_dim * sizeof(float);
  // FC record size is container/dtype-dependent (int8 / Section A / plain) —
  // peek the tag at the record start so the offset walk works for int4-all,
  // mixed 8/4/4, AND plain-container models.
  auto fc_bytes = [this](size_t at, unsigned int K, unsigned int N) -> size_t {
    return qint4_record_bytes(at, K, N);
  };

  size_t off = *offset_inout;

  if (cfg_.is_gemma2) {
    // #63 Gemma2 layout: input_ln | q | k | v | o | post_attn_ln | pre_ffn_ln |
    // gate | up | down | post_ffn_ln. NO q/k-norm; 4 sandwich norms (all [H]).
    // (1+w) is already baked into the gammas by the converter.
    auto load_hnorm = [&](void **f32, void **f16, const char *nm) -> bool {
      const float *g = reinterpret_cast<const float *>(weight_mmap_ + off);
      if (!load_norm_to_svm_fp32(cl_ctx_, cl_q_, g, K_h, f32, nm)) return false;
      if (!load_hidden_norm_to_svm_fp16(cl_ctx_, cl_q_, g, K_h, f16, nm))
        return false;
      off += norm_bytes;
      return true;
    };
    if (!load_hnorm(&lw.attn_norm_gamma_svm, &lw.attn_norm_gamma_svm_fp16,
                    "input_ln")) return false;
    if (!load_qint4_weight_at(off, K_h, N_q, &lw.wq, "wq")) return false;
    off += fc_bytes(off, K_h, N_q);
    if (!load_qint4_weight_at(off, K_h, N_kv, &lw.wk, "wk")) return false;
    off += fc_bytes(off, K_h, N_kv);
    if (!load_qint4_weight_at(off, K_h, N_kv, &lw.wv, "wv")) return false;
    off += fc_bytes(off, K_h, N_kv);
    if (!load_qint4_weight_at(off, N_q, K_h, &lw.wo, "wo")) return false;
    off += fc_bytes(off, N_q, K_h);
    if (!load_hnorm(&lw.post_attn_norm_gamma_svm,
                    &lw.post_attn_norm_gamma_svm_fp16, "post_attn_ln"))
      return false;
    if (!load_hnorm(&lw.ffn_norm_gamma_svm, &lw.ffn_norm_gamma_svm_fp16,
                    "pre_ffn_ln")) return false;
    if (!load_qint4_weight_at(off, K_h, I, &lw.ffn_up, "ffn_up")) return false;
    off += fc_bytes(off, K_h, I);
    if (!load_qint4_weight_at(off, K_h, I, &lw.ffn_gate, "ffn_gate"))
      return false;
    off += fc_bytes(off, K_h, I);
    if (!load_qint4_weight_at(off, I, K_h, &lw.ffn_down, "ffn_down"))
      return false;
    off += fc_bytes(off, I, K_h);
    if (!load_hnorm(&lw.post_ffn_norm_gamma_svm,
                    &lw.post_ffn_norm_gamma_svm_fp16, "post_ffn_ln"))
      return false;
  } else {

  // attn_norm gamma -> SVM fp32 AND fp16 (#46m)
  if (!load_norm_to_svm_fp32(
        cl_ctx_, cl_q_,
        reinterpret_cast<const float *>(weight_mmap_ + off), K_h,
        &lw.attn_norm_gamma_svm, "attn_norm")) return false;
  if (!load_hidden_norm_to_svm_fp16(
        cl_ctx_, cl_q_,
        reinterpret_cast<const float *>(weight_mmap_ + off), K_h,
        &lw.attn_norm_gamma_svm_fp16, "attn_norm")) return false;
  off += norm_bytes;

  // wq -> v8c backing
  if (!load_qint4_weight_at(off, K_h, N_q, &lw.wq, "wq")) return false;
  off += fc_bytes(off, K_h, N_q);

  // q_norm gamma -> SVM fp16
  if (!load_qk_norm_to_svm_fp16(
        cl_ctx_, cl_q_,
        reinterpret_cast<const float *>(weight_mmap_ + off),
        cfg_.head_dim, &lw.q_norm_gamma_svm_fp16, "q_norm")) return false;
  off += head_norm_bytes;

  // wk -> v8c backing
  if (!load_qint4_weight_at(off, K_h, N_kv, &lw.wk, "wk")) return false;
  off += fc_bytes(off, K_h, N_kv);

  // k_norm gamma -> SVM fp16
  if (!load_qk_norm_to_svm_fp16(
        cl_ctx_, cl_q_,
        reinterpret_cast<const float *>(weight_mmap_ + off),
        cfg_.head_dim, &lw.k_norm_gamma_svm_fp16, "k_norm")) return false;
  off += head_norm_bytes;

  // wv -> v8c backing
  if (!load_qint4_weight_at(off, K_h, N_kv, &lw.wv, "wv")) return false;
  off += fc_bytes(off, K_h, N_kv);

  // wo -> v8c backing
  if (!load_qint4_weight_at(off, N_q, K_h, &lw.wo, "wo")) return false;
  off += fc_bytes(off, N_q, K_h);

  // ffn_norm gamma -> SVM fp32 AND fp16 (#46m)
  if (!load_norm_to_svm_fp32(
        cl_ctx_, cl_q_,
        reinterpret_cast<const float *>(weight_mmap_ + off), K_h,
        &lw.ffn_norm_gamma_svm, "ffn_norm")) return false;
  if (!load_hidden_norm_to_svm_fp16(
        cl_ctx_, cl_q_,
        reinterpret_cast<const float *>(weight_mmap_ + off), K_h,
        &lw.ffn_norm_gamma_svm_fp16, "ffn_norm")) return false;
  off += norm_bytes;

  // ffn_up, ffn_gate, ffn_down -> v8c backings
  if (!load_qint4_weight_at(off, K_h, I, &lw.ffn_up, "ffn_up"))
    return false;
  off += fc_bytes(off, K_h, I);
  if (!load_qint4_weight_at(off, K_h, I, &lw.ffn_gate, "ffn_gate"))
    return false;
  off += fc_bytes(off, K_h, I);
  if (!load_qint4_weight_at(off, I, K_h, &lw.ffn_down, "ffn_down"))
    return false;
  off += fc_bytes(off, I, K_h);
  } // end else (Qwen3 layout)

  // K, V cache SVM, sized for max_seq_len_used.
  const size_t cache_bytes =
    (size_t)max_seq_len_used * cfg_.num_heads_KV * cfg_.head_dim *
    sizeof(uint16_t);
  lw.cache_k_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, cache_bytes, 0);
  lw.cache_v_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, cache_bytes, 0);
  if (lw.cache_k_svm == nullptr || lw.cache_v_svm == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] layer %u: KV cache SVMAlloc failed\n",
                 layer_id);
    return false;
  }
  // Zero-init.
  for (void *p : {lw.cache_k_svm, lw.cache_v_svm}) {
    cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, p,
                                 cache_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;
    std::memset(p, 0, cache_bytes);
    clEnqueueSVMUnmap(cl_q_, p, 0, nullptr, nullptr);
  }
  clFinish(cl_q_);

  // OHWI image2d V experiment mirror buffer (#46f). cl_mem (not SVM) so
  // it can be wrapped via image2d_from_buffer; same size as cache_v_svm.
  // Layout: OHWI-reversed [hKV, d, max_seq_len_used] fp16.
  cl_int img_err = CL_SUCCESS;
  lw.cache_v_buf_ohwi =
    clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE, cache_bytes, nullptr, &img_err);
  if (img_err != CL_SUCCESS || lw.cache_v_buf_ohwi == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] layer %u: clCreateBuffer(cache_v_buf_ohwi) "
                 "err=%d (image2d V experiment disabled)\n",
                 layer_id, img_err);
    lw.cache_v_buf_ohwi = nullptr;
  } else {
    lw.cache_v_buf_ohwi_bytes = cache_bytes;
    // Zero-init via clEnqueueFillBuffer.
    const uint16_t zero = 0;
    clEnqueueFillBuffer(cl_q_, lw.cache_v_buf_ohwi, &zero, sizeof(uint16_t), 0,
                        cache_bytes, 0, nullptr, nullptr);
    clFinish(cl_q_);
    // Build the V image2d_from_buffer view once.
    cl_image_format img_fmt{CL_RGBA, CL_UNSIGNED_INT32};
    cl_image_desc vd{};
    vd.image_type = CL_MEM_OBJECT_IMAGE2D;
    vd.image_width = (size_t)max_seq_len_used / 8;        // 8 halves/texel
    vd.image_height = (size_t)cfg_.num_heads_KV * cfg_.head_dim;
    vd.image_row_pitch = (size_t)max_seq_len_used * sizeof(uint16_t);
    vd.buffer = lw.cache_v_buf_ohwi;
    cl_int ie = CL_SUCCESS;
    lw.cache_v_image_ohwi =
      clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &img_fmt, &vd, nullptr, &ie);
    if (ie != CL_SUCCESS || lw.cache_v_image_ohwi == nullptr) {
      std::fprintf(stderr,
                   "[qwen3-gpu] layer %u: clCreateImage(cache_v_image_ohwi) "
                   "err=%d (image2d V experiment disabled)\n",
                   layer_id, ie);
      lw.cache_v_image_ohwi = nullptr;
    }
  }

  // #46h: K image2d mirror. Same memory size as cache_k_svm, but
  // cl_mem so image2d_from_buffer can wrap. K's OHWI layout has
  // O=cache_size (rows), I=d_h (cols) per head, so the image2d
  // axes are: width = d_h/8 texels, height = H_kv * S_max.
  cl_int ke = CL_SUCCESS;
  lw.cache_k_buf_ohwi =
    clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE, cache_bytes, nullptr, &ke);
  if (ke != CL_SUCCESS || lw.cache_k_buf_ohwi == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] layer %u: clCreateBuffer(cache_k_buf_ohwi) "
                 "err=%d (K image2d disabled)\n",
                 layer_id, ke);
    lw.cache_k_buf_ohwi = nullptr;
  } else {
    lw.cache_k_buf_ohwi_bytes = cache_bytes;
    const uint16_t zero = 0;
    clEnqueueFillBuffer(cl_q_, lw.cache_k_buf_ohwi, &zero, sizeof(uint16_t), 0,
                        cache_bytes, 0, nullptr, nullptr);
    clFinish(cl_q_);
    cl_image_format kf{CL_RGBA, CL_UNSIGNED_INT32};
    cl_image_desc kd{};
    kd.image_type = CL_MEM_OBJECT_IMAGE2D;
    kd.image_width = (size_t)cfg_.head_dim / 8;
    kd.image_height =
      (size_t)cfg_.num_heads_KV * (size_t)max_seq_len_used;
    kd.image_row_pitch = (size_t)cfg_.head_dim * sizeof(uint16_t);
    kd.buffer = lw.cache_k_buf_ohwi;
    cl_int kie = CL_SUCCESS;
    lw.cache_k_image_ohwi =
      clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &kf, &kd, nullptr, &kie);
    if (kie != CL_SUCCESS || lw.cache_k_image_ohwi == nullptr) {
      std::fprintf(stderr,
                   "[qwen3-gpu] layer %u: clCreateImage(cache_k_image_ohwi) "
                   "err=%d\n", layer_id, kie);
      lw.cache_k_image_ohwi = nullptr;
    }
  }

  *offset_inout = off;
  // Record the cache stride so the OHWI attention wrapper can pass it
  // as `max_seq_len` (the per-head index stride in the OHWI layout).
  // All layers share the same max_seq_len_used in our setup.
  if (kv_cache_max_seq_len_ == 0)
    kv_cache_max_seq_len_ = max_seq_len_used;
  std::fprintf(stderr,
               "[qwen3-gpu] layer %u loaded; advanced offset to %zu "
               "(~%zu MB)\n",
               layer_id, off, off / (1024 * 1024));
  return true;
}

cl_mem Qwen3Forward::forward_one_layer(unsigned int layer_id, cl_mem in_fp32,
                                       unsigned int position) {
  if (layer_id >= layers_.size() || layers_[layer_id].wq.backing == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] forward_one_layer(%u): not loaded\n", layer_id);
    return nullptr;
  }
  LayerWeights &lw = layers_[layer_id];
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  cl_int err = CL_SUCCESS;

  const unsigned int M_pad = 4;
  const unsigned int K_h = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;
  const unsigned int I = cfg_.intermediate_size;

  // (a) Pad in_fp32 [K_h] into [M_pad, K_h] for the v8c tile.
  cl_mem in_padded = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    (size_t)M_pad * K_h * sizeof(float),
                                    nullptr, &err);
  float zero = 0.0f;
  clEnqueueFillBuffer(cl_q_, in_padded, &zero, sizeof(float), 0,
                      (size_t)M_pad * K_h * sizeof(float), 0, nullptr,
                      nullptr);
  clEnqueueCopyBuffer(cl_q_, in_fp32, in_padded, 0, 0,
                      (size_t)K_h * sizeof(float), 0, nullptr, nullptr);
  cl_mem attn_normed =
    clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                   (size_t)M_pad * K_h * sizeof(float), nullptr, &err);
  // attn_norm
  {
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_kernel, "rmsnorm_cl");
    float eps = cfg_.rms_norm_eps;
    int H = (int)M_pad, W = (int)K_h;
    kp->SetKernelArguments(0, &in_padded, sizeof(cl_mem));
    kp->SetKernelArguments(1, &attn_normed, sizeof(cl_mem));
    kp->SetKernelSVMArguments(2, lw.attn_norm_gamma_svm);
    kp->SetKernelArguments(3, &eps, sizeof(float));
    kp->SetKernelArguments(4, &H, sizeof(int));
    kp->SetKernelArguments(5, &W, sizeof(int));
    std::array<size_t, 1> gws = {(size_t)M_pad * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // (b) Shared act quant for Q/K/V.
  cl_mem act_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 (size_t)M_pad * K_h, nullptr, &err);
  cl_mem act_scale = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(float) * M_pad, nullptr, &err);
  cl_mem act_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 sizeof(int) * M_pad, nullptr, &err);
  cl_mem act_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 sizeof(int) * M_pad, nullptr, &err);
  nntrainer::quantize_act_v8c_fp32_cl(attn_normed, act_i8, act_scale,
                                      act_zp, act_rs, M_pad, K_h);
  cl_image_format afmt{CL_RGBA, CL_UNSIGNED_INT32};
  cl_image_desc adesc{};
  adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  adesc.image_width = K_h / 16;
  adesc.image_height = M_pad;
  adesc.image_row_pitch = K_h;
  adesc.buffer = act_i8;
  cl_mem act_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &adesc, nullptr, &err);

  cl_mem y_q = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                              sizeof(uint16_t) * (size_t)M_pad * N_q,
                              nullptr, &err);
  cl_mem y_k = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                              sizeof(uint16_t) * (size_t)M_pad * N_kv,
                              nullptr, &err);
  cl_mem y_v = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                              sizeof(uint16_t) * (size_t)M_pad * N_kv,
                              nullptr, &err);
  nntrainer::gemm_int8_v8c_cl(act_image, lw.wq.weight_image, act_scale,
                              lw.wq.scale_buf, act_rs, act_zp,
                              lw.wq.row_sum_w_int4, y_q, M_pad, N_q, K_h);
  nntrainer::gemm_int8_v8c_cl(act_image, lw.wk.weight_image, act_scale,
                              lw.wk.scale_buf, act_rs, act_zp,
                              lw.wk.row_sum_w_int4, y_k, M_pad, N_kv, K_h);
  nntrainer::gemm_int8_v8c_cl(act_image, lw.wv.weight_image, act_scale,
                              lw.wv.scale_buf, act_rs, act_zp,
                              lw.wv.row_sum_w_int4, y_v, M_pad, N_kv, K_h);
  clFinish(cl_q_);

  // (c) q_norm / k_norm in place, then RoPE in place (cos/sin tables
  //     are shared across layers — already precomputed for this position).
  auto disp_qk_norm = [&](cl_mem io, void *gamma, unsigned int num_heads) {
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_fp16_kernel,
                                   "rmsnorm_cl_fp16");
    uint16_t eps_h = f2h(cfg_.rms_norm_eps);
    int B = 1, C = 1, H = (int)num_heads, W = (int)cfg_.head_dim;
    kp->SetKernelArguments(0, &io, sizeof(cl_mem));
    kp->SetKernelArguments(1, &io, sizeof(cl_mem));
    kp->SetKernelSVMArguments(2, gamma);
    kp->SetKernelArguments(3, &eps_h, sizeof(uint16_t));
    kp->SetKernelArguments(4, &B, sizeof(int));
    kp->SetKernelArguments(5, &C, sizeof(int));
    kp->SetKernelArguments(6, &H, sizeof(int));
    kp->SetKernelArguments(7, &W, sizeof(int));
    std::array<size_t, 2> gws = {1, (size_t)num_heads};
    std::array<size_t, 2> lws = {1, 1};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 2, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
  };
  disp_qk_norm(y_q, lw.q_norm_gamma_svm_fp16, cfg_.num_heads_Q);
  disp_qk_norm(y_k, lw.k_norm_gamma_svm_fp16, cfg_.num_heads_KV);
  clFinish(cl_q_);

  if (layer0_rope_position_ != (int)position) {
    if (!precompute_rope_for_position(position)) return nullptr;
  }
  run_layer0_rope_on_qk(y_q, y_k);

  // (d) Write K, V into this layer's SVM cache at `position`.
  const size_t kv_row_bytes = (size_t)N_kv * sizeof(uint16_t);
  auto copy_cl_to_svm = [&](cl_mem src, void *dst_svm_base,
                            size_t offset_bytes) {
    cl_int e;
    void *p = clEnqueueMapBuffer(cl_q_, src, CL_TRUE, CL_MAP_READ, 0,
                                 kv_row_bytes, 0, nullptr, nullptr, &e);
    if (!p) return false;
    void *dst = static_cast<uint8_t *>(dst_svm_base) + offset_bytes;
    if (clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, dst, kv_row_bytes, 0,
                        nullptr, nullptr) != CL_SUCCESS)
      return false;
    std::memcpy(dst, p, kv_row_bytes);
    clEnqueueSVMUnmap(cl_q_, dst, 0, nullptr, nullptr);
    clEnqueueUnmapMemObject(cl_q_, src, p, 0, nullptr, nullptr);
    return true;
  };
  copy_cl_to_svm(y_k, lw.cache_k_svm, (size_t)position * kv_row_bytes);
  copy_cl_to_svm(y_v, lw.cache_v_svm, (size_t)position * kv_row_bytes);
  clFinish(cl_q_);

  // (e) Attention dispatch via SVM.
  const size_t q_row_bytes = (size_t)N_q * sizeof(uint16_t);
  void *q_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, q_row_bytes, 0);
  void *o_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, q_row_bytes, 0);
  {
    cl_int e;
    void *p = clEnqueueMapBuffer(cl_q_, y_q, CL_TRUE, CL_MAP_READ, 0,
                                 q_row_bytes, 0, nullptr, nullptr, &e);
    clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, q_svm, q_row_bytes, 0,
                    nullptr, nullptr);
    std::memcpy(q_svm, p, q_row_bytes);
    clEnqueueSVMUnmap(cl_q_, q_svm, 0, nullptr, nullptr);
    clEnqueueUnmapMemObject(cl_q_, y_q, p, 0, nullptr, nullptr);
    clFinish(cl_q_);
  }
  bool attn_ok = nntrainer::two_conv_attention_prefill_f16_cl(
    static_cast<const uint16_t *>(q_svm),
    static_cast<const uint16_t *>(lw.cache_k_svm),
    static_cast<const uint16_t *>(lw.cache_v_svm),
    static_cast<uint16_t *>(o_svm), 1, position + 1, cfg_.num_heads_Q,
    cfg_.num_heads_KV, cfg_.head_dim, true, true);
  if (!attn_ok) {
    std::fprintf(stderr,
                 "[qwen3-gpu] layer %u attention failed\n", layer_id);
  }

  // (f) wo: O_svm fp16 -> wo_out fp16 -> residual_1 fp32 = in + wo_out.
  cl_mem o_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 (size_t)M_pad * N_q * sizeof(float),
                                 nullptr, &err);
  clEnqueueFillBuffer(cl_q_, o_fp32, &zero, sizeof(float), 0,
                      (size_t)M_pad * N_q * sizeof(float), 0, nullptr,
                      nullptr);
  {
    auto kp = cl->registerClKernel(kConvertFp16ToFp32Kernel, "cvt_h2f");
    int n = (int)N_q;
    kp->SetKernelSVMArguments(0, o_svm);
    kp->SetKernelArguments(1, &o_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)N_q + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }
  cl_mem wo_act_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    (size_t)M_pad * N_q, nullptr, &err);
  cl_mem wo_act_scale = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                       sizeof(float) * M_pad, nullptr, &err);
  cl_mem wo_act_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(int) * M_pad, nullptr, &err);
  cl_mem wo_act_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(int) * M_pad, nullptr, &err);
  cl_mem wo_y_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(uint16_t) * (size_t)M_pad * K_h,
                                    nullptr, &err);
  nntrainer::quantize_act_v8c_fp32_cl(o_fp32, wo_act_i8, wo_act_scale,
                                      wo_act_zp, wo_act_rs, M_pad, N_q);
  cl_image_desc wo_adesc{};
  wo_adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  wo_adesc.image_width = N_q / 16;
  wo_adesc.image_height = M_pad;
  wo_adesc.image_row_pitch = N_q;
  wo_adesc.buffer = wo_act_i8;
  cl_mem wo_act_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &wo_adesc, nullptr, &err);
  nntrainer::gemm_int8_v8c_cl(wo_act_image, lw.wo.weight_image, wo_act_scale,
                              lw.wo.scale_buf, wo_act_rs, wo_act_zp,
                              lw.wo.row_sum_w_int4, wo_y_fp16, M_pad, K_h, N_q);
  clFinish(cl_q_);
  cl_mem wo_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  (size_t)K_h * sizeof(float), nullptr, &err);
  {
    auto kp = cl->registerClKernel(kConvertFp16ToFp32Kernel, "cvt_h2f");
    int n = (int)K_h;
    kp->SetKernelArguments(0, &wo_y_fp16, sizeof(cl_mem));
    kp->SetKernelArguments(1, &wo_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)K_h + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }
  cl_mem residual_1 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                     (size_t)K_h * sizeof(float), nullptr,
                                     &err);
  {
    auto kp = cl->registerClKernel(kAddFp32Kernel, "add_fp32");
    int n = (int)K_h;
    kp->SetKernelArguments(0, &in_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(1, &wo_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &residual_1, sizeof(cl_mem));
    kp->SetKernelArguments(3, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)K_h + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // (g) ffn_norm + ffn_up/gate (shared quant) + swiglu + ffn_down +
  //     residual_2.
  cl_mem ffn_in_padded =
    clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                   (size_t)M_pad * K_h * sizeof(float), nullptr, &err);
  clEnqueueFillBuffer(cl_q_, ffn_in_padded, &zero, sizeof(float), 0,
                      (size_t)M_pad * K_h * sizeof(float), 0, nullptr,
                      nullptr);
  clEnqueueCopyBuffer(cl_q_, residual_1, ffn_in_padded, 0, 0,
                      (size_t)K_h * sizeof(float), 0, nullptr, nullptr);
  cl_mem ffn_normed = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                     (size_t)M_pad * K_h * sizeof(float),
                                     nullptr, &err);
  {
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_kernel, "rmsnorm_cl");
    float eps = cfg_.rms_norm_eps;
    int H = (int)M_pad, W = (int)K_h;
    kp->SetKernelArguments(0, &ffn_in_padded, sizeof(cl_mem));
    kp->SetKernelArguments(1, &ffn_normed, sizeof(cl_mem));
    kp->SetKernelSVMArguments(2, lw.ffn_norm_gamma_svm);
    kp->SetKernelArguments(3, &eps, sizeof(float));
    kp->SetKernelArguments(4, &H, sizeof(int));
    kp->SetKernelArguments(5, &W, sizeof(int));
    std::array<size_t, 1> gws = {(size_t)M_pad * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }
  cl_mem fa_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                (size_t)M_pad * K_h, nullptr, &err);
  cl_mem fa_sc = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                sizeof(float) * M_pad, nullptr, &err);
  cl_mem fa_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                sizeof(int) * M_pad, nullptr, &err);
  cl_mem fa_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                sizeof(int) * M_pad, nullptr, &err);
  nntrainer::quantize_act_v8c_fp32_cl(ffn_normed, fa_i8, fa_sc, fa_zp, fa_rs,
                                      M_pad, K_h);
  cl_image_desc fa_adesc{};
  fa_adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  fa_adesc.image_width = K_h / 16;
  fa_adesc.image_height = M_pad;
  fa_adesc.image_row_pitch = K_h;
  fa_adesc.buffer = fa_i8;
  cl_mem fa_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &fa_adesc, nullptr, &err);
  cl_mem up_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  sizeof(uint16_t) * (size_t)M_pad * I,
                                  nullptr, &err);
  cl_mem gate_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(uint16_t) * (size_t)M_pad * I,
                                    nullptr, &err);
  nntrainer::gemm_int8_v8c_cl(fa_image, lw.ffn_up.weight_image, fa_sc,
                              lw.ffn_up.scale_buf, fa_rs, fa_zp,
                              lw.ffn_up.row_sum_w_int4, up_fp16, M_pad, I,
                              K_h);
  nntrainer::gemm_int8_v8c_cl(fa_image, lw.ffn_gate.weight_image, fa_sc,
                              lw.ffn_gate.scale_buf, fa_rs, fa_zp,
                              lw.ffn_gate.row_sum_w_int4, gate_fp16, M_pad, I,
                              K_h);
  clFinish(cl_q_);
  cl_mem up_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  (size_t)I * sizeof(float), nullptr, &err);
  cl_mem gate_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    (size_t)I * sizeof(float), nullptr, &err);
  auto disp_cvt = [&](cl_mem hin, cl_mem fout, unsigned int n) {
    auto kp = cl->registerClKernel(kConvertFp16ToFp32Kernel, "cvt_h2f");
    int ni = (int)n;
    kp->SetKernelArguments(0, &hin, sizeof(cl_mem));
    kp->SetKernelArguments(1, &fout, sizeof(cl_mem));
    kp->SetKernelArguments(2, &ni, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)n + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
  };
  disp_cvt(up_fp16, up_fp32, I);
  disp_cvt(gate_fp16, gate_fp32, I);
  clFinish(cl_q_);
  cl_mem swiglu_out = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                     (size_t)M_pad * I * sizeof(float),
                                     nullptr, &err);
  clEnqueueFillBuffer(cl_q_, swiglu_out, &zero, sizeof(float), 0,
                      (size_t)M_pad * I * sizeof(float), 0, nullptr, nullptr);
  {
    auto kp = cl->registerClKernel(kSwigluFp32Kernel, "swiglu_fp32");
    int n = (int)I;
    kp->SetKernelArguments(0, &gate_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(1, &up_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &swiglu_out, sizeof(cl_mem));
    kp->SetKernelArguments(3, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)I + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }
  cl_mem dn_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                (size_t)M_pad * I, nullptr, &err);
  cl_mem dn_sc = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                sizeof(float) * M_pad, nullptr, &err);
  cl_mem dn_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                sizeof(int) * M_pad, nullptr, &err);
  cl_mem dn_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                sizeof(int) * M_pad, nullptr, &err);
  nntrainer::quantize_act_v8c_fp32_cl(swiglu_out, dn_i8, dn_sc, dn_zp, dn_rs,
                                      M_pad, I);
  cl_image_desc dn_adesc{};
  dn_adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  dn_adesc.image_width = I / 16;
  dn_adesc.image_height = M_pad;
  dn_adesc.image_row_pitch = I;
  dn_adesc.buffer = dn_i8;
  cl_mem dn_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &dn_adesc, nullptr, &err);
  cl_mem dn_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  sizeof(uint16_t) * (size_t)M_pad * K_h,
                                  nullptr, &err);
  nntrainer::gemm_int8_v8c_cl(dn_image, lw.ffn_down.weight_image, dn_sc,
                              lw.ffn_down.scale_buf, dn_rs, dn_zp,
                              lw.ffn_down.row_sum_w_int4, dn_fp16, M_pad, K_h,
                              I);
  clFinish(cl_q_);
  cl_mem dn_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  (size_t)K_h * sizeof(float), nullptr, &err);
  disp_cvt(dn_fp16, dn_fp32, K_h);
  clFinish(cl_q_);
  cl_mem out_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                   (size_t)K_h * sizeof(float), nullptr,
                                   &err);
  {
    auto kp = cl->registerClKernel(kAddFp32Kernel, "add_fp32");
    int n = (int)K_h;
    kp->SetKernelArguments(0, &residual_1, sizeof(cl_mem));
    kp->SetKernelArguments(1, &dn_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &out_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(3, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)K_h + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // Cleanup all the per-call scratch.
  clReleaseMemObject(dn_fp32);
  clReleaseMemObject(dn_fp16);
  clReleaseMemObject(dn_image);
  clReleaseMemObject(dn_rs);
  clReleaseMemObject(dn_zp);
  clReleaseMemObject(dn_sc);
  clReleaseMemObject(dn_i8);
  clReleaseMemObject(swiglu_out);
  clReleaseMemObject(gate_fp32);
  clReleaseMemObject(up_fp32);
  clReleaseMemObject(gate_fp16);
  clReleaseMemObject(up_fp16);
  clReleaseMemObject(fa_image);
  clReleaseMemObject(fa_rs);
  clReleaseMemObject(fa_zp);
  clReleaseMemObject(fa_sc);
  clReleaseMemObject(fa_i8);
  clReleaseMemObject(ffn_normed);
  clReleaseMemObject(ffn_in_padded);
  clReleaseMemObject(residual_1);
  clReleaseMemObject(wo_fp32);
  clReleaseMemObject(wo_y_fp16);
  clReleaseMemObject(wo_act_image);
  clReleaseMemObject(wo_act_rs);
  clReleaseMemObject(wo_act_zp);
  clReleaseMemObject(wo_act_scale);
  clReleaseMemObject(wo_act_i8);
  clReleaseMemObject(o_fp32);
  if (q_svm) clSVMFree(cl_ctx_, q_svm);
  if (o_svm) clSVMFree(cl_ctx_, o_svm);
  clReleaseMemObject(y_v);
  clReleaseMemObject(y_k);
  clReleaseMemObject(y_q);
  clReleaseMemObject(act_image);
  clReleaseMemObject(act_rs);
  clReleaseMemObject(act_zp);
  clReleaseMemObject(act_scale);
  clReleaseMemObject(act_i8);
  clReleaseMemObject(attn_normed);
  clReleaseMemObject(in_padded);

  return out_fp32; // caller owns
}

bool Qwen3Forward::ensure_forward_scratch_allocated(unsigned int max_M) {
  if (cl_ctx_ == nullptr) return false;
  // v8c kernel tile alignment: round up to multiple of 4.
  const unsigned int M_pad = ((max_M + 3) / 4) * 4;
  if (scratch_max_M_ >= M_pad && scratch_.in_padded != nullptr) {
    return true; // already big enough
  }
  // Free existing scratch (if any) before re-allocating to larger size.
  if (scratch_.in_padded != nullptr) {
    auto rel = [&](cl_mem &m) { if (m) { clReleaseMemObject(m); m = nullptr; } };
    auto svm = [&](void *&p) { if (p && cl_ctx_) { clSVMFree(cl_ctx_, p); p = nullptr; } };
    rel(scratch_.in_padded);    rel(scratch_.attn_normed);
    rel(scratch_.qkv_act_i8);   rel(scratch_.qkv_act_scale);
    rel(scratch_.qkv_act_zp);   rel(scratch_.qkv_act_rs);
    rel(scratch_.y_q);          rel(scratch_.y_k);          rel(scratch_.y_v);
    svm(scratch_.q_svm);        svm(scratch_.o_svm);
    rel(scratch_.o_fp32);
    rel(scratch_.wo_act_i8);    rel(scratch_.wo_act_scale);
    rel(scratch_.wo_act_zp);    rel(scratch_.wo_act_rs);
    rel(scratch_.wo_y_fp16);    rel(scratch_.wo_fp32);
    rel(scratch_.residual_1);
    rel(scratch_.ffn_in_padded);rel(scratch_.ffn_normed);
    rel(scratch_.fa_i8);        rel(scratch_.fa_sc);
    rel(scratch_.fa_zp);        rel(scratch_.fa_rs);
    rel(scratch_.up_fp16);      rel(scratch_.gate_fp16);
    rel(scratch_.up_fp32);      rel(scratch_.gate_fp32);
    rel(scratch_.swiglu_out);
    rel(scratch_.dn_i8);        rel(scratch_.dn_sc);
    rel(scratch_.dn_zp);        rel(scratch_.dn_rs);
    rel(scratch_.dn_fp16);      rel(scratch_.dn_fp32);
    // Increment 2: cached activation image views (must die with their buffers).
    rel(scratch_.qkv_act_img);  rel(scratch_.wo_act_img);
    rel(scratch_.fa_act_img);   rel(scratch_.dn_act_img);
  }

  const unsigned int K_h = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;
  const unsigned int I = cfg_.intermediate_size;

  cl_int err = CL_SUCCESS;
  auto alloc = [&](cl_mem &m, cl_mem_flags flags, size_t bytes,
                   const char *tag) -> bool {
    m = clCreateBuffer(cl_ctx_, flags, bytes, nullptr, &err);
    if (err != CL_SUCCESS || m == nullptr) {
      std::fprintf(stderr, "[qwen3-gpu] scratch alloc %s (%zu B) err=%d\n",
                   tag, bytes, err);
      return false;
    }
    return true;
  };

  // #46m: residual stream is fp16 throughout (paper-aligned).
  if (!alloc(scratch_.in_padded,    CL_MEM_READ_WRITE, (size_t)M_pad * K_h * sizeof(float), "in_padded"))    return false;     // #47j fp32 inter-layer residual
  if (!alloc(scratch_.attn_normed,  CL_MEM_READ_WRITE, (size_t)M_pad * K_h * sizeof(uint16_t), "attn_normed"))  return false;
  if (!alloc(scratch_.qkv_act_i8,   CL_MEM_READ_WRITE, (size_t)M_pad * K_h,                 "qkv_act_i8"))   return false;
  if (!alloc(scratch_.qkv_act_scale,CL_MEM_READ_WRITE, sizeof(float) * M_pad,                "qkv_act_scale"))return false;
  if (!alloc(scratch_.qkv_act_zp,   CL_MEM_READ_WRITE, sizeof(int) * M_pad,                  "qkv_act_zp"))   return false;
  if (!alloc(scratch_.qkv_act_rs,   CL_MEM_READ_WRITE, sizeof(int) * M_pad,                  "qkv_act_rs"))   return false;
  if (!alloc(scratch_.y_q,          CL_MEM_READ_WRITE, sizeof(uint16_t) * (size_t)M_pad * N_q,  "y_q"))       return false;
  if (!alloc(scratch_.y_k,          CL_MEM_READ_WRITE, sizeof(uint16_t) * (size_t)M_pad * N_kv, "y_k"))       return false;
  if (!alloc(scratch_.y_v,          CL_MEM_READ_WRITE, sizeof(uint16_t) * (size_t)M_pad * N_kv, "y_v"))       return false;
  // SVM bridge sized for M_pad rows of Q (largest per-call attention input).
  scratch_.q_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, (size_t)M_pad * N_q * sizeof(uint16_t), 0);
  scratch_.o_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, (size_t)M_pad * N_q * sizeof(uint16_t), 0);
  if (!scratch_.q_svm || !scratch_.o_svm) {
    std::fprintf(stderr, "[qwen3-gpu] scratch SVM alloc failed\n");
    return false;
  }
  // #46m: o_fp32 now holds fp16 (kept name for back-compat).
  if (!alloc(scratch_.o_fp32,       CL_MEM_READ_WRITE, (size_t)M_pad * N_q * sizeof(uint16_t), "o_fp32"))       return false;
  if (!alloc(scratch_.wo_act_i8,    CL_MEM_READ_WRITE, (size_t)M_pad * N_q,                  "wo_act_i8"))   return false;
  if (!alloc(scratch_.wo_act_scale, CL_MEM_READ_WRITE, sizeof(float) * M_pad,                "wo_act_scale"))return false;
  if (!alloc(scratch_.wo_act_zp,    CL_MEM_READ_WRITE, sizeof(int) * M_pad,                  "wo_act_zp"))   return false;
  if (!alloc(scratch_.wo_act_rs,    CL_MEM_READ_WRITE, sizeof(int) * M_pad,                  "wo_act_rs"))   return false;
  if (!alloc(scratch_.wo_y_fp16,    CL_MEM_READ_WRITE, sizeof(uint16_t) * (size_t)M_pad * K_h,"wo_y_fp16"))   return false;
  // #46m: residual fp16.
  if (!alloc(scratch_.wo_fp32,      CL_MEM_READ_WRITE, (size_t)M_pad * K_h * sizeof(uint16_t),"wo_fp32"))     return false;
  if (!alloc(scratch_.residual_1,   CL_MEM_READ_WRITE, (size_t)M_pad * K_h * sizeof(float),"residual_1"))  return false;     // #47j fp32 residual accumulation (last-layer massive-activation overflow)
  if (!alloc(scratch_.ffn_in_padded,CL_MEM_READ_WRITE, (size_t)M_pad * K_h * sizeof(float),"ffn_in_padded"))return false;  // #47j fp32
  if (!alloc(scratch_.ffn_normed,   CL_MEM_READ_WRITE, (size_t)M_pad * K_h * sizeof(uint16_t),"ffn_normed"))   return false;
  if (!alloc(scratch_.fa_i8,        CL_MEM_READ_WRITE, (size_t)M_pad * K_h,                  "fa_i8"))       return false;
  if (!alloc(scratch_.fa_sc,        CL_MEM_READ_WRITE, sizeof(float) * M_pad,                "fa_sc"))       return false;
  if (!alloc(scratch_.fa_zp,        CL_MEM_READ_WRITE, sizeof(int) * M_pad,                  "fa_zp"))       return false;
  if (!alloc(scratch_.fa_rs,        CL_MEM_READ_WRITE, sizeof(int) * M_pad,                  "fa_rs"))       return false;
  if (!alloc(scratch_.up_fp16,      CL_MEM_READ_WRITE, sizeof(uint16_t) * (size_t)M_pad * I, "up_fp16"))     return false;
  if (!alloc(scratch_.gate_fp16,    CL_MEM_READ_WRITE, sizeof(uint16_t) * (size_t)M_pad * I, "gate_fp16"))   return false;
  if (!alloc(scratch_.up_fp32,      CL_MEM_READ_WRITE, (size_t)M_pad * I * sizeof(float),    "up_fp32"))     return false;
  if (!alloc(scratch_.gate_fp32,    CL_MEM_READ_WRITE, (size_t)M_pad * I * sizeof(float),    "gate_fp32"))   return false;
  if (!alloc(scratch_.swiglu_out,   CL_MEM_READ_WRITE, (size_t)M_pad * I * sizeof(float), "swiglu_out"))  return false;  // #47i: fp32 swiglu product (avoids fp16 overflow of silu(gate)*up)
  if (!alloc(scratch_.dn_i8,        CL_MEM_READ_WRITE, (size_t)M_pad * I,                    "dn_i8"))       return false;
  if (!alloc(scratch_.dn_sc,        CL_MEM_READ_WRITE, sizeof(float) * M_pad,                "dn_sc"))       return false;
  if (!alloc(scratch_.dn_zp,        CL_MEM_READ_WRITE, sizeof(int) * M_pad,                  "dn_zp"))       return false;
  if (!alloc(scratch_.dn_rs,        CL_MEM_READ_WRITE, sizeof(int) * M_pad,                  "dn_rs"))       return false;
  if (!alloc(scratch_.dn_fp16,      CL_MEM_READ_WRITE, sizeof(uint16_t) * (size_t)M_pad * K_h,"dn_fp16"))     return false;
  if (!alloc(scratch_.dn_fp32,      CL_MEM_READ_WRITE, (size_t)M_pad * K_h * sizeof(uint16_t),"dn_fp32"))     return false;

  // Increment 2: cache the int8 activation image2d views once (sized for the
  // max M_pad). Reused every layer; reads only touch valid rows so a
  // max-height view serves any M<=M_pad. Routed through the tv factory.
  // #v8c-buf: when the buffer-load FC path is active (NNTR_V8C_BUF=1, e.g.
  // Intel NEO) the GEMMs index the raw int8 scratch buffer directly, so no
  // image2d views are needed — skip their creation entirely.
  const bool skip_act_img = []() {
    const char *e = std::getenv("NNTR_V8C_BUF");
    return e && std::atoi(e) != 0;
  }();
  if (!skip_act_img) {
    scratch_.qkv_act_img = make_act_image2d(cl_ctx_, scratch_.qkv_act_i8, K_h, M_pad, img_caps_);
    scratch_.wo_act_img  = make_act_image2d(cl_ctx_, scratch_.wo_act_i8,  N_q, M_pad, img_caps_);
    scratch_.fa_act_img  = make_act_image2d(cl_ctx_, scratch_.fa_i8,      K_h, M_pad, img_caps_);
    scratch_.dn_act_img  = make_act_image2d(cl_ctx_, scratch_.dn_i8,      I,   M_pad, img_caps_);
    if (!scratch_.qkv_act_img || !scratch_.wo_act_img || !scratch_.fa_act_img ||
        !scratch_.dn_act_img) {
      std::fprintf(stderr, "[qwen3-gpu] scratch act-image cache create failed\n");
      return false;
    }
  }

  scratch_max_M_ = M_pad;
  std::fprintf(stderr,
               "[qwen3-gpu] forward scratch pool allocated for max_M=%u "
               "(hidden=%u inter=%u hQ=%u hKV=%u d=%u)\n",
               M_pad, K_h, I, cfg_.num_heads_Q, cfg_.num_heads_KV,
               cfg_.head_dim);
  return true;
}

bool Qwen3Forward::forward_one_layer_v2(unsigned int layer_id,
                                        cl_mem in_fp32, cl_mem out_fp32,
                                        unsigned int position,
                                        unsigned int M) {
  if (M == 0) return false;
  if (!ensure_forward_scratch_allocated(M)) return false;
  if (layer_id >= layers_.size() || layers_[layer_id].wq.backing == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] forward_one_layer_v2(%u): not loaded\n",
                 layer_id);
    return false;
  }
  LayerWeights &lw = layers_[layer_id];
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  // #46f/h/l: NNTR_OHWI_IMG=1 enables the V image2d path (and the wv FC
  // fused-OHWI-scatter from #46l). Read once per process; same flag
  // controls scatter / wv dispatch / attention call.
  static const bool use_ohwi_img = []() {
    const char *e = std::getenv("NNTR_OHWI_IMG");
    return e && std::atoi(e) != 0;
  }();

  // #v8c-buf (paper §3.4 device specialization): NNTR_V8C_BUF=1 selects the
  // buffer-load v8c FC kernels (no sampled-image reads) for runtimes like
  // Intel NEO that cannot compile integer-coord read_imageui. When set we pass
  // the raw cl_mem buffers (scratch int8 act + weight backing buffer) into the
  // act_image/weight_image arg slots; the wrapper dispatches the _buf kernel.
  // Default 0 keeps the Adreno image2d path bit-identical.
  static const bool use_v8c_buf = []() {
    const char *e = std::getenv("NNTR_V8C_BUF");
    return e && std::atoi(e) != 0;
  }();
  // NNTR_FC_BUF=1 (probe, lib-side pair in gemm_int8_v8c_cl): FC GEMMs take
  // the buffer-load kernels while attention/images stay on the image path.
  static const bool use_fc_buf = [&]() {
    if (use_v8c_buf) return true;
    const char *e = std::getenv("NNTR_FC_BUF");
    return e && std::atoi(e) != 0;
  }();

  // v8c tile alignment: M_pad >= M, multiple of 4.
  const unsigned int M_pad = ((M + 3) / 4) * 4;
  const unsigned int K_h = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;
  const unsigned int I = cfg_.intermediate_size;

  // Profiling helpers (no-op when profile_stages_ is false).
  auto NOW = []() { return std::chrono::steady_clock::now(); };
  auto MS  = [](auto t1, auto t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
             .count() / 1000.0;
  };
  std::chrono::steady_clock::time_point t_stage;
  auto stage_begin = [&]() {
    if (profile_stages_) { clFinish(cl_q_); t_stage = NOW(); }
  };
  // Intel NEO: the command queue is CL_QUEUE_OUT_OF_ORDER and does NOT
  // auto-serialize data-dependent kernels across stages the way Adreno's
  // driver does in practice. Each stage here writes scratch buffers the
  // next stage reads (e.g. rmsnorm→quant→GEMM→...), and the cross-stage
  // bridges (KV/Q SVM map) read with no event dependency — so a later
  // stage can consume a not-yet-written buffer, producing inf/NaN. Insert
  // an ordering barrier at every stage boundary on Intel (gated by the
  // existing device-specialization signal NNTR_V8C_BUF). Barriers carry
  // NO math change and add no host stall (unlike clFinish); Adreno keeps
  // the original barrier-free fast path bit-for-bit.
  static const bool ooo_stage_barrier = []() {
    const char *e = std::getenv("NNTR_V8C_BUF");
    return e && std::atoi(e) != 0;
  }();
  auto stage_end_add = [&](double &accum) {
    if (profile_stages_) { clFinish(cl_q_); accum += MS(NOW(), t_stage); }
    else if (ooo_stage_barrier)
      clEnqueueBarrierWithWaitList(cl_q_, 0, nullptr, nullptr);
  };
  // Intel NEO intra-stage ordering: the wo/ffn blocks chain several
  // dependent kernels (copy→rmsnorm→quant→GEMM→elementwise) inside a
  // single profiling "stage". The stage-boundary barrier above does not
  // order WITHIN a stage, and the lib quant/GEMM wrappers enqueue without
  // a trailing barrier, so on the OOO queue a consumer can read a buffer
  // its producer has not finished writing → inf/NaN. bar() inserts a pure
  // ordering barrier; no-op on Adreno (gated by NNTR_V8C_BUF).
  auto bar = [&]() {
    if (ooo_stage_barrier)
      clEnqueueBarrierWithWaitList(cl_q_, 0, nullptr, nullptr);
  };
  // drainbar(): the 3 barriers before the sandwich post_attn/post_ffn rmsnorms
  // and before geglu were UNCONDITIONAL (OOO-Intel ordering), but on Adreno's
  // in-order queue they force a redundant full GPU drain. Measured (M=1024,
  // Adreno 840): removing them cut inter-kernel idle 161->58ms and lifted prefill
  // 834->908 TPS, TOKEN-IDENTICAL. So gate them on the OOO signal like bar()
  // (no-op on Adreno; kept for the Intel/NNTR_V8C_BUF path where the OOO queue
  // needs explicit ordering). NNTR_DRAINBAR=1 forces them back on for A/B.
  static const bool force_drainbar = []() {
    const char *e = std::getenv("NNTR_DRAINBAR");
    return e && std::atoi(e) != 0;
  }();
  auto drainbar = [&]() {
    if (ooo_stage_barrier || force_drainbar)
      clEnqueueBarrierWithWaitList(cl_q_, 0, nullptr, nullptr);
  };

  // (a) pad in_fp32 -> in_padded fp16 (cvt at boundary, #46m), then
  // attn_norm fp16. Residual stream is fp16 throughout (paper §3.7).
  // Intel NEO (OOO queue): the caller (main.cpp / prefill loop) enqueues the
  // input copy into in_fp32 WITHOUT a finish before calling us. On an
  // out-of-order queue our in_fp32->in_padded copy below can run before that
  // input copy lands → in_padded copies stale zeros → whole layer zero.
  // Order the caller's writes before our first read. Adreno-gated off.
  if (ooo_stage_barrier)
    clEnqueueBarrierWithWaitList(cl_q_, 0, nullptr, nullptr);
  stage_begin();
  float zero = 0.0f;
  const uint16_t zero_h = 0;
  // ML Drift reaudit (2026-06-12): residual chaining. The previous layer's
  // end-of-layer add already wrote residual_2 straight into in_padded, so
  // the 9.4 MB fill+CopyBuffer below is pure shuttling — skip it. Layer 0
  // always stages from the caller (flag never set across chains: the last
  // layer writes out_fp32 and leaves it false). NNTR_RESID_CHAIN=0 restores
  // the copy every layer.
  static const bool resid_chain = []() {
    const char *e = std::getenv("NNTR_RESID_CHAIN");
    return !e || std::atoi(e) != 0;
  }();
  const bool have_chained_input =
    resid_chain && layer_id > 0 && chain_in_padded_valid_;
  chain_in_padded_valid_ = false; // consumed (or invalidated)
  if (!have_chained_input) {
    // #47j: in_padded is FP32 (the inter-layer residual is kept in fp32 — the
    // last layer's massive activations exceed the fp16 max, and truncating the
    // residual to fp16 here would inf). Pad rows [M, M_pad) zeroed; [0, M) is a
    // straight fp32 copy of the caller's fp32 residual input.
    if (M_pad > M)
      clEnqueueFillBuffer(cl_q_, scratch_.in_padded, &zero, sizeof(float),
                          (size_t)M * K_h * sizeof(float),
                          (size_t)(M_pad - M) * K_h * sizeof(float), 0,
                          nullptr, nullptr);
    clEnqueueCopyBuffer(cl_q_, in_fp32, scratch_.in_padded, 0, 0,
                        (size_t)M * K_h * sizeof(float), 0, nullptr, nullptr);
  }
  bar();  // fill/copy(in_padded) -> attn rmsnorm
  // #80 NNTR_FUSE_NORMQUANT: fold attn-rmsnorm + QKV int8 act-quant into one
  // cooperative pass (no fp16 normed round-trip, one fewer dispatch).
  // Bit-identical to the split path. Default ON (=0 kill switch) since the
  // 2026-06-12 cooled interleaved A/B: together with NNTR_FUSE_ADDNORM,
  // M=1024 967->977 TPS, token-ID identical (logit lsb drift only), decode
  // and all other M unchanged. The earlier NEUTRAL verdict was taken in the
  // idle-dominated 841-era; the chain is ~98.5% GPU-bound now.
  static const bool fuse_normquant = []() {
    const char *e = std::getenv("NNTR_FUSE_NORMQUANT");
    return !e || std::atoi(e) != 0;
  }();
  cl_mem act_image = scratch_.qkv_act_img; // increment 2: cached view (no create)
  if (fuse_normquant) {
    auto kp = cl->registerClKernel(kFusedNormQuantKernel,
                                   "rmsnorm_f32in_quant_fused");
    uint16_t eps_h = f2h(cfg_.rms_norm_eps);
    int n_rows = (int)M_pad, W = (int)K_h;
    kp->SetKernelArguments(0, &scratch_.in_padded, sizeof(cl_mem));
    kp->SetKernelSVMArguments(1, lw.attn_norm_gamma_svm_fp16);
    kp->SetKernelArguments(2, &scratch_.qkv_act_i8, sizeof(cl_mem));
    kp->SetKernelArguments(3, &scratch_.qkv_act_scale, sizeof(cl_mem));
    kp->SetKernelArguments(4, &scratch_.qkv_act_zp, sizeof(cl_mem));
    kp->SetKernelArguments(5, &scratch_.qkv_act_rs, sizeof(cl_mem));
    kp->SetKernelArguments(6, &eps_h, sizeof(uint16_t));
    kp->SetKernelArguments(7, &n_rows, sizeof(int));
    kp->SetKernelArguments(8, &W, sizeof(int));
    constexpr size_t FNQ_LWS = 64;
    std::array<size_t, 1> gws = {FNQ_LWS * (size_t)n_rows};
    std::array<size_t, 1> lws = {FNQ_LWS};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    stage_end_add(timings_.qkv_quant_image_ms);
  } else {
    {
      auto kp = cl->registerClKernel(nntrainer::rmsnorm_fp16_kernel,
                                     "rmsnorm_f32in_f16out_coop");
      uint16_t eps_h = f2h(cfg_.rms_norm_eps);
      int n_rows = (int)M_pad, W = (int)K_h;
      kp->SetKernelArguments(0, &scratch_.in_padded, sizeof(cl_mem));
      kp->SetKernelArguments(1, &scratch_.attn_normed, sizeof(cl_mem));
      kp->SetKernelSVMArguments(2, lw.attn_norm_gamma_svm_fp16);
      kp->SetKernelArguments(3, &eps_h, sizeof(uint16_t));
      kp->SetKernelArguments(4, &n_rows, sizeof(int));
      kp->SetKernelArguments(5, &W, sizeof(int));
      constexpr size_t RMSN_LWS = 64;
      std::array<size_t, 1> gws = {RMSN_LWS * (size_t)n_rows};
      std::array<size_t, 1> lws = {RMSN_LWS};
      cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                            lws.data(), 0, nullptr, nullptr);
    }
    stage_end_add(timings_.pad_attn_norm_ms);
    // (b) shared act quant Q/K/V (fp16 act, #46m).
    stage_begin();
    nntrainer::quantize_act_v8c_fp16_cl(
      scratch_.attn_normed, scratch_.qkv_act_i8, scratch_.qkv_act_scale,
      scratch_.qkv_act_zp, scratch_.qkv_act_rs, M_pad, K_h);
    stage_end_add(timings_.qkv_quant_image_ms);
  }

  // (c) Q/K/V GEMMs against persistent y_q/y_k/y_v. 8/4/4: int8 weights for
  // q/k/v (is_int8) dispatch the int8×int8 kernel; int4 keeps v8c int8×int4.
  // Both wrappers share the same signature, so a function pointer selects.
  stage_begin();
  auto *qgemm = lw.wq.is_int8 ? &nntrainer::gemm_int8_int8_v8c_cl
                              : &gemm_int8_v8c_cl_legacy;
  auto *kgemm = lw.wk.is_int8 ? &nntrainer::gemm_int8_int8_v8c_cl
                              : &gemm_int8_v8c_cl_legacy;
  // Buffer path: act arg = raw int8 scratch buffer; weight arg = backing buffer.
  cl_mem qkv_act_arg = use_fc_buf ? scratch_.qkv_act_i8 : act_image;
  cl_mem wq_arg = use_fc_buf ? lw.wq.weight_buf : lw.wq.weight_image;
  cl_mem wk_arg = use_fc_buf ? lw.wk.weight_buf : lw.wk.weight_image;
  cl_mem wv_arg = use_fc_buf ? lw.wv.weight_buf : lw.wv.weight_image;
  qgemm(qkv_act_arg, wq_arg, scratch_.qkv_act_scale, lw.wq.scale_buf,
        scratch_.qkv_act_rs, scratch_.qkv_act_zp, lw.wq.row_sum_w_int4,
        scratch_.y_q, M_pad, N_q, K_h);
  kgemm(qkv_act_arg, wk_arg, scratch_.qkv_act_scale, lw.wk.scale_buf,
        scratch_.qkv_act_rs, scratch_.qkv_act_zp, lw.wk.row_sum_w_int4,
        scratch_.y_k, M_pad, N_kv, K_h);
  // #46l tried: when OHWI_IMG active, wv writes directly into
  // cache_v_buf_ohwi (gemm_int8_v8c_v_ohwi_cl), eliminating the separate
  // v_scatter_ohwi_t pass. BUT the profiler showed this direct GEMM is ~7x
  // SLOWER than the equivalent buffer GEMM (v_ohwi 6.8ms vs the same-shape
  // k_proj ~1ms at M=1024) because it does strided stores into the
  // TRANSPOSED [hKV,d,S] image layout. #65: default to the fast buffer GEMM
  // -> scratch_.y_v, then a cheap v_scatter_ohwi_t pass (mirrors the K path,
  // ~0.1ms). NNTR_V_DIRECT=1 restores the old fused direct write for A/B
  // (must produce token-identical output, only slower).
  static const bool v_direct = []() {
    const char *e = std::getenv("NNTR_V_DIRECT");
    return e && std::atoi(e) != 0;
  }();
  const bool v_gpu_scatter =
    use_ohwi_img && lw.cache_v_buf_ohwi != nullptr && !v_direct;
  if (use_ohwi_img && lw.cache_v_buf_ohwi != nullptr && v_direct) {
    auto *vohwi = lw.wv.is_int8 ? &nntrainer::gemm_int8_int8_v8c_v_ohwi_cl
                                : &nntrainer::gemm_int8_v8c_v_ohwi_cl;
    vohwi(qkv_act_arg, wv_arg, scratch_.qkv_act_scale,
          lw.wv.scale_buf, scratch_.qkv_act_rs, scratch_.qkv_act_zp,
          lw.wv.row_sum_w_int4, lw.cache_v_buf_ohwi, M_pad, N_kv, K_h,
          cfg_.head_dim, kv_cache_max_seq_len_, position, /*M_real=*/M);
  } else {
    // Fast path (default) + non-image fallback: concat write to scratch_.y_v.
    auto *vgemm = lw.wv.is_int8 ? &nntrainer::gemm_int8_int8_v8c_cl
                                : &gemm_int8_v8c_cl_legacy;
    vgemm(qkv_act_arg, wv_arg, scratch_.qkv_act_scale,
          lw.wv.scale_buf, scratch_.qkv_act_rs, scratch_.qkv_act_zp,
          lw.wv.row_sum_w_int4, scratch_.y_v, M_pad, N_kv, K_h);
  }
  // stage_end_add does its own clFinish when profiling; skip the
  // unconditional host stall in production. #46i.
  stage_end_add(timings_.qkv_gemm_ms);

  // (d) q_norm / k_norm in place. Multi-token: M token rows × num_heads
  //     heads × head_dim. Kernel iterates (B*C, H) where index is
  //     ((n*C+c)*H + h)*W. Set B=M, C=num_heads, H=1, W=head_dim →
  //     each WI covers one (token, head) head_dim-row.
  auto disp_qk_norm = [&](cl_mem io, void *gamma, unsigned int num_heads) {
    // #61: q/k-norm has W = head_dim = 128 (W8 = 16). At the default LWS=64 only
    // 16/64 WIs do work (75% idle) + a 6-round LDS barrier tree -> 35ms on Intel
    // (4x Adreno). On the Intel/buffer path use LWS=16 (perfect occupancy) +
    // subgroup-reduce (no LDS, no barriers). Adreno (use_v8c_buf unset) keeps
    // LWS=64 + LDS tree, bit-identical. Gate on head_dim==128 so LWS=16==W8.
    const bool sg = use_v8c_buf && cfg_.head_dim == 128;
    const char *copts = sg ? "-DRMSN_SG -DRMSN_LWS=16" : "";
    const size_t RMSN_LWS = sg ? 16 : 64;
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_fp16_kernel,
                                   "rmsnorm_cl_fp16_coop", copts);
    uint16_t eps_h = f2h(cfg_.rms_norm_eps);
    int n_rows = (int)M * (int)num_heads, W = (int)cfg_.head_dim;
    kp->SetKernelArguments(0, &io, sizeof(cl_mem));
    kp->SetKernelArguments(1, &io, sizeof(cl_mem));
    kp->SetKernelSVMArguments(2, gamma);
    kp->SetKernelArguments(3, &eps_h, sizeof(uint16_t));
    kp->SetKernelArguments(4, &n_rows, sizeof(int));
    kp->SetKernelArguments(5, &W, sizeof(int));
    std::array<size_t, 1> gws = {RMSN_LWS * (size_t)n_rows};
    std::array<size_t, 1> lws = {RMSN_LWS};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
  };
  stage_begin();
  // #63 Gemma2 has NO q/k-norm (RoPE applies to raw projected Q/K). Skip.
  if (cfg_.has_qk_norm) {
  disp_qk_norm(scratch_.y_q, lw.q_norm_gamma_svm_fp16, cfg_.num_heads_Q);
  disp_qk_norm(scratch_.y_k, lw.k_norm_gamma_svm_fp16, cfg_.num_heads_KV);
  // Intel NEO (OOO queue): q_norm/k_norm write y_q/y_k in place, and the
  // RoPE kernels below read+write the SAME y_q/y_k in place. These are four
  // data-dependent kernels enqueued back-to-back inside ONE stage; the
  // stage-boundary barrier (after RoPE) does not order WITHIN the stage. On
  // the OOO queue RoPE can start before its q_norm/k_norm producer finishes
  // → it reads the un-normalized (or partially written) head row, corrupting
  // Q/K. At the small, growing prefill M of the greedy loop the per-position
  // RoPE + per-row norm contention window actually opens (the decode M=1 and
  // the position-0 / repeated-row sweep masked it), so this surfaces as the
  // step~13 inf/NaN. bar() is pure ordering, no math; no-op on Adreno
  // (gated by NNTR_V8C_BUF) which serializes same-buffer commands in practice.
  bar();  // q_norm/k_norm(y_q,y_k) -> RoPE in-place reads
  } // end if (has_qk_norm)

  // RoPE on Q/K via batched LUT kernel (#45b / Path 4). Single
  // dispatch covers all M tokens × num_heads × half_d, looking up
  // cos/sin from the precomputed full LUT. Works for M=1 (decode)
  // and M>1 (prefill) without the per-token dispatch storm that
  // made prefill RoPE infeasible before.
  if (!precompute_rope_full_lut(cfg_.max_seq_len)) return false;
  if (!dispatch_rope_batched(scratch_.y_q, M, cfg_.num_heads_Q, position) ||
      !dispatch_rope_batched(scratch_.y_k, M, cfg_.num_heads_KV, position)) {
    return false;
  }
  // Intel NEO (OOO queue): the q_norm/k_norm/RoPE kernels write y_q/y_k in
  // place, but the following KV/Q bridges read y_q/y_k/y_v via blocking
  // clEnqueueMapBuffer with NO event dependency on those kernels — on an
  // out-of-order queue the map can run BEFORE the rmsnorm/rope completes,
  // so the bridge copies PRE-norm (O(1900)) Q/K into the SVM caches → the
  // QK dot overflows fp16 to +inf → softmax NaN → zero/NaN attention out.
  // A barrier here orders the in-place writes before the bridge reads.
  // Adreno serializes same-buffer commands in practice, so gate on the
  // existing Intel device-specialization signal (NNTR_V8C_BUF).
  if (use_v8c_buf)
    clEnqueueBarrierWithWaitList(cl_q_, 0, nullptr, nullptr);
  stage_end_add(timings_.qk_norm_rope_ms);  // #46i: no extra clFinish

  // (e) Write K, V (M rows) to this layer's SVM cache starting at
  //     `position`. y_k/y_v are [M_pad * N_kv] fp16; we copy only the
  //     valid M*N_kv prefix.
  //
  //   K cache: OHWI layout [hKV, max_seq_len, d] — sequential per-head.
  //            Per token row t we scatter to per-head offsets so the
  //            qk_matmul_f16_ohwi kernel can read each head's K
  //            contiguously (paper §3.8). This costs O(M*hKV) small
  //            memcpys per layer instead of one contiguous copy, but
  //            (i) M*hKV at 1024*8 = 8K copies of 256B is still ~2 MB
  //            total in <10 ms, and (ii) the attention speedup from
  //            cache-friendly access dwarfs the write-side cost.
  //   V cache: concat layout [max_seq_len, hKV*d] — V OHWI_T is task
  //            #46d (needs new sv_matmul kernel variant).
  const size_t kv_row_bytes = (size_t)N_kv * sizeof(uint16_t);
  const size_t kv_total_bytes = (size_t)M * kv_row_bytes;
  const size_t d_bytes = (size_t)cfg_.head_dim * sizeof(uint16_t);
  const size_t max_S = kv_cache_max_seq_len_;
  stage_begin();
  // K scatter:
  //   NNTR_OHWI_IMG=1 + cache_k_buf_ohwi → GPU k_scatter_ohwi kernel
  //                                        (no CPU sync map, like V).
  //                                        Optionally ALSO writes SVM if
  //                                        the K-image attention path is
  //                                        off (caller still reads SVM).
  //   default                           → CPU map + per-(t,h) memcpy.
  // #57: the CPU map+scatter that fills cache_k_svm is DEAD WORK whenever
  // attention reads the K image (cache_k_image_ohwi, a view of the
  // GPU-scattered cache_k_buf_ohwi) instead of SVM K. That is the default
  // fast path (NNTR_OHWI_IMG=1, NNTR_OHWI_KIMG default-on, NNTR_FLASH off):
  // the GPU k_scatter_ohwi above already populated the buffer the image
  // wraps, so the blocking map (a per-layer queue drain + M*hKV host memcpys
  // + refill bubble = the whole host_kv_ms) produces a cache_k_svm nobody
  // reads. Skip it. Keep it ONLY when a path that actually reads SVM K runs:
  // NNTR_FLASH (3387), NNTR_OHWI_KIMG=0 img_view (3416), or no K image.
  // The two env reads below MUST mirror use_flash (~3377) and use_k_image
  // (~3402) so the scatter and the attention dispatch never disagree.
  static const bool flash_reads_svm_k = []() {
    const char *e = std::getenv("NNTR_FLASH");
    return e && std::atoi(e) != 0;
  }();
  static const bool kimg_attn_on = []() {
    const char *e = std::getenv("NNTR_OHWI_KIMG");
    return e ? (std::atoi(e) != 0) : true;
  }();
  const bool gpu_k_scatter =
    use_ohwi_img && (lw.cache_k_buf_ohwi != nullptr);
  const bool need_k_svm_after_scatter =
    flash_reads_svm_k ||
    !(kimg_attn_on && lw.cache_k_image_ohwi != nullptr);

  // #89 Intel dispatch-idle fix: when attention reads the SVM K/V cache (Intel
  // flash path: use_v8c_buf, no image path), fill cache_k_svm / cache_v_svm with
  // GPU scatter kernels that stay on the command queue instead of the per-layer
  // host-blocking CPU map+memcpy (clEnqueueMapBuffer CL_TRUE + SVMMap CL_TRUE +
  // M*hKV memcpys = a queue drain that bubbles the GPU ~60 ms at M=1024). The
  // OHWI K scatter reuses k_scatter_ohwi with an SVM dst; V is a flat concat copy
  // into cache_v_svm at the position offset. NNTR_GPU_KV_SVM=0 restores the CPU
  // path (token-identical A/B). Adreno (use_ohwi_img) unaffected (flag false).
  static const bool gpu_kv_svm_env = []() {
    const char *e = std::getenv("NNTR_GPU_KV_SVM");
    if (e)
      return std::atoi(e) != 0;
    const char *b = std::getenv("NNTR_V8C_BUF");
    return b && std::atoi(b) != 0; // default ON for the Intel/buffer path
  }();
  const bool use_gpu_kv_svm =
    gpu_kv_svm_env && !gpu_k_scatter && !use_ohwi_img &&
    lw.cache_k_svm != nullptr && lw.cache_v_svm != nullptr;

  if (use_gpu_kv_svm) {
    // GPU OHWI K scatter straight into the SVM cache (dst = cache_k_svm). Same
    // OHWI [hKV, max_S, d] bytes the CPU path below writes; stays on the queue.
    auto kp = cl->registerClKernel(kKScatterOhwiKernel, "k_scatter_ohwi");
    if (!kp) {
      std::fprintf(stderr,
                   "[qwen3-gpu] layer %u: k_scatter_ohwi(SVM) register failed\n",
                   layer_id);
      return false;
    }
    cl_mem src_mem = scratch_.y_k;
    int Mi = (int)M, hKVi = (int)cfg_.num_heads_KV, di = (int)cfg_.head_dim,
        max_Si = (int)max_S, pos_i = (int)position;
    if (!kp->SetKernelArguments(0, &src_mem, sizeof(cl_mem)) ||
        !kp->SetKernelSVMArguments(1, lw.cache_k_svm) ||
        !kp->SetKernelArguments(2, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(3, &hKVi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &di, sizeof(int)) ||
        !kp->SetKernelArguments(5, &max_Si, sizeof(int)) ||
        !kp->SetKernelArguments(6, &pos_i, sizeof(int)))
      return false;
    constexpr size_t LWS_Z = 64;
    std::array<size_t, 3> gws = {(size_t)Mi, (size_t)hKVi,
                                 ((size_t)di + LWS_Z - 1) / LWS_Z * LWS_Z};
    std::array<size_t, 3> lws = {1, 1, LWS_Z};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    // GPU V concat scatter: copy y_v (cl_mem) -> cache_v_svm + position offset
    // (SVM). cache_v_svm is concat [max_S, hKV*d]; dst row n = position+t.
    auto vp = cl->registerClKernel(kCopySvmFp16Kernel, "copy_svm_to_clmem_fp16");
    if (!vp) {
      std::fprintf(stderr,
                   "[qwen3-gpu] layer %u: V scatter(SVM) register failed\n",
                   layer_id);
      return false;
    }
    int nv = (int)((size_t)M * N_kv);
    void *v_dst = static_cast<uint16_t *>(lw.cache_v_svm) +
                  (size_t)position * (size_t)N_kv;
    vp->SetKernelArguments(0, &scratch_.y_v, sizeof(cl_mem)); // src cl_mem
    vp->SetKernelSVMArguments(1, v_dst);                       // dst SVM
    vp->SetKernelArguments(2, &nv, sizeof(int));
    std::array<size_t, 1> vgws = {(((size_t)nv + 63) / 64) * 64};
    std::array<size_t, 1> vlws = {64};
    cl->command_queue_inst_.enqueueKernel(vp->GetKernel(), 1, vgws.data(),
                                          vlws.data(), 0, nullptr, nullptr);
  }

  if (gpu_k_scatter) {
    // GPU-side OHWI K scatter. Stays on the command queue with the prior
    // q_norm/k_norm/RoPE kernels — no host sync, no map.
    auto kp = cl->registerClKernel(kKScatterOhwiKernel, "k_scatter_ohwi");
    if (!kp) {
      std::fprintf(stderr,
                   "[qwen3-gpu] layer %u: k_scatter_ohwi register failed\n",
                   layer_id);
      return false;
    }
    cl_mem src_mem = scratch_.y_k;
    cl_mem dst_mem = lw.cache_k_buf_ohwi;
    int Mi = (int)M;
    int hKVi = (int)cfg_.num_heads_KV;
    int di = (int)cfg_.head_dim;
    int max_Si = (int)max_S;
    int pos_i = (int)position;
    if (!kp->SetKernelArguments(0, &src_mem, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &dst_mem, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(2, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(3, &hKVi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &di, sizeof(int)) ||
        !kp->SetKernelArguments(5, &max_Si, sizeof(int)) ||
        !kp->SetKernelArguments(6, &pos_i, sizeof(int))) {
      return false;
    }
    // x in stride-1, coalesce on innermost dim.
    constexpr size_t LWS_X = 1, LWS_Y = 1, LWS_Z = 64;
    const size_t gws_x = (size_t)Mi;
    const size_t gws_y = (size_t)hKVi;
    const size_t gws_z = ((size_t)di + LWS_Z - 1) / LWS_Z * LWS_Z;
    std::array<size_t, 3> gws = {gws_x, gws_y, gws_z};
    std::array<size_t, 3> lws = {LWS_X, LWS_Y, LWS_Z};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
  }

  if (v_gpu_scatter) {
    // #65 V scatter: scratch_.y_v [t,hKV,d] -> cache_v_buf_ohwi TRANSPOSED
    // [hKV,d,S] layout (same bytes the old direct v_ohwi GEMM produced, just
    // far faster: fast buffer GEMM above + this cheap scatter). Dispatch
    // geometry choice (read- vs write-coalesce) is decided below.
    auto kp = cl->registerClKernel(kVScatterOhwiTKernel, "v_scatter_ohwi_t");
    if (!kp) {
      std::fprintf(stderr,
                   "[qwen3-gpu] layer %u: v_scatter_ohwi_t register failed\n",
                   layer_id);
      return false;
    }
    cl_mem src_mem = scratch_.y_v;
    cl_mem dst_mem = lw.cache_v_buf_ohwi;
    int Mi = (int)M;
    int hKVi = (int)cfg_.num_heads_KV;
    int di = (int)cfg_.head_dim;
    int max_Si = (int)max_S;
    int pos_i = (int)position;
    if (!kp->SetKernelArguments(0, &src_mem, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &dst_mem, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(2, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(3, &hKVi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &di, sizeof(int)) ||
        !kp->SetKernelArguments(5, &max_Si, sizeof(int)) ||
        !kp->SetKernelArguments(6, &pos_i, sizeof(int))) {
      return false;
    }
    // Dispatch geometry: coalesce the READ on the d (gws_z) axis — src
    // concat [t,hKV,d] is contiguous in d, and Adreno absorbs the scattered
    // dst writes (stride max_S) in the write-combiner while scattered reads
    // stall the wave. Same-kernel measurement @M=1024 on Adreno 840:
    // layer-graph read-coalesced dispatch 197us/call vs this path's legacy
    // t-coalesced (write-coalesce) 982us/call = 5x. k_scatter above already
    // ships the same {1,1,64} 3D shape. NNTR_VSCAT_TCOAL=1 restores the
    // legacy write-coalesced dispatch for A/B (token-identical, only
    // slower: the kernel is a pure 1:1 scatter copy, no arithmetic).
    static const bool vscat_tcoal = []() {
      const char *e = std::getenv("NNTR_VSCAT_TCOAL");
      return e && std::atoi(e) != 0;
    }();
    if (vscat_tcoal) {
      constexpr size_t LWS_X = 64, LWS_Y = 1, LWS_Z = 1;
      const size_t gws_x = ((size_t)Mi + LWS_X - 1) / LWS_X * LWS_X;
      const size_t gws_y = (size_t)hKVi;
      const size_t gws_z = (size_t)di;
      std::array<size_t, 3> gws = {gws_x, gws_y, gws_z};
      std::array<size_t, 3> lws = {LWS_X, LWS_Y, LWS_Z};
      cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                            lws.data(), 0, nullptr, nullptr);
    } else {
      constexpr size_t LWS_Z = 64;
      const size_t gws_z = ((size_t)di + LWS_Z - 1) / LWS_Z * LWS_Z;
      std::array<size_t, 3> gws = {(size_t)Mi, (size_t)hKVi, gws_z};
      std::array<size_t, 3> lws = {1, 1, LWS_Z};
      cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                            lws.data(), 0, nullptr, nullptr);
    }
  }

  if ((!gpu_k_scatter || need_k_svm_after_scatter) && !use_gpu_kv_svm) {
    // CPU-side scatter path: writes cache_k_svm (and, in legacy NNTR_OHWI
    // _IMG-without-KIMG mode, also keeps SVM in sync for the SVM-K
    // attention kernel).
    auto _h0 = NOW();  // [host-timing] K scatter CPU map bridge (BLOCKS host)
    cl_int e;
    void *p_k = clEnqueueMapBuffer(cl_q_, scratch_.y_k, CL_TRUE, CL_MAP_READ,
                                   0, kv_total_bytes, 0, nullptr, nullptr,
                                   &e);
    const size_t cache_bytes_total =
      max_S * cfg_.num_heads_KV * cfg_.head_dim * sizeof(uint16_t);
    clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, lw.cache_k_svm,
                    cache_bytes_total, 0, nullptr, nullptr);
    const uint16_t *src = static_cast<const uint16_t *>(p_k);
    uint16_t *dst = static_cast<uint16_t *>(lw.cache_k_svm);
    const size_t hKV = cfg_.num_heads_KV;
    const size_t d = cfg_.head_dim;
    for (unsigned int t = 0; t < M; ++t) {
      for (size_t h = 0; h < hKV; ++h) {
        const uint16_t *src_row = src + (size_t)t * hKV * d + h * d;
        const size_t dst_off = h * max_S * d + ((size_t)position + t) * d;
        std::memcpy(dst + dst_off, src_row, d_bytes);
      }
    }
    clEnqueueSVMUnmap(cl_q_, lw.cache_k_svm, 0, nullptr, nullptr);
    clEnqueueUnmapMemObject(cl_q_, scratch_.y_k, p_k, 0, nullptr, nullptr);
    timings_.host_kv_ms += MS(NOW(), _h0);  // [host-timing] K bridge
  }
  // V scatter:
  //   default               → concat copy into cache_v_svm (sv_matmul_f16)
  //   NNTR_OHWI_IMG=1 (#46l)→ NO-OP: wv FC already wrote OHWI-reversed
  //                           directly into cache_v_buf_ohwi.
  if (!use_ohwi_img && !use_gpu_kv_svm) {
    auto _h0 = NOW();  // [host-timing] V write bridge (BLOCKS host)
    cl_int e;
    void *p_v = clEnqueueMapBuffer(cl_q_, scratch_.y_v, CL_TRUE, CL_MAP_READ,
                                   0, kv_total_bytes, 0, nullptr, nullptr,
                                   &e);
    void *dst = static_cast<uint8_t *>(lw.cache_v_svm) +
                (size_t)position * kv_row_bytes;
    clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, dst, kv_total_bytes, 0,
                    nullptr, nullptr);
    std::memcpy(dst, p_v, kv_total_bytes);
    clEnqueueSVMUnmap(cl_q_, dst, 0, nullptr, nullptr);
    clEnqueueUnmapMemObject(cl_q_, scratch_.y_v, p_v, 0, nullptr, nullptr);
    timings_.host_kv_ms += MS(NOW(), _h0);  // [host-timing] V bridge
  }
  // #46l NNTR_OHWI_IMG=1: wv FC already wrote OHWI-reversed into
  // cache_v_buf_ohwi (gemm_int8_v8c_v_ohwi_cl). No separate scatter.
  stage_end_add(timings_.kv_write_ms);  // #46i: no extra clFinish

  // (f) attention via SVM. Upload all M Q rows. N_kv (cache_to) =
  //     position + M (all rows just written are visible to attention).
  stage_begin();
  const size_t q_total_bytes = (size_t)M * N_q * sizeof(uint16_t);
  // Attention-mode decision — hoisted above the Q bridge so the bridge can
  // be skipped when the kvimg path binds Q/O directly. Per-branch docs stay
  // with the dispatch block below.
  static const int flash_env = []() {
    const char *e = std::getenv("NNTR_FLASH");
    return e ? (std::atoi(e) != 0 ? 1 : 0) : -1;  // -1 = unset
  }();
  const bool use_flash =
    (flash_env >= 0) ? (flash_env == 1) : (use_v8c_buf && !use_ohwi_img);
  static const bool use_flash_img = []() {
    const char *e = std::getenv("NNTR_FLASH_IMG");
    return e && std::atoi(e) != 0;
  }();
  static const bool use_k_image = []() {
    const char *e = std::getenv("NNTR_OHWI_KIMG");
    return e ? (std::atoi(e) != 0) : true;
  }();
  // ML Drift reaudit #4 (2026-06-12): on the default Adreno kvimg attention
  // path, bind Q and O directly as cl_mem inside the attention kernels
  // (q_clmem = y_q, o_clmem = o_fp32) and skip both copy_svm_to_clmem_fp16
  // round-trips (2 x M*N_q fp16 blits per layer). The lib wrappers grew the
  // q_clmem/o_clmem params for the layer-graph static-residency work; the
  // kernels are unchanged. NNTR_QO_BRIDGE=1 restores the copies for A/B.
  static const bool force_qo_bridge = []() {
    const char *e = std::getenv("NNTR_QO_BRIDGE");
    return e && std::atoi(e) != 0;
  }();
  static const bool q_host_bridge = []() {
    const char *e = std::getenv("NNTR_Q_HOST_BRIDGE");
    return e && std::atoi(e) != 0;
  }();
  const bool attn_flash_img_branch = use_flash_img && use_ohwi_img &&
    lw.cache_k_image_ohwi != nullptr && lw.cache_v_image_ohwi != nullptr;
  const bool attn_kvimg_branch = !attn_flash_img_branch && !use_flash &&
    use_ohwi_img && use_k_image && lw.cache_k_image_ohwi != nullptr;
  const bool attn_direct_qo =
    attn_kvimg_branch && !force_qo_bridge && !q_host_bridge;
  // #66 enqueue-batching: the Q bridge moves the qkv-GEMM output y_q (cl_mem)
  // into q_svm (SVM) that the attention kernel reads. The old path did a
  // host map(CL_TRUE)+4MB memcpy+clFinish PER LAYER — a host stall that
  // bubbles the GPU. Replace with a GPU copy that stays on the in-order
  // queue: copy_svm_to_clmem_fp16 is just dst[i]=src[i], so bind src=y_q
  // (cl_mem) + dst=q_svm (SVM). attention reads q_svm next on the same queue
  // (coherent — the o_svm path already does kernel-write-SVM->kernel-read-SVM).
  // NNTR_Q_HOST_BRIDGE=1 restores the old host path for A/B (token-identical).
  if (attn_direct_qo) {
    // kvimg attention binds y_q directly below — no Q bridge needed.
  } else if (q_host_bridge) {
    auto _h0 = NOW();  // [host-timing] Q SVM-upload bridge (BLOCKS host)
    cl_int e;
    void *p = clEnqueueMapBuffer(cl_q_, scratch_.y_q, CL_TRUE, CL_MAP_READ,
                                 0, q_total_bytes, 0, nullptr, nullptr, &e);
    clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, scratch_.q_svm,
                    q_total_bytes, 0, nullptr, nullptr);
    std::memcpy(scratch_.q_svm, p, q_total_bytes);
    clEnqueueSVMUnmap(cl_q_, scratch_.q_svm, 0, nullptr, nullptr);
    clEnqueueUnmapMemObject(cl_q_, scratch_.y_q, p, 0, nullptr, nullptr);
    clFinish(cl_q_);
    timings_.host_q_ms += MS(NOW(), _h0);  // [host-timing] Q bridge
  } else {
    auto kp =
      cl->registerClKernel(kCopySvmFp16Kernel, "copy_svm_to_clmem_fp16");
    if (!kp) {
      std::fprintf(stderr, "[qwen3-gpu] layer %u: Q copy register failed\n",
                   layer_id);
      return false;
    }
    int n = (int)((size_t)M * N_q);
    kp->SetKernelArguments(0, &scratch_.y_q, sizeof(cl_mem));   // src cl_mem
    kp->SetKernelSVMArguments(1, scratch_.q_svm);               // dst SVM
    kp->SetKernelArguments(2, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)n + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
  }
  // Attention dispatch:
  //   NNTR_FLASH=1   → fused flash-attention single kernel (online softmax,
  //                    no scores DRAM). Uses the SAME SVM buffers as the
  //                    _ohwi_cl fallback below (Q concat, K OHWI, V concat).
  //   default        → _ohwi_cl   (K OHWI, V concat — half-OHWI, 192 TPS@1K)
  //   NNTR_OHWI_IMG  → _ohwi_img_cl (K OHWI SVM, V cl_mem→image2d, #46f)
  // #59 device specialization: NNTR_FLASH explicit overrides; else default the
  // fused (vectorized) flash attention ON for the Intel/buffer path
  // (use_v8c_buf) when the image attention path is unavailable (!use_ohwi_img)
  // — vec-flash is +59% over the scalar 3-kernel there (Intel Arc M=1024
  // 727 -> 1153 TPS, token 7212). Adreno (use_v8c_buf unset, use_ohwi_img set)
  // keeps the image 3-kernel path: image attention beats flash 3x there.
  // #58: NNTR_FLASH_IMG=1 → fused single-kernel attention over the SAME two
  // OHWI images as the default 3-kernel kvimg_view path (K image + reversed-V
  // image), but with the score row kept in LDS instead of round-tripping the
  // 32 MB [H,M,N_kv] scores tensor through DRAM. 3 enqueues -> 1. Requires the
  // OHWI image path (both image views built). Falls through if unavailable.
  // (use_flash / use_flash_img / use_k_image are hoisted above the Q bridge.)
  // #89 With GPU K/V SVM scatter (no host-blocking drain) the Q bridge + K/V
  // scatter writes must be ordered before flash reads them on the Intel OOO
  // queue. bar() is a no-op on Adreno (in-order) and when use_gpu_kv_svm off.
  if (use_gpu_kv_svm)
    bar();
  bool attn_ok;
  if (attn_flash_img_branch) {
    attn_ok = nntrainer::fused_row_attention_f16_ohwi_img_cl(
      static_cast<const uint16_t *>(scratch_.q_svm),
      lw.cache_k_image_ohwi, lw.cache_v_image_ohwi,
      static_cast<uint16_t *>(scratch_.o_svm), M, position + M,
      cfg_.num_heads_Q, cfg_.num_heads_KV, cfg_.head_dim,
      kv_cache_max_seq_len_, true);
  } else if (use_flash) {
    // Fused flash path. K is cache_k_svm (OHWI [H_kv, max_S, d]) so we
    // pass max_seq_len as the OHWI row-stride; V is cache_v_svm (concat).
    attn_ok = nntrainer::flash_attention_prefill_f16_cl(
      static_cast<const uint16_t *>(scratch_.q_svm),
      static_cast<const uint16_t *>(lw.cache_k_svm),
      static_cast<const uint16_t *>(lw.cache_v_svm),
      static_cast<uint16_t *>(scratch_.o_svm), M, position + M,
      cfg_.num_heads_Q, cfg_.num_heads_KV, cfg_.head_dim,
      kv_cache_max_seq_len_, true, /*svm_inputs=*/true);
  } else if (use_ohwi_img) {
    // K image2d path (qk_matmul_f16_ohwi_img): default ON. The earlier
    // #46h "NEUTRAL" verdict was a measurement artifact of clFinish-
    // bracketed stage timing (queue catch-up). CL-event per-kernel
    // profiling (NNTR_OPENCL_PROFILING) shows the buffer qk_matmul does
    // SCALAR fp32 FMA + uncoalesced cross-WI K reads = 1381 ms at M=1024,
    // while the image2d variant does vectorized dot(float4) over texture-
    // cached K = 222 ms — a 6.2× kernel speedup and +35% prefill wall
    // (224→303 TPS @ M=1024), token-identical (758, logit 3.20271).
    // Opt out with NNTR_OHWI_KIMG=0 for ablation.
    if (attn_kvimg_branch) {
      attn_ok = nntrainer::two_conv_attention_prefill_f16_ohwi_kvimg_view_cl(
        static_cast<const uint16_t *>(scratch_.q_svm),
        lw.cache_k_image_ohwi, lw.cache_v_image_ohwi,
        static_cast<uint16_t *>(scratch_.o_svm), M, position + M,
        cfg_.num_heads_Q, cfg_.num_heads_KV, cfg_.head_dim,
        kv_cache_max_seq_len_, true,
        cfg_.attn_logit_softcap,  // #63 Gemma2 QK soft-cap (0 on Qwen3)
        // ML Drift reaudit #4: direct cl_mem Q/O binding (no SVM bridges).
        attn_direct_qo ? static_cast<void *>(scratch_.y_q) : nullptr,
        attn_direct_qo ? static_cast<void *>(scratch_.o_fp32) : nullptr);
    } else {
      attn_ok = nntrainer::two_conv_attention_prefill_f16_ohwi_img_view_cl(
        static_cast<const uint16_t *>(scratch_.q_svm),
        static_cast<const uint16_t *>(lw.cache_k_svm), lw.cache_v_image_ohwi,
        static_cast<uint16_t *>(scratch_.o_svm), M, position + M,
        cfg_.num_heads_Q, cfg_.num_heads_KV, cfg_.head_dim,
        kv_cache_max_seq_len_, true);
    }
  } else {
    attn_ok = nntrainer::two_conv_attention_prefill_f16_ohwi_cl(
      static_cast<const uint16_t *>(scratch_.q_svm),
      static_cast<const uint16_t *>(lw.cache_k_svm),
      static_cast<const uint16_t *>(lw.cache_v_svm),
      static_cast<uint16_t *>(scratch_.o_svm), M, position + M,
      cfg_.num_heads_Q, cfg_.num_heads_KV, cfg_.head_dim,
      kv_cache_max_seq_len_, true, /*svm_inputs=*/true);
  }
  stage_end_add(timings_.attn_dispatch_ms);
  if (!attn_ok) {
    std::fprintf(stderr,
                 "[qwen3-gpu] layer %u attention failed\n", layer_id);
    return false;
  }

  // (g) wo (#46m): o_svm is fp16 SVM; quantize directly without cvt.
  //     Then v8c FC -> wo_y_fp16, then add_fp16(in_padded + wo_y) -> residual_1.
  stage_begin();
  // GPU copy o_svm (fp16 SVM, attention output) → scratch_.o_fp32
  // (fp16 cl_mem, name kept). Async on the queue, no host stall.
  // Skipped when the kvimg attention wrote o_fp32 directly (attn_direct_qo).
  if (!attn_direct_qo) {
    auto _h0 = NOW();  // [host-timing] o_svm->o_fp32 copy_svm bridge (host enqueue)
    auto kp =
      cl->registerClKernel(kCopySvmFp16Kernel, "copy_svm_to_clmem_fp16");
    int n = (int)(M * N_q);
    kp->SetKernelSVMArguments(0, scratch_.o_svm);
    kp->SetKernelArguments(1, &scratch_.o_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &n, sizeof(int));
    std::array<size_t, 1> gws = {(((size_t)n + 63) / 64) * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    timings_.host_copy_svm_ms += MS(NOW(), _h0);  // [host-timing] copy_svm
  }
  // Intel NEO (OOO queue): order copy_svm_to_clmem_fp16 (writes o_fp32)
  // before quantize_act reads it. Adreno-gated off via use_v8c_buf.
  if (use_v8c_buf)
    clEnqueueBarrierWithWaitList(cl_q_, 0, nullptr, nullptr);
  nntrainer::quantize_act_v8c_fp16_cl(scratch_.o_fp32, scratch_.wo_act_i8,
                                      scratch_.wo_act_scale,
                                      scratch_.wo_act_zp, scratch_.wo_act_rs,
                                      M_pad, N_q);
  bar();  // quant(wo_act) -> wo GEMM
  cl_mem wo_act_image = scratch_.wo_act_img; // increment 2: cached view
  auto *ogemm = lw.wo.is_int8 ? &nntrainer::gemm_int8_int8_v8c_cl
                              : &gemm_int8_v8c_cl_legacy;
  cl_mem wo_act_arg = use_fc_buf ? scratch_.wo_act_i8 : wo_act_image;
  cl_mem wo_arg = use_fc_buf ? lw.wo.weight_buf : lw.wo.weight_image;
  ogemm(wo_act_arg, wo_arg, scratch_.wo_act_scale,
        lw.wo.scale_buf, scratch_.wo_act_rs, scratch_.wo_act_zp,
        lw.wo.row_sum_w_int4, scratch_.wo_y_fp16, M_pad, K_h, N_q);
  drainbar();  // FC(wo) -> sandwich post_attn rmsnorm (Adreno: in-order serializes)
  // #63 Gemma2 sandwich norm: residual_1 = x + post_attention_layernorm(attn_out).
  // Normalize wo_y in place (rmsnorm reads its own lanes before writing them, so
  // in==out is safe per work-item). Gated; Qwen3 adds the raw attn_out.
  if (cfg_.sandwich_norm) {
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_fp16_kernel,
                                   "rmsnorm_cl_fp16_coop");
    uint16_t eps_h = f2h(cfg_.rms_norm_eps);
    int n_rows = (int)M_pad, W = (int)K_h;
    kp->SetKernelArguments(0, &scratch_.wo_y_fp16, sizeof(cl_mem));
    kp->SetKernelArguments(1, &scratch_.wo_y_fp16, sizeof(cl_mem)); // in place
    kp->SetKernelSVMArguments(2, lw.post_attn_norm_gamma_svm_fp16);
    kp->SetKernelArguments(3, &eps_h, sizeof(uint16_t));
    kp->SetKernelArguments(4, &n_rows, sizeof(int));
    kp->SetKernelArguments(5, &W, sizeof(int));
    constexpr size_t RMSN_LWS = 64;
    std::array<size_t, 1> gws = {RMSN_LWS * (size_t)n_rows};
    std::array<size_t, 1> lws = {RMSN_LWS};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    drainbar();  // sandwich rmsnorm(wo_y in-place) -> fused_add reads it
                 // (Adreno in-order serializes; OOO needs the barrier)
  }
  // #71 fuse_add_norm: fold this residual add into the ffn-rmsnorm kernel
  // below (one cooperative pass computes residual_1 AND ffn_normed). When
  // off (NNTR_FUSE_ADDNORM=0 kill switch), the standalone add runs here as
  // before. Default ON since the 2026-06-12 cooled interleaved A/B — see
  // the fuse_normquant note above for the measurement.
  static const bool fuse_addnorm = []() {
    const char *e = std::getenv("NNTR_FUSE_ADDNORM");
    return !e || std::atoi(e) != 0;
  }();
  // #47j: in_padded(FP32) + wo_y(fp16) → residual_1 (FP32). The residual is
  // accumulated in fp32 because the last layer's massive activations exceed
  // the fp16 max (65504) -> inf -> NaN cascade through ffn_norm.
  if (!fuse_addnorm) {
    auto kp =
      cl->registerClKernel(kFusedAddH2fFp32Kernel, "fused_add_h2f_fp32");
    int n = (int)(M * K_h);
    kp->SetKernelArguments(0, &scratch_.in_padded, sizeof(cl_mem));  // fp32
    kp->SetKernelArguments(1, &scratch_.wo_y_fp16, sizeof(cl_mem));  // fp16
    kp->SetKernelArguments(2, &scratch_.residual_1, sizeof(cl_mem)); // fp32
    kp->SetKernelArguments(3, &n, sizeof(int));
    std::array<size_t, 1> gws = {(((size_t)M * K_h) + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
  }
  stage_end_add(timings_.wo_ms);  // #46i: no extra clFinish

  // (h) ffn block: pad residual_1 -> ffn_in_padded, rmsnorm -> ffn_normed,
  //     shared quant, ffn_up + ffn_gate, cvt fp16->fp32, swiglu, quant,
  //     ffn_down, cvt, add residual -> out_fp32 (caller-managed).
  stage_begin();
  // #47j: residual_1 and ffn_in_padded are FP32 (fp32 residual accumulation).
  uint16_t eps_h = f2h(cfg_.rms_norm_eps);
  if (fuse_addnorm && fuse_normquant) {
    // #80b 3-in-1: residual_1 = in_padded + wo_y (fp32) AND fa_i8/sc/zp/rs =
    // int8-quant(rmsnorm(residual_1)*gamma) in ONE pass — no fp16 ffn_normed
    // round-trip and no standalone quant dispatch. M_pad rows (covers the GEMM
    // pad rows; rows<M bit-identical to #71->quant).
    auto kp = cl->registerClKernel(kFusedAddRmsnormQuantKernel,
                                   "fused_add_rmsnorm_quant");
    int n_rows = (int)M_pad, W = (int)K_h;
    kp->SetKernelArguments(0, &scratch_.in_padded, sizeof(cl_mem));
    kp->SetKernelArguments(1, &scratch_.wo_y_fp16, sizeof(cl_mem));
    kp->SetKernelArguments(2, &scratch_.residual_1, sizeof(cl_mem));
    kp->SetKernelSVMArguments(3, lw.ffn_norm_gamma_svm_fp16);
    kp->SetKernelArguments(4, &scratch_.fa_i8, sizeof(cl_mem));
    kp->SetKernelArguments(5, &scratch_.fa_sc, sizeof(cl_mem));
    kp->SetKernelArguments(6, &scratch_.fa_zp, sizeof(cl_mem));
    kp->SetKernelArguments(7, &scratch_.fa_rs, sizeof(cl_mem));
    kp->SetKernelArguments(8, &eps_h, sizeof(uint16_t));
    kp->SetKernelArguments(9, &n_rows, sizeof(int));
    kp->SetKernelArguments(10, &W, sizeof(int));
    std::array<size_t, 1> gws = {(size_t)64 * n_rows};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
  } else if (fuse_addnorm) {
    // #71 one cooperative pass: residual_1 = in_padded + wo_y (fp32, for the
    // later residual add) AND ffn_normed = rmsnorm(residual_1)*gamma (fp16).
    // Replaces standalone add + pad-fill + copy + rmsnorm.
    auto kp = cl->registerClKernel(kFusedAddRmsnormKernel, "fused_add_rmsnorm");
    int W = (int)K_h;
    kp->SetKernelArguments(0, &scratch_.in_padded, sizeof(cl_mem));   // a fp32
    kp->SetKernelArguments(1, &scratch_.wo_y_fp16, sizeof(cl_mem));   // b fp16
    kp->SetKernelArguments(2, &scratch_.residual_1, sizeof(cl_mem));  // resid fp32
    kp->SetKernelArguments(3, &scratch_.ffn_normed, sizeof(cl_mem));  // normed fp16
    kp->SetKernelSVMArguments(4, lw.ffn_norm_gamma_svm_fp16);
    kp->SetKernelArguments(5, &eps_h, sizeof(uint16_t));
    kp->SetKernelArguments(6, &W, sizeof(int));
    std::array<size_t, 1> gws = {(size_t)64 * M};  // M rows × LWS 64
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    // pad rows of ffn_normed = rmsnorm(0) = 0
    if (M_pad > M) {
      const uint16_t zero_h = 0;
      clEnqueueFillBuffer(cl_q_, scratch_.ffn_normed, &zero_h, sizeof(uint16_t),
                          (size_t)M * K_h * sizeof(uint16_t),
                          (size_t)(M_pad - M) * K_h * sizeof(uint16_t), 0,
                          nullptr, nullptr);
    }
  } else {
    // Only zero the pad rows [M, M_pad); the copy below overwrites [0, M).
    if (M_pad > M)
      clEnqueueFillBuffer(cl_q_, scratch_.ffn_in_padded, &zero,
                          sizeof(float), (size_t)M * K_h * sizeof(float),
                          (size_t)(M_pad - M) * K_h * sizeof(float), 0,
                          nullptr, nullptr);
    clEnqueueCopyBuffer(cl_q_, scratch_.residual_1, scratch_.ffn_in_padded, 0,
                        0, (size_t)M * K_h * sizeof(float), 0, nullptr,
                        nullptr);
    bar();  // fill/copy(ffn_in_padded) -> ffn rmsnorm
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_fp16_kernel,
                                   "rmsnorm_f32in_f16out_coop");
    int n_rows = (int)M_pad, W = (int)K_h;
    kp->SetKernelArguments(0, &scratch_.ffn_in_padded, sizeof(cl_mem));
    kp->SetKernelArguments(1, &scratch_.ffn_normed, sizeof(cl_mem));
    kp->SetKernelSVMArguments(2, lw.ffn_norm_gamma_svm_fp16);
    kp->SetKernelArguments(3, &eps_h, sizeof(uint16_t));
    kp->SetKernelArguments(4, &n_rows, sizeof(int));
    kp->SetKernelArguments(5, &W, sizeof(int));
    constexpr size_t RMSN_LWS = 64;
    std::array<size_t, 1> gws = {RMSN_LWS * (size_t)n_rows};
    std::array<size_t, 1> lws = {RMSN_LWS};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
  }
  if (!(fuse_addnorm && fuse_normquant)) {
    bar();  // ffn rmsnorm(ffn_normed) -> quant
    nntrainer::quantize_act_v8c_fp16_cl(scratch_.ffn_normed, scratch_.fa_i8,
                                        scratch_.fa_sc, scratch_.fa_zp,
                                        scratch_.fa_rs, M_pad, K_h);
  }
  bar();  // quant(fa) -> ffn_up/gate GEMM
  cl_mem fa_image = scratch_.fa_act_img; // increment 2: cached view
  cl_mem fa_act_arg = use_fc_buf ? scratch_.fa_i8 : fa_image;
  cl_mem fup_arg = use_fc_buf ? lw.ffn_up.weight_buf : lw.ffn_up.weight_image;
  cl_mem fgate_arg =
    use_fc_buf ? lw.ffn_gate.weight_buf : lw.ffn_gate.weight_image;
  nntrainer::gemm_int8_v8c_cl(fa_act_arg, fup_arg, scratch_.fa_sc,
                              lw.ffn_up.scale_buf, scratch_.fa_rs,
                              scratch_.fa_zp, lw.ffn_up.row_sum_w_int4,
                              scratch_.up_fp16, M_pad, I, K_h);
  nntrainer::gemm_int8_v8c_cl(fa_act_arg, fgate_arg,
                              scratch_.fa_sc, lw.ffn_gate.scale_buf,
                              scratch_.fa_rs, scratch_.fa_zp,
                              lw.ffn_gate.row_sum_w_int4, scratch_.gate_fp16,
                              M_pad, I, K_h);
  // GPU-side barrier instead of host clFinish: keeps serialization
  // for the OOO queue's next consumers (cvt h2f) without stalling
  // the host. #46i — biggest single overhead lever in FFN block.
  drainbar();  // FC(gate/up) -> geglu (Adreno: in-order serializes same-buffer)
  auto disp_cvt = [&](cl_mem hin, cl_mem fout, unsigned int n) {
    auto kp = cl->registerClKernel(kConvertFp16ToFp32Kernel, "cvt_h2f");
    int ni = (int)n;
    kp->SetKernelArguments(0, &hin, sizeof(cl_mem));
    kp->SetKernelArguments(1, &fout, sizeof(cl_mem));
    kp->SetKernelArguments(2, &ni, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)n + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
  };
  // #47i: swiglu product silu(gate)*up in FP32. gate/up individually fit
  // fp16, but their product overflows fp16 (e.g. silu(30)*3000 = 90000 >
  // 65504) at the last layer's massive activations -> inf -> the per-row
  // int8 quant scale becomes NaN -> down GEMM spreads NaN to all 1024 dims
  // -> garbage prefill output for those rows. Computing the product in fp32
  // (h2f kernel) lets the per-row int8 quant absorb the large magnitude.
  // #81 NNTR_FUSE_GEGLUQUANT=1 (Gemma2): collapse geglu + the fp32 act-quant
  // into one cooperative pass, dropping the [M,9216] fp32 swiglu_out round-trip
  // (157MB->81MB DRAM @ M=1024). Bit-identical dn_i8/sc/zp/rs (same fp32 geglu,
  // same quant math). Default off; flip on after a token-identical A/B win.
  static const bool fuse_gegluquant = []() {
    const char *e = std::getenv("NNTR_FUSE_GEGLUQUANT");
    return e ? (std::atoi(e) != 0) : true;  // default ON: bit-identical, +2% @M1024
  }();
  if (cfg_.mlp_geglu && fuse_gegluquant) {
    auto kp = cl->registerClKernel(kFusedGegluQuantKernel, "fused_geglu_quant");
    int n_rows = (int)M_pad, Kw = (int)I;
    kp->SetKernelArguments(0, &scratch_.up_fp16, sizeof(cl_mem));    // gate_proj(x)
    kp->SetKernelArguments(1, &scratch_.gate_fp16, sizeof(cl_mem));  // up_proj(x)
    kp->SetKernelArguments(2, &scratch_.dn_i8, sizeof(cl_mem));
    kp->SetKernelArguments(3, &scratch_.dn_sc, sizeof(cl_mem));
    kp->SetKernelArguments(4, &scratch_.dn_zp, sizeof(cl_mem));
    kp->SetKernelArguments(5, &scratch_.dn_rs, sizeof(cl_mem));
    kp->SetKernelArguments(6, &n_rows, sizeof(int));
    kp->SetKernelArguments(7, &Kw, sizeof(int));
    std::array<size_t, 1> gws = {(size_t)64 * (size_t)n_rows};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    bar();  // fused geglu+quant(dn) -> ffn_down GEMM
  } else {
  {
    // #63 Gemma2 uses GeGLU (gelu_tanh) instead of SwiGLU (silu).
    auto kp = cfg_.mlp_geglu
      ? cl->registerClKernel(kFusedGegluH2fFp32Kernel, "fused_geglu_h2f_fp32")
      : cl->registerClKernel(kFusedSwigluH2fFp32Kernel, "fused_swiglu_h2f_fp32");
    int n = (int)(M * I);
    // #64 FFN gate/up de-swap. The loader uses an INVERTED naming: lw.ffn_up
    // holds gate_proj weights and lw.ffn_gate holds up_proj weights (see
    // load_layer + converter records h=gate_proj,i=up_proj). The GEMMs above
    // therefore wrote gate_proj(x) into scratch_.up_fp16 and up_proj(x) into
    // scratch_.gate_fp16. The kernel applies the activation to its arg0
    // ("gate" param). (Geg/Swi)GLU must activate gate_proj, so arg0 MUST be
    // scratch_.up_fp16 (=gate_proj output) and arg1 scratch_.gate_fp16
    // (=up_proj output). Previously these were swapped -> act(up)*gate, which
    // silently corrupted EVERY model's FFN (never-coherent C++ forward).
    kp->SetKernelArguments(0, &scratch_.up_fp16, sizeof(cl_mem));    // gate_proj(x)
    kp->SetKernelArguments(1, &scratch_.gate_fp16, sizeof(cl_mem));  // up_proj(x)
    kp->SetKernelArguments(2, &scratch_.swiglu_out, sizeof(cl_mem)); // fp32
    kp->SetKernelArguments(3, &n, sizeof(int));
    std::array<size_t, 1> gws = {(((size_t)M * I) + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
  }
  bar();  // swiglu(swiglu_out) -> quant
  nntrainer::quantize_act_v8c_fp32_cl(scratch_.swiglu_out, scratch_.dn_i8,
                                      scratch_.dn_sc, scratch_.dn_zp,
                                      scratch_.dn_rs, M_pad, I);
  bar();  // quant(dn) -> ffn_down GEMM
  }  // #81 end NNTR_FUSE_GEGLUQUANT else (2-kernel geglu + act-quant path)
  cl_mem dn_image = scratch_.dn_act_img; // increment 2: cached view
  cl_mem dn_act_arg = use_fc_buf ? scratch_.dn_i8 : dn_image;
  cl_mem fdown_arg =
    use_fc_buf ? lw.ffn_down.weight_buf : lw.ffn_down.weight_image;
  nntrainer::gemm_int8_v8c_cl(dn_act_arg, fdown_arg,
                              scratch_.dn_sc, lw.ffn_down.scale_buf,
                              scratch_.dn_rs, scratch_.dn_zp,
                              lw.ffn_down.row_sum_w_int4, scratch_.dn_fp16,
                              M_pad, K_h, I);
  drainbar();  // FC(down) -> sandwich post_ffn rmsnorm (Adreno: in-order serializes)
  // #63 Gemma2 sandwich norm: out = residual_1 + post_feedforward_layernorm(ffn_out).
  // Normalize dn_fp16 in place. Gated; Qwen3 adds the raw ffn_out.
  if (cfg_.sandwich_norm) {
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_fp16_kernel,
                                   "rmsnorm_cl_fp16_coop");
    uint16_t eps_h = f2h(cfg_.rms_norm_eps);
    int n_rows = (int)M_pad, W = (int)K_h;
    kp->SetKernelArguments(0, &scratch_.dn_fp16, sizeof(cl_mem));
    kp->SetKernelArguments(1, &scratch_.dn_fp16, sizeof(cl_mem)); // in place
    kp->SetKernelSVMArguments(2, lw.post_ffn_norm_gamma_svm_fp16);
    kp->SetKernelArguments(3, &eps_h, sizeof(uint16_t));
    kp->SetKernelArguments(4, &n_rows, sizeof(int));
    kp->SetKernelArguments(5, &W, sizeof(int));
    constexpr size_t RMSN_LWS = 64;
    std::array<size_t, 1> gws = {RMSN_LWS * (size_t)n_rows};
    std::array<size_t, 1> lws = {RMSN_LWS};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    drainbar();  // sandwich rmsnorm(dn_fp16 in-place) -> fused_add reads it
                 // (Adreno in-order serializes; OOO needs the barrier)
  }
  // #47j: end-of-layer boundary. residual_1 (FP32) + dn_fp16 → residual_2.
  // Residual chaining: non-last layers write residual_2 straight into
  // scratch_.in_padded (the next layer's input slot — its last reader, the
  // residual_1 fold above, has already run on this in-order queue), so the
  // next layer skips its 9.4 MB staging copy. The last layer writes the
  // caller's out_fp32 as before (out_norm/lm_head read it). Mid-chain
  // out_fp32 ping-pong buffers are never read by the caller.
  {
    const bool resid_chain_here = [&]() {
      static const bool on = []() {
        const char *e = std::getenv("NNTR_RESID_CHAIN");
        return !e || std::atoi(e) != 0;
      }();
      return on && (layer_id + 1 < cfg_.num_layers);
    }();
    cl_mem resid2_dst = resid_chain_here ? scratch_.in_padded : out_fp32;
    auto kp =
      cl->registerClKernel(kFusedAddH2fFp32Kernel, "fused_add_h2f_fp32");
    int n = (int)(M * K_h);
    kp->SetKernelArguments(0, &scratch_.residual_1, sizeof(cl_mem)); // fp32
    kp->SetKernelArguments(1, &scratch_.dn_fp16, sizeof(cl_mem));    // fp16
    kp->SetKernelArguments(2, &resid2_dst, sizeof(cl_mem));
    kp->SetKernelArguments(3, &n, sizeof(int));
    std::array<size_t, 1> gws = {(((size_t)M * K_h) + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    if (resid_chain_here)
      chain_in_padded_valid_ = true;
  }
  stage_end_add(timings_.ffn_ms);  // #46i: no extra clFinish
  if (profile_stages_) timings_.calls += 1;

  // increment 2: act images are cached in scratch_ (freed with the scratch
  // buffers in ensure_forward_scratch_allocated/destructor); no release here.
  return true;
}

bool Qwen3Forward::load_output_norm(size_t file_offset) {
  if (output_norm_gamma_svm_ != nullptr) return true;
  const size_t bytes = (size_t)cfg_.hidden_size * sizeof(float);
  if (file_offset + bytes > weight_bytes_) {
    std::fprintf(stderr,
                 "[qwen3-gpu] output_norm offset %zu + %zu > file %zu\n",
                 file_offset, bytes, weight_bytes_);
    return false;
  }
  const float *src =
    reinterpret_cast<const float *>(weight_mmap_ + file_offset);
  std::fprintf(stderr,
               "[qwen3-gpu] output_norm off=%zu first 8: %g %g %g %g %g %g "
               "%g %g\n", file_offset, src[0], src[1], src[2], src[3],
               src[4], src[5], src[6], src[7]);
  if (!load_norm_to_svm_fp32(cl_ctx_, cl_q_, src, cfg_.hidden_size,
                             &output_norm_gamma_svm_, "output_norm"))
    return false;
  // #46m: also load fp16 version for rmsnorm_cl_fp16.
  if (!load_hidden_norm_to_svm_fp16(cl_ctx_, cl_q_, src, cfg_.hidden_size,
                                    &output_norm_gamma_svm_fp16_,
                                    "output_norm"))
    return false;
  std::fprintf(stderr,
               "[qwen3-gpu] output_norm gamma -> SVM (fp32, %zu B)\n", bytes);
  return true;
}

bool Qwen3Forward::run_output_norm(cl_mem inout_fp32) {
  if (output_norm_gamma_svm_ == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] output_norm gamma not loaded\n");
    return false;
  }
  // Wrap inout in a [M_pad=1, hidden] view; rmsnorm.cl expects H rows
  // of W columns and we just need one row.
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  auto kp = cl->registerClKernel(nntrainer::rmsnorm_kernel, "rmsnorm_cl");
  float eps = cfg_.rms_norm_eps;
  int H = 1, W = (int)cfg_.hidden_size;
  if (!kp ||
      !kp->SetKernelArguments(0, &inout_fp32, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(1, &inout_fp32, sizeof(cl_mem)) ||
      !kp->SetKernelSVMArguments(2, output_norm_gamma_svm_) ||
      !kp->SetKernelArguments(3, &eps, sizeof(float)) ||
      !kp->SetKernelArguments(4, &H, sizeof(int)) ||
      !kp->SetKernelArguments(5, &W, sizeof(int))) {
    std::fprintf(stderr, "[qwen3-gpu] output_norm args failed\n");
    return false;
  }
  std::array<size_t, 1> gws = {64};
  std::array<size_t, 1> lws = {64};
  cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                        lws.data(), 0, nullptr, nullptr);
  clFinish(cl_q_);
  return true;
}

cl_mem Qwen3Forward::embedding_lookup_to_fp32_clmem(unsigned int token_id) {
  if (weight_mmap_ == nullptr || cl_ctx_ == nullptr) return nullptr;
  if (token_id >= cfg_.vocab_size) {
    std::fprintf(stderr,
                 "[qwen3-gpu] embedding_lookup: token_id %u >= vocab %u\n",
                 token_id, cfg_.vocab_size);
    return nullptr;
  }
  const unsigned int H = cfg_.hidden_size;
  // Q6_K row stride: H * 210 / 256 bytes. For H=1024 -> 4 blocks * 210 =
  // 840 bytes per row. Embedding starts at file offset 0.
  if (H % Q6_K_BLOCK_ELTS != 0) {
    std::fprintf(stderr,
                 "[qwen3-gpu] embedding_lookup: hidden %u not multiple of "
                 "Q6_K block (%zu)\n", H, Q6_K_BLOCK_ELTS);
    return nullptr;
  }
  const size_t row_bytes = (H / Q6_K_BLOCK_ELTS) * Q6_K_BLOCK_BYTES;
  const size_t row_off = (size_t)token_id * row_bytes;
  if (row_off + row_bytes > embed_table_bytes()) {
    std::fprintf(stderr,
                 "[qwen3-gpu] embedding_lookup: row offset %zu past embed "
                 "table %zu\n", row_off, embed_table_bytes());
    return nullptr;
  }
  std::vector<float> host_row(H);
  nntrainer::dequantize_row_q6_K(weight_mmap_ + row_off, host_row.data(),
                                 (int64_t)H);
  // #63 Gemma2 scales the INPUT embedding by sqrt(hidden) (~48). The tied
  // lm_head output projection is NOT scaled, so this is applied here only.
  if (cfg_.embed_scale > 0.0f) {
    const float s = cfg_.embed_scale;
    for (unsigned int i = 0; i < H; ++i) host_row[i] *= s;
  }
  std::fprintf(stderr,
               "[qwen3-gpu] embedding[%u] dequant first 8:", token_id);
  for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %g", host_row[i]);
  std::fprintf(stderr, "\n");

  cl_int err = CL_SUCCESS;
  cl_mem buf = clCreateBuffer(cl_ctx_,
                              CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              (size_t)H * sizeof(float), host_row.data(),
                              &err);
  if (err != CL_SUCCESS || buf == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] embedding_lookup: clCreateBuffer err=%d\n",
                 err);
    return nullptr;
  }
  return buf;
}

int Qwen3Forward::run_lm_head_and_argmax_cpu(cl_mem post_norm_fp32) {
  if (weight_mmap_ == nullptr || cl_ctx_ == nullptr ||
      post_norm_fp32 == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] lm_head: not initialized\n");
    return -1;
  }
  const unsigned int H = cfg_.hidden_size;
  const unsigned int V = cfg_.vocab_size;
  if (H % Q6_K_BLOCK_ELTS != 0) {
    std::fprintf(stderr,
                 "[qwen3-gpu] lm_head: hidden %u not multiple of Q6_K\n", H);
    return -1;
  }
  // Read post_norm to host (fp32 [hidden]).
  std::vector<float> hidden(H);
  cl_int err = clEnqueueReadBuffer(cl_q_, post_norm_fp32, CL_TRUE, 0,
                                   (size_t)H * sizeof(float), hidden.data(),
                                   0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] lm_head: hidden read err=%d\n", err);
    return -1;
  }

  // Per-vocab row: dequant Q6_K -> 1024 fp32, dot with hidden, store
  // logit. Argmax incrementally to avoid the [vocab=151936] logits
  // buffer. Embedding row stride = (H/256) * 210 bytes.
  const size_t row_bytes = (H / Q6_K_BLOCK_ELTS) * Q6_K_BLOCK_BYTES;
  const uint8_t *embed_base = weight_mmap_;

  std::vector<float> dequant_row(H);
  // Optional full fp32 logit dump (first call only): NNTR_DUMP_LOGITS=<path>.
  const char *dump_path = std::getenv("NNTR_DUMP_LOGITS");
  std::vector<float> all_logits;
  if (dump_path != nullptr) all_logits.resize((size_t)V);
  float best_logit = -std::numeric_limits<float>::infinity();
  int best_token = -1;
  for (unsigned int v = 0; v < V; ++v) {
    nntrainer::dequantize_row_q6_K(embed_base + (size_t)v * row_bytes,
                                   dequant_row.data(), (int64_t)H);
    float dot = 0.0f;
    for (unsigned int i = 0; i < H; ++i)
      dot += dequant_row[i] * hidden[i];
    if (dump_path != nullptr) all_logits[(size_t)v] = dot;
    if (dot > best_logit) {
      best_logit = dot;
      best_token = (int)v;
    }
  }
  if (dump_path != nullptr) {
    static bool dumped = false;
    if (!dumped) {
      std::FILE *fp = std::fopen(dump_path, "wb");
      if (fp != nullptr) {
        std::fwrite(all_logits.data(), sizeof(float), (size_t)V, fp);
        std::fclose(fp);
        dumped = true;
        std::fprintf(stderr,
                     "[NNTR_DUMP_LOGITS] wrote %u fp32 logits to %s\n", V,
                     dump_path);
      } else {
        std::fprintf(stderr, "[NNTR_DUMP_LOGITS] failed to open %s\n",
                     dump_path);
      }
    }
  }
  std::fprintf(stderr,
               "[qwen3-gpu] lm_head: argmax token=%d logit=%g (vocab=%u)\n",
               best_token, best_logit, V);
  return best_token;
}

int Qwen3Forward::run_lm_head_and_argmax_gpu(cl_mem post_norm_fp32) {
  if (weight_mmap_ == nullptr || cl_ctx_ == nullptr ||
      post_norm_fp32 == nullptr)
    return -1;
  const unsigned int H = cfg_.hidden_size;
  const unsigned int V = cfg_.vocab_size;
  if (H % Q6_K_BLOCK_ELTS != 0)
    return -1;
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  if (cl == nullptr)
    return -1;
  const size_t row_bytes = (H / Q6_K_BLOCK_ELTS) * Q6_K_BLOCK_BYTES;
  if (row_bytes > 1890) // kernel stages one row in a 1904 B LDS array
    return -1;
  const size_t table_bytes = (size_t)V * row_bytes;
  cl_int err = CL_SUCCESS;
  if (lm_head_q6k_buf_ == nullptr) {
    auto _u0 = std::chrono::steady_clock::now();
    // +16 B slack: the kernel's coalesced row staging over-reads up to 14 B
    // past the LAST row's end.
    lm_head_q6k_buf_ = clCreateBuffer(cl_ctx_, CL_MEM_READ_ONLY,
                                      table_bytes + 16, nullptr, &err);
    if (err == CL_SUCCESS && lm_head_q6k_buf_ != nullptr) {
      err = clEnqueueWriteBuffer(cl_q_, lm_head_q6k_buf_, CL_TRUE, 0,
                                 table_bytes, weight_mmap_, 0, nullptr,
                                 nullptr);
    }
    if (err != CL_SUCCESS || lm_head_q6k_buf_ == nullptr) {
      std::fprintf(stderr,
                   "[qwen3-gpu] lm_head(gpu): table upload failed err=%d "
                   "(%zu MB) — falling back to CPU\n",
                   err, table_bytes >> 20);
      if (lm_head_q6k_buf_ != nullptr)
        clReleaseMemObject(lm_head_q6k_buf_);
      lm_head_q6k_buf_ = nullptr;
      return -1;
    }
    const double up_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - _u0)
        .count() /
      1000.0;
    std::fprintf(stderr,
                 "[qwen3-gpu] lm_head(gpu): Q6_K table uploaded (%zu MB, "
                 "%.1f ms, one-time)\n",
                 table_bytes >> 20, up_ms);
  }
  if (lm_head_logits_buf_ == nullptr) {
    lm_head_logits_buf_ = clCreateBuffer(
      cl_ctx_, CL_MEM_READ_WRITE, (size_t)V * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS || lm_head_logits_buf_ == nullptr) {
      lm_head_logits_buf_ = nullptr;
      return -1;
    }
  }
  auto kp = cl->registerClKernel(kQ6kGemvKernel, "q6k_gemv_lmhead");
  if (!kp)
    return -1;
  int Vi = (int)V, Hi = (int)H;
  if (!kp->SetKernelArguments(0, &lm_head_q6k_buf_, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(1, &post_norm_fp32, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(2, &lm_head_logits_buf_, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(3, &Vi, sizeof(int)) ||
      !kp->SetKernelArguments(4, &Hi, sizeof(int)))
    return -1;
  // One 64-WI workgroup per vocab row.
  std::array<size_t, 1> gws = {(size_t)V * 64};
  std::array<size_t, 1> lws = {64};
  cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                        lws.data(), 0, nullptr, nullptr);
  // GPU stage-1 argmax: 64 candidates instead of a 1 MB logits readback.
  if (lm_head_bestv_buf_ == nullptr) {
    lm_head_bestv_buf_ = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                        64 * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) lm_head_bestv_buf_ = nullptr;
  }
  if (lm_head_besti_buf_ == nullptr) {
    lm_head_besti_buf_ = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                        64 * sizeof(int), nullptr, &err);
    if (err != CL_SUCCESS) lm_head_besti_buf_ = nullptr;
  }
  auto ap = cl->registerClKernel(kQ6kGemvKernel, "argmax_f32_stage1");
  if (!ap || lm_head_bestv_buf_ == nullptr || lm_head_besti_buf_ == nullptr)
    return -1;
  int Vi2 = (int)V;
  if (!ap->SetKernelArguments(0, &lm_head_logits_buf_, sizeof(cl_mem)) ||
      !ap->SetKernelArguments(1, &Vi2, sizeof(int)) ||
      !ap->SetKernelArguments(2, &lm_head_bestv_buf_, sizeof(cl_mem)) ||
      !ap->SetKernelArguments(3, &lm_head_besti_buf_, sizeof(cl_mem)))
    return -1;
  std::array<size_t, 1> agws = {64 * 64};
  std::array<size_t, 1> alws = {64};
  cl->command_queue_inst_.enqueueKernel(ap->GetKernel(), 1, agws.data(),
                                        alws.data(), 0, nullptr, nullptr);
  float bestv[64];
  int besti[64];
  err = clEnqueueReadBuffer(cl_q_, lm_head_bestv_buf_, CL_FALSE, 0,
                            sizeof(bestv), bestv, 0, nullptr, nullptr);
  cl_int err2 = clEnqueueReadBuffer(cl_q_, lm_head_besti_buf_, CL_TRUE, 0,
                                    sizeof(besti), besti, 0, nullptr, nullptr);
  if (err != CL_SUCCESS || err2 != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] lm_head(gpu): argmax read err=%d/%d\n",
                 err, err2);
    return -1;
  }
  // Final pick over 64 candidates — same tie rule as the CPU scan
  // (strict >, ties keep the smaller index).
  float best_logit = bestv[0];
  int best_token = besti[0];
  for (int g = 1; g < 64; ++g) {
    if (bestv[g] > best_logit ||
        (bestv[g] == best_logit && besti[g] < best_token)) {
      best_logit = bestv[g];
      best_token = besti[g];
    }
  }
  std::fprintf(stderr,
               "[qwen3-gpu] lm_head: argmax token=%d logit=%g (vocab=%u) "
               "[gpu]\n",
               best_token, best_logit, V);
  return best_token;
}

int Qwen3Forward::run_lm_head_and_argmax(cl_mem post_norm_fp32) {
  static const bool gpu_on = []() {
    const char *e = std::getenv("NNTR_LMHEAD_GPU");
    return !e || std::atoi(e) != 0;
  }();
  // The CPU path owns the NNTR_DUMP_LOGITS debug dump.
  if (gpu_on && std::getenv("NNTR_DUMP_LOGITS") == nullptr) {
    int r = run_lm_head_and_argmax_gpu(post_norm_fp32);
    if (r >= 0)
      return r;
    std::fprintf(stderr,
                 "[qwen3-gpu] lm_head(gpu) unavailable — CPU fallback\n");
  }
  return run_lm_head_and_argmax_cpu(post_norm_fp32);
}

bool Qwen3Forward::allocate_layer0_kv_cache_svm() {
  if (layer0_cache_k_svm_ != nullptr && layer0_cache_v_svm_ != nullptr)
    return true;
  if (cl_ctx_ == nullptr) return false;
  // [max_seq_len * hKV * d] fp16 — concat layout (same as the
  // existing CausalLM KVCacheManager for the non-OHWI path).
  const size_t cache_bytes =
    (size_t)cfg_.max_seq_len * cfg_.num_heads_KV * cfg_.head_dim *
    sizeof(uint16_t);

  auto alloc_one = [this, cache_bytes](void **dst, const char *tag) -> bool {
    if (*dst != nullptr) return true;
    *dst = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, cache_bytes, 0);
    if (*dst == nullptr) {
      std::fprintf(stderr, "[qwen3-gpu] %s SVMAlloc(%zu B) failed\n", tag,
                   cache_bytes);
      return false;
    }
    // Zero-init via SVM map + memset (clEnqueueSVMMemFill is OpenCL 2.0
    // but unavailable through some Adreno OpenCL loaders; map+memset
    // works universally on coarse-grained SVM).
    cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, *dst,
                                 cache_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::fprintf(stderr, "[qwen3-gpu] %s SVMMap zero err=%d\n", tag, err);
      return false;
    }
    std::memset(*dst, 0, cache_bytes);
    err = clEnqueueSVMUnmap(cl_q_, *dst, 0, nullptr, nullptr);
    clFinish(cl_q_);
    return err == CL_SUCCESS;
  };
  if (!alloc_one(&layer0_cache_k_svm_, "cache_K") ||
      !alloc_one(&layer0_cache_v_svm_, "cache_V"))
    return false;

  std::fprintf(stderr,
               "[qwen3-gpu] KV cache SVM allocated: each %zu MB "
               "(max_seq=%u, hKV=%u, d=%u, fp16)\n",
               cache_bytes / (1024 * 1024), cfg_.max_seq_len,
               cfg_.num_heads_KV, cfg_.head_dim);
  return true;
}

} // namespace causallm_gpu

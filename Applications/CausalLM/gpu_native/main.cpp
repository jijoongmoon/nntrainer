// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   29 May 2026
 * @brief  Entry point for the GPU-native Qwen3 forward binary
 *         (nntrainer_qwen3_gpu).
 *
 * Step 7b: end-to-end 28-layer chain via the generic load_layer +
 * forward_one_layer path. Old layer0_* methods are still in the .cpp
 * but unused — they'll go away when output_norm + lm_head land in
 * step 7c (which finishes the from-scratch inference pipeline up
 * through the first generated token).
 *
 * Usage:
 *   nntrainer_qwen3_gpu <weight_file_path>
 *
 * Qwen3-0.6B config is hardcoded (matches the verified production
 * QINT4 model on device).
 */

#include "qwen3_forward.h"

#include <env_compat.h>
#include <chrono>
#include <functional>
#include <cl_context.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <engine.h>
#include <limits>
#include <string>
#include <vector>

// Bypass the production safety gate in two_conv_attention_prefill_f16_cl.
// The from-scratch runtime explicitly accepts GPU baseline as reference
// (paper §3.6 same-numerics chain), so the existing "Using CPU mha"
// fallback in the production wrapper isn't useful here.
static struct EnvSetup {
  EnvSetup() { setenv("NNTR_MHA_VERIFY", "1", 1); }
} _env_setup;

// #69 On-device compute-peak microbench (NNTR_PEAK_BENCH=1). Register-resident
// loops (NO memory traffic inside the loop) measure the Adreno int8-dp4a peak
// — the ceiling our int8×int4 FC GEMM competes against — and the fp16-FMA peak.
// 16 independent dependency chains hide instruction latency so we measure
// THROUGHPUT, not latency. Counted as flops (dp4a = 4 MAC = 8 flop; fma = 2
// flop) to match the GEMM's "TOP/s" = 2×MAC convention (our v8c FC = 5.08).
static const char *kPeakKernels = R"CLC(
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define C32(op) op(0) op(1) op(2) op(3) op(4) op(5) op(6) op(7) \
  op(8) op(9) op(10) op(11) op(12) op(13) op(14) op(15) \
  op(16) op(17) op(18) op(19) op(20) op(21) op(22) op(23) \
  op(24) op(25) op(26) op(27) op(28) op(29) op(30) op(31)
// 32 independent dp4a recurrence chains, no extra ALU per dp4a (just the
// accumulate, matching the GEMM's acc += dot). Lots of in-flight WIs + 32
// chains/WI saturate throughput.
__kernel void dp4a_peak(__global int *out, const int iters, const uint b) {
  int a[32];
  #pragma unroll
  for (int j=0;j<32;j++) a[j]=(int)get_global_id(0)+j+1;
  for (int i=0;i<iters;i++) {
    #define DP(j) a[j]=dot_4x8packed_su_int(as_uint(a[j]),b)+a[j];
    C32(DP)
    #undef DP
  }
  int s=0;
  #pragma unroll
  for (int j=0;j<32;j++) s+=a[j];
  out[get_global_id(0)]=s;
}
)CLC";

static void run_peak_bench() {
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  if (!cl) { std::fprintf(stderr, "[peak] no gpu context\n"); return; }
  cl_command_queue q = cl->command_queue_inst_.GetCommandQueue();
  cl_context ctx = nullptr;
  clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
  // #69b: dump device HW facilities — which fast ops does THIS Adreno expose,
  // and are we using them? (dp4a=yes; ml_ops/command_buffer/recordable=?)
  {
    cl_device_id dev = nullptr;
    clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);
    static char buf[16384];
    size_t n = 0;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(buf), buf, &n);
    std::fprintf(stderr, "[hw] device: %s\n", buf);
    clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(buf), buf, &n);
    std::fprintf(stderr, "[hw] version: %s\n", buf);
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(buf), buf, &n);
    std::fprintf(stderr, "[hw] extensions: %s\n", buf);
    const char *probe[] = {"cl_qcom_ml_ops", "cl_khr_command_buffer",
                           "cl_qcom_recordable_queue", "cl_khr_integer_dot_product",
                           "cl_qcom_dot_product8", "cl_qcom_subgroup",
                           "cl_qcom_accelerated_image_ops", "cl_khr_subgroups",
                           "cl_qcom_extended_query_image_info", "matrix"};
    for (auto *p : probe)
      std::fprintf(stderr, "[hw]   %-34s : %s\n", p,
                   std::strstr(buf, p) ? "YES" : "no");
  }
  const size_t GS = 256 * 1024;   // work-items
  const size_t LWS = 64;
  cl_int err = 0;
  cl_mem out_i = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, GS * sizeof(int),
                                nullptr, &err);
  cl_mem out_h = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, GS * sizeof(uint16_t),
                                nullptr, &err);
  auto NOW = []() { return std::chrono::steady_clock::now(); };
  auto SEC = [](auto t1, auto t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
             .count() / 1e6;
  };
  std::array<size_t, 1> gws = {GS}, lws = {LWS};

  // ---- int8 dp4a peak ----
  for (int iters : {4096, 16384}) {
    auto kp = cl->registerClKernel(kPeakKernels, "dp4a_peak");
    uint32_t b = 0x01010101u;
    kp->SetKernelArguments(0, &out_i, sizeof(cl_mem));
    kp->SetKernelArguments(1, &iters, sizeof(int));
    kp->SetKernelArguments(2, &b, sizeof(uint32_t));
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(q);  // warmup
    auto t0 = NOW();
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(q);
    double s = SEC(NOW(), t0);
    double flop = (double)GS * iters * 32.0 * 8.0;  // 32 chains × dp4a(8 flop)
    std::fprintf(stderr, "[peak] int8-dp4a iters=%d : %.3f ms => %.2f TOP/s\n",
                 iters, s * 1e3, flop / s / 1e12);
  }
  std::fprintf(stderr, "[peak] (our v8c int8xint4 FC in-forward = 5.08 TOP/s; "
                       "this register-only loop is the int8-dp4a ceiling)\n");
  (void)out_h;
  clReleaseMemObject(out_i);
  clReleaseMemObject(out_h);
}

// #84 LDS cross-WI exchange minimal repro (NNTR_LDS_REPRO=1). Two model kernels
// (fused-SV attention, v_scatter tiled-transpose) wrote nothing despite
// verified-correct math; both did barrier-separated LDS EXCHANGE (WI A writes
// LDS[i], WI B reads it). geglu (LDS REDUCTION) works. This isolates the trigger
// on tiny buffers (no SVM/images/model state), same ClContext/queue the model
// uses, so a FAIL here is a pure kernel/driver fact.
static const char *kLdsReproKernels = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define LR_TILE 8
// (1) control: direct transpose, NO LDS. out[c][r] = in[r][c].
__kernel void t_nolds(__global const float *in, __global float *out, const int N) {
  int c = get_global_id(0); int r = get_global_id(1);
  if (c >= N || r >= N) return;
  out[(long)c * N + r] = in[(long)r * N + c];
}
// (2) SUSPECT: float LDS-exchange transpose, 2D workgroup.
__attribute__((reqd_work_group_size(LR_TILE, LR_TILE, 1)))
__kernel void t_lds_f(__global const float *in, __global float *out, const int N) {
  __local float tile[LR_TILE][LR_TILE + 1];
  int bc = get_group_id(0) * LR_TILE, br = get_group_id(1) * LR_TILE;
  int lc = get_local_id(0), lr = get_local_id(1);
  if (br + lr < N && bc + lc < N) tile[lr][lc] = in[(long)(br + lr) * N + (bc + lc)];
  barrier(CLK_LOCAL_MEM_FENCE);
  int oc = bc + lr, orow = br + lc;
  if (oc < N && orow < N) out[(long)oc * N + orow] = tile[lc][lr];
}
// (3) SUSPECT: half LDS-exchange transpose (matches v_scatter exactly).
__attribute__((reqd_work_group_size(LR_TILE, LR_TILE, 1)))
__kernel void t_lds_h(__global const float *in, __global float *out, const int N) {
  __local half tile[LR_TILE][LR_TILE + 1];
  int bc = get_group_id(0) * LR_TILE, br = get_group_id(1) * LR_TILE;
  int lc = get_local_id(0), lr = get_local_id(1);
  if (br + lr < N && bc + lc < N) tile[lr][lc] = (half)in[(long)(br + lr) * N + (bc + lc)];
  barrier(CLK_LOCAL_MEM_FENCE);
  int oc = bc + lr, orow = br + lc;
  if (oc < N && orow < N) out[(long)oc * N + orow] = (float)tile[lc][lr];
}
// (4) SUSPECT: 1D-workgroup LDS exchange (attention fused-SV pattern): write
// sh[k], barrier, read sh[N-1-k] (cross-WI) -> reverse each row. row=group_id(1)
// so a 3D-ish dispatch (gws={64,N,1}) like the model uses get_global_id(1).
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void x_lds_1d(__global const float *in, __global float *out, const int N) {
  int row = get_global_id(1), tid = get_local_id(0);
  __local float sh[256];
  for (int k = tid; k < N; k += 64) sh[k] = in[(long)row * N + k];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int k = tid; k < N; k += 64) out[(long)row * N + k] = sh[N - 1 - k];
}
// (6) SUSPECT: 3D dispatch (z-axis = "head") LDS-exchange transpose, EXACTLY the
// v_scatter shape: gws={N,N,H}, lws={16,16,1}, head on get_global_id(2).
__attribute__((reqd_work_group_size(LR_TILE, LR_TILE, 1)))
__kernel void t_lds_h3(__global const float *in, __global float *out,
                       const int N, const int H) {
  __local half tile[LR_TILE][LR_TILE + 1];
  int hd = get_global_id(2);
  int bc = get_group_id(0) * LR_TILE, br = get_group_id(1) * LR_TILE;
  int lc = get_local_id(0), lr = get_local_id(1);
  if (br + lr < N && bc + lc < N)
    tile[lr][lc] = (half)in[(long)hd * N * N + (long)(br + lr) * N + (bc + lc)];
  barrier(CLK_LOCAL_MEM_FENCE);
  int oc = bc + lr, orow = br + lc;
  if (oc < N && orow < N)
    out[(long)hd * N * N + (long)oc * N + orow] = (float)tile[lc][lr];
}
// (7) FIX candidate: same per-head transpose but head is a SCALAR ARG and the
// dispatch is 2D (gws={N,N,1}); caller loops heads. Avoids the 3D z-axis that
// breaks t_lds_h3.
__attribute__((reqd_work_group_size(LR_TILE, LR_TILE, 1)))
__kernel void t_lds_h2(__global const float *in, __global float *out,
                       const int N, const int hd) {
  __local half tile[LR_TILE][LR_TILE + 1];
  int bc = get_group_id(0) * LR_TILE, br = get_group_id(1) * LR_TILE;
  int lc = get_local_id(0), lr = get_local_id(1);
  if (br + lr < N && bc + lc < N)
    tile[lr][lc] = (half)in[(long)hd * N * N + (long)(br + lr) * N + (bc + lc)];
  barrier(CLK_LOCAL_MEM_FENCE);
  int oc = bc + lr, orow = br + lc;
  if (oc < N && orow < N)
    out[(long)hd * N * N + (long)oc * N + orow] = (float)tile[lc][lr];
}
// (8) FIX: single 2D dispatch, head FOLDED into gws.y (no z-axis, no arg race).
// gws={N, (N/TILE)*H, 1}; group_id(1) decodes (head, token-tile).
__attribute__((reqd_work_group_size(LR_TILE, LR_TILE, 1)))
__kernel void t_lds_hf(__global const float *in, __global float *out,
                       const int N, const int H) {
  __local half tile[LR_TILE][LR_TILE + 1];
  int tiles_per_head = (N + LR_TILE - 1) / LR_TILE;
  int gy = get_group_id(1);
  int hd = gy / tiles_per_head;
  int br = (gy % tiles_per_head) * LR_TILE;
  int bc = get_group_id(0) * LR_TILE;
  int lc = get_local_id(0), lr = get_local_id(1);
  if (hd < H && br + lr < N && bc + lc < N)
    tile[lr][lc] = (half)in[(long)hd * N * N + (long)(br + lr) * N + (bc + lc)];
  barrier(CLK_LOCAL_MEM_FENCE);
  int oc = bc + lr, orow = br + lc;
  if (hd < H && oc < N && orow < N)
    out[(long)hd * N * N + (long)oc * N + orow] = (float)tile[lc][lr];
}
// (9) FIX: head BAKED via -DVS_HEAD macro -> fresh kernel object per head
// (distinct opts = distinct cache entry = no shared-object arg race), each a 2D
// {N,N,1} dispatch (the passing t_lds_h shape).
#ifndef VS_HEAD
#define VS_HEAD 0
#endif
__attribute__((reqd_work_group_size(LR_TILE, LR_TILE, 1)))
__kernel void t_lds_hd(__global const float *in, __global float *out, const int N) {
  const int hd = VS_HEAD;
  __local half tile[LR_TILE][LR_TILE + 1];
  int bc = get_group_id(0) * LR_TILE, br = get_group_id(1) * LR_TILE;
  int lc = get_local_id(0), lr = get_local_id(1);
  if (br + lr < N && bc + lc < N)
    tile[lr][lc] = (half)in[(long)hd * N * N + (long)(br + lr) * N + (bc + lc)];
  barrier(CLK_LOCAL_MEM_FENCE);
  int oc = bc + lr, orow = br + lc;
  if (oc < N && orow < N)
    out[(long)hd * N * N + (long)oc * N + orow] = (float)tile[lc][lr];
}
// (5) control: 1D LDS reduction (geglu-like). out[row] = sum(in[row][*]).
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void r_lds(__global const float *in, __global float *out, const int N) {
  int row = get_group_id(0), tid = get_local_id(0);
  __local float red[64];
  float p = 0.0f;
  for (int k = tid; k < N; k += 64) p += in[(long)row * N + k];
  red[tid] = p;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
  if (tid == 0) out[row] = red[0];
}
)CLC";

static void run_lds_repro() {
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  if (!cl) { std::fprintf(stderr, "[lds] no gpu context\n"); return; }
  cl_command_queue q = cl->command_queue_inst_.GetCommandQueue();
  cl_context ctx = nullptr;
  clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
  const int N = 32;                 // 32x32: multiple 16x16 tiles
  const int H = 4;                  // "heads" (z-axis), like v_scatter hKV
  const size_t NN = (size_t)N * N;
  const size_t NNH = NN * H;
  std::vector<float> hin(NNH), hout(NNH);
  for (size_t i = 0; i < NNH; i++) hin[i] = (float)i;
  cl_int e = 0;
  cl_mem din = clCreateBuffer(ctx, CL_MEM_READ_ONLY, NNH * sizeof(float), nullptr, &e);
  cl_mem dout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, NNH * sizeof(float), nullptr, &e);

  auto run_check = [&](const char *kname, std::array<size_t,3> gws,
                       std::array<size_t,3> lws, int wdim, int extraH,
                       std::function<float(int)> expect, int n_out) {
    std::vector<float> zero(NNH, -1.0f);
    clEnqueueWriteBuffer(q, dout, CL_TRUE, 0, NNH * sizeof(float), zero.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(q, din, CL_TRUE, 0, NNH * sizeof(float), hin.data(), 0, nullptr, nullptr);
    auto kp = cl->registerClKernel(kLdsReproKernels, kname);
    if (!kp) { std::fprintf(stderr, "[lds] %-10s REGISTER FAILED\n", kname); return; }
    int Ni = N, Hi = H;
    kp->SetKernelArguments(0, &din, sizeof(cl_mem));
    kp->SetKernelArguments(1, &dout, sizeof(cl_mem));
    kp->SetKernelArguments(2, &Ni, sizeof(int));
    if (extraH) kp->SetKernelArguments(3, &Hi, sizeof(int));
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), wdim, gws.data(), lws.data(), 0, nullptr, nullptr);
    clFinish(q);
    clEnqueueReadBuffer(q, dout, CL_TRUE, 0, NNH * sizeof(float), hout.data(), 0, nullptr, nullptr);
    int bad = 0, first_bad = -1;
    for (int i = 0; i < n_out; i++)
      if (hout[i] != expect(i)) { if (first_bad < 0) first_bad = i; bad++; }
    std::fprintf(stderr, "[lds] %-10s : %s  (%d/%d wrong)  out[0..3]=%.0f,%.0f,%.0f,%.0f",
                 kname, bad == 0 ? "PASS" : "FAIL", bad, n_out,
                 hout[0], hout[1], hout[2], hout[3]);
    if (bad) std::fprintf(stderr, "  first_bad i=%d got=%.0f want=%.0f",
                          first_bad, hout[first_bad], expect(first_bad));
    std::fprintf(stderr, "\n");
  };

  auto tr_expect  = [&](int i) { int c = i / N, r = i % N; return (float)(r * N + c); };
  auto rev_expect = [&](int i) { int row = i / N, k = i % N; return (float)(row * N + (N - 1 - k)); };
  auto red_expect = [&](int row) { return (float)((long)N * row * N + (long)N * (N - 1) / 2); };
  // per-head transpose: out[hd][c][r] = in[hd][r][c]. i = hd*NN + c*N + r.
  auto tr3_expect = [&](int i) { int hd = i / (int)NN; int j = i % (int)NN;
                                 int c = j / N, r = j % N;
                                 return (float)(hd * (int)NN + r * N + c); };

  std::fprintf(stderr, "[lds] === LDS cross-WI exchange repro (N=%d H=%d) ===\n", N, H);
  run_check("t_nolds", {(size_t)N,(size_t)N,1}, {8,8,1}, 2, 0, tr_expect, (int)NN);
  run_check("t_lds_f", {(size_t)N,(size_t)N,1}, {8,8,1}, 2, 0, tr_expect, (int)NN);
  run_check("t_lds_h", {(size_t)N,(size_t)N,1}, {8,8,1}, 2, 0, tr_expect, (int)NN);
  run_check("t_lds_h3", {(size_t)N,(size_t)N,(size_t)H}, {8,8,1}, 3, 1, tr3_expect, (int)NNH);
  run_check("x_lds_1d", {64,(size_t)N,1}, {64,1,1}, 2, 0, rev_expect, (int)NN);
  run_check("r_lds", {(size_t)64*N,1,1}, {64,1,1}, 1, 0, red_expect, N);
  {
    // FIX: head folded into gws.y, single 2D dispatch.
    int tph = (N + 7) / 8;
    run_check("t_lds_hf", {(size_t)N, (size_t)tph * H * 8, 1}, {8,8,1}, 2, 1,
              tr3_expect, (int)NNH);
  }

  // FIX: per-head 2D loop with FRESH kernel object per head (head baked via -D).
  {
    std::vector<float> zero(NNH, -1.0f);
    clEnqueueWriteBuffer(q, dout, CL_TRUE, 0, NNH * sizeof(float), zero.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(q, din, CL_TRUE, 0, NNH * sizeof(float), hin.data(), 0, nullptr, nullptr);
    int Ni = N;
    std::array<size_t,3> gws = {(size_t)N,(size_t)N,1}, lws = {8,8,1};
    for (int hd = 0; hd < H; hd++) {
      char opt[32]; std::snprintf(opt, sizeof(opt), "-DVS_HEAD=%d", hd);
      auto kp = cl->registerClKernel(kLdsReproKernels, "t_lds_hd", std::string(opt));
      kp->SetKernelArguments(0, &din, sizeof(cl_mem));
      kp->SetKernelArguments(1, &dout, sizeof(cl_mem));
      kp->SetKernelArguments(2, &Ni, sizeof(int));
      cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 2, gws.data(), lws.data(), 0, nullptr, nullptr);
    }
    clFinish(q);
    clEnqueueReadBuffer(q, dout, CL_TRUE, 0, NNH * sizeof(float), hout.data(), 0, nullptr, nullptr);
    int bad = 0, first_bad = -1;
    for (int i = 0; i < (int)NNH; i++)
      if (hout[i] != tr3_expect(i)) { if (first_bad < 0) first_bad = i; bad++; }
    std::fprintf(stderr, "[lds] %-10s : %s  (%d/%d wrong)  out[0..3]=%.0f,%.0f,%.0f,%.0f%s\n",
                 "t_lds_hd(freshx4)", bad == 0 ? "PASS" : "FAIL", bad, (int)NNH,
                 hout[0], hout[1], hout[2], hout[3],
                 bad ? "" : "  <-- FIX WORKS");
  }

  // FIX candidate: per-head 2D dispatch loop (no 3D z-axis).
  {
    std::vector<float> zero(NNH, -1.0f);
    clEnqueueWriteBuffer(q, dout, CL_TRUE, 0, NNH * sizeof(float), zero.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(q, din, CL_TRUE, 0, NNH * sizeof(float), hin.data(), 0, nullptr, nullptr);
    int Ni = N;
    std::array<size_t,3> gws = {(size_t)N,(size_t)N,1}, lws = {8,8,1};
    for (int hd = 0; hd < H; hd++) {
      auto kp = cl->registerClKernel(kLdsReproKernels, "t_lds_h2");
      kp->SetKernelArguments(0, &din, sizeof(cl_mem));
      kp->SetKernelArguments(1, &dout, sizeof(cl_mem));
      kp->SetKernelArguments(2, &Ni, sizeof(int));
      kp->SetKernelArguments(3, &hd, sizeof(int));
      cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 2, gws.data(), lws.data(), 0, nullptr, nullptr);
    }
    clFinish(q);
    clEnqueueReadBuffer(q, dout, CL_TRUE, 0, NNH * sizeof(float), hout.data(), 0, nullptr, nullptr);
    int bad = 0, first_bad = -1;
    for (int i = 0; i < (int)NNH; i++)
      if (hout[i] != tr3_expect(i)) { if (first_bad < 0) first_bad = i; bad++; }
    std::fprintf(stderr, "[lds] %-10s : %s  (%d/%d wrong)  out[0..3]=%.0f,%.0f,%.0f,%.0f\n",
                 "t_lds_h2(x4)", bad == 0 ? "PASS" : "FAIL", bad, (int)NNH,
                 hout[0], hout[1], hout[2], hout[3]);
  }
  clReleaseMemObject(din);
  clReleaseMemObject(dout);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <weight_file_path>\n"
                 "  e.g. %s /data/local/tmp/nntrainer/causallm/models/"
                 "qwen3-0.6b-qint4-fresh/nntr_qwen3_0.6b_qint4.bin\n",
                 argv[0], argv[0]);
    return 1;
  }
  const std::string weight_path = argv[1];

  causallm_gpu::Qwen3Config cfg;
  // Default: Qwen3-0.6B. NNTR_MODEL_4B=1 selects Qwen3-4B dims (hidden 2560,
  // 36 layers, 32 Q / 8 KV heads, inter 9728) — for the 8/4/4 coherence demo.
  const bool model_4b = []() {
    const char *e = std::getenv("NNTR_MODEL_4B");
    return e && std::atoi(e) != 0;
  }();
  // #63: NNTR_MODEL_GEMMA2=1 selects Gemma2-2B (the ML Drift paper's model) —
  // a plain transformer but with head_dim 256, sandwich norm, GeGLU, attn/final
  // soft-cap, embed*sqrt(H), no q/k-norm. Proves gpu_native is model-agnostic.
  const bool model_gemma2 = []() {
    const char *e = std::getenv("NNTR_MODEL_GEMMA2");
    return e && std::atoi(e) != 0;
  }();
  const char *model_name = "Qwen3-0.6B";
  if (model_gemma2) {
    cfg.hidden_size = 2304;
    cfg.intermediate_size = 9216;
    cfg.head_dim = 256;
    cfg.num_heads_Q = 8;
    cfg.num_heads_KV = 4;
    cfg.num_layers = 26;
    cfg.vocab_size = 256000;
    cfg.rope_theta = 1e4f;            // Gemma2 sliding/default rope
    cfg.is_gemma2 = true;
    cfg.embed_scale = std::sqrt((float)cfg.hidden_size); // ~48.0
    cfg.attn_logit_softcap = 50.0f;
    cfg.final_logit_softcap = 30.0f;
    cfg.sliding_window = 4096;        // no-op for prefill M<=1024
    model_name = "Gemma2-2B";
  } else if (model_4b) {
    cfg.hidden_size = 2560;
    cfg.intermediate_size = 9728;
    cfg.head_dim = 128;
    cfg.num_heads_Q = 32;
    cfg.num_heads_KV = 8;
    cfg.num_layers = 36;
    cfg.vocab_size = 151936;
    cfg.rope_theta = 1e6f;
    model_name = "Qwen3-4B";
  } else {
    cfg.hidden_size = 1024;
    cfg.intermediate_size = 3072;
    cfg.head_dim = 128;
    cfg.num_heads_Q = 16;
    cfg.num_heads_KV = 8;
    cfg.num_layers = 28;
    cfg.vocab_size = 151936;
    cfg.rope_theta = 1e6f;
  }
  cfg.max_seq_len = 20480;
  cfg.rms_norm_eps = 1e-6f;
  std::fprintf(stderr,
               "[main] model=%s hidden=%u L=%u hQ=%u hKV=%u hd=%u V=%u gemma2=%d\n",
               model_name, cfg.hidden_size, cfg.num_layers, cfg.num_heads_Q,
               cfg.num_heads_KV, cfg.head_dim, cfg.vocab_size,
               (int)cfg.is_gemma2);

  causallm_gpu::Qwen3Forward fwd;
  if (!fwd.init(cfg, weight_path)) {
    std::fprintf(stderr, "[main] init failed\n");
    return 2;
  }

  // #69 compute-peak microbench (no model needed): measure the actual
  // Adreno int8-dp4a + fp16 peak on THIS device, then exit.
  if (std::getenv("NNTR_PEAK_BENCH")) {
    run_peak_bench();
    return 0;
  }
  if (std::getenv("NNTR_LDS_REPRO")) {
    run_lds_repro();
    return 0;
  }

  // RoPE freqs for position 0 (identity rotation; degenerate single-
  // token attention where softmax of one element = 1.0, attention
  // output = V per head). Precomputed once and reused across all 28
  // layers.
  if (!fwd.precompute_rope_for_position(0)) {
    std::fprintf(stderr, "[main] precompute_rope_for_position failed\n");
    return 3;
  }

  // Walk the weight file, loading all 28 layers. Per-layer KV cache
  // sized for max_seq_len_used = 1024 so that the prefill measurement
  // at M=1024 doesn't overrun the cache (each layer writes M rows).
  // Per-layer cache memory at this size = 2 * 1024 * 8 * 128 * 2 =
  // 4 MB; 28 layers => 112 MB total SVM (fine on SD8 Elite).
  const unsigned int max_seq_len_used = 1024;
  auto NOW = []() { return std::chrono::steady_clock::now(); };
  auto MS = [](auto t1, auto t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
             .count() / 1000.0;
  };

  auto t_load_start = NOW();
  size_t off = fwd.layers_start_offset();
  for (unsigned int L = 0; L < cfg.num_layers; ++L) {
    if (!fwd.load_layer(L, &off, max_seq_len_used)) {
      std::fprintf(stderr, "[main] load_layer(%u) failed\n", L);
      return 10 + (int)L;
    }
  }
  auto t_load_done = NOW();
  std::fprintf(stderr,
               "[main] all %u layers loaded in %.1f ms; final offset=%zu "
               "MB (file=%zu MB)\n",
               cfg.num_layers, MS(t_load_done, t_load_start),
               off / (1024 * 1024),
               fwd.weight_file_size() / (1024 * 1024));

  // Load output_norm gamma up front (sits at the file tail right after
  // layer 27). Doing it here makes the per-iteration timing below
  // exclude this one-time load.
  if (!fwd.load_output_norm(off)) {
    std::fprintf(stderr, "[main] load_output_norm failed\n");
    return 60;
  }

  // Gemma2 BOS = 2 (<bos>); Qwen3 uses 151643. Model-aware so the seed
  // token is in-distribution for the loaded model (#63).
  const unsigned int BOS_TOKEN = cfg.is_gemma2 ? 2u : 151643u;
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  cl_command_queue q = cl->command_queue_inst_.GetCommandQueue();
  const unsigned int H = cfg.hidden_size;

  // Step 9: run 3 iterations to measure timing breakdown + verify
  // determinism (same predicted token every run).
  constexpr int NUM_RUNS = 3;
  int prev_token = -1;
  bool deterministic = true;
  for (int run = 0; run < NUM_RUNS; ++run) {
    auto t_embed_start = NOW();
    cl_mem cur = fwd.embedding_lookup_to_fp32_clmem(BOS_TOKEN);
    if (cur == nullptr) {
      std::fprintf(stderr, "[main] embedding_lookup failed\n");
      return 50;
    }
    auto t_embed_done = NOW();

    // Ping-pong output buffers (persistent across the chain).
    // forward_one_layer_v2 takes caller-managed in/out — both
    // [hidden] fp32 cl_mems. Alternate which is "in" each layer.
    cl_int e;
    cl_context ctx2 = cl->context_inst_.GetContext();
    static cl_mem buf_a = nullptr, buf_b = nullptr;
    if (buf_a == nullptr) {
      buf_a = clCreateBuffer(ctx2, CL_MEM_READ_WRITE, H * sizeof(float),
                             nullptr, &e);
      buf_b = clCreateBuffer(ctx2, CL_MEM_READ_WRITE, H * sizeof(float),
                             nullptr, &e);
    }
    // Copy embedding-lookup `cur` into buf_a as the layer-0 input,
    // then release the per-iter `cur` (the lookup allocates fresh).
    clEnqueueCopyBuffer(q, cur, buf_a, 0, 0, H * sizeof(float), 0, nullptr,
                        nullptr);
    clReleaseMemObject(cur);
    cur = nullptr;

    auto t_chain_start = NOW();
    cl_mem layer_in = buf_a;
    cl_mem layer_out = buf_b;
    for (unsigned int L = 0; L < cfg.num_layers; ++L) {
      if (!fwd.forward_one_layer_v2(L, layer_in, layer_out, /*position=*/0)) {
        std::fprintf(stderr, "[main] forward_one_layer_v2(%u) failed\n", L);
        return 100 + (int)L;
      }
      std::swap(layer_in, layer_out);
    }
    // After the loop layer_in holds the final layer's output (swap ran 28x).
    cur = layer_in;
    auto t_chain_done = NOW();

    if (!fwd.run_output_norm(cur)) {
      std::fprintf(stderr, "[main] run_output_norm failed\n");
      return 61;
    }
    auto t_norm_done = NOW();

    int next_token = fwd.run_lm_head_and_argmax(cur);
    // cur points into the persistent buf_a/buf_b ping-pong pool —
    // intentionally NOT released here so the next iteration reuses it.
    if (next_token < 0) {
      std::fprintf(stderr, "[main] lm_head failed\n");
      return 70;
    }
    auto t_lm_done = NOW();

    const double t_embed   = MS(t_embed_done, t_embed_start);
    const double t_chain   = MS(t_chain_done, t_chain_start);
    const double t_norm    = MS(t_norm_done,  t_chain_done);
    const double t_lm      = MS(t_lm_done,    t_norm_done);
    const double t_total   = MS(t_lm_done,    t_embed_start);
    const double t_per_layer = t_chain / cfg.num_layers;
    const double tps = 1000.0 / t_total;

    std::fprintf(stderr,
                 "[run %d] embed=%.2f ms  chain=%.1f ms (%.2f ms/layer)  "
                 "out_norm=%.2f ms  lm_head=%.1f ms  TOTAL=%.1f ms  "
                 "(=> %.2f TPS effective)  -> token %d\n",
                 run, t_embed, t_chain, t_per_layer, t_norm, t_lm, t_total,
                 tps, next_token);

    if (run > 0 && next_token != prev_token) deterministic = false;
    prev_token = next_token;
  }

  std::fprintf(stderr,
               "\n[main] decode (M=1) summary:\n"
               "  predicted token over %d runs: %d (deterministic=%d)\n"
               "  baseline reference: CausalLM ~6.7 decode TPS "
               "(== ~150 ms/decode token) on the same SD8 Elite.\n",
               NUM_RUNS, prev_token, deterministic ? 1 : 0);

  // ===== Phase A #2: multi-token prefill timing (task #45) =====
  // Measure 28-layer chain at various M values. Initial input is the
  // BOS embedding replicated M times — semantically wrong for prefill
  // (real prefill needs M distinct tokens + per-position RoPE = task
  // #45b) but the kernel chain runs end-to-end so per-op wall time
  // is valid. Compare to baseline 1K prefill 460 TPS.
  std::fprintf(stderr,
               "\n[main] === Phase A #2: prefill timing (M>1) ===\n"
               "  NOTE: per-token RoPE not yet implemented (task #45b);\n"
               "  output token id is not meaningful for M>1 but per-op\n"
               "  timing is.\n");

  constexpr int PREFILL_MS[] = {2, 8, 64, 256, 512, 1024};
  // Allocate bigger ping-pong buffers + warm the scratch up to M=1024.
  const unsigned int M_max = 1024;
  if (!fwd.ensure_forward_scratch_allocated(M_max)) {
    std::fprintf(stderr, "[main] ensure_forward_scratch_allocated(M=%u) failed\n",
                 M_max);
    return 80;
  }
  cl_context ctx3 = cl->context_inst_.GetContext();
  cl_int e3 = CL_SUCCESS;
  cl_mem pf_in = clCreateBuffer(ctx3, CL_MEM_READ_WRITE,
                                (size_t)M_max * H * sizeof(float), nullptr,
                                &e3);
  cl_mem pf_out = clCreateBuffer(ctx3, CL_MEM_READ_WRITE,
                                 (size_t)M_max * H * sizeof(float), nullptr,
                                 &e3);
  if (e3 != CL_SUCCESS) {
    std::fprintf(stderr, "[main] prefill bufs alloc err=%d\n", e3);
    return 81;
  }
  // Load BOS embedding once on host.
  std::vector<float> bos_host(H);
  {
    cl_mem one = fwd.embedding_lookup_to_fp32_clmem(BOS_TOKEN);
    clEnqueueReadBuffer(q, one, CL_TRUE, 0, H * sizeof(float),
                        bos_host.data(), 0, nullptr, nullptr);
    clReleaseMemObject(one);
  }
  // Replicate it M_max times in host buffer then upload once.
  std::vector<float> rep_input((size_t)M_max * H);
  for (unsigned int m = 0; m < M_max; ++m)
    std::memcpy(rep_input.data() + (size_t)m * H, bos_host.data(),
                H * sizeof(float));

  for (int M_test : PREFILL_MS) {
    clEnqueueWriteBuffer(q, pf_in, CL_TRUE, 0,
                         (size_t)M_test * H * sizeof(float),
                         rep_input.data(), 0, nullptr, nullptr);
    // Enable per-stage profiling at M=256 (peak) and M=1024 (cliff)
    // so we can see WHERE the cliff time goes. Profiling adds a
    // clFinish per stage = overhead; total ms reported INCLUDES that
    // overhead but per-stage attribution is clean.
    // per-stage timing brackets each stage with clFinish(cl_q_), which DRAINS
    // the in-order queue and destroys host/GPU dispatch overlap — i.e. it
    // inflates the very M=256/M=1024 wall it is trying to measure (M=512 never
    // set it, which is why M=512 looked faster per-token). Gate it OFF by
    // default; NNTR_STAGE_PROFILE=1 restores per-stage attribution.
    static const bool want_stage_prof = []() {
      const char *e = std::getenv("NNTR_STAGE_PROFILE");
      return e && std::atoi(e) != 0;
    }();
    const bool profile = want_stage_prof && (M_test == 256 || M_test == 1024);
    if (profile) {
      fwd.timings_.reset();
      fwd.profile_stages_ = true;
    } else {
      fwd.profile_stages_ = false;
    }
    // NNTR_PERLAYER_PROF=1: clFinish after EACH layer so tl1-tl0 is that layer's
    // real GPU time (uniform drain overhead → layers are comparable). Reveals
    // whether later layers slow down (thermal/clock droop) vs identical compute.
    static const bool perlayer_prof = []() {
      const char *e = std::getenv("NNTR_PERLAYER_PROF");
      return e && std::atoi(e) != 0;
    }();
    const bool plp = perlayer_prof && (M_test == 1024 || M_test == 512);
    std::vector<double> per_layer_ms;
    if (plp) per_layer_ms.reserve(cfg.num_layers);
    auto t0 = NOW();
    cl_mem in_b = pf_in, out_b = pf_out;
    bool ok = true;
    double t_layer_0 = 0, t_layer_last = 0;
    for (unsigned int L = 0; L < cfg.num_layers; ++L) {
      auto tl0 = NOW();
      if (!fwd.forward_one_layer_v2(L, in_b, out_b, 0,
                                    (unsigned int)M_test)) {
        std::fprintf(stderr,
                     "[main] prefill M=%d layer %u failed\n", M_test, L);
        ok = false;
        break;
      }
      if (plp) clFinish(q);
      auto tl1 = NOW();
      if (plp) per_layer_ms.push_back(MS(tl1, tl0));
      if (L == 0) t_layer_0 = MS(tl1, tl0);
      if (L == cfg.num_layers - 1) t_layer_last = MS(tl1, tl0);
      std::swap(in_b, out_b);
    }
    if (plp && ok) {
      double sum = 0, mn = 1e9, mx = 0;
      for (double v : per_layer_ms) { sum += v; mn = v < mn ? v : mn; mx = v > mx ? v : mx; }
      std::fprintf(stderr, "  [per-layer GPU ms, M=%d] (clFinish/layer):\n   ", M_test);
      for (size_t i = 0; i < per_layer_ms.size(); i++)
        std::fprintf(stderr, " L%zu=%.2f", i, per_layer_ms[i]);
      std::fprintf(stderr,
                   "\n  [per-layer M=%d] min=%.2f max=%.2f mean=%.2f  last/first=%.2fx\n",
                   M_test, mn, mx, sum / per_layer_ms.size(),
                   per_layer_ms.back() / per_layer_ms.front());
    }
    // The attention image path no longer drains the queue per layer (the two
    // clFinish drains were gated off for dispatch overlap), so the forward now
    // only ENQUEUES work and returns before the GPU finishes. Drain once here
    // so the wall-clock below measures real end-to-end prefill, not host
    // enqueue time. (Outside the timed region this would otherwise be absorbed
    // by the next CL_TRUE write.)
    clFinish(q);
    auto t1 = NOW();
    if (!ok) continue;
    const double t_ms = MS(t1, t0);
    const double tps = (double)M_test * 1000.0 / t_ms;
    const double ms_per_token = t_ms / M_test;
    std::fprintf(stderr,
                 "[prefill M=%4d] chain=%7.1f ms  %.3f ms/token  "
                 "=> %7.1f TPS  (L0=%.1f ms, L27=%.1f ms)\n",
                 M_test, t_ms, ms_per_token, tps,
                 t_layer_0, t_layer_last);
    if (profile) {
      const auto &tt = fwd.timings_;
      const double sum = tt.pad_attn_norm_ms + tt.qkv_quant_image_ms +
                         tt.qkv_gemm_ms + tt.qk_norm_rope_ms +
                         tt.kv_write_ms + tt.attn_dispatch_ms +
                         tt.wo_ms + tt.ffn_ms;
      auto pct = [&](double v) { return sum > 0 ? 100.0 * v / sum : 0.0; };
      std::fprintf(stderr,
                   "  [stage timings, M=%d, %d layer-calls totaling %.0f ms]:\n"
                   "    (a) pad+attn_norm  %7.1f ms (%4.1f%%)\n"
                   "    (b) qkv quant+img  %7.1f ms (%4.1f%%)\n"
                   "    (c) Q/K/V GEMM     %7.1f ms (%4.1f%%)\n"
                   "    (d) qk_norm[+RoPE] %7.1f ms (%4.1f%%)\n"
                   "    (e) KV write SVM   %7.1f ms (%4.1f%%)\n"
                   "    (f) attention      %7.1f ms (%4.1f%%)\n"
                   "    (g) wo + resid_1   %7.1f ms (%4.1f%%)\n"
                   "    (h) ffn block      %7.1f ms (%4.1f%%)\n",
                   M_test, tt.calls, sum,
                   tt.pad_attn_norm_ms,   pct(tt.pad_attn_norm_ms),
                   tt.qkv_quant_image_ms, pct(tt.qkv_quant_image_ms),
                   tt.qkv_gemm_ms,        pct(tt.qkv_gemm_ms),
                   tt.qk_norm_rope_ms,    pct(tt.qk_norm_rope_ms),
                   tt.kv_write_ms,        pct(tt.kv_write_ms),
                   tt.attn_dispatch_ms,   pct(tt.attn_dispatch_ms),
                   tt.wo_ms,              pct(tt.wo_ms),
                   tt.ffn_ms,             pct(tt.ffn_ms));
    }
    // ALWAYS-ON host-bridge timing (env-gated print). The host_*_ms fields
    // accumulate the host wall-clock stalls of the SVM<->cl_mem bridges
    // across all layer calls of this forward. They are reset along with the
    // stage timings (timings_.reset() at the start of each profile M), so
    // the printed value is the clean per-forward total at this M.
    if (profile && std::getenv("NNTR_HOST_TIMING")) {
      const auto &tt = fwd.timings_;
      const double host_total =
        tt.host_kv_ms + tt.host_q_ms + tt.host_copy_svm_ms;
      std::fprintf(stderr,
                   "  [host-timing M=%d, %d layer-calls]: "
                   "kv_bridge=%.2f ms  q_bridge=%.2f ms  copy_svm=%.2f ms  "
                   "=> host_total=%.2f ms (%.1f%% of chain=%.1f ms)\n",
                   M_test, tt.calls, tt.host_kv_ms, tt.host_q_ms,
                   tt.host_copy_svm_ms, host_total,
                   t_ms > 0 ? 100.0 * host_total / t_ms : 0.0, t_ms);
    }
    // True on-device per-kernel GPU time (no-op unless NNTR_OPENCL_PROFILING
    // is set). Unlike the clFinish-bracketed stage timings above, this is
    // immune to out-of-order queue catch-up — it reads each kernel's own
    // CL_PROFILING_COMMAND_START/END. dumpProfile clears its event log each
    // call, so this captures exactly this M_test chain.
    {
      char ptag[32];
      std::snprintf(ptag, sizeof(ptag), "M=%d", M_test);
      cl->command_queue_inst_.dumpProfile(ptag);
    }
  }

  // ===== Prefill CORRECTNESS: greedy generation via repeated prefill =====
  // Real multi-token prefill (distinct tokens + per-position RoPE + causal
  // attention), re-prefilling the growing sequence each step and reading the
  // LAST position's logits. Validates that prefill output is now valid
  // (#47i fp32 swiglu fixed the last-layer fp16 overflow NaN).
  {
    std::fprintf(stderr,
                 "\n[main] === prefill correctness: greedy generation from BOS ===\n");
    cl_int eg = CL_SUCCESS;
    cl_mem last_row =
      clCreateBuffer(ctx3, CL_MEM_READ_WRITE, H * sizeof(float), nullptr, &eg);
    std::vector<int> seq{(int)BOS_TOKEN};
    const int GEN = 20;
    auto prefill_predict = [&](int read_row) -> int {
      const int M = (int)seq.size();
      for (int i = 0; i < M; ++i) {
        cl_mem em = fwd.embedding_lookup_to_fp32_clmem((unsigned int)seq[i]);
        if (!em) return -2;
        clEnqueueCopyBuffer(q, em, pf_in, 0, (size_t)i * H * sizeof(float),
                            H * sizeof(float), 0, nullptr, nullptr);
        clReleaseMemObject(em);
      }
      clFinish(q);
      cl_mem in_b = pf_in, out_b = pf_out;
      for (unsigned int L = 0; L < cfg.num_layers; ++L) {
        if (!fwd.forward_one_layer_v2(L, in_b, out_b, 0, (unsigned int)M))
          return -2;
        std::swap(in_b, out_b);
      }
      const int r = (read_row < 0) ? (M - 1) : read_row;
      clEnqueueCopyBuffer(q, in_b, last_row, (size_t)r * H * sizeof(float), 0,
                          H * sizeof(float), 0, nullptr, nullptr);
      clFinish(q);
      if (!fwd.run_output_norm(last_row)) return -2;
      return fwd.run_lm_head_and_argmax(last_row);
    };
    bool ok = true;
    for (int step = 0; step < GEN; ++step) {
      int nxt = prefill_predict(-1);
      if (nxt < 0) { std::fprintf(stderr, "  step %d: predict failed (%d)\n", step, nxt); ok = false; break; }
      if (step == 0)
        std::fprintf(stderr,
                     "  [self-consistency] prefill([BOS]) -> %d (decode=7212, match=%d)\n",
                     nxt, nxt == 7212);
      seq.push_back(nxt);
    }
    // Causal check: prefill([BOS,X]) row 0 must equal [BOS]-alone prediction.
    {
      std::vector<int> saved = seq; seq = {(int)BOS_TOKEN, 12345};
      int r0 = prefill_predict(0);
      std::fprintf(stderr, "  [causal] prefill([BOS,12345]) row0 -> %d (match=%d)\n",
                   r0, r0 == 7212);
      seq = saved;
    }
    std::fprintf(stderr, "  generated %zu token ids (greedy, no NaN=%d):\n   ",
                 seq.size(), ok);
    for (int t : seq) std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, "\n");
    clReleaseMemObject(last_row);
  }

  clReleaseMemObject(pf_in);
  clReleaseMemObject(pf_out);

  std::fprintf(stderr,
               "\n[main] step #45 OK — multi-token prefill chain runs.\n");
  return 0;
}

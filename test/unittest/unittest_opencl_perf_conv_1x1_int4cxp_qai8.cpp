// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_opencl_perf_conv_1x1_int4cxp_qai8.cpp
 * @date    15 May 2026
 * @brief   Pure-GPU timing for the B4-int4-int8 kernel. Uses
 *          cl_event + clGetEventProfilingInfo on a CL_QUEUE_PROFILING_ENABLE
 *          queue, so the reported numbers exclude host packing, allocation,
 *          and any CPU-side reference work. Reports median / min / p95
 *          across N timed iterations after a small warmup, plus arithmetic
 *          throughput in GOps/s.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cl_context.h>
#include <engine.h>
#include <opencl_command_queue_manager.h>
#include <opencl_context_manager.h>
#include <opencl_loader.h>
#include <phwc4_int8_quant_layout.h>
#include <phwc4_layout.h>
#include <weight_pack_int4cxp_layout.h>

namespace nntrainer {
extern const std::string conv_1x1_int4cxp_qai8_kernel;
}

namespace {

struct GpuFixture {
  GpuFixture() {
    auto *cc = static_cast<nntrainer::ClContext *>(
      nntrainer::Engine::Global().getRegisteredContext("gpu"));
    (void)cc;
  }
};
static GpuFixture s_fixture;

struct PerfStats {
  double median_us = 0.0;
  double min_us = 0.0;
  double p95_us = 0.0;
};

static PerfStats summarize(std::vector<double> &samples) {
  std::sort(samples.begin(), samples.end());
  PerfStats r;
  if (samples.empty())
    return r;
  r.min_us = samples.front();
  r.median_us = samples[samples.size() / 2];
  r.p95_us =
    samples[std::min(samples.size() - 1, (samples.size() * 95) / 100)];
  return r;
}

// One MAC = 1 multiply + 1 add  = 2 ops in the conventional accounting
// (matches MLPerf / cuBLAS). For an LLM matmul (B=1) the workload is
// 2 * C_in * C_out * H * W * B ops.
static double ops_for_shape(int B, int C_in, int C_out, int H, int W) {
  return 2.0 * static_cast<double>(B) * C_in * C_out * H * W;
}

// Build kernel, allocate device buffers, fill with random data once, then
// run warmup + timed iterations on a profile-enabled queue. Returns PerfStats
// over the kernel-only times (no host packing, no buffer fill, no readback).
static PerfStats bench_conv_1x1_int4_qai8(int B, int C_in, int C_out, int H,
                                          int W, int warmup, int iters) {
  auto *cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  (void)cc;
  const cl_context ctx =
    nntrainer::opencl::ContextManager::Global().GetContext();
  const cl_device_id dev =
    nntrainer::opencl::ContextManager::Global().GetDeviceId();
  const cl_queue_properties qprops[] = {CL_QUEUE_PROPERTIES,
                                        CL_QUEUE_PROFILING_ENABLE, 0};
  cl_int q_err = CL_SUCCESS;
  cl_command_queue queue =
    clCreateCommandQueueWithProperties(ctx, dev, qprops, &q_err);
  EXPECT_EQ(q_err, CL_SUCCESS) << "queue: " << q_err;

  const char *src = nntrainer::conv_1x1_int4cxp_qai8_kernel.c_str();
  const size_t src_len = nntrainer::conv_1x1_int4cxp_qai8_kernel.size();
  cl_int err = CL_SUCCESS;
  cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
  EXPECT_EQ(err, CL_SUCCESS) << "createProgramWithSource: " << err;
  err = clBuildProgram(prog, 1, &dev, "-cl-std=CL3.0", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &log_size);
    std::vector<char> log(log_size + 1, '\0');
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(),
                          nullptr);
    ADD_FAILURE() << "buildProgram(" << err << "):\n" << log.data();
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    return {};
  }
  cl_kernel kern = clCreateKernel(prog, "conv_1x1_int4cxp_qai8", &err);
  EXPECT_EQ(err, CL_SUCCESS) << "createKernel: " << err;

  // Fill all buffers with deterministic noise. Bench is not checking
  // correctness — that's the job of unittest_opencl_conv_1x1_int4cxp_qai8.
  std::mt19937 rng(0xb44u + B * 13 + C_in * 17 + C_out * 19 + H * 29 + W * 37);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<std::int8_t> x_int8(
    nntrainer::phwc4_int8::num_bytes(B, C_in, H, W));
  std::vector<float> sx(nntrainer::phwc4_int8::num_scales(B, H, W));
  for (auto &v : x_int8)
    v = static_cast<std::int8_t>((rng() & 0xFF) - 128);
  for (auto &v : sx)
    v = std::fabs(dist(rng)) + 1e-3f;

  std::vector<std::uint8_t> wq(
    nntrainer::weight_pack_int4cxp::num_bytes(C_out, C_in, 1, 1));
  std::vector<float> sw(
    nntrainer::weight_pack_int4cxp::num_scales(C_out));
  for (auto &v : wq)
    v = static_cast<std::uint8_t>(rng() & 0xFF);
  for (auto &v : sw)
    v = std::fabs(dist(rng)) + 1e-3f;

  const std::size_t y_elems =
    nntrainer::phwc4::num_elements(B, C_out, H, W);

  cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              x_int8.size(), x_int8.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS);
  cl_mem d_sx = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sx.size() * sizeof(float), sx.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS);
  cl_mem d_wq = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               wq.size(), wq.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS);
  cl_mem d_sw = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sw.size() * sizeof(float), sw.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS);
  cl_mem d_y =
    clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, y_elems * sizeof(float), nullptr,
                   &err);
  EXPECT_EQ(err, CL_SUCCESS);

  int iB = B, iCin = C_in, iCout = C_out, iH = H, iW = W;
  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &d_x);
  err |= clSetKernelArg(kern, 1, sizeof(cl_mem), &d_sx);
  err |= clSetKernelArg(kern, 2, sizeof(cl_mem), &d_wq);
  err |= clSetKernelArg(kern, 3, sizeof(cl_mem), &d_sw);
  err |= clSetKernelArg(kern, 4, sizeof(cl_mem), &d_y);
  err |= clSetKernelArg(kern, 5, sizeof(int), &iB);
  err |= clSetKernelArg(kern, 6, sizeof(int), &iCin);
  err |= clSetKernelArg(kern, 7, sizeof(int), &iCout);
  err |= clSetKernelArg(kern, 8, sizeof(int), &iH);
  err |= clSetKernelArg(kern, 9, sizeof(int), &iW);
  EXPECT_EQ(err, CL_SUCCESS);

  const std::size_t slice_o = (C_out + 3) / 4;
  const std::size_t total = slice_o * static_cast<std::size_t>(B) * H * W;
  const std::size_t local = 64;
  const std::size_t global = ((total + local - 1) / local) * local;

  // Warmup — drives the JIT and primes any per-queue setup costs. Adreno in
  // particular wants the first few launches to settle before timing.
  for (int i = 0; i < warmup; ++i) {
    err = clEnqueueNDRangeKernel(queue, kern, 1, nullptr, &global, &local, 0,
                                 nullptr, nullptr);
    EXPECT_EQ(err, CL_SUCCESS);
  }
  clFinish(queue);

  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    cl_event e = nullptr;
    err = clEnqueueNDRangeKernel(queue, kern, 1, nullptr, &global, &local, 0,
                                 nullptr, &e);
    EXPECT_EQ(err, CL_SUCCESS);
    clWaitForEvents(1, &e);
    cl_ulong tstart = 0, tend = 0;
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                            &tstart, nullptr);
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                            &tend, nullptr);
    clReleaseEvent(e);
    samples.push_back(static_cast<double>(tend - tstart) / 1000.0);
  }

  clReleaseMemObject(d_x);
  clReleaseMemObject(d_sx);
  clReleaseMemObject(d_wq);
  clReleaseMemObject(d_sw);
  clReleaseMemObject(d_y);
  clReleaseKernel(kern);
  clReleaseProgram(prog);
  clReleaseCommandQueue(queue);

  return summarize(samples);
}

} // namespace

static void report(const char *name, int B, int C_in, int C_out, int H, int W,
                   const PerfStats &s) {
  const double ops = ops_for_shape(B, C_in, C_out, H, W);
  const double median_s = s.median_us * 1e-6;
  const double throughput_gops = (median_s > 0.0) ? (ops / median_s / 1e9) : 0.0;
  // One line per shape — easy to grep.
  std::printf("[BENCH] kernel=%-22s shape=B=%d,Ci=%d,Co=%d,H=%d,W=%d "
              "median_us=%.2f min_us=%.2f p95_us=%.2f GOps=%.3f "
              "throughput_GOps/s=%.2f\n",
              name, B, C_in, C_out, H, W, s.median_us, s.min_us, s.p95_us,
              ops / 1e9, throughput_gops);
  std::fflush(stdout);
}

TEST(perf_conv_1x1_int4_qai8, small_baseline) {
  // (1, 1024, 1024, 1, 64) — baseline at small but realistic scale.
  auto s = bench_conv_1x1_int4_qai8(1, 1024, 1024, 1, 64, 3, 10);
  report("conv_1x1_int4cxp_qai8", 1, 1024, 1024, 1, 64, s);
  EXPECT_GT(s.median_us, 0.0);
}

TEST(perf_conv_1x1_int4_qai8, decode_one_token_hidden_4k) {
  // (1, 4096, 4096, 1, 1) — single-token decode, memory-bound regime.
  auto s = bench_conv_1x1_int4_qai8(1, 4096, 4096, 1, 1, 3, 30);
  report("conv_1x1_int4cxp_qai8", 1, 4096, 4096, 1, 1, s);
  EXPECT_GT(s.median_us, 0.0);
}

TEST(perf_conv_1x1_int4_qai8, prefill_seq128_hidden_4k) {
  // (1, 4096, 4096, 1, 128) — prefill / hidden projection.
  auto s = bench_conv_1x1_int4_qai8(1, 4096, 4096, 1, 128, 3, 10);
  report("conv_1x1_int4cxp_qai8", 1, 4096, 4096, 1, 128, s);
  EXPECT_GT(s.median_us, 0.0);
}

TEST(perf_conv_1x1_int4_qai8, llama_ffn_up_proj_seq128) {
  // LLaMA-7B-ish FFN up-projection: hidden 4096 -> intermediate 11008.
  auto s = bench_conv_1x1_int4_qai8(1, 4096, 11008, 1, 128, 3, 5);
  report("conv_1x1_int4cxp_qai8", 1, 4096, 11008, 1, 128, s);
  EXPECT_GT(s.median_us, 0.0);
}

TEST(perf_conv_1x1_int4_qai8, llama_ffn_down_proj_seq128) {
  // FFN down-projection: intermediate 11008 -> hidden 4096.
  auto s = bench_conv_1x1_int4_qai8(1, 11008, 4096, 1, 128, 3, 5);
  report("conv_1x1_int4cxp_qai8", 1, 11008, 4096, 1, 128, s);
  EXPECT_GT(s.median_us, 0.0);
}

GTEST_API_ int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
  return result;
}

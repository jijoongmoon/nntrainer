// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_opencl_perf_rmsnorm_residual_fused.cpp
 * @date    15 May 2026
 * @brief   Pure-GPU timing for the §3.6 RMSNorm + residual-add fused
 *          kernel. Replaces the unfused two-kernel sequence (element-
 *          wise add, then RMSNorm). For decode the unfused cost is
 *          dominated by 2x ~50 us kernel-launch dispatch overhead;
 *          fusion should approximately halve that — and the longer
 *          prefill cases will show what the per-token compute side
 *          alone costs.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
#include <phwc4_layout.h>

namespace nntrainer {
extern const std::string rmsnorm_residual_fused_fp16_kernel;
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

static PerfStats summarize(std::vector<double> &s) {
  std::sort(s.begin(), s.end());
  PerfStats r;
  if (s.empty())
    return r;
  r.min_us = s.front();
  r.median_us = s[s.size() / 2];
  r.p95_us = s[std::min(s.size() - 1, (s.size() * 95) / 100)];
  return r;
}

// Convert a fp32 value to its fp16 binary representation (IEEE 754
// half-precision). Stored in a uint16_t for portable transit through
// OpenCL host APIs that don't know about the C++ half type.
static std::uint16_t f32_to_f16_bits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const std::uint32_t sign = (u >> 16) & 0x8000u;
  const int exp = ((u >> 23) & 0xff) - 127 + 15;
  const std::uint32_t mant = u & 0x7fffffu;
  if (exp <= 0)
    return static_cast<std::uint16_t>(sign);
  if (exp >= 31)
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  return static_cast<std::uint16_t>(sign | (exp << 10) | (mant >> 13));
}

static PerfStats bench_rmsnorm_fused(int B, int C, int H, int W, int warmup,
                                     int iters) {
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
  EXPECT_EQ(q_err, CL_SUCCESS);

  const char *src = nntrainer::rmsnorm_residual_fused_fp16_kernel.c_str();
  const size_t src_len = nntrainer::rmsnorm_residual_fused_fp16_kernel.size();
  cl_int err = CL_SUCCESS;
  cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
  EXPECT_EQ(err, CL_SUCCESS);
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
  cl_kernel kern =
    clCreateKernel(prog, "rmsnorm_residual_fused_fp16", &err);
  EXPECT_EQ(err, CL_SUCCESS);

  // Random fp32 -> fp16 bits for X, R, gamma.
  std::mt19937 rng(0x9fu + B * 7 + C * 13 + H * 19 + W * 23);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  const std::size_t elems = nntrainer::phwc4::num_elements(B, C, H, W);
  std::vector<std::uint16_t> x16(elems), r16(elems);
  for (auto &v : x16)
    v = f32_to_f16_bits(dist(rng));
  for (auto &v : r16)
    v = f32_to_f16_bits(dist(rng));
  std::vector<std::uint16_t> gamma16(C);
  for (auto &v : gamma16)
    v = f32_to_f16_bits(std::fabs(dist(rng)) + 1e-3f);

  cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              x16.size() * sizeof(std::uint16_t), x16.data(),
                              &err);
  EXPECT_EQ(err, CL_SUCCESS);
  cl_mem d_r = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              r16.size() * sizeof(std::uint16_t), r16.data(),
                              &err);
  EXPECT_EQ(err, CL_SUCCESS);
  cl_mem d_g =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   gamma16.size() * sizeof(std::uint16_t), gamma16.data(),
                   &err);
  EXPECT_EQ(err, CL_SUCCESS);
  cl_mem d_y =
    clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                   elems * sizeof(std::uint16_t), nullptr, &err);
  EXPECT_EQ(err, CL_SUCCESS);

  const float eps = 1e-5f;
  int iB = B, iC = C, iH = H, iW = W;
  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &d_x);
  err |= clSetKernelArg(kern, 1, sizeof(cl_mem), &d_r);
  err |= clSetKernelArg(kern, 2, sizeof(cl_mem), &d_g);
  err |= clSetKernelArg(kern, 3, sizeof(cl_mem), &d_y);
  err |= clSetKernelArg(kern, 4, sizeof(int), &iB);
  err |= clSetKernelArg(kern, 5, sizeof(int), &iC);
  err |= clSetKernelArg(kern, 6, sizeof(int), &iH);
  err |= clSetKernelArg(kern, 7, sizeof(int), &iW);
  err |= clSetKernelArg(kern, 8, sizeof(float), &eps);
  EXPECT_EQ(err, CL_SUCCESS);

  const std::size_t total_tokens = static_cast<std::size_t>(B) * H * W;
  const std::size_t local = 64;
  const std::size_t global = total_tokens * local;

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
  clReleaseMemObject(d_r);
  clReleaseMemObject(d_g);
  clReleaseMemObject(d_y);
  clReleaseKernel(kern);
  clReleaseProgram(prog);
  clReleaseCommandQueue(queue);
  return summarize(samples);
}

} // namespace

static void report(int B, int C, int H, int W, const PerfStats &s) {
  std::printf("[BENCH] kernel=%-36s shape=B=%d,C=%d,H=%d,W=%d "
              "median_us=%.2f min_us=%.2f p95_us=%.2f\n",
              "rmsnorm_residual_fused_fp16", B, C, H, W, s.median_us,
              s.min_us, s.p95_us);
  std::fflush(stdout);
}

TEST(perf_rmsnorm_fused, decode_hidden_1024) {
  auto s = bench_rmsnorm_fused(1, 1024, 1, 1, 3, 50);
  report(1, 1024, 1, 1, s);
  EXPECT_GT(s.median_us, 0.0);
}

TEST(perf_rmsnorm_fused, decode_hidden_4096) {
  auto s = bench_rmsnorm_fused(1, 4096, 1, 1, 3, 50);
  report(1, 4096, 1, 1, s);
  EXPECT_GT(s.median_us, 0.0);
}

TEST(perf_rmsnorm_fused, decode_hidden_11008) {
  auto s = bench_rmsnorm_fused(1, 11008, 1, 1, 3, 50);
  report(1, 11008, 1, 1, s);
  EXPECT_GT(s.median_us, 0.0);
}

TEST(perf_rmsnorm_fused, prefill_hidden_4k_seq128) {
  auto s = bench_rmsnorm_fused(1, 4096, 1, 128, 3, 20);
  report(1, 4096, 1, 128, s);
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

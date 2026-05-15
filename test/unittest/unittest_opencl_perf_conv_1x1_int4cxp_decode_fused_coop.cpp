// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_opencl_perf_conv_1x1_int4cxp_decode_fused_coop.cpp
 * @date    15 May 2026
 * @brief   Pure-GPU timing for the §3.6 op-fused decode kernel. The
 *          fused kernel folds (per-token INT8 quant) + (int4 weight x
 *          int8 act matmul) + (fp16 output) into one program — so the
 *          comparison number to beat is "B5 (90 us) + B2 image2d decode
 *          (371 us)" ~= 460 us total kernel time on Adreno 830.
 *          This benchmark times the single fused kernel only.
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
#include <weight_pack_int4cxp_layout.h>

namespace nntrainer {
extern const std::string conv_1x1_int4cxp_decode_fused_coop_fp16o_kernel;
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

static double ops_for_shape(int C_in, int C_out) {
  // Decode shape is B=H=W=1, so total ops = 2 * C_in * C_out.
  return 2.0 * static_cast<double>(C_in) * C_out;
}

static PerfStats bench_decode_fused_coop(int C_in, int C_out, int warmup,
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
  EXPECT_EQ(q_err, CL_SUCCESS) << "queue: " << q_err;

  const char *src =
    nntrainer::conv_1x1_int4cxp_decode_fused_coop_fp16o_kernel.c_str();
  const size_t src_len =
    nntrainer::conv_1x1_int4cxp_decode_fused_coop_fp16o_kernel.size();
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
    clCreateKernel(prog, "conv_1x1_int4cxp_decode_fused_coop_fp16o", &err);
  EXPECT_EQ(err, CL_SUCCESS);

  // Decode-shape data: B=H=W=1.
  std::mt19937 rng(0xfdu + C_in * 13 + C_out * 41);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> x_fp32(nntrainer::phwc4::num_elements(1, C_in, 1, 1));
  for (auto &v : x_fp32)
    v = dist(rng);

  std::vector<std::uint8_t> wq(
    nntrainer::weight_pack_int4cxp::num_bytes(C_out, C_in, 1, 1));
  std::vector<float> sw(nntrainer::weight_pack_int4cxp::num_scales(C_out));
  for (auto &v : wq)
    v = static_cast<std::uint8_t>(rng() & 0xFF);
  for (auto &v : sw)
    v = std::fabs(dist(rng)) + 1e-3f;

  // Image2d wraps the int4 weight bytes (same packing as B2: RGBA8, 2 i4
  // groups per texel along X).
  const int slice_i_total = (C_in + 3) / 4;
  const int slice_o = (C_out + 3) / 4;
  const int slice_i_pairs = (slice_i_total + 1) / 2;
  cl_image_format fmt = {CL_RGBA, CL_UNSIGNED_INT8};
  cl_image_desc desc{};
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = static_cast<size_t>(slice_i_pairs);
  desc.image_height = static_cast<size_t>(slice_o * 4);
  cl_mem d_wq_img =
    clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
  EXPECT_EQ(err, CL_SUCCESS) << "createImage: " << err;
  std::vector<std::uint8_t> wq_rgba(
    static_cast<std::size_t>(slice_o * 4) * slice_i_pairs * 4, 0);
  for (int o = 0; o < slice_o * 4 && o < C_out; ++o) {
    const std::size_t src_row = static_cast<std::size_t>(o) * slice_i_total * 2;
    const std::size_t dst_row =
      static_cast<std::size_t>(o) * slice_i_pairs * 4;
    const std::size_t copy_bytes =
      std::min<std::size_t>(slice_i_total * 2, slice_i_pairs * 4);
    std::memcpy(wq_rgba.data() + dst_row, wq.data() + src_row, copy_bytes);
  }
  const size_t origin[3] = {0, 0, 0};
  const size_t region[3] = {static_cast<size_t>(slice_i_pairs),
                            static_cast<size_t>(slice_o * 4), 1};
  err = clEnqueueWriteImage(queue, d_wq_img, CL_TRUE, origin, region,
                            slice_i_pairs * 4, 0, wq_rgba.data(), 0, nullptr,
                            nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "writeImage: " << err;

  cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              x_fp32.size() * sizeof(float), x_fp32.data(),
                              &err);
  EXPECT_EQ(err, CL_SUCCESS);
  cl_mem d_sw = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sw.size() * sizeof(float), sw.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS);
  const std::size_t y_elems = nntrainer::phwc4::num_elements(1, C_out, 1, 1);
  cl_mem d_y =
    clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, y_elems * sizeof(std::uint16_t),
                   nullptr, &err);
  EXPECT_EQ(err, CL_SUCCESS);

  int iCin = C_in, iCout = C_out;
  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &d_x);
  err |= clSetKernelArg(kern, 1, sizeof(cl_mem), &d_wq_img);
  err |= clSetKernelArg(kern, 2, sizeof(cl_mem), &d_sw);
  err |= clSetKernelArg(kern, 3, sizeof(cl_mem), &d_y);
  err |= clSetKernelArg(kern, 4, sizeof(int), &iCin);
  err |= clSetKernelArg(kern, 5, sizeof(int), &iCout);
  EXPECT_EQ(err, CL_SUCCESS);

  const std::size_t total = static_cast<std::size_t>(slice_o);
  const std::size_t local = 64;
  const std::size_t global = ((total + local - 1) / local) * local;

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
  clReleaseMemObject(d_sw);
  clReleaseMemObject(d_wq_img);
  clReleaseMemObject(d_y);
  clReleaseKernel(kern);
  clReleaseProgram(prog);
  clReleaseCommandQueue(queue);
  return summarize(samples);
}

} // namespace

static void report(int C_in, int C_out, const PerfStats &s) {
  const double ops = ops_for_shape(C_in, C_out);
  const double median_s = s.median_us * 1e-6;
  const double throughput_gops =
    (median_s > 0.0) ? (ops / median_s / 1e9) : 0.0;
  std::printf("[BENCH] kernel=%-44s shape=B=1,Ci=%d,Co=%d,H=1,W=1 "
              "median_us=%.2f min_us=%.2f p95_us=%.2f GOps=%.3f "
              "throughput_GOps/s=%.2f\n",
              "conv_1x1_int4cxp_decode_fused_coop_fp16o", C_in, C_out,
              s.median_us, s.min_us, s.p95_us, ops / 1e9, throughput_gops);
  std::fflush(stdout);
}

TEST(perf_conv_1x1_decode_fused_coop, small_baseline) {
  auto s = bench_decode_fused_coop(1024, 1024, 3, 30);
  report(1024, 1024, s);
  EXPECT_GT(s.median_us, 0.0);
}

TEST(perf_conv_1x1_decode_fused_coop, decode_hidden_4k) {
  // Compare to B2 image2d decode: 371 us for the matmul alone, plus a
  // separate B5 quant kernel dispatch (~50-100 us). Fused goal: well
  // under ~400 us, and a single dispatch.
  auto s = bench_decode_fused_coop(4096, 4096, 3, 30);
  report(4096, 4096, s);
  EXPECT_GT(s.median_us, 0.0);
}

TEST(perf_conv_1x1_decode_fused_coop, decode_ffn_up_4k_11k) {
  // FFN up-projection at decode (one token, projecting to intermediate).
  auto s = bench_decode_fused_coop(4096, 11008, 3, 15);
  report(4096, 11008, s);
  EXPECT_GT(s.median_us, 0.0);
}

TEST(perf_conv_1x1_decode_fused_coop, decode_ffn_down_11k_4k) {
  auto s = bench_decode_fused_coop(11008, 4096, 3, 15);
  report(11008, 4096, s);
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

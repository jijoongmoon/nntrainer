// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_opencl_perf_conv_1x1_int4cxp_qai8_img2d_tile_w4.cpp
 * @date    15 May 2026
 * @brief   Pure-GPU timing for the B2 image2d-weight variant of the
 *          B4-int4-int8 1x1 conv kernel. Same shapes and harness as
 *          unittest_opencl_perf_conv_1x1_int4cxp_qai8.cpp; reported numbers
 *          can be diffed against that file's baseline to size the
 *          texture-cache win on Adreno.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#include <algorithm>
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
#include <phwc4_int8_quant_layout.h>
#include <phwc4_layout.h>
#include <weight_pack_int4cxp_layout.h>

namespace nntrainer {
extern const std::string conv_1x1_int4cxp_qai8_img2d_w_tile_w4_kernel;
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

static double ops_for_shape(int B, int C_in, int C_out, int H, int W) {
  return 2.0 * static_cast<double>(B) * C_in * C_out * H * W;
}

static PerfStats bench_tile_w4(int B, int C_in, int C_out, int H, int W,
                             int warmup, int iters) {
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
    nntrainer::conv_1x1_int4cxp_qai8_img2d_w_tile_w4_kernel.c_str();
  const size_t src_len =
    nntrainer::conv_1x1_int4cxp_qai8_img2d_w_tile_w4_kernel.size();
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
    clCreateKernel(prog, "conv_1x1_int4cxp_qai8_img2d_w_tile_w4", &err);
  EXPECT_EQ(err, CL_SUCCESS);

  // Random fill — perf test, not correctness.
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
  std::vector<float> sw(nntrainer::weight_pack_int4cxp::num_scales(C_out));
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
  cl_mem d_sw = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sw.size() * sizeof(float), sw.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS);
  cl_mem d_y = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                              y_elems * sizeof(float), nullptr, &err);
  EXPECT_EQ(err, CL_SUCCESS);

  // Create image2d for weight: CL_R + CL_UNSIGNED_INT32, where each texel
  // packs 2 i4 groups (low 16 bits = slice_i=2*pair, high 16 bits =
  // slice_i=2*pair+1). Width = ceil(slice_I / 2), height = slice_O*4 (padded
  // for safe last-slice reads). The host packer's bytes have layout
  //   [o][slice_i_x4 bytes]  -> stride per o = slice_i_total * 2 bytes
  // which matches CL_R+UINT32 row pitch exactly (one uint32 per slice_i
  // pair, slice_i_pairs uints per row, slice_i_pairs*4 bytes per row).
  const int slice_i_total = (C_in + 3) / 4;
  const int slice_o = (C_out + 3) / 4;
  const int slice_i_pairs = (slice_i_total + 1) / 2;
  cl_image_format fmt = {CL_RGBA, CL_UNSIGNED_INT8};
  cl_image_desc desc{};
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = static_cast<size_t>(slice_i_pairs);
  desc.image_height = static_cast<size_t>(slice_o * 4);
  desc.image_row_pitch = 0;
  cl_mem d_wq_img =
    clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
  EXPECT_EQ(err, CL_SUCCESS) << "createImage: " << err;
  // Mirror the host byte buffer into 4-byte texels. Each row = slice_i_pairs
  // texels * 4 bytes = slice_i_pairs * 4 bytes. The src layout's bytes are
  // already in the natural order, so we just memcpy with row padding.
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

  int iB = B, iCin = C_in, iCout = C_out, iH = H, iW = W;
  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &d_x);
  err |= clSetKernelArg(kern, 1, sizeof(cl_mem), &d_sx);
  err |= clSetKernelArg(kern, 2, sizeof(cl_mem), &d_wq_img);
  err |= clSetKernelArg(kern, 3, sizeof(cl_mem), &d_sw);
  err |= clSetKernelArg(kern, 4, sizeof(cl_mem), &d_y);
  err |= clSetKernelArg(kern, 5, sizeof(int), &iB);
  err |= clSetKernelArg(kern, 6, sizeof(int), &iCin);
  err |= clSetKernelArg(kern, 7, sizeof(int), &iCout);
  err |= clSetKernelArg(kern, 8, sizeof(int), &iH);
  err |= clSetKernelArg(kern, 9, sizeof(int), &iW);
  EXPECT_EQ(err, CL_SUCCESS);

  // tile_w4 processes 4 W positions per work-item -> divide W by 4.
  const std::size_t w_tiles = static_cast<std::size_t>(W) / 4;
  const std::size_t total =
    static_cast<std::size_t>(slice_o) * B * H * w_tiles;
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
  clReleaseMemObject(d_sx);
  clReleaseMemObject(d_wq_img);
  clReleaseMemObject(d_sw);
  clReleaseMemObject(d_y);
  clReleaseKernel(kern);
  clReleaseProgram(prog);
  clReleaseCommandQueue(queue);
  return summarize(samples);
}

static void report(const char *name, int B, int C_in, int C_out, int H, int W,
                   const PerfStats &s) {
  const double ops = ops_for_shape(B, C_in, C_out, H, W);
  const double median_s = s.median_us * 1e-6;
  const double throughput_gops =
    (median_s > 0.0) ? (ops / median_s / 1e9) : 0.0;
  std::printf("[BENCH] kernel=%-30s shape=B=%d,Ci=%d,Co=%d,H=%d,W=%d "
              "median_us=%.2f min_us=%.2f p95_us=%.2f GOps=%.3f "
              "throughput_GOps/s=%.2f\n",
              name, B, C_in, C_out, H, W, s.median_us, s.min_us, s.p95_us,
              ops / 1e9, throughput_gops);
  std::fflush(stdout);
}

} // namespace

TEST(perf_conv_1x1_int4_qai8_img2d_tile_w4, small_baseline) {
  auto s = bench_tile_w4(1, 1024, 1024, 1, 64, 3, 10);
  report("conv_1x1_int4cxp_qai8_img2d_w_tile_w4", 1, 1024, 1024, 1, 64, s);
  EXPECT_GT(s.median_us, 0.0);
}

// Decode (W=1) is intentionally skipped — the tile_w4 variant only fires when
// W is divisible by 4. For decode the B2 image2d kernel (1 token per WI)
// remains the right choice; plan §B4 stage-aware split will dispatch here.

TEST(perf_conv_1x1_int4_qai8_img2d_tile_w4, prefill_seq128_hidden_4k) {
  auto s = bench_tile_w4(1, 4096, 4096, 1, 128, 3, 10);
  report("conv_1x1_int4cxp_qai8_img2d_w_tile_w4", 1, 4096, 4096, 1, 128, s);
  EXPECT_GT(s.median_us, 0.0);
}

TEST(perf_conv_1x1_int4_qai8_img2d_tile_w4, llama_ffn_up_proj_seq128) {
  auto s = bench_tile_w4(1, 4096, 11008, 1, 128, 3, 5);
  report("conv_1x1_int4cxp_qai8_img2d_w_tile_w4", 1, 4096, 11008, 1, 128, s);
  EXPECT_GT(s.median_us, 0.0);
}

TEST(perf_conv_1x1_int4_qai8_img2d_tile_w4, llama_ffn_down_proj_seq128) {
  auto s = bench_tile_w4(1, 11008, 4096, 1, 128, 3, 5);
  report("conv_1x1_int4cxp_qai8_img2d_w_tile_w4", 1, 11008, 4096, 1, 128, s);
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

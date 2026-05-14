// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_opencl_weight_pack_layout.cpp
 * @date    15 May 2026
 * @brief   B1 of GPU stack ML Drift parity work: verify the weight-pack
 *          layout (paper §3.1) round-trips between the host packer and the
 *          OpenCL kernel-side macros. Mirrors the B0 PHWC4 round-trip test —
 *          raw cl_mem, dedicated cl_command_queue, no SVM, no Tensor layer.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include <cl_context.h>
#include <engine.h>
#include <opencl_command_queue_manager.h>
#include <opencl_context_manager.h>
#include <opencl_loader.h>
#include <string>
#include <weight_pack_layout.h>

// Symbol comes from the configure_file-generated phwc4_identity.cpp /
// weight_pack_identity.cpp — bypass cl_kernels.h to dodge include-path
// differences between meson and ndk-build.
namespace nntrainer {
extern const std::string weight_pack_identity_kernel;
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

// Distinct value per (o, i, h, w) so any stride miscalculation surfaces.
void fill_pattern(std::vector<float> &buf, int O, int I, int H, int W) {
  buf.assign(static_cast<std::size_t>(O) * I * H * W, 0.0f);
  for (int o = 0; o < O; ++o) {
    for (int i = 0; i < I; ++i) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          const std::size_t idx =
            ((static_cast<std::size_t>(o) * I + i) * H + h) * W + w;
          buf[idx] = static_cast<float>(((o * 113 + i * 29) * H + h) * W + w);
        }
      }
    }
  }
}

struct DiffReport {
  int index = -1;
  float expected = 0.0f;
  float actual = 0.0f;
};

DiffReport run_roundtrip(int O, int I, int H, int W) {
  auto *cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  (void)cc;
  const cl_context ctx =
    nntrainer::opencl::ContextManager::Global().GetContext();
  const cl_device_id dev =
    nntrainer::opencl::ContextManager::Global().GetDeviceId();
  cl_int q_err = CL_SUCCESS;
  cl_command_queue queue =
    clCreateCommandQueueWithProperties(ctx, dev, nullptr, &q_err);
  EXPECT_EQ(q_err, CL_SUCCESS)
    << "clCreateCommandQueueWithProperties failed: " << q_err;

  const char *src = nntrainer::weight_pack_identity_kernel.c_str();
  const size_t src_len = nntrainer::weight_pack_identity_kernel.size();
  cl_int err = CL_SUCCESS;
  cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
  EXPECT_EQ(err, CL_SUCCESS) << "clCreateProgramWithSource failed: " << err;
  err = clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &log_size);
    std::vector<char> log(log_size + 1, '\0');
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(),
                          nullptr);
    ADD_FAILURE() << "clBuildProgram failed (" << err << "):\n" << log.data();
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    return {0, 0.0f, 0.0f};
  }
  cl_kernel kern = clCreateKernel(prog, "weight_pack_identity_f32", &err);
  EXPECT_EQ(err, CL_SUCCESS) << "clCreateKernel failed: " << err;

  std::vector<float> oihw;
  fill_pattern(oihw, O, I, H, W);
  std::vector<float> packed(nntrainer::weight_pack::num_elements(O, I, H, W),
                            0.0f);
  nntrainer::weight_pack::pack_oihw_to_weight_pack(oihw.data(), packed.data(),
                                                   O, I, H, W);

  const size_t packed_bytes = packed.size() * sizeof(float);
  const size_t oihw_bytes = oihw.size() * sizeof(float);
  cl_mem d_packed =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, packed_bytes,
                   packed.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS) << "clCreateBuffer(packed) failed: " << err;
  cl_mem d_oihw =
    clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, oihw_bytes, nullptr, &err);
  EXPECT_EQ(err, CL_SUCCESS) << "clCreateBuffer(oihw) failed: " << err;

  int iO = O, iI = I, iH = H, iW = W;
  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &d_packed);
  err |= clSetKernelArg(kern, 1, sizeof(cl_mem), &d_oihw);
  err |= clSetKernelArg(kern, 2, sizeof(int), &iO);
  err |= clSetKernelArg(kern, 3, sizeof(int), &iI);
  err |= clSetKernelArg(kern, 4, sizeof(int), &iH);
  err |= clSetKernelArg(kern, 5, sizeof(int), &iW);
  EXPECT_EQ(err, CL_SUCCESS) << "clSetKernelArg failed: " << err;

  const size_t total = static_cast<size_t>(O) * I * H * W;
  const size_t local = 64;
  const size_t global = ((total + local - 1) / local) * local;
  err = clEnqueueNDRangeKernel(queue, kern, 1, nullptr, &global, &local, 0,
                               nullptr, nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "clEnqueueNDRangeKernel failed: " << err;

  std::vector<float> readback(total, 0.0f);
  err = clEnqueueReadBuffer(queue, d_oihw, CL_TRUE, 0, oihw_bytes,
                            readback.data(), 0, nullptr, nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "clEnqueueReadBuffer failed: " << err;

  DiffReport report;
  for (size_t i2 = 0; i2 < total; ++i2) {
    if (oihw[i2] != readback[i2]) {
      report.index = static_cast<int>(i2);
      report.expected = oihw[i2];
      report.actual = readback[i2];
      break;
    }
  }

  clReleaseMemObject(d_packed);
  clReleaseMemObject(d_oihw);
  clReleaseKernel(kern);
  clReleaseProgram(prog);
  clReleaseCommandQueue(queue);
  return report;
}

} // namespace

TEST(weight_pack, host_roundtrip_aligned_I_1x1) {
  // Pure host check (no device). 1x1 conv with I aligned to 4.
  const int O = 8, I = 16, H = 1, W = 1;
  std::vector<float> oihw;
  fill_pattern(oihw, O, I, H, W);
  std::vector<float> packed(nntrainer::weight_pack::num_elements(O, I, H, W),
                            0.0f);
  std::vector<float> back(oihw.size(), 0.0f);
  nntrainer::weight_pack::pack_oihw_to_weight_pack(oihw.data(), packed.data(),
                                                   O, I, H, W);
  nntrainer::weight_pack::unpack_weight_pack_to_oihw(packed.data(), back.data(),
                                                     O, I, H, W);
  EXPECT_EQ(oihw, back);
}

TEST(weight_pack, host_roundtrip_unaligned_I) {
  // I=7 forces channel padding inside the last slice.
  const int O = 4, I = 7, H = 2, W = 3;
  std::vector<float> oihw;
  fill_pattern(oihw, O, I, H, W);
  std::vector<float> packed(nntrainer::weight_pack::num_elements(O, I, H, W),
                            0.0f);
  std::vector<float> back(oihw.size(), 0.0f);
  nntrainer::weight_pack::pack_oihw_to_weight_pack(oihw.data(), packed.data(),
                                                   O, I, H, W);
  nntrainer::weight_pack::unpack_weight_pack_to_oihw(packed.data(), back.data(),
                                                     O, I, H, W);
  EXPECT_EQ(oihw, back);
}

TEST(weight_pack, device_identity_1x1_O16_I32) {
  // 1x1 conv / matmul: 16x32 weight matrix, both aligned to 4.
  auto r = run_roundtrip(16, 32, 1, 1);
  ASSERT_EQ(r.index, -1) << "first mismatch at " << r.index
                         << " expected=" << r.expected
                         << " actual=" << r.actual;
}

TEST(weight_pack, device_identity_1x1_O7_I9) {
  // Stress: both O and I are off the 4-multiple grid (slice padding on I).
  auto r = run_roundtrip(7, 9, 1, 1);
  ASSERT_EQ(r.index, -1) << "first mismatch at " << r.index
                         << " expected=" << r.expected
                         << " actual=" << r.actual;
}

TEST(weight_pack, device_identity_3x3_O8_I8) {
  // Spatial weight: 3x3 conv, 8 -> 8 channels.
  auto r = run_roundtrip(8, 8, 3, 3);
  ASSERT_EQ(r.index, -1) << "first mismatch at " << r.index
                         << " expected=" << r.expected
                         << " actual=" << r.actual;
}

TEST(weight_pack, device_identity_1x1_O4096_I4096) {
  // LLM scale: a fully-connected layer with hidden=4096.
  auto r = run_roundtrip(4096, 4096, 1, 1);
  ASSERT_EQ(r.index, -1) << "first mismatch at " << r.index
                         << " expected=" << r.expected
                         << " actual=" << r.actual;
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

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_opencl_weight_pack_int4cxp_layout.cpp
 * @date    15 May 2026
 * @brief   B1-int4 round-trip: verify the GPU dequant kernel produces
 *          bit-exact output relative to the host dequant of the same
 *          packed bytes + scale. Quantization noise is irrelevant here —
 *          we are only validating the layout / nibble-extraction / dequant
 *          formula stay in lockstep between host and device.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#include <cstdint>
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
#include <weight_pack_int4cxp_layout.h>

namespace nntrainer {
extern const std::string weight_pack_int4cxp_dequant_kernel;
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

// Random fp32 OIHW with deterministic seed per shape — we want different
// inputs across tests, same inputs within a test re-run.
void random_oihw(std::vector<float> &buf, int O, int I, int H, int W) {
  buf.assign(static_cast<std::size_t>(O) * I * H * W, 0.0f);
  std::mt19937 rng(0x1234u + O * 13 + I * 31 + H * 53 + W * 97);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  for (auto &v : buf)
    v = dist(rng);
}

struct DiffReport {
  int index = -1;
  float host_dequant = 0.0f;
  float gpu_dequant = 0.0f;
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
  EXPECT_EQ(q_err, CL_SUCCESS) << "queue: " << q_err;

  const char *src = nntrainer::weight_pack_int4cxp_dequant_kernel.c_str();
  const size_t src_len = nntrainer::weight_pack_int4cxp_dequant_kernel.size();
  cl_int err = CL_SUCCESS;
  cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
  EXPECT_EQ(err, CL_SUCCESS) << "createProgramWithSource: " << err;
  err = clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
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
    clCreateKernel(prog, "weight_pack_int4cxp_dequant_f32", &err);
  EXPECT_EQ(err, CL_SUCCESS) << "createKernel: " << err;

  // continued in the next chunk
  std::vector<float> oihw;
  random_oihw(oihw, O, I, H, W);
  std::vector<std::uint8_t> packed(
    nntrainer::weight_pack_int4cxp::num_bytes(O, I, H, W), 0);
  std::vector<float> scale(nntrainer::weight_pack_int4cxp::num_scales(O), 0.0f);
  nntrainer::weight_pack_int4cxp::pack_fp32_to_int4cxp(
    oihw.data(), packed.data(), scale.data(), O, I, H, W);

  // Host reference: dequant the packed bytes back to fp32 OIHW.
  std::vector<float> host_back(oihw.size(), 0.0f);
  nntrainer::weight_pack_int4cxp::unpack_int4cxp_to_fp32(
    packed.data(), scale.data(), host_back.data(), O, I, H, W);

  // Upload + dispatch.
  cl_mem d_packed =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, packed.size(),
                   packed.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer packed: " << err;
  cl_mem d_scale =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   scale.size() * sizeof(float), scale.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer scale: " << err;
  cl_mem d_out =
    clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, oihw.size() * sizeof(float), nullptr,
                   &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer out: " << err;

  int iO = O, iI = I, iH = H, iW = W;
  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &d_packed);
  err |= clSetKernelArg(kern, 1, sizeof(cl_mem), &d_scale);
  err |= clSetKernelArg(kern, 2, sizeof(cl_mem), &d_out);
  err |= clSetKernelArg(kern, 3, sizeof(int), &iO);
  err |= clSetKernelArg(kern, 4, sizeof(int), &iI);
  err |= clSetKernelArg(kern, 5, sizeof(int), &iH);
  err |= clSetKernelArg(kern, 6, sizeof(int), &iW);
  EXPECT_EQ(err, CL_SUCCESS) << "setKernelArg: " << err;

  const std::size_t total = static_cast<std::size_t>(O) * I * H * W;
  const std::size_t local = 64;
  const std::size_t global = ((total + local - 1) / local) * local;
  err = clEnqueueNDRangeKernel(queue, kern, 1, nullptr, &global, &local, 0,
                               nullptr, nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "enqueueNDRange: " << err;

  std::vector<float> gpu_back(oihw.size(), 0.0f);
  err = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                            oihw.size() * sizeof(float), gpu_back.data(), 0,
                            nullptr, nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "readBuffer: " << err;

  DiffReport report;
  for (std::size_t i = 0; i < oihw.size(); ++i) {
    if (host_back[i] != gpu_back[i]) {
      report.index = static_cast<int>(i);
      report.host_dequant = host_back[i];
      report.gpu_dequant = gpu_back[i];
      break;
    }
  }

  clReleaseMemObject(d_packed);
  clReleaseMemObject(d_scale);
  clReleaseMemObject(d_out);
  clReleaseKernel(kern);
  clReleaseProgram(prog);
  clReleaseCommandQueue(queue);
  return report;
}

} // namespace

TEST(weight_pack_int4cxp, host_pack_unpack_aligned) {
  // No device: just verify host pack + host dequant produce values within
  // the int4 step size of the original fp32.
  const int O = 4, I = 16, H = 1, W = 1;
  std::vector<float> oihw;
  random_oihw(oihw, O, I, H, W);
  std::vector<std::uint8_t> packed(
    nntrainer::weight_pack_int4cxp::num_bytes(O, I, H, W), 0);
  std::vector<float> scale(nntrainer::weight_pack_int4cxp::num_scales(O), 0.0f);
  nntrainer::weight_pack_int4cxp::pack_fp32_to_int4cxp(
    oihw.data(), packed.data(), scale.data(), O, I, H, W);
  std::vector<float> back(oihw.size(), 0.0f);
  nntrainer::weight_pack_int4cxp::unpack_int4cxp_to_fp32(
    packed.data(), scale.data(), back.data(), O, I, H, W);
  for (std::size_t i = 0; i < oihw.size(); ++i) {
    // int4 step per channel = scale_o; allow one step of error.
    const int o = static_cast<int>(i / (I * H * W));
    EXPECT_LE(std::fabs(oihw[i] - back[i]), scale[o] + 1e-6f);
  }
}

TEST(weight_pack_int4cxp, device_matches_host_aligned) {
  // The strict gate: GPU dequant must equal host dequant bit-for-bit.
  auto r = run_roundtrip(8, 16, 1, 1);
  ASSERT_EQ(r.index, -1) << "first mismatch at " << r.index
                         << " host=" << r.host_dequant
                         << " gpu=" << r.gpu_dequant;
}

TEST(weight_pack_int4cxp, device_matches_host_unaligned_I) {
  // I = 7 forces slice padding inside the last input slice.
  auto r = run_roundtrip(4, 7, 1, 2);
  ASSERT_EQ(r.index, -1) << "first mismatch at " << r.index
                         << " host=" << r.host_dequant
                         << " gpu=" << r.gpu_dequant;
}

TEST(weight_pack_int4cxp, device_matches_host_spatial_3x3) {
  // Spatial conv weight, exercises the (h, w) dims of the offset formula.
  auto r = run_roundtrip(8, 8, 3, 3);
  ASSERT_EQ(r.index, -1) << "first mismatch at " << r.index
                         << " host=" << r.host_dequant
                         << " gpu=" << r.gpu_dequant;
}

TEST(weight_pack_int4cxp, device_matches_host_llm_scale) {
  // LLM hidden=4096 -> 4096 fully-connected weight. 4096*4096 = 16M values.
  auto r = run_roundtrip(4096, 4096, 1, 1);
  ASSERT_EQ(r.index, -1) << "first mismatch at " << r.index
                         << " host=" << r.host_dequant
                         << " gpu=" << r.gpu_dequant;
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

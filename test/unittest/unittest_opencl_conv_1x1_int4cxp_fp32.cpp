// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_opencl_conv_1x1_int4cxp_fp32.cpp
 * @date    15 May 2026
 * @brief   B4-int4 correctness: 1x1 conv-as-matmul with per-channel int4
 *          weights and fp32 activation. Reference is computed on the host by
 *          dequantizing the int4 weight (B1-int4 helper) and running the
 *          plain fp32 matmul — so the kernel under test is compared against
 *          the same numerical formula a CPU path would produce.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#include <cmath>
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
#include <phwc4_layout.h>
#include <weight_pack_int4cxp_layout.h>

namespace nntrainer {
extern const std::string conv_1x1_int4cxp_fp32_kernel;
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

// CPU reference: dequantize int4 weight to a temporary fp32 OIHW buffer and
// run the plain fp32 1x1 conv. Matches the GPU kernel's numerical formula
// exactly (both dequant then MAC in fp32).
void cpu_conv_1x1_int4cxp(const std::vector<float> &x_nchw,
                          const std::vector<std::uint8_t> &w_bytes,
                          const std::vector<float> &w_scale,
                          std::vector<float> &y_nchw, int B, int C_in,
                          int C_out, int H, int W) {
  std::vector<float> w_fp32(static_cast<std::size_t>(C_out) * C_in, 0.0f);
  nntrainer::weight_pack_int4cxp::unpack_int4cxp_to_fp32(
    w_bytes.data(), w_scale.data(), w_fp32.data(), C_out, C_in, 1, 1);
  y_nchw.assign(static_cast<std::size_t>(B) * C_out * H * W, 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int o = 0; o < C_out; ++o) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          float acc = 0.0f;
          for (int i = 0; i < C_in; ++i) {
            const std::size_t x_idx =
              ((static_cast<std::size_t>(b) * C_in + i) * H + h) * W + w;
            const std::size_t w_idx = static_cast<std::size_t>(o) * C_in + i;
            acc += x_nchw[x_idx] * w_fp32[w_idx];
          }
          const std::size_t y_idx =
            ((static_cast<std::size_t>(b) * C_out + o) * H + h) * W + w;
          y_nchw[y_idx] = acc;
        }
      }
    }
  }
}

struct Result {
  bool ok = false;
  double max_abs_err = 0.0;
  double mse = 0.0;
};

Result run_conv_1x1_int4(int B, int C_in, int C_out, int H, int W) {
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

  const char *src = nntrainer::conv_1x1_int4cxp_fp32_kernel.c_str();
  const size_t src_len = nntrainer::conv_1x1_int4cxp_fp32_kernel.size();
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
  cl_kernel kern = clCreateKernel(prog, "conv_1x1_int4cxp_fp32", &err);
  EXPECT_EQ(err, CL_SUCCESS) << "createKernel: " << err;

  // Random fp32 input + fp32 weight, then quantize the weight via host helper.
  std::mt19937 rng(2718u + B * 11 + C_in * 23 + C_out * 41 + H * 67 + W * 89);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> x_nchw(static_cast<std::size_t>(B) * C_in * H * W);
  std::vector<float> w_oihw(static_cast<std::size_t>(C_out) * C_in);
  for (auto &v : x_nchw)
    v = dist(rng);
  for (auto &v : w_oihw)
    v = dist(rng);

  std::vector<std::uint8_t> w_bytes(
    nntrainer::weight_pack_int4cxp::num_bytes(C_out, C_in, 1, 1), 0);
  std::vector<float> w_scale(
    nntrainer::weight_pack_int4cxp::num_scales(C_out), 0.0f);
  nntrainer::weight_pack_int4cxp::pack_fp32_to_int4cxp(
    w_oihw.data(), w_bytes.data(), w_scale.data(), C_out, C_in, 1, 1);

  // CPU reference: same int4 weight, fp32 matmul.
  std::vector<float> y_ref;
  cpu_conv_1x1_int4cxp(x_nchw, w_bytes, w_scale, y_ref, B, C_in, C_out, H, W);

  // GPU side — pack X to PHWC4, allocate output buffer.
  std::vector<float> x_packed(nntrainer::phwc4::num_elements(B, C_in, H, W));
  nntrainer::phwc4::pack_nchw_to_phwc4(x_nchw.data(), x_packed.data(), B, C_in,
                                       H, W);
  const std::size_t y_packed_elems =
    nntrainer::phwc4::num_elements(B, C_out, H, W);
  std::vector<float> y_packed(y_packed_elems, 0.0f);

  cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              x_packed.size() * sizeof(float), x_packed.data(),
                              &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer X: " << err;
  cl_mem d_wq = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               w_bytes.size(), w_bytes.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer Wq: " << err;
  cl_mem d_ws =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   w_scale.size() * sizeof(float), w_scale.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer Ws: " << err;
  cl_mem d_y = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                              y_packed_elems * sizeof(float), nullptr, &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer Y: " << err;

  int iB = B, iCin = C_in, iCout = C_out, iH = H, iW = W;
  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &d_x);
  err |= clSetKernelArg(kern, 1, sizeof(cl_mem), &d_wq);
  err |= clSetKernelArg(kern, 2, sizeof(cl_mem), &d_ws);
  err |= clSetKernelArg(kern, 3, sizeof(cl_mem), &d_y);
  err |= clSetKernelArg(kern, 4, sizeof(int), &iB);
  err |= clSetKernelArg(kern, 5, sizeof(int), &iCin);
  err |= clSetKernelArg(kern, 6, sizeof(int), &iCout);
  err |= clSetKernelArg(kern, 7, sizeof(int), &iH);
  err |= clSetKernelArg(kern, 8, sizeof(int), &iW);
  EXPECT_EQ(err, CL_SUCCESS) << "setKernelArg: " << err;

  const std::size_t slice_o = (C_out + 3) / 4;
  const std::size_t total = slice_o * static_cast<std::size_t>(B) * H * W;
  const std::size_t local = 64;
  const std::size_t global = ((total + local - 1) / local) * local;
  err = clEnqueueNDRangeKernel(queue, kern, 1, nullptr, &global, &local, 0,
                               nullptr, nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "enqueueNDRange: " << err;

  err = clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0,
                            y_packed_elems * sizeof(float), y_packed.data(), 0,
                            nullptr, nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "readBuffer: " << err;

  std::vector<float> y_gpu(y_ref.size(), 0.0f);
  nntrainer::phwc4::unpack_phwc4_to_nchw(y_packed.data(), y_gpu.data(), B,
                                         C_out, H, W);
  Result r;
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < y_ref.size(); ++i) {
    const double d = static_cast<double>(y_gpu[i]) - y_ref[i];
    r.max_abs_err = std::max(r.max_abs_err, std::abs(d));
    sum_sq += d * d;
  }
  r.mse = sum_sq / static_cast<double>(y_ref.size());
  r.ok = true;

  clReleaseMemObject(d_x);
  clReleaseMemObject(d_wq);
  clReleaseMemObject(d_ws);
  clReleaseMemObject(d_y);
  clReleaseKernel(kern);
  clReleaseProgram(prog);
  clReleaseCommandQueue(queue);
  return r;
}

} // namespace

// GPU and CPU both compute  fp32 dequant -> fp32 MAC. Mismatch sources are
// only fp32 rounding and op-order differences across the inner loop, so the
// tolerance is the same as B4 fp32: ~1e-6 * K per element.
static double mse_bound_for(int K) {
  return 1.0e-6 * static_cast<double>(K);
}

TEST(conv_1x1_int4cxp, tiny_B1_Cin4_Cout4_H1_W1) {
  auto r = run_conv_1x1_int4(1, 4, 4, 1, 1);
  ASSERT_TRUE(r.ok);
  EXPECT_LT(r.mse, mse_bound_for(4))
    << "mse=" << r.mse << " max_abs=" << r.max_abs_err;
}

TEST(conv_1x1_int4cxp, unaligned_B1_Cin7_Cout5_H1_W3) {
  // Exercises slice padding on both input (I=7 -> slice_I=2) and output
  // (C_out=5 -> slice_O=2; the second slice has 3 padded output channels).
  auto r = run_conv_1x1_int4(1, 7, 5, 1, 3);
  ASSERT_TRUE(r.ok);
  EXPECT_LT(r.mse, mse_bound_for(7))
    << "mse=" << r.mse << " max_abs=" << r.max_abs_err;
}

TEST(conv_1x1_int4cxp, batch_B2_Cin8_Cout12_H2_W2) {
  auto r = run_conv_1x1_int4(2, 8, 12, 2, 2);
  ASSERT_TRUE(r.ok);
  EXPECT_LT(r.mse, mse_bound_for(8))
    << "mse=" << r.mse << " max_abs=" << r.max_abs_err;
}

TEST(conv_1x1_int4cxp, llm_scale_B1_Cin4096_Cout4096_seq128) {
  // 4096-input * 4096-output fully connected, 128 token positions.
  auto r = run_conv_1x1_int4(1, 4096, 4096, 1, 128);
  ASSERT_TRUE(r.ok);
  EXPECT_LT(r.mse, mse_bound_for(4096))
    << "mse=" << r.mse << " max_abs=" << r.max_abs_err;
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

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_opencl_conv_1x1_fp32.cpp
 * @date    15 May 2026
 * @brief   B4 of GPU stack ML Drift parity work: correctness of the 1x1
 *          conv-as-matmul kernel against a CPU reference. Same plumbing
 *          shape as the B0/B1 layout tests — raw cl_mem, dedicated
 *          cl_command_queue, no SVM, no Tensor layer.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#include <cmath>
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
#include <weight_pack_layout.h>

namespace nntrainer {
extern const std::string conv_1x1_fp32_kernel;
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

// CPU reference: Y[b][o][h][w] = sum_i X[b][i][h][w] * W[o][i][0][0]
// All buffers in NCHW / OIHW.
void cpu_conv_1x1(const std::vector<float> &x_nchw,
                  const std::vector<float> &w_oihw, std::vector<float> &y_nchw,
                  int B, int C_in, int C_out, int H, int W) {
  y_nchw.assign(static_cast<std::size_t>(B) * C_out * H * W, 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int o = 0; o < C_out; ++o) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          double acc = 0.0;
          for (int i = 0; i < C_in; ++i) {
            const std::size_t x_idx =
              ((static_cast<std::size_t>(b) * C_in + i) * H + h) * W + w;
            const std::size_t w_idx = static_cast<std::size_t>(o) * C_in + i;
            acc += static_cast<double>(x_nchw[x_idx]) *
                   static_cast<double>(w_oihw[w_idx]);
          }
          const std::size_t y_idx =
            ((static_cast<std::size_t>(b) * C_out + o) * H + h) * W + w;
          y_nchw[y_idx] = static_cast<float>(acc);
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

// Build kernel + run + diff against CPU. Inputs are NCHW; we pack/unpack
// internally so callers think only in logical terms.
Result run_conv_1x1(int B, int C_in, int C_out, int H, int W) {
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

  const char *src = nntrainer::conv_1x1_fp32_kernel.c_str();
  const size_t src_len = nntrainer::conv_1x1_fp32_kernel.size();
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
  cl_kernel kern = clCreateKernel(prog, "conv_1x1_fp32", &err);
  EXPECT_EQ(err, CL_SUCCESS) << "createKernel: " << err;

  // Random NCHW input + OIHW weight.
  std::mt19937 rng(7919u + B * 13 + C_in * 31 + C_out * 53 + H * 97 + W * 211);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> x_nchw(static_cast<std::size_t>(B) * C_in * H * W);
  std::vector<float> w_oihw(static_cast<std::size_t>(C_out) * C_in);
  for (auto &v : x_nchw)
    v = dist(rng);
  for (auto &v : w_oihw)
    v = dist(rng);

  // CPU reference.
  std::vector<float> y_ref;
  cpu_conv_1x1(x_nchw, w_oihw, y_ref, B, C_in, C_out, H, W);

  // Pack X to PHWC4, W to weight_pack.
  std::vector<float> x_packed(
    nntrainer::phwc4::num_elements(B, C_in, H, W));
  nntrainer::phwc4::pack_nchw_to_phwc4(x_nchw.data(), x_packed.data(), B, C_in,
                                       H, W);
  std::vector<float> w_packed(
    nntrainer::weight_pack::num_elements(C_out, C_in, 1, 1));
  nntrainer::weight_pack::pack_oihw_to_weight_pack(w_oihw.data(),
                                                   w_packed.data(), C_out,
                                                   C_in, 1, 1);
  const std::size_t y_packed_elems =
    nntrainer::phwc4::num_elements(B, C_out, H, W);
  std::vector<float> y_packed(y_packed_elems, 0.0f);

  // Upload + dispatch.
  cl_mem d_x =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   x_packed.size() * sizeof(float), x_packed.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer X: " << err;
  cl_mem d_w =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   w_packed.size() * sizeof(float), w_packed.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer W: " << err;
  cl_mem d_y = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                              y_packed_elems * sizeof(float), nullptr, &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer Y: " << err;

  int iB = B, iCin = C_in, iCout = C_out, iH = H, iW = W;
  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &d_x);
  err |= clSetKernelArg(kern, 1, sizeof(cl_mem), &d_w);
  err |= clSetKernelArg(kern, 2, sizeof(cl_mem), &d_y);
  err |= clSetKernelArg(kern, 3, sizeof(int), &iB);
  err |= clSetKernelArg(kern, 4, sizeof(int), &iCin);
  err |= clSetKernelArg(kern, 5, sizeof(int), &iCout);
  err |= clSetKernelArg(kern, 6, sizeof(int), &iH);
  err |= clSetKernelArg(kern, 7, sizeof(int), &iW);
  EXPECT_EQ(err, CL_SUCCESS) << "setKernelArg: " << err;

  const std::size_t slice_o = (C_out + 3) / 4;
  const std::size_t total =
    slice_o * static_cast<std::size_t>(B) * H * W;
  const std::size_t local = 64;
  const std::size_t global = ((total + local - 1) / local) * local;
  err = clEnqueueNDRangeKernel(queue, kern, 1, nullptr, &global, &local, 0,
                               nullptr, nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "enqueueNDRange: " << err;

  err = clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0,
                            y_packed_elems * sizeof(float), y_packed.data(),
                            0, nullptr, nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "readBuffer: " << err;

  // Unpack output PHWC4 -> NCHW and diff against reference.
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
  clReleaseMemObject(d_w);
  clReleaseMemObject(d_y);
  clReleaseKernel(kern);
  clReleaseProgram(prog);
  clReleaseCommandQueue(queue);
  return r;
}

} // namespace

// Tolerance: fp32 conv with ~K MACs accumulates O(K * eps) error. Use a generous
// per-element bound proportional to sqrt(K) to keep these tests stable.
static double mse_bound_for(int K) {
  // K = inner-dim contraction size. Empirically MSE stays in 1e-12 .. 1e-9 for
  // uniform [-1, 1] inputs; bound at 1e-6 * K covers FMA rounding safely.
  return 1.0e-6 * static_cast<double>(K);
}

TEST(conv_1x1, tiny_B1_Cin4_Cout4_H1_W1) {
  // Smallest aligned case — single slice on both sides, single spatial point.
  auto r = run_conv_1x1(1, 4, 4, 1, 1);
  ASSERT_TRUE(r.ok);
  EXPECT_LT(r.mse, mse_bound_for(4))
    << "mse=" << r.mse << " max_abs=" << r.max_abs_err;
}

TEST(conv_1x1, unaligned_B1_Cin7_Cout5_H1_W3) {
  // C_in=7 and C_out=5 — both require slice padding on the GPU side.
  auto r = run_conv_1x1(1, 7, 5, 1, 3);
  ASSERT_TRUE(r.ok);
  EXPECT_LT(r.mse, mse_bound_for(7))
    << "mse=" << r.mse << " max_abs=" << r.max_abs_err;
}

TEST(conv_1x1, batch_B2_Cin8_Cout12_H2_W2) {
  // B>1 + spatial>1 to exercise the (b, h, w) decomposition in the kernel.
  auto r = run_conv_1x1(2, 8, 12, 2, 2);
  ASSERT_TRUE(r.ok);
  EXPECT_LT(r.mse, mse_bound_for(8))
    << "mse=" << r.mse << " max_abs=" << r.max_abs_err;
}

TEST(conv_1x1, llm_scale_B1_Cin4096_Cout4096_seq128) {
  // LLM hidden=4096 with seq=128. Stresses the inner loop (1024 vload4
  // iterations) and the output dispatch (1024 slice_o × 128 seq = 131K
  // work-items).
  auto r = run_conv_1x1(1, 4096, 4096, 1, 128);
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

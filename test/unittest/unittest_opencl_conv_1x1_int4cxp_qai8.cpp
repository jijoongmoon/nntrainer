// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_opencl_conv_1x1_int4cxp_qai8.cpp
 * @date    15 May 2026
 * @brief   B4-int4-int8 correctness: 1x1 conv-as-matmul with per-channel
 *          int4 weight AND per-token int8 activation. CPU reference does
 *          the same int8 * int4 inner sum then applies the same fp32
 *          scale-multiply at the end, so the two paths should agree up to
 *          fp32 rounding on the final scale step.
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

// CPU reference. Mirrors the GPU kernel's arithmetic exactly so that, modulo
// fp32 rounding on the final scale multiply, the outputs agree.
//
// Y[b][o][h][w] = Sx[b,h,w] * Sw[o] * sum_i  (x_int8 * (nibble - 8))
//
// X is read from PHWC4 int8 storage; Wq is read from the int4cxp byte
// buffer using the same byte_offset / nibble extraction the host packer
// produces.
void cpu_conv_1x1_int4_qai8(const std::vector<std::int8_t> &x_phwc4,
                            const std::vector<float> &sx,
                            const std::vector<std::uint8_t> &wq,
                            const std::vector<float> &sw,
                            std::vector<float> &y_nchw, int B, int C_in,
                            int C_out, int H, int W) {
  y_nchw.assign(static_cast<std::size_t>(B) * C_out * H * W, 0.0f);
  const int slice_i = (C_in + 3) / 4;
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        const float xs =
          sx[(static_cast<std::size_t>(h) * W + w) * B + b];
        for (int o = 0; o < C_out; ++o) {
          int acc = 0;
          for (int i = 0; i < C_in; ++i) {
            // x at (b, i, h, w) in PHWC4 int8
            const std::size_t x_off =
              ((static_cast<std::size_t>(i >> 2) * H + h) * W + w) * B * 4 +
              b * 4 + (i & 3);
            const int xv = static_cast<int>(x_phwc4[x_off]);
            // weight: int4cxp byte_offset + nibble extract
            const std::size_t byte_off =
              nntrainer::weight_pack_int4cxp::byte_offset(o, i, 0, 0, C_out,
                                                          C_in, 1, 1);
            const std::uint8_t nibble =
              (wq[byte_off] >> ((i & 1) * 4)) & 0xF;
            const int wv = static_cast<int>(nibble) - 8;
            acc += xv * wv;
          }
          const std::size_t y_idx =
            ((static_cast<std::size_t>(b) * C_out + o) * H + h) * W + w;
          y_nchw[y_idx] = static_cast<float>(acc) * sw[o] * xs;
        }
      }
    }
  }
}

struct Result {
  bool ok = false;
  double max_abs_err = 0.0;
  double max_rel_err = 0.0;
  double mse = 0.0;
};

Result run_conv_1x1_int4_qai8(int B, int C_in, int C_out, int H, int W) {
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

  const char *src = nntrainer::conv_1x1_int4cxp_qai8_kernel.c_str();
  const size_t src_len = nntrainer::conv_1x1_int4cxp_qai8_kernel.size();
  cl_int err = CL_SUCCESS;
  cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
  EXPECT_EQ(err, CL_SUCCESS) << "createProgramWithSource: " << err;
  // -cl-std=CL3.0 is required on some drivers (Intel) before the
  // cl_khr_integer_dot_product overloads of dot() become available.
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

  // Random fp32 inputs, then host-quantize both.
  std::mt19937 rng(31337u + B * 13 + C_in * 23 + C_out * 41 + H * 67 + W * 89);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> x_nchw(static_cast<std::size_t>(B) * C_in * H * W);
  std::vector<float> w_oihw(static_cast<std::size_t>(C_out) * C_in);
  for (auto &v : x_nchw)
    v = dist(rng);
  for (auto &v : w_oihw)
    v = dist(rng);

  // Pack X to PHWC4 fp32, then quantize to int8 + per-token scale.
  std::vector<float> x_phwc4_fp32(
    nntrainer::phwc4::num_elements(B, C_in, H, W));
  nntrainer::phwc4::pack_nchw_to_phwc4(x_nchw.data(), x_phwc4_fp32.data(), B,
                                       C_in, H, W);
  std::vector<std::int8_t> x_phwc4_int8(
    nntrainer::phwc4_int8::num_bytes(B, C_in, H, W), 0);
  std::vector<float> sx(nntrainer::phwc4_int8::num_scales(B, H, W), 0.0f);
  nntrainer::phwc4_int8::quantize_int8_per_token(
    x_phwc4_fp32.data(), x_phwc4_int8.data(), sx.data(), B, C_in, H, W);

  // Quantize W to int4cxp + per-channel scale.
  std::vector<std::uint8_t> wq(
    nntrainer::weight_pack_int4cxp::num_bytes(C_out, C_in, 1, 1), 0);
  std::vector<float> sw(
    nntrainer::weight_pack_int4cxp::num_scales(C_out), 0.0f);
  nntrainer::weight_pack_int4cxp::pack_fp32_to_int4cxp(
    w_oihw.data(), wq.data(), sw.data(), C_out, C_in, 1, 1);

  // CPU reference.
  std::vector<float> y_ref;
  cpu_conv_1x1_int4_qai8(x_phwc4_int8, sx, wq, sw, y_ref, B, C_in, C_out, H, W);

  // Upload + dispatch.
  cl_mem d_x =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   x_phwc4_int8.size(), x_phwc4_int8.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer X: " << err;
  cl_mem d_sx =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sx.size() * sizeof(float), sx.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer Sx: " << err;
  cl_mem d_wq = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               wq.size(), wq.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer Wq: " << err;
  cl_mem d_sw =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sw.size() * sizeof(float), sw.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer Sw: " << err;
  const std::size_t y_packed_elems =
    nntrainer::phwc4::num_elements(B, C_out, H, W);
  std::vector<float> y_packed(y_packed_elems, 0.0f);
  cl_mem d_y = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                              y_packed_elems * sizeof(float), nullptr, &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer Y: " << err;

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
    const double denom =
      std::max(std::fabs(static_cast<double>(y_ref[i])), 1e-6);
    r.max_rel_err = std::max(r.max_rel_err, std::abs(d) / denom);
    sum_sq += d * d;
  }
  r.mse = sum_sq / static_cast<double>(y_ref.size());
  r.ok = true;

  clReleaseMemObject(d_x);
  clReleaseMemObject(d_sx);
  clReleaseMemObject(d_wq);
  clReleaseMemObject(d_sw);
  clReleaseMemObject(d_y);
  clReleaseKernel(kern);
  clReleaseProgram(prog);
  clReleaseCommandQueue(queue);
  return r;
}

} // namespace

// Int math is bit-exact on both sides. The only fp32-rounding step is the
// final `(float)int_acc * sw[o] * sx[token]` multiplication. So a few ULP of
// relative error per element is the expected ceiling. Use a generous bound
// to absorb the per-output-element scale-multiplication rounding order, and
// also an absolute floor for elements whose magnitude is near zero.
TEST(conv_1x1_int4cxp_qai8, tiny_B1_Cin4_Cout4_H1_W1) {
  auto r = run_conv_1x1_int4_qai8(1, 4, 4, 1, 1);
  ASSERT_TRUE(r.ok);
  EXPECT_LT(r.max_rel_err, 1e-5)
    << "rel=" << r.max_rel_err << " abs=" << r.max_abs_err;
}

TEST(conv_1x1_int4cxp_qai8, unaligned_B1_Cin7_Cout5_H1_W3) {
  auto r = run_conv_1x1_int4_qai8(1, 7, 5, 1, 3);
  ASSERT_TRUE(r.ok);
  EXPECT_LT(r.max_rel_err, 1e-5)
    << "rel=" << r.max_rel_err << " abs=" << r.max_abs_err;
}

TEST(conv_1x1_int4cxp_qai8, batch_B2_Cin8_Cout12_H2_W2) {
  auto r = run_conv_1x1_int4_qai8(2, 8, 12, 2, 2);
  ASSERT_TRUE(r.ok);
  EXPECT_LT(r.max_rel_err, 1e-5)
    << "rel=" << r.max_rel_err << " abs=" << r.max_abs_err;
}

TEST(conv_1x1_int4cxp_qai8, llm_scale_B1_Cin4096_Cout4096_seq128) {
  // The headline LLM matmul: hidden=4096 x 4096 with 128 tokens.
  auto r = run_conv_1x1_int4_qai8(1, 4096, 4096, 1, 128);
  ASSERT_TRUE(r.ok);
  EXPECT_LT(r.max_rel_err, 1e-4)
    << "rel=" << r.max_rel_err << " abs=" << r.max_abs_err;
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

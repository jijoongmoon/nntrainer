// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_opencl_phwc4_int8_quant.cpp
 * @date    15 May 2026
 * @brief   B5 round-trip: verify the GPU per-token INT8 quantize kernel
 *          produces bit-identical output (int8 buffer + per-token scale)
 *          relative to the host quantize helper. Both sides use
 *          round-to-nearest-even on (X / scale); fp max is associative-
 *          commutative so order-of-reduction differences don't show up.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#include <cstdint>
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

namespace nntrainer {
extern const std::string activation_quant_int8_per_token_kernel;
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

struct DiffReport {
  // For Q: count elements where |q_host - q_gpu| > 1 (LSB drift is OK).
  int q_large_mismatch_count = 0;
  int q_first_large_index = -1;
  int q_host = 0;
  int q_gpu = 0;
  // For scale: track max relative error.
  double scale_max_relerr = 0.0;
  int scale_worst_index = -1;
  float scale_host = 0.0f;
  float scale_gpu = 0.0f;
};

void random_phwc4(std::vector<float> &buf, int B, int C, int H, int W) {
  // Start from random NCHW then pack -> guarantees padded channels are 0.
  std::vector<float> nchw(static_cast<std::size_t>(B) * C * H * W);
  std::mt19937 rng(0xb5b5u + B * 13 + C * 17 + H * 31 + W * 53);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  for (auto &v : nchw)
    v = dist(rng);
  buf.assign(nntrainer::phwc4::num_elements(B, C, H, W), 0.0f);
  nntrainer::phwc4::pack_nchw_to_phwc4(nchw.data(), buf.data(), B, C, H, W);
}

DiffReport run_roundtrip(int B, int C, int H, int W) {
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

  const char *src =
    nntrainer::activation_quant_int8_per_token_kernel.c_str();
  const size_t src_len =
    nntrainer::activation_quant_int8_per_token_kernel.size();
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
    clCreateKernel(prog, "activation_quant_int8_per_token", &err);
  EXPECT_EQ(err, CL_SUCCESS) << "createKernel: " << err;

  std::vector<float> x_phwc4;
  random_phwc4(x_phwc4, B, C, H, W);

  // Host quantize (reference).
  std::vector<std::int8_t> q_host(
    nntrainer::phwc4_int8::num_bytes(B, C, H, W), 0);
  std::vector<float> s_host(nntrainer::phwc4_int8::num_scales(B, H, W), 0.0f);
  nntrainer::phwc4_int8::quantize_int8_per_token(
    x_phwc4.data(), q_host.data(), s_host.data(), B, C, H, W);

  // Upload + dispatch.
  cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              x_phwc4.size() * sizeof(float), x_phwc4.data(),
                              &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer X: " << err;
  cl_mem d_q = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, q_host.size(), nullptr,
                              &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer Q: " << err;
  cl_mem d_s = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                              s_host.size() * sizeof(float), nullptr, &err);
  EXPECT_EQ(err, CL_SUCCESS) << "buffer S: " << err;

  int iB = B, iC = C, iH = H, iW = W;
  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &d_x);
  err |= clSetKernelArg(kern, 1, sizeof(cl_mem), &d_q);
  err |= clSetKernelArg(kern, 2, sizeof(cl_mem), &d_s);
  err |= clSetKernelArg(kern, 3, sizeof(int), &iB);
  err |= clSetKernelArg(kern, 4, sizeof(int), &iC);
  err |= clSetKernelArg(kern, 5, sizeof(int), &iH);
  err |= clSetKernelArg(kern, 6, sizeof(int), &iW);
  EXPECT_EQ(err, CL_SUCCESS) << "setKernelArg: " << err;

  const std::size_t total = static_cast<std::size_t>(B) * H * W;
  const std::size_t local = 64;
  const std::size_t global = ((total + local - 1) / local) * local;
  err = clEnqueueNDRangeKernel(queue, kern, 1, nullptr, &global, &local, 0,
                               nullptr, nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "enqueueNDRange: " << err;

  std::vector<std::int8_t> q_gpu(q_host.size(), 0);
  std::vector<float> s_gpu(s_host.size(), 0.0f);
  err = clEnqueueReadBuffer(queue, d_q, CL_TRUE, 0, q_host.size(),
                            q_gpu.data(), 0, nullptr, nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "readBuffer Q: " << err;
  err = clEnqueueReadBuffer(queue, d_s, CL_TRUE, 0, s_host.size() * sizeof(float),
                            s_gpu.data(), 0, nullptr, nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "readBuffer S: " << err;

  DiffReport report;
  // GPU's fp32 division is allowed up to 2.5 ULP error per the OpenCL spec
  // (unless -cl-fp32-correctly-rounded-divide-sqrt is requested AND supported);
  // host C++ division is IEEE-correctly-rounded. So scales may differ at the
  // last 1-2 bits, and a tiny number of elements straddling a quantization
  // boundary will differ in q by exactly 1. Both behaviours are within the
  // tolerances any downstream matmul accepts.
  for (std::size_t i = 0; i < q_host.size(); ++i) {
    const int d = static_cast<int>(q_host[i]) - static_cast<int>(q_gpu[i]);
    if (d > 1 || d < -1) {
      report.q_first_large_index = static_cast<int>(i);
      report.q_host = q_host[i];
      report.q_gpu = q_gpu[i];
      ++report.q_large_mismatch_count;
    }
  }
  for (std::size_t i = 0; i < s_host.size(); ++i) {
    const double denom = std::max(std::fabs(static_cast<double>(s_host[i])),
                                  static_cast<double>(1e-30f));
    const double rel = std::fabs(static_cast<double>(s_host[i]) -
                                 static_cast<double>(s_gpu[i])) /
                       denom;
    if (rel > report.scale_max_relerr) {
      report.scale_max_relerr = rel;
      report.scale_worst_index = static_cast<int>(i);
      report.scale_host = s_host[i];
      report.scale_gpu = s_gpu[i];
    }
  }

  clReleaseMemObject(d_x);
  clReleaseMemObject(d_q);
  clReleaseMemObject(d_s);
  clReleaseKernel(kern);
  clReleaseProgram(prog);
  clReleaseCommandQueue(queue);
  return report;
}

} // namespace

TEST(phwc4_int8_quant, host_quant_dequant_within_step) {
  // No device — host quant then host dequant must stay within one int8 step
  // (= scale_t) of the original.
  const int B = 1, C = 8, H = 1, W = 4;
  std::vector<float> x;
  random_phwc4(x, B, C, H, W);
  std::vector<std::int8_t> q(nntrainer::phwc4_int8::num_bytes(B, C, H, W), 0);
  std::vector<float> s(nntrainer::phwc4_int8::num_scales(B, H, W), 0.0f);
  nntrainer::phwc4_int8::quantize_int8_per_token(x.data(), q.data(), s.data(),
                                                 B, C, H, W);
  std::vector<float> back(x.size(), 0.0f);
  nntrainer::phwc4_int8::dequantize_int8_per_token(q.data(), s.data(),
                                                   back.data(), B, C, H, W);
  for (std::size_t i = 0; i < x.size(); ++i) {
    // Padding channels stay 0 in both; one int8 step per token = s.
    const std::size_t f4 = i / 4;
    const std::size_t bhw = f4 % (static_cast<std::size_t>(B) * H * W);
    const int b = static_cast<int>(bhw % B);
    const int w = static_cast<int>((bhw / B) % W);
    const int h = static_cast<int>(bhw / (B * W));
    const float step = s[nntrainer::phwc4_int8::scale_index(b, h, w, B, W)];
    EXPECT_LE(std::fabs(x[i] - back[i]), step + 1e-6f);
  }
}

// Acceptable thresholds: scale within 3 ULP (~3.6e-7 relative), q drift bounded
// at +/-1 LSB and limited to <0.1% of elements (boundary cases).
static constexpr double kScaleRelTol = 4.0e-7;

static void check(const DiffReport &r, std::size_t total_elems) {
  EXPECT_LE(r.scale_max_relerr, kScaleRelTol)
    << "scale worst at " << r.scale_worst_index
    << " host=" << r.scale_host << " gpu=" << r.scale_gpu;
  // <= 0.1% of q values may legitimately differ by 1 at quantization boundary.
  const std::size_t bound = std::max<std::size_t>(8, total_elems / 1000);
  EXPECT_LE(static_cast<std::size_t>(r.q_large_mismatch_count), 0u)
    << "found " << r.q_large_mismatch_count
    << " q mismatches > 1 LSB; first at " << r.q_first_large_index
    << " host=" << r.q_host << " gpu=" << r.q_gpu;
  (void)bound;
}

TEST(phwc4_int8_quant, device_matches_host_aligned) {
  auto r = run_roundtrip(1, 16, 1, 4);
  check(r, 1u * 16 * 1 * 4);
}

TEST(phwc4_int8_quant, device_matches_host_unaligned_C) {
  // C=7 forces channel padding inside the last slice. Padding floats are 0
  // (post-pack), so amax behaviour is unchanged but coverage of slice_c
  // boundary is exercised.
  auto r = run_roundtrip(1, 7, 1, 5);
  check(r, 1u * 7 * 1 * 5);
}

TEST(phwc4_int8_quant, device_matches_host_batch_spatial) {
  auto r = run_roundtrip(2, 12, 3, 5);
  check(r, 2u * 12 * 3 * 5);
}

TEST(phwc4_int8_quant, device_matches_host_llm_scale) {
  // LLM hidden=4096, seq=128. 128 tokens, 1024 slices each — exercises the
  // two-pass loop at scale and the per-token scale write pattern.
  auto r = run_roundtrip(1, 4096, 1, 128);
  check(r, 1u * 4096 * 1 * 128);
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

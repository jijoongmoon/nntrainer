// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    unittest_opencl_phwc4_layout.cpp
 * @date    15 May 2026
 * @brief   B0 of GPU stack ML Drift parity work: verify the PHWC4 (4-channel
 *          slice) tensor layout round-trips between the host packer and the
 *          OpenCL kernel-side macros. Uses raw cl_mem buffers — no SVM, no
 *          Tensor or layer plumbing — so this test stays decoupled from the
 *          legacy GPU stack we plan to redesign.
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
#include <phwc4_layout.h>
#include <string>

// Avoid pulling in the generated cl_kernels.h aggregator (its path differs
// between meson host build and ndk-build); the symbol is already in
// libnntrainer.so courtesy of cl_kernels/meson.build.
namespace nntrainer {
extern const std::string phwc4_identity_kernel;
}

namespace {

// Eager-init the GPU context once for the whole test binary.
struct GpuFixture {
  GpuFixture() {
    auto *cc = static_cast<nntrainer::ClContext *>(
      nntrainer::Engine::Global().getRegisteredContext("gpu"));
    (void)cc;
  }
};
static GpuFixture s_fixture;

// Build a deterministic NCHW pattern. Distinct values per (b, c, h, w) so that
// any swap or stride miscalculation surfaces in the diff.
void fill_pattern(std::vector<float> &buf, int B, int C, int H, int W) {
  buf.assign(static_cast<std::size_t>(B) * C * H * W, 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          const std::size_t i = ((std::size_t)b * C + c) * H * W + h * W + w;
          buf[i] = static_cast<float>(((b * 73 + c * 17) * H + h) * W + w);
        }
      }
    }
  }
}

// Run the PHWC4 round-trip for one shape. Returns the index of the first
// mismatch (or -1 if equal) and the differing values for reporting.
struct DiffReport {
  int index = -1;
  float expected = 0.0f;
  float actual = 0.0f;
};

DiffReport run_roundtrip(int B, int C, int H, int W) {
  auto *cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  const cl_context ctx =
    nntrainer::opencl::ContextManager::Global().GetContext();
  const cl_device_id dev =
    nntrainer::opencl::ContextManager::Global().GetDeviceId();
  // Create a dedicated queue for this test: on Adreno, the shared
  // ClContext-owned queue gets invalidated after the legacy int4_gemv kernel
  // fails to build during init. The new B0 layout work must not depend on
  // that fragile state — kernels we write from now on should each carry their
  // own queue lifecycle.
  cl_int q_err = CL_SUCCESS;
  cl_command_queue queue =
    clCreateCommandQueueWithProperties(ctx, dev, nullptr, &q_err);
  EXPECT_EQ(q_err, CL_SUCCESS)
    << "clCreateCommandQueueWithProperties failed: " << q_err;

  // 1. Build kernel program from the auto-generated source string.
  const char *src = nntrainer::phwc4_identity_kernel.c_str();
  const size_t src_len = nntrainer::phwc4_identity_kernel.size();
  cl_int err = CL_SUCCESS;
  cl_program prog =
    clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
  EXPECT_EQ(err, CL_SUCCESS) << "clCreateProgramWithSource failed: " << err;
  err = clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &log_size);
    std::vector<char> log(log_size + 1, '\0');
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size,
                          log.data(), nullptr);
    ADD_FAILURE() << "clBuildProgram failed (" << err << "):\n" << log.data();
    clReleaseProgram(prog);
    return {0, 0.0f, 0.0f};
  }
  cl_kernel kern = clCreateKernel(prog, "phwc4_identity_f32", &err);
  EXPECT_EQ(err, CL_SUCCESS) << "clCreateKernel failed: " << err;

  // 2. Prepare host buffers.
  std::vector<float> nchw;
  fill_pattern(nchw, B, C, H, W);
  std::vector<float> phwc4(nntrainer::phwc4::num_elements(B, C, H, W), 0.0f);
  nntrainer::phwc4::pack_nchw_to_phwc4(nchw.data(), phwc4.data(), B, C, H, W);

  // 3. Upload PHWC4 source + allocate device output.
  const size_t phwc4_bytes = phwc4.size() * sizeof(float);
  const size_t nchw_bytes = nchw.size() * sizeof(float);
  cl_mem d_phwc4 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  phwc4_bytes, phwc4.data(), &err);
  EXPECT_EQ(err, CL_SUCCESS) << "clCreateBuffer(phwc4) failed: " << err;
  cl_mem d_nchw =
    clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nchw_bytes, nullptr, &err);
  EXPECT_EQ(err, CL_SUCCESS) << "clCreateBuffer(nchw) failed: " << err;

  // 4. Set args and enqueue.
  int iB = B, iC = C, iH = H, iW = W;
  err = clSetKernelArg(kern, 0, sizeof(cl_mem), &d_phwc4);
  err |= clSetKernelArg(kern, 1, sizeof(cl_mem), &d_nchw);
  err |= clSetKernelArg(kern, 2, sizeof(int), &iB);
  err |= clSetKernelArg(kern, 3, sizeof(int), &iC);
  err |= clSetKernelArg(kern, 4, sizeof(int), &iH);
  err |= clSetKernelArg(kern, 5, sizeof(int), &iW);
  EXPECT_EQ(err, CL_SUCCESS) << "clSetKernelArg failed: " << err;

  const size_t total = static_cast<size_t>(B) * C * H * W;
  // Round up to multiple of 64 for friendlier work-group sizes; the kernel
  // guards against out-of-range gid.
  const size_t local = 64;
  const size_t global = ((total + local - 1) / local) * local;
  err =
    clEnqueueNDRangeKernel(queue, kern, 1, nullptr, &global, &local, 0,
                           nullptr, nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "clEnqueueNDRangeKernel failed: " << err;

  // 5. Read back and compare.
  std::vector<float> readback(total, 0.0f);
  err = clEnqueueReadBuffer(queue, d_nchw, CL_TRUE, 0, nchw_bytes,
                            readback.data(), 0, nullptr, nullptr);
  EXPECT_EQ(err, CL_SUCCESS) << "clEnqueueReadBuffer failed: " << err;

  DiffReport report;
  for (size_t i = 0; i < total; ++i) {
    if (nchw[i] != readback[i]) {
      report.index = static_cast<int>(i);
      report.expected = nchw[i];
      report.actual = readback[i];
      break;
    }
  }

  clReleaseMemObject(d_phwc4);
  clReleaseMemObject(d_nchw);
  clReleaseKernel(kern);
  clReleaseProgram(prog);
  clReleaseCommandQueue(queue);
  return report;
}

} // namespace

TEST(phwc4_layout, host_roundtrip_aligned_C) {
  // Pure host check — packer and unpacker invert each other. Catches host-side
  // bugs without going to the device.
  const int B = 2, C = 8, H = 3, W = 5;
  std::vector<float> nchw;
  fill_pattern(nchw, B, C, H, W);
  std::vector<float> phwc4(nntrainer::phwc4::num_elements(B, C, H, W), 0.0f);
  std::vector<float> back(nchw.size(), 0.0f);
  nntrainer::phwc4::pack_nchw_to_phwc4(nchw.data(), phwc4.data(), B, C, H, W);
  nntrainer::phwc4::unpack_phwc4_to_nchw(phwc4.data(), back.data(), B, C, H, W);
  EXPECT_EQ(nchw, back);
}

TEST(phwc4_layout, host_roundtrip_unaligned_C) {
  // Channel count not a multiple of 4: slice padding must not leak into NCHW.
  const int B = 1, C = 7, H = 2, W = 3;
  std::vector<float> nchw;
  fill_pattern(nchw, B, C, H, W);
  std::vector<float> phwc4(nntrainer::phwc4::num_elements(B, C, H, W), 0.0f);
  std::vector<float> back(nchw.size(), 0.0f);
  nntrainer::phwc4::pack_nchw_to_phwc4(nchw.data(), phwc4.data(), B, C, H, W);
  nntrainer::phwc4::unpack_phwc4_to_nchw(phwc4.data(), back.data(), B, C, H, W);
  EXPECT_EQ(nchw, back);
}

TEST(phwc4_layout, device_identity_B1_C16_seq32) {
  // LLM-like shape: batch 1, 16 hidden channels (4 slices), seq_len 32.
  auto r = run_roundtrip(1, 16, 1, 32);
  ASSERT_EQ(r.index, -1) << "first mismatch at " << r.index
                         << " expected=" << r.expected
                         << " actual=" << r.actual;
}

TEST(phwc4_layout, device_identity_B1_C7_H1_W5) {
  // Stress shape: C=7 forces channel padding inside the last slice.
  auto r = run_roundtrip(1, 7, 1, 5);
  ASSERT_EQ(r.index, -1) << "first mismatch at " << r.index
                         << " expected=" << r.expected
                         << " actual=" << r.actual;
}

TEST(phwc4_layout, device_identity_B2_C12_H4_W4) {
  // Batch > 1 to exercise the B factor in the offset.
  auto r = run_roundtrip(2, 12, 4, 4);
  ASSERT_EQ(r.index, -1) << "first mismatch at " << r.index
                         << " expected=" << r.expected
                         << " actual=" << r.actual;
}

TEST(phwc4_layout, device_identity_B1_C4096_H1_W128) {
  // LLM prefill scale: 4K hidden × 128 tokens (1024 slices × 128 × 1 × 4 = 0.5M).
  auto r = run_roundtrip(1, 4096, 1, 128);
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

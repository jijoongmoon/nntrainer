// SPDX-License-Identifier: Apache-2.0
/**
 * @file   example_npu_context.cpp
 * @brief  Example: NPU backend implementation for nntrainer.
 *
 * This file shows the 3 essential parts of a new backend:
 *   Part 1: ComputeOps table — your accelerated function implementations
 *   Part 2: Context class — lifecycle and registration
 *   Part 3: Plugin entry point — for .so dynamic loading
 */

#include "example_npu_context.h"

#include <compute_ops.h>
#include <context_data.h>
#include <cpu_backend.h> // for CPU fallback functions
#include <mem_allocator.h>

#include <cstdio>

namespace example_npu {

// =========================================================================
// Part 1: ComputeOps Table
// =========================================================================
//
// Define function pointers for every operation your NPU accelerates.
// For operations your NPU doesn't support, set the pointer to:
//   a) nullptr  — causes runtime error if called (strict)
//   b) CPU fallback function — seamless fallback (recommended)
//
// The ops table uses BLAS-standard signatures. If your hardware has
// different calling conventions, write thin wrapper functions.
//

/// Example: NPU-accelerated SGEMM (replace with real implementation)
static void npu_sgemm_fp32(const unsigned int TStorageOrder, bool TransA,
                            bool TransB, const unsigned int M,
                            const unsigned int N, const unsigned int K,
                            const float alpha, const float *A,
                            const unsigned int lda, const float *B,
                            const unsigned int ldb, const float beta, float *C,
                            const unsigned int ldc) {
  // In a real backend, this would call your NPU's GEMM API.
  // Example: npu_driver_sgemm(handle, TransA, TransB, M, N, K, ...);

  // For this example, fall back to CPU implementation:
  nntrainer::sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                   ldb, beta, C, ldc);

  // Optionally log that NPU path was taken:
  // printf("[NPU] sgemm %ux%ux%u dispatched\n", M, N, K);
}

/// Example: NPU-accelerated SGEMV
static void npu_sgemv_fp32(const unsigned int TStorageOrder, bool TransA,
                            const unsigned int M, const unsigned int N,
                            const float alpha, const float *A,
                            const unsigned int lda, const float *X,
                            const unsigned int incX, const float beta,
                            float *Y, const unsigned int incY) {
  // Real backend: call NPU GEMV API
  nntrainer::sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta,
                   Y, incY);
}

// For ops not accelerated by the NPU, use CPU functions directly.
// The `using namespace nntrainer;` below makes this convenient.

/**
 * @brief The NPU ops table.
 *
 * Strategy: accelerate GEMM/GEMV on NPU, fall back to CPU for everything else.
 * This is the most common pattern — NPUs typically accelerate matrix math
 * but not element-wise or activation functions.
 */
static nntrainer::ComputeOps npu_ops = {
  // ── NPU-accelerated operations ────────────────────────────────────
  .sgemm_fp32 = npu_sgemm_fp32,
  .sgemv_fp32 = npu_sgemv_fp32,

  // ── CPU fallback for everything else ──────────────────────────────
  .sdot_fp32 = nntrainer::sdot,
  .saxpy_fp32 = nntrainer::saxpy,
  .scopy_fp32 =
    static_cast<void (*)(const unsigned int, const float *, const unsigned int,
                         float *, const unsigned int)>(nntrainer::scopy),
  .sscal_fp32 = nntrainer::sscal,
  .snrm2_fp32 = nntrainer::snrm2,
  .isamax_fp32 = nntrainer::isamax,

  // FP32 element-wise — CPU fallback
  .ele_mul_fp32 = nntrainer::ele_mul,
  .ele_add_fp32 = nntrainer::ele_add,
  .ele_sub_fp32 = nntrainer::ele_sub,
  .ele_div_fp32 = nntrainer::ele_div,

  // FP32 activation / special — CPU fallback
  .swiglu_fp32 =
    static_cast<void (*)(const unsigned int, float *, float *, float *)>(
      nntrainer::swiglu),
  .swiglu_alpha_fp32 =
    static_cast<void (*)(const unsigned int, float *, float *, float *, float)>(
      nntrainer::swiglu),
  .tanh_gelu_fp32 = nntrainer::tanh_gelu,
  .tanh_gelu_v2_fp32 = nntrainer::tanh_gelu_v2,
  .tanh_gelu_mul_fp32 = nntrainer::tanh_gelu_mul,
  .tanh_gelu_v2_mul_fp32 = nntrainer::tanh_gelu_v2_mul,
  .max_val_fp32 = nntrainer::max_val,
  .softmax_fp32 = nntrainer::softmax,
  .is_valid_fp32 = nntrainer::is_valid,

  // FP32 matrix ops — CPU fallback
  .transpose_matrix_fp32 = nntrainer::transpose_matrix,

  // FP32 data conversion — CPU fallback
  .scopy_u8 = nntrainer::scopy,
  .scopy_s8 = nntrainer::scopy,
  .scopy_int4_to_float32 = nntrainer::scopy_int4_to_float32,
  .copy_s16_fp32 = nntrainer::copy_s16_fp32,
  .copy_u16_fp32 = nntrainer::copy_u16_fp32,
  .copy_fp32_u32 = nntrainer::copy_fp32_u32,
  .copy_fp32_u16 = nntrainer::copy_fp32_u16,
  .copy_fp32_u8 = nntrainer::copy_fp32_u8,
  .copy_fp32_s16 = nntrainer::copy_fp32_s16,
  .copy_fp32_s8 = nntrainer::copy_fp32_s8,

  // Quantized GEMM — CPU fallback
  .gemm_q4_0_fp32 = nntrainer::gemm_q4_0<float>,
  .gemm_q4_K_fp32 = nntrainer::gemm_q4_K,
  .gemm_q6_K_fp32 = nntrainer::gemm_q6_K<float>,

  // Quantized weight packing — CPU fallback
  .unpack_q4_0 = nntrainer::unpack_q4_0,
  .unpack_q4_0x8_transpose16 = nntrainer::unpack_q4_0x8_transpose16,

  // GPU-accelerated batch ops — not available on NPU example
  .gemm_q4_0_batch_fp32 = nullptr,
  .gemm_q4_0_accel_fp32 = nullptr,
  .gemv_int4_batch_fp32 = nullptr,
  .gemm_int4_batch_fp32 = nullptr,
  .gemv_int4_accel_fp32 = nullptr,
  .sgemm_int4_accel_fp32 = nullptr,

  // FP16 — omitted for brevity (set all to nullptr or CPU fallback)
  // In a real backend, fill these in if your NPU supports FP16.
#ifdef ENABLE_FP16
  .sgemm_fp16 = nullptr,
  .sgemv_fp16 = nullptr,
  .sdot_fp16 = nullptr,
  .saxpy_fp16 = nullptr,
  .scopy_fp16 = nullptr,
  .scopy_fp32_to_fp16 = nullptr,
  .scopy_fp16_to_fp32 = nullptr,
  .sscal_fp16 = nullptr,
  .snrm2_fp16 = nullptr,
  .isamax_fp16 = nullptr,
  .ele_mul_fp16 = nullptr,
  .ele_add_fp16 = nullptr,
  .ele_sub_fp16 = nullptr,
  .ele_div_fp16 = nullptr,
  .swiglu_fp16 = nullptr,
  .max_val_fp16 = nullptr,
  .softmax_fp16 = nullptr,
  .is_valid_fp16 = nullptr,
  .inv_sqrt_inplace_fp16 = nullptr,
  .transpose_matrix_fp16 = nullptr,
  .scopy_int4_to_float16 = nullptr,
  .scopy_int8_to_float16_u = nullptr,
  .scopy_int8_to_float16_s = nullptr,
  .shgemm = nullptr,
  .shgemv = nullptr,
  .hsgemm = nullptr,
  .hsgemv = nullptr,
  .gemm_q4_0_fp16 = nullptr,
  .gemm_q6_K_fp16 = nullptr,
  .compute_rotary_embedding_value = nullptr,
#endif
};

nntrainer::ComputeOps *get_npu_ops() { return &npu_ops; }

// =========================================================================
// Part 2: Context Class Implementation
// =========================================================================

ExampleNpuContext::ExampleNpuContext()
  : Context(std::make_shared<nntrainer::ContextData>()) {}

void ExampleNpuContext::initialize() noexcept {
  try {
    // Step 1: Initialize your hardware
    // npu_driver_init();

    // Step 2: Set memory allocator and compute ops table
    if (auto cd = getContextData()) {
      cd->setMemAllocator(std::make_shared<nntrainer::MemAllocator>());
      // THIS IS THE KEY STEP — sets which ops tensor.dot() will call
      cd->setComputeOps(get_npu_ops());
    }

    // Step 4: Register NPU-specific layers (optional)
    add_default_object();

  } catch (std::exception &e) {
    fprintf(stderr, "ExampleNpuContext init failed: %s\n", e.what());
  }
}

void ExampleNpuContext::add_default_object() {
  // Register NPU-specific layer implementations here.
  // If your NPU reuses CPU layers (just accelerates tensor ops),
  // you can leave this empty.
  //
  // Example:
  // registerFactory(nntrainer::createLayer<NpuFullyConnectedLayer>,
  //                 "fully_connected", LayerType::LAYER_FC);
}

nntrainer::Context::PtrType<nntrainer::Layer>
ExampleNpuContext::createLayerObject(const std::string &type,
                                     const std::vector<std::string> &props) {
  // For this example, return nullptr to fall back to CPU layer creation.
  // In a real backend, create NPU-optimized layers here.
  return nullptr;
}

// =========================================================================
// Part 3: Plugin Entry Point (for .so dynamic loading)
// =========================================================================
//
// If your backend is built as a shared library plugin (.so), export these
// symbols so Engine can discover and load it at runtime:
//
//   engine.registerPluggableContext("path/to/libyour_backend.so");
//
// For static linking, register directly in engine.cpp instead.
//

} // namespace example_npu

// Uncomment these for .so plugin builds:
//
// extern "C" nntrainer::ContextPluggable ml_train_context_pluggable = {
//   []() -> nntrainer::Context * {
//     return &example_npu::ExampleNpuContext::Global();
//   },
//   [](nntrainer::Context *) {
//     // Singleton — no delete needed
//   }
// };

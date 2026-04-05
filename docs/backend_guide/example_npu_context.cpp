// SPDX-License-Identifier: Apache-2.0
/**
 * @file   example_npu_context.cpp
 * @brief  Example NPU backend — complete implementation.
 *
 * This shows the 4 essential parts of a new backend:
 *   Part 1: ComputeOps wrappers (vendor_backend:: namespace)
 *   Part 2: ComputeOps table (maps wrappers to ops table)
 *   Part 3: Context initialization (set ops + vendor data)
 *   Part 4: Plugin entry point (for .so dynamic loading)
 */

#include "example_npu_context.h"

#include <compute_ops.h>
#include <context_data.h>
#include <cpu_backend.h>
#include <mem_allocator.h>

#include <cstdio>

namespace example_npu {

// =========================================================================
// Part 1: ComputeOps wrappers
// =========================================================================
//
// Each wrapper adapts the vendor's implementation to the ComputeOps
// function pointer signature. For ops your NPU accelerates, call your
// vendor SDK. For others, fall back to CPU.
//
// This follows the SAME pattern as:
//   - arm_backend::  in arm_ops_table.cpp
//   - x86_backend::  in x86_ops_table.cpp
//   - cl_sgemm_fp32  in cl_compute_ops.cpp
//

namespace npu_backend {

/// NPU-accelerated SGEMM (replace with real NPU SDK call)
static void sgemm_fp32(const unsigned int TStorageOrder, bool TransA,
                        bool TransB, const unsigned int M, const unsigned int N,
                        const unsigned int K, const float alpha, const float *A,
                        const unsigned int lda, const float *B,
                        const unsigned int ldb, const float beta, float *C,
                        const unsigned int ldc) {
  // In production: npu_driver_sgemm(handle, TransA, TransB, M, N, K, ...);
  // For this example: fall back to CPU
  nntrainer::sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                   ldb, beta, C, ldc);
}

/// NPU-accelerated SGEMV
static void sgemv_fp32(const unsigned int TStorageOrder, bool TransA,
                        const unsigned int M, const unsigned int N,
                        const float alpha, const float *A,
                        const unsigned int lda, const float *X,
                        const unsigned int incX, const float beta, float *Y,
                        const unsigned int incY) {
  nntrainer::sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta,
                   Y, incY);
}

// For ops NOT accelerated by NPU, use CPU functions directly
static float sdot_fp32(const unsigned int N, const float *X,
                        const unsigned int iX, const float *Y,
                        const unsigned int iY) {
  return nntrainer::sdot(N, X, iX, Y, iY);
}
static void saxpy_fp32(const unsigned int N, const float a, const float *X,
                        const unsigned int iX, float *Y,
                        const unsigned int iY) {
  nntrainer::saxpy(N, a, X, iX, Y, iY);
}
static void scopy_fp32(const unsigned int N, const float *X,
                        const unsigned int iX, float *Y,
                        const unsigned int iY) {
  nntrainer::scopy(N, X, iX, Y, iY);
}
static void sscal_fp32(const unsigned int N, const float a, float *X,
                        const unsigned int iX) {
  nntrainer::sscal(N, a, X, iX);
}
static float snrm2_fp32(const unsigned int N, const float *X,
                         const unsigned int iX) {
  return nntrainer::snrm2(N, X, iX);
}
static unsigned int isamax_fp32(const unsigned int N, const float *X,
                                 const unsigned int iX) {
  return nntrainer::isamax(N, X, iX);
}
static void ele_mul_fp32(const unsigned int N, const float *X, const float *Y,
                          float *Z, float a, float b, unsigned int is,
                          unsigned int os) {
  nntrainer::ele_mul(N, X, Y, Z, a, b, is, os);
}
static void ele_add_fp32(const unsigned int N, const float *X, const float *Y,
                          float *Z, float a, float b, unsigned int is,
                          unsigned int os) {
  nntrainer::ele_add(N, X, Y, Z, a, b, is, os);
}
static void ele_sub_fp32(const unsigned int N, const float *X, const float *Y,
                          float *Z, float a, float b, unsigned int is,
                          unsigned int os) {
  nntrainer::ele_sub(N, X, Y, Z, a, b, is, os);
}
static void ele_div_fp32(const unsigned int N, const float *X, const float *Y,
                          float *Z, float a, float b, unsigned int is,
                          unsigned int os) {
  nntrainer::ele_div(N, X, Y, Z, a, b, is, os);
}
static void swiglu_fp32(const unsigned int N, float *X, float *Y, float *Z) {
  nntrainer::swiglu(N, X, Y, Z);
}
static void swiglu_alpha_fp32(const unsigned int N, float *X, float *Y,
                               float *Z, float alpha) {
  nntrainer::swiglu(N, X, Y, Z, alpha);
}
static void tanh_gelu_fp32(const unsigned int N, const float *X, float *Y) {
  nntrainer::tanh_gelu(N, X, Y);
}
static void gelu_v2_fp32(const unsigned int N, const float *X, float *Y) {
  nntrainer::gelu_v2(N, X, Y);
}
static void tanh_gelu_v2_fp32(const unsigned int N, const float *X, float *Y) {
  nntrainer::tanh_gelu_v2(N, X, Y);
}
static void tanh_gelu_mul_fp32(const unsigned int N, float *X, float *Y,
                                float *Z) {
  nntrainer::tanh_gelu_mul(N, X, Y, Z);
}
static void tanh_gelu_v2_mul_fp32(const unsigned int N, float *X, float *Y,
                                   float *Z) {
  nntrainer::tanh_gelu_v2_mul(N, X, Y, Z);
}
static float max_val_fp32(const unsigned int N, float *X) {
  return nntrainer::max_val(N, X);
}
static void softmax_fp32(const unsigned int N, float *X, float *Y) {
  nntrainer::softmax(N, X, Y);
}
static bool is_valid_fp32(const unsigned int N, const float *X) {
  return nntrainer::is_valid(N, X);
}
static void transpose_matrix_fp32(const unsigned int M, const unsigned int N,
                                   const float *s, unsigned int lds, float *d,
                                   unsigned int ldd) {
  nntrainer::transpose_matrix(M, N, s, lds, d, ldd);
}
static void gemm_q4_0_fp32(const unsigned int M, const unsigned int N,
                             const unsigned int K, const float *A,
                             const unsigned int lda, const void *B,
                             const unsigned int ldb, float *C,
                             const unsigned int ldc) {
  nntrainer::gemm_q4_0(M, N, K, A, lda, B, ldb, C, ldc);
}
static void gemm_q4_K_fp32(const unsigned int M, const unsigned int N,
                             const unsigned int K, const float *A,
                             const unsigned int lda, const void *B,
                             const unsigned int ldb, float *C,
                             const unsigned int ldc) {
  nntrainer::gemm_q4_K(M, N, K, A, lda, B, ldb, C, ldc);
}
static void gemm_q6_K_fp32(const unsigned int M, const unsigned int N,
                             const unsigned int K, const float *A,
                             const unsigned int lda, const void *B,
                             const unsigned int ldb, float *C,
                             const unsigned int ldc) {
  nntrainer::gemm_q6_K(M, N, K, A, lda, B, ldb, C, ldc);
}
static size_t quantize_q4_0_impl(const float *src, void *dst, int64_t nrow,
                                  int64_t n_per_row, const float *qw) {
  return nntrainer::quantize_q4_0(src, dst, nrow, n_per_row, qw);
}
static void dequantize_row_q4_0_impl(const void *x, float *y, int64_t k) {
  nntrainer::dequantize_row_q4_0(x, y, k);
}
static void repack_q4_0_impl(void *dst, void *src, size_t ds,
                              const unsigned int M, const unsigned int N) {
  nntrainer::repack_q4_0(dst, src, ds, M, N);
}
static void clamp_fp32(const float *in, float *out, size_t len, float lb,
                        float ub) {
  nntrainer::clamp(in, out, len, lb, ub);
}
static void scopy_int8_to_fp32_u(const unsigned int N, const uint8_t *X,
                                  const unsigned int iX, float *Y,
                                  const unsigned int iY) {
  nntrainer::scopy_int8_to_float32(N, X, iX, Y, iY);
}
static void scopy_int8_to_fp32_s(const unsigned int N, const int8_t *X,
                                  const unsigned int iX, float *Y,
                                  const unsigned int iY) {
  nntrainer::scopy_int8_to_float32(N, X, iX, Y, iY);
}

} // namespace npu_backend

// =========================================================================
// Part 2: ComputeOps table
// =========================================================================

static nntrainer::ComputeOps npu_ops = {
  // ── NPU-accelerated (or CPU fallback) ──
  .sgemm_fp32 = npu_backend::sgemm_fp32,
  .sgemv_fp32 = npu_backend::sgemv_fp32,
  .sdot_fp32 = npu_backend::sdot_fp32,
  .saxpy_fp32 = npu_backend::saxpy_fp32,
  .scopy_fp32 = npu_backend::scopy_fp32,
  .sscal_fp32 = npu_backend::sscal_fp32,
  .snrm2_fp32 = npu_backend::snrm2_fp32,
  .isamax_fp32 = npu_backend::isamax_fp32,
  .ele_mul_fp32 = npu_backend::ele_mul_fp32,
  .ele_add_fp32 = npu_backend::ele_add_fp32,
  .ele_sub_fp32 = npu_backend::ele_sub_fp32,
  .ele_div_fp32 = npu_backend::ele_div_fp32,
  .swiglu_fp32 = npu_backend::swiglu_fp32,
  .swiglu_alpha_fp32 = npu_backend::swiglu_alpha_fp32,
  .tanh_gelu_fp32 = npu_backend::tanh_gelu_fp32,
  .gelu_v2_fp32 = npu_backend::gelu_v2_fp32,
  .tanh_gelu_v2_fp32 = npu_backend::tanh_gelu_v2_fp32,
  .tanh_gelu_mul_fp32 = npu_backend::tanh_gelu_mul_fp32,
  .tanh_gelu_v2_mul_fp32 = npu_backend::tanh_gelu_v2_mul_fp32,
  .max_val_fp32 = npu_backend::max_val_fp32,
  .softmax_fp32 = npu_backend::softmax_fp32,
  .is_valid_fp32 = npu_backend::is_valid_fp32,
  .transpose_matrix_fp32 = npu_backend::transpose_matrix_fp32,
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
  .gemm_q4_0_fp32 = npu_backend::gemm_q4_0_fp32,
  .gemm_q4_K_fp32 = npu_backend::gemm_q4_K_fp32,
  .gemm_q6_K_fp32 = npu_backend::gemm_q6_K_fp32,
  .unpack_q4_0 = nntrainer::unpack_q4_0,
  .unpack_q4_0x8_transpose16 = nntrainer::unpack_q4_0x8_transpose16,
  .quantize_q4_0 = npu_backend::quantize_q4_0_impl,
  .dequantize_row_q4_0 = npu_backend::dequantize_row_q4_0_impl,
  .repack_q4_0 = npu_backend::repack_q4_0_impl,
  .clamp_fp32 = npu_backend::clamp_fp32,
  .scopy_int8_to_fp32_u = npu_backend::scopy_int8_to_fp32_u,
  .scopy_int8_to_fp32_s = npu_backend::scopy_int8_to_fp32_s,
  // GPU-accelerated batch ops — not available on this NPU
  .gemm_q4_0_batch_fp32 = nullptr,
  .gemm_q4_0_accel_fp32 = nullptr,
  .gemv_int4_batch_fp32 = nullptr,
  .gemm_int4_batch_fp32 = nullptr,
  .gemv_int4_accel_fp32 = nullptr,
  .sgemm_int4_accel_fp32 = nullptr,
#ifdef ENABLE_FP16
  // FP16 — set to nullptr (fill in if your NPU supports FP16)
  .sgemm_fp16 = nullptr, .sgemv_fp16 = nullptr, .sdot_fp16 = nullptr,
  .saxpy_fp16 = nullptr, .scopy_fp16 = nullptr,
  .scopy_fp32_to_fp16 = nullptr, .scopy_fp16_to_fp32 = nullptr,
  .sscal_fp16 = nullptr, .snrm2_fp16 = nullptr, .isamax_fp16 = nullptr,
  .ele_mul_fp16 = nullptr, .ele_add_fp16 = nullptr,
  .ele_sub_fp16 = nullptr, .ele_div_fp16 = nullptr,
  .swiglu_fp16 = nullptr, .max_val_fp16 = nullptr,
  .softmax_fp16 = nullptr, .is_valid_fp16 = nullptr,
  .inv_sqrt_inplace_fp16 = nullptr, .transpose_matrix_fp16 = nullptr,
  .scopy_int4_to_float16 = nullptr,
  .scopy_int8_to_float16_u = nullptr, .scopy_int8_to_float16_s = nullptr,
  .shgemm = nullptr, .shgemv = nullptr,
  .hsgemm = nullptr, .hsgemv = nullptr,
  .gemm_q4_0_fp16 = nullptr, .gemm_q6_K_fp16 = nullptr,
  .compute_rotary_embedding_value = nullptr,
#endif
};

// =========================================================================
// Part 3: Context initialization
// =========================================================================

void ExampleNpuContext::initialize() noexcept {
  try {
    // 1. Initialize CPU backend (for fallback ops)
    nntrainer::init_backend();

    // 2. Set ComputeOps on our ContextData
    //    ALL contexts must set this — even NPU (for tensor ops outside
    //    the NPU graph, like preprocessing and postprocessing)
    if (auto cd = getContextData())
      cd->setComputeOps(&npu_ops);

    // 3. Set memory allocator
    if (auto cd = getContextData())
      cd->setMemAllocator(std::make_shared<nntrainer::MemAllocator>());

    // 4. Initialize NPU hardware
    auto *npu_data = getContextData()->as<NpuBackendData>();
    if (npu_data) {
      npu_data->getNpuData()->initialize("default");
    }

    // 5. Register NPU-specific layers (optional)
    // registerFactory(createLayer<NpuGraphLayer>, "npu_graph", ...);

  } catch (std::exception &e) {
    fprintf(stderr, "ExampleNpuContext init failed: %s\n", e.what());
  }
}

nntrainer::Context::PtrType<nntrainer::Layer>
ExampleNpuContext::createLayerObject(const std::string &type,
                                     const std::vector<std::string> &props) {
  return nullptr; // Fall back to CPU layer creation
}

// =========================================================================
// Part 4: Plugin entry point (for .so dynamic loading)
// =========================================================================
//
// Build as shared library and load via:
//   engine.registerContext("path/to/libexample_npu.so");
//
// Or register statically in engine.cpp:
//   auto &npu = example_npu::ExampleNpuContext::Global();
//   registerContext("example_npu", &npu);

} // namespace example_npu

// Uncomment for .so plugin:
// extern "C" nntrainer::ContextPluggable ml_train_context_pluggable = {
//   []() -> nntrainer::Context * {
//     return &example_npu::ExampleNpuContext::Global();
//   },
//   [](nntrainer::Context *) { /* Singleton, no delete */ }
// };

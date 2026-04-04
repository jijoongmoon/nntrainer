// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 NNtrainer contributors
 *
 * @file   compute_ops.cpp
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 * @brief  Global ComputeOps pointer + shared CPU ops table.
 *
 * The CPU ops table is shared across ARM, x86, and fallback backends.
 * All three backends define the same global functions (sgemm, sgemv, etc.)
 * via cpu_backend.h, which selects the architecture-specific header at
 * compile time. So the same function pointers resolve to different
 * implementations depending on which .cpp files are linked.
 */

#include <compute_ops.h>
#include <cpu_backend.h>

namespace nntrainer {

ComputeOps *g_compute_ops = nullptr;

void ensureComputeOps() {
  if (g_compute_ops == nullptr) {
    init_backend();
  }
}

/**
 * @brief Shared CPU ops table for all CPU architectures.
 *
 * The function pointers (nntrainer::sgemm, etc.) resolve to ARM NEON,
 * x86 AVX, or fallback implementations at link time based on which
 * architecture-specific .cpp is compiled.
 */
static ComputeOps cpu_ops = {
  // FP32 BLAS
  .sgemm_fp32 = nntrainer::sgemm,
  .sgemv_fp32 = nntrainer::sgemv,
  .sdot_fp32 = nntrainer::sdot,
  .saxpy_fp32 = nntrainer::saxpy,
  .scopy_fp32 =
    static_cast<void (*)(const unsigned int, const float *, const unsigned int,
                         float *, const unsigned int)>(nntrainer::scopy),
  .sscal_fp32 = nntrainer::sscal,
  .snrm2_fp32 = nntrainer::snrm2,
  .isamax_fp32 = nntrainer::isamax,
  // FP32 element-wise
  .ele_mul_fp32 = nntrainer::ele_mul,
  .ele_add_fp32 = nntrainer::ele_add,
  .ele_sub_fp32 = nntrainer::ele_sub,
  .ele_div_fp32 = nntrainer::ele_div,
  // FP32 activation / special
  .swiglu_fp32 =
    static_cast<void (*)(const unsigned int, float *, float *, float *)>(
      nntrainer::swiglu),
  .swiglu_alpha_fp32 = static_cast<void (*)(const unsigned int, float *, float *,
                                            float *, float)>(nntrainer::swiglu),
  .tanh_gelu_fp32 = nntrainer::tanh_gelu,
  .tanh_gelu_v2_fp32 = nntrainer::tanh_gelu_v2,
  .tanh_gelu_mul_fp32 = nntrainer::tanh_gelu_mul,
  .tanh_gelu_v2_mul_fp32 = nntrainer::tanh_gelu_v2_mul,
  .max_val_fp32 = nntrainer::max_val,
  .softmax_fp32 = nntrainer::softmax,
  .is_valid_fp32 = nntrainer::is_valid,
  // FP32 matrix ops
  .transpose_matrix_fp32 = nntrainer::transpose_matrix,
  // FP32 data conversion
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
  // Quantized GEMM
  .gemm_q4_0_fp32 = nntrainer::gemm_q4_0<float>,
  .gemm_q4_K_fp32 = nntrainer::gemm_q4_K,
  .gemm_q6_K_fp32 = nntrainer::gemm_q6_K<float>,
  // Quantized weight packing
  .unpack_q4_0 = nntrainer::unpack_q4_0,
  .unpack_q4_0x8_transpose16 = nntrainer::unpack_q4_0x8_transpose16,
  // GPU-accelerated quantized ops (not available on CPU)
  .gemm_q4_0_batch_fp32 = nullptr,
  .gemm_q4_0_accel_fp32 = nullptr,
  .gemv_int4_batch_fp32 = nullptr,
  .gemm_int4_batch_fp32 = nullptr,
  .gemv_int4_accel_fp32 = nullptr,
  .sgemm_int4_accel_fp32 = nullptr,
#ifdef ENABLE_FP16
  // FP16 BLAS
  .sgemm_fp16 = nntrainer::sgemm,
  .sgemv_fp16 = nntrainer::sgemv,
  .sdot_fp16 = nntrainer::sdot,
  .saxpy_fp16 = nntrainer::saxpy,
  .scopy_fp16 =
    static_cast<void (*)(const unsigned int, const _FP16 *, const unsigned int,
                         _FP16 *, const unsigned int)>(nntrainer::scopy),
  .scopy_fp32_to_fp16 =
    static_cast<void (*)(const unsigned int, const float *, const unsigned int,
                         _FP16 *, const unsigned int)>(nntrainer::scopy),
  .scopy_fp16_to_fp32 =
    static_cast<void (*)(const unsigned int, const _FP16 *, const unsigned int,
                         float *, const unsigned int)>(nntrainer::scopy),
  .sscal_fp16 = nntrainer::sscal,
  .snrm2_fp16 = nntrainer::snrm2,
  .isamax_fp16 = nntrainer::isamax,
  // FP16 element-wise
  .ele_mul_fp16 = nntrainer::ele_mul,
  .ele_add_fp16 = nntrainer::ele_add,
  .ele_sub_fp16 = nntrainer::ele_sub,
  .ele_div_fp16 = nntrainer::ele_div,
  // FP16 activation / special
  .swiglu_fp16 = nntrainer::swiglu,
  .max_val_fp16 = nntrainer::max_val,
  .softmax_fp16 = nntrainer::softmax,
  .is_valid_fp16 = nntrainer::is_valid,
  .inv_sqrt_inplace_fp16 = nntrainer::inv_sqrt_inplace,
  // FP16 matrix ops
  .transpose_matrix_fp16 = nntrainer::transpose_matrix,
  // FP16 data conversion
  .scopy_int4_to_float16 = nntrainer::scopy_int4_to_float16,
  .scopy_int8_to_float16_u =
    static_cast<void (*)(const unsigned int, const uint8_t *,
                         const unsigned int, _FP16 *, const unsigned int)>(
      nntrainer::scopy_int8_to_float16),
  .scopy_int8_to_float16_s =
    static_cast<void (*)(const unsigned int, const int8_t *,
                         const unsigned int, _FP16 *, const unsigned int)>(
      nntrainer::scopy_int8_to_float16),
  // Mixed precision BLAS
  .shgemm = nntrainer::shgemm,
  .shgemv = nntrainer::shgemv,
  .hsgemm = nntrainer::hsgemm,
  .hsgemv = nntrainer::hsgemv,
  // Quantized GEMM (FP16)
  .gemm_q4_0_fp16 = nntrainer::gemm_q4_0<_FP16>,
  .gemm_q6_K_fp16 = nntrainer::gemm_q6_K<_FP16>,
  // Rotary embedding
  .compute_rotary_embedding_value = nntrainer::compute_rotary_embedding_value,
#endif
};

ComputeOps *get_cpu_ops() { return &cpu_ops; }

} // namespace nntrainer

// SPDX-License-Identifier: Apache-2.0
/**
 * @file   arm_ops_table.cpp
 * @brief  ARM-specific ComputeOps table. References functions defined in
 *         arm_compute_backend.cpp (same translation unit via arm_compute_backend.h).
 *         Each function pointer resolves to the ARM implementation.
 */
#include <arm_compute_backend.h>
#include <compute_ops.h>
#include <cpu_backend.h>

namespace nntrainer {

// Forward declarations - these are defined in arm_compute_backend.cpp
// and arm_compute_backend_fp16.cpp within namespace nntrainer.
// By referencing them here, the linker binds to ARM-specific implementations.

static ComputeOps arm_ops = {
  // FP32 BLAS
  .sgemm_fp32 = sgemm,
  .sgemv_fp32 = sgemv,
  .sdot_fp32 = sdot,
  .saxpy_fp32 = saxpy,
  .scopy_fp32 =
    static_cast<void (*)(const unsigned int, const float *, const unsigned int,
                         float *, const unsigned int)>(scopy),
  .sscal_fp32 = sscal,
  .snrm2_fp32 = snrm2,
  .isamax_fp32 = isamax,
  // FP32 element-wise
  .ele_mul_fp32 = ele_mul,
  .ele_add_fp32 = ele_add,
  .ele_sub_fp32 = ele_sub,
  .ele_div_fp32 = ele_div,
  // FP32 activation / special
  .swiglu_fp32 =
    static_cast<void (*)(const unsigned int, float *, float *, float *)>(
      swiglu),
  .swiglu_alpha_fp32 = static_cast<void (*)(const unsigned int, float *, float *,
                                            float *, float)>(swiglu),
  .tanh_gelu_fp32 = tanh_gelu,
  .tanh_gelu_v2_fp32 = tanh_gelu_v2,
  .tanh_gelu_mul_fp32 = tanh_gelu_mul,
  .tanh_gelu_v2_mul_fp32 = tanh_gelu_v2_mul,
  .max_val_fp32 = max_val,
  .softmax_fp32 = softmax,
  .is_valid_fp32 = is_valid,
  // FP32 matrix ops
  .transpose_matrix_fp32 = transpose_matrix,
  // FP32 data conversion
  .scopy_u8 = scopy,
  .scopy_s8 = scopy,
  .scopy_int4_to_float32 = scopy_int4_to_float32,
  .copy_s16_fp32 = copy_s16_fp32,
  .copy_u16_fp32 = copy_u16_fp32,
  .copy_fp32_u32 = copy_fp32_u32,
  .copy_fp32_u16 = copy_fp32_u16,
  .copy_fp32_u8 = copy_fp32_u8,
  .copy_fp32_s16 = copy_fp32_s16,
  .copy_fp32_s8 = copy_fp32_s8,
  // Quantized GEMM
  .gemm_q4_0_fp32 = gemm_q4_0<float>,
  .gemm_q4_K_fp32 = gemm_q4_K,
  .gemm_q6_K_fp32 = gemm_q6_K<float>,
  // Quantized weight packing
  .unpack_q4_0 = unpack_q4_0,
  .unpack_q4_0x8_transpose16 = unpack_q4_0x8_transpose16,
  // GPU-accelerated quantized ops (not available on CPU)
  .gemm_q4_0_batch_fp32 = nullptr,
  .gemm_q4_0_accel_fp32 = nullptr,
  .gemv_int4_batch_fp32 = nullptr,
  .gemm_int4_batch_fp32 = nullptr,
  .gemv_int4_accel_fp32 = nullptr,
  .sgemm_int4_accel_fp32 = nullptr,
#ifdef ENABLE_FP16
  // FP16 BLAS
  .sgemm_fp16 = sgemm,
  .sgemv_fp16 = sgemv,
  .sdot_fp16 = sdot,
  .saxpy_fp16 = saxpy,
  .scopy_fp16 =
    static_cast<void (*)(const unsigned int, const _FP16 *, const unsigned int,
                         _FP16 *, const unsigned int)>(scopy),
  .scopy_fp32_to_fp16 =
    static_cast<void (*)(const unsigned int, const float *, const unsigned int,
                         _FP16 *, const unsigned int)>(scopy),
  .scopy_fp16_to_fp32 =
    static_cast<void (*)(const unsigned int, const _FP16 *, const unsigned int,
                         float *, const unsigned int)>(scopy),
  .sscal_fp16 = sscal,
  .snrm2_fp16 = snrm2,
  .isamax_fp16 = isamax,
  // FP16 element-wise
  .ele_mul_fp16 = ele_mul,
  .ele_add_fp16 = ele_add,
  .ele_sub_fp16 = ele_sub,
  .ele_div_fp16 = ele_div,
  // FP16 activation / special
  .swiglu_fp16 = swiglu,
  .max_val_fp16 = max_val,
  .softmax_fp16 = softmax,
  .is_valid_fp16 = is_valid,
  .inv_sqrt_inplace_fp16 = inv_sqrt_inplace,
  // FP16 matrix ops
  .transpose_matrix_fp16 = transpose_matrix,
  // FP16 data conversion
  .scopy_int4_to_float16 = scopy_int4_to_float16,
  .scopy_int8_to_float16_u =
    static_cast<void (*)(const unsigned int, const uint8_t *,
                         const unsigned int, _FP16 *, const unsigned int)>(
      scopy_int8_to_float16),
  .scopy_int8_to_float16_s =
    static_cast<void (*)(const unsigned int, const int8_t *,
                         const unsigned int, _FP16 *, const unsigned int)>(
      scopy_int8_to_float16),
  // Mixed precision BLAS
  .shgemm = shgemm,
  .shgemv = shgemv,
  .hsgemm = hsgemm,
  .hsgemv = hsgemv,
  // Quantized GEMM (FP16)
  .gemm_q4_0_fp16 = gemm_q4_0<_FP16>,
  .gemm_q6_K_fp16 = gemm_q6_K<_FP16>,
  // Rotary embedding
  .compute_rotary_embedding_value = compute_rotary_embedding_value,
#endif
};

ComputeOps *get_arm_ops() { return &arm_ops; }

} // namespace nntrainer

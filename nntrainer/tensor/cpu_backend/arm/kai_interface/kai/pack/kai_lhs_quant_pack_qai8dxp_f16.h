/**
 * @file kai_lhs_quant_pack_qai8dxp_f16.h
 * @date   2026-05-24
 * @see    https://github.com/ARM-software/kleidiai
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  fp16-input variant of kai_lhs_quant_pack_qai8dxp_f32.
 *
 * Same packed-output byte layout (int8 qai8dxp + per-row int32 zero point
 * + fp32 scale) as the upstream KAI fp32 packer. The only difference is
 * the LHS source dtype: this entry point reads _FP16 (Arm __fp16 /
 * _Float16) and lifts to fp32 internally via NEON vcvt_f32_f16 so the
 * quantization arithmetic keeps fp32 precision.
 *
 * Output bytes are bit-identical to running
 * kai_run_lhs_quant_pack_qai8dxp_f32 on a fp32-promoted copy of the same
 * input — i.e. the existing qai8dxp_qsi4cxp matmul micro-kernels can
 * consume the result without modification.
 */
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Gets the m step value. See kai_get_m_step_lhs_quant_pack_qai8dxp_f32.
size_t kai_get_m_step_lhs_quant_pack_qai8dxp_f16(size_t mr);

/// Gets the offset in bytes for the (unpacked) LHS matrix. Same semantics
/// as the fp32 variant but stride is measured in _FP16 elements when the
/// caller passes lhs_stride = k * sizeof(_FP16).
size_t kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16(size_t m_idx,
                                                     size_t lhs_stride);

/// Gets the offset in bytes for the packed LHS matrix. Layout matches the
/// fp32 variant byte-for-byte.
size_t kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16(size_t m_idx,
                                                            size_t k, size_t mr,
                                                            size_t kr,
                                                            size_t sr);

/// Gets the size in bytes for the quantized and packed LHS matrix. Identical
/// to the fp32 variant — output bytes do not depend on input dtype.
size_t kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16(size_t m, size_t k,
                                                          size_t mr, size_t kr,
                                                          size_t sr);

/// Run the micro-kernel to quantize and pack the LHS matrix from fp16.
///
/// @param[in]  m           Number of LHS rows.
/// @param[in]  k           Common dimension K. Must be a multiple of 8.
/// @param[in]  mr/kr/sr    Matches the matmul micro-kernel's getters.
/// @param[in]  m_idx_start Starting row index.
/// @param[in]  lhs         LHS data, fp16 elements. Caller passes
///                         lhs_stride in bytes (typically k * 2).
/// @param[in]  lhs_stride  Row stride in bytes between LHS rows.
/// @param[out] lhs_packed  Output buffer. Same byte layout as the fp32
///                         packer's output; size given by
///                         kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16.
void kai_run_lhs_quant_pack_qai8dxp_f16(size_t m, size_t k, size_t mr,
                                        size_t kr, size_t sr,
                                        size_t m_idx_start, const void *lhs,
                                        size_t lhs_stride, void *lhs_packed);

#ifdef __cplusplus
}
#endif

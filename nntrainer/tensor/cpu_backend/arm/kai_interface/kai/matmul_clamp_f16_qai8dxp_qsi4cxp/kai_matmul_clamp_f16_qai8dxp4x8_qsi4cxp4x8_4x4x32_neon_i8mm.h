/**
 * @file kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h
 * @date   2026-05-24
 * @see    https://github.com/ARM-software/kleidiai
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  fp16-dst variant of
 *         kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.
 *
 * Same matmul micro-kernel (4x4 output tile, 4x8 LHS pack, 4x8 RHS pack,
 * 32-K accumulation per inner loop, i8mm smmla) but the epilogue casts
 * the fp32 accumulator down to fp16 with fcvtn and stores via 8-byte /
 * 4-byte / 2-byte half-precision stores instead of the fp32 sizes.
 *
 * The packed LHS / RHS layout is unchanged — this kernel consumes the
 * exact same bytes that the qai8dxp f32 / f16 LHS packer and the
 * qsi4cxp_qs4cxs1s0 RHS packer produce. Output dst must be a tightly
 * packed fp16 buffer with dst_stride_col == sizeof(_FP16) == 2 and
 * dst_stride_row given in bytes (typically n * 2).
 *
 * The clamp arithmetic is still performed in fp32 (before the fcvtn) so
 * scalar_min / scalar_max are fp32; callers should ensure
 * |scalar_max| <= 65504 so the fcvtn does not saturate to +Inf.
 */
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t
kai_get_m_step_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(void);

size_t
kai_get_n_step_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(void);

size_t kai_get_mr_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(void);

size_t kai_get_nr_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(void);

size_t kai_get_kr_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(void);

size_t kai_get_sr_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(void);

size_t
kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
  size_t m_idx, size_t k);

size_t
kai_get_rhs_packed_offset_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
  size_t n_idx, size_t k);

size_t
kai_get_dst_offset_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
  size_t m_idx, size_t n_idx, size_t dst_stride);

size_t kai_get_dst_size_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
  size_t m, size_t n);

/// Runs the int8 x int4 matmul micro-kernel followed by clamp and fp16
/// store. The clamp is applied in fp32 just before the fcvtn — pass
/// scalar_min / scalar_max in fp32 with magnitude <= 65504 so the
/// fp32 -> fp16 cast does not saturate.
///
/// @param[in]  dst            Tightly packed fp16 destination matrix.
/// @param[in]  dst_stride_row Stride in bytes between two rows of dst.
/// @param[in]  dst_stride_col Stride in bytes between two columns of
///                            dst. For this kernel it must equal
///                            sizeof(_FP16) (= 2).
void kai_run_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
  size_t m, size_t n, size_t k, const void *lhs_packed, const void *rhs_packed,
  void *dst, size_t dst_stride_row, size_t dst_stride_col, float scalar_min,
  float scalar_max);

#ifdef __cplusplus
}
#endif

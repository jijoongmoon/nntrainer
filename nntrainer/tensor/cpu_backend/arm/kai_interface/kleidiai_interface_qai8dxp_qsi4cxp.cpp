// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Arm Limited and/or its affiliates
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   kleidiai_interface_qai8dxp_qsi4cxp.cpp
 * @date   15 September 2025
 * @see    https://github.com/ARM-software/kleidiai
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong
 *
 * @brief  Modified computational backend components of
 * kleidiai. Portions of this file are derived from Arm
 * Limited code licensed under the Apache License, Version 2.0, with
 * modifications
 *
 * @note   Licensed under the Apache License, Version 2.0 (the "License");
 *         you may not use this file except in compliance with the License.
 *         You may obtain a copy of the License at
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * @modifications
 *   - [2025-09-15] Integrated and adapted Arm-provided code into
 *     nntrainer CPU backend
 *
 * @bug    No known bugs except for NYI items
 */
//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates
// <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <kleidiai_interface.h>
#include <thread_manager.h>

#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h"

// Include micro-kernel variants
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.h"

#if defined(__ARM_FEATURE_MATMUL_INT8)
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h"
// fp16-dst i8mm matmul fork (ours, additive) — i8mm-gated like the entries above
#include "kai/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h"
#endif

// Include packinng kernels
// fp16 LHS packer (ours, additive) — pure NEON, no i8mm dependency
#include "kai/pack/kai_lhs_quant_pack_qai8dxp_f16.h"
#include "kai/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/pack/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.h"
#include "kai/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"

#define INT4_MIN (-8)
#define INT4_MAX (7)

namespace {

// Micro-kernel interface
/**
 * @brief kai_matmul_ukernel_f32_qa8dxp_qs4cxp
 */
struct kai_matmul_ukernel_f32_qa8dxp_qs4cxp {
  kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel ukernel;
  std::string name = {};
};

kai_matmul_ukernel_f32_qa8dxp_qs4cxp ukernel_variants[] = {
  {kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
   "matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   "matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   "matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
   "matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod"},
#if defined(__ARM_FEATURE_MATMUL_INT8)
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   "matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   "matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm"},
#endif

};

const size_t num_ukernel_variants =
  sizeof(ukernel_variants) / sizeof(ukernel_variants[0]);
} // namespace

namespace nntrainer {
std::string __kai_get_num_ukernel_name_qai8dxp_qsi4cxp(size_t idx_variant) {
  return ukernel_variants[idx_variant].name;
}

size_t __kai_get_num_ukernel_variants_qai8dxp_qsi4cxp() {
  return num_ukernel_variants;
}

size_t __kai_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(size_t n, size_t k,
                                                   size_t idx_variant,
                                                   bool is_nxk) {
  const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
  const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

  if (is_nxk) {
    return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(n, k, nr, kr,
                                                                  sr);
  } else {
    return kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(n, k, nr, kr,
                                                                  sr);
  }
}

void __kai_rhs_pack_qsi4cxp_qs4cxs1s0(size_t n, size_t k,
                                      void *rhs_packed_mtx_qs4cx,
                                      void *rhs_native_mtx_qs4cx,
                                      void *rhs_scales_f32, size_t idx_variant,
                                      bool is_nxk) {
  const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
  const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

  if (is_nxk) {
    struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params nxk_params;
    nxk_params.lhs_zero_point = 1;
    nxk_params.rhs_zero_point = 8;
    // RHS packing
    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
      1, n, k, nr, kr, sr,                     // Packing arguments
      (const uint8_t *)(rhs_native_mtx_qs4cx), // RHS
      NULL,                                    // Bias
      (const float *)(rhs_scales_f32),         // Scale
      rhs_packed_mtx_qs4cx,                    // RHS packed
      0, &nxk_params);
  } else {
    struct kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params kxn_params;
    kxn_params.lhs_zero_point = 1;
    kxn_params.rhs_zero_point = 8;
    // RHS packing
    kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(
      1, n, k, nr, kr, sr,                     // Packing arguments
      (const uint8_t *)(rhs_native_mtx_qs4cx), // RHS
      NULL,                                    // Bias
      (const float *)(rhs_scales_f32),         // Scale
      rhs_packed_mtx_qs4cx,                    // RHS packed
      0, &kxn_params);
  }
}

void __kai_gemm_qai8dxp_qsi4cxp_rhs_unpacked(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs4cx, void *rhs_scales_f32, float *dst_act_mtx_f32,
  size_t idx_variant, bool is_nxk, float lower_bound, float upper_bound) {
  const size_t mr = ukernel_variants[idx_variant].ukernel.get_mr();
  const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
  const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

  // Get the size in bytes for the packed matrices
  const size_t lhs_packed_size =
    kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr);
  const size_t rhs_packed_size =
    __kai_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(n, k, idx_variant, is_nxk);

  // Allocate the matrices
  uint8_t *lhs_packed_mtx_qa8dx = new uint8_t[lhs_packed_size];
  uint8_t *rhs_packed_mtx_qs4cx = new uint8_t[rhs_packed_size];

  // RHS packing
  __kai_rhs_pack_qsi4cxp_qs4cxs1s0(n, k, rhs_packed_mtx_qs4cx,
                                   rhs_native_mtx_qs4cx, rhs_scales_f32,
                                   idx_variant, is_nxk);

  // call packed gemm
  __kai_gemm_qai8dxp_qsi4cxp(m, n, k, lhs_native_mtx_f32, rhs_packed_mtx_qs4cx,
                             dst_act_mtx_f32, idx_variant);

  delete[] lhs_packed_mtx_qa8dx;
  delete[] rhs_packed_mtx_qs4cx;
}

void __kai_gemm_qai8dxp_qsi4cxp(size_t m, size_t n, size_t k,
                                void *lhs_native_mtx_f32,
                                void *rhs_packed_mtx_qs4cx,
                                float *dst_act_mtx_f32, size_t idx_variant,
                                float lower_bound, float upper_bound) {
  const size_t mr = ukernel_variants[idx_variant].ukernel.get_mr();
  const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
  const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

  // LHS packing
  const size_t lhs_packed_size =
    kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr);
  uint8_t *lhs_packed_mtx_qa8dx = new uint8_t[lhs_packed_size];

  size_t m_step = ukernel_variants[idx_variant].ukernel.get_m_step();
  size_t m_loop = (m + m_step - 1) / m_step;

  size_t n_step = ukernel_variants[idx_variant].ukernel.get_n_step();
  size_t n_loop = (n + n_step - 1) / n_step;

  auto &tm = nntrainer::ThreadManager::Global();

  /// @todo find better heuristic
  if (m_step > n_step) {
    // parallelize over m
    tm.parallel_for(0, m_loop, [&](size_t i) {
      size_t m_start = i * m_step;
      size_t m_to_process = std::min(m_step, m - m_start);

      size_t lhs_offset = m_start * k;
      float *lhs_ptr = (float *)lhs_native_mtx_f32 + lhs_offset;

      const size_t lhs_packed_offset =
        ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, k);
      uint8_t *lhs_packed_ptr = lhs_packed_mtx_qa8dx + lhs_packed_offset;

      // LHS packing
      kai_run_lhs_quant_pack_qai8dxp_f32(m_to_process, k,   // Dimensions
                                         mr, kr, sr, 0,     // Packing arguments
                                         lhs_ptr,           // LHS
                                         k * sizeof(float), // LHS stride
                                         lhs_packed_ptr);   // LHS packed

      const size_t rhs_packed_offset =
        ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, k);
      const void *rhs_packed_ptr =
        (const void *)((const char *)rhs_packed_mtx_qs4cx + rhs_packed_offset);

      const size_t dst_stride = n * sizeof(float);
      const size_t dst_offset =
        ukernel_variants[idx_variant].ukernel.get_dst_offset(m_start, 0,
                                                             dst_stride);
      float *dst_ptr = (float *)((uint8_t *)dst_act_mtx_f32 + dst_offset);

      ukernel_variants[idx_variant].ukernel.run_matmul(
        m_to_process, n, k,      // Dimensions
        lhs_packed_ptr,          // LHS packed
        rhs_packed_ptr,          // RHS packed
        dst_ptr,                 // DST
        dst_stride,              // DST stride (row)
        sizeof(float),           // DST stride (col)
        lower_bound, upper_bound // Min and max for the clamp operation
      );
    });
  } else {
    // parallelize over m
    tm.parallel_for(0, m_loop, [&](size_t i) {
      size_t m_start = i * m_step;
      size_t m_to_process = std::min(m_step, m - m_start);

      size_t lhs_offset = m_start * k;
      float *lhs_ptr = (float *)lhs_native_mtx_f32 + lhs_offset;

      const size_t lhs_packed_offset =
        ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, k);
      uint8_t *lhs_packed_ptr = lhs_packed_mtx_qa8dx + lhs_packed_offset;

      // LHS packing
      kai_run_lhs_quant_pack_qai8dxp_f32(m_to_process, k,   // Dimensions
                                         mr, kr, sr, 0,     // Packing arguments
                                         lhs_ptr,           // LHS
                                         k * sizeof(float), // LHS stride
                                         lhs_packed_ptr);   // LHS packed
    });

    // parallelize over n
    tm.parallel_for(0, n_loop, [&](size_t i) {
      size_t n_start = i * n_step;
      size_t n_to_process = std::min(n_step, n - n_start);

      uint8_t *lhs_packed_ptr = lhs_packed_mtx_qa8dx;

      const size_t rhs_packed_offset =
        ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(n_start, k);
      const void *rhs_packed_ptr =
        (const void *)((const char *)rhs_packed_mtx_qs4cx + rhs_packed_offset);

      const size_t dst_stride = n * sizeof(float);
      const size_t dst_offset =
        ukernel_variants[idx_variant].ukernel.get_dst_offset(0, n_start,
                                                             dst_stride);
      float *dst_ptr = (float *)((uint8_t *)dst_act_mtx_f32 + dst_offset);

      ukernel_variants[idx_variant].ukernel.run_matmul(
        m, n_to_process, k,      // Dimensions
        lhs_packed_ptr,          // LHS packed
        rhs_packed_ptr,          // RHS packed
        dst_ptr,                 // DST
        dst_stride,              // DST stride (row)
        sizeof(float),           // DST stride (col)
        lower_bound, upper_bound // Min and max for the clamp operation
      );
    });
  }

  delete[] lhs_packed_mtx_qa8dx;
}

// NOTE: the old nntr_kai_gemm_qai8dxp_qsi4cxp_olp dispatcher was dropped in the
// merge — upstream folded its single-thread / n-parallel logic into
// __kai_gemm_qai8dxp_qsi4cxp above, so the f32 facade now calls that directly.
// Only the fp16-dst fork remains ours-only (i8mm-gated, matching upstream).
#if defined(__ARM_FEATURE_MATMUL_INT8)
void nntr_kai_gemm_qai8dxp_qsi4cxp_olp_f16(
  size_t m, size_t n, size_t k, const void *lhs_native_mtx_f16,
  void *rhs_packed_mtx_qs4cx, void *dst_act_mtx_f16, bool transB,
  float lower_bound, float upper_bound) {
  // Locked to the 4x4x32 i8mm variant — the only KAI matmul we have a
  // fp16-dst fork for. The packed RHS bytes must come from a 4x4x32-
  // compatible packing (nr=4, kr=16, sr=2), which is what
  // Int4Utils::quantizeAndPackKai emits and what the load-time
  // assembleKaiRhsPacked helper turns into a full KAI rhs_packed buffer.
  (void)transB; // RHS layout is fixed for this fp16 path

  constexpr size_t mr = 4;
  constexpr size_t kr = 16;
  constexpr size_t sr = 2;
  constexpr size_t kai_fp16_size = sizeof(uint16_t);

  // LHS packing direct from fp16 — no fp32 promotion buffer. Packed once here
  // (shared, read-only) and reused by every N-column worker below.
  const size_t lhs_packed_size =
    kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16(m, k, mr, kr, sr);
  uint8_t *lhs_packed_mtx_qa8dx = new uint8_t[lhs_packed_size];
  kai_run_lhs_quant_pack_qai8dxp_f16(m, k, mr, kr, sr, /*m_idx_start=*/0,
                                     lhs_native_mtx_f16, k * kai_fp16_size,
                                     lhs_packed_mtx_qa8dx);

  const size_t dst_stride = n * kai_fp16_size;
  const size_t lhs_offset =
    kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
      0, k);
  const void *lhs_ptr = (const char *)lhs_packed_mtx_qa8dx + lhs_offset;

  // Parallelize the matmul over N (output columns). Each worker owns a disjoint
  // n_step-wide band of the RHS/DST (LHS is shared read-only), so no sync is
  // needed and parallel_for is a barrier before the free below. Without
  // this the whole GEMM ran on a single core — the FP32 facade
  // (__kai_gemm_qai8dxp_qsi4cxp) already splits n this way; the fp16-dst fork
  // had been left single-threaded, which made every QS4CX-FP16 FC (and the
  // untied QS4CX lm_head) the decode bottleneck.
  const size_t n_step =
    kai_get_n_step_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm();
  const size_t n_loop = (n + n_step - 1) / n_step;
  auto &tm = nntrainer::ThreadManager::Global();

  tm.parallel_for(0, n_loop, [&](size_t i) {
    const size_t n_start = i * n_step;
    const size_t n_to_process = std::min(n_step, n - n_start);

    const size_t rhs_offset =
      kai_get_rhs_packed_offset_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
        n_start, k);
    const size_t dst_offset =
      kai_get_dst_offset_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
        0, n_start, dst_stride);
    const void *rhs_ptr = (const char *)rhs_packed_mtx_qs4cx + rhs_offset;
    void *dst_ptr = (uint8_t *)dst_act_mtx_f16 + dst_offset;

    kai_run_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
      m, n_to_process, k, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, kai_fp16_size,
      lower_bound, upper_bound);
  });

  delete[] lhs_packed_mtx_qa8dx;
}
#endif // defined(__ARM_FEATURE_MATMUL_INT8)

} // namespace nntrainer

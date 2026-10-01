// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nntr_ggml_impl_avxvnni.cpp
 * @date   01 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  AVX-VNNI build of the Q4_0 8x8 int8 GEMM/GEMV kernels
 *
 * This translation unit is compiled with -mavxvnni, so __AVXVNNI__ selects
 * the vpdpbusd branch of the int8 dot helpers. It contains only
 * nntr_gemm_q4_0_8x8_q8_0_avxvnni() and nntr_gemv_q4_0_8x8_q8_0_avxvnni();
 * nntr_ggml_impl_avx.cpp calls them only when cpuid reports AVX-VNNI.
 */

#define NNTR_GGML_AVXVNNI_VARIANT 1
#include "nntr_ggml_impl_avx.cpp"

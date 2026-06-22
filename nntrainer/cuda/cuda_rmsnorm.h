// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_rmsnorm.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device RMSNorm op for the CUDA backend. Row-wise:
 *          y = x * rsqrt(mean(x^2) + eps) * gamma  (gamma optional / raw, no
 *          (1+gamma) bias -- matches ReshapedRMSNormLayer). Sum of squares is
 *          accumulated in FP32. Callers must pass device-accessible (UVM)
 *          pointers.
 */

#ifndef __CUDA_RMSNORM_H__
#define __CUDA_RMSNORM_H__

namespace nntrainer::cuda {

/**
 * @brief FP32 row-wise RMSNorm on device (UVM) pointers.
 *
 * @param in     [rows, width] row-major input (device-accessible)
 * @param gamma  [width] per-feature scale, or nullptr for the gamma-free norm
 * @param out    [rows, width] row-major output (device-accessible)
 * @param eps    epsilon added to the mean of squares
 * @param rows   number of rows (one block per row)
 * @param width  feature size (the normalized dimension)
 * @return true on success
 */
bool cuda_rmsnorm_fp32(const float *in, const float *gamma, float *out,
                       float eps, unsigned int rows, unsigned int width);

} // namespace nntrainer::cuda

#endif // __CUDA_RMSNORM_H__

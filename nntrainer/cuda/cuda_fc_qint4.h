// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_fc_qint4.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Fused QINT4 (Int4QTensor) GEMM for the CUDA FC layer:
 *          Y[M,N] = X[M,K] * dequant(W), where W is a row-major [K,N] QINT4
 *          weight (signed nibbles, even index = high nibble, FP32 per-group
 *          scales scale[i/group]). The int4 weight is read and dequantized
 *          INLINE inside the kernel -- no dense FP32 weight buffer -- so it
 *          fits the memory budget of real-size (e2b) models. This is the
 *          correctness/memory floor; a dp4a (int8-act x int4-weight) kernel is
 *          a later perf refinement.
 */

#ifndef __CUDA_FC_QINT4_H__
#define __CUDA_FC_QINT4_H__

namespace nntrainer::cuda {

/**
 * @brief Y[M,N] = X[M,K] * dequant(QINT4 W[K,N]) on device (UVM) pointers.
 *
 * @param X       [M,K] row-major FP32 activation (device-accessible)
 * @param nibbles packed int4 weight, (K*N+1)/2 bytes (device-accessible)
 * @param scales  FP32 scales, K*N/group entries; weight elem i uses scale[i/group]
 * @param Y       [M,N] row-major FP32 output (device-accessible)
 * @param M,N,K   GEMM dims
 * @param group   scale group size (Int4QTensor::getGroupSize(), default 32)
 * @return true on success
 */
bool cuda_fc_qint4_gemm_fp32(const float *X, const unsigned char *nibbles,
                             const float *scales, float *Y, unsigned int M,
                             unsigned int N, unsigned int K, unsigned int group);

} // namespace nntrainer::cuda

#endif // __CUDA_FC_QINT4_H__

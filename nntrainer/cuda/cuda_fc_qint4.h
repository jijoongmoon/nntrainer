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

/**
 * @brief Y[M,N] = X[M,K] * dequant(QINT4 W) where W is stored in the KAI
 *        qsi4cxp 4x4x32 "Section A" super-row layout -- the actual in-memory
 *        form of every loaded Int4QTensor (Int4Utils::packPlainToSectionA).
 *        The signed int4 weight for (output channel n, input channel k) is
 *        decoded inline by inverting the Section-A index permutation; no dense
 *        FP32 weight buffer is materialised (fits real-size e2b models). Per-
 *        output-channel fp16 scale (one per N), converted to fp32 in-kernel.
 *
 * @param X            [M,K] row-major FP32 activation (device-accessible)
 * @param section_a    Section-A nibble payload = weight.getData() (device-acc)
 * @param scales_fp16  N fp16 per-channel scales = weight.getScale() (device-acc)
 * @param Y            [M,N] row-major FP32 output (device-accessible)
 * @param M,N,K        GEMM dims; requires N%4==0 and K%32==0 (load invariant)
 * @return true on success
 */
bool cuda_fc_qint4_sectionA_gemm_fp32(const float *X,
                                      const unsigned char *section_a,
                                      const unsigned short *scales_fp16,
                                      float *Y, unsigned int M, unsigned int N,
                                      unsigned int K);

/**
 * @brief Same as cuda_fc_qint4_sectionA_gemm_fp32 but for HOST-resident inputs.
 *        engine=cuda tensors currently live on the host heap (FloatTensor /
 *        Int4QTensor allocate() bypass the UVM pool), so the QINT4 weight,
 *        activation and output are not device-accessible. This wrapper mirrors
 *        the weight into device memory ONCE (cached by the host weight pointer
 *        -- weights are constant) and stages the activation in / output out per
 *        call, then runs the device kernel. It is the CUDA analogue of the
 *        OpenCL cl_mem residency bridge; a future UVM-resident tensor pool
 *        would let the zero-copy path above be taken instead.
 *
 * @param host_X        [M,K] FP32 activation on the host heap
 * @param host_secA     Section-A nibble payload on the host heap (cache key)
 * @param host_scales   N fp16 per-channel scales on the host heap
 * @param host_Y        [M,N] FP32 output on the host heap (written back)
 */
bool cuda_fc_qint4_sectionA_gemm_fp32_resident(const float *host_X,
                                               const unsigned char *host_secA,
                                               const unsigned short *host_scales,
                                               float *host_Y, unsigned int M,
                                               unsigned int N, unsigned int K);

} // namespace nntrainer::cuda

#endif // __CUDA_FC_QINT4_H__

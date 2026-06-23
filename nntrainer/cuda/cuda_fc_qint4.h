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

/**
 * @brief w4a8 dp4a fast path: Y[M,N] = X[M,K] * dequant(QINT4 W) using Ada
 *        __dp4a (4-way int8 dot-accumulate). The FP32 activation is quantized
 *        to int8 per-row (symmetric, absmax/127); the KAI Section-A int4 weight
 *        is repacked ONCE into plain row-major int4 [N,K] (cached by the
 *        section_a pointer, reusing the validated inverse mapping); the GEMM
 *        accumulates int32 via __dp4a and rescales by act_scale[m]*w_scale[n].
 *        Much higher arithmetic throughput than the naive FP32 kernel, and no
 *        FP32 weight blow-up (stays int4 in memory). All pointers must be
 *        device-accessible (UVM). Requires N%4==0, K%32==0 (load invariant).
 *
 * @param X            [M,K] row-major FP32 activation (device-accessible)
 * @param section_a    Section-A nibble payload = weight.getData() (device-acc)
 * @param scales_fp16  N fp16 per-channel scales = weight.getScale() (device-acc)
 * @param Y            [M,N] row-major FP32 output (device-accessible)
 * @param M,N,K        GEMM dims
 * @return true on success
 */
bool cuda_fc_qint4_sectionA_dp4a_gemm_fp32(const float *X,
                                           const unsigned char *section_a,
                                           const unsigned short *scales_fp16,
                                           float *Y, unsigned int M,
                                           unsigned int N, unsigned int K);

/**
 * @brief FP16-activation variant of the dp4a fast path (the real CausalLM
 *        models use QINT4-FP16: int4 weight + fp16 activations). The fp16
 *        activation is read directly into the int8 quantizer and the fp16
 *        output is produced by a final float->half conversion.
 *
 * @param Xh           [M,K] row-major fp16 activation (device-accessible)
 * @param section_a    Section-A nibble payload (device-accessible)
 * @param scales_fp16  N fp16 per-channel scales (device-accessible)
 * @param Yh           [M,N] row-major fp16 output (device-accessible)
 */
bool cuda_fc_qint4_sectionA_dp4a_gemm_fp16(const unsigned short *Xh,
                                           const unsigned char *section_a,
                                           const unsigned short *scales_fp16,
                                           unsigned short *Yh, unsigned int M,
                                           unsigned int N, unsigned int K);

/**
 * @brief w4a8 on the INT8 Tensor Cores via cuBLAS (prefill FC). Same quant
 *        scheme as the dp4a path (per-row asym int8 activation + symmetric int4
 *        weight, fp16 io), but the int8xint8->int32 matmul runs on IMMA Tensor
 *        Cores (cublasGemmEx CUDA_R_8I) instead of __dp4a on the int ALU --
 *        ~10x the GEMM throughput at prefill M. The int4 weight is unpacked to
 *        int8 [K,N] ONCE (cached by section_a) so the unpack stays off the
 *        per-call path. Bit-identical to dp4a (exact int32 accumulate). Best
 *        for large M (prefill); decode M=1 stays on the dp4a GEMV (BW-bound,
 *        Tensor Cores do not help).
 */
bool cuda_fc_qint4_sectionA_cublas_i8_gemm_fp16(const unsigned short *Xh,
                                                const unsigned char *section_a,
                                                const unsigned short *scales_fp16,
                                                unsigned short *Yh,
                                                unsigned int M, unsigned int N,
                                                unsigned int K);

/**
 * @brief High-accuracy fp16 path: FP32-precision activation (no int8 quant),
 *        naive Section-A decode-GEMM. Selected by NNTR_FC_CUDA_DP4A=0 for an
 *        fp16 activation (and as an accuracy reference vs the dp4a w4a8 path).
 */
bool cuda_fc_qint4_sectionA_gemm_fp16_naive(const unsigned short *Xh,
                                            const unsigned char *section_a,
                                            const unsigned short *scales_fp16,
                                            unsigned short *Yh, unsigned int M,
                                            unsigned int N, unsigned int K);

} // namespace nntrainer::cuda

#endif // __CUDA_FC_QINT4_H__

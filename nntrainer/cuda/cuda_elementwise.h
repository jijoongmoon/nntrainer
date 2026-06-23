// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_elementwise.h
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device element-wise ops (geglu / add / scalar-mul / slice) for the
 *          gemma4 decode path -- the small host ops that break the GPU chain.
 *          fp16 I/O, FP32 math; all reduce per-op host work to one kernel.
 */

#ifndef __CUDA_ELEMENTWISE_H__
#define __CUDA_ELEMENTWISE_H__

namespace nntrainer::cuda {

/** @brief out[i] = gelu_tanh(gate[i]) * up[i], gelu_tanh = pytorch-tanh approx */
bool cuda_geglu_fp16(const unsigned short *gate, const unsigned short *up,
                     unsigned short *out, unsigned int n);

/** @brief out[i] = a[i] + b[i] (residual add) */
bool cuda_add_fp16(const unsigned short *a, const unsigned short *b,
                   unsigned short *out, unsigned int n);

/** @brief out[i] = in[i] * scalar */
bool cuda_scalar_mul_fp16(const unsigned short *in, unsigned short *out,
                          unsigned int n, float scalar);

/** @brief out[i] = cap * tanh(in[i] / cap) -- final logit softcapping */
bool cuda_softcap_fp16(const unsigned short *in, unsigned short *out,
                       unsigned int n, float cap);

/** @brief out[r*fs + f] = in[r*in_width + layer_off + f] (per-layer slice) */
bool cuda_slice_copy_fp16(const unsigned short *in, unsigned short *out,
                          unsigned int rows, unsigned int in_width,
                          unsigned int layer_off, unsigned int fs);

} // namespace nntrainer::cuda

#endif // __CUDA_ELEMENTWISE_H__

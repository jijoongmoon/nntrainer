// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_rope.h
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device RoPE (rotary position embedding) for the gemma4 decode path.
 *
 * Matches the host compute_rotary_emb_value split-half convention exactly:
 * for each head and each k in [0, head_dim/2):
 *   out[k]        = in[k]*cos[k] - in[k+half]*sin[k]
 *   out[k+half]   = in[k]*sin[k] + in[k+half]*cos[k]
 * (full rotation over head_dim; the per-position cos/sin LUT row has head_dim/2
 * fp16 entries). FP32 math, fp16 I/O. One block per head.
 */

#ifndef __CUDA_ROPE_H__
#define __CUDA_ROPE_H__

namespace nntrainer::cuda {

/**
 * @brief  Apply RoPE on the device to an interleaved fp16 [num_heads*head_dim]
 *         row (one token). in/out are device-accessible (UVM); cos_row/sin_row
 *         are the host LUT row for the token's absolute position (head_dim/2
 *         fp16 entries) -- mirrored to the device internally.
 * @param in        [num_heads*head_dim] fp16 bits (device-accessible), one token
 * @param out       [num_heads*head_dim] fp16 bits (device-accessible); may == in
 * @param cos_row   [head_dim/2] fp16 bits (host or device) for this position
 * @param sin_row   [head_dim/2] fp16 bits
 * @param num_heads number of heads packed in the row
 * @param head_dim  per-head dim (256 sliding / 512 full); half = head_dim/2
 * @return true on success
 */
bool cuda_rope_fp16(const unsigned short *in, unsigned short *out,
                    const unsigned short *cos_row, const unsigned short *sin_row,
                    int num_heads, int head_dim);

} // namespace nntrainer::cuda

#endif // __CUDA_ROPE_H__

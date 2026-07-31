// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   rms_reverse_norm_cl_op.h
 * @date   29 July 2026
 * @brief  OpenCL reverse-RMSNorm whole-op kernel dispatch
 *         (out = out_scale * normalize(in * weight)).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details Relocated verbatim from RMSReverseNormLayer::incremental_forwarding
 * (the gpu_svm branch) and blas_kernels.cpp (the rms_reverse_norm_cl_fp16
 * dispatch) so the reverse-norm Layer is a single backend-neutral Layer that
 * dispatches through ComputeOps (ClComputeOps::rms_reverse_norm forwards
 * here). The SVM residency contract that used to gate the layer's CL branch
 * is checked HERE, and a miss is reported to the caller with a NAMED reason
 * (rms_reverse_norm_cl_last_reject_reason) instead of falling through
 * silently. (N4).
 *
 * This header is deliberately NOT in the meson install list: the raw kernel
 * dispatch is a private backend detail behind the ComputeOps virtual, not
 * public ABI (the reverse of what 652e8f2e4 did to blas_kernels.h).
 */

#ifndef __RMS_REVERSE_NORM_CL_OP_H__
#define __RMS_REVERSE_NORM_CL_OP_H__

#include <cl_context.h>

namespace nntrainer {

class Tensor;

/**
 * @brief Reverse-RMSNorm over rows [row_offset, row_offset + active_rows) of
 *        width in.width(): out = out_scale * normalize(in * weight), with the
 *        per-feature @a weight folded INSIDE the RMS denominator (it couples
 *        all features — this is NOT expressible as rmsnorm*gamma) and
 *        @a out_scale applied as a post-norm scalar. FP32 accumulation.
 *
 * @return true when the GPU kernel was dispatched. false when the call does
 *         not meet the CL contract (non-FP16 activation, an operand outside
 *         the SVM pool, or a non-FP16 weight/out_scale); the reject reason is
 *         recorded for rms_reverse_norm_cl_last_reject_reason() and NOTHING
 *         was computed — the caller owns the (named) host fallback.
 */
bool rms_reverse_norm_cl_op(Tensor &in, Tensor &out, const Tensor &weight,
                            const Tensor &out_scale, float epsilon,
                            unsigned int active_rows, unsigned int row_offset);

/**
 * @brief Why the last rms_reverse_norm_cl_op() call on this thread returned
 *        false (V8C_REJECT-style; "none" when it never rejected). Used by
 *        ClComputeOps::rms_reverse_norm to NAME the host bounce.
 */
const char *rms_reverse_norm_cl_last_reject_reason();

} // namespace nntrainer

#endif // __RMS_REVERSE_NORM_CL_OP_H__

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dit_rope.cpp
 * @date   16 June 2026
 * @brief  Adjacent-pair head-0-only rotary embedding for the Token2Wav DiT.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <stdexcept>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#endif

#include "dit_rope.h"

namespace causallm {

static constexpr size_t X_IDX = 0;
static constexpr size_t COS_IDX = 1;
static constexpr size_t SIN_IDX = 2;
static constexpr size_t OUT_IDX = 0;

void DiTRoPELayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 3, std::invalid_argument)
    << "dit_rope expects 3 inputs (x, cos, sin)";
  const auto &in = context.getInputDimensions();
  const unsigned int head_dim = in[COS_IDX].width();
  NNTR_THROW_IF(head_dim == 0 || (head_dim % 2) != 0 ||
                  in[X_IDX].width() % head_dim != 0,
                std::invalid_argument)
    << "dit_rope: x width " << in[X_IDX].width()
    << " must be a multiple of an even head_dim " << head_dim;
  context.setOutputDimensions({in[X_IDX]});
}

void DiTRoPELayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // producer FCs may be in-flight cuBLAS kernels under NNTR_CUDA_ASYNC
  nntrainer::cuda::drain_if_async();
#endif

  nntrainer::Tensor &x = context.getInput(X_IDX);
  nntrainer::Tensor &cos = context.getInput(COS_IDX);
  nntrainer::Tensor &sin = context.getInput(SIN_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  NNTR_THROW_IF(x.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "dit_rope supports FP32 only";

  const unsigned int hd = cos.width(); // head_dim (=64)
  const unsigned int width = x.width(); // num_heads * head_dim
  const unsigned int seq = x.height();
  const bool cos_batched = cos.batch() == x.batch();

  for (unsigned int b = 0; b < x.batch(); ++b) {
    for (unsigned int s = 0; s < seq; ++s) {
      const float *xrow = x.getData<float>() + x.getIndex(b, 0, s, 0);
      float *orow = out.getData<float>() + out.getIndex(b, 0, s, 0);
      const unsigned int cb = cos_batched ? b : 0;
      const float *crow = cos.getData<float>() + cos.getIndex(cb, 0, s, 0);
      const float *srow = sin.getData<float>() + sin.getIndex(cb, 0, s, 0);

      // head 0: adjacent-pair rotation; out = x*cos + rotate_half(x)*sin
      // rotate_half: [2j] = -x[2j+1], [2j+1] = x[2j]
      for (unsigned int i = 0; i < hd; ++i) {
        const float rh = (i % 2 == 0) ? -xrow[i + 1] : xrow[i - 1];
        orow[i] = xrow[i] * crow[i] + rh * srow[i];
      }
      // heads 1..: pass through unchanged
      for (unsigned int i = hd; i < width; ++i)
        orow[i] = xrow[i];
    }
  }
}

} // namespace causallm

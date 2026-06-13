// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   mrope_apply.cpp
 * @date   13 June 2026
 * @brief  Apply precomputed rotary embedding (cos/sin) to a q or k tensor.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <stdexcept>

#include "mrope_apply.h"

namespace causallm {

static constexpr size_t X_IDX = 0;
static constexpr size_t COS_IDX = 1;
static constexpr size_t SIN_IDX = 2;
static constexpr size_t OUT_IDX = 0;

void MRoPEApplyLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 3, std::invalid_argument)
    << "mrope_apply expects 3 inputs (x, cos, sin)";
  const auto &in = context.getInputDimensions();
  const unsigned int head_dim = in[COS_IDX].width();
  NNTR_THROW_IF(head_dim == 0 || in[X_IDX].width() % head_dim != 0,
                std::invalid_argument)
    << "mrope_apply: x width " << in[X_IDX].width()
    << " not a multiple of head_dim " << head_dim;
  context.setOutputDimensions({in[X_IDX]});
}

void MRoPEApplyLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &x = context.getInput(X_IDX);
  nntrainer::Tensor &cos = context.getInput(COS_IDX);
  nntrainer::Tensor &sin = context.getInput(SIN_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  NNTR_THROW_IF(x.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "mrope_apply supports FP32 only";

  const unsigned int hd = cos.width();
  const unsigned int half = hd / 2;
  const unsigned int num_heads = x.width() / hd;
  const unsigned int iter = to - from;

  for (unsigned int b = 0; b < x.batch(); ++b) {
    for (unsigned int s = 0; s < iter; ++s) {
      const unsigned int gpos = from + s;
      // cos/sin side inputs are indexed by GLOBAL position (batch 0 row gpos)
      const float *crow = cos.getData<float>() + cos.getIndex(0, 0, gpos, 0);
      const float *srow = sin.getData<float>() + sin.getIndex(0, 0, gpos, 0);
      float *xrow = x.getData<float>() + x.getIndex(b, 0, s, 0);
      float *orow = out.getData<float>() + out.getIndex(b, 0, s, 0);
      for (unsigned int h = 0; h < num_heads; ++h) {
        float *xv = xrow + static_cast<size_t>(h) * hd;
        float *ov = orow + static_cast<size_t>(h) * hd;
        for (unsigned int i = 0; i < hd; ++i) {
          const float rh = (i < half) ? -xv[i + half] : xv[i - half];
          ov[i] = xv[i] * crow[i] + rh * srow[i];
        }
      }
    }
  }
}

void MRoPEApplyLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  nntrainer::TensorDim x_dim = context.getInput(X_IDX).getDim();
  nntrainer::TensorDim out_dim = context.getOutput(OUT_IDX).getDim();
  x_dim.height(input_dimensions[0].height());
  out_dim.height(input_dimensions[0].height());
  context.updateInput(X_IDX, x_dim);
  context.updateOutput(OUT_IDX, out_dim);
}

} // namespace causallm

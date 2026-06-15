// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   snake_beta.cpp
 * @date   15 June 2026
 * @brief  Snake-beta activation used by the Qwen2.5-Omni Token2Wav BigVGAN.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <stdexcept>

#include "snake_beta.h"

namespace causallm {

static constexpr size_t X_IDX = 0;
static constexpr size_t OUT_IDX = 0;
static constexpr size_t ALPHA_IDX = 0;
static constexpr size_t BETA_IDX = 1;
static constexpr float NO_DIV_BY_ZERO = 1e-9f;

void SnakeBetaLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "snake_beta expects a single input";

  const auto &in = context.getInputDimensions()[0];
  const unsigned int channels = in.channel();
  NNTR_THROW_IF(channels == 0, std::invalid_argument)
    << "snake_beta: input channel dim must be non-zero";

  // alpha then beta: request in load (DFS-from-output) order so the converter
  // can emit [alpha, beta] per instance. ZEROS is only a placeholder; the
  // checkpoint overwrites with trained log-domain values.
  nntrainer::TensorDim wdim({1, 1, 1, channels});
  wt_idx[ALPHA_IDX] =
    context.requestWeight(wdim, nntrainer::Initializer::ZEROS,
                          nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
                          "alpha", false);
  wt_idx[BETA_IDX] =
    context.requestWeight(wdim, nntrainer::Initializer::ZEROS,
                          nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
                          "beta", false);

  context.setOutputDimensions({in});
}

void SnakeBetaLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  nntrainer::Tensor &x = context.getInput(X_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  NNTR_THROW_IF(x.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "snake_beta supports FP32 only";

  const float *alpha = context.getWeight(wt_idx[ALPHA_IDX]).getData<float>();
  const float *beta = context.getWeight(wt_idx[BETA_IDX]).getData<float>();

  const unsigned int C = x.channel();
  // NCHW is contiguous over (height, width) for a fixed (batch, channel).
  const unsigned int HW = x.height() * x.width();

  for (unsigned int b = 0; b < x.batch(); ++b) {
    for (unsigned int c = 0; c < C; ++c) {
      const float a = std::exp(alpha[c]);
      const float inv_b = 1.0f / (std::exp(beta[c]) + NO_DIV_BY_ZERO);
      const float *xc = x.getData<float>() + x.getIndex(b, c, 0, 0);
      float *oc = out.getData<float>() + out.getIndex(b, c, 0, 0);
      for (unsigned int i = 0; i < HW; ++i) {
        const float s = std::sin(a * xc[i]);
        oc[i] = xc[i] + inv_b * s * s;
      }
    }
  }
}

} // namespace causallm

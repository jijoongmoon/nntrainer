// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dit_modulate.cpp
 * @date   16 June 2026
 * @brief  AdaLN modulation (no-affine LN + scale/shift) for the Token2Wav DiT.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <stdexcept>
#include <string>

#include "dit_modulate.h"

namespace causallm {

static constexpr size_t X_IDX = 0;
static constexpr size_t COND_IDX = 1;
static constexpr size_t OUT_IDX = 0;
static constexpr float LN_EPS = 1e-6f;

void DiTModulateLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "dit_modulate expects 2 inputs (x, cond)";
  context.setOutputDimensions({context.getInputDimensions()[X_IDX]});
}

void DiTModulateLayer::forwarding(nntrainer::RunLayerContext &context,
                                  bool training) {
  nntrainer::Tensor &x = context.getInput(X_IDX);
  nntrainer::Tensor &cond = context.getInput(COND_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  NNTR_THROW_IF(x.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "dit_modulate supports FP32 only";

  const unsigned int C = x.width();
  const unsigned int seq = x.height();
  const bool cond_batched = cond.batch() == x.batch();

  for (unsigned int b = 0; b < x.batch(); ++b) {
    const unsigned int cb = cond_batched ? b : 0;
    const float *cr = cond.getData<float>() + cond.getIndex(cb, 0, 0, 0);
    const float *scale = cr + scale_off;
    const float *shift = cr + shift_off;
    for (unsigned int s = 0; s < seq; ++s) {
      const float *xr = x.getData<float>() + x.getIndex(b, 0, s, 0);
      float *orow = out.getData<float>() + out.getIndex(b, 0, s, 0);
      // no-affine LayerNorm over C
      float mean = 0.0f;
      for (unsigned int c = 0; c < C; ++c)
        mean += xr[c];
      mean /= C;
      float var = 0.0f;
      for (unsigned int c = 0; c < C; ++c) {
        float d = xr[c] - mean;
        var += d * d;
      }
      var /= C;
      const float inv = 1.0f / std::sqrt(var + LN_EPS);
      for (unsigned int c = 0; c < C; ++c) {
        float n = (xr[c] - mean) * inv;
        orow[c] = n * (1.0f + scale[c]) + shift[c];
      }
    }
  }
}

void DiTModulateLayer::setProperty(const std::vector<std::string> &values) {
  for (const auto &v : values) {
    auto pos = v.find('=');
    if (pos == std::string::npos)
      continue;
    const std::string key = v.substr(0, pos);
    const std::string val = v.substr(pos + 1);
    if (key == "scale_off")
      scale_off = static_cast<unsigned int>(std::stoul(val));
    else if (key == "shift_off")
      shift_off = static_cast<unsigned int>(std::stoul(val));
  }
}

} // namespace causallm

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dit_gate.cpp
 * @date   16 June 2026
 * @brief  Gated residual (residual + broadcast(gate)*x) for the Token2Wav DiT.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <stdexcept>
#include <string>

#include "dit_gate.h"

namespace causallm {

static constexpr size_t RES_IDX = 0;
static constexpr size_t X_IDX = 1;
static constexpr size_t COND_IDX = 2;
static constexpr size_t OUT_IDX = 0;

void DiTGateLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 3, std::invalid_argument)
    << "dit_gate expects 3 inputs (residual, x, cond)";
  context.setOutputDimensions({context.getInputDimensions()[RES_IDX]});
}

void DiTGateLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
  nntrainer::Tensor &res = context.getInput(RES_IDX);
  nntrainer::Tensor &x = context.getInput(X_IDX);
  nntrainer::Tensor &cond = context.getInput(COND_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  NNTR_THROW_IF(x.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "dit_gate supports FP32 only";

  const unsigned int C = x.width();
  const unsigned int seq = x.height();
  const bool cond_batched = cond.batch() == x.batch();

  for (unsigned int b = 0; b < x.batch(); ++b) {
    const unsigned int cb = cond_batched ? b : 0;
    const float *gate = cond.getData<float>() + cond.getIndex(cb, 0, 0, 0) + gate_off;
    for (unsigned int s = 0; s < seq; ++s) {
      const float *rr = res.getData<float>() + res.getIndex(b, 0, s, 0);
      const float *xr = x.getData<float>() + x.getIndex(b, 0, s, 0);
      float *orow = out.getData<float>() + out.getIndex(b, 0, s, 0);
      for (unsigned int c = 0; c < C; ++c)
        orow[c] = rr[c] + gate[c] * xr[c];
    }
  }
}

void DiTGateLayer::setProperty(const std::vector<std::string> &values) {
  for (const auto &v : values) {
    auto pos = v.find('=');
    if (pos == std::string::npos)
      continue;
    if (v.substr(0, pos) == "gate_off")
      gate_off = static_cast<unsigned int>(std::stoul(v.substr(pos + 1)));
  }
}

} // namespace causallm

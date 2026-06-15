// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   scale_layer.cpp
 * @date   15 June 2026
 * @brief  Elementwise multiply-by-constant layer (no weights).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <stdexcept>
#include <string>

#include "scale_layer.h"

namespace causallm {

static constexpr size_t X_IDX = 0;
static constexpr size_t OUT_IDX = 0;

void ScaleLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "scale expects a single input";
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void ScaleLayer::forwarding(nntrainer::RunLayerContext &context,
                            bool training) {
  nntrainer::Tensor &x = context.getInput(X_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  NNTR_THROW_IF(x.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "scale supports FP32 only";

  const float *xp = x.getData<float>();
  float *op = out.getData<float>();
  const size_t n = x.size();
  const float s = scale_value;
  for (size_t i = 0; i < n; ++i)
    op[i] = s * xp[i];
}

void ScaleLayer::setProperty(const std::vector<std::string> &values) {
  for (const auto &v : values) {
    auto pos = v.find('=');
    if (pos == std::string::npos)
      continue;
    const std::string key = v.substr(0, pos);
    const std::string val = v.substr(pos + 1);
    if (key == "scale")
      scale_value = std::stof(val);
    // ignore name/other keys handled by the graph
  }
}

} // namespace causallm

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dit_act.cpp
 * @date   18 July 2026
 * @brief  Elementwise activation with a CUDA-stream drain for the DiT.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <stdexcept>

#include <cpu_backend.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#endif

#include "dit_act.h"

namespace causallm {

void DiTActLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "dit_act expects 1 input";
  NNTR_THROW_IF(fn != "tanh_gelu" && fn != "swish", std::invalid_argument)
    << "dit_act: unsupported fn: " << fn;
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void DiTActLayer::setProperty(const std::vector<std::string> &values) {
  for (const auto &value : values) {
    const auto pos = value.find('=');
    NNTR_THROW_IF(pos == std::string::npos, std::invalid_argument)
      << "dit_act: invalid property: " << value;
    const std::string key = value.substr(0, pos);
    if (key == "fn")
      fn = value.substr(pos + 1);
    else
      NNTR_THROW_IF(true, std::invalid_argument)
        << "dit_act: unknown property: " << key;
  }
}

void DiTActLayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  nntrainer::Tensor &in = context.getInput(0);
  nntrainer::Tensor &out = context.getOutput(0);

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // the producing FC may be an in-flight cuBLAS kernel under NNTR_CUDA_ASYNC
  nntrainer::cuda::drain_if_async();
#endif

  const float *x = in.getData<float>();
  float *y = out.getData<float>();
  const size_t n = in.size();
  if (fn == "tanh_gelu") {
    // exact same routine the core activation layer uses (acti_func.h)
    nntrainer::tanh_gelu(static_cast<unsigned int>(n), x, y);
  } else { // swish = x * sigmoid(x), matching ActiFunc::swish elementwise
    for (size_t i = 0; i < n; ++i)
      y[i] = x[i] * (1.0f / (1.0f + std::exp(-x[i])));
  }
}

} // namespace causallm

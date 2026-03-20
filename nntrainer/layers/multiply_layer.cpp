// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   multiply_layer.cpp
 * @date   10 Oct 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is multiply layer class (operation layer)
 *
 */

#include <multiply_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void MultiplyLayer::finalize(InitLayerContext &context) {
  auto const &input_dims = context.getInputDimensions();
  TensorDim out_dim = input_dims[0];

  if (input_dims.size() > 1) {
    TensorDim dim1 = input_dims[1];
    // Compute broadcast output shape: for each dimension, take the max
    // when one of the inputs has size 1 (standard broadcasting rules).
    for (unsigned int i = 0; i < 4; ++i) {
      if (out_dim[i] != dim1[i]) {
        if (out_dim[i] == 1) {
          out_dim.setTensorDim(i, dim1[i]);
        } else if (dim1[i] != 1) {
          throw std::invalid_argument(
            "MultiplyLayer: incompatible shapes for broadcasting at dim " +
            std::to_string(i) + " (" + std::to_string(out_dim[i]) + " vs " +
            std::to_string(dim1[i]) + ")");
        }
      }
    }
  }

  context.setOutputDimensions({out_dim});
}

void MultiplyLayer::forwarding_operation(const Tensor &input0,
                                         const Tensor &input1, Tensor &hidden) {
  input0.multiply(input1, hidden);
}

void MultiplyLayer::calcDerivative(RunLayerContext &context) {
  context.getOutgoingDerivative(0).copy(
    context.getIncomingDerivative(SINGLE_INOUT_IDX)
      .multiply(context.getInput(1)));

  context.getOutgoingDerivative(1).copy(
    context.getIncomingDerivative(SINGLE_INOUT_IDX)
      .multiply(context.getInput(0)));
}

void MultiplyLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, multiply_props);
  if (!remain_props.empty()) {
    std::string msg = "[MultiplyLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   gather_layer.cpp
 * @date   02 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is gather layer class (operation layer)
 *
 * Implements torch.gather semantics:
 *   output[b][c][h][w] = input[b][idx][h][w]  (axis=1)
 *   output[b][c][h][w] = input[b][c][idx][w]  (axis=2)
 *   output[b][c][h][w] = input[b][c][h][idx]  (axis=3)
 * where idx = (unsigned int)index[b][c][h][w]
 */

#include "common_properties.h"
#include <gather_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <stdexcept>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void GatherLayer::finalize(InitLayerContext &context) {
  axis = std::get<props::Axis>(gather_props).get();
  TensorDim inputDim = context.getInputDimensions()[0];
  TensorDim indexDim = context.getInputDimensions()[1];

  if (axis < 1 || axis > 3) {
    throw std::invalid_argument(
      "The axis property of GatherLayer should be between 1 and 3.");
  }

  if (inputDim[0] != indexDim[0]) {
    throw std::invalid_argument(
      "The batch size of the input and index should be same.");
  }

  TensorDim outputDim = TensorDim(inputDim);
  outputDim.setTensorDim(axis, indexDim[axis]);
  context.setOutputDimensions({outputDim});
}

void GatherLayer::forwarding_operation(const Tensor &input, const Tensor &index,
                                       Tensor &output) {
  unsigned int out_b = output.batch();
  unsigned int out_c = output.channel();
  unsigned int out_h = output.height();
  unsigned int out_w = output.width();

  for (unsigned int b = 0; b < out_b; ++b) {
    for (unsigned int c = 0; c < out_c; ++c) {
      for (unsigned int h = 0; h < out_h; ++h) {
        for (unsigned int w = 0; w < out_w; ++w) {
          unsigned int idx =
            static_cast<unsigned int>(index.getValue<float>(b, c, h, w));

          unsigned int src_c = c, src_h = h, src_w = w;
          switch (axis) {
          case 1:
            src_c = idx;
            break;
          case 2:
            src_h = idx;
            break;
          case 3:
            src_w = idx;
            break;
          }

          output.setValue(b, c, h, w,
                         input.getValue<float>(b, src_c, src_h, src_w));
        }
      }
    }
  }
}

void GatherLayer::calcDerivative(RunLayerContext &context) {
  Tensor &input_deriv = context.getOutgoingDerivative(0);
  const Tensor &output_deriv =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  const Tensor &index = context.getInput(1);

  input_deriv.setValue(0.0f);

  unsigned int out_b = output_deriv.batch();
  unsigned int out_c = output_deriv.channel();
  unsigned int out_h = output_deriv.height();
  unsigned int out_w = output_deriv.width();

  for (unsigned int b = 0; b < out_b; ++b) {
    for (unsigned int c = 0; c < out_c; ++c) {
      for (unsigned int h = 0; h < out_h; ++h) {
        for (unsigned int w = 0; w < out_w; ++w) {
          unsigned int idx =
            static_cast<unsigned int>(index.getValue<float>(b, c, h, w));

          unsigned int src_c = c, src_h = h, src_w = w;
          switch (axis) {
          case 1:
            src_c = idx;
            break;
          case 2:
            src_h = idx;
            break;
          case 3:
            src_w = idx;
            break;
          }

          float grad = output_deriv.getValue<float>(b, c, h, w);
          float current =
            input_deriv.getValue<float>(b, src_c, src_h, src_w);
          input_deriv.setValue(b, src_c, src_h, src_w, current + grad);
        }
      }
    }
  }
}

void GatherLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gather_props);
  if (!remain_props.empty()) {
    std::string msg = "[GatherLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */

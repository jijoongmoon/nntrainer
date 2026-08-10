// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   multiply_layer.cpp
 * @date   10 Oct 2024
 * @see    https://github.com/nntrainer/nntrainer
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

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#endif

namespace nntrainer {

void MultiplyLayer::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(multiply_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(multiply_props).get();
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void MultiplyLayer::forwarding_operation(const Tensor &input0,
                                         const Tensor &input1, Tensor &hidden) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Tensor::multiply is a pure host element loop with no ComputeOps dispatch:
  // on a CUDA graph both operands are device-written (attention output, gate
  // sigmoid) and the only ordering is the producers' per-op drains. Inside a
  // deferred-drain or async region this supplies the missing drain; in plain
  // sync mode it is a no-op.
  nntrainer::cuda::drain_if_async();
#endif
  input0.multiply(input1, hidden);
}

void MultiplyLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  BinaryOperationLayer::incremental_forwarding(context, from, to, training);
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

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   attention_gate.cpp
 * @date   07 April 2026
 * @brief  Sigmoid gating layer for attention output (Qwen3.5)
 * @see    https://github.com/nntrainer/nntrainer
 * @author NNTrainer Authors
 * @bug    No known bugs except for NYI items
 */

#include <cmath>

#include "attention_gate.h"

namespace causallm {

void AttentionGateLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "AttentionGate layer requires 2 inputs (attn_output, gate)";

  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions({dim[0]});
}

void AttentionGateLayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {}

void AttentionGateLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  nntrainer::Tensor &attn_out = context.getInput(0);
  nntrainer::Tensor &gate = context.getInput(1);
  nntrainer::Tensor &output = context.getOutput(0);

  ml::train::TensorDim in_dim = attn_out.getDim();
  ml::train::TensorDim step_dim = in_dim;
  step_dim.batch(1);
  step_dim.height(to - from);

  ml::train::TensorDim out_dim = output.getDim();
  ml::train::TensorDim out_step_dim = out_dim;
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    nntrainer::Tensor a_step = attn_out.getSharedDataTensor(
      step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor g_step = gate.getSharedDataTensor(
      step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor o_step = output.getSharedDataTensor(
      out_step_dim, b * out_dim.getFeatureLen(), true);

    // output = attn_output * sigmoid(gate)
    auto gate_sigmoid = g_step.apply<float>(
      [](float x) -> float { return 1.0f / (1.0f + std::exp(-x)); });
    a_step.multiply(gate_sigmoid, o_step);
  }
}

void AttentionGateLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_attention_gate_layer() {
  auto layer = new AttentionGateLayer();
  return layer;
}

void destroy_attention_gate_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_attention_gate_layer, destroy_attention_gate_layer};
}

#endif

} // namespace causallm

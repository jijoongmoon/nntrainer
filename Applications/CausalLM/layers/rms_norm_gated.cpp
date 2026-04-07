// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   rms_norm_gated.cpp
 * @date   07 April 2026
 * @brief  RMS Normalization with SiLU gating implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author NNTrainer Authors
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <iostream>

#include "rms_norm_gated.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void RMSNormGatedLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  /// Expects 2 inputs: hidden_states and gate
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "RMSNormGated layer expects 2 inputs (hidden_states, gate)";

  context.setOutputDimensions({dim[0]});

  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[0] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);
}

void RMSNormGatedLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {}

void RMSNormGatedLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_gated_props).get();

  nntrainer::Tensor &hidden = context.getInput(0);
  nntrainer::Tensor &gate = context.getInput(1);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[0]);

  ml::train::TensorDim in_dim = hidden.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim step_dim = in_dim;
  step_dim.batch(1);
  step_dim.height(to - from);

  ml::train::TensorDim out_step_dim = out_dim;
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor h_step =
      hidden.getSharedDataTensor(step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor g_step =
      gate.getSharedDataTensor(step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor o_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    // RMSNorm(hidden)
    auto variance = h_step.multiply(h_step).average(3).add(epsilon);
    variance.inv_sqrt_i();
    h_step.multiply(variance, o_step);
    o_step.multiply_i(gamma);

    // * SiLU(gate)
    // SiLU(x) = x * sigmoid(x)
    /// @todo implement efficient SiLU
    auto gate_sigmoid = g_step.apply<float>(
      [](float x) -> float { return x / (1.0f + std::exp(-x)); });
    o_step.multiply_i(gate_sigmoid);
  }
}

void RMSNormGatedLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_gated_layer() {
  auto layer = new RMSNormGatedLayer();
  return layer;
}

void destroy_rms_norm_gated_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_rms_norm_gated_layer, destroy_rms_norm_gated_layer};
}

#endif

} // namespace causallm

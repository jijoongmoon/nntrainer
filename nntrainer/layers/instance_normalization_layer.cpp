// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 NNTrainer contributors
 *
 * @file   instance_normalization_layer.cpp
 * @date   15 Mar 2025
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1607.08022
 * @brief  Instance Normalization Layer Class for Neural Network
 *
 */

#include <cmath>
#include <stdexcept>

#include <layer_context.h>
#include <instance_normalization_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum INParams {
  gamma,
  beta,
  deviation,
  variance,
  inv_std_dev,
};

InstanceNormalizationLayer::InstanceNormalizationLayer() :
  Layer(),
  in_props(props::Epsilon(), props::GammaInitializer(),
           props::BetaInitializer(), props::WeightDecay(),
           props::BiasDecay()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void InstanceNormalizationLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 1) {
    throw std::invalid_argument(
      "Only one input is allowed for instance normalization layer");
  }

  auto const &input_dim = context.getInputDimensions()[0];
  context.setOutputDimensions({input_dim});

  unsigned int channels = input_dim.channel();

  auto gamma_initializer = std::get<props::GammaInitializer>(in_props).get();
  auto beta_initializer = std::get<props::BetaInitializer>(in_props).get();
  auto weight_decay = std::get<props::WeightDecay>(in_props);
  auto bias_decay = std::get<props::BiasDecay>(in_props);

  // gamma and beta are per-channel [1, C, 1, 1]
  TensorDim affine_dim(context.getFormat(), context.getWeightDataType());
  affine_dim.setTensorDim(1, channels);

  wt_idx[INParams::gamma] = context.requestWeight(
    affine_dim, gamma_initializer, WeightRegularizer::NONE, 1.0f,
    weight_decay, "gamma", true);
  wt_idx[INParams::beta] = context.requestWeight(
    affine_dim, beta_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
    "beta", true);

  // deviation: same as input [B, C, H, W]
  wt_idx[INParams::deviation] =
    context.requestTensor(input_dim, "deviation", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);

  // variance and inv_std_dev: per (B, C) -> [B, C, 1, 1]
  TensorDim stat_dim(context.getFormat(), context.getWeightDataType());
  stat_dim.batch(input_dim.batch());
  stat_dim.channel(channels);

  wt_idx[INParams::variance] =
    context.requestTensor(stat_dim, "variance", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  wt_idx[INParams::inv_std_dev] =
    context.requestTensor(stat_dim, "inv_std_dev", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
}

void InstanceNormalizationLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain = loadProperties(values, in_props);
  NNTR_THROW_IF(!remain.empty(), std::invalid_argument)
    << "[InstanceNorm] Unknown properties count " +
         std::to_string(values.size());
}

void InstanceNormalizationLayer::forwarding(RunLayerContext &context,
                                             bool training) {
  const float epsilon = std::get<props::Epsilon>(in_props).get();

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &gamma = context.getWeight(wt_idx[INParams::gamma]);
  Tensor &beta = context.getWeight(wt_idx[INParams::beta]);

  Tensor &deviation = context.getTensor(wt_idx[INParams::deviation]);
  Tensor &variance = context.getTensor(wt_idx[INParams::variance]);
  Tensor &inv_std_dev = context.getTensor(wt_idx[INParams::inv_std_dev]);

  auto dim = input.getDim();
  unsigned int batch = dim.batch();
  unsigned int channels = dim.channel();
  unsigned int height = dim.height();
  unsigned int width = dim.width();
  unsigned int spatial_size = height * width;

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int c = 0; c < channels; ++c) {
      // Compute mean
      float sum = 0.0f;
      for (unsigned int h = 0; h < height; ++h) {
        for (unsigned int w = 0; w < width; ++w) {
          sum += input.getValue(b, c, h, w);
        }
      }
      float mean = sum / spatial_size;

      // Compute variance and deviation
      float var_sum = 0.0f;
      for (unsigned int h = 0; h < height; ++h) {
        for (unsigned int w = 0; w < width; ++w) {
          float val = input.getValue(b, c, h, w) - mean;
          deviation.setValue(b, c, h, w, val);
          var_sum += val * val;
        }
      }
      float var = var_sum / spatial_size;
      float inv_std = 1.0f / std::sqrt(var + epsilon);

      variance.setValue(b, c, 0, 0, var);
      inv_std_dev.setValue(b, c, 0, 0, inv_std);

      // Normalize and apply affine
      float g_val = gamma.getValue(0, c, 0, 0);
      float b_val = beta.getValue(0, c, 0, 0);
      for (unsigned int h = 0; h < height; ++h) {
        for (unsigned int w = 0; w < width; ++w) {
          float normed = deviation.getValue(b, c, h, w) * inv_std;
          output.setValue(b, c, h, w, normed * g_val + b_val);
        }
      }
    }
  }
}

void InstanceNormalizationLayer::incremental_forwarding(
  RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  forwarding(context, training);
}

void InstanceNormalizationLayer::calcDerivative(RunLayerContext &context) {
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);

  const Tensor &gamma = context.getWeight(wt_idx[INParams::gamma]);
  Tensor &deviation = context.getTensor(wt_idx[INParams::deviation]);
  Tensor &inv_std_dev = context.getTensor(wt_idx[INParams::inv_std_dev]);

  auto dim = incoming_derivative.getDim();
  unsigned int batch = dim.batch();
  unsigned int channels = dim.channel();
  unsigned int height = dim.height();
  unsigned int width = dim.width();
  unsigned int spatial_size = height * width;

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int c = 0; c < channels; ++c) {
      float inv_std = inv_std_dev.getValue(b, c, 0, 0);
      float g_val = gamma.getValue(0, c, 0, 0);

      float sum_dy = 0.0f;
      float sum_dy_x = 0.0f;

      for (unsigned int h = 0; h < height; ++h) {
        for (unsigned int w = 0; w < width; ++w) {
          float dy = incoming_derivative.getValue(b, c, h, w) * g_val;
          float x_hat = deviation.getValue(b, c, h, w) * inv_std;
          sum_dy += dy;
          sum_dy_x += dy * x_hat;
        }
      }

      for (unsigned int h = 0; h < height; ++h) {
        for (unsigned int w = 0; w < width; ++w) {
          float dy = incoming_derivative.getValue(b, c, h, w) * g_val;
          float x_hat = deviation.getValue(b, c, h, w) * inv_std;
          float dx =
            inv_std *
            (dy - (sum_dy + x_hat * sum_dy_x) / (float)spatial_size);
          outgoing_derivative.setValue(b, c, h, w, dx);
        }
      }
    }
  }
}

void InstanceNormalizationLayer::calcGradient(RunLayerContext &context) {
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &deviation = context.getTensor(wt_idx[INParams::deviation]);
  Tensor &inv_std_dev = context.getTensor(wt_idx[INParams::inv_std_dev]);
  Tensor &d_gamma = context.getWeightGrad(wt_idx[INParams::gamma]);
  Tensor &d_beta = context.getWeightGrad(wt_idx[INParams::beta]);

  d_gamma.setZero();
  d_beta.setZero();

  auto dim = incoming_derivative.getDim();
  unsigned int batch = dim.batch();
  unsigned int channels = dim.channel();
  unsigned int height = dim.height();
  unsigned int width = dim.width();

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int c = 0; c < channels; ++c) {
      float inv_std = inv_std_dev.getValue(b, c, 0, 0);
      for (unsigned int h = 0; h < height; ++h) {
        for (unsigned int w = 0; w < width; ++w) {
          float dy = incoming_derivative.getValue(b, c, h, w);
          float x_hat = deviation.getValue(b, c, h, w) * inv_std;
          d_gamma.setValue(0, c, 0, 0,
                           d_gamma.getValue(0, c, 0, 0) + dy * x_hat);
          d_beta.setValue(0, c, 0, 0, d_beta.getValue(0, c, 0, 0) + dy);
        }
      }
    }
  }
}

void InstanceNormalizationLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(in_props, method, this);
}

void InstanceNormalizationLayer::setBatch(RunLayerContext &context,
                                           unsigned int batch) {
  context.updateTensor(wt_idx[INParams::deviation], batch);
  context.updateTensor(wt_idx[INParams::variance], batch);
  context.updateTensor(wt_idx[INParams::inv_std_dev], batch);
}

} /* namespace nntrainer */

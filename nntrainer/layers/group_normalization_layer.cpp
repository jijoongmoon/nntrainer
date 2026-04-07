// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 NNTrainer contributors
 *
 * @file   group_normalization_layer.cpp
 * @date   15 Mar 2025
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1803.08494
 * @brief  Group Normalization Layer Class for Neural Network
 *
 */

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <layer_context.h>
#include <group_normalization_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum GNParams {
  gamma,
  beta,
  deviation,
  variance,
  inv_std_dev,
};

GroupNormalizationLayer::GroupNormalizationLayer() :
  Layer(),
  num_groups(1),
  gn_props(props::Epsilon(), props::GammaInitializer(),
           props::BetaInitializer(), props::WeightDecay(),
           props::BiasDecay()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void GroupNormalizationLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 1) {
    throw std::invalid_argument(
      "Only one input is allowed for group normalization layer");
  }

  auto const &input_dim = context.getInputDimensions()[0];
  context.setOutputDimensions({input_dim});

  unsigned int channels = input_dim.channel();
  NNTR_THROW_IF(num_groups == 0, std::invalid_argument)
    << "[GroupNorm] num_groups must be > 0";
  NNTR_THROW_IF(channels % num_groups != 0, std::invalid_argument)
    << "[GroupNorm] channels (" << channels
    << ") must be divisible by num_groups (" << num_groups << ")";

  auto gamma_initializer = std::get<props::GammaInitializer>(gn_props).get();
  auto beta_initializer = std::get<props::BetaInitializer>(gn_props).get();
  auto weight_decay = std::get<props::WeightDecay>(gn_props);
  auto bias_decay = std::get<props::BiasDecay>(gn_props);

  // gamma and beta are per-channel [1, C, 1, 1]
  TensorDim affine_dim(context.getFormat(), context.getWeightDataType());
  affine_dim.setTensorDim(1, channels);

  wt_idx[GNParams::gamma] = context.requestWeight(
    affine_dim, gamma_initializer, WeightRegularizer::NONE, 1.0f,
    weight_decay, "gamma", true);
  wt_idx[GNParams::beta] = context.requestWeight(
    affine_dim, beta_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
    "beta", true);

  // deviation: same as input [B, C, H, W]
  wt_idx[GNParams::deviation] =
    context.requestTensor(input_dim, "deviation", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);

  // variance and inv_std_dev: [B, num_groups, 1, 1]
  TensorDim group_dim(context.getFormat(), context.getWeightDataType());
  group_dim.batch(input_dim.batch());
  group_dim.channel(num_groups);

  wt_idx[GNParams::variance] =
    context.requestTensor(group_dim, "variance", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  wt_idx[GNParams::inv_std_dev] =
    context.requestTensor(group_dim, "inv_std_dev", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
}

void GroupNormalizationLayer::setProperty(
  const std::vector<std::string> &values) {
  std::vector<std::string> remain;
  for (auto &v : values) {
    auto pos = v.find('=');
    if (pos != std::string::npos) {
      std::string key = v.substr(0, pos);
      std::string val = v.substr(pos + 1);
      if (key == "num_groups") {
        num_groups = std::stoul(val);
        continue;
      }
    }
    remain.push_back(v);
  }
  auto leftover = loadProperties(remain, gn_props);
  NNTR_THROW_IF(!leftover.empty(), std::invalid_argument)
    << "[GroupNorm] Unknown properties count " + std::to_string(values.size());
}

void GroupNormalizationLayer::forwarding(RunLayerContext &context,
                                         bool training) {
  const float epsilon = std::get<props::Epsilon>(gn_props).get();

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &gamma = context.getWeight(wt_idx[GNParams::gamma]);
  Tensor &beta = context.getWeight(wt_idx[GNParams::beta]);

  Tensor &deviation = context.getTensor(wt_idx[GNParams::deviation]);
  Tensor &variance = context.getTensor(wt_idx[GNParams::variance]);
  Tensor &inv_std_dev = context.getTensor(wt_idx[GNParams::inv_std_dev]);

  auto dim = input.getDim();
  unsigned int batch = dim.batch();
  unsigned int channels = dim.channel();
  unsigned int height = dim.height();
  unsigned int width = dim.width();
  unsigned int channels_per_group = channels / num_groups;
  unsigned int group_size = channels_per_group * height * width;

  // Per-group mean and variance
  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int g = 0; g < num_groups; ++g) {
      float sum = 0.0f;
      unsigned int c_start = g * channels_per_group;

      for (unsigned int c = c_start; c < c_start + channels_per_group; ++c) {
        for (unsigned int h = 0; h < height; ++h) {
          for (unsigned int w = 0; w < width; ++w) {
            sum += input.getValue(b, c, h, w);
          }
        }
      }
      float mean = sum / group_size;

      float var_sum = 0.0f;
      for (unsigned int c = c_start; c < c_start + channels_per_group; ++c) {
        for (unsigned int h = 0; h < height; ++h) {
          for (unsigned int w = 0; w < width; ++w) {
            float val = input.getValue(b, c, h, w) - mean;
            deviation.setValue(b, c, h, w, val);
            var_sum += val * val;
          }
        }
      }
      float var = var_sum / group_size;
      float inv_std = 1.0f / std::sqrt(var + epsilon);

      variance.setValue(b, g, 0, 0, var);
      inv_std_dev.setValue(b, g, 0, 0, inv_std);

      for (unsigned int c = c_start; c < c_start + channels_per_group; ++c) {
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
}

void GroupNormalizationLayer::calcDerivative(RunLayerContext &context) {
  const float epsilon = std::get<props::Epsilon>(gn_props).get();

  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);

  const Tensor &gamma = context.getWeight(wt_idx[GNParams::gamma]);
  Tensor &deviation = context.getTensor(wt_idx[GNParams::deviation]);
  Tensor &inv_std_dev = context.getTensor(wt_idx[GNParams::inv_std_dev]);

  auto dim = incoming_derivative.getDim();
  unsigned int batch = dim.batch();
  unsigned int channels = dim.channel();
  unsigned int height = dim.height();
  unsigned int width = dim.width();
  unsigned int channels_per_group = channels / num_groups;
  unsigned int group_size = channels_per_group * height * width;

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int g = 0; g < num_groups; ++g) {
      float inv_std = inv_std_dev.getValue(b, g, 0, 0);
      unsigned int c_start = g * channels_per_group;

      float sum_dy = 0.0f;
      float sum_dy_x = 0.0f;

      for (unsigned int c = c_start; c < c_start + channels_per_group; ++c) {
        float g_val = gamma.getValue(0, c, 0, 0);
        for (unsigned int h = 0; h < height; ++h) {
          for (unsigned int w = 0; w < width; ++w) {
            float dy = incoming_derivative.getValue(b, c, h, w) * g_val;
            float x_hat = deviation.getValue(b, c, h, w) * inv_std;
            sum_dy += dy;
            sum_dy_x += dy * x_hat;
          }
        }
      }

      for (unsigned int c = c_start; c < c_start + channels_per_group; ++c) {
        float g_val = gamma.getValue(0, c, 0, 0);
        for (unsigned int h = 0; h < height; ++h) {
          for (unsigned int w = 0; w < width; ++w) {
            float dy = incoming_derivative.getValue(b, c, h, w) * g_val;
            float x_hat = deviation.getValue(b, c, h, w) * inv_std;
            float dx =
              inv_std *
              (dy - (sum_dy + x_hat * sum_dy_x) / (float)group_size);
            outgoing_derivative.setValue(b, c, h, w, dx);
          }
        }
      }
    }
  }
}

void GroupNormalizationLayer::calcGradient(RunLayerContext &context) {
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &deviation = context.getTensor(wt_idx[GNParams::deviation]);
  Tensor &inv_std_dev = context.getTensor(wt_idx[GNParams::inv_std_dev]);
  Tensor &d_gamma = context.getWeightGrad(wt_idx[GNParams::gamma]);
  Tensor &d_beta = context.getWeightGrad(wt_idx[GNParams::beta]);

  d_gamma.setZero();
  d_beta.setZero();

  auto dim = incoming_derivative.getDim();
  unsigned int batch = dim.batch();
  unsigned int channels = dim.channel();
  unsigned int height = dim.height();
  unsigned int width = dim.width();
  unsigned int channels_per_group = channels / num_groups;

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int g = 0; g < num_groups; ++g) {
      float inv_std = inv_std_dev.getValue(b, g, 0, 0);
      unsigned int c_start = g * channels_per_group;
      for (unsigned int c = c_start; c < c_start + channels_per_group; ++c) {
        for (unsigned int h = 0; h < height; ++h) {
          for (unsigned int w = 0; w < width; ++w) {
            float dy = incoming_derivative.getValue(b, c, h, w);
            float x_hat = deviation.getValue(b, c, h, w) * inv_std;
            d_gamma.setValue(0, c, 0, 0,
                             d_gamma.getValue(0, c, 0, 0) + dy * x_hat);
            d_beta.setValue(0, c, 0, 0,
                            d_beta.getValue(0, c, 0, 0) + dy);
          }
        }
      }
    }
  }
}

void GroupNormalizationLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(gn_props, method, this);
}

void GroupNormalizationLayer::setBatch(RunLayerContext &context,
                                       unsigned int batch) {
  context.updateTensor(wt_idx[GNParams::deviation], batch);
  context.updateTensor(wt_idx[GNParams::variance], batch);
  context.updateTensor(wt_idx[GNParams::inv_std_dev], batch);
}

} /* namespace nntrainer */

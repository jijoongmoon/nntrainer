// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   custom_rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cpu_backend.h>
#include <reshaped_rms_norm.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ReshapedRMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  feature_size = std::get<props::FeatureSize>(rms_props);

  NNTR_THROW_IF(dim[0].width() % feature_size != 0, std::invalid_argument)
    << "feature size must be a divisor of width";

  auto const &input_dim = dim[0];
  nntrainer::TensorDim::TensorType tensor_type(context.getFormat(),
                                               context.getWeightDataType());

  nntrainer::TensorDim gamma_dim(1, 1, 1, feature_size, tensor_type);
  auto gamma_initializer =
    std::get<props::RMS_NORM_GAMMA_INIT>(rms_props).get();
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, gamma_initializer, nntrainer::WeightRegularizer::NONE, 1.0f,
    0.0f, "gamma", true);

  /**
   * Reshaped dimensions: the input (batch, channel, height, width) is reshaped
   * to (batch, channel, height * (width / feature_size), feature_size).
   * The reduced dim collapses the last axis (feature_size) to 1.
   */
  unsigned int reshaped_height =
    input_dim.height() * (input_dim.width() / feature_size);
  nntrainer::TensorDim reshaped_dim(input_dim.batch(), input_dim.channel(),
                                    reshaped_height, feature_size, tensor_type);
  nntrainer::TensorDim reduced_dim(input_dim.batch(), input_dim.channel(),
                                   reshaped_height, 1, tensor_type);

  /** caches variance = mean(x^2) + epsilon */
  wt_idx[RMSParams::variance] =
    context.requestTensor(reduced_dim, "variance", nntrainer::Initializer::NONE,
                          false, nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  /** caches the inverse RMS (1 / sqrt(variance)) */
  wt_idx[RMSParams::inv_std_dev] = context.requestTensor(
    reduced_dim, "inv_std_dev", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::ITERATION_LIFESPAN);

  /** temporary tensor (reshaped input size) for calcDerivative */
  wt_idx[RMSParams::temp_origin_size] = context.requestTensor(
    reshaped_dim, "temp_origin_size", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::CALC_DERIV_LIFESPAN);
  /** temporary tensor (reduced size) for calcDerivative */
  wt_idx[RMSParams::temp_reduced_size] = context.requestTensor(
    reduced_dim, "temp_reduced_size", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::CALC_DERIV_LIFESPAN);
}

void ReshapedRMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  nntrainer::Tensor &variance = context.getTensor(wt_idx[RMSParams::variance]);
  nntrainer::Tensor &inv_std_dev =
    context.getTensor(wt_idx[RMSParams::inv_std_dev]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  /** Reshape to (batch, channel, height * (width / feature_size), feature_size)
   */
  ml::train::TensorDim reshaped_dim = in_dim;
  reshaped_dim.width(feature_size);
  reshaped_dim.height(in_dim.height() * (in_dim.width() / feature_size));

  in.reshape(reshaped_dim);
  out.reshape(reshaped_dim);

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    /** variance = mean(x^2) + epsilon */
    in.multiply(in, out);
    out.average(3, variance);
    variance.add_i(epsilon);

    /** inv_std_dev = 1 / sqrt(variance) */
    variance.pow(-0.5f, inv_std_dev);

    /** output = x * inv_std_dev * gamma */
    in.multiply(inv_std_dev, out);
  } else {
    throw std::invalid_argument(
      "Error: not yet implemented for this data type");
  }
  out.multiply_i(gamma);

  /** Reshape back to original dimensions */
  in.reshape(in_dim);
  out.reshape(out_dim);
}

void ReshapedRMSNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  unsigned int _from = from;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  // set reshaped dim to (1, 1, -1, feature_size)
  ml::train::TensorDim step_reshaped_dim = in_step_dim;

  step_reshaped_dim.width(feature_size);
  step_reshaped_dim.height(in_step_dim.height() *
                           (in_dim.width() / feature_size));

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    // reshape in_step
    // reshape out_step
    in_step.reshape(step_reshaped_dim);
    out_step.reshape(step_reshaped_dim);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      ///@todo rms_norm_wrt_width_something() should be refactored to
      /// nntrainer::Tensor operation.
#ifdef ENABLE_FP16
      nntrainer::rms_norm_wrt_width_fp16_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);
#else
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);
#endif
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }
    out_step.multiply_i(gamma);

    // reshape again out_step
    out_step.reshape(out_step_dim);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }
}

void ReshapedRMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void ReshapedRMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  const bool trainable = context.getTrainable();

  nntrainer::Tensor &outgoing_derivative =
    context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);

  const nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  nntrainer::Tensor &variance = context.getTensor(wt_idx[RMSParams::variance]);
  nntrainer::Tensor &inv_std_dev =
    context.getTensor(wt_idx[RMSParams::inv_std_dev]);

  nntrainer::Tensor &temp_origin_size =
    context.getTensor(wt_idx[RMSParams::temp_origin_size]);
  nntrainer::Tensor &temp_reduced_size =
    context.getTensor(wt_idx[RMSParams::temp_reduced_size]);

  ml::train::TensorDim in_dim = input.getDim();
  ml::train::TensorDim out_dim = outgoing_derivative.getDim();

  /** Reshape to (batch, channel, height * (width / feature_size), feature_size)
   */
  ml::train::TensorDim reshaped_dim = in_dim;
  reshaped_dim.width(feature_size);
  reshaped_dim.height(in_dim.height() * (in_dim.width() / feature_size));

  /** We need non-const access for reshape, use const_cast as reshape does not
   * modify data */
  const_cast<nntrainer::Tensor &>(input).reshape(reshaped_dim);
  const_cast<nntrainer::Tensor &>(incoming_derivative).reshape(reshaped_dim);
  outgoing_derivative.reshape(reshaped_dim);

  /**
   * RMS Norm backward pass (same as rms_norm but on reshaped tensors):
   *   y = gamma * x * inv_rms
   *   where inv_rms = 1 / sqrt(mean(x^2) + eps)
   *
   *   dL/dx = inv_rms * gamma * (dL/dy - x * mean(dL/dy * x) / variance)
   *   dL/dgamma = sum over remain_axes of (dL/dy * x * inv_rms)
   */

  /** temp_origin = dL/dy * x */
  incoming_derivative.multiply(input, temp_origin_size);

  /** temp_reduced = mean(dL/dy * x) along width axis (feature_size dim) */
  temp_origin_size.average({3}, temp_reduced_size);

  /** temp_reduced = mean(dL/dy * x) / variance */
  temp_reduced_size.divide_i(variance);

  if (trainable) {
    /** d_gamma = sum of (dL/dy * x * inv_rms) along remain axes (batch,
     * channel, height) */
    nntrainer::Tensor &d_gamma =
      context.getWeightGrad(wt_idx[RMSParams::gamma]);
    temp_origin_size.multiply_i(inv_std_dev);
    temp_origin_size.sum({0, 1, 2}, d_gamma);
  }

  /** outgoing = dL/dy - x * mean(dL/dy * x) / variance */
  input.multiply(temp_reduced_size, outgoing_derivative);
  incoming_derivative.subtract(outgoing_derivative, outgoing_derivative);

  /** outgoing *= inv_rms * gamma */
  inv_std_dev.multiply_i(gamma);
  outgoing_derivative.multiply_i(inv_std_dev);

  /** Reshape back to original dimensions */
  const_cast<nntrainer::Tensor &>(input).reshape(in_dim);
  const_cast<nntrainer::Tensor &>(incoming_derivative).reshape(out_dim);
  outgoing_derivative.reshape(out_dim);
}

void ReshapedRMSNormLayer::calcGradient(nntrainer::RunLayerContext &context) {
  /** d_gamma is already computed in calcDerivative */
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new ReshapedRMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   activation_layer.cpp
 * @date   17 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Layer Class for Neural Network
 *
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

#include <activation_layer.h>
#include <blas_interface.h>
#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer_internal.h>
#include <parse_util.h>
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {

const std::string ActivationLayer::type = "activation";

/**
 * @brief     Initialize the layer
 *
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int ActivationLayer::initialize(Manager &manager) {

  output_dim = input_dim;

  return ML_ERROR_NONE;
}

void ActivationLayer::forwarding(bool training) {
  Tensor &hidden_ = net_hidden[0]->getVariableRef();
  /// @note @a _act_fn is expected to work out of place and not modify @a input
  acti_func.run_fn(net_input[0]->getVariableRef(), hidden_);
}

void ActivationLayer::calcDerivative() {
  Tensor &deriv = net_hidden[0]->getGradientRef();
  Tensor &ret = net_input[0]->getGradientRef();
  Tensor &in = net_hidden[0]->getVariableRef();

  ret = acti_func.run_prime_fn(in, ret, deriv);
}

int ActivationLayer::setActivation(
  std::function<Tensor &(Tensor const &, Tensor &)> const &activation_fn,
  std::function<Tensor &(Tensor &, Tensor &, Tensor const &)> const
    &activation_prime_fn) {
  acti_func.setActivation(activation_fn, activation_prime_fn);

  return ML_ERROR_NONE;
}

int ActivationLayer::setActivation(
  std::function<Tensor &(Tensor const &, Tensor &)> const &activation_fn,
  std::function<Tensor &(Tensor &, Tensor &)> const &activation_prime_fn) {

  acti_func.setActivation(activation_fn, activation_prime_fn);

  return ML_ERROR_NONE;
}

int ActivationLayer::setActivation(
  std::function<float(float const)> const &activation_fn,
  std::function<float(float const)> const &activation_prime_fn) {

  acti_func.setActivation(activation_fn, activation_prime_fn);

  return ML_ERROR_NONE;
}

/**
 * @brief setActivation by preset ActivationType
 *
 * @param[in] ActivationType ActivationType ActivationType to be set
 */
void ActivationLayer::setActivation(ActivationType acti_type) {
  Layer::setActivation(acti_type);

  switch (acti_type) {
  case ActivationType::ACT_TANH:
    acti_func.setActiFunc(ActivationType::ACT_TANH);
    break;
  case ActivationType::ACT_SIGMOID:
    acti_func.setActiFunc(ActivationType::ACT_SIGMOID);
    break;
  case ActivationType::ACT_SOFTMAX:
    acti_func.setActiFunc(ActivationType::ACT_SOFTMAX);
    break;
  case ActivationType::ACT_RELU:
    acti_func.setActiFunc(ActivationType::ACT_RELU);
    break;
  case ActivationType::ACT_NONE:
    acti_func.setActiFunc(ActivationType::ACT_NONE);
    break;
  case ActivationType::ACT_UNKNOWN:
  default:
    throw std::runtime_error("Error: Not Supported Activation Type");
  }
}

}; // namespace nntrainer

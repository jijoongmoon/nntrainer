/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	input_layer.cpp
 * @date	14 May 2020
 * @brief	This is Input Layer Class for Neural Network
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <input_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

InputLayer::InputLayer() :
  Layer(),
  input_props(props::Normalization(), props::Standardization(),
              props::TensorDtype()) {}

namespace {

/**
 * @brief Parse a tensor_dtype property value into a DataType enum, falling
 *        back to FP32 (only used when the string is non-empty).
 */
ml::train::TensorDim::DataType parse_dtype(const std::string &s) {
  using DT = ml::train::TensorDim::DataType;
  if (s == "FP32" || s == "fp32")
    return DT::FP32;
  if (s == "FP16" || s == "fp16")
    return DT::FP16;
  if (s == "UINT8" || s == "uint8")
    return DT::UINT8;
  if (s == "UINT16" || s == "uint16")
    return DT::UINT16;
  if (s == "UINT32" || s == "uint32")
    return DT::UINT32;
  if (s == "QINT8" || s == "qint8")
    return DT::QINT8;
  if (s == "QINT16" || s == "qint16")
    return DT::QINT16;
  if (s == "Q4_K" || s == "q4_k")
    return DT::Q4_K;
  if (s == "Q6_K" || s == "q6_k")
    return DT::Q6_K;
  if (s == "Q4_0" || s == "q4_0")
    return DT::Q4_0;
  throw std::invalid_argument("InputLayer: unknown tensor_dtype value '" + s +
                              "'");
}

} // namespace

void InputLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, input_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[InputLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void InputLayer::forwarding(RunLayerContext &context, bool training) {

  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  std::unique_ptr<nntrainer::Quantizer> quantizer;
  if (!context.getInPlace()) {
    Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
    ///@note: The following code simply copies incoming values (fp32) to the
    // input tensor. Supported types include QINT4, QINT8, UINT8, UINT16,
    // UINT32, FP16, and FP32.
    hidden_.copyData(input_);
  }

  if (std::get<props::Normalization>(input_props))
    hidden_.normalization_i();
  if (std::get<props::Standardization>(input_props))
    hidden_.standardization_i();
}

void InputLayer::calcDerivative(RunLayerContext &context) {
  throw exception::not_supported(
    "calcDerivative for input layer is not supported");
}

void InputLayer::exportTo(Exporter &exporter,
                          const ml::train::ExportMethods &method) const {
  exporter.saveResult(input_props, method, this);
}

void InputLayer::finalize(InitLayerContext &context) {

  // If the user pinned a per-input dtype with `tensor_dtype=...`, honour that
  // (typed inputs — KV caches, INT-coded tokens, etc.). Otherwise fall back
  // to the model's global activation dtype, matching legacy behaviour.
  const auto &dtype_prop = std::get<props::TensorDtype>(input_props);
  const bool has_explicit_dtype = !dtype_prop.empty();
  const ml::train::TensorDim::DataType output_dtype =
    has_explicit_dtype ? parse_dtype(dtype_prop.get())
                       : context.getActivationDataType();

  std::vector<TensorDim> output_dims = context.getInputDimensions();
  for (auto &d : output_dims) {
    d.setDataType(output_dtype);
  }

  context.setOutputDimensions(output_dims);
  is_inplace = true;
  if (output_dtype != ml::train::TensorDim::DataType::FP32)
    is_inplace = false;
}

void InputLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

} /* namespace nntrainer */

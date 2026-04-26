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
 * @file	input_layer.h
 * @date	14 May 2020
 * @brief	This is Input Layer Class of Neural Network
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __INPUT_LAYER_H__
#define __INPUT_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

namespace props {

/**
 * @brief Per-input-layer override of the output tensor data type.
 *
 * Without this, InputLayer::finalize() coerces the output dtype to the
 * model's global activation dtype (set via "model_tensor_type"), which makes
 * it impossible for, e.g., an FP16 KV-cache or INT-coded tokens placeholder
 * to coexist with FP32 activations.
 *
 * Accepted values: "FP32", "FP16", "UINT8", "UINT16", "UINT32", "QINT8" ...
 *                  (anything ml::train::TensorDim::DataType supports as a
 *                  string token via str_converter).
 *
 * Mainly used by the symbolic Tensor → input layer auto-creation path in
 * tensor_api so a Tensor declared with FP16 dim wires up a matching FP16
 * input layer at compile time.
 */
class TensorDtype final : public nntrainer::Property<std::string> {
public:
  /**
   * @brief Default-constructed TensorDtype is an UNSET property
   *        (`empty() == true`). Use the explicit-value constructor — or set
   *        via `setProperty({"tensor_dtype=FP16"})` — to pin the dtype.
   */
  TensorDtype() = default;
  explicit TensorDtype(const std::string &value) { set(value); }
  static constexpr const char *key = "tensor_dtype";
  using prop_tag = nntrainer::str_prop_tag;
};

} // namespace props

/**
 * @class   Input Layer
 * @note    input layers requires to be only single input, consider making the
 * class deal with multiple inputs
 * @brief   Just Handle the Input of Network
 */
class InputLayer : public Layer {
public:
  /**
   * @brief     Constructor of InputLayer
   */
  InputLayer();

  /**
   * @brief     Destructor of InputLayer
   */
  ~InputLayer() = default;

  /**
   *  @brief  Move constructor of Pooling 2D Layer.
   *  @param[in] Input &&
   */
  InputLayer(InputLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs InputLayer to be moved.
   */
  InputLayer &operator=(InputLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  InPlaceType initializeInPlace() final {
    is_inplace = true;
    return InPlaceType::NON_RESTRICTING;
  }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return InputLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  static constexpr const char *type = "input";

private:
  std::tuple<props::Normalization, props::Standardization, props::TensorDtype>
    input_props;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __INPUT_LAYER_H__ */

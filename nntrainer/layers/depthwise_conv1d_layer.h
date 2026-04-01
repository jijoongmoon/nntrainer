// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   depthwise_conv1d_layer.h
 * @date   01 Apr 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Depthwise Convolution 1D Layer Class for Neural Network
 *
 */

#ifndef __DEPTHWISE_CONV1D_LAYER_H_
#define __DEPTHWISE_CONV1D_LAYER_H_
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   Depthwise Convolution 1D Layer
 * @brief   Depthwise Convolution 1D Layer
 *
 * Each input channel is convolved independently with its own filter
 * (groups == in_channels). Output channels == input channels.
 * Weight shape: [channels, 1, 1, kernel_size]
 */
class DepthwiseConv1DLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Depthwise Conv 1D Layer
   */
  DepthwiseConv1DLayer();

  /**
   * @brief     Destructor of Depthwise Conv 1D Layer
   */
  ~DepthwiseConv1DLayer() = default;

  /**
   *  @brief  Move constructor of Depthwise Conv 1D Layer.
   *  @param[in] DepthwiseConv1DLayer &&
   */
  DepthwiseConv1DLayer(DepthwiseConv1DLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs DepthwiseConv1DLayer to be moved.
   */
  DepthwiseConv1DLayer &operator=(DepthwiseConv1DLayer &&rhs) = default;

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
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return DepthwiseConv1DLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "depthwiseconv1d";

private:
  std::tuple<props::FilterSize, props::KernelSize, props::Stride,
             props::Padding1D, props::Dilation>
    conv_props;

  std::array<unsigned int, 2> padding; /**< computed padding [left, right] */
  std::array<unsigned int, 2> wt_idx;  /**< indices of the weights */
  unsigned int col_buf_idx;            /**< index of the im2col buffer tensor */

  enum DepthwiseConvParams { weight, bias };
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __DEPTHWISE_CONV1D_LAYER_H_ */

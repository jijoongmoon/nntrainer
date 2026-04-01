// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   depthwise_conv1d_layer.cpp
 * @date   01 Apr 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Depthwise Convolution 1D Layer Class for Neural Network
 *
 * Depthwise convolution applies a single filter per input channel.
 * Equivalent to PyTorch nn.Conv1d with groups == in_channels.
 * Weight shape: [channels, 1, 1, kernel_size]  (batch=C, channel=1, h=1, w=K)
 *
 */
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#include <depthwise_conv1d_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

DepthwiseConv1DLayer::DepthwiseConv1DLayer() :
  LayerImpl(),
  conv_props(props::FilterSize(), props::KernelSize(), props::Stride(),
             props::Padding1D(), props::Dilation()),
  padding({0, 0}) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void DepthwiseConv1DLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Depthwise Convolution 1D layer takes only one input";

  const TensorDim &in_dim = context.getInputDimensions()[0];

  NNTR_THROW_IF(in_dim.height() != 1, std::invalid_argument)
    << "DepthwiseConv1D layer requires input with height 1";

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  unsigned int channels = in_dim.channel();
  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();
  unsigned int dilation = std::get<props::Dilation>(conv_props).get();

  // For depthwise, filters must equal input channels
  unsigned int filter_size = std::get<props::FilterSize>(conv_props).get();
  NNTR_THROW_IF(filter_size != channels, std::invalid_argument)
    << "DepthwiseConv1D: filters (" << filter_size
    << ") must equal input channels (" << channels << ")";

  padding = std::get<props::Padding1D>(conv_props)
              .compute(in_dim, kernel_size, stride, dilation);

  auto in_t_type = in_dim.getTensorType();
  in_t_type.data_type = context.getWeightDataType();

  // Weight: [channels, 1, 1, kernel_size] — one filter per channel
  TensorDim kernel_dim(channels, 1, 1, kernel_size, in_t_type);

  // Bias: [1, channels, 1, 1]
  TensorDim bias_dim(1, channels, 1, 1, in_t_type);

  wt_idx[DepthwiseConvParams::weight] = context.requestWeight(
    kernel_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "filter", true, 0);

  if (disable_bias.empty() || disable_bias.get() == false) {
    wt_idx[DepthwiseConvParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true, 0);
  }

  // Compute output dimension
  unsigned int eff_in_width = in_dim.width() + padding[0] + padding[1];
  unsigned int eff_k_width = (kernel_size - 1) * dilation + 1;

  NNTR_THROW_IF(eff_in_width < eff_k_width, std::invalid_argument)
    << "DepthwiseConv1D: input width + padding is smaller than effective "
       "kernel size";

  unsigned int out_width = (eff_in_width - eff_k_width) / stride + 1;

  TensorDim out_dim;
  out_dim.batch(in_dim.batch());
  out_dim.channel(channels);
  out_dim.height(1);
  out_dim.width(out_width);
  out_dim.setTensorType(in_dim.getTensorType());

  context.setOutputDimensions({out_dim});
}

void DepthwiseConv1DLayer::forwarding(RunLayerContext &context, bool training) {
  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();
  unsigned int dilation = std::get<props::Dilation>(conv_props).get();

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &filter = context.getWeight(wt_idx[DepthwiseConvParams::weight]);

  const TensorDim &in_dim = input_.getDim();
  const TensorDim &out_dim = hidden_.getDim();

  unsigned int batch = in_dim.batch();
  unsigned int channels = in_dim.channel();
  unsigned int in_width = in_dim.width();
  unsigned int out_width = out_dim.width();
  unsigned int pad_left = padding[0];

  hidden_.setZero();

  /**
   * Depthwise 1D convolution: each channel is convolved independently.
   *
   * For each batch b, channel c, output position ow:
   *   output[b][c][0][ow] = sum_{k=0}^{kernel_size-1}
   *     input[b][c][0][ow*stride + k*dilation - pad_left] * filter[c][0][0][k]
   */
  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int c = 0; c < channels; ++c) {
      for (unsigned int ow = 0; ow < out_width; ++ow) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < kernel_size; ++k) {
          int iw = static_cast<int>(ow * stride + k * dilation) -
                   static_cast<int>(pad_left);
          if (iw >= 0 && iw < static_cast<int>(in_width)) {
            sum += input_.getValue(b, c, 0, iw) * filter.getValue(c, 0, 0, k);
          }
        }
        hidden_.setValue(b, c, 0, ow, sum);
      }
    }
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(wt_idx[DepthwiseConvParams::bias]);
    int status = hidden_.add_i(bias);
    if (status != ML_ERROR_NONE) {
      throw std::invalid_argument(
        "[DepthwiseConv1D] adding bias failed");
    }
  }
}

void DepthwiseConv1DLayer::calcDerivative(RunLayerContext &context) {
  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();
  unsigned int dilation = std::get<props::Dilation>(conv_props).get();

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &filter = context.getWeight(wt_idx[DepthwiseConvParams::weight]);

  const TensorDim &deriv_dim = derivative.getDim();
  const TensorDim &in_deriv_dim = input_derivative.getDim();

  unsigned int batch = deriv_dim.batch();
  unsigned int channels = deriv_dim.channel();
  unsigned int out_width = deriv_dim.width();
  unsigned int in_width = in_deriv_dim.width();
  unsigned int pad_left = padding[0];

  input_derivative.setZero();

  /**
   * Backprop to input: for each output gradient, scatter to input positions.
   *
   * d_input[b][c][0][iw] += sum_{ow where iw is used}
   *   d_output[b][c][0][ow] * filter[c][0][0][k]
   */
  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int c = 0; c < channels; ++c) {
      for (unsigned int ow = 0; ow < out_width; ++ow) {
        float d_out = derivative.getValue(b, c, 0, ow);
        for (unsigned int k = 0; k < kernel_size; ++k) {
          int iw = static_cast<int>(ow * stride + k * dilation) -
                   static_cast<int>(pad_left);
          if (iw >= 0 && iw < static_cast<int>(in_width)) {
            float cur = input_derivative.getValue(b, c, 0, iw);
            input_derivative.setValue(b, c, 0, iw,
                                     cur + d_out * filter.getValue(c, 0, 0, k));
          }
        }
      }
    }
  }
}

void DepthwiseConv1DLayer::calcGradient(RunLayerContext &context) {
  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();
  unsigned int dilation = std::get<props::Dilation>(conv_props).get();

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &delK = context.getWeightGrad(wt_idx[DepthwiseConvParams::weight]);
  delK.setZero();

  const TensorDim &deriv_dim = derivative.getDim();
  const TensorDim &in_dim = input_.getDim();

  unsigned int batch = deriv_dim.batch();
  unsigned int channels = deriv_dim.channel();
  unsigned int out_width = deriv_dim.width();
  unsigned int in_width = in_dim.width();
  unsigned int pad_left = padding[0];

  /**
   * Weight gradient: correlate input with output gradient.
   *
   * d_filter[c][0][0][k] += sum_{b, ow}
   *   d_output[b][c][0][ow] * input[b][c][0][ow*stride + k*dilation - pad_left]
   */
  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int c = 0; c < channels; ++c) {
      for (unsigned int ow = 0; ow < out_width; ++ow) {
        float d_out = derivative.getValue(b, c, 0, ow);
        for (unsigned int k = 0; k < kernel_size; ++k) {
          int iw = static_cast<int>(ow * stride + k * dilation) -
                   static_cast<int>(pad_left);
          if (iw >= 0 && iw < static_cast<int>(in_width)) {
            float cur = delK.getValue(c, 0, 0, k);
            delK.setValue(c, 0, 0, k,
                          cur + d_out * input_.getValue(b, c, 0, iw));
          }
        }
      }
    }
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &delBias =
      context.getWeightGrad(wt_idx[DepthwiseConvParams::bias]);
    delBias.setZero();
    /**
     * Bias gradient: sum output gradient over batch and spatial dims.
     * d_bias[c] = sum_{b, ow} d_output[b][c][0][ow]
     */
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (unsigned int ow = 0; ow < out_width; ++ow) {
          sum += derivative.getValue(b, c, 0, ow);
        }
        float cur = delBias.getValue(0, c, 0, 0);
        delBias.setValue(0, c, 0, 0, cur + sum);
      }
    }
  }
}

void DepthwiseConv1DLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void DepthwiseConv1DLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

} // namespace nntrainer

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   depthwise_conv1d_layer.cpp
 * @date   22 March 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Depthwise Convolution 1D Layer Class for Neural Network
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

enum DepthwiseConv1DParams { weight, bias };

DepthwiseConv1DLayer::DepthwiseConv1DLayer() :
  LayerImpl(),
  padding({0, 0}),
  conv_props(props::KernelSize(), props::Stride(), props::Padding1D(),
             props::Dilation()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void DepthwiseConv1DLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Depthwise Convolution 1D layer takes only one input";

  NNTR_THROW_IF(context.getInputDimensions()[SINGLE_INOUT_IDX].height() != 1,
                std::invalid_argument)
    << "DepthwiseConv1D layer requires input with height 1";

  const TensorDim &in_dim = context.getInputDimensions()[0];
  unsigned int channels = in_dim.channel();

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

  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();
  unsigned int dilation = std::get<props::Dilation>(conv_props).get();

  auto in_t_type = in_dim.getTensorType();
  in_t_type.data_type = context.getWeightDataType();

  padding = std::get<props::Padding1D>(conv_props)
              .compute(in_dim, kernel_size, stride, dilation);

  /** Weight shape: (channels, kernel_size)
   *  Each channel has its own independent kernel */
  TensorDim weight_dim(channels, kernel_size);
  weight_dim.setTensorType(in_t_type);

  TensorDim bias_dim(1, channels, 1, 1, in_t_type);

  wt_idx[DepthwiseConv1DParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "filter", true, 0);

  if (disable_bias.empty() || disable_bias.get() == false) {
    wt_idx[DepthwiseConv1DParams::bias] =
      context.requestWeight(bias_dim, bias_initializer,
                            WeightRegularizer::NONE, 1.0f, bias_decay, "bias",
                            true, 0);
  }

  unsigned int eff_in_width = in_dim.width() + padding[0] + padding[1];
  unsigned int eff_k_width = (kernel_size - 1) * dilation + 1;

  NNTR_THROW_IF(eff_in_width < eff_k_width, std::invalid_argument)
    << "Failed to initialize: in width + padding is smaller than effective "
       "kernel";

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
  Tensor &filter_kernel =
    context.getWeight(wt_idx[DepthwiseConv1DParams::weight]);

  const TensorDim &in_dim = input_.getDim();
  const TensorDim &out_dim = hidden_.getDim();
  unsigned int batch = in_dim.batch();
  unsigned int channels = in_dim.channel();
  int in_width = in_dim.width();
  unsigned int out_width = out_dim.width();
  unsigned int pad_left = padding[0];

  hidden_.setZero();

  auto compute_forward = [&]<typename T>(T) {
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int c = 0; c < channels; ++c) {
        for (unsigned int ow = 0; ow < out_width; ++ow) {
          T sum = static_cast<T>(0);
          int base_w = static_cast<int>(ow * stride) - static_cast<int>(pad_left);
          for (unsigned int k = 0; k < kernel_size; ++k) {
            int iw = base_w + static_cast<int>(k * dilation);
            if (iw >= 0 && iw < in_width) {
              sum += input_.getValue<T>(b, c, 0, iw) *
                     filter_kernel.getValue<T>(0, 0, c, k);
            }
          }
          hidden_.setValue(b, c, 0, ow, sum);
        }
      }
    }
  };

  if (input_.getDataType() == nntrainer::Tdatatype::FP32) {
    compute_forward(float{});
  }
#ifdef ENABLE_FP16
  else if (input_.getDataType() == nntrainer::Tdatatype::FP16) {
    compute_forward(_FP16{});
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias_kernel =
      context.getWeight(wt_idx[DepthwiseConv1DParams::bias]);
    int status = hidden_.add_i(bias_kernel);
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
  Tensor &filter_kernel =
    context.getWeight(wt_idx[DepthwiseConv1DParams::weight]);

  const TensorDim &deriv_dim = derivative.getDim();
  unsigned int batch = deriv_dim.batch();
  unsigned int channels = deriv_dim.channel();
  unsigned int out_width = deriv_dim.width();
  int in_width = input_derivative.getDim().width();
  unsigned int pad_left = padding[0];

  input_derivative.setZero();

  auto compute_deriv = [&]<typename T>(T) {
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int c = 0; c < channels; ++c) {
        for (unsigned int ow = 0; ow < out_width; ++ow) {
          T grad_out = derivative.getValue<T>(b, c, 0, ow);
          int base_w = static_cast<int>(ow * stride) - static_cast<int>(pad_left);
          for (unsigned int k = 0; k < kernel_size; ++k) {
            int iw = base_w + static_cast<int>(k * dilation);
            if (iw >= 0 && iw < in_width) {
              T *addr = input_derivative.getAddress<T>(b, c, 0, iw);
              *addr += grad_out * filter_kernel.getValue<T>(0, 0, c, k);
            }
          }
        }
      }
    }
  };

  if (derivative.getDataType() == nntrainer::Tdatatype::FP32) {
    compute_deriv(float{});
  }
#ifdef ENABLE_FP16
  else if (derivative.getDataType() == nntrainer::Tdatatype::FP16) {
    compute_deriv(_FP16{});
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }
}

void DepthwiseConv1DLayer::calcGradient(RunLayerContext &context) {
  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();
  unsigned int dilation = std::get<props::Dilation>(conv_props).get();

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &delK = context.getWeightGrad(wt_idx[DepthwiseConv1DParams::weight]);

  const TensorDim &deriv_dim = derivative.getDim();
  unsigned int batch = deriv_dim.batch();
  unsigned int channels = deriv_dim.channel();
  unsigned int out_width = deriv_dim.width();
  int in_width = input_.getDim().width();
  unsigned int pad_left = padding[0];

  delK.setZero();

  auto compute_grad = [&]<typename T>(T) {
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int c = 0; c < channels; ++c) {
        for (unsigned int ow = 0; ow < out_width; ++ow) {
          T grad_out = derivative.getValue<T>(b, c, 0, ow);
          int base_w = static_cast<int>(ow * stride) - static_cast<int>(pad_left);
          for (unsigned int k = 0; k < kernel_size; ++k) {
            int iw = base_w + static_cast<int>(k * dilation);
            if (iw >= 0 && iw < in_width) {
              T *addr = delK.getAddress<T>(0, 0, c, k);
              *addr += grad_out * input_.getValue<T>(b, c, 0, iw);
            }
          }
        }
      }
    }
  };

  if (derivative.getDataType() == nntrainer::Tdatatype::FP32) {
    compute_grad(float{});
  }
#ifdef ENABLE_FP16
  else if (derivative.getDataType() == nntrainer::Tdatatype::FP16) {
    compute_grad(_FP16{});
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &delBias =
      context.getWeightGrad(wt_idx[DepthwiseConv1DParams::bias]);
    delBias.setZero();
    /** Sum over batch, height(1), width dimensions to get bias gradient
     *  derivative shape: (batch, channels, 1, out_width)
     *  delBias shape: (1, 1, 1, channels) */
    derivative.sum({0, 2, 3}, delBias);
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

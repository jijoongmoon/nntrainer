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
 * Uses im2col approach: input patches are unrolled into a column matrix once,
 * then per-channel dot products compute the convolution efficiently.
 * The im2col buffer is pre-allocated in finalize() to avoid repeated allocation.
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

/**
 * @brief Fill im2col buffer for depthwise 1D convolution.
 *
 * For each channel c and kernel position k, copies the input value at the
 * corresponding strided+dilated position into col[c*K + k][ow].
 * Out-of-bounds positions (due to padding) are left as zero.
 *
 * @param[in]  in       Input tensor slice for one batch [1, C, 1, W]
 * @param[out] col      Column buffer [C*K, OW], must be pre-zeroed
 * @param[in]  channels Number of input channels
 * @param[in]  in_width Input width
 * @param[in]  out_width Output width
 * @param[in]  kernel_size Kernel size K
 * @param[in]  stride Convolution stride
 * @param[in]  dilation Kernel dilation
 * @param[in]  pad_left Left padding
 */
static void depthwise_im2col_1d(const Tensor &in, Tensor &col,
                                unsigned int channels, unsigned int in_width,
                                unsigned int out_width,
                                unsigned int kernel_size, unsigned int stride,
                                unsigned int dilation, unsigned int pad_left) {
  for (unsigned int c = 0; c < channels; ++c) {
    for (unsigned int k = 0; k < kernel_size; ++k) {
      unsigned int row = c * kernel_size + k;
      for (unsigned int ow = 0; ow < out_width; ++ow) {
        int iw = static_cast<int>(ow * stride + k * dilation) -
                 static_cast<int>(pad_left);
        if (iw >= 0 && iw < static_cast<int>(in_width)) {
          col.setValue(0, 0, row, ow, in.getValue(0, c, 0, iw));
        }
      }
    }
  }
}

/**
 * @brief col2im for depthwise 1D convolution (backprop to input).
 *
 * Scatters column values back to the input derivative positions.
 *
 * @param[in]  col        Column buffer [C*K, OW]
 * @param[out] in_deriv   Input derivative [1, C, 1, W], must be pre-zeroed
 * @param[in]  channels   Number of channels
 * @param[in]  in_width   Input width
 * @param[in]  out_width  Output width
 * @param[in]  kernel_size Kernel size K
 * @param[in]  stride     Convolution stride
 * @param[in]  dilation   Kernel dilation
 * @param[in]  pad_left   Left padding
 */
static void depthwise_col2im_1d(const Tensor &col, Tensor &in_deriv,
                                unsigned int channels, unsigned int in_width,
                                unsigned int out_width,
                                unsigned int kernel_size, unsigned int stride,
                                unsigned int dilation, unsigned int pad_left) {
  for (unsigned int c = 0; c < channels; ++c) {
    for (unsigned int k = 0; k < kernel_size; ++k) {
      unsigned int row = c * kernel_size + k;
      for (unsigned int ow = 0; ow < out_width; ++ow) {
        int iw = static_cast<int>(ow * stride + k * dilation) -
                 static_cast<int>(pad_left);
        if (iw >= 0 && iw < static_cast<int>(in_width)) {
          float cur = in_deriv.getValue(0, c, 0, iw);
          in_deriv.setValue(0, c, 0, iw, cur + col.getValue(0, 0, row, ow));
        }
      }
    }
  }
}

DepthwiseConv1DLayer::DepthwiseConv1DLayer() :
  LayerImpl(),
  conv_props(props::FilterSize(), props::KernelSize(), props::Stride(),
             props::Padding1D(), props::Dilation()),
  padding({0, 0}),
  weight_col_expanded(false) {
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

  // Pre-allocate im2col buffer: [1, 1, C*K, OW]
  // Reused across forward passes to avoid repeated allocation.
  TensorDim col_dim(1, 1, channels * kernel_size, out_width,
                    in_dim.getTensorType());
  col_buf_idx = context.requestTensor(col_dim, "im2col_buf",
                                      Initializer::NONE, false,
                                      TensorLifespan::ITERATION_LIFESPAN);

  // Pre-allocate weight im2col buffer: [1, 1, C*K, OW]
  // Weight[C,1,1,K] is expanded (tiled) into [C*K, OW] once and stored.
  // For quantized inference: weight is 4-bit quantized, then tiled into this
  // buffer. During forward, only input needs im2col, then element-wise
  // INT4×INT8 multiply + reduce-sum replaces per-channel dot products.
  TensorDim weight_col_dim(1, 1, channels * kernel_size, out_width,
                           in_dim.getTensorType());
  weight_col_idx = context.requestTensor(weight_col_dim, "weight_col",
                                         Initializer::NONE, false,
                                         TensorLifespan::MAX_LIFESPAN);
}

void DepthwiseConv1DLayer::expandWeightCol(const Tensor &filter,
                                           Tensor &weight_col,
                                           unsigned int channels,
                                           unsigned int kernel_size,
                                           unsigned int out_width) {
  // Tile weight[c, 0, 0, k] across all OW positions:
  //   weight_col[0, 0, c*K+k, ow] = filter[c, 0, 0, k]  for all ow
  for (unsigned int c = 0; c < channels; ++c) {
    for (unsigned int k = 0; k < kernel_size; ++k) {
      unsigned int row = c * kernel_size + k;
      float val = filter.getValue(c, 0, 0, k);
      for (unsigned int ow = 0; ow < out_width; ++ow) {
        weight_col.setValue(0, 0, row, ow, val);
      }
    }
  }
}

void DepthwiseConv1DLayer::forwarding(RunLayerContext &context, bool training) {
  unsigned int kernel_size = std::get<props::KernelSize>(conv_props).get();
  unsigned int stride = std::get<props::Stride>(conv_props).get();
  unsigned int dilation = std::get<props::Dilation>(conv_props).get();

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &filter = context.getWeight(wt_idx[DepthwiseConvParams::weight]);
  Tensor &col_buf = context.getTensor(col_buf_idx);
  Tensor &weight_col = context.getTensor(weight_col_idx);

  const TensorDim &in_dim = input_.getDim();
  const TensorDim &out_dim = hidden_.getDim();

  unsigned int batch = in_dim.batch();
  unsigned int channels = in_dim.channel();
  unsigned int in_width = in_dim.width();
  unsigned int out_width = out_dim.width();
  unsigned int pad_left = padding[0];

  // Expand weight [C,1,1,K] → weight_col [1,1,C*K,OW] by tiling across OW.
  // Done only once; the tiled weight_col is reused across all forward calls.
  // For quantized inference the stored weight_col can be INT4 quantized.
  if (!weight_col_expanded) {
    expandWeightCol(filter, weight_col, channels, kernel_size, out_width);
    weight_col_expanded = true;
  }

  /**
   * Depthwise 1D convolution using im2col + element-wise multiply + reduce.
   *
   * For each batch:
   *   1. im2col: unroll input patches into col_buf [C*K, OW]
   *   2. Element-wise multiply: col_buf *= weight_col  (both [C*K, OW])
   *   3. Reduce-sum every K rows to get output [C, OW]:
   *      output[c, ow] = sum_{k=0}^{K-1} col_buf[c*K+k, ow]
   *
   * This approach replaces per-channel dot products with bulk element-wise
   * ops, which maps directly to quantized INT4×INT8 element-wise multiply
   * followed by reduction.
   */
  for (unsigned int b = 0; b < batch; ++b) {
    col_buf.setZero();
    Tensor in_sub = input_.getBatchSlice(b, 1);

    depthwise_im2col_1d(in_sub, col_buf, channels, in_width, out_width,
                        kernel_size, stride, dilation, pad_left);

    // Element-wise multiply: col_buf[C*K, OW] *= weight_col[C*K, OW]
    col_buf.multiply_i(weight_col);

    // Reduce-sum every K rows → output[C, OW]
    for (unsigned int c = 0; c < channels; ++c) {
      for (unsigned int ow = 0; ow < out_width; ++ow) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < kernel_size; ++k) {
          sum += col_buf.getValue(0, 0, c * kernel_size + k, ow);
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
      throw std::invalid_argument("[DepthwiseConv1D] adding bias failed");
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
  Tensor &col_buf = context.getTensor(col_buf_idx);

  const TensorDim &deriv_dim = derivative.getDim();
  const TensorDim &in_deriv_dim = input_derivative.getDim();

  unsigned int batch = deriv_dim.batch();
  unsigned int channels = deriv_dim.channel();
  unsigned int out_width = deriv_dim.width();
  unsigned int in_width = in_deriv_dim.width();
  unsigned int pad_left = padding[0];

  input_derivative.setZero();

  /**
   * Backprop to input using col2im:
   * For each batch:
   *   1. Per-channel: col_block = weight[c]^T . d_output[c,:]
   *      weight[c] is [1, K], d_output is [1, OW] → col_block is [K, OW]
   *   2. col2im: scatter col_buf back to input derivative positions
   */
  for (unsigned int b = 0; b < batch; ++b) {
    col_buf.setZero();

    for (unsigned int c = 0; c < channels; ++c) {
      Tensor w_row = filter.getSharedDataTensor(
        {1, 1, 1, kernel_size}, c * kernel_size);
      Tensor col_block = col_buf.getSharedDataTensor(
        {1, 1, kernel_size, out_width}, c * kernel_size * out_width);
      Tensor deriv_row = derivative.getSharedDataTensor(
        {1, 1, 1, out_width}, (b * channels + c) * out_width);

      // col_block[K, OW] = w_row^T[K, 1] . deriv_row[1, OW]
      w_row.dot(deriv_row, col_block, true, false);
    }

    Tensor in_deriv_sub = input_derivative.getBatchSlice(b, 1);
    depthwise_col2im_1d(col_buf, in_deriv_sub, channels, in_width, out_width,
                        kernel_size, stride, dilation, pad_left);
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
  Tensor &col_buf = context.getTensor(col_buf_idx);

  const TensorDim &deriv_dim = derivative.getDim();
  const TensorDim &in_dim = input_.getDim();

  unsigned int batch = deriv_dim.batch();
  unsigned int channels = deriv_dim.channel();
  unsigned int out_width = deriv_dim.width();
  unsigned int in_width = in_dim.width();
  unsigned int pad_left = padding[0];

  /**
   * Weight gradient using im2col:
   * For each batch:
   *   1. im2col input patches into col_buf
   *   2. Per-channel: delK[c] += d_output[c,:] . col_block[c]^T
   *      d_output is [1, OW], col_block is [K, OW]
   *      → delK[c] is [1, K]
   */
  for (unsigned int b = 0; b < batch; ++b) {
    col_buf.setZero();
    Tensor in_sub = input_.getBatchSlice(b, 1);

    depthwise_im2col_1d(in_sub, col_buf, channels, in_width, out_width,
                        kernel_size, stride, dilation, pad_left);

    for (unsigned int c = 0; c < channels; ++c) {
      Tensor delK_row = delK.getSharedDataTensor(
        {1, 1, 1, kernel_size}, c * kernel_size);
      Tensor col_block = col_buf.getSharedDataTensor(
        {1, 1, kernel_size, out_width}, c * kernel_size * out_width);
      Tensor deriv_row = derivative.getSharedDataTensor(
        {1, 1, 1, out_width}, (b * channels + c) * out_width);

      // delK[c] += deriv_row[1, OW] . col_block^T[OW, K] = [1, K]
      deriv_row.dot(col_block, delK_row, false, true, 1.0f);
    }
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &delBias =
      context.getWeightGrad(wt_idx[DepthwiseConvParams::bias]);
    delBias.setZero();
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

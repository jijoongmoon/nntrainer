// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv1d_transpose.cpp
 * @date   15 June 2026
 * @brief  Convolution 1D transpose (deconvolution) layer.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <limits>
#include <string>

#include <conv2d_transpose_layer.h>
#include <nntrainer_error.h>

#include "conv1d_transpose.h"

namespace causallm {

Conv1DTransposeLayer::Conv1DTransposeLayer() :
  LayerImpl(),
  conv_props(nntrainer::props::FilterSize(), nntrainer::props::KernelSize(),
             nntrainer::props::Stride(), nntrainer::props::Padding1D(),
             nntrainer::props::Dilation()) {
  conv2d_transpose_layer =
    std::make_unique<nntrainer::Conv2DTransposeLayer>();
}

Conv1DTransposeLayer::~Conv1DTransposeLayer() {}

void Conv1DTransposeLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Conv1DTranspose layer takes only one input";

  NNTR_THROW_IF(context.getInputDimensions()[0].height() != 1,
                std::invalid_argument)
    << "Conv1DTranspose layer requires input with height 1";

  const nntrainer::TensorDim &in_dim = context.getInputDimensions()[0];
  const unsigned int kernel_size =
    std::get<nntrainer::props::KernelSize>(conv_props).get();
  const unsigned int stride =
    std::get<nntrainer::props::Stride>(conv_props).get();
  const unsigned int dilation =
    std::get<nntrainer::props::Dilation>(conv_props).get();

  // explicit/"same"/"valid" padding is resolved here; BigVGAN uses explicit
  // p=(k-stride)//2 so compute() returns it verbatim (case 1).
  const std::array<unsigned int, 2> padding =
    std::get<nntrainer::props::Padding1D>(conv_props)
      .compute(in_dim, kernel_size, stride, dilation);
  const std::string padding_str =
    "0,0," + std::to_string(padding[0]) + "," + std::to_string(padding[1]);

  auto setPropertyKV = [this](const std::string &key,
                              const std::string &value) {
    auto const &prop = key + "=" + value;
    conv2d_transpose_layer->setProperty({prop});
  };

  setPropertyKV(
    nntrainer::props::FilterSize::key,
    std::to_string(std::get<nntrainer::props::FilterSize>(conv_props).get()));
  setPropertyKV(nntrainer::props::KernelSize::key,
                "1," + std::to_string(kernel_size));
  setPropertyKV(nntrainer::props::Stride::key, "1," + std::to_string(stride));
  setPropertyKV(nntrainer::props::Padding2D::key, padding_str);
  setPropertyKV(nntrainer::props::Dilation::key,
                "1," + std::to_string(dilation));

  conv2d_transpose_layer->finalize(context);
}

void Conv1DTransposeLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {
  conv2d_transpose_layer->forwarding(context, training);
}

void Conv1DTransposeLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  conv2d_transpose_layer->calcDerivative(context);
}

void Conv1DTransposeLayer::calcGradient(nntrainer::RunLayerContext &context) {
  conv2d_transpose_layer->calcGradient(context);
}

void Conv1DTransposeLayer::exportTo(
  nntrainer::Exporter &exporter,
  const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void Conv1DTransposeLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

} // namespace causallm

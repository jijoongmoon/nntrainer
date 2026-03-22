// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file unittest_layers_depthwise_convolution1d.cpp
 * @date 22 March 2026
 * @brief Depthwise Conv1d Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <depthwise_conv1d_layer.h>
#include <layers_common_tests.h>

auto semantic_depthwise_conv1d = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::DepthwiseConv1DLayer>,
  nntrainer::DepthwiseConv1DLayer::type, {"kernel_size=3", "padding=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(DepthwiseConvolution1D, LayerSemantics,
                     ::testing::Values(semantic_depthwise_conv1d));

auto depthwise_conv1d_sb_minimum = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DepthwiseConv1DLayer>,
  {"kernel_size=3"}, "1:3:1:8",
  "depthwise_conv1d_sb_minimum.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto depthwise_conv1d_mb_minimum = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DepthwiseConv1DLayer>,
  {"kernel_size=3"}, "3:3:1:8",
  "depthwise_conv1d_mb_minimum.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto depthwise_conv1d_sb_same = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DepthwiseConv1DLayer>,
  {"kernel_size=3", "padding=same"}, "1:4:1:8",
  "depthwise_conv1d_sb_same.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto depthwise_conv1d_mb_same = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DepthwiseConv1DLayer>,
  {"kernel_size=3", "padding=same"}, "3:4:1:8",
  "depthwise_conv1d_mb_same.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto depthwise_conv1d_sb_stride = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DepthwiseConv1DLayer>,
  {"kernel_size=3", "stride=2"}, "1:3:1:8",
  "depthwise_conv1d_sb_stride.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto depthwise_conv1d_mb_stride = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DepthwiseConv1DLayer>,
  {"kernel_size=3", "stride=2"}, "3:3:1:8",
  "depthwise_conv1d_mb_stride.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto depthwise_conv1d_sb_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DepthwiseConv1DLayer>,
  {"kernel_size=3", "dilation=2"}, "1:3:1:11",
  "depthwise_conv1d_sb_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto depthwise_conv1d_mb_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DepthwiseConv1DLayer>,
  {"kernel_size=3", "dilation=2"}, "3:3:1:11",
  "depthwise_conv1d_mb_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto depthwise_conv1d_sb_causal = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DepthwiseConv1DLayer>,
  {"kernel_size=4", "padding=causal"}, "1:4:1:8",
  "depthwise_conv1d_sb_causal.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto depthwise_conv1d_mb_causal = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DepthwiseConv1DLayer>,
  {"kernel_size=4", "padding=causal"}, "3:4:1:8",
  "depthwise_conv1d_mb_causal.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto depthwise_conv1d_sb_no_bias = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DepthwiseConv1DLayer>,
  {"kernel_size=3", "disable_bias=true"}, "1:3:1:8",
  "depthwise_conv1d_sb_no_bias.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto depthwise_conv1d_mb_no_bias = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DepthwiseConv1DLayer>,
  {"kernel_size=3", "disable_bias=true"}, "3:3:1:8",
  "depthwise_conv1d_mb_no_bias.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(
  DepthwiseConvolution1D, LayerGoldenTest,
  ::testing::Values(
    depthwise_conv1d_sb_minimum, depthwise_conv1d_mb_minimum,
    depthwise_conv1d_sb_same, depthwise_conv1d_mb_same,
    depthwise_conv1d_sb_stride, depthwise_conv1d_mb_stride,
    depthwise_conv1d_sb_dilation, depthwise_conv1d_mb_dilation,
    depthwise_conv1d_sb_causal, depthwise_conv1d_mb_causal,
    depthwise_conv1d_sb_no_bias, depthwise_conv1d_mb_no_bias));

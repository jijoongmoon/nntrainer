// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_rms_norm.cpp
 * @date   31 March 2026
 * @brief  Unit test for CausalLM RMS normalization layer
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <rms_norm.h>

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LayerPropertySemantics);

auto semantic_rms_norm = LayerSemanticsParamType(
  nntrainer::createLayer<causallm::RMSNormLayer>,
  causallm::RMSNormLayer::type, {},
  0 /* not from app context */, false /* must not fail */, 1 /* num inputs */);

GTEST_PARAMETER_TEST(RMSNorm, LayerSemantics,
                     ::testing::Values(semantic_rms_norm));

auto rms_option = LayerGoldenTestParamOptions::SKIP_COSINE_SIMILARITY;

auto rms_norm_training = LayerGoldenTestParamType(
  nntrainer::createLayer<causallm::RMSNormLayer>, {"epsilon=1e-7"},
  "2:3:3:3", "rms_norm_training.nnlayergolden", rms_option, "nchw", "fp32",
  "fp32");

GTEST_PARAMETER_TEST(RMSNorm, LayerGoldenTest,
                     ::testing::Values(rms_norm_training));

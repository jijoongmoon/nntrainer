// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file unittest_models_mixed_precision.cpp
 * @date 3 May 2024
 * @brief unittest models to cover mixed precision
 * @see	https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <memory>

#include <ini_wrapper.h>
#include <neuralnet.h>
#include <nntrainer_test_util.h>

#include <models_golden_test.h>

using namespace nntrainer;

static std::unique_ptr<NeuralNetwork> fc_mixed_training() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty(
    {"batch_size=1", "model_tensor_type=FP16-FP16", "loss_scale=65536"});

  auto graph = makeGraph({
    {"input", {"name=in", "input_shape=1:1:3", "input_dtype=FP16"}},
    {"Fully_connected", {"name=fc", "input_layers=in", "unit=10"}},
    {"mse", {"name=loss", "input_layers=fc"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer(
    "adam", {"learning_rate = 0.1", "torch_ref=true"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> fc_mixed_training_nan_sgd() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty(
    {"batch_size=1", "model_tensor_type=FP16-FP16", "loss_scale=65536"});

  auto graph = makeGraph({
    {"input", {"name=in", "input_shape=1:1:1", "input_dtype=FP16"}},
    {"Fully_connected", {"name=fc0", "input_layers=in", "unit=1"}},
    {"Fully_connected", {"name=fc1", "input_layers=fc0", "unit=1"}},
    {"mse", {"name=loss", "input_layers=fc1"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));

  return nn;
}

GTEST_PARAMETER_TEST(
  MixedPrecision, nntrainerModelTest,
  ::testing::ValuesIn({
    mkModelTc_V2(fc_mixed_training, "fc_mixed_training",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(fc_mixed_training_nan_sgd, "fc_mixed_training_nan_sgd",
                 ModelTestOption::ALL_V2),
  }),
  [](const testing::TestParamInfo<nntrainerModelTest::ParamType> &info)
    -> const auto & { return std::get<1>(info.param); });

#ifdef ENABLE_FP16
/**
 * @brief An FP16-activation model of one fully connected layer
 * @param input_dtype dtype the input layer is declared with
 */
static std::unique_ptr<NeuralNetwork>
fp16_fc_model(const std::string &input_dtype) {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=1", "model_tensor_type=FP16-FP16"});

  auto graph = makeGraph({
    {"input", {"name=in", "input_shape=1:1:4", "input_dtype=" + input_dtype}},
    {"fully_connected",
     {"name=fc", "input_layers=in", "unit=3", "weight_initializer=ones",
      "bias_initializer=zeros"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }
  return nn;
}

/**
 * @brief Set the fully connected weight to ones and the bias to zeros
 * @note An inference model does not run the initializers.
 */
static void set_fc_ones(NeuralNetwork &nn) {
  for (auto &node : nn.getFlatGraph()) {
    for (unsigned int i = 0; i < node->getNumWeights(); ++i) {
      const bool is_bias =
        node->getWeightName(i).find("bias") != std::string::npos;
      node->getWeight(i).setValue(is_bias ? 0.0f : 1.0f);
    }
  }
}

/**
 * @brief An FP32 input layer feeding an FP16 fully connected layer is rejected
 * at initialize instead of producing NaN at inference.
 */
TEST(MixedPrecisionInputDtype, fp32_input_into_fp16_fc_n) {
  auto nn = fp16_fc_model("FP32");
  EXPECT_EQ(nn->compile(ml::train::ExecutionMode::INFERENCE), ML_ERROR_NONE);
  EXPECT_THROW(nn->initialize(ml::train::ExecutionMode::INFERENCE),
               std::invalid_argument);
}

/**
 * @brief An FP16 input layer feeding an FP16 fully connected layer runs and
 * returns the expected FP16 output.
 */
TEST(MixedPrecisionInputDtype, fp16_input_into_fp16_fc_p) {
  auto nn = fp16_fc_model("FP16");
  EXPECT_EQ(nn->compile(ml::train::ExecutionMode::INFERENCE), ML_ERROR_NONE);
  EXPECT_EQ(nn->initialize(ml::train::ExecutionMode::INFERENCE), ML_ERROR_NONE);
  set_fc_ones(*nn);

  std::vector<_FP16> input = {
    static_cast<_FP16>(1.0f), static_cast<_FP16>(2.0f),
    static_cast<_FP16>(3.0f), static_cast<_FP16>(4.0f)};
  std::vector<float *> in = {reinterpret_cast<float *>(input.data())};
  std::vector<float *> out = nn->inference(1, in);
  ASSERT_EQ(out.size(), 1u);
  const _FP16 *answer = reinterpret_cast<const _FP16 *>(out[0]);
  for (unsigned int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(static_cast<float>(answer[i]), 10.0f);
}

/**
 * @brief A tensor of another dtype than the declared input is rejected when
 * it is bound to the model input.
 */
TEST(MixedPrecisionInputDtype, fp32_tensor_into_fp16_input_n) {
  auto nn = fp16_fc_model("FP16");
  EXPECT_EQ(nn->compile(ml::train::ExecutionMode::INFERENCE), ML_ERROR_NONE);
  EXPECT_EQ(nn->initialize(ml::train::ExecutionMode::INFERENCE), ML_ERROR_NONE);
  set_fc_ones(*nn);

  auto x =
    std::make_shared<Tensor>(TensorDim(1, 1, 1, 4,
                                       {ml::train::TensorDim::Format::NCHW,
                                        ml::train::TensorDim::DataType::FP32}));
  x->setValue(1.0f);
  EXPECT_THROW(nn->inference(sharedConstTensors{x}), std::invalid_argument);
}

/**
 * @brief A tensor of the declared input dtype is accepted.
 */
TEST(MixedPrecisionInputDtype, fp16_tensor_into_fp16_input_p) {
  auto nn = fp16_fc_model("FP16");
  EXPECT_EQ(nn->compile(ml::train::ExecutionMode::INFERENCE), ML_ERROR_NONE);
  EXPECT_EQ(nn->initialize(ml::train::ExecutionMode::INFERENCE), ML_ERROR_NONE);
  set_fc_ones(*nn);

  auto x =
    std::make_shared<Tensor>(TensorDim(1, 1, 1, 4,
                                       {ml::train::TensorDim::Format::NCHW,
                                        ml::train::TensorDim::DataType::FP16}));
  x->setValue(1.0f);
  sharedConstTensors out;
  EXPECT_NO_THROW(out = nn->inference(sharedConstTensors{x}));
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FLOAT_EQ(static_cast<float>(out[0]->getValue<_FP16>(0, 0, 0, 0)),
                  4.0f);
}
#endif

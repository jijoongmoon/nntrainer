// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Donghak Park <donghak.park@samsung.com>
 *
 * @file   integration_test_fsu.cpp
 * @date   20 Dec 2024
 * @brief  Unit Test for Asynch FSU
 * @see    https://github.com/nntrainer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <app_context.h>
#include <gtest/gtest.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <util_func.h>
#include <vector>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
std::vector<float> ori_answer;
std::vector<float> fsu_answer;

void RemoveWeightFile(std::string file_path) { remove(file_path.c_str()); }

/**
 * @brief Input dtype for a model_tensor_type ("<weight>-<activation>").
 * @note The input layer keeps the dtype it is declared with, so an FP16
 * activation model has to ask for an FP16 input explicitly.
 */
static std::string inputDtype(const std::string &weight_act_type) {
  return weight_act_type.substr(weight_act_type.find('-') + 1);
}

/**
 * @brief Run one inference on the input 0, 1, ..., feature_size - 1 and
 * return the output as floats.
 * @note The float* inference API maps the caller's buffers with the model's
 * own input/output dtype, so an FP16 activation model is fed and read
 * through _FP16 buffers rather than float ones.
 */
static std::vector<float> RunInference(ml::train::Model &model,
                                       unsigned int feature_size,
                                       const std::string &weight_act_type) {
  std::vector<float> result;
  result.reserve(feature_size);
#ifdef ENABLE_FP16
  if (inputDtype(weight_act_type) == "FP16") {
    std::vector<_FP16> input(feature_size);
    for (unsigned int j = 0; j < feature_size; ++j)
      input[j] = static_cast<_FP16>(static_cast<float>(j));
    std::vector<float *> in = {reinterpret_cast<float *>(input.data())};
    std::vector<float *> answer = model.inference(1, in);
    const _FP16 *out = reinterpret_cast<const _FP16 *>(answer[0]);
    for (unsigned int i = 0; i < feature_size; i++)
      result.push_back(static_cast<float>(out[i]));
    return result;
  }
#endif
  std::vector<float> input(feature_size);
  for (unsigned int j = 0; j < feature_size; ++j)
    input[j] = static_cast<float>(j);
  std::vector<float *> in = {input.data()};
  std::vector<float *> answer = model.inference(1, in);
  for (unsigned int i = 0; i < feature_size; i++)
    result.push_back(answer[0][i]);
  return result;
}

void MakeWeight(unsigned int feature_size, unsigned int layer_num,
                unsigned int look_ahead, std::string file_path,
                std::string weight_act_type) {

  ModelHandle _model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "mse")});
  _model->addLayer(ml::train::createLayer(
    "input",
    {nntrainer::withKey("name", "input0"),
     nntrainer::withKey("input_shape", "1:1:" + std::to_string(feature_size)),
     nntrainer::withKey("input_dtype", inputDtype(weight_act_type))}));

  for (unsigned int i = 0; i < layer_num; i++) {
    _model->addLayer(ml::train::createLayer(
      "fully_connected",
      {nntrainer::withKey("unit", std::to_string(feature_size)),
       nntrainer::withKey("weight_initializer", "xavier_uniform"),
       nntrainer::withKey("bias_initializer", "zeros")}));
  }

  _model->setProperty(
    {nntrainer::withKey("batch_size", 1), nntrainer::withKey("epochs", 1),
     nntrainer::withKey("model_tensor_type", weight_act_type)});
  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  int status = _model->setOptimizer(std::move(optimizer));

  status = _model->compile();
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = _model->initialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
  _model->save(file_path);
}

void MakeAnswer(unsigned int feature_size, unsigned int layer_num,
                unsigned int look_ahead, std::string file_path,
                std::string weight_act_type) {

  ModelHandle _model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "mse")});
  _model->addLayer(ml::train::createLayer(
    "input",
    {nntrainer::withKey("name", "input0"),
     nntrainer::withKey("input_shape", "1:1:" + std::to_string(feature_size)),
     nntrainer::withKey("input_dtype", inputDtype(weight_act_type))}));

  for (unsigned int i = 0; i < layer_num; i++) {
    _model->addLayer(ml::train::createLayer(
      "fully_connected",
      {nntrainer::withKey("unit", std::to_string(feature_size)),
       nntrainer::withKey("weight_initializer", "xavier_uniform"),
       nntrainer::withKey("bias_initializer", "zeros")}));
  }

  _model->setProperty(
    {nntrainer::withKey("batch_size", 1), nntrainer::withKey("epochs", 1),
     nntrainer::withKey("model_tensor_type", weight_act_type)});
  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  int status = _model->setOptimizer(std::move(optimizer));

  status = _model->compile(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = _model->initialize(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);
  _model->load(file_path);

  ori_answer = RunInference(*_model, feature_size, weight_act_type);
}

void MakeAndRunModel(unsigned int feature_size, unsigned int layer_num,
                     unsigned int look_ahead, std::string file_path,
                     std::string weight_act_type) {

  ModelHandle _model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "mse")});
  _model->addLayer(ml::train::createLayer(
    "input",
    {nntrainer::withKey("name", "input0"),
     nntrainer::withKey("input_shape", "1:1:" + std::to_string(feature_size)),
     nntrainer::withKey("input_dtype", inputDtype(weight_act_type))}));

  for (unsigned int i = 0; i < layer_num; i++) {
    _model->addLayer(ml::train::createLayer(
      "fully_connected",
      {nntrainer::withKey("unit", std::to_string(feature_size)),
       nntrainer::withKey("weight_initializer", "xavier_uniform"),
       nntrainer::withKey("bias_initializer", "zeros")}));
  }

  _model->setProperty(
    {nntrainer::withKey("batch_size", 1), nntrainer::withKey("epochs", 1),
     nntrainer::withKey("fsu", "true"),
     nntrainer::withKey("fsu_lookahead", std::to_string(look_ahead)),
     nntrainer::withKey("model_tensor_type", weight_act_type)});

  int status = _model->compile(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = _model->initialize(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);
  _model->load(file_path);

  fsu_answer = RunInference(*_model, feature_size, weight_act_type);
}

/**
 * @brief FSU TestParm with variout LookAhead
 *
 */
class LookAheadParm
  : public ::testing::TestWithParam<
      std::tuple<unsigned int, std::string, unsigned int, unsigned int>> {};
// look_ahead_parm, weight_act_type, feature_size, layer_num
TEST_P(LookAheadParm, simple_fc) {
  auto param = GetParam();
  unsigned int look_ahead_parm = std::get<0>(param);
  std::string weight_act_type = std::get<1>(param);
  unsigned int feature_size = std::get<2>(param);
  unsigned int layer_num = std::get<3>(param);
  std::string file_path = "weight_file" + std::to_string(look_ahead_parm) +
                          "_" + std::to_string(feature_size) + "_" +
                          std::to_string(layer_num) + ".bin";

  ori_answer.clear();
  fsu_answer.clear();

  EXPECT_NO_THROW(MakeWeight(feature_size, layer_num, look_ahead_parm,
                             file_path, weight_act_type));
  EXPECT_NO_THROW(MakeAnswer(feature_size, layer_num, look_ahead_parm,
                             file_path, weight_act_type));
  EXPECT_NO_THROW(MakeAndRunModel(feature_size, layer_num, look_ahead_parm,
                                  file_path, weight_act_type));
  EXPECT_EQ(ori_answer, fsu_answer);
  RemoveWeightFile(file_path);
}

#ifdef ENABLE_FP16
INSTANTIATE_TEST_SUITE_P(
  LookAheadParmTest, LookAheadParm,
  ::testing::Values(std::make_tuple(2, "FP16-FP16", 1024, 8),
                    std::make_tuple(4, "FP16-FP16", 1024, 8),
                    std::make_tuple(2, "FP32-FP32", 1024, 8),
                    std::make_tuple(4, "FP32-FP32", 1024, 8),
                    std::make_tuple(4, "FP32-FP32", 512, 6),
                    std::make_tuple(4, "FP32-FP32", 256, 6),
                    std::make_tuple(4, "FP32-FP32", 100, 6)));
#else
INSTANTIATE_TEST_SUITE_P(
  LookAheadParmTest, LookAheadParm,
  ::testing::Values(std::make_tuple(2, "FP32-FP32", 1024, 8),
                    std::make_tuple(4, "FP32-FP32", 1024, 8),
                    std::make_tuple(4, "FP32-FP32", 512, 6),
                    std::make_tuple(4, "FP32-FP32", 256, 6),
                    std::make_tuple(4, "FP32-FP32", 100, 6)));
#endif

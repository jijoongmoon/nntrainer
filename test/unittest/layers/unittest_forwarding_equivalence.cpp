// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_forwarding_equivalence.cpp
 * @brief  End-to-end test comparing incremental_forwarding vs forwarding paths
 *         using a small transformer model with shared weights.
 */

#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <app_context.h>
#include <layer_context.h>
#include <model.h>
#include <tensor.h>
#include <tensor_api.h>

#include <embedding_layer.h>
#include <lm_head.h>
#include <mha_core.h>
#include <rms_norm.h>
#include <swiglu.h>

using ml::train::createLayer;

static std::string withKey(const std::string &key, const std::string &value) {
  return key + "=" + value;
}

static std::string withKey(const std::string &key, unsigned int value) {
  return key + "=" + std::to_string(value);
}

static const unsigned int DIM = 8;
static const unsigned int NUM_HEADS = 2;
static const unsigned int NUM_VOCAB = 32;
static const unsigned int NUM_TO_GENERATE = 4;
static const std::string WEIGHT_PATH = "/tmp/test_tiny_transformer.bin";

static void registerCustomLayers() {
  auto &ac = nntrainer::AppContext::Global();
  try {
    ac.registerFactory(nntrainer::createLayer<causallm::MHACoreLayer>);
  } catch (...) {
  }
  try {
    ac.registerFactory(nntrainer::createLayer<causallm::RMSNormLayer>);
  } catch (...) {
  }
  try {
    ac.registerFactory(nntrainer::createLayer<causallm::SwiGLULayer>);
  } catch (...) {
  }
  try {
    ac.registerFactory(nntrainer::createLayer<causallm::EmbeddingLayer>);
  } catch (...) {
  }
  try {
    ac.registerFactory(nntrainer::createLayer<causallm::LmHeadLayer>);
  } catch (...) {
  }
}

static std::unique_ptr<ml::train::Model>
buildTinyTransformer(unsigned int seq_len) {
  registerCustomLayers();

  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty(
    {withKey("batch_size", 1u), withKey("epochs", "1"),
     withKey("model_tensor_type", "FP32-FP32")});

  using ml::train::LayerHandle;
  using Tensor = ml::train::Tensor;

  LayerHandle input_layer = createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(seq_len))});
  Tensor input = input_layer(Tensor());

  LayerHandle embedding = createLayer(
    "embedding_layer",
    {withKey("name", "embedding0"), withKey("in_dim", NUM_VOCAB),
     withKey("out_dim", DIM), withKey("weight_initializer", "xavier_uniform")});
  Tensor x = embedding(input);

  LayerHandle norm1 = createLayer(
    "rms_norm",
    {withKey("name", "norm1"), withKey("epsilon", "1e-5")});
  x = norm1(x);

  LayerHandle q_proj = createLayer(
    "fully_connected",
    {withKey("name", "q_proj"), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("weight_initializer", "xavier_uniform")});
  Tensor q = q_proj(x);

  LayerHandle k_proj = createLayer(
    "fully_connected",
    {withKey("name", "k_proj"), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("weight_initializer", "xavier_uniform")});
  Tensor k = k_proj(x);

  LayerHandle v_proj = createLayer(
    "fully_connected",
    {withKey("name", "v_proj"), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("weight_initializer", "xavier_uniform")});
  Tensor v = v_proj(x);

  LayerHandle attn = createLayer(
    "mha_core",
    {withKey("name", "attn0"), withKey("num_heads", NUM_HEADS),
     withKey("num_heads_kv", NUM_HEADS),
     withKey("max_timestep", std::to_string(seq_len + NUM_TO_GENERATE)),
     withKey("rope_theta", "10000"),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", "true")});
  Tensor a = attn({q, k, v});

  LayerHandle o_proj = createLayer(
    "fully_connected",
    {withKey("name", "o_proj"), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("weight_initializer", "xavier_uniform")});
  Tensor o = o_proj(a);

  LayerHandle lm_head = createLayer(
    "lm_head",
    {withKey("name", "lm_head"), withKey("unit", NUM_VOCAB),
     withKey("disable_bias", "true"), withKey("weight_initializer", "xavier_uniform")});
  Tensor output = lm_head(o);

  std::vector<Tensor> all_inputs = {input};
  std::vector<Tensor> outputs = {output};
  model->compile(all_inputs, outputs, ml::train::ExecutionMode::INFERENCE);

  return model;
}

/**
 * @brief Test incremental_inference prefill path
 */
TEST(ForwardingEquivalence, incremental_prefill_produces_output) {
  const unsigned int SEQ_LEN = 4;
  auto model = buildTinyTransformer(SEQ_LEN);

  // Save weights for reproducibility
  model->save(WEIGHT_PATH);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float *> input = {input_data.data()};
  std::vector<float *> label;

  auto output = model->incremental_inference(1, input, label, SEQ_LEN, 0,
                                             SEQ_LEN, false);
  ASSERT_FALSE(output.empty());
  ASSERT_NE(output[0], nullptr);

  // Verify no NaN or Inf (logits may be 0 with default weight init)
  float *logits = output[0];
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    EXPECT_FALSE(std::isnan(logits[i])) << "NaN at index " << i;
    EXPECT_FALSE(std::isinf(logits[i])) << "Inf at index " << i;
  }

  for (auto &out : output)
    delete[] out;

  std::remove(WEIGHT_PATH.c_str());
}

/**
 * @brief Test forwarding path with resetInputDimension
 */
TEST(ForwardingEquivalence, forwarding_prefill_produces_output) {
  const unsigned int SEQ_LEN = 4;
  auto model = buildTinyTransformer(SEQ_LEN);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float *> input = {input_data.data()};
  std::vector<float *> label;

  // resetInputDimension to set height=SEQ_LEN
  ml::train::TensorDim input_dim(1, 1, SEQ_LEN, 1);
  model->resetInputDimension({input_dim});

  // cache_index is 0 by default in newly constructed model
  auto output = model->inference(1, input, label);
  ASSERT_FALSE(output.empty());
  ASSERT_NE(output[0], nullptr);

  // Verify no NaN or Inf
  float *logits = output[0];
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    EXPECT_FALSE(std::isnan(logits[i])) << "NaN at index " << i;
    EXPECT_FALSE(std::isinf(logits[i])) << "Inf at index " << i;
  }
}

/**
 * @brief Compare incremental_forwarding vs forwarding with same weights.
 *        Build one model, save weights, load into both paths, compare.
 */
TEST(ForwardingEquivalence, incremental_vs_forwarding_same_output) {
  const unsigned int SEQ_LEN = 4;

  // Build model and save weights
  auto model_ref = buildTinyTransformer(SEQ_LEN);
  model_ref->save(WEIGHT_PATH);

  std::vector<float> input_data = {1.0f, 5.0f, 10.0f, 15.0f};
  std::vector<float *> label;

  // --- Path 1: incremental_forwarding ---
  auto model1 = buildTinyTransformer(SEQ_LEN);
  model1->load(WEIGHT_PATH);

  std::vector<float *> input1 = {input_data.data()};
  auto output1 = model1->incremental_inference(1, input1, label, SEQ_LEN, 0,
                                               SEQ_LEN, false);
  ASSERT_FALSE(output1.empty());
  std::vector<float> logits1(output1[0], output1[0] + NUM_VOCAB);
  for (auto &out : output1)
    delete[] out;

  // --- Path 2: forwarding via resetInputDimension ---
  auto model2 = buildTinyTransformer(SEQ_LEN);
  model2->load(WEIGHT_PATH);

  ml::train::TensorDim input_dim(1, 1, SEQ_LEN, 1);
  model2->resetInputDimension({input_dim});

  // cache_index is 0 by default in newly constructed model
  std::vector<float *> input2 = {input_data.data()};
  auto output2 = model2->inference(1, input2, label);
  ASSERT_FALSE(output2.empty());
  std::vector<float> logits2(output2[0], output2[0] + NUM_VOCAB);

  // --- Compare ---
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    EXPECT_NEAR(logits1[i], logits2[i], 1e-4)
      << "Mismatch at vocab index " << i
      << ": incremental=" << logits1[i] << " forwarding=" << logits2[i];
  }

  std::remove(WEIGHT_PATH.c_str());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

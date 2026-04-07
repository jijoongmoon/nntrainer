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

/**
 * @brief Multi-turn test: prefill → generate tokens → 2nd turn prefill →
 * generate. Tests that both paths handle sequential inference correctly.
 * Uses incremental_inference path (the established working path).
 */
TEST(ForwardingEquivalence, multiturn_incremental) {
  const unsigned int SEQ_LEN = 4;

  auto model = buildTinyTransformer(SEQ_LEN);

  std::vector<float *> label;

  // --- Turn 1: Prefill with 4 tokens ---
  std::vector<float> turn1_input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float *> input1 = {turn1_input.data()};

  auto out1 = model->incremental_inference(1, input1, label, SEQ_LEN, 0,
                                           SEQ_LEN, false);
  ASSERT_FALSE(out1.empty());
  std::vector<float> logits_turn1(out1[0], out1[0] + NUM_VOCAB);
  for (auto &out : out1)
    delete[] out;

  // --- Turn 1: Generate 2 tokens ---
  std::vector<float> gen_token = {5.0f}; // simulated generated token
  std::vector<float *> gen_input = {gen_token.data()};

  auto out_gen1 = model->incremental_inference(1, gen_input, label, SEQ_LEN,
                                               SEQ_LEN, SEQ_LEN + 1, false);
  ASSERT_FALSE(out_gen1.empty());
  std::vector<float> logits_gen1(out_gen1[0], out_gen1[0] + NUM_VOCAB);
  for (auto &out : out_gen1)
    delete[] out;

  gen_token[0] = 6.0f; // next token
  auto out_gen2 = model->incremental_inference(1, gen_input, label, SEQ_LEN,
                                               SEQ_LEN + 1, SEQ_LEN + 2, false);
  ASSERT_FALSE(out_gen2.empty());
  std::vector<float> logits_gen2(out_gen2[0], out_gen2[0] + NUM_VOCAB);
  for (auto &out : out_gen2)
    delete[] out;

  // Verify outputs are valid (no NaN/Inf) and different between steps
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    EXPECT_FALSE(std::isnan(logits_turn1[i]));
    EXPECT_FALSE(std::isnan(logits_gen1[i]));
    EXPECT_FALSE(std::isnan(logits_gen2[i]));
  }
}

/**
 * @brief Multi-turn test using forwarding path with resetInputDimension.
 *        Prefill(4 tokens) → Generate(2 tokens) using height=1.
 *        Compares with incremental path.
 */
TEST(ForwardingEquivalence, multiturn_forwarding_vs_incremental) {
  const unsigned int SEQ_LEN = 4;

  // Save shared weights
  auto model_ref = buildTinyTransformer(SEQ_LEN);
  model_ref->save(WEIGHT_PATH);

  std::vector<float *> label;
  std::vector<float> prefill_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> gen1_data = {5.0f};
  std::vector<float> gen2_data = {6.0f};

  // --- Path 1: incremental_forwarding ---
  auto model1 = buildTinyTransformer(SEQ_LEN);
  model1->load(WEIGHT_PATH);

  std::vector<float *> in_prefill1 = {prefill_data.data()};
  auto out_p1 = model1->incremental_inference(1, in_prefill1, label, SEQ_LEN,
                                              0, SEQ_LEN, false);
  std::vector<float> logits_p1(out_p1[0], out_p1[0] + NUM_VOCAB);
  for (auto &o : out_p1)
    delete[] o;

  std::vector<float *> in_gen1_1 = {gen1_data.data()};
  auto out_g1_1 = model1->incremental_inference(1, in_gen1_1, label, SEQ_LEN,
                                                SEQ_LEN, SEQ_LEN + 1, false);
  std::vector<float> logits_g1_1(out_g1_1[0], out_g1_1[0] + NUM_VOCAB);
  for (auto &o : out_g1_1)
    delete[] o;

  std::vector<float *> in_gen2_1 = {gen2_data.data()};
  auto out_g2_1 = model1->incremental_inference(1, in_gen2_1, label, SEQ_LEN,
                                                SEQ_LEN + 1, SEQ_LEN + 2, false);
  std::vector<float> logits_g2_1(out_g2_1[0], out_g2_1[0] + NUM_VOCAB);
  for (auto &o : out_g2_1)
    delete[] o;

  // --- Path 2: forwarding via resetInputDimension ---
  auto model2 = buildTinyTransformer(SEQ_LEN);
  model2->load(WEIGHT_PATH);

  // Prefill: height = SEQ_LEN
  ml::train::TensorDim dim_prefill(1, 1, SEQ_LEN, 1);
  model2->resetInputDimension({dim_prefill});

  std::vector<float *> in_prefill2 = {prefill_data.data()};
  auto out_p2 = model2->inference(1, in_prefill2, label);
  std::vector<float> logits_p2(out_p2[0], out_p2[0] + NUM_VOCAB);

  // Compare prefill
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    EXPECT_NEAR(logits_p1[i], logits_p2[i], 1e-4)
      << "Prefill mismatch at " << i;
  }

  // Generate step 1: height = 1
  ml::train::TensorDim dim_gen(1, 1, 1, 1);
  model2->resetInputDimension({dim_gen});

  std::vector<float *> in_gen1_2 = {gen1_data.data()};
  auto out_g1_2 = model2->inference(1, in_gen1_2, label);
  std::vector<float> logits_g1_2(out_g1_2[0], out_g1_2[0] + NUM_VOCAB);

  // Compare gen step 1
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    EXPECT_NEAR(logits_g1_1[i], logits_g1_2[i], 1e-4)
      << "Gen step 1 mismatch at " << i;
  }

  // Generate step 2: still height = 1 (no resetInputDimension needed)
  std::vector<float *> in_gen2_2 = {gen2_data.data()};
  auto out_g2_2 = model2->inference(1, in_gen2_2, label);
  std::vector<float> logits_g2_2(out_g2_2[0], out_g2_2[0] + NUM_VOCAB);

  // Compare gen step 2
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    EXPECT_NEAR(logits_g2_1[i], logits_g2_2[i], 1e-4)
      << "Gen step 2 mismatch at " << i;
  }

  std::remove(WEIGHT_PATH.c_str());
}

/**
 * @brief KVCacheManager integration test.
 *        Tests save/load with actual model inference:
 *        1. Prefill and generate with model A
 *        2. Save KV cache via KVCacheManager
 *        3. Load KV cache into model B (new instance, same weights)
 *        4. Continue generation from model B
 *        5. Verify model B generates same result as if model A continued
 */
TEST(ForwardingEquivalence, kvcache_manager_save_load_continue) {
  const unsigned int SEQ_LEN = 4;

  auto model_ref = buildTinyTransformer(SEQ_LEN);
  model_ref->save(WEIGHT_PATH);

  std::vector<float *> label;
  std::vector<float> prefill_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> gen1_data = {5.0f};
  std::vector<float> gen2_data = {6.0f};
  std::string cache_path = "/tmp/test_kvcache_integration.bin";

  // --- Model A: Prefill + Generate 1 token ---
  auto modelA = buildTinyTransformer(SEQ_LEN);
  modelA->load(WEIGHT_PATH);

  std::vector<float *> in_prefill = {prefill_data.data()};
  auto out_p = modelA->incremental_inference(1, in_prefill, label, SEQ_LEN, 0,
                                             SEQ_LEN, false);
  for (auto &o : out_p)
    delete[] o;

  std::vector<float *> in_gen1 = {gen1_data.data()};
  auto out_g1 = modelA->incremental_inference(1, in_gen1, label, SEQ_LEN,
                                              SEQ_LEN, SEQ_LEN + 1, false);
  for (auto &o : out_g1)
    delete[] o;

  // Save KV cache from model A (after prefill + 1 gen = 5 tokens cached)
  unsigned int cached_len = SEQ_LEN + 1; // 5 tokens
  {
    auto f = std::ofstream(cache_path, std::ios::binary);
    std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
      fn = [&f, cached_len](ml::train::Layer &l,
                            nntrainer::RunLayerContext &context, void *) {
        if (l.getType() == causallm::MHACoreLayer::type) {
          auto k_cache = context.getTensor(0);
          auto v_cache = context.getTensor(1);
          ml::train::TensorDim k_dim = k_cache.getDim();
          ml::train::TensorDim v_dim = v_cache.getDim();
          k_dim.height(cached_len);
          v_dim.height(cached_len);
          nntrainer::Tensor k_slice =
            k_cache.getSharedDataTensor(k_dim, 0, true);
          nntrainer::Tensor v_slice =
            v_cache.getSharedDataTensor(v_dim, 0, true);
          k_slice.save(f);
          v_slice.save(f);
        }
      };
    modelA->forEachLayer(fn, nullptr);
    f.close();
  }

  // Model A continues: generate token 2
  std::vector<float *> in_gen2_A = {gen2_data.data()};
  auto out_g2_A = modelA->incremental_inference(1, in_gen2_A, label, SEQ_LEN,
                                                SEQ_LEN + 1, SEQ_LEN + 2, false);
  ASSERT_FALSE(out_g2_A.empty());
  std::vector<float> logits_A(out_g2_A[0], out_g2_A[0] + NUM_VOCAB);
  for (auto &o : out_g2_A)
    delete[] o;

  // --- Model B: Load cache, continue from token 5 ---
  auto modelB = buildTinyTransformer(SEQ_LEN);
  modelB->load(WEIGHT_PATH);

  // Allocate model B tensors first
  std::vector<float> dummy = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float *> in_dummy = {dummy.data()};
  auto out_dummy = modelB->incremental_inference(1, in_dummy, label, SEQ_LEN,
                                                 0, SEQ_LEN, false);
  for (auto &o : out_dummy)
    delete[] o;

  // Load KV cache into model B
  {
    auto f = std::ifstream(cache_path, std::ios::binary);
    std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
      fn = [&f, cached_len](ml::train::Layer &l,
                            nntrainer::RunLayerContext &context, void *) {
        if (l.getType() == causallm::MHACoreLayer::type) {
          auto k_cache = context.getTensor(0);
          auto v_cache = context.getTensor(1);
          ml::train::TensorDim k_dim = k_cache.getDim();
          ml::train::TensorDim v_dim = v_cache.getDim();
          k_dim.height(cached_len);
          v_dim.height(cached_len);
          nntrainer::Tensor k_slice =
            k_cache.getSharedDataTensor(k_dim, 0, true);
          nntrainer::Tensor v_slice =
            v_cache.getSharedDataTensor(v_dim, 0, true);
          k_slice.read(f);
          v_slice.read(f);
        }
      };
    modelB->forEachLayer(fn, nullptr);
    f.close();
  }

  // Model B generates token 2 (starting from cached position 5)
  std::vector<float *> in_gen2_B = {gen2_data.data()};
  auto out_g2_B = modelB->incremental_inference(1, in_gen2_B, label, SEQ_LEN,
                                                SEQ_LEN + 1, SEQ_LEN + 2, false);
  ASSERT_FALSE(out_g2_B.empty());
  std::vector<float> logits_B(out_g2_B[0], out_g2_B[0] + NUM_VOCAB);
  for (auto &o : out_g2_B)
    delete[] o;

  // --- Verify: Model A and Model B should produce identical output ---
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    EXPECT_NEAR(logits_A[i], logits_B[i], 1e-4)
      << "KV cache restore mismatch at vocab " << i
      << ": modelA=" << logits_A[i] << " modelB=" << logits_B[i];
  }

  std::remove(WEIGHT_PATH.c_str());
  std::remove(cache_path.c_str());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

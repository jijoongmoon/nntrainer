// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_lfm2.cpp
 * @date   19 June 2026
 * @brief  Tiny LFM2 CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <lfm2_causallm.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

constexpr int tiny_lfm2_num_layers = 2;

/**
 * @brief Tiny LFM2 CausalLM adapter for common model tests
 */
using TinyLfm2CausalLM =
  causallm_test::CausalLMTestAdapter<causallm::Lfm2CausalLM>;

/**
 * @brief Populate deterministic tiny LFM2 weights for golden token tests
 *
 * Zero all FC weights; set RMS norm scales to 1; set embedding[1][0]=1,
 * embedding[4][0]=2. This produces analytically known prefill logits.
 */
void setupLfm2DeterministicWeights(TinyLfm2CausalLM &model) {
  model.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "output_of_causallm")
        return;

      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        weight.setValue(0.0f);
        if (layer.getType() == "rms_norm" ||
            layer.getType() == "reshaped_rms_norm") {
          weight.setValue(1.0f);
        } else if (layer.getName() == "embedding0") {
          weight.setValue(0.0f);
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        }
      }
    });
}

/**
 * @brief Make the tiny LFM2 model config
 *
 * Uses layer_types=["attention","conv"] to exercise both the attention and
 * conv hybrid paths with a single tiny model.
 */
causallm::json makeTinyLfm2Config() {
  return {
    {"architectures", {"Lfm2ForCausalLM"}},
    {"bos_token_id", 0},
    {"conv_L_cache", 3},
    {"conv_bias", false},
    {"conv_dim", 64},
    {"conv_dim_out", 64},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"layer_types", {"attention", "conv"}},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", tiny_lfm2_num_layers},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-6},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Make the expected tiny LFM2 prefill logits
 *
 * Derivation: all FC weights = 0, all RMS norm scales = 1.
 *   layer0 (attention): residual carries embedding[tok4]=[2,0..] unchanged.
 *   layer1 (conv):      explicit residual (input.add(proj_back)) does the same.
 *   output_norm on [2,0..0] (64-dim): RMS≈0.25, normed≈[8,0..0].
 *   tied LM head: logit[j] ≈ 8*emb[j][0].
 *   LFM2's Q/K reshaped_rms_norm reduces numerical error vs Qwen family:
 *   logit[1] ≈ 7.9999361, logit[4] ≈ 15.9998722 (empirically confirmed).
 */
std::vector<float> makeExpectedLfm2Logits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 7.9999361f;
  logits[4] = 15.9998722f;
  return logits;
}

/**
 * @brief Make the tiny LFM2 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeLfm2LayerDtypeMap(const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32")
    dtype_map["embedding0"] =
      causallm_test::toTensorDataType(data_type.embedding_dtype);

  if (data_type.fc_layer_dtype != "FP32") {
    const auto dtype =
      causallm_test::toTensorDataType(data_type.fc_layer_dtype);
    // layer0: attention block FC layers
    dtype_map["layer0_wq"] = dtype;
    dtype_map["layer0_wk"] = dtype;
    dtype_map["layer0_wv"] = dtype;
    dtype_map["layer0_attention_out"] = dtype;
    dtype_map["layer0_ffn_up"] = dtype;
    dtype_map["layer0_ffn_gate"] = dtype;
    dtype_map["layer0_ffn_down"] = dtype;
    // layer1: conv block FC layers (causal_conv1d is always FP32 by design)
    dtype_map["layer1_conv_in_proj"] = dtype;
    dtype_map["layer1_conv_out_proj"] = dtype;
    dtype_map["layer1_ffn_up"] = dtype;
    dtype_map["layer1_ffn_gate"] = dtype;
    dtype_map["layer1_ffn_down"] = dtype;
  }

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  return dtype_map;
}

/**
 * @brief Make a LFM2 tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makeLfm2Case(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "LFM2_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedLfm2Logits(),
     data_type.name == "FP32" ? 1e-4f : 1e-3f},
    makeTinyLfm2Config,
    makeLfm2LayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyLfm2CausalLM>(cfg, generation_cfg, nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupLfm2DeterministicWeights(static_cast<TinyLfm2CausalLM &>(runner));
    },
  };
}

/**
 * @brief Parameterized fixture for tiny LFM2 model cases
 */
class Lfm2TinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "Lfm2TinyModelTest";
    std::string test_name = "Unknown";

    if (info != nullptr) {
      suite_name = info->test_suite_name();
      test_name = info->name();
    }

    return causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                GetParam().name);
  }
};

TEST_P(Lfm2TinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto model =
    GetParam().create_model(config.model, config.generation, config.nntrainer);

  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

TEST_P(Lfm2TinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

TEST_P(Lfm2TinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

INSTANTIATE_TEST_SUITE_P(
  LFM2, Lfm2TinyModelTest,
  ::testing::Values(makeLfm2Case(causallm_test::makeTinyFp32DataType()),
                    makeLfm2Case(causallm_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

/**
 * @brief Prompt-budget regime tests for the Lfm2CausalLM entry points
 *
 * The committed fixture (init_seq_len 4, max_seq_len 8, num_to_generate 1)
 * gives min(INIT_SEQ_LEN = 4, promptTokenBudget(8, 1, 0, false) = 6) = 4, i.e.
 * exactly the old INIT_SEQ_LEN-only clamp, so every test above exercises the
 * UNCHANGED branch. These tests build the regime the clamp was changed for,
 *
 *   max_seq_len < init_seq_len + num_to_generate + 1
 *
 * which the class's own checks permit -- run_with_embeddings() rejects only
 * n_tokens > INIT_SEQ_LEN and MAX_SEQ_LEN < INIT_SEQ_LEN -- and which no
 * committed configuration reaches.
 *
 * Both entry points are covered: run(), whose clamp now yields one prompt
 * token so the generation loop keeps its whole budget, and
 * run_with_embeddings(), whose caller-chosen n_tokens is refused by name at
 * the value that would otherwise write past the token-history row stride.
 *
 * These cases need USE_EMBEDDING = true (that is the only path on which
 * Lfm2CausalLM::run() applies its own clamp; with it false, run() delegates
 * straight to CausalLM::run()). Two consequences shape the fixture:
 *
 *  - embedding0 is then disconnected and pruned from the graph, so the fed
 *    embeddings come only from embedding_bin_path. Its contents mirror
 *    setupLfm2DeterministicWeights().
 *  - tie_word_embeddings must be false. With it true AND USE_EMBEDDING true,
 *    save_weight() omits the LM head's record (it still carries
 *    shared_from = embedding0) while load_weight() expects it, and the
 *    round-trip fails with a layout mismatch -- a pre-existing asymmetry of
 *    that combination, unrelated to the budget. The untied head below is
 *    given the tied head's values by hand, so the arithmetic is unchanged.
 */

/**
 * @brief Write the FP32 embedding bin matching setupLfm2DeterministicWeights
 *
 * NUM_VOCAB x DIM row-major floats: embedding[1][0] = 1, embedding[4][0] = 2,
 * every other entry zero. With all FC weights zero the residual carries the
 * LAST position's embedding unchanged, so the LM head yields
 * logit[j] = 8 * emb[j][0] and greedy decoding always selects token 4.
 */
std::filesystem::path
writeTinyLfm2EmbeddingBin(const std::filesystem::path &dir) {
  constexpr size_t vocab = 32;
  constexpr size_t dim = 64;
  std::vector<float> weights(vocab * dim, 0.0f);
  weights[1 * dim + 0] = 1.0f;
  weights[4 * dim + 0] = 2.0f;

  const auto path = dir / "lfm2_tiny_embedding.bin";
  std::ofstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("failed to open " + path.string());
  file.write(reinterpret_cast<const char *>(weights.data()),
             static_cast<std::streamsize>(weights.size() * sizeof(float)));
  if (!file.good())
    throw std::runtime_error("failed to write " + path.string());
  return path;
}

/**
 * @brief Deterministic weights for the untied, embedding-bypass LFM2 model
 *
 * Same as setupLfm2DeterministicWeights() except that the LM head is written
 * explicitly instead of being tied: its weight is [DIM, NUM_VOCAB], and the
 * normalized residual is [8, 0..0], so W[0][1] = 1 and W[0][4] = 2 reproduce
 * logit[1] = 8 and logit[4] = 16 exactly as the tied head does.
 */
void setupLfm2UntiedDeterministicWeights(TinyLfm2CausalLM &model) {
  model.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        weight.setValue(0.0f);
        if (layer.getType() == "rms_norm" ||
            layer.getType() == "reshaped_rms_norm") {
          weight.setValue(1.0f);
        } else if (layer.getName() == "output_of_causallm") {
          weight.setValue(0, 0, 0, 1, 1.0f);
          weight.setValue(0, 0, 0, 4, 2.0f);
        }
      }
    });
}

/**
 * @brief Make an LFM2 case in a chosen (init, max, num_to_generate) regime
 */
causallm_test::TinyCausalLMCase
makeLfm2BudgetCase(const std::string &name, unsigned int init_seq_len,
                   unsigned int max_seq_len, int num_to_generate,
                   const std::filesystem::path &embedding_bin) {
  auto test_case = makeLfm2Case(causallm_test::makeTinyFp32DataType());
  test_case.name = name;
  test_case.make_model_config = []() {
    auto cfg = makeTinyLfm2Config();
    cfg["tie_word_embeddings"] = false;
    return cfg;
  };
  test_case.setup_weights = [](causallm_test::TinyCausalLMRunner &runner) {
    setupLfm2UntiedDeterministicWeights(
      static_cast<TinyLfm2CausalLM &>(runner));
  };
  test_case.make_nntrainer_config =
    [init_seq_len, max_seq_len, num_to_generate,
     embedding_bin](const std::filesystem::path &tokenizer_path,
                    const causallm_test::TinyCausalLMDataType &data_type) {
      auto nntr_cfg =
        causallm_test::makeTinyNntrainerConfig(tokenizer_path, data_type);
      nntr_cfg["init_seq_len"] = init_seq_len;
      nntr_cfg["max_seq_len"] = max_seq_len;
      nntr_cfg["num_to_generate"] = num_to_generate;
      nntr_cfg["use_embedding"] = true;
      nntr_cfg["embedding_bin_path"] = embedding_bin.string();
      return nntr_cfg;
    };
  return test_case;
}

/**
 * @brief Build a deterministic LFM2 model with weights saved and re-loaded
 *
 * Mirrors the file-local makeLoadedDeterministicModel() of causallm_test_utils
 * but returns the concrete adapter, so a test can reach run_with_embeddings()
 * and getGeneratedIds() directly. The load is not optional here: it is what
 * populates the lookupEmbedding() cache from embedding_bin_path.
 */
std::unique_ptr<TinyLfm2CausalLM>
makeLoadedLfm2(const causallm_test::TinyCausalLMCase &test_case,
               const causallm_test::TinyCausalLMFiles &files) {
  auto source_config =
    causallm_test::makeTinyCausalLMConfig(test_case, files.tokenizer_path);
  auto source = std::make_unique<TinyLfm2CausalLM>(
    source_config.model, source_config.generation, source_config.nntrainer);
  source->initializeModel();
  test_case.setup_weights(*source);
  source->saveWeightWithDtype(
    files.weight_path.string(),
    test_case.make_layer_dtype_map(test_case.data_type));

  auto loaded_config =
    causallm_test::makeTinyCausalLMConfig(test_case, files.tokenizer_path);
  auto loaded = std::make_unique<TinyLfm2CausalLM>(
    loaded_config.model, loaded_config.generation, loaded_config.nntrainer);
  loaded->initializeModel();
  loaded->loadWeight(files.weight_path.string());
  return loaded;
}

/**
 * @brief run(): a window tighter than init + generate + 1 keeps the budget
 *
 * init_seq_len 4, max_seq_len 8, num_to_generate 4: 8 < 4 + 4 + 1, so
 * promptTokenBudget(8, 4, 0, false) = 3, one below INIT_SEQ_LEN.
 *
 * The prompt "world hello tok4 hello tok5" is 5 tokens, longer than either
 * clamp, so the two differ:
 *
 *   clamp                       prompt  loop start  loop end          emitted
 *   INIT_SEQ_LEN = 4                 4           5   5 + min(3,4) = 8   1 + 3
 *   min(INIT, budget) = 3            3           4   4 + min(4,4) = 8   1 + 4
 *
 * Both end at the same absolute position; yielding one prompt token is what
 * turns 3 of the 4 requested generation steps into 4 of 4. Asserting the
 * generated count asserts exactly the token the window clamp used to eat.
 */
TEST(Lfm2GenerationBudgetTest, TightWindowDeliversTheWholeGenerationBudget) {
  const auto files = causallm_test::makeTinyCausalLMFiles(
    "Lfm2GenerationBudgetTest", "TightWindowDeliversTheWholeGenerationBudget",
    "LFM2_FP32");
  const auto embedding_bin = writeTinyLfm2EmbeddingBin(files.dir);
  const auto test_case =
    makeLfm2BudgetCase("LFM2_budget_4_8_4", 4, 8, 4, embedding_bin);

  auto model = makeLoadedLfm2(test_case, files);
  ASSERT_NO_THROW(model->runPrompt("world hello tok4 hello tok5"));

  // One token sampled by the prefill + the full num_to_generate loop.
  EXPECT_EQ(model->getGeneratedIds().size(), 5u);
  for (unsigned int id : model->getGeneratedIds())
    EXPECT_EQ(id, 4u);

  // The prompt was clamped to 3, so the first generated token sits at column 3
  // and the loop fills 4..7: the row stride is used to the end and nothing is
  // written past it.
  EXPECT_EQ(model->tokenAt(0), 2u); // "world"
  EXPECT_EQ(model->tokenAt(1), 1u); // "hello"
  EXPECT_EQ(model->tokenAt(2), 4u); // "tok4"
  for (size_t pos = 3; pos < 8; ++pos)
    EXPECT_EQ(model->tokenAt(pos), 4u) << "at token-history column " << pos;
}

/**
 * @brief run_with_embeddings(): n_tokens == INIT_SEQ_LEN == MAX_SEQ_LEN refused
 *
 * The class's own checks permit that configuration, and it makes the PREFILL
 * store its sampled token at ids_history[MAX_SEQ_LEN] -- the next batch row,
 * or past the end of the allocation for the last one. Refused by name at the
 * entry point, before any work and before the over-budget warning, so the
 * warning never promises a run that then throws.
 */
TEST(Lfm2GenerationBudgetTest, EmbeddingsAtTheRowStrideAreRefusedByName) {
  const auto files = causallm_test::makeTinyCausalLMFiles(
    "Lfm2GenerationBudgetTest", "EmbeddingsAtTheRowStrideAreRefusedByName",
    "LFM2_FP32");
  const auto embedding_bin = writeTinyLfm2EmbeddingBin(files.dir);
  // init_seq_len == max_seq_len == 4, num_to_generate 2 = maxNumToGenerate(4,0)
  const auto test_case =
    makeLfm2BudgetCase("LFM2_budget_4_4_2", 4, 4, 2, embedding_bin);

  auto model = makeLoadedLfm2(test_case, files);

  constexpr size_t dim = 64;
  std::vector<float> embeds(4 * dim, 0.0f);
  embeds[3 * dim + 0] = 2.0f; // last position carries embedding[4]
  const std::vector<int> seed_tokens{2, 1, 4, 4};

  try {
    model->run_with_embeddings(embeds.data(), 4, seed_tokens, false, false);
    FAIL() << "run_with_embeddings accepted n_tokens == max_seq_len";
  } catch (const std::invalid_argument &e) {
    const std::string what(e.what());
    EXPECT_NE(what.find("run_with_embeddings"), std::string::npos) << what;
    EXPECT_NE(what.find("must be less than max_seq_len"), std::string::npos)
      << what;
  }

  // One token below the stride is the degraded-but-completed band: it warns,
  // runs to the end and emits only the token the prefill sampled, because
  // generation_begin == 4 is already the window. That is what the warning
  // says will happen, and now the run keeps that promise.
  std::vector<float> shorter(3 * dim, 0.0f);
  shorter[2 * dim + 0] = 2.0f;
  ASSERT_NO_THROW(
    model->run_with_embeddings(shorter.data(), 3, {2, 1, 4}, false, false));
  EXPECT_EQ(model->getGeneratedIds().size(), 1u);
  EXPECT_EQ(model->tokenAt(3), 4u);
}
} // namespace

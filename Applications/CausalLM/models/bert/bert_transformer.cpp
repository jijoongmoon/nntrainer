// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   bert_transformer.h
 * @date   29 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Please refer to the following code :
 *  https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/bert/modeling_bert.py
 */

#include <bert_transformer.h>
#include <llm_util.hpp>
#include <model.h>

#include <app_context.h>

namespace causallm {

namespace {
/**
 * @brief Convert a float to a string with enough precision to preserve
 * values as small as BERT's layer_norm_eps (1e-12).
 */
std::string toStringPrecise(float v) {
  std::ostringstream oss;
  oss << std::setprecision(20) << v;
  return oss.str();
}
} // namespace

json &BertTransformer::sanitizeConfig(json &cfg) {
  if (!cfg.contains("rope_theta")) {
    cfg["rope_theta"] = 0u;
  }

  if (!cfg.contains("rms_norm_eps")) {
    float layer_norm_eps = cfg.value("layer_norm_eps", 1e-12f);
    cfg["rms_norm_eps"] = layer_norm_eps;
  }

  if (!cfg.contains("tie_word_embeddings")) {
    cfg["tie_word_embeddings"] = false;
  }

  if (!cfg.contains("use_bidirectional_attention") &&
      !cfg.contains("is_causal")) {
    cfg["is_causal"] = false;
  }

  if (!cfg.contains("num_key_value_heads")) {
    cfg["num_key_value_heads"] = cfg["num_attention_heads"];
  }

  return cfg;
}

void BertTransformer::setupParameters(json &cfg, json &generation_cfg,
                                      json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
}

void BertTransformer::constructModel() {

  std::vector<LayerHandle> layers;

  // create model
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  /** --------- Inputs --------- */
  layers.push_back(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  layers.push_back(createLayer(
    "input", {withKey("name", "position_ids"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  layers.push_back(createLayer(
    "input", {withKey("name", "token_type_ids"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  /** --------- Token / Position / TokenType Embeddings --------- */
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  layers.push_back(
    createLayer(embedding_type,
                {withKey("name", "embedding0"), withKey("in_dim", NUM_VOCAB),
                 withKey("weight_dtype", EMBEDDING_DTYPE),
                 withKey("out_dim", DIM), withKey("input_layers", "input0")}));

  layers.push_back(
    createLayer("embedding_layer", {withKey("name", "position_embedding"),
                                    withKey("in_dim", MAX_POSITION_EMBEDDINGS),
                                    withKey("weight_dtype", EMBEDDING_DTYPE),
                                    withKey("out_dim", DIM),
                                    withKey("input_layers", "position_ids")}));

  layers.push_back(createLayer("embedding_layer",
                               {withKey("name", "token_type_embedding"),
                                withKey("in_dim", TYPE_VOCAB_SIZE),
                                withKey("weight_dtype", EMBEDDING_DTYPE),
                                withKey("out_dim", DIM),
                                withKey("input_layers", "token_type_ids")}));

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "embedding_sum"),
     withKey("input_layers",
             "embedding0,position_embedding,token_type_embedding")}));

  layers.push_back(createLayer("layer_normalization",
                               {withKey("name", "embedding_norm"),
                                withKey("epsilon", toStringPrecise(NORM_EPS)),
                                withKey("axis", 3), withKey("packed", "false"),
                                withKey("input_layers", "embedding_sum")}));

  /** --------- Encoder blocks --------- */
  for (int i = 0; i < NUM_LAYERS; ++i) {
    std::vector<LayerHandle> block;
    if (i == 0)
      block = createTransformerDecoderBlock(0, "embedding_norm");
    else
      block = createTransformerDecoderBlock(i, "layer" + std::to_string(i - 1) +
                                                 "_ffn_norm");
    layers.insert(layers.end(), block.begin(), block.end());
  }

  for (auto &layer : layers) {
    model->addLayer(layer);
  }
}

std::vector<LayerHandle>
BertTransformer::createTransformerDecoderBlock(const int layer_id,
                                               std::string input_name) {

  std::vector<LayerHandle> layers;

  // Self-attention sub-block
  auto att_layers = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                    input_name, input_name, input_name);
  layers.insert(layers.end(), att_layers.begin(), att_layers.end());

  // Residual (input + attention_out) + post LayerNorm
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_res"),
     withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                               "_attention_out")}));

  layers.push_back(createLayer(
    "layer_normalization",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", toStringPrecise(NORM_EPS)), withKey("axis", 3),
     withKey("packed", "false"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_attention_res")}));

  // Feed-forward sub-block
  auto ffn_layers =
    createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
              "layer" + std::to_string(layer_id) + "_attention_norm");
  layers.insert(layers.end(), ffn_layers.begin(), ffn_layers.end());

  // Residual (normed + ffn_down) + post LayerNorm
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_res"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_attention_norm,layer" +
                               std::to_string(layer_id) + "_ffn_down")}));

  layers.push_back(createLayer(
    "layer_normalization",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("epsilon", toStringPrecise(NORM_EPS)), withKey("axis", 3),
     withKey("packed", "false"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_res")}));

  return layers;
}

std::vector<LayerHandle>
BertTransformer::createAttention(const int layer_id, int seq_len, int n_heads,
                                 int head_dim, std::string query_name,
                                 std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;
  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // Q layer (bias enabled for BERT)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", Q), withKey("unit", head_dim * n_heads),
     withKey("disable_bias", "false"), withKey("input_layers", query_name),
     withKey("weight_initializer", "ones")}));

  // K layer (bias enabled for BERT)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("input_layers", key_name),
     withKey("weight_initializer", "ones")}));

  // V layer (bias enabled for BERT)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("input_layers", value_name),
     withKey("weight_initializer", "ones")}));

  // Attention core layer (bidirectional, no RoPE)
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN)),
    withKey("rope_theta", ROPE_THETA),
    withKey("is_causal", "false"),
    withKey("input_layers", {Q, K, V})};
  layers.push_back(createLayer("mha_core", a_params));

  // O layer (bias enabled for BERT)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", O), withKey("unit", DIM), withKey("disable_bias", "false"),
     withKey("input_layers", A), withKey("weight_initializer", "ones")}));

  return layers;
}

std::vector<LayerHandle> BertTransformer::createMlp(const int layer_id, int dim,
                                                    int hidden_dim,
                                                    std::string input_name) {

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_fc1"),
     withKey("unit", hidden_dim), withKey("disable_bias", "false"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));

  layers.push_back(createLayer(
    "activation",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_act"),
     withKey("activation", "gelu"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_fc1")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "false"),
     withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_act"),
     withKey("weight_initializer", "ones")}));

  return layers;
}

void BertTransformer::run(const WSTR prompt, bool do_sample,
                          const WSTR system_prompt, const WSTR tail_prompt,
                          bool log_output) {

  try {
    std::vector<float *> results = encode(prompt, system_prompt, tail_prompt);

    if (log_output) {

      std::cout << "Embedding Result (" << BATCH_SIZE
                << " batch(es)):" << std::endl;
      for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
        std::cout << "Batch " << b << ": [";
        // Print first few elements as sample
        int print_dim = (DIM > 10) ? 10 : DIM;
        for (int i = 0; i < print_dim; ++i) {
          std::cout << results[0][b * DIM + i]
                    << (i == print_dim - 1 ? "" : ", ");
        }
        if (DIM > 10)
          std::cout << ", ...";
        std::cout << "] (Total DIM: " << DIM << ")" << std::endl;
      }
    }

    // output should be deallocated after use.
    for (auto out : results) {
      delete[] out;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error during embedding run: " << e.what() << std::endl;
  }
}

void BertTransformer::registerCustomLayers() {
  Transformer::registerCustomLayers();
}

} // namespace causallm

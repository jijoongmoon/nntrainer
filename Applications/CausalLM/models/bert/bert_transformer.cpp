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
#include <iomanip>
#include <llm_util.hpp>
#include <model.h>
#include <sstream>

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

std::pair<Tensor, Tensor> BertTransformer::constructModel() {

  /** --------- Inputs --------- */
  Tensor x =
    Tensor({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");
  position_ids_input =
    Tensor({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "position_ids");
  token_type_ids_input = Tensor(
    {1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "token_type_ids");

  /** --------- Token / Position / TokenType Embeddings --------- */
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  LayerHandle word_emb(createLayer(
    embedding_type,
    {withKey("name", "embedding0"), withKey("in_dim", NUM_VOCAB),
     withKey("weight_dtype", EMBEDDING_DTYPE), withKey("out_dim", DIM)}));
  Tensor word = word_emb(x);

  LayerHandle pos_emb(
    createLayer("embedding_layer", {withKey("name", "position_embedding"),
                                    withKey("in_dim", MAX_POSITION_EMBEDDINGS),
                                    withKey("weight_dtype", EMBEDDING_DTYPE),
                                    withKey("out_dim", DIM)}));
  Tensor pos = pos_emb(position_ids_input);

  LayerHandle type_emb(
    createLayer("embedding_layer", {withKey("name", "token_type_embedding"),
                                    withKey("in_dim", TYPE_VOCAB_SIZE),
                                    withKey("weight_dtype", EMBEDDING_DTYPE),
                                    withKey("out_dim", DIM)}));
  Tensor type_e = type_emb(token_type_ids_input);

  // Sum the three embeddings (Addition takes a list of inputs).
  LayerHandle sum_layer(
    createLayer("addition", {withKey("name", "embedding_sum")}));
  Tensor summed = sum_layer({word, pos, type_e});

  // First LayerNorm (post-embedding)
  LayerHandle emb_norm(createLayer(
    "layer_normalization", {withKey("name", "embedding_norm"),
                            withKey("epsilon", toStringPrecise(NORM_EPS)),
                            withKey("axis", 3), withKey("packed", "false")}));
  Tensor h = emb_norm(summed);

  /** --------- Encoder blocks --------- */
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  return {x, h};
}

Tensor BertTransformer::createTransformerDecoderBlock(const int layer_id,
                                                      Tensor input) {

  // Self-attention sub-block (Q=K=V=input for self-attention)
  Tensor att = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                               input, input, input);

  // Residual (input + attention_out) + post LayerNorm
  LayerHandle att_res(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_res")}));
  Tensor att_res_t = att_res({input, att});

  LayerHandle att_norm(createLayer(
    "layer_normalization",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", toStringPrecise(NORM_EPS)), withKey("axis", 3),
     withKey("packed", "false")}));
  Tensor att_norm_t = att_norm(att_res_t);

  // Feed-forward sub-block
  Tensor ffn = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, att_norm_t);

  // Residual (normed + ffn_down) + post LayerNorm
  LayerHandle ffn_res(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_res")}));
  Tensor ffn_res_t = ffn_res({att_norm_t, ffn});

  LayerHandle ffn_norm(createLayer(
    "layer_normalization",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("epsilon", toStringPrecise(NORM_EPS)), withKey("axis", 3),
     withKey("packed", "false")}));
  return ffn_norm(ffn_res_t);
}

Tensor BertTransformer::createAttention(const int layer_id, int seq_len,
                                        int n_heads, int head_dim, Tensor query,
                                        Tensor key, Tensor value) {

  // Q layer (bias enabled for BERT)
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));
  Tensor q = wq(query);

  // K layer (bias enabled for BERT)
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor k = wk(key);

  // V layer (bias enabled for BERT)
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor v = wv(value);

  // Attention core layer (bidirectional, no RoPE — BERT is an encoder)
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(INIT_SEQ_LEN)),
     withKey("rope_theta", ROPE_THETA), withKey("is_causal", "false")}));
  Tensor a = mha({q, k, v});

  // O layer (bias enabled for BERT)
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));
  return wo(a);
}

Tensor BertTransformer::createMlp(const int layer_id, int dim, int hidden_dim,
                                  Tensor input) {

  LayerHandle fc1(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_fc1"),
     withKey("unit", hidden_dim), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));
  Tensor mid = fc1(input);

  LayerHandle act(createLayer(
    "activation",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_act"),
     withKey("activation", "gelu")}));
  Tensor act_t = act(mid);

  LayerHandle fc2(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));
  return fc2(act_t);
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

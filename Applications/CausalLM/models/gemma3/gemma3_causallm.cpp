// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file	gemma3_causallm.cpp
 * @date	24 Dec 2025
 * @brief	This defines a gemma3 causal language model.
 * @see		https://github.com/nnstreamer/
 * @author	Seungbaek Hong <sb92.hong@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <gemma3_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>
#include <reshaped_rms_norm.h>

namespace causallm {

json &Gemma3Transformer::sanitizeConfig(json &cfg) {
  if (!cfg.contains("tie_word_embeddings")) {
    cfg["tie_word_embeddings"] = true;
  }
  return cfg;
}

json &Gemma3Transformer::sanitizeGenerationConfig(json &gen_cfg,
                                                  const json &cfg) {
  if (!gen_cfg.contains("eos_token_id")) {
    if (cfg.contains("eos_token_id")) {
      auto eos = cfg["eos_token_id"];
      if (eos.is_number()) {
        gen_cfg["eos_token_id"] =
          std::vector<unsigned int>{eos.get<unsigned int>()};
      } else {
        gen_cfg["eos_token_id"] = eos;
      }
    }
  } else {
    auto eos = gen_cfg["eos_token_id"];
    if (eos.is_number()) {
      gen_cfg["eos_token_id"] =
        std::vector<unsigned int>{eos.get<unsigned int>()};
    }
  }

  return gen_cfg;
}

void Gemma3Transformer::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);
  if (cfg.contains("layer_types")) {
    layer_types = cfg["layer_types"].get<std::vector<std::string>>();
  }
  if (cfg.contains("attn_logit_softcapping") &&
      !cfg["attn_logit_softcapping"].is_null()) {
    ATTN_LOGIT_SOFTCAPPING = cfg["attn_logit_softcapping"].get<float>();
  }
}

Tensor
Gemma3Transformer::createTransformerDecoderBlock(const int layer_id,
                                                  Tensor input) {

  using ml::train::createLayer;

  // Pre-attention norm
  LayerHandle att_norm = createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")});
  Tensor normed = att_norm(input);

  // Self attention
  Tensor att_out =
    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM, normed,
                    normed, normed);

  // Post-attention norm (Gemma3-specific)
  LayerHandle post_att_norm = createLayer(
    "rms_norm",
    {withKey("name",
             "layer" + std::to_string(layer_id) + "_post_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")});
  Tensor att_normed = post_att_norm(att_out);

  // Residual add (input + post_attention_norm(attention_out))
  Tensor post_att = input.add(att_normed);

  // Pre-FFN norm
  LayerHandle ffn_norm = createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "pre_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")});
  Tensor ffn_normed = ffn_norm(post_att);

  // Feed forward (GeGLU)
  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_normed);

  // Post-FFN norm (Gemma3-specific)
  LayerHandle post_ffn_norm = createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "post_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")});
  Tensor ffn_normed_out = post_ffn_norm(ffn_out);

  // Residual add (post_attention + post_ffn_norm(ffn_out))
  Tensor decoder_out = post_att.add(ffn_normed_out);

  return decoder_out;
}

Tensor Gemma3Transformer::createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           Tensor query, Tensor key,
                                           Tensor value) {

  using ml::train::createLayer;

  auto Q_name = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  auto K_name = "layer" + std::to_string(layer_id) + "_wk";
  auto K_norm = "layer" + std::to_string(layer_id) + "_k_norm";
  auto V_name = "layer" + std::to_string(layer_id) + "_wv";
  auto A_name = "layer" + std::to_string(layer_id) + "_attention";
  auto O_name = "layer" + std::to_string(layer_id) + "_attention_out";

  // V projection (Gemma3: no bias)
  LayerHandle v_proj = createLayer(
    "fully_connected",
    {withKey("name", V_name), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)});
  Tensor v = v_proj(value);

  // K projection (Gemma3: no bias)
  LayerHandle k_proj = createLayer(
    "fully_connected",
    {withKey("name", K_name), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)});
  Tensor k = k_proj(key);

  // K reshaped norm
  LayerHandle k_norm = createLayer(
    "reshaped_rms_norm",
    {withKey("name", K_norm), withKey("packed", "false"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))});
  Tensor k_normed = k_norm(k);

  // Q projection (Gemma3: no bias)
  LayerHandle q_proj = createLayer(
    "fully_connected",
    {withKey("name", Q_name), withKey("unit", head_dim * n_heads),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)});
  Tensor q = q_proj(query);

  // Q reshaped norm
  LayerHandle q_norm = createLayer(
    "reshaped_rms_norm",
    {withKey("name", Q_norm), withKey("packed", "false"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))});
  Tensor q_normed = q_norm(q);

  // Determine sliding window for this layer
  unsigned int window_size = UINT_MAX;
  if (!layer_types.empty()) {
    if (layer_id < static_cast<int>(layer_types.size())) {
      if (layer_types[layer_id] == "sliding_attention") {
        window_size = SLIDING_WINDOW;
      }
    }
  } else {
    window_size = SLIDING_WINDOW;
  }

  // Determine RoPE theta (sliding layers use 10000, global use config value)
  float rope_theta = ROPE_THETA;
  if (!layer_types.empty() &&
      layer_id < static_cast<int>(layer_types.size())) {
    if (layer_types[layer_id] == "sliding_attention") {
      rope_theta = 10000.0f;
    }
  }

  // Attention core layer
  std::vector<std::string> a_params = {
    withKey("name", A_name),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", window_size),
    withKey("rope_theta", std::to_string(rope_theta)),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("attn_logit_softcapping", std::to_string(ATTN_LOGIT_SOFTCAPPING)),
    withKey("is_causal", IS_CAUSAL ? "true" : "false")};
  LayerHandle attn = createLayer("mha_core", a_params);
  Tensor a = attn({q_normed, k_normed, v, key_cache_tensors[layer_id],
                   val_cache_tensors[layer_id]});

  // O projection (Gemma3: no bias)
  LayerHandle o_proj = createLayer(
    "fully_connected",
    {withKey("name", O_name), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)});
  Tensor o = o_proj(a);

  return o;
}

Tensor Gemma3Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                                     Tensor input) {

  using ml::train::createLayer;

  // Gate projection
  LayerHandle ffn_gate = createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)});
  Tensor gate = ffn_gate(input);

  // GeLU activation (Gemma3 uses tanh_gelu, not silu)
  LayerHandle gelu = createLayer(
    "activation",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate_gelu"),
     withKey("activation", "tanh_gelu")});
  Tensor gate_activated = gelu(gate);

  // Up projection
  LayerHandle ffn_up = createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)});
  Tensor up = ffn_up(input);

  // Multiply gate * up (GeGLU)
  LayerHandle geglu = createLayer(
    "multiply",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_geglu")});
  Tensor activated = geglu({gate_activated, up});

  // Down projection
  LayerHandle ffn_down = createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)});
  Tensor down = ffn_down(activated);

  return down;
}

void Gemma3Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void Gemma3CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Gemma3Transformer::registerCustomLayers();
}

} // namespace causallm

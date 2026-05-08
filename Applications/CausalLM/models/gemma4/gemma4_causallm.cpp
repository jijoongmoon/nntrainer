// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemma4_causallm.cpp
 * @date   07 Apr 2026
 * @brief  This defines a Gemma4 causal language model.
 * @see    https://github.com/nnstreamer/
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gemma4_causallm.h>

#include <algorithm>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <logit_softcapping.h>
#include <model.h>
#include <per_layer_slice.h>
#include <reshaped_rms_norm.h>
#include <scalar_multiply.h>

namespace causallm {

bool Gemma4Transformer::isKVSharedLayer(int layer_id) const {
  const int first_kv_shared_layer_idx = NUM_LAYERS - NUM_KV_SHARED_LAYERS;
  return layer_id >= first_kv_shared_layer_idx && first_kv_shared_layer_idx > 0;
}

void Gemma4Transformer::appendSkipPrefillIfNeeded(
  std::vector<std::string> &props, bool enable_skip) const {
  if (enable_skip && ENABLE_SKIP_PREFILL_OPT) {
    props.emplace_back(withKey("skip_prefill", "true"));
  }
}

json &Gemma4Transformer::sanitizeConfig(json &cfg) {
  if (cfg.contains("text_config") && cfg["text_config"].is_object()) {
    const auto &text_cfg = cfg["text_config"];
    for (auto it = text_cfg.begin(); it != text_cfg.end(); ++it) {
      if (!cfg.contains(it.key())) {
        cfg[it.key()] = it.value();
      }
    }
  }

  if (!cfg.contains("tie_word_embeddings")) {
    cfg["tie_word_embeddings"] = true;
  }

  if (!cfg.contains("head_dim") && cfg.contains("hidden_size") &&
      cfg.contains("num_attention_heads")) {
    cfg["head_dim"] = cfg["hidden_size"].get<unsigned int>() /
                      cfg["num_attention_heads"].get<unsigned int>();
  }

  return cfg;
}

json &Gemma4Transformer::sanitizeGenerationConfig(json &gen_cfg,
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

void Gemma4Transformer::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);

  if (cfg.contains("layer_types")) {
    layer_types = cfg["layer_types"].get<std::vector<std::string>>();
  }

  if (cfg.contains("attn_logit_softcapping") &&
      !cfg["attn_logit_softcapping"].is_null()) {
    ATTN_LOGIT_SOFTCAPPING = cfg["attn_logit_softcapping"].get<float>();
  }
  if (cfg.contains("final_logit_softcapping") &&
      !cfg["final_logit_softcapping"].is_null()) {
    FINAL_LOGIT_SOFTCAPPING = cfg["final_logit_softcapping"].get<float>();
  }

  GLOBAL_HEAD_DIM =
    cfg.contains("global_head_dim") && !cfg["global_head_dim"].is_null()
      ? cfg["global_head_dim"].get<unsigned int>()
      : HEAD_DIM;

  NUM_GLOBAL_KEY_VALUE_HEADS =
    cfg.contains("num_global_key_value_heads") &&
        !cfg["num_global_key_value_heads"].is_null()
      ? cfg["num_global_key_value_heads"].get<unsigned int>()
      : NUM_KEY_VALUE_HEADS;

  ATTENTION_K_EQ_V =
    cfg.contains("attention_k_eq_v") && cfg["attention_k_eq_v"].get<bool>();

  NNTR_THROW_IF(!cfg.contains("hidden_size_per_layer_input") ||
                  cfg["hidden_size_per_layer_input"].is_null() ||
                  cfg["hidden_size_per_layer_input"].get<unsigned int>() == 0,
                std::invalid_argument)
    << "[Gemma4] hidden_size_per_layer_input must be provided and > 0";
  NNTR_THROW_IF(!cfg.contains("vocab_size_per_layer_input") ||
                  cfg["vocab_size_per_layer_input"].is_null() ||
                  cfg["vocab_size_per_layer_input"].get<unsigned int>() == 0,
                std::invalid_argument)
    << "[Gemma4] vocab_size_per_layer_input must be provided and > 0";
  HIDDEN_SIZE_PER_LAYER_INPUT =
    cfg["hidden_size_per_layer_input"].get<unsigned int>();
  VOCAB_SIZE_PER_LAYER_INPUT =
    cfg["vocab_size_per_layer_input"].get<unsigned int>();

  FULL_ATTENTION_ROPE_THETA = ROPE_THETA;
  SLIDING_ATTENTION_ROPE_THETA = ROPE_THETA;
  FULL_ATTENTION_ROPE_TYPE = "default";
  SLIDING_ATTENTION_ROPE_TYPE = "default";
  FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR = 1.0f;
  SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR = 1.0f;

  NUM_KV_SHARED_LAYERS = cfg.contains("num_kv_shared_layers") &&
                             !cfg["num_kv_shared_layers"].is_null()
                           ? cfg["num_kv_shared_layers"].get<int>()
                           : 0;
  USE_DOUBLE_WIDE_MLP = cfg.contains("use_double_wide_mlp") &&
                        cfg["use_double_wide_mlp"].get<bool>();
  ENABLE_SKIP_PREFILL_OPT =
    nntr_cfg.contains("skip_prefill") && nntr_cfg["skip_prefill"].get<bool>();

  if (cfg.contains("rope_parameters") && cfg["rope_parameters"].is_object()) {
    const auto &rope_params = cfg["rope_parameters"];
    if (rope_params.contains("full_attention") &&
        rope_params["full_attention"].contains("rope_theta")) {
      FULL_ATTENTION_ROPE_THETA =
        rope_params["full_attention"]["rope_theta"].get<unsigned int>();
    }
    if (rope_params.contains("full_attention") &&
        rope_params["full_attention"].contains("rope_type") &&
        !rope_params["full_attention"]["rope_type"].is_null()) {
      FULL_ATTENTION_ROPE_TYPE =
        rope_params["full_attention"]["rope_type"].get<std::string>();
    }
    if (rope_params.contains("full_attention") &&
        rope_params["full_attention"].contains("partial_rotary_factor") &&
        !rope_params["full_attention"]["partial_rotary_factor"].is_null()) {
      FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR =
        rope_params["full_attention"]["partial_rotary_factor"].get<float>();
    }

    if (rope_params.contains("sliding_attention") &&
        rope_params["sliding_attention"].contains("rope_theta")) {
      SLIDING_ATTENTION_ROPE_THETA =
        rope_params["sliding_attention"]["rope_theta"].get<unsigned int>();
    }

    if (rope_params.contains("sliding_attention") &&
        rope_params["sliding_attention"].contains("rope_type") &&
        !rope_params["sliding_attention"]["rope_type"].is_null()) {
      SLIDING_ATTENTION_ROPE_TYPE =
        rope_params["sliding_attention"]["rope_type"].get<std::string>();
    }
    if (rope_params.contains("sliding_attention") &&
        rope_params["sliding_attention"].contains("partial_rotary_factor") &&
        !rope_params["sliding_attention"]["partial_rotary_factor"].is_null()) {
      SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR =
        rope_params["sliding_attention"]["partial_rotary_factor"].get<float>();
    }
  }

  EMBEDDING_SCALE = std::sqrt(static_cast<float>(DIM));
  EMBEDDING_PER_LAYER_SCALE =
    std::sqrt(static_cast<float>(HIDDEN_SIZE_PER_LAYER_INPUT));
}

void Gemma4Transformer::constructModel() {

  std::vector<LayerHandle> layers;

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  layers.push_back(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  layers.push_back(createLayer(
    embedding_type,
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));

  std::string decoder_input = "embedding0";

  const unsigned int per_layer_total_dim =
    NUM_LAYERS * HIDDEN_SIZE_PER_LAYER_INPUT;

  // try using same low bit precision as fc layers
  layers.push_back(createLayer(
    "embedding_layer",
    {withKey("name", "per_layer_input_embedding"),
     withKey("in_dim", std::to_string(VOCAB_SIZE_PER_LAYER_INPUT)),
     withKey("out_dim", std::to_string(per_layer_total_dim)),
     withKey("weight_dtype", FC_LAYER_DTYPE), withKey("input_layers", "input0"),
     withKey("scale", EMBEDDING_PER_LAYER_SCALE)}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "per_layer_input_projection"),
     withKey("unit", std::to_string(per_layer_total_dim)),
     withKey("disable_bias", "true"), withKey("input_layers", "embedding0"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  float ple_proj_scale = 1.0f / std::sqrt(static_cast<float>(DIM));
  layers.push_back(createLayer(
    "scalar_multiply", {
                         withKey("name", "per_layer_model_proj_scale"),
                         withKey("input_layers", "per_layer_input_projection"),
                         withKey("packed", "false"),
                         withKey("multiplier", std::to_string(ple_proj_scale)),
                       }));

  layers.push_back(createLayer(
    "reshaped_rms_norm",
    {
      withKey("name", "per_layer_projection_norm"),
      withKey("input_layers", "per_layer_model_proj_scale"),
      withKey("epsilon", std::to_string(NORM_EPS)),
      withKey("feature_size", std::to_string(HIDDEN_SIZE_PER_LAYER_INPUT)),
      withKey("packed", "false"),
    }));

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "per_layer_input_sum"),
     withKey("input_layers",
             "per_layer_input_embedding,per_layer_projection_norm")}));

  // TODO : change per_layer_input_scale to non hard-coded way

  float per_layer_input_scale = std::sqrt(0.5f);

  layers.push_back(
    createLayer("scalar_multiply",
                {
                  withKey("name", "per_layer_input_scale"),
                  withKey("input_layers", "per_layer_input_sum"),
                  withKey("packed", "false"),
                  withKey("multiplier", std::to_string(per_layer_input_scale)),
                }));

  for (int i = 0; i < NUM_LAYERS; ++i) {
    std::vector<LayerHandle> transformer =
      createTransformerDecoderBlock(i, decoder_input);
    layers.insert(layers.end(), transformer.begin(), transformer.end());
    decoder_input = "layer" + std::to_string(i) + "_layer_scalar";
  }

  std::vector<std::string> output_norm_props = {
    withKey("name", "output_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("input_layers", decoder_input), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(output_norm_props, true);
  layers.push_back(createLayer("rms_norm", output_norm_props));

  for (auto &layer : layers) {
    model->addLayer(layer);
  }
}

std::vector<LayerHandle>
Gemma4Transformer::createTransformerDecoderBlock(const int layer_id,
                                                 std::string input_name) {

  std::vector<LayerHandle> layers;

  // Gemma4TextRMSNorm scales by `weight` (initialized to ones), which matches
  // NNTrainer `rms_norm` behavior used here.
  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  std::vector<std::string> attn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
    withKey("input_layers", input_name),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(attn_norm_props, is_kv_shared_layer);
  layers.push_back(createLayer("rms_norm", attn_norm_props));

  int shared_kv_layer_id = -1;

  const int first_kv_shared_layer_idx = NUM_LAYERS - NUM_KV_SHARED_LAYERS;

  if (is_kv_shared_layer && !layer_types.empty() &&
      first_kv_shared_layer_idx <= static_cast<int>(layer_types.size())) {
    const auto &curr_layer_type = layer_types[layer_id];
    const std::vector<std::string> prev_layers(
      layer_types.begin(), layer_types.begin() + first_kv_shared_layer_idx);
    auto rev_it =
      std::find(prev_layers.rbegin(), prev_layers.rend(), curr_layer_type);
    NNTR_THROW_IF(rev_it == prev_layers.rend(), std::invalid_argument)
      << "[Gemma4] Could not find shared KV source layer for layer " << layer_id
      << " with layer_type=" << curr_layer_type;
    shared_kv_layer_id =
      static_cast<int>(prev_layers.size()) - 1 -
      static_cast<int>(std::distance(prev_layers.rbegin(), rev_it));
  }

  std::vector<LayerHandle> att_layer;
  if (shared_kv_layer_id >= 0) {
    att_layer = createSharedAttention(
      layer_id, shared_kv_layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
      "layer" + std::to_string(layer_id) + "_attention_norm");
  } else {
    att_layer =
      createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                      "layer" + std::to_string(layer_id) + "_attention_norm",
                      "layer" + std::to_string(layer_id) + "_attention_norm",
                      "layer" + std::to_string(layer_id) + "_attention_norm");
  }
  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  std::vector<std::string> post_attn_norm_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_post_attention_norm"),
    withKey("input_layers",
            "layer" + std::to_string(layer_id) + "_attention_out"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(post_attn_norm_props, is_kv_shared_layer);
  layers.push_back(createLayer("rms_norm", post_attn_norm_props));

  std::vector<std::string> post_attention_add_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_post_attention"),
    withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                              "_post_attention_norm")};
  appendSkipPrefillIfNeeded(post_attention_add_props, is_kv_shared_layer);
  layers.push_back(createLayer("addition", post_attention_add_props));

  std::vector<std::string> pre_ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_pre_ffn_norm"),
    withKey("input_layers",
            "layer" + std::to_string(layer_id) + "_post_attention"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(pre_ffn_norm_props, is_kv_shared_layer);
  layers.push_back(createLayer("rms_norm", pre_ffn_norm_props));

  auto ffn_layer =
    createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
              "layer" + std::to_string(layer_id) + "_pre_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  std::vector<std::string> post_ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(post_ffn_norm_props, is_kv_shared_layer);
  layers.push_back(createLayer("rms_norm", post_ffn_norm_props));

  const std::string decoder_output_name =
    "layer" + std::to_string(layer_id) + "_decoder_output_base";

  std::vector<std::string> decoder_output_base_props = {
    withKey("name", decoder_output_name),
    withKey("input_layers", "layer" + std::to_string(layer_id) +
                              "_post_attention,layer" +
                              std::to_string(layer_id) + "_post_ffn_norm")};
  appendSkipPrefillIfNeeded(decoder_output_base_props, is_kv_shared_layer);
  layers.push_back(createLayer("addition", decoder_output_base_props));

  // Select [B, S, hidden_size_per_layer_input] from packed per-layer input
  // [B, S, num_layers*hidden_size_per_layer_input]
  std::vector<std::string> per_layer_slice_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_per_layer_input"),
    withKey("input_layers", "per_layer_input_scale"),
    withKey("feature_size", std::to_string(HIDDEN_SIZE_PER_LAYER_INPUT)),
    withKey("layer_index", std::to_string(layer_id))};
  appendSkipPrefillIfNeeded(per_layer_slice_props, is_kv_shared_layer);
  layers.push_back(createLayer("per_layer_slice", per_layer_slice_props));

  std::vector<std::string> per_layer_input_gate_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_per_layer_input_gate"),
    withKey("unit", std::to_string(HIDDEN_SIZE_PER_LAYER_INPUT)),
    withKey("disable_bias", "true"),
    withKey("input_layers", decoder_output_name),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(per_layer_input_gate_props, is_kv_shared_layer);
  layers.push_back(createLayer("fully_connected", per_layer_input_gate_props));

  std::vector<std::string> per_layer_input_act_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_per_layer_input_act"),
    withKey("activation", "tanh_gelu"),
    withKey("input_layers",
            "layer" + std::to_string(layer_id) + "_per_layer_input_gate")};
  appendSkipPrefillIfNeeded(per_layer_input_act_props, is_kv_shared_layer);
  layers.push_back(createLayer("activation", per_layer_input_act_props));

  std::vector<std::string> per_layer_input_mul_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_per_layer_input_mul"),
    withKey("input_layers", "layer" + std::to_string(layer_id) +
                              "_per_layer_input_act,layer" +
                              std::to_string(layer_id) + "_per_layer_input")};
  appendSkipPrefillIfNeeded(per_layer_input_mul_props, is_kv_shared_layer);
  layers.push_back(createLayer("multiply", per_layer_input_mul_props));

  std::vector<std::string> per_layer_input_proj_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_per_layer_input_proj"),
    withKey("unit", std::to_string(DIM)),
    withKey("disable_bias", "true"),
    withKey("input_layers",
            "layer" + std::to_string(layer_id) + "_per_layer_input_mul"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(per_layer_input_proj_props, is_kv_shared_layer);
  layers.push_back(createLayer("fully_connected", per_layer_input_proj_props));

  std::vector<std::string> post_per_layer_input_norm_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_post_per_layer_input_norm"),
    withKey("input_layers",
            "layer" + std::to_string(layer_id) + "_per_layer_input_proj"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(post_per_layer_input_norm_props,
                            is_kv_shared_layer);
  layers.push_back(createLayer("rms_norm", post_per_layer_input_norm_props));

  std::vector<std::string> decoder_output_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
    withKey("input_layers", decoder_output_name + ",layer" +
                              std::to_string(layer_id) +
                              "_post_per_layer_input_norm")};
  appendSkipPrefillIfNeeded(decoder_output_props, is_kv_shared_layer);
  layers.push_back(createLayer("addition", decoder_output_props));

  std::vector<std::string> layer_scalar_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_layer_scalar"),
    withKey("input_layers",
            "layer" + std::to_string(layer_id) + "_decoder_output"),
    withKey("packed", "false"),
    withKey("use_weight", "true"),
  };
  appendSkipPrefillIfNeeded(layer_scalar_props, is_kv_shared_layer);
  layers.push_back(createLayer("scalar_multiply", layer_scalar_props));

  return layers;
}

std::vector<LayerHandle> Gemma4Transformer::createSharedAttention(
  const int layer_id, const int shared_kv_layer_id, int seq_len, int n_heads,
  int head_dim, std::string query_name) {
  std::vector<LayerHandle> layers;
  (void)seq_len;
  (void)head_dim;

  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";
  auto shared_K_norm = "layer" + std::to_string(shared_kv_layer_id) + "_k_norm";
  auto shared_V_norm = "layer" + std::to_string(shared_kv_layer_id) + "_v_norm";
  auto Q_scaled = "layer" + std::to_string(layer_id) + "_q_scaled";

  bool is_sliding = true;
  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  if (!layer_types.empty() && layer_id < static_cast<int>(layer_types.size())) {
    is_sliding = layer_types[layer_id] == "sliding_attention";
  }

  int curr_head_dim = is_sliding ? HEAD_DIM : GLOBAL_HEAD_DIM;
  int curr_kv_heads = (is_sliding || !ATTENTION_K_EQ_V)
                        ? NUM_KEY_VALUE_HEADS
                        : NUM_GLOBAL_KEY_VALUE_HEADS;

  // Q layer [B, S, H] -> [B, S, Nq*Dh]
  std::vector<std::string> q_params = {withKey("name", Q),
                                       withKey("unit", curr_head_dim * n_heads),
                                       withKey("disable_bias", "true"),
                                       withKey("input_layers", query_name),
                                       withKey("weight_initializer", "ones"),
                                       withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(q_params, is_kv_shared_layer);
  layers.push_back(createLayer("fully_connected", q_params));

  // q_norm on per-head projection [B, S, Nq*Dh]
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm), withKey("input_layers", Q),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim))};
  appendSkipPrefillIfNeeded(q_norm_params, is_kv_shared_layer);
  layers.push_back(createLayer("reshaped_rms_norm", q_norm_params));

  // Gemma4TextAttention uses scaling=1.0 after q_norm/k_norm.
  // mha_core backend applies 1/sqrt(head_dim) to QK, so pre-scale Q by
  // sqrt(head_dim) to preserve Gemma4 semantics.

  // TODO : fix AVX kernel to not make it divide by 1/sqrt(head_dim) on gemma4
  layers.push_back(createLayer(
    "scalar_multiply",
    {withKey("name", Q_scaled), withKey("input_layers", Q_norm),
     withKey("packed", "false"),
     withKey("multiplier",
             std::to_string(std::sqrt(static_cast<float>(curr_head_dim))))}));

  unsigned int window_size = is_sliding ? SLIDING_WINDOW : UINT_MAX;
  unsigned int rope_theta =
    is_sliding ? SLIDING_ATTENTION_ROPE_THETA : FULL_ATTENTION_ROPE_THETA;

  const std::string &rope_type =
    is_sliding ? SLIDING_ATTENTION_ROPE_TYPE : FULL_ATTENTION_ROPE_TYPE;
  const float rope_partial_rotary_factor =
    is_sliding ? SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR
               : FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR;

  // Shared attention core receives [Q_norm, shared_K_norm, shared_V_norm]
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", curr_kv_heads),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE + 1)),
    withKey("sliding_window", window_size),
    withKey("use_rope", "true"),
    withKey("rope_theta", std::to_string(rope_theta)),
    withKey("rope_scaling_type", rope_type),
    withKey("rope_partial_rotary_factor",
            std::to_string(rope_partial_rotary_factor)),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("attn_logit_softcapping", std::to_string(ATTN_LOGIT_SOFTCAPPING)),
    withKey("is_causal", IS_CAUSAL ? "true" : "false"),
    withKey("input_layers", {Q_scaled, shared_K_norm, shared_V_norm})};
  appendSkipPrefillIfNeeded(a_params, is_kv_shared_layer);
  layers.push_back(createLayer("mha_core", a_params));

  // O layer [B, S, Nq*Dh] -> [B, S, H]
  std::vector<std::string> o_params = {withKey("name", O),
                                       withKey("unit", DIM),
                                       withKey("disable_bias", "true"),
                                       withKey("input_layers", A),
                                       withKey("weight_initializer", "ones"),
                                       withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(o_params, is_kv_shared_layer);
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

std::vector<LayerHandle> Gemma4Transformer::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  std::string query_name, std::string key_name, std::string value_name) {
  std::vector<LayerHandle> layers;

  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto K_norm = "layer" + std::to_string(layer_id) + "_k_norm";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto V_norm = "layer" + std::to_string(layer_id) + "_v_norm";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";
  auto Q_scaled = "layer" + std::to_string(layer_id) + "_q_scaled";

  bool is_sliding = true;
  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  if (!layer_types.empty() && layer_id < static_cast<int>(layer_types.size())) {
    is_sliding = layer_types[layer_id] == "sliding_attention";
  }

  int curr_head_dim = is_sliding ? HEAD_DIM : GLOBAL_HEAD_DIM;
  int curr_kv_heads = (is_sliding || !ATTENTION_K_EQ_V)
                        ? NUM_KEY_VALUE_HEADS
                        : NUM_GLOBAL_KEY_VALUE_HEADS;

  // Q layer [B, S, H] -> [B, S, Nq*Dh]
  std::vector<std::string> q_params = {withKey("name", Q),
                                       withKey("unit", curr_head_dim * n_heads),
                                       withKey("disable_bias", "true"),
                                       withKey("input_layers", query_name),
                                       withKey("weight_initializer", "ones"),
                                       withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(q_params, is_kv_shared_layer);
  layers.push_back(createLayer("fully_connected", q_params));

  // K layer [B, S, H] -> [B, S, Nk*Dh]
  std::vector<std::string> k_params = {
    withKey("name", K),
    withKey("unit", curr_head_dim * curr_kv_heads),
    withKey("disable_bias", "true"),
    withKey("input_layers", key_name),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(k_params, is_kv_shared_layer);
  layers.push_back(createLayer("fully_connected", k_params));

  // V layer [B, S, H] -> [B, S, Nk*Dh]
  std::vector<std::string> v_params = {
    withKey("name", V),
    withKey("unit", curr_head_dim * curr_kv_heads),
    withKey("disable_bias", "true"),
    withKey("input_layers", value_name),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(v_params, is_kv_shared_layer);
  layers.push_back(createLayer("fully_connected", v_params));

  // q_norm on per-head projection [B, S, Nq*Dh]
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm), withKey("input_layers", Q),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim))};
  appendSkipPrefillIfNeeded(q_norm_params, is_kv_shared_layer);
  layers.push_back(createLayer("reshaped_rms_norm", q_norm_params));

  // Gemma4TextAttention uses scaling=1.0 after q_norm/k_norm.
  // mha_core backend applies 1/sqrt(head_dim) to QK, so pre-scale Q by
  // sqrt(head_dim) to preserve Gemma4 semantics.

  // TODO : fix AVX kernel to not make it divide by 1/sqrt(head_dim) on gemma4
  layers.push_back(createLayer(
    "scalar_multiply",
    {withKey("name", Q_scaled), withKey("input_layers", Q_norm),
     withKey("packed", "false"),
     withKey("multiplier",
             std::to_string(std::sqrt(static_cast<float>(curr_head_dim))))}));

  // k_norm on per-head projection [B, S, Nk*Dh]
  std::vector<std::string> k_norm_params = {
    withKey("name", K_norm), withKey("input_layers", K),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim))};
  appendSkipPrefillIfNeeded(k_norm_params, is_kv_shared_layer);
  layers.push_back(createLayer("reshaped_rms_norm", k_norm_params));

  // v_norm on per-head projection [B, S, Nk*Dh] (no learned scale)
  std::vector<std::string> v_norm_params = {
    withKey("name", V_norm), withKey("input_layers", V),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim))};
  v_norm_params.push_back(withKey("use_gamma", "false"));
  appendSkipPrefillIfNeeded(v_norm_params, is_kv_shared_layer);
  layers.push_back(createLayer("reshaped_rms_norm", v_norm_params));

  unsigned int window_size = is_sliding ? SLIDING_WINDOW : UINT_MAX;
  unsigned int rope_theta =
    is_sliding ? SLIDING_ATTENTION_ROPE_THETA : FULL_ATTENTION_ROPE_THETA;
  const std::string &rope_type =
    is_sliding ? SLIDING_ATTENTION_ROPE_TYPE : FULL_ATTENTION_ROPE_TYPE;
  const float rope_partial_rotary_factor =
    is_sliding ? SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR
               : FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR;

  // Attention core receives [Q_norm, K_norm, V_norm]
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", curr_kv_heads),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE + 1)),
    withKey("sliding_window", window_size),
    withKey("use_rope", "true"),
    withKey("rope_theta", std::to_string(rope_theta)),
    withKey("rope_scaling_type", rope_type),
    withKey("rope_partial_rotary_factor",
            std::to_string(rope_partial_rotary_factor)),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("attn_logit_softcapping", std::to_string(ATTN_LOGIT_SOFTCAPPING)),
    withKey("is_causal", IS_CAUSAL ? "true" : "false"),
    withKey("input_layers", {Q_scaled, K_norm, V_norm})};
  appendSkipPrefillIfNeeded(a_params, is_kv_shared_layer);
  layers.push_back(createLayer("mha_core", a_params));

  // O layer [B, S, Nq*Dh] -> [B, S, H]
  std::vector<std::string> o_params = {withKey("name", O),
                                       withKey("unit", DIM),
                                       withKey("disable_bias", "true"),
                                       withKey("input_layers", A),
                                       withKey("weight_initializer", "ones"),
                                       withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(o_params, is_kv_shared_layer);
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

std::vector<LayerHandle> Gemma4Transformer::createMlp(const int layer_id,
                                                      int dim, int hidden_dim,
                                                      std::string input_name) {
  std::vector<LayerHandle> layers;

  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  const int curr_hidden_dim =
    hidden_dim * ((USE_DOUBLE_WIDE_MLP && is_kv_shared_layer) ? 2 : 1);

  std::vector<std::string> ffn_gate_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
    withKey("unit", curr_hidden_dim),
    withKey("disable_bias", "true"),
    withKey("input_layers", input_name),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(ffn_gate_props, is_kv_shared_layer);
  layers.push_back(createLayer("fully_connected", ffn_gate_props));

  std::vector<std::string> ffn_gate_gelu_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate_gelu"),
    withKey("activation", "tanh_gelu"),
    withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_gate")};
  appendSkipPrefillIfNeeded(ffn_gate_gelu_props, is_kv_shared_layer);
  layers.push_back(createLayer("activation", ffn_gate_gelu_props));

  std::vector<std::string> ffn_up_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
    withKey("unit", curr_hidden_dim),
    withKey("disable_bias", "true"),
    withKey("input_layers", input_name),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(ffn_up_props, is_kv_shared_layer);
  layers.push_back(createLayer("fully_connected", ffn_up_props));

  std::vector<std::string> ffn_geglu_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_geglu"),
    withKey("input_layers", "layer" + std::to_string(layer_id) +
                              "_ffn_gate_gelu,layer" +
                              std::to_string(layer_id) + "_ffn_up")};
  appendSkipPrefillIfNeeded(ffn_geglu_props, is_kv_shared_layer);
  layers.push_back(createLayer("multiply", ffn_geglu_props));

  std::vector<std::string> ffn_down_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
    withKey("unit", dim),
    withKey("disable_bias", "true"),
    withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_geglu"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(ffn_down_props, is_kv_shared_layer);
  layers.push_back(createLayer("fully_connected", ffn_down_props));

  return layers;
}

void Gemma4Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::PerLayerSliceLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ScalarMultiplyLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::LogitSoftCappingLayer>);

  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void Gemma4CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Gemma4Transformer::registerCustomLayers();
}

void Gemma4CausalLM::constructModel() {
  Gemma4Transformer::constructModel();

  // create lm_head layer (using fully_connected option)
  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";

  // add lmhead
  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("input_layers", "output_norm"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };
  appendSkipPrefillIfNeeded(lmhead_prop, true);

  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));

  model->addLayer(createLayer(lmhead_type, lmhead_prop));

  if (FINAL_LOGIT_SOFTCAPPING > 0.0f) {
    std::vector<std::string> final_softcap_props = {
      withKey("name", "output_of_causallm_softcapped"),
      withKey("input_layers", "output_of_causallm"),
      withKey("activation_type", "tanh"), withKey("apply_rows", "1"),
      withKey("softcap_value", std::to_string(FINAL_LOGIT_SOFTCAPPING))};
    appendSkipPrefillIfNeeded(final_softcap_props, true);
    model->addLayer(createLayer("logit_softcapping", final_softcap_props));
  }
}

} // namespace causallm

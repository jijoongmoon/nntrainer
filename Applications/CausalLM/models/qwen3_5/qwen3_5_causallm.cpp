/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   qwen3_5_causallm.cpp
 * @date   07 April 2026
 * @brief  Qwen3.5 hybrid CausalLM implementation
 * @see    https://github.com/nnstreamer/
 * @author NNTrainer Authors
 * @bug    No known bugs except for NYI items
 */
#include <llm_util.hpp>
#include <model.h>
#include <qwen3_5_causallm.h>

#include <app_context.h>
#include <attention_gate.h>
#include <engine.h>
#include <gated_delta_net.h>
#include <reshaped_rms_norm.h>
#include <rms_norm_gated.h>

namespace causallm {

void Qwen3_5Transformer::setupParameters(json &cfg, json &generation_cfg,
                                          json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);

  FULL_ATTENTION_INTERVAL =
    cfg.contains("full_attention_interval")
      ? cfg["full_attention_interval"].get<unsigned int>()
      : 4;

  LINEAR_CONV_KERNEL_DIM = cfg.contains("linear_conv_kernel_dim")
                             ? cfg["linear_conv_kernel_dim"].get<int>()
                             : 4;

  LINEAR_KEY_HEAD_DIM = cfg.contains("linear_key_head_dim")
                          ? cfg["linear_key_head_dim"].get<int>()
                          : 128;

  LINEAR_VALUE_HEAD_DIM = cfg.contains("linear_value_head_dim")
                            ? cfg["linear_value_head_dim"].get<int>()
                            : 128;

  LINEAR_NUM_KEY_HEADS = cfg.contains("linear_num_key_heads")
                           ? cfg["linear_num_key_heads"].get<int>()
                           : 16;

  LINEAR_NUM_VALUE_HEADS = cfg.contains("linear_num_value_heads")
                             ? cfg["linear_num_value_heads"].get<int>()
                             : 16;

  PARTIAL_ROTARY_FACTOR = cfg.contains("partial_rotary_factor")
                            ? cfg["partial_rotary_factor"].get<float>()
                            : 0.25f;
}

std::vector<LayerHandle>
Qwen3_5Transformer::createTransformerDecoderBlock(const int layer_id,
                                                   std::string input_name) {
  std::vector<LayerHandle> layers;

  // Pre-attention RMS norm (input_layernorm / attn_norm)
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  std::string norm_name =
    "layer" + std::to_string(layer_id) + "_attention_norm";

  // Route to full attention or linear attention (GatedDeltaNet)
  if (isFullAttentionLayer(layer_id)) {
    auto att_layers = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS,
                                      HEAD_DIM, norm_name, norm_name, norm_name);
    layers.insert(layers.end(), att_layers.begin(), att_layers.end());
  } else {
    auto gdn_layers = createGatedDeltaNet(layer_id, norm_name);
    layers.insert(layers.end(), gdn_layers.begin(), gdn_layers.end());
  }

  // Residual add after attention/linear_attention
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                               "_attention_out")}));

  // Post-attention RMS norm (post_attention_layernorm)
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // MLP (SwiGLU, same as base Transformer)
  auto ffn_layers = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                               "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layers.begin(), ffn_layers.end());

  // Residual add after MLP
  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_decoder_add,layer" + std::to_string(layer_id) +
                               "_ffn_down")}));

  return layers;
}

std::vector<LayerHandle> Qwen3_5Transformer::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  std::string query_name, std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;
  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_gate = "layer" + std::to_string(layer_id) + "_wq_gate";
  auto Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto K_norm = "layer" + std::to_string(layer_id) + "_k_norm";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto AG = "layer" + std::to_string(layer_id) + "_attn_gate";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // Q layer: query projection (first half of original 2x Q)
  // q_proj: hidden_size -> num_heads * head_dim
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", q_params));

  // Q gate layer: sigmoid gate (second half of original 2x Q)
  // gate_proj: hidden_size -> num_heads * head_dim
  std::vector<std::string> qg_params = {
    withKey("name", Q_gate), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", qg_params));

  // Q-reshaped-norm layer
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm), withKey("input_layers", Q),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))};
  layers.push_back(createLayer("reshaped_rms_norm", q_norm_params));

  // K layer
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", key_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", k_params));

  // K-reshaped-norm layer
  std::vector<std::string> k_norm_params = {
    withKey("name", K_norm), withKey("input_layers", K),
    withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(head_dim))};
  layers.push_back(createLayer("reshaped_rms_norm", k_norm_params));

  // V layer
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", v_params));

  // Attention core layer
  // @todo: partial_rotary_factor support needs to be added to mha_core
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", UINT_MAX),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("input_layers", {Q_norm, K_norm, V})};
  layers.push_back(createLayer("mha_core", a_params));

  // Attention gate: output = attn_output * sigmoid(gate)
  std::vector<std::string> ag_params = {
    withKey("name", AG),
    withKey("input_layers", A + "," + Q_gate)};
  layers.push_back(createLayer("attention_gate", ag_params));

  // O layer (output projection)
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", DIM), withKey("disable_bias", "true"),
    withKey("input_layers", AG), withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

std::vector<LayerHandle>
Qwen3_5Transformer::createGatedDeltaNet(const int layer_id,
                                         std::string input_name) {
  std::vector<LayerHandle> layers;

  int key_dim = LINEAR_KEY_HEAD_DIM * LINEAR_NUM_KEY_HEADS;
  int value_dim = LINEAR_VALUE_HEAD_DIM * LINEAR_NUM_VALUE_HEADS;
  int conv_dim = key_dim * 2 + value_dim;

  auto GDN = "layer" + std::to_string(layer_id) + "_attention_out";

  std::vector<std::string> gdn_params = {
    withKey("name", GDN),
    withKey("input_layers", input_name),
    withKey("num_heads", LINEAR_NUM_VALUE_HEADS),
    withKey("num_key_heads", LINEAR_NUM_KEY_HEADS),
    withKey("key_head_dim", LINEAR_KEY_HEAD_DIM),
    withKey("value_head_dim", LINEAR_VALUE_HEAD_DIM),
    withKey("conv_kernel_size", LINEAR_CONV_KERNEL_DIM),
    withKey("epsilon", std::to_string(NORM_EPS))};
  layers.push_back(createLayer("gated_delta_net", gdn_params));

  return layers;
}

void Qwen3_5Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::GatedDeltaNetLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::RMSNormGatedLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::AttentionGateLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void Qwen3_5CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Qwen3_5Transformer::registerCustomLayers();
}

} // namespace causallm

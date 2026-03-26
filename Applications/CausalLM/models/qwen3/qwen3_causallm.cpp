/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 *
 * @file	qwen3_causallm.cpp
 * @date	23 July 2025
 * @brief	This defines a qwen3 causal language model.
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <llm_util.hpp>
#include <model.h>
#include <qwen3_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <reshaped_rms_norm.h>

namespace causallm {

Tensor Qwen3Transformer::createAttention(const int layer_id, int seq_len,
                                          int n_heads, int head_dim,
                                          Tensor query, Tensor key,
                                          Tensor value) {

  using ml::train::createLayer;

  auto Q_name = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_norm_name = "layer" + std::to_string(layer_id) + "_q_norm";
  auto K_name = "layer" + std::to_string(layer_id) + "_wk";
  auto K_norm_name = "layer" + std::to_string(layer_id) + "_k_norm";
  auto V_name = "layer" + std::to_string(layer_id) + "_wv";
  auto A_name = "layer" + std::to_string(layer_id) + "_attention";
  auto O_name = "layer" + std::to_string(layer_id) + "_attention_out";

  // V projection
  LayerHandle v_proj = createLayer(
    "fully_connected",
    {withKey("name", V_name), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")});
  Tensor v = v_proj(value);

  // K projection
  LayerHandle k_proj = createLayer(
    "fully_connected",
    {withKey("name", K_name), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")});
  Tensor k = k_proj(key);

  // K reshaped norm
  LayerHandle k_norm = createLayer(
    "reshaped_rms_norm",
    {withKey("name", K_norm_name), withKey("packed", "false"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))});
  Tensor k_normed = k_norm(k);

  // Q projection
  LayerHandle q_proj = createLayer(
    "fully_connected",
    {withKey("name", Q_name), withKey("unit", head_dim * n_heads),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")});
  Tensor q = q_proj(query);

  // Q reshaped norm
  LayerHandle q_norm = createLayer(
    "reshaped_rms_norm",
    {withKey("name", Q_norm_name), withKey("packed", "false"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))});
  Tensor q_normed = q_norm(q);

  // Attention core layer
  LayerHandle attn = createLayer(
    "mha_core",
    {withKey("name", A_name), withKey("num_heads", n_heads),
     withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("sliding_window", SLIDING_WINDOW),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE))});
  Tensor a = attn({q_normed, k_normed, v});

  // O projection
  LayerHandle o_proj = createLayer(
    "fully_connected",
    {withKey("name", O_name), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")});
  Tensor o = o_proj(a);

  return o;
}

void Qwen3Transformer::registerCustomLayers() {
  CausalLM::registerCustomLayers();

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

void Qwen3CausalLM::registerCustomLayers() {
  Qwen3Transformer::registerCustomLayers();
}

} // namespace causallm

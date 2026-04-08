// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   qwen2_causallm.h
 * @date   6 January 2026
 * @brief  This defines a qwen2 causal language model.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <llm_util.hpp>
#include <model.h>
#include <qwen2_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <reshaped_rms_norm.h>

namespace causallm {

Tensor Qwen2Transformer::createAttention(const int layer_id, int seq_len,
                                          int n_heads, int head_dim,
                                          Tensor query, Tensor key,
                                          Tensor value, Tensor cache_key,
                                          Tensor cache_value) {

  using ml::train::createLayer;

  auto Q_name = "layer" + std::to_string(layer_id) + "_wq";
  auto K_name = "layer" + std::to_string(layer_id) + "_wk";
  auto V_name = "layer" + std::to_string(layer_id) + "_wv";
  auto A_name = "layer" + std::to_string(layer_id) + "_attention";
  auto O_name = "layer" + std::to_string(layer_id) + "_attention_out";

  // V projection
  LayerHandle v_proj = createLayer(
    "fully_connected",
    {withKey("name", V_name), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")});
  Tensor v = v_proj(value);

  // K projection
  LayerHandle k_proj = createLayer(
    "fully_connected",
    {withKey("name", K_name), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")});
  Tensor k = k_proj(key);

  // Q projection
  LayerHandle q_proj = createLayer(
    "fully_connected",
    {withKey("name", Q_name), withKey("unit", head_dim * n_heads),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")});
  Tensor q = q_proj(query);

  // Attention core layer
  LayerHandle attn = createLayer(
    "mha_core",
    {withKey("name", A_name), withKey("num_heads", n_heads),
     withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("sliding_window", SLIDING_WINDOW),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false")});
  Tensor a = attn({q, k, v, cache_key, cache_value});

  // O projection
  LayerHandle o_proj = createLayer(
    "fully_connected",
    {withKey("name", O_name), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")});
  Tensor o = o_proj(a);

  return o;
}

} // namespace causallm

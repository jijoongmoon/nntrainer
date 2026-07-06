// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gauss4_causallm.h
 * @date   16 Jun 2026
 * @brief  Gauss4 CausalLM model.
 * @author Joonseok Oh <jrock.oh@samsung.com>
 *
 */

#include "gauss4_causallm.h"

#include <app_context.h>
#include <engine.h>
#include <factory.h>
#include <llm_util.hpp>
#include <model.h>
#include <reshaped_rms_norm.h>
#include <rms_reverse_norm.h>

#include <fp16.h>
#include <iostream>
#include <string>
#include <tensor.h>

namespace causallm {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void Gauss4Transformer::appendSkipPrefillIfNeeded(
  std::vector<std::string> &props, bool skip) const {
  if (skip && ENABLE_SKIP_PREFILL_OPT) {
    props.emplace_back(withKey("skip_prefill", "true"));
  }
}

// ---------------------------------------------------------------------------
// setupParameters
// ---------------------------------------------------------------------------

void Gauss4Transformer::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  // Let base class read common params (DIM, NUM_HEADS, HEAD_DIM, etc.)
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);

  // EMBEDDING_SCALE = 1.0 (PyTorch: h = embed_tokens(ids) -- no scaling)
  EMBEDDING_SCALE = 1.0f;

  // The new config has NO top-level rope_theta and NO sliding_window_pattern.
  // Base class: SLIDING_WINDOW_PATTERN defaults to 1 when key absent.
  // Gauss4 rule: pattern=5 (every 5th layer is full attention).
  // Override if not explicitly set in config.
  if (!cfg.contains("sliding_window_pattern")) {
    SLIDING_WINDOW_PATTERN = 5;
  }

  // Gauss4 stores per-attention-type rope_theta under
  // rope_parameters.{sliding_attention,full_attention}.rope_theta
  // (sliding = 500000, full = 8000000). Apply the correct theta PER LAYER TYPE:
  // sliding layers use the sliding value, full-attention layers (every 5th) use
  // the full value. (The base ROPE_THETA reads the legacy global
  // rope_parameters.rope_theta = 10000 and is not used by the attention
  // builders, which read the per-type members below.)
  if (cfg.contains("rope_parameters")) {
    const auto &rp = cfg["rope_parameters"];
    if (rp.contains("sliding_attention") &&
        rp["sliding_attention"].contains("rope_theta")) {
      SLIDING_ATTENTION_ROPE_THETA =
        rp["sliding_attention"]["rope_theta"].get<double>();
    }
    if (rp.contains("full_attention") &&
        rp["full_attention"].contains("rope_theta")) {
      FULL_ATTENTION_ROPE_THETA =
        rp["full_attention"]["rope_theta"].get<double>();
    }
    // Keep the base single ROPE_THETA in sync (defensive) for any legacy path.
    ROPE_THETA = SLIDING_ATTENTION_ROPE_THETA;
  }

  USE_KV_SHARING =
    cfg.contains("kv_sharing") ? cfg["kv_sharing"].get<bool>() : true;

  if (USE_KV_SHARING) {
    // Read from config.json (nntr_config.json has STALE values)
    // sliding_attention_kv_layer=19 -> capture at INPUT of layer 19 = output of
    // 18 full_attention_kv_layer=20    -> capture at INPUT of layer 20 = output
    // of 19
    NUM_SEQUENTIAL_LAYERS = cfg.contains("num_sequential_layers")
                              ? cfg["num_sequential_layers"].get<unsigned int>()
                              : 20;

    unsigned int sliding_cfg =
      cfg.contains("sliding_attention_kv_layer")
        ? cfg["sliding_attention_kv_layer"].get<unsigned int>()
        : 19;
    SLIDING_ATTENTION_KV_LAYER = sliding_cfg - 1; // 19-1 = 18

    unsigned int full_cfg =
      cfg.contains("full_attention_kv_layer")
        ? cfg["full_attention_kv_layer"].get<unsigned int>()
        : 20;
    FULL_ATTENTION_KV_LAYER = full_cfg - 1; // 20-1 = 19
  } else {
    NUM_SEQUENTIAL_LAYERS = static_cast<unsigned int>(NUM_LAYERS);
    SLIDING_ATTENTION_KV_LAYER = 0;
    FULL_ATTENTION_KV_LAYER = 0;
  }

  LATENT_SIZE_PER_GATE = cfg.contains("latent_size_per_gate")
                           ? cfg["latent_size_per_gate"].get<unsigned int>()
                           : 256;

  MLP_LRA_RANK = cfg.contains("mlp_lra_rank")
                   ? cfg["mlp_lra_rank"].get<unsigned int>()
                   : 512;

  HIDDEN_SIZE_PER_LAYER_INPUT =
    cfg.contains("hidden_size_per_layer_input")
      ? cfg["hidden_size_per_layer_input"].get<unsigned int>()
      : 192;

  PLE_ACT =
    cfg.contains("ple_act") ? cfg["ple_act"].get<std::string>() : "sigmoid";

  PLE_MIX_METHOD =
    cfg.contains("ple_mix_method") ? cfg["ple_mix_method"].get<int>() : 1;

  PLE_PRE_MLP =
    cfg.contains("ple_pre_mlp") ? cfg["ple_pre_mlp"].get<bool>() : true;

  ENABLE_SKIP_PREFILL_OPT =
    nntr_cfg.contains("skip_prefill") && nntr_cfg["skip_prefill"].get<bool>();
}

// ---------------------------------------------------------------------------
// KV cache placeholder helpers
// ---------------------------------------------------------------------------

std::pair<Tensor, Tensor>
Gauss4Transformer::createGauss4KVCachePlaceholders(const int layer_id,
                                                   unsigned int kv_width) {
  const unsigned int max_timestep = static_cast<unsigned int>(MAX_SEQ_LEN);
#ifdef ENABLE_FP16
  // On Android (ENABLE_FP16), the KV cache is FP16 — mirror the base
  // CausalLM convention (transformer.cpp createKVCachePlaceholders /
  // causal_lm.cpp allocateAndBindKVCache). The mha_core FP16 path writes FP16
  // K/V values into the cache via copyData; a UINT16 cache would hit
  // UIntTensor::copyData which has no FP16 source case and throws
  // "Unsupported data type".
  ml::train::TensorDim cache_dim(
    {BATCH_SIZE, 1, max_timestep, kv_width},
    {ml::train::TensorDim::Format::NCHW, ml::train::TensorDim::DataType::FP16});
  Tensor cache_k(cache_dim, "cache_k_l" + std::to_string(layer_id));
  Tensor cache_v(cache_dim, "cache_v_l" + std::to_string(layer_id));
  return {cache_k, cache_v};
#else
  const std::string cache_shape = std::to_string(BATCH_SIZE) +
                                  ":1:" + std::to_string(max_timestep) + ":" +
                                  std::to_string(kv_width);

  LayerHandle cache_k_input(createLayer(
    "input",
    {withKey("name", "cache_k_l" + std::to_string(layer_id)),
     withKey("input_shape", cache_shape), withKey("input_dtype", "UINT16")}));
  LayerHandle cache_v_input(createLayer(
    "input",
    {withKey("name", "cache_v_l" + std::to_string(layer_id)),
     withKey("input_shape", cache_shape), withKey("input_dtype", "UINT16")}));

  return {cache_k_input(Tensor()), cache_v_input(Tensor())};
#endif
}

// ---------------------------------------------------------------------------
// constructModel
// ---------------------------------------------------------------------------

std::pair<Tensor, Tensor> Gauss4Transformer::constructModel() {
  // Input token ids
  Tensor x({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");

  // Main embedding (no sqrt scaling -- PyTorch: h = embed_tokens(ids))
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";
  LayerHandle embedding(createLayer(
    embedding_type,
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));
  Tensor h = embedding(x);

  // ── Normal decoder layers 0 .. NUM_SEQUENTIAL_LAYERS-1 ──────────────────
  kv_sharing_sliding_tensor = Tensor();
  kv_sharing_full_tensor = Tensor();

  for (int i = 0; i < static_cast<int>(NUM_SEQUENTIAL_LAYERS); ++i) {
    h = createTransformerDecoderBlock(i, h);

    // Capture hidden state AFTER layer i for KV-sharing norms + rotation:
    // PyTorch (modeling_gauss4.py Gauss4Model.forward()):
    //   at i == sliding_attention_kv_layer (19):
    //     shared_sliding = rotation_sliding(norm_kv_sharing_sliding(h))
    //   at i == full_attention_kv_layer (20):
    //     shared_full = rotation_full(norm_kv_sharing_full(h))
    //
    // C++ captures the ROTATION output as the shared tensor.
    // SLIDING_ATTENTION_KV_LAYER = 18 (config 19-1) -> after layer 18 = layer
    // 19 input FULL_ATTENTION_KV_LAYER    = 19 (config 20-1) -> after layer 19
    // = layer 20 input
    if (USE_KV_SHARING && i == static_cast<int>(SLIDING_ATTENTION_KV_LAYER)) {
      // norm_kv_sharing_sliding
      LayerHandle norm_sliding(
        createLayer("rms_norm", {withKey("name", "norm_kv_sharing_sliding"),
                                 withKey("epsilon", std::to_string(NORM_EPS)),
                                 withKey("packed", "false"),
                                 withKey("engine", causallm_engine())}));
      Tensor normed_sliding = norm_sliding(h);

      // rotation_sliding: FC(2688->2688, no bias)
      LayerHandle rotation_sliding(createLayer(
        "fully_connected",
        {withKey("name", "rotation_sliding"), withKey("unit", DIM),
         withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
         withKey("weight_dtype", FC_LAYER_DTYPE),
         withKey("engine", causallm_engine())}));
      kv_sharing_sliding_tensor = rotation_sliding(normed_sliding);
    }
    if (USE_KV_SHARING && i == static_cast<int>(FULL_ATTENTION_KV_LAYER)) {
      // norm_kv_sharing_full
      LayerHandle norm_full(
        createLayer("rms_norm", {withKey("name", "norm_kv_sharing_full"),
                                 withKey("epsilon", std::to_string(NORM_EPS)),
                                 withKey("packed", "false"),
                                 withKey("engine", causallm_engine())}));
      Tensor normed_full = norm_full(h);

      // rotation_full: FC(2688->2688, no bias)
      LayerHandle rotation_full(createLayer(
        "fully_connected",
        {withKey("name", "rotation_full"), withKey("unit", DIM),
         withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
         withKey("weight_dtype", FC_LAYER_DTYPE),
         withKey("engine", causallm_engine())}));
      kv_sharing_full_tensor = rotation_full(normed_full);
    }
  }

  // ── KV-shared layers NUM_SEQUENTIAL_LAYERS .. NUM_LAYERS-1 ──────────────
  if (USE_KV_SHARING) {
    for (int i = static_cast<int>(NUM_SEQUENTIAL_LAYERS); i < NUM_LAYERS; ++i) {
      h = createTransformerDecoderBlock(i, h);
    }
  }

  // Output norm (skip_prefill=true)
  std::vector<std::string> output_norm_props = {
    withKey("name", "output_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  output_norm_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(output_norm_props, true);
  LayerHandle out_norm(createLayer("rms_norm", output_norm_props));
  h = out_norm(h);

  return {x, h};
}

// ---------------------------------------------------------------------------
// createTransformerDecoderBlock
// ---------------------------------------------------------------------------

Tensor Gauss4Transformer::createTransformerDecoderBlock(const int layer_id,
                                                        Tensor input) {
  const bool is_shared_layer =
    USE_KV_SHARING && layer_id >= static_cast<int>(NUM_SEQUENTIAL_LAYERS);
  const bool is_sliding =
    ((layer_id + 1) % static_cast<int>(SLIDING_WINDOW_PATTERN)) != 0;

  // ── Input (pre-attention) RMS norm ──────────────────────────────────────
  std::vector<std::string> attn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  attn_norm_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(attn_norm_props, is_shared_layer);
  LayerHandle attn_norm(createLayer("rms_norm", attn_norm_props));
  Tensor normed = attn_norm(input);

  // ── Attention ────────────────────────────────────────────────────────────
  Tensor att_out;
  if (is_shared_layer) {
    Tensor shared_kv_raw =
      is_sliding ? kv_sharing_sliding_tensor : kv_sharing_full_tensor;

    // PyTorch Gauss4DecoderLayer.forward() normalizes shared_states through the
    // SAME input_layernorm before passing it to k_proj / v_proj:
    //   shared_states = self.input_layernorm(shared_states)
    // Mirror that by creating a second rms_norm that shares weights with the
    // attention_norm already constructed above ("layer{id}_attention_norm").
    // NOTE: kv_norm intentionally does NOT have skip_prefill=true even though
    // this is a shared layer.  During prefill, kv_norm must run to produce the
    // normalized shared_kv tensor so that wk/wv can compute the K/V projections
    // and populate the KV cache for positions 0..(prefill_len-1).  Without
    // this, shared-layer KV caches stay zeroed for all prefill positions,
    // causing a systematic attention error in the first decode steps.
    const std::string kv_norm_name =
      "layer" + std::to_string(layer_id) + "_attention_norm_kv";
    const std::string shared_from_name =
      "layer" + std::to_string(layer_id) + "_attention_norm";
    std::vector<std::string> kv_norm_props = {
      withKey("name", kv_norm_name),
      withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
      withKey("shared_from", shared_from_name)};
    kv_norm_props.push_back(withKey("engine", causallm_engine()));
    // skip_prefill deliberately omitted here — see note above.
    LayerHandle kv_norm(createLayer("rms_norm", kv_norm_props));
    Tensor shared_kv = kv_norm(shared_kv_raw);

    att_out = createSharedAttention(layer_id, normed, shared_kv);
  } else {
    att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                              normed, normed, normed);
  }

  // ── Residual 1: decoder_add = input + attn_out ────────────────────────
  std::vector<std::string> dec_add_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add")};
  dec_add_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(dec_add_props, is_shared_layer);
  LayerHandle dec_add_layer(createLayer("addition", dec_add_props));
  Tensor decoder_add = dec_add_layer({input, att_out});

  // ── Post-attention (pre-FFN) RMS norm ──────────────────────────────────
  std::vector<std::string> ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  ffn_norm_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(ffn_norm_props, is_shared_layer);
  LayerHandle ffn_norm(createLayer("rms_norm", ffn_norm_props));
  Tensor ffn_in = ffn_norm(decoder_add);

  // ── PLE input capture (ple_pre_mlp=true) ─────────────────────────────
  // PyTorch: ple_states = hidden_states (= ffn_norm output, BEFORE MLP)
  Tensor ple_input = PLE_PRE_MLP ? ffn_in : Tensor();

  // ── FFN (LRA MLP) ────────────────────────────────────────────────────────
  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_in);

  // ── Per-Layer Embedding (PLE) ─────────────────────────────────────────
  // For ple_pre_mlp=true: PLE gate input = ffn_norm output (pre-MLP)
  // For ple_pre_mlp=false (old): PLE gate input = ffn_output (post-MLP)
  Tensor input0({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");
  Tensor ple_gate_input = PLE_PRE_MLP ? ple_input : ffn_out;
  Tensor ple_out =
    createPerLayerEmbedding(layer_id, ple_gate_input, input0, is_shared_layer);

  // ── 3-way residual: decoder_add + ffn_output + ple_out ───────────────
  std::vector<std::string> dec_out_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output")};
  dec_out_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(dec_out_props, is_shared_layer);
  LayerHandle dec_out_layer(createLayer("addition", dec_out_props));
  Tensor decoder_output = dec_out_layer({decoder_add, ffn_out, ple_out});

  return decoder_output;
}

// ---------------------------------------------------------------------------
// createAttention (layers 0-19, normal)
// ---------------------------------------------------------------------------

Tensor Gauss4Transformer::createAttention(const int layer_id, int /*seq_len*/,
                                          int n_heads, int /*head_dim*/,
                                          Tensor query, Tensor /*key*/,
                                          Tensor /*value*/) {
  const unsigned int kv_unit = static_cast<unsigned int>(HEAD_DIM) *
                               static_cast<unsigned int>(NUM_KEY_VALUE_HEADS);
  const unsigned int q_unit =
    static_cast<unsigned int>(HEAD_DIM) * static_cast<unsigned int>(n_heads);
  const unsigned int kv_width = kv_unit;

  const bool is_sliding =
    ((layer_id + 1) % static_cast<int>(SLIDING_WINDOW_PATTERN)) != 0;
  const float rope_theta =
    is_sliding ? static_cast<float>(SLIDING_ATTENTION_ROPE_THETA)
               : static_cast<float>(FULL_ATTENTION_ROPE_THETA);
  const unsigned int window_size = is_sliding ? SLIDING_WINDOW : UINT_MAX;

  const std::string lname = "layer" + std::to_string(layer_id);

  // ── Q projection ────────────────────────────────────────────────────────
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", lname + "_wq"), withKey("unit", q_unit),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", causallm_engine())}));
  Tensor q = wq(query);

  // ── K projection ────────────────────────────────────────────────────────
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", lname + "_wk"), withKey("unit", kv_unit),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", causallm_engine())}));
  Tensor k = wk(query); // K/V also from attn_norm output

  // ── V projection ────────────────────────────────────────────────────────
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", lname + "_wv"), withKey("unit", kv_unit),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", causallm_engine())}));
  Tensor v = wv(query);

  // ── MHA core with KV cache placeholders ─────────────────────────────────
  auto [cache_k, cache_v] = createGauss4KVCachePlaceholders(layer_id, kv_width);

  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", lname + "_attention"), withKey("num_heads", n_heads),
     withKey("num_heads_kv", NUM_KEY_VALUE_HEADS),
     withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
     withKey("max_position_embeddings",
             std::to_string(MAX_POSITION_EMBEDDINGS)),
     withKey("sliding_window", window_size), withKey("use_rope", "true"),
     withKey("rope_theta", std::to_string(rope_theta)),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("attn_logit_softcapping", "0.0"),
     withKey("is_causal", IS_CAUSAL ? "true" : "false"),
     withKey("use_gemm_attention", USE_FLASH_ATTENTION ? "true" : "false"),
     withKey("engine", causallm_engine())}));
  Tensor attn_raw = mha({q, k, v, cache_k, cache_v});

  // ── reshaped_rms_norm on attention output (gated_norm) ──────────────────
  // feature_size = head_dim = 128 (kv_channels norm)
  LayerHandle gated_norm(createLayer(
    "reshaped_rms_norm", {withKey("name", lname + "_gated_norm"),
                          withKey("epsilon", std::to_string(NORM_EPS)),
                          withKey("feature_size", std::to_string(HEAD_DIM)),
                          withKey("packed", "false"),
                          withKey("engine", causallm_engine())}));
  Tensor normed_attn = gated_norm(attn_raw);

  // ── Lowrank-element-wise gate ────────────────────────────────────────────
  // Gate INPUT = attention_norm output = query (PyTorch: gate =
  // gate_down(hidden_states))
  const std::string gate_name = lname + "_attention_gate";

  LayerHandle gate_down(
    createLayer("fully_connected", {withKey("name", gate_name + "_down"),
                                    withKey("unit", LATENT_SIZE_PER_GATE),
                                    withKey("disable_bias", "true"),
                                    withKey("weight_initializer", "ones"),
                                    withKey("weight_dtype", FC_LAYER_DTYPE),
                                    withKey("engine", causallm_engine())}));
  Tensor g_down = gate_down(query);

  LayerHandle gate_up(createLayer(
    "fully_connected",
    {withKey("name", gate_name + "_up"), withKey("unit", q_unit),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", causallm_engine())}));
  Tensor g_up = gate_up(g_down);

  // sigmoid activation
  LayerHandle gate_sigmoid(
    createLayer("activation", {withKey("name", gate_name + "_sigmoid"),
                               withKey("activation", "sigmoid")}));
  Tensor gate_act = gate_sigmoid(g_up);

  // multiply: sigmoid(gate) * reshaped_rms_norm(attn_out)
  LayerHandle gate_mult(
    createLayer("multiply", {withKey("name", gate_name + "_mult")}));
  Tensor gated = gate_mult({gate_act, normed_attn});

  // ── O projection ─────────────────────────────────────────────────────────
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", lname + "_attention_out"), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE),
     withKey("engine", causallm_engine())}));
  return wo(gated);
}

// ---------------------------------------------------------------------------
// createSharedAttention (layers 20-34, skip_prefill=true)
// ---------------------------------------------------------------------------

Tensor Gauss4Transformer::createSharedAttention(const int layer_id,
                                                Tensor query,
                                                Tensor shared_kv) {
  const unsigned int kv_unit = static_cast<unsigned int>(HEAD_DIM) *
                               static_cast<unsigned int>(NUM_KEY_VALUE_HEADS);
  const unsigned int q_unit =
    static_cast<unsigned int>(HEAD_DIM) * static_cast<unsigned int>(NUM_HEADS);
  const unsigned int kv_width = kv_unit;
  const bool is_sliding =
    ((layer_id + 1) % static_cast<int>(SLIDING_WINDOW_PATTERN)) != 0;
  const float rope_theta =
    is_sliding ? static_cast<float>(SLIDING_ATTENTION_ROPE_THETA)
               : static_cast<float>(FULL_ATTENTION_ROPE_THETA);
  const unsigned int window_size = is_sliding ? SLIDING_WINDOW : UINT_MAX;

  const std::string lname = "layer" + std::to_string(layer_id);

  // Q (skip_prefill)
  std::vector<std::string> q_props = {
    withKey("name", lname + "_wq"), withKey("unit", q_unit),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  q_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(q_props, true);
  LayerHandle wq(createLayer("fully_connected", q_props));
  Tensor q = wq(query);

  // K from shared_kv — NO skip_prefill so K is computed and written to the KV
  // cache during prefill, populating positions 0..(prefill_len-1).  The
  // mha_core layer still has skip_prefill=true, so it returns early after
  // writing K to the cache, skipping the attention computation (which is not
  // needed during prefill).
  std::vector<std::string> k_props = {
    withKey("name", lname + "_wk"), withKey("unit", kv_unit),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  k_props.push_back(withKey("engine", causallm_engine()));
  // skip_prefill deliberately omitted — K must be written to cache at prefill.
  LayerHandle wk(createLayer("fully_connected", k_props));
  Tensor k = wk(shared_kv);

  // V from shared_kv — NO skip_prefill for the same reason as wk above.
  std::vector<std::string> v_props = {
    withKey("name", lname + "_wv"), withKey("unit", kv_unit),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  v_props.push_back(withKey("engine", causallm_engine()));
  // skip_prefill deliberately omitted — V must be written to cache at prefill.
  LayerHandle wv(createLayer("fully_connected", v_props));
  Tensor v = wv(shared_kv);

  // KV cache placeholders (each layer has its own cache even when KV-shared)
  auto [cache_k, cache_v] = createGauss4KVCachePlaceholders(layer_id, kv_width);

  std::vector<std::string> mha_props = {
    withKey("name", lname + "_attention"),
    withKey("num_heads", NUM_HEADS),
    withKey("num_heads_kv", NUM_KEY_VALUE_HEADS),
    withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
    withKey("max_position_embeddings", std::to_string(MAX_POSITION_EMBEDDINGS)),
    withKey("sliding_window", window_size),
    withKey("use_rope", "true"),
    withKey("rope_theta", std::to_string(rope_theta)),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("attn_logit_softcapping", "0.0"),
    withKey("is_causal", IS_CAUSAL ? "true" : "false")};
  mha_props.push_back(withKey("engine", causallm_engine()));
  mha_props.push_back(
    withKey("use_gemm_attention", USE_FLASH_ATTENTION ? "true" : "false"));
  appendSkipPrefillIfNeeded(mha_props, true);
  LayerHandle mha(createLayer("mha_core", mha_props));
  Tensor attn_raw = mha({q, k, v, cache_k, cache_v});

  // reshaped_rms_norm -- feature_size = head_dim = 128
  std::vector<std::string> gn_props = {
    withKey("name", lname + "_gated_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(HEAD_DIM)),
    withKey("packed", "false")};
  gn_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(gn_props, true);
  LayerHandle gated_norm(createLayer("reshaped_rms_norm", gn_props));
  Tensor normed_attn = gated_norm(attn_raw);

  // Gate: input = attention_norm output = query
  const std::string gate_name = lname + "_attention_gate";

  std::vector<std::string> gd_props = {
    withKey("name", gate_name + "_down"), withKey("unit", LATENT_SIZE_PER_GATE),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  gd_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(gd_props, true);
  LayerHandle gate_down(createLayer("fully_connected", gd_props));
  Tensor g_down = gate_down(query);

  std::vector<std::string> gu_props = {
    withKey("name", gate_name + "_up"), withKey("unit", q_unit),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  gu_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(gu_props, true);
  LayerHandle gate_up(createLayer("fully_connected", gu_props));
  Tensor g_up = gate_up(g_down);

  std::vector<std::string> gs_props = {withKey("name", gate_name + "_sigmoid"),
                                       withKey("activation", "sigmoid")};
  appendSkipPrefillIfNeeded(gs_props, true);
  LayerHandle gate_sigmoid(createLayer("activation", gs_props));
  Tensor gate_act = gate_sigmoid(g_up);

  std::vector<std::string> gm_props = {withKey("name", gate_name + "_mult")};
  appendSkipPrefillIfNeeded(gm_props, true);
  LayerHandle gate_mult(createLayer("multiply", gm_props));
  Tensor gated = gate_mult({gate_act, normed_attn});

  // O projection
  std::vector<std::string> o_props = {
    withKey("name", lname + "_attention_out"), withKey("unit", DIM),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  o_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(o_props, true);
  LayerHandle wo(createLayer("fully_connected", o_props));
  return wo(gated);
}

// ---------------------------------------------------------------------------
// createMlp (LRA MLP)
// ---------------------------------------------------------------------------
// PyTorch Gauss4LRAMLP.forward():
//   up_states = linear_up(x)
//   gate_states = up_states + gate_down(gate_up(x))     # add
//   ffn = act_fn(gate_states) * up_states                # silu(add) * up
//   return linear_fc2(ffn)
//
// nntrainer swiglu(input0, input1) = silu(input0) * input1
//   => swiglu(ffn_add, ffn_linear_up) == silu(add) * linear_up  [matches PT]

Tensor Gauss4Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                                    Tensor input) {
  const bool is_shared =
    USE_KV_SHARING && layer_id >= static_cast<int>(NUM_SEQUENTIAL_LAYERS);
  const std::string lname = "layer" + std::to_string(layer_id);

  // linear_up: DIM -> intermediate_size
  std::vector<std::string> lu_props = {
    withKey("name", lname + "_ffn_linear_up"), withKey("unit", hidden_dim),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  lu_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(lu_props, is_shared);
  LayerHandle ffn_linear_up(createLayer("fully_connected", lu_props));
  Tensor linear_up = ffn_linear_up(input);

  // gate_up: DIM -> MLP_LRA_RANK (512)
  std::vector<std::string> gu_props = {
    withKey("name", lname + "_ffn_gate_up"), withKey("unit", MLP_LRA_RANK),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  gu_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(gu_props, is_shared);
  LayerHandle ffn_gate_up(createLayer("fully_connected", gu_props));
  Tensor g_up = ffn_gate_up(input);

  // gate_down: MLP_LRA_RANK -> intermediate_size
  std::vector<std::string> gd_props = {
    withKey("name", lname + "_ffn_gate_down"), withKey("unit", hidden_dim),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  gd_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(gd_props, is_shared);
  LayerHandle ffn_gate_down(createLayer("fully_connected", gd_props));
  Tensor g_down = ffn_gate_down(g_up);

  // add: linear_up + gate_down
  std::vector<std::string> add_props = {withKey("name", lname + "_ffn_add")};
  add_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(add_props, is_shared);
  LayerHandle ffn_add(createLayer("addition", add_props));
  Tensor ffn_sum = ffn_add({linear_up, g_down});

  // swiglu(add, linear_up) = silu(add) * linear_up  [matches PyTorch]
  std::vector<std::string> swiglu_props = {
    withKey("name", lname + "_ffn_swiglu")};
  swiglu_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(swiglu_props, is_shared);
  LayerHandle ffn_swiglu(createLayer("swiglu", swiglu_props));
  Tensor swiglu_out = ffn_swiglu({ffn_sum, linear_up});

  // linear_fc2: intermediate_size -> DIM
  std::vector<std::string> fc2_props = {
    withKey("name", lname + "_ffn_output"), withKey("unit", dim),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  fc2_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(fc2_props, is_shared);
  LayerHandle ffn_fc2(createLayer("fully_connected", fc2_props));
  return ffn_fc2(swiglu_out);
}

// ---------------------------------------------------------------------------
// createPerLayerEmbedding (PLE mix_method=1, ple_pre_mlp=true)
// ---------------------------------------------------------------------------
// PyTorch PerLayerEmbedding.forward() with ple_mix_method=1, ple_act=sigmoid:
//   hidden_states = per_layer_input_gate(hidden_states)    # FC 2688->192
//   output_ple    = sigmoid(hidden_states) + cur_embedding # ADD (method=1)
//   output_ple    = per_layer_projection(output_ple)        # FC 192->2688
//   output_ple    = post_per_layer_input_norm(output_ple)   # ReverseRMSNorm

Tensor Gauss4Transformer::createPerLayerEmbedding(const int layer_id,
                                                  Tensor ple_input,
                                                  Tensor input0,
                                                  bool skip_prefill) {
  const std::string lname = "layer" + std::to_string(layer_id);

  // per_layer_input_gate: DIM -> HIDDEN_SIZE_PER_LAYER_INPUT
  std::vector<std::string> pg_props = {
    withKey("name", lname + "_PLE_input_gate"),
    withKey("unit", HIDDEN_SIZE_PER_LAYER_INPUT),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  pg_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(pg_props, skip_prefill);
  LayerHandle ple_gate(createLayer("fully_connected", pg_props));
  Tensor gate = ple_gate(ple_input);

  if (PLE_MIX_METHOD == 1) {
    // method=1: sigmoid(gate) THEN add(emb)
    // sigmoid activation FIRST
    std::vector<std::string> act_props = {
      withKey("name", lname + "_PLE_activation"),
      withKey("activation", PLE_ACT)};
    appendSkipPrefillIfNeeded(act_props, skip_prefill);
    LayerHandle ple_act_layer(createLayer("activation", act_props));
    Tensor activated = ple_act_layer(gate);

    // embedding lookup: vocab -> HIDDEN_SIZE_PER_LAYER_INPUT
    LayerHandle ple_emb(createLayer(
      "embedding_layer",
      {withKey("name", lname + "_PLE"),
       withKey("in_dim", std::to_string(NUM_VOCAB)),
       withKey("out_dim", std::to_string(HIDDEN_SIZE_PER_LAYER_INPUT)),
       withKey("weight_dtype", EMBEDDING_DTYPE)}));
    Tensor emb = ple_emb(input0);

    // add: sigmoid(gate) + emb
    std::vector<std::string> add_props = {withKey("name", lname + "_PLE_add")};
    add_props.push_back(withKey("engine", causallm_engine()));
    appendSkipPrefillIfNeeded(add_props, skip_prefill);
    LayerHandle ple_add(createLayer("addition", add_props));
    Tensor mix_out = ple_add({activated, emb});

    // projection: HIDDEN_SIZE_PER_LAYER_INPUT -> DIM
    std::vector<std::string> pp_props = {
      withKey("name", lname + "_PLE_projection"), withKey("unit", DIM),
      withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
      withKey("weight_dtype", FC_LAYER_DTYPE)};
    pp_props.push_back(withKey("engine", causallm_engine()));
    appendSkipPrefillIfNeeded(pp_props, skip_prefill);
    LayerHandle ple_proj(createLayer("fully_connected", pp_props));
    Tensor projected = ple_proj(mix_out);

    // post_norm: ReverseRMSNorm (weight + out_scale)
    std::vector<std::string> pn_props = {
      withKey("name", lname + "_PLE_post_norm"),
      withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
    appendSkipPrefillIfNeeded(pn_props, skip_prefill);
    LayerHandle ple_norm(createLayer("rms_reverse_norm", pn_props));
    return ple_norm(projected);
  } else {
    // method=3 (legacy): sigmoid(gate * emb) -> projection -> post_norm
    // embedding lookup: vocab -> HIDDEN_SIZE_PER_LAYER_INPUT
    LayerHandle ple_emb(createLayer(
      "embedding_layer",
      {withKey("name", lname + "_PLE"),
       withKey("in_dim", std::to_string(NUM_VOCAB)),
       withKey("out_dim", std::to_string(HIDDEN_SIZE_PER_LAYER_INPUT)),
       withKey("weight_dtype", EMBEDDING_DTYPE)}));
    Tensor emb = ple_emb(input0);

    // multiply: gate * emb (element-wise)
    std::vector<std::string> mul_props = {
      withKey("name", lname + "_PLE_multiply")};
    appendSkipPrefillIfNeeded(mul_props, skip_prefill);
    LayerHandle ple_mul(createLayer("multiply", mul_props));
    Tensor mul_out = ple_mul({gate, emb});

    // sigmoid activation
    std::vector<std::string> act_props = {
      withKey("name", lname + "_PLE_activation"),
      withKey("activation", PLE_ACT)};
    appendSkipPrefillIfNeeded(act_props, skip_prefill);
    LayerHandle ple_act_layer(createLayer("activation", act_props));
    Tensor activated = ple_act_layer(mul_out);

    // projection: HIDDEN_SIZE_PER_LAYER_INPUT -> DIM
    std::vector<std::string> pp_props = {
      withKey("name", lname + "_PLE_projection"), withKey("unit", DIM),
      withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
      withKey("weight_dtype", FC_LAYER_DTYPE)};
    pp_props.push_back(withKey("engine", causallm_engine()));
    appendSkipPrefillIfNeeded(pp_props, skip_prefill);
    LayerHandle ple_proj(createLayer("fully_connected", pp_props));
    Tensor projected = ple_proj(activated);

    // post_norm: ReverseRMSNorm (weight + out_scale)
    std::vector<std::string> pn_props = {
      withKey("name", lname + "_PLE_post_norm"),
      withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
    appendSkipPrefillIfNeeded(pn_props, skip_prefill);
    LayerHandle ple_norm(createLayer("rms_reverse_norm", pn_props));
    return ple_norm(projected);
  }
}

// ---------------------------------------------------------------------------
// registerCustomLayers
// ---------------------------------------------------------------------------

void Gauss4Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::RMSReverseNormLayer>);
  } catch (const std::invalid_argument &e) {
    std::cerr << "[Gauss4] registerCustomLayers warning: " << e.what()
              << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Gauss4CausalLM::constructModel
// ---------------------------------------------------------------------------

std::pair<Tensor, Tensor> Gauss4CausalLM::constructModel() {
  auto [x, h] = Gauss4Transformer::constructModel();

  // lm_head selection:
  //   LMHEAD_UNTIE=true  -> "fully_connected": output_of_causallm carries its
  //     OWN weight (a separate transposed [hidden,vocab] copy of the tied
  //     embedding, synthesized by the converter). gauss4 is tied in HF and its
  //     embedding must stay Q4_0 (hidden=2688 / PLE=192 are not 256-divisible,
  //     so Q6_K is impossible), but a Q4_0 lm_head has NO GPU GEMV kernel
  //     (tie_word_embedding.cpp's Q4_0 branch is CPU-only). Untying to a
  //     per-channel QS4CX lm_head lets the output projection run the fast v8c
  //     int4 GPU GEMV, mirroring gemma4. The quantizer builds this same untied
  //     graph with an FP32 source weight and the dtype map quantizes
  //     output_of_causallm to QS4CX on save; inference rebuilds it as QS4CX.
  //   tie_word_embeddings=true (not untied) -> "tie_word_embeddings" (shares
  //     embedding0); tie_word_embeddings=false -> "lm_head" (separate FC).
  const bool lmhead_untied = LMHEAD_UNTIE;
  const std::string lmhead_type =
    lmhead_untied
      ? "fully_connected"
      : (TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head");

  std::vector<std::string> lmhead_props = {
    withKey("name", "output_of_causallm"), withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"), withKey("weight_dtype", LMHEAD_DTYPE)};
  lmhead_props.push_back(withKey("engine", causallm_engine()));
  appendSkipPrefillIfNeeded(lmhead_props, true);

  if (TIE_WORD_EMBEDDINGS && !lmhead_untied)
    lmhead_props.emplace_back(withKey("shared_from", "embedding0"));

  LayerHandle lmhead(createLayer(lmhead_type, lmhead_props));
  Tensor y = lmhead(h);

  return {x, y};
}

// ---------------------------------------------------------------------------
// Gauss4CausalLM::registerCustomLayers
// ---------------------------------------------------------------------------

void Gauss4CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Gauss4Transformer::registerCustomLayers();
}

// ---------------------------------------------------------------------------
// Gauss4CausalLM::allocateAndBindKVCache
// ---------------------------------------------------------------------------

void Gauss4CausalLM::allocateAndBindKVCache() {
  const unsigned int kv_width = static_cast<unsigned int>(HEAD_DIM) *
                                static_cast<unsigned int>(NUM_KEY_VALUE_HEADS);

  if (!kv_cache.isAllocated()) {
#ifdef ENABLE_FP16
    const auto cache_dtype = ml::train::TensorDim::DataType::FP16;
#else
    const auto cache_dtype = ml::train::TensorDim::DataType::UINT16;
#endif
    std::vector<unsigned int> kv_widths(static_cast<size_t>(NUM_LAYERS),
                                        kv_width);

    kv_cache.allocate(static_cast<unsigned int>(NUM_LAYERS), BATCH_SIZE,
                      static_cast<unsigned int>(MAX_SEQ_LEN), kv_widths,
                      cache_dtype);
    kv_cache_bound = false;
  }

  if (kv_cache_bound)
    return;

  for (int i = 0; i < NUM_LAYERS; ++i) {
    auto &kc = kv_cache.getKeyCache(i);
    auto &vc = kv_cache.getValueCache(i);

    auto find_cache_placeholder = [this](const std::string &base_name) {
      for (const auto &suffix : {":0", ":input0", ":out0", ""}) {
        auto *tensor = model->getTensor(base_name + suffix);
        if (tensor != nullptr)
          return tensor;
      }
      return static_cast<nntrainer::Tensor *>(nullptr);
    };

    auto *kp =
      model->getTensor("layer" + std::to_string(i) + "_attention:input3");
    auto *vp =
      model->getTensor("layer" + std::to_string(i) + "_attention:input4");
    if (kp == nullptr)
      kp = find_cache_placeholder("cache_k_l" + std::to_string(i));
    if (vp == nullptr)
      vp = find_cache_placeholder("cache_v_l" + std::to_string(i));

    NNTR_THROW_IF(kp == nullptr || vp == nullptr, std::runtime_error)
      << "[Gauss4] allocateAndBindKVCache: cache_k_l" << i << " / cache_v_l"
      << i << " not found in compiled graph";
    NNTR_THROW_IF(kp->getDim() != kc.getDim() || vp->getDim() != vc.getDim(),
                  std::runtime_error)
      << "[Gauss4] allocateAndBindKVCache: shape mismatch for layer " << i;

    kp->setData(kc.getMemoryData(), kc.getOffset(), false);
    vp->setData(vc.getMemoryData(), vc.getOffset(), false);
  }

  kv_cache_bound = true;
}

} // namespace causallm

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_vision_encoder.cpp
 * @date   13 June 2026
 * @brief  Qwen2.5-Omni Thinker vision encoder (Qwen2.5-VL-style ViT).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <vision_attention.h>
#include <vision_rope.h>

#include <qwen25_omni_vision_encoder.h>

namespace causallm {

void Qwen25OmniVisionEncoder::setupParameters(json &cfg, json &generation_cfg,
                                              json &nntr_cfg) {
  (void)generation_cfg;

  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");
  MEMORY_SWAP = nntr_cfg.value("fsu", false);
  head_weight_file = nntr_cfg.value(
    "vision_head_file_name",
    std::string("nntr_qwen2.5_omni_vision_head.bin"));

  json v = cfg;
  if (cfg.contains("thinker_config") &&
      cfg["thinker_config"].contains("vision_config"))
    v = cfg["thinker_config"]["vision_config"];
  else if (cfg.contains("vision_config"))
    v = cfg["vision_config"];

  DIM = v.value("hidden_size", v.value("embed_dim", 1280));
  NUM_LAYERS = v.value("depth", 32);
  NUM_HEADS = v.value("num_heads", 16);
  NUM_KEY_VALUE_HEADS = NUM_HEADS;
  GQA_SIZE = 1;
  HEAD_DIM = DIM / NUM_HEADS;
  INTERMEDIATE_SIZE = v.value("intermediate_size", 3420);
  OUT_HIDDEN = v.value("out_hidden_size", 2048);
  SPATIAL_MERGE = v.value("spatial_merge_size", 2);
  PATCH_SIZE = v.value("patch_size", 14);
  WINDOW_SIZE = v.value("window_size", 112);
  FULLATT = v.value("fullatt_block_indexes", std::vector<int>{7, 15, 23, 31});
  const unsigned int patch = PATCH_SIZE;
  const unsigned int tpatch = v.value("temporal_patch_size", 2);
  const unsigned int in_ch = v.value("in_channels", v.value("in_chans", 3));
  PATCH_DIM = in_ch * tpatch * patch * patch;
  MERGE_HIDDEN = DIM * SPATIAL_MERGE * SPATIAL_MERGE;
  NORM_EPS = 1e-6;
  ROPE_THETA = 0; // 2D-RoPE applied externally via vision_rope
  IS_CAUSAL = false;

  // grid for this run (raw patch counts); the patch file header overrides at
  // run() time but the compiled graph needs fixed dims, so take from config.
  GRID_H = nntr_cfg.value("grid_h", 8);
  GRID_W = nntr_cfg.value("grid_w", 8);
  GRID_T = nntr_cfg.value("grid_t", 1);

  const unsigned int seq = GRID_T * GRID_H * GRID_W;
  INIT_SEQ_LEN = seq;
  MAX_SEQ_LEN = seq;
  NUM_TO_GENERATE = 0;
}

Tensor Qwen25OmniVisionEncoder::createBlock(int layer_id, Tensor input) {
  const std::string p = "layer" + std::to_string(layer_id) + "_";

  LayerHandle norm1(createLayer(
    "rms_norm", {withKey("name", p + "norm1"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")}));
  Tensor normed = norm1(input);

  auto fc = [&](const std::string &name) {
    return createLayer("fully_connected",
                       {withKey("name", name), withKey("unit", DIM),
                        withKey("disable_bias", "false"),
                        withKey("weight_dtype", FC_LAYER_DTYPE)});
  };
  LayerHandle wq(fc(p + "wq")), wk(fc(p + "wk")), wv(fc(p + "wv"));
  Tensor q = wq(normed), k = wk(normed), v = wv(normed);

  auto rope = [&](const std::string &name) {
    return createLayer(
      "vision_rope",
      {withKey("name", name), withKey("grid_h", GRID_H),
       withKey("grid_w", GRID_W), withKey("grid_t", GRID_T),
       withKey("num_heads", NUM_HEADS),
       withKey("head_dim", HEAD_DIM),
       withKey("spatial_merge_size", SPATIAL_MERGE)});
  };
  LayerHandle rope_q(rope(p + "rope_q")), rope_k(rope(p + "rope_k"));
  q = rope_q(q);
  k = rope_k(k);

  const bool is_full =
    std::find(FULLATT.begin(), FULLATT.end(), layer_id) != FULLATT.end();
  LayerHandle attn(createLayer(
    "vision_attention",
    {withKey("name", p + "attention"), withKey("num_heads", NUM_HEADS),
     withKey("head_dim", HEAD_DIM), withKey("grid_h", GRID_H),
     withKey("grid_w", GRID_W), withKey("grid_t", GRID_T),
     withKey("window_size", WINDOW_SIZE),
     withKey("patch_size", PATCH_SIZE),
     withKey("spatial_merge_size", SPATIAL_MERGE),
     withKey("is_full", is_full ? "true" : "false")}));
  Tensor a = attn({q, k, v});

  LayerHandle proj(createLayer(
    "fully_connected", {withKey("name", p + "attention_out"),
                        withKey("unit", DIM), withKey("disable_bias", "false"),
                        withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor att = proj(a);

  LayerHandle attn_res(
    createLayer("addition", {withKey("name", p + "attn_residual")}));
  Tensor h = attn_res({input, att});

  LayerHandle norm2(createLayer(
    "rms_norm", {withKey("name", p + "norm2"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")}));
  Tensor n2 = norm2(h);

  auto ffc = [&](const std::string &name, unsigned int unit) {
    return createLayer("fully_connected",
                       {withKey("name", name), withKey("unit", unit),
                        withKey("disable_bias", "false"),
                        withKey("weight_dtype", FC_LAYER_DTYPE)});
  };
  LayerHandle gate(ffc(p + "ffn_gate", INTERMEDIATE_SIZE));
  LayerHandle up(ffc(p + "ffn_up", INTERMEDIATE_SIZE));
  Tensor g = gate(n2), u = up(n2);

  LayerHandle swiglu(
    createLayer("swiglu", {withKey("name", p + "ffn_swiglu")}));
  Tensor act = swiglu({g, u});

  LayerHandle down(ffc(p + "ffn_down", DIM));
  Tensor d = down(act);

  LayerHandle ffn_res(
    createLayer("addition", {withKey("name", p + "ffn_residual")}));
  return ffn_res({h, d});
}

std::pair<Tensor, Tensor> Qwen25OmniVisionEncoder::constructModel() {
  const unsigned int seq = GRID_T * GRID_H * GRID_W;

  // flattened patches: [B, 1, seq, PATCH_DIM]
  Tensor x({BATCH_SIZE, 1, seq, PATCH_DIM}, "input0");

  // patch_embed: Conv3d(kernel==stride==patch) == Linear(PATCH_DIM -> DIM)
  LayerHandle patch_embed(createLayer(
    "fully_connected",
    {withKey("name", "patch_embed"), withKey("unit", DIM),
     withKey("disable_bias", "true"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor h = patch_embed(x);

  for (int i = 0; i < NUM_LAYERS; ++i)
    h = createBlock(i, h);

  // merger.ln_q (RMSNorm over DIM, per patch — row count stays seq). The 2x2
  // merge reshape changes the row count, which the shared from/to slicing of
  // incremental_inference cannot express, so the reshape + MLP run as a
  // separate head graph (see initialize()/run()).
  LayerHandle ln_q(createLayer(
    "rms_norm", {withKey("name", "merger_ln_q"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")}));
  h = ln_q(h);

  return {x, h}; // [B, 1, seq, DIM]
}

void Qwen25OmniVisionEncoder::initialize() {
  Transformer::initialize(); // compiles patch_embed + blocks + ln_q

  const unsigned int merged =
    (GRID_T * GRID_H * GRID_W) / (SPATIAL_MERGE * SPATIAL_MERGE);

  head_model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  head_model->setProperty({withKey("batch_size", BATCH_SIZE),
                           withKey("epochs", "1"),
                           withKey("model_tensor_type", MODEL_TENSOR_TYPE)});

  // Head input is the block output reinterpreted as [merged, MERGE_HIDDEN]
  // (4 consecutive patches concatenated — a pure regroup of contiguous rows).
  Tensor hin({BATCH_SIZE, 1, merged, MERGE_HIDDEN}, "vision_head_input0");
  LayerHandle fc1(createLayer(
    "fully_connected",
    {withKey("name", "merger_mlp0"), withKey("unit", MERGE_HIDDEN),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor hh = fc1(hin);
  LayerHandle gelu(createLayer(
    "activation",
    {withKey("name", "merger_gelu"), withKey("activation", "gelu")}));
  hh = gelu(hh);
  LayerHandle fc2(createLayer(
    "fully_connected",
    {withKey("name", "merger_mlp2"), withKey("unit", OUT_HIDDEN),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  hh = fc2(hh);

  if (head_model->compile(hin, hh, ml::train::ExecutionMode::INFERENCE))
    throw std::invalid_argument("Vision merger head compilation failed.");
}

void Qwen25OmniVisionEncoder::load_weight(const std::string &weight_path) {
  Transformer::load_weight(weight_path);
  std::filesystem::path head_path =
    std::filesystem::path(weight_path).parent_path() / head_weight_file;
  try {
    head_model->load(head_path.string(),
                     ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load vision merger head from " +
                             head_path.string() + ": " + e.what());
  }
}

void Qwen25OmniVisionEncoder::registerCustomLayers() {
  Transformer::registerCustomLayers();
  const auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::VisionRopeLayer>);
  } catch (std::invalid_argument &e) {
    // already registered
  }
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::VisionAttentionLayer>);
  } catch (std::invalid_argument &e) {
    // already registered
  }
}

void Qwen25OmniVisionEncoder::run(const WSTR prompt, bool do_sample,
                                  const WSTR system_prompt,
                                  const WSTR tail_prompt, bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;
  if (!is_initialized)
    throw std::runtime_error("Vision encoder is not initialized.");

  const std::string path(prompt);
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Failed to open patch feature file: " + path);
  int32_t gh = 0, gw = 0;
  f.read(reinterpret_cast<char *>(&gh), sizeof(int32_t));
  f.read(reinterpret_cast<char *>(&gw), sizeof(int32_t));
  if (gh != static_cast<int32_t>(GRID_H) || gw != static_cast<int32_t>(GRID_W))
    throw std::runtime_error(
      "patch grid " + std::to_string(gh) + "x" + std::to_string(gw) +
      " != compiled grid " + std::to_string(GRID_H) + "x" +
      std::to_string(GRID_W) + " (set grid_h/grid_w in nntr_config.json)");

  const unsigned int seq = GRID_T * GRID_H * GRID_W;
  std::vector<float> patches(static_cast<size_t>(seq) * PATCH_DIM);
  f.read(reinterpret_cast<char *>(patches.data()),
         patches.size() * sizeof(float));
  if (!f)
    throw std::runtime_error("Truncated patch feature file: " + path);

  std::vector<float> embd = encode(patches.data());
  const unsigned int merged = seq / (SPATIAL_MERGE * SPATIAL_MERGE);

  const std::string out_path = path + ".embd";
  std::ofstream of(out_path, std::ios::binary);
  const int32_t n = static_cast<int32_t>(merged);
  const int32_t d = static_cast<int32_t>(OUT_HIDDEN);
  of.write(reinterpret_cast<const char *>(&n), sizeof(int32_t));
  of.write(reinterpret_cast<const char *>(&d), sizeof(int32_t));
  of.write(reinterpret_cast<const char *>(embd.data()),
           embd.size() * sizeof(float));
  of.close();

  if (log_output) {
    std::cout << "vision tokens: " << merged << " (dim " << OUT_HIDDEN
              << ") -> " << out_path << "\nfirst values:" << std::setprecision(7);
    for (unsigned int i = 0; i < 8 && i < OUT_HIDDEN; ++i)
      std::cout << " " << embd[i];
    std::cout << std::endl;
  }
  has_run_ = true;
}

std::vector<float> Qwen25OmniVisionEncoder::encode(const float *patches) {
  if (!is_initialized)
    throw std::runtime_error("Vision encoder is not initialized.");
  const unsigned int seq = GRID_T * GRID_H * GRID_W;
  const unsigned int merged = seq / (SPATIAL_MERGE * SPATIAL_MERGE);

  std::vector<float *> in = {const_cast<float *>(patches)};
  std::vector<float *> label;
  std::vector<float *> out =
    model->incremental_inference(BATCH_SIZE, in, label, seq, 0, seq, false);

  // [seq, DIM] block output -> [merged, MERGE_HIDDEN]: a pure regroup of the
  // contiguous buffer (4 consecutive patches concatenated), fed to the head.
  std::vector<float *> hin = {out[0]};
  std::vector<float *> hout = head_model->incremental_inference(
    BATCH_SIZE, hin, label, merged, 0, merged, false);

  return std::vector<float>(
    hout[0], hout[0] + static_cast<size_t>(merged) * OUT_HIDDEN);
}

} // namespace causallm

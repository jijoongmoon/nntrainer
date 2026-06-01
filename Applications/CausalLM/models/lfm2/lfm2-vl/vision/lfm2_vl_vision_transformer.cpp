// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_vision_transformer.cpp
 * @date   13 May 2026
 * @brief  CLIP/SigLIP-style Vision Transformer encoder.
 */

#include <lfm2_vl_vision_transformer.h>
#include <llm_util.hpp>
#include <model.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace causallm {

namespace {

std::string toStringPrecise(float v) {
  std::ostringstream oss;
  oss << std::setprecision(20) << v;
  return oss.str();
}

const char *PATCH_EMBED_NAME = "v_patch_embd";
const char *PATCH_SEQ_NAME = "v_patch_seq";
const char *POST_LN_NAME = "v_post_ln";

std::string blkName(int i, const char *suffix) {
  return "v_blk_" + std::to_string(i) + "_" + suffix;
}

} // namespace

json &Lfm2VlVisionTransformer::sanitizeConfig(json &cfg) {
  // base Transformer::setupParameters() calls .get<T>() on these without a
  // default, so missing keys would throw. Values written here are overwritten
  // by Lfm2VlVisionTransformer::setupParameters() immediately afterwards.
  if (!cfg.contains("vocab_size"))
    cfg["vocab_size"] = 0u;
  if (!cfg.contains("max_position_embeddings")) {
    unsigned int img = cfg.value("image_size", 256u);
    unsigned int patch = cfg.value("patch_size", 16u);
    cfg["max_position_embeddings"] = (img / patch) * (img / patch);
  }
  if (!cfg.contains("rope_theta"))
    cfg["rope_theta"] = 0u;
  if (!cfg.contains("tie_word_embeddings"))
    cfg["tie_word_embeddings"] = false;
  if (!cfg.contains("rms_norm_eps"))
    cfg["rms_norm_eps"] = cfg.value("layer_norm_eps", 1e-6f);
  if (!cfg.contains("num_attention_heads"))
    cfg["num_attention_heads"] = 12;
  if (!cfg.contains("hidden_size"))
    cfg["hidden_size"] = 768;
  if (!cfg.contains("num_hidden_layers"))
    cfg["num_hidden_layers"] = 12;
  if (!cfg.contains("intermediate_size"))
    cfg["intermediate_size"] = 3072;
  return cfg;
}

void Lfm2VlVisionTransformer::setupParameters(json &cfg, json &generation_cfg,
                                         json &nntr_cfg) {
  // Vision Transformer parameters. Defaults follow LFM2.5-VL's vision tower
  // (SigLIP2-style, 86M).
  DIM = cfg.value("hidden_size", 768);
  NUM_HEADS = cfg.value("num_attention_heads", 12);
  NUM_KEY_VALUE_HEADS = cfg.value("num_key_value_heads", NUM_HEADS);
  HEAD_DIM = cfg.contains("head_dim") ? cfg["head_dim"].get<int>()
                                      : DIM / NUM_HEADS;
  NUM_LAYERS = cfg.value("num_hidden_layers", 12);
  INTERMEDIATE_SIZE = cfg.value("intermediate_size", 3072);
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  // ViT-specific.
  IMAGE_SIZE = cfg.value("image_size", 256u);
  PATCH_SIZE = cfg.value("patch_size", 16u);
  NUM_CHANNELS = cfg.value("num_channels", 3u);
  NUM_PATCHES = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE);

  // Non-causal, no RoPE.
  IS_CAUSAL = false;
  ROPE_THETA = 0;
  NORM_EPS = cfg.value("layer_norm_eps", 1e-6f);

  // The "sequence length" of the vision graph is the patch grid size.
  // MAX_SEQ_LEN sizes the working/KV-cache memory pool; keep it generous
  // (TimmViT uses 1000) so the mha_core cache (max_timestep = NUM_PATCHES + 1)
  // and attention buffers fit. Read from nntr_config so it stays tunable.
  INIT_SEQ_LEN = NUM_PATCHES;
  MAX_SEQ_LEN = NUM_PATCHES + 1;
  MAX_POSITION_EMBEDDINGS = NUM_PATCHES + 1;

  // Misc.
  BATCH_SIZE = nntr_cfg.value("batch_size", 1u);
  MEMORY_SWAP = nntr_cfg.value("memory_swap", false);
  FSU_LOOKAHEAD = nntr_cfg.value("fsu_lookahead", 0u);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", std::string("FP32"));
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", std::string("FP32"));
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", std::string("FP32"));

  // Fields the LLM path uses but ViT does not. Kept defined so any base-class
  // code path that touches them stays well-formed.
  NUM_VOCAB = 0;
  TIE_WORD_EMBEDDINGS = false;
  EMBEDDING_SCALE = 1.0f;
  NUM_TO_GENERATE = 0;
  USE_VOCAB_SELECTION = false;
  SLIDING_WINDOW = UINT_MAX;
  ATTN_LOGIT_SOFTCAPPING = 0.0f;
}

std::pair<Tensor, Tensor> Lfm2VlVisionTransformer::constructModel() {

  // [B, C, H, W] image input.
  Tensor x =
    Tensor({BATCH_SIZE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE}, "input0");

  // Conv2D patch embedding: kernel = stride = PATCH_SIZE, padding "valid".
  // Output: [B, DIM, H/P, W/P].
  const std::string ps = std::to_string(PATCH_SIZE);
  LayerHandle patch_embed(createLayer(
    "conv2d",
    {withKey("name", PATCH_EMBED_NAME), withKey("filters", DIM),
     withKey("kernel_size", ps + "," + ps), withKey("stride", ps + "," + ps),
     withKey("padding", "0,0"), withKey("bias_initializer", "zeros"),
     withKey("weight_initializer", "xavier_uniform")}));
  Tensor patch = patch_embed(x);

  // Conv2D output is [B, DIM, H/P, W/P] (channel-major). Reshape to
  // [B, 1, DIM, NUM_PATCHES] then permute the DIM and patch axes to get the
  // proper token sequence [B, 1, NUM_PATCHES, DIM] (each row = one patch's
  // full DIM feature vector). Mirrors TimmViT's patch_embed reorder.
  LayerHandle patch_seq(createLayer(
    "reshape",
    {withKey("name", PATCH_SEQ_NAME),
     withKey("target_shape",
             "1:" + std::to_string(DIM) + ":" + std::to_string(NUM_PATCHES))}));
  Tensor h = patch_seq(patch);

  LayerHandle patch_perm(createLayer(
    "permute",
    {withKey("name", "v_patch_perm"), withKey("direction", {1, 3, 2})}));
  h = patch_perm(h);

  // Learnable position embedding [NUM_PATCHES, DIM], added to the patch
  // sequence (loaded from the weight file as tensor "v_pos_embd").
  LayerHandle pos_embed(createLayer(
    "weight",
    {withKey("name", "v_pos_embd"),
     withKey("dim", "1:1:" + std::to_string(NUM_PATCHES) + ":" +
                      std::to_string(DIM)),
     withKey("tensor_dtype", "FP32"), withKey("weight_name", "v_pos_embd")}));
  Tensor pos = pos_embed(x);

  LayerHandle pos_add(createLayer("addition", {withKey("name", "v_pos_add")}));
  h = pos_add({h, pos});

  // Encoder blocks.
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createEncoderBlock(i, h);
  }

  // Final post-LN.
  LayerHandle post_ln(createLayer(
    "layer_normalization",
    {withKey("name", POST_LN_NAME),
     withKey("epsilon", toStringPrecise(NORM_EPS)), withKey("axis", 3),
     withKey("packed", "false")}));
  h = post_ln(h);

  return {x, h};
}

Tensor Lfm2VlVisionTransformer::createEncoderBlock(int layer_id, Tensor input) {

  // Pre-LN.
  LayerHandle ln1(createLayer(
    "layer_normalization",
    {withKey("name", blkName(layer_id, "ln1")),
     withKey("epsilon", toStringPrecise(NORM_EPS)), withKey("axis", 3),
     withKey("packed", "false")}));
  Tensor x = ln1(input);

  // Self-attention.
  Tensor a = createSelfAttention(layer_id, x);

  // Residual.
  LayerHandle attn_res(createLayer(
    "addition", {withKey("name", blkName(layer_id, "attn_res"))}));
  Tensor h = attn_res({input, a});

  // Pre-LN before MLP.
  LayerHandle ln2(createLayer(
    "layer_normalization",
    {withKey("name", blkName(layer_id, "ln2")),
     withKey("epsilon", toStringPrecise(NORM_EPS)), withKey("axis", 3),
     withKey("packed", "false")}));
  Tensor n = ln2(h);

  // MLP.
  Tensor m = createVitMlp(layer_id, n);

  // Residual.
  LayerHandle ffn_res(createLayer(
    "addition", {withKey("name", blkName(layer_id, "ffn_res"))}));
  return ffn_res({h, m});
}

Tensor Lfm2VlVisionTransformer::createSelfAttention(int layer_id, Tensor x) {

  // Q / K / V projections (CLIP/SigLIP-style: bias enabled).
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", blkName(layer_id, "attn_q")),
     withKey("unit", HEAD_DIM * NUM_HEADS), withKey("disable_bias", "false"),
     withKey("weight_initializer", "xavier_uniform")}));
  Tensor q = wq(x);

  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", blkName(layer_id, "attn_k")),
     withKey("unit", HEAD_DIM * NUM_HEADS / GQA_SIZE),
     withKey("disable_bias", "false"),
     withKey("weight_initializer", "xavier_uniform")}));
  Tensor k = wk(x);

  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", blkName(layer_id, "attn_v")),
     withKey("unit", HEAD_DIM * NUM_HEADS / GQA_SIZE),
     withKey("disable_bias", "false"),
     withKey("weight_initializer", "xavier_uniform")}));
  Tensor v = wv(x);

  // Bidirectional attention, no RoPE, no external KV cache.
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", blkName(layer_id, "attn")),
     withKey("num_heads", NUM_HEADS),
     withKey("num_heads_kv", NUM_HEADS / GQA_SIZE),
     withKey("max_timestep", std::to_string(NUM_PATCHES + 1)),
     withKey("rope_theta", ROPE_THETA), withKey("is_causal", "false"),
     withKey("use_gemm_attention", "true")}));
  Tensor a = mha({q, k, v});

  // Output projection (with bias).
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", blkName(layer_id, "attn_out")), withKey("unit", DIM),
     withKey("disable_bias", "false"),
     withKey("weight_initializer", "xavier_uniform")}));
  return wo(a);
}

Tensor Lfm2VlVisionTransformer::createVitMlp(int layer_id, Tensor x) {

  LayerHandle up(createLayer(
    "fully_connected",
    {withKey("name", blkName(layer_id, "ffn_up")),
     withKey("unit", INTERMEDIATE_SIZE), withKey("disable_bias", "false"),
     withKey("weight_initializer", "xavier_uniform")}));
  Tensor h = up(x);

  LayerHandle act(createLayer(
    "activation", {withKey("name", blkName(layer_id, "ffn_act")),
                   withKey("activation", "gelu")}));
  h = act(h);

  LayerHandle down(createLayer(
    "fully_connected",
    {withKey("name", blkName(layer_id, "ffn_down")), withKey("unit", DIM),
     withKey("disable_bias", "false"),
     withKey("weight_initializer", "xavier_uniform")}));
  return down(h);
}

void Lfm2VlVisionTransformer::run(const WSTR image_tensor_path, bool /*do_sample*/,
                             const WSTR /*system_prompt*/,
                             const WSTR /*tail_prompt*/, bool log_output) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Lfm2VlVisionTransformer is not initialized. Call initialize() first.");
  }

  // Load preprocessed image tensor (raw fp32, NCHW, BATCH_SIZE x C x H x W).
  const size_t n_elems =
    static_cast<size_t>(BATCH_SIZE) * NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE;
  std::vector<float> image(n_elems);
  std::ifstream in(image_tensor_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open image tensor file: " +
                             image_tensor_path);
  }
  in.read(reinterpret_cast<char *>(image.data()),
          static_cast<std::streamsize>(n_elems * sizeof(float)));
  if (in.gcount() !=
      static_cast<std::streamsize>(n_elems * sizeof(float))) {
    throw std::runtime_error(
      "Image tensor file is smaller than expected. Need " +
      std::to_string(n_elems) + " fp32 elements (NCHW=" +
      std::to_string(BATCH_SIZE) + "x" + std::to_string(NUM_CHANNELS) + "x" +
      std::to_string(IMAGE_SIZE) + "x" + std::to_string(IMAGE_SIZE) + ").");
  }

  // Forward.
  std::vector<float *> in_ptrs = {image.data()};
  int vit_iters = 1;
  if (const char *it = std::getenv("NNTR_VIT_ITERS"))
    vit_iters = std::max(1, std::atoi(it));
  std::vector<float *> out_ptrs;
  std::vector<float *> vit_label;
  auto vit_t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < vit_iters; ++r)
    out_ptrs = model->incremental_inference(BATCH_SIZE, in_ptrs, vit_label,
                                            NUM_PATCHES, 0, NUM_PATCHES, false);
  auto vit_t1 = std::chrono::high_resolution_clock::now();
  std::cout << "[vit infer] "
            << std::chrono::duration<double, std::milli>(vit_t1 - vit_t0).count() /
                 vit_iters
            << " ms/iter over " << vit_iters << " iters" << std::endl;

  // Debug: dump the full feature tensor to a file for parity comparison.
  if (const char *out_path = std::getenv("NNTR_VIT_OUT")) {
    if (!out_ptrs.empty() && out_ptrs[0] != nullptr) {
      const size_t n_out =
        static_cast<size_t>(BATCH_SIZE) * NUM_PATCHES * DIM;
      std::ofstream of(out_path, std::ios::binary);
      of.write(reinterpret_cast<const char *>(out_ptrs[0]),
               static_cast<std::streamsize>(n_out * sizeof(float)));
      std::cout << "[NNTR_VIT_OUT] wrote " << n_out << " floats to " << out_path
                << std::endl;
    }
  }

  if (log_output && !out_ptrs.empty() && out_ptrs[0] != nullptr) {
    std::cout << "Lfm2VlVisionTransformer features [" << NUM_PATCHES << "x" << DIM
              << "], first 10 values: ";
    for (int i = 0; i < 10 && i < DIM; ++i) {
      std::cout << out_ptrs[0][i] << (i == 9 ? "" : ", ");
    }
    std::cout << " ..." << std::endl;
  }

  // NOTE: inference() output buffers are not owned here; deleting them
  // double-frees (pre-existing). Left to process teardown for this debug path.
  for (auto p : (std::vector<float *>){}) {
    delete[] p;
  }
}

} // namespace causallm

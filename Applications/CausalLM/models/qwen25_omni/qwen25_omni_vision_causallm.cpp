// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_vision_causallm.cpp
 * @date   13 June 2026
 * @brief  Qwen2.5-Omni Thinker with image input (image + text in / text out).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#include <app_context.h>
#include <embedding_injection.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>
#include <mrope_apply.h>
#include <qwen25_omni_vision_causallm.h>

namespace causallm {

void Qwen25OmniVisionCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  const auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingInjectionLayer>);
  } catch (std::invalid_argument &e) {
  }
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::MRoPEApplyLayer>);
  } catch (std::invalid_argument &e) {
  }
}

Tensor Qwen25OmniVisionCausalLM::createAttention(const int layer_id,
                                                 int seq_len, int n_heads,
                                                 int head_dim, Tensor query,
                                                 Tensor key, Tensor value) {
  const std::string p = "layer" + std::to_string(layer_id) + "_";

  // Q/K/V (Qwen2 carries bias on q/k/v)
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", p + "wq"), withKey("unit", head_dim * n_heads),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor q = wq(query);
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", p + "wk"), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor k = wk(key);
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", p + "wv"), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor v = wv(value);

  // M-RoPE on q and k (host-computed cos/sin); mha_core then runs theta=0.
  LayerHandle rope_q(
    createLayer("mrope_apply", {withKey("name", p + "rope_q")}));
  q = rope_q({q, mrope_cos_t, mrope_sin_t});
  LayerHandle rope_k(
    createLayer("mrope_apply", {withKey("name", p + "rope_k")}));
  k = rope_k({k, mrope_cos_t, mrope_sin_t});

  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);

  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", p + "attention"), withKey("num_heads", n_heads),
     withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("sliding_window", SLIDING_WINDOW), withKey("rope_theta", "0"),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false")}));
  Tensor a = mha({q, k, v, cache_k, cache_v});

  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", p + "attention_out"), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones")}));
  return wo(a);
}

std::pair<Tensor, Tensor> Qwen25OmniVisionCausalLM::constructModel() {
  Tensor x =
    Tensor({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");

  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";
  LayerHandle embedding(createLayer(
    embedding_type,
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));
  Tensor h = embedding(x);

  // image embedding side input + injection at <|IMAGE|> positions
  const std::string img_shape = std::to_string(BATCH_SIZE) + ":1:" +
                                std::to_string(INIT_SEQ_LEN) + ":" +
                                std::to_string(DIM);
  LayerHandle img_input(createLayer(
    "input",
    {withKey("name", "img_embd"), withKey("input_shape", img_shape)}));
  img_embd_t = img_input(Tensor());
  LayerHandle inject(createLayer(
    "embedding_injection",
    {withKey("name", "img_inject"),
     withKey("token_id", std::to_string(IMAGE_TOKEN_ID) + "," +
                           std::to_string(VIDEO_TOKEN_ID))}));
  h = inject({h, x, img_embd_t});

  // shared M-RoPE cos/sin side inputs [B,1,MAX_SEQ_LEN,HEAD_DIM]
  const std::string rope_shape = std::to_string(BATCH_SIZE) + ":1:" +
                                 std::to_string(MAX_SEQ_LEN) + ":" +
                                 std::to_string(HEAD_DIM);
  LayerHandle cos_input(createLayer(
    "input",
    {withKey("name", "mrope_cos"), withKey("input_shape", rope_shape)}));
  mrope_cos_t = cos_input(Tensor());
  LayerHandle sin_input(createLayer(
    "input",
    {withKey("name", "mrope_sin"), withKey("input_shape", rope_shape)}));
  mrope_sin_t = sin_input(Tensor());

  for (int i = 0; i < NUM_LAYERS; ++i)
    h = createTransformerDecoderBlock(i, h);

  LayerHandle out_norm(
    createLayer("rms_norm", {withKey("name", "output_norm"),
                             withKey("epsilon", std::to_string(NORM_EPS)),
                             withKey("packed", "false")}));
  h = out_norm(h);

  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";
  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"), withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"), withKey("weight_dtype", LMHEAD_DTYPE)};
  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));
  LayerHandle lmhead(createLayer(lmhead_type, lmhead_prop));
  Tensor y = lmhead(h);

  return {x, y};
}

void Qwen25OmniVisionCausalLM::initialize() {
  CausalLM::initialize();

  img_buf.assign(static_cast<size_t>(INIT_SEQ_LEN) * DIM, 0.0f);
  mrope_cos.assign(static_cast<size_t>(MAX_SEQ_LEN) * HEAD_DIM, 1.0f);
  mrope_sin.assign(static_cast<size_t>(MAX_SEQ_LEN) * HEAD_DIM, 0.0f);

  if (vision_encoder_path.empty()) {
    std::cerr << "[Warning] nntr_config has no vision_encoder_path; only text "
                 "prompts will work."
              << std::endl;
    return;
  }
  vision_cfg = LoadJsonFile(vision_encoder_path + "/config.json");
  vision_gen_cfg = json::object();
  vision_nntr_cfg = LoadJsonFile(vision_encoder_path + "/nntr_config.json");
  vision_encoder = std::make_unique<Qwen25OmniVisionEncoder>(
    vision_cfg, vision_gen_cfg, vision_nntr_cfg);
  vision_encoder->initialize();
  vision_encoder->load_weight(
    vision_encoder_path + "/" +
    vision_nntr_cfg["model_file_name"].get<std::string>());
}

std::vector<float *>
Qwen25OmniVisionCausalLM::buildInferenceInputs(float *input_sample) {
  // The compiled graph orders inputs by creation: input0, then img_embd,
  // mrope_cos, mrope_sin (created right after the embedding), then the
  // per-layer KV caches (name-sorted, matching the base text path).
  std::vector<std::pair<std::string, float *>> caches;
  caches.reserve(static_cast<size_t>(NUM_LAYERS) * 2);
  for (int i = 0; i < NUM_LAYERS; ++i) {
    caches.emplace_back(
      "cache_k_l" + std::to_string(i),
      reinterpret_cast<float *>(kv_cache.getKeyCache(i).getData()));
    caches.emplace_back(
      "cache_v_l" + std::to_string(i),
      reinterpret_cast<float *>(kv_cache.getValueCache(i).getData()));
  }
  std::sort(caches.begin(), caches.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  std::vector<float *> in;
  in.reserve(4 + caches.size());
  in.push_back(input_sample);
  in.push_back(img_buf.data());
  in.push_back(mrope_cos.data());
  in.push_back(mrope_sin.data());
  for (const auto &c : caches)
    in.push_back(c.second);
  return in;
}

void Qwen25OmniVisionCausalLM::buildMRoPETables(
  const std::vector<std::array<int, 3>> &pos3d) {
  const unsigned int hd = HEAD_DIM;
  const unsigned int half = hd / 2; // 64
  const float theta = static_cast<float>(ROPE_THETA);

  std::vector<float> inv_freq(half);
  for (unsigned int j = 0; j < half; ++j)
    inv_freq[j] = 1.0f / std::pow(theta, static_cast<float>(2 * j) / hd);

  // section boundaries: mrope_section doubled -> [t,h,w,t,h,w]
  std::vector<int> sec2;
  for (int s : MROPE_SECTION)
    sec2.push_back(s);
  for (int s : MROPE_SECTION)
    sec2.push_back(s);
  // dim -> position axis (0=t,1=h,2=w)
  std::vector<int> axis(hd, 0);
  {
    unsigned int d = 0;
    for (size_t c = 0; c < sec2.size(); ++c)
      for (int n = 0; n < sec2[c] && d < hd; ++n, ++d)
        axis[d] = static_cast<int>(c % 3);
  }

  int max_pos = 0;
  for (const auto &p : pos3d)
    max_pos = std::max({max_pos, p[0], p[1], p[2]});

  for (unsigned int pidx = 0; pidx < MAX_SEQ_LEN; ++pidx) {
    int t, hh, ww;
    if (pidx < pos3d.size()) {
      t = pos3d[pidx][0];
      hh = pos3d[pidx][1];
      ww = pos3d[pidx][2];
    } else {
      const int val = max_pos + 1 + static_cast<int>(pidx - pos3d.size());
      t = hh = ww = val; // generated text: sequential
    }
    float *crow = &mrope_cos[static_cast<size_t>(pidx) * hd];
    float *srow = &mrope_sin[static_cast<size_t>(pidx) * hd];
    for (unsigned int d = 0; d < hd; ++d) {
      const int pos = axis[d] == 0 ? t : (axis[d] == 1 ? hh : ww);
      const float angle = pos * inv_freq[d % half];
      crow[d] = std::cos(angle);
      srow[d] = std::sin(angle);
    }
  }
}

void Qwen25OmniVisionCausalLM::run(const WSTR prompt, bool do_sample,
                                   const WSTR system_prompt,
                                   const WSTR tail_prompt, bool log_output) {
  static const std::string kImage = "image:";
  static const std::string kVideo = "video:";
  const bool is_image = prompt.rfind(kImage, 0) == 0;
  const bool is_video = prompt.rfind(kVideo, 0) == 0;

  // ---- pure text: sequential 3D positions, no vision ----
  if (!is_image && !is_video) {
    std::fill(img_buf.begin(), img_buf.end(), 0.0f);
    std::vector<std::array<int, 3>> pos3d;
    auto ids = tokenizer->Encode(system_prompt + prompt + tail_prompt);
    for (size_t i = 0; i < ids.size(); ++i)
      pos3d.push_back({(int)i, (int)i, (int)i});
    buildMRoPETables(pos3d);
    CausalLM::run(prompt, do_sample, system_prompt, tail_prompt, log_output);
    return;
  }

  if (!vision_encoder)
    throw std::runtime_error(
      "vision prompt given but vision_encoder_path is not configured");

  // "<image|video>:<patch_file> <question>"
  const int token_id = is_video ? VIDEO_TOKEN_ID : IMAGE_TOKEN_ID;
  const char *placeholder = is_video ? "<|VIDEO|>" : "<|IMAGE|>";
  const std::string rest = prompt.substr(kImage.size()); // both prefixes len 6
  const auto space = rest.find(' ');
  if (space == std::string::npos)
    throw std::invalid_argument("syntax: <image|video>:<patch_file> <question>");
  const std::string patch_path = rest.substr(0, space);
  const std::string question = rest.substr(space + 1);

  // read the flattened patches ([int32 gh][int32 gw][fp32 t*gh*gw x patch_dim])
  std::ifstream f(patch_path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Failed to open patch file: " + patch_path);
  int32_t gh = 0, gw = 0;
  f.read(reinterpret_cast<char *>(&gh), sizeof(int32_t));
  f.read(reinterpret_cast<char *>(&gw), sizeof(int32_t));
  if (gh != static_cast<int32_t>(vision_encoder->gridH()) ||
      gw != static_cast<int32_t>(vision_encoder->gridW()))
    throw std::runtime_error(
      "patch grid does not match the vision encoder's compiled grid");
  const unsigned int grid_t = vision_encoder->gridT();
  const unsigned int seq =
    grid_t * static_cast<unsigned int>(gh) * gw;
  std::vector<float> patches(static_cast<size_t>(seq) *
                             vision_encoder->patchDim());
  f.read(reinterpret_cast<char *>(patches.data()),
         patches.size() * sizeof(float));
  if (!f)
    throw std::runtime_error("Truncated patch file: " + patch_path);
  f.close();

  const int merge = 2;
  const int llm_h = gh / merge, llm_w = gw / merge;
  const int per_frame = llm_h * llm_w;
  const size_t n_vis = static_cast<size_t>(grid_t) * per_frame;

  std::vector<float> vis_embd = vision_encoder->encode(patches.data());
  if (vis_embd.size() != n_vis * DIM)
    throw std::runtime_error("vision embedding size mismatch: " +
                             std::to_string(vis_embd.size()) + " vs " +
                             std::to_string(n_vis * DIM));

  // temporal position step per frame (HF: arange(t)*second_per_grid*25)
  const int t_step =
    is_video ? static_cast<int>(std::lround(VIDEO_SECOND_PER_GRID *
                                            POSITION_ID_PER_SECONDS))
             : 0;

  // build prompt: system + user(<|vision_bos|> placeholder*n <|vision_eos|> Q)
  std::string vis = "<|vision_bos|>";
  for (size_t i = 0; i < n_vis; ++i)
    vis += placeholder;
  vis += "<|vision_eos|>";
  const std::string full =
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n" +
    vis + question + "<|im_end|>\n<|im_start|>assistant\n";

  auto ids = tokenizer->Encode(full);

  // ---- scatter vision embeddings + build 3D positions (get_rope_index) ----
  std::fill(img_buf.begin(), img_buf.end(), 0.0f);
  std::vector<std::array<int, 3>> pos3d(ids.size());
  size_t scattered = 0;
  int run_max = -1; // running max position (st_idx = run_max + 1)
  for (size_t i = 0; i < ids.size(); ++i) {
    if (static_cast<int>(ids[i]) == token_id) {
      const int st = run_max + 1; // vision block start (after vision_bos)
      const int g = static_cast<int>(scattered);
      const int frame = g / per_frame;
      const int sp = g % per_frame;
      pos3d[i] = {st + frame * t_step, st + sp / llm_w, st + sp % llm_w};
      if (i < INIT_SEQ_LEN)
        std::memcpy(&img_buf[i * static_cast<size_t>(DIM)],
                    &vis_embd[g * static_cast<size_t>(DIM)],
                    DIM * sizeof(float));
      ++scattered;
    } else {
      const int pos = run_max + 1; // text / vision_bos / vision_eos
      pos3d[i] = {pos, pos, pos};
      run_max = pos;
    }
    // after the last vision token, advance run_max to the block's max pos
    if (static_cast<int>(ids[i]) == token_id &&
        (i + 1 >= ids.size() ||
         static_cast<int>(ids[i + 1]) != token_id)) {
      const int st = pos3d[i][0] - (static_cast<int>(grid_t) - 1) * t_step;
      run_max = st + std::max({(static_cast<int>(grid_t) - 1) * t_step,
                               llm_h - 1, llm_w - 1});
    }
  }
  if (scattered != n_vis)
    throw std::runtime_error("vision token/embedding count mismatch: " +
                             std::to_string(scattered) + " vs " +
                             std::to_string(n_vis));
  buildMRoPETables(pos3d);

  if (log_output)
    std::cout << "[" << (is_video ? "Video" : "Image") << "] " << patch_path
              << " grid t=" << grid_t << " " << gh << "x" << gw << " -> "
              << n_vis << " tokens" << std::endl;

  CausalLM::run(full, do_sample, "", "", log_output);
}

} // namespace causallm

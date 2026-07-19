// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_talker_causallm.cpp
 * @date   13 June 2026
 * @brief  Qwen2.5-Omni Talker: codec-token LM (Phase 1 of speech output).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>
#include <mrope_apply.h>
#include <nntrainer_error.h>
#include <tensor.h>

#include <qwen25_omni_causallm.h>
#include <qwen25_omni_talker_causallm.h>
#include <qwen25_omni_token2wav.h>

namespace causallm {

/**
 * @brief Non-inplace identity tap: copies its input to a FRESH output buffer so
 *        that, wired with no consumer, it becomes a graph-leaf output and is
 *        returned by incremental_inference. (The core "identity" layer is
 *        in-place, so its output would alias — and get overwritten by — the
 *        tapped activation; this one forces a copy.) Used to expose the
 *        Thinker's output_norm (hidden) and embedding0 (token embed).
 */
class CaptureTapLayer final : public nntrainer::Layer {
public:
  CaptureTapLayer() : Layer() {}
  ~CaptureTapLayer() {}

  void finalize(nntrainer::InitLayerContext &context) override {
    NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
      << "capture_tap expects 1 input";
    context.setOutputDimensions({context.getInputDimensions()[0]});
  }

  void forwarding(nntrainer::RunLayerContext &context, bool) override {
    context.getOutput(0).copyData(context.getInput(0));
  }

  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool) override {
    nntrainer::Tensor &in = context.getInput(0);
    nntrainer::Tensor &out = context.getOutput(0);
    const unsigned int w = in.width();
    const unsigned int iter = to - from;
    for (unsigned int b = 0; b < in.batch(); ++b)
      for (unsigned int s = 0; s < iter; ++s)
        std::memcpy(out.getData<float>() + out.getIndex(b, 0, s, 0),
                    in.getData<float>() + in.getIndex(b, 0, s, 0),
                    static_cast<size_t>(w) * sizeof(float));
  }

  void calcDerivative(nntrainer::RunLayerContext &) override {}
  bool supportBackwarding() const override { return false; }
  void exportTo(nntrainer::Exporter &,
                const ml::train::ExportMethods &) const override {}
  const std::string getType() const override { return type; }
  void setProperty(const std::vector<std::string> &) override {}

  void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override {
    nntrainer::TensorDim in_dim = context.getInput(0).getDim();
    nntrainer::TensorDim out_dim = context.getOutput(0).getDim();
    in_dim.height(input_dimensions[0].height());
    out_dim.height(input_dimensions[0].height());
    context.updateInput(0, in_dim);
    context.updateOutput(0, out_dim);
  }

  inline static const std::string type = "capture_tap";
};

/**
 * @brief Thinker subclass that exposes per-token activation capture and greedy
 *        reply generation. Subclassing grants access to the protected model /
 *        KV-cache machinery; the captured output_norm (final hidden) and
 *        embedding0 (token embed) build the Talker's thinker_reply_part.
 */
class ThinkerForCapture : public Qwen25OmniCausalLM {
public:
  ThinkerForCapture(json &cfg, json &gen, json &nntr) :
    Transformer(Qwen25OmniCausalLM::flattenThinkerTextConfig(cfg), gen, nntr,
                ModelType::CAUSALLM),
    Qwen25OmniCausalLM(cfg, gen, nntr) {}

  int dim() const { return DIM; }

  /** Rebuild the decoder graph capturing the embedding0 and output_norm output
   *  tensors so they can be exposed as graph outputs (see initialize()). */
  std::pair<Tensor, Tensor> constructModel() override {
    Tensor x(
      {1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");
    const std::string embedding_type =
      TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";
    LayerHandle embedding(createLayer(
      embedding_type,
      {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
       "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
       "scale=" + std::to_string(EMBEDDING_SCALE)}));
    Tensor h = embedding(x);
    // tap the token embedding (leaf -> returned as an inference output)
    LayerHandle embed_tap(
      createLayer("capture_tap", {withKey("name", "embed_tap")}));
    embedding0_out_ = embed_tap(h);
    for (int i = 0; i < NUM_LAYERS; ++i)
      h = createTransformerDecoderBlock(i, h);
    LayerHandle out_norm(
      createLayer("rms_norm", {withKey("name", "output_norm"),
                               withKey("epsilon", std::to_string(NORM_EPS)),
                               withKey("packed", "false")}));
    h = out_norm(h);
    // tap the final hidden (leaf -> returned as an inference output)
    LayerHandle hidden_tap(
      createLayer("capture_tap", {withKey("name", "hidden_tap")}));
    output_norm_out_ = hidden_tap(h);
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

  /** Compile with [lm_head, output_norm, embedding0] as outputs so the latter
   *  two are preserved (not aliased) and returned per inference. lm_head is the
   *  first output, so the DFS-from-output weight order is unchanged. */
  void initialize() override {
    registerCustomLayers();
    model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    std::vector<std::string> model_props = {
      withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
      withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
    if (MEMORY_SWAP) {
      model_props.emplace_back(withKey("fsu", "true"));
      model_props.emplace_back(withKey("fsu_lookahead", FSU_LOOKAHEAD));
    }
    model->setProperty(model_props);
    auto [x, y] = constructModel();
    std::vector<Tensor> outs = {y, output_norm_out_, embedding0_out_};
    if (model->compile(x, outs, ml::train::ExecutionMode::INFERENCE))
      throw std::invalid_argument("ThinkerForCapture compilation failed.");
    is_initialized = true;
  }

  /** Row-by-row prefill of @p ids; snapshot output_norm (hidden) + embedding0
   *  (token embed) per row from the RETURNED outputs (fresh, un-aliased).
   *  Incremental (causal, exact) — avoids the batched-prefill last-row issue. */
  void captureActivations(const std::vector<unsigned int> &ids,
                          std::vector<float> &hidden,
                          std::vector<float> &embed) {
    allocateAndBindKVCache();
    setKVCachePosition(0);
    const size_t L = ids.size();
    hidden.assign(L * DIM, 0.0f);
    embed.assign(L * DIM, 0.0f);
    auto *in_sample =
      static_cast<float *>(malloc(sizeof(float) * MAX_SEQ_LEN));
    std::vector<float *> label;
    for (size_t r = 0; r < L; ++r) {
      in_sample[0] = static_cast<float>(ids[r]);
      std::vector<float *> in = buildInferenceInputs(in_sample);
      std::vector<float *> out = model->incremental_inference(
        1, in, label, static_cast<unsigned int>(L),
        static_cast<unsigned int>(r), static_cast<unsigned int>(r) + 1, false);
      NNTR_THROW_IF(out.size() < 3, std::runtime_error)
        << "thinker capture expects 3 outputs (got " << out.size() << ")";
      std::memcpy(&hidden[r * DIM], out[1],
                  static_cast<size_t>(DIM) * sizeof(float)); // output_norm
      std::memcpy(&embed[r * DIM], out[2],
                  static_cast<size_t>(DIM) * sizeof(float)); // embedding0
      for (auto o : out)
        delete[] o;
    }
    free(in_sample);
  }

  /** Embedding lookup for special tokens (position-independent). */
  void captureEmbeds(const std::vector<unsigned int> &ids,
                     std::vector<float> &embed) {
    std::vector<float> hidden;
    captureActivations(ids, hidden, embed);
  }

  /** Greedy reply generation; returns up to @p max_new generated ids. */
  std::vector<unsigned int>
  generateReply(const std::vector<unsigned int> &prompt_ids, int max_new) {
    allocateAndBindKVCache();
    setKVCachePosition(0);
    const size_t P = prompt_ids.size();
    auto *in_sample =
      static_cast<float *>(malloc(sizeof(float) * MAX_SEQ_LEN));
    std::vector<float *> label;
    auto argmax = [this](float *logits) {
      return static_cast<unsigned int>(
        std::distance(logits, std::max_element(logits, logits + NUM_VOCAB)));
    };
    unsigned int next = 0;
    for (size_t r = 0; r < P; ++r) { // row-by-row prefill
      in_sample[0] = static_cast<float>(prompt_ids[r]);
      std::vector<float *> in = buildInferenceInputs(in_sample);
      std::vector<float *> out = model->incremental_inference(
        1, in, label, static_cast<unsigned int>(P),
        static_cast<unsigned int>(r), static_cast<unsigned int>(r) + 1, false);
      if (r == P - 1)
        next = argmax(out[0]);
      for (auto o : out)
        delete[] o;
    }
    std::vector<unsigned int> reply;
    reply.push_back(next);
    for (int k = 1; k < max_new; ++k) {
      if (std::find(EOS_TOKEN_ID.begin(), EOS_TOKEN_ID.end(), next) !=
          EOS_TOKEN_ID.end())
        break;
      in_sample[0] = static_cast<float>(next);
      std::vector<float *> in = buildInferenceInputs(in_sample);
      std::vector<float *> out = model->incremental_inference(
        1, in, label, static_cast<unsigned int>(P),
        static_cast<unsigned int>(P) + k - 1, static_cast<unsigned int>(P) + k);
      next = argmax(out[0]);
      reply.push_back(next);
      for (auto o : out)
        delete[] o;
    }
    free(in_sample);
    return reply;
  }

private:
  Tensor embedding0_out_, output_norm_out_; /**< captured for multi-output */
};

void Qwen25OmniTalkerCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  const auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::MRoPEApplyLayer>);
  } catch (std::invalid_argument &e) {
  }
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::CaptureTapLayer>);
  } catch (std::invalid_argument &e) {
  }
}

Tensor Qwen25OmniTalkerCausalLM::createAttention(const int layer_id,
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

std::pair<Tensor, Tensor> Qwen25OmniTalkerCausalLM::constructModel() {
  // input0 = host-computed fused inputs_embeds [B, 1, INIT_SEQ_LEN, emb_size].
  Tensor x({BATCH_SIZE, 1, static_cast<unsigned int>(INIT_SEQ_LEN),
            static_cast<unsigned int>(EMBEDDING_SIZE)},
           "input0");

  // thinker_to_talker_proj: emb_size(2048) -> hidden(896), with bias.
  LayerHandle proj(createLayer(
    "fully_connected",
    {withKey("name", "thinker_to_talker_proj"), withKey("unit", DIM),
     withKey("disable_bias", "false"), withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor h = proj(x);

  // shared M-RoPE cos/sin side inputs [B,1,MAX_SEQ_LEN,HEAD_DIM]
  const std::string rope_shape = std::to_string(BATCH_SIZE) + ":1:" +
                                 std::to_string(MAX_SEQ_LEN) + ":" +
                                 std::to_string(HEAD_DIM);
  LayerHandle cos_input(createLayer(
    "input", {withKey("name", "mrope_cos"), withKey("input_shape", rope_shape)}));
  mrope_cos_t = cos_input(Tensor());
  LayerHandle sin_input(createLayer(
    "input", {withKey("name", "mrope_sin"), withKey("input_shape", rope_shape)}));
  mrope_sin_t = sin_input(Tensor());

  for (int i = 0; i < NUM_LAYERS; ++i)
    h = createTransformerDecoderBlock(i, h);

  LayerHandle out_norm(
    createLayer("rms_norm", {withKey("name", "output_norm"),
                             withKey("epsilon", std::to_string(NORM_EPS)),
                             withKey("packed", "false")}));
  h = out_norm(h);

  // codec_head: hidden(896) -> codec vocab(8448), no bias, no tying.
  LayerHandle codec_head(createLayer(
    "fully_connected",
    {withKey("name", "codec_head"), withKey("unit", NUM_VOCAB),
     withKey("disable_bias", "true"), withKey("weight_dtype", LMHEAD_DTYPE)}));
  Tensor y = codec_head(h);

  return {x, y};
}

Qwen25OmniTalkerCausalLM::Qwen25OmniTalkerCausalLM(json &cfg,
                                                   json &generation_cfg,
                                                   json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
  CausalLM(cfg, generation_cfg, nntr_cfg) {
  EMBEDDING_SIZE = cfg.value("embedding_size", 2048);
  CODEC_BOS = cfg.value("tts_codec_start_token_id", 8293);
  CODEC_EOS = cfg.value("tts_codec_end_token_id", 8294);
  CODEC_PAD = cfg.value("tts_codec_pad_token_id", 8292);
  CODEC_MASK = cfg.value("tts_codec_mask_token_id", 8296);
  if (cfg.contains("rope_scaling") &&
      cfg["rope_scaling"].contains("mrope_section"))
    MROPE_SECTION = cfg["rope_scaling"]["mrope_section"].get<std::vector<int>>();
  codec_embed_path = nntr_cfg.value("codec_embed_path", std::string());
  SPEAKER_BOS = nntr_cfg.value("speaker_bos_token", 151872);
  TEXT_EOS = cfg.value("tts_text_end_token_id", 151861);
  TEXT_PAD = cfg.value("tts_text_pad_token_id", 151859);
  THINKER_MAX_NEW = nntr_cfg.value("thinker_max_new_tokens", 16);
  TALKER_MAX_NEW = nntr_cfg.value("talker_max_new_tokens", 128);
  thinker_model_path = nntr_cfg.value("thinker_model_path", std::string());
  thinker_nntr_config =
    nntr_cfg.value("thinker_nntr_config", std::string("nntr_config.json"));
  token2wav_model_path =
    nntr_cfg.value("token2wav_model_path", std::string());
  speech_output = nntr_cfg.value("speech_output", std::string("speech.wav"));
}

Qwen25OmniTalkerCausalLM::~Qwen25OmniTalkerCausalLM() {
  if (talker_in_ != nullptr)
    free(talker_in_);
}

void Qwen25OmniTalkerCausalLM::loadCodecEmbed() {
  if (!codec_embed.empty())
    return;
  if (codec_embed_path.empty())
    throw std::runtime_error(
      "codec_embed_path not set in nntr_config (needed for end-to-end)");
  std::ifstream f(codec_embed_path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Failed to open codec_embed: " + codec_embed_path);
  codec_embed.assign(static_cast<size_t>(NUM_VOCAB) * EMBEDDING_SIZE, 0.0f);
  f.read(reinterpret_cast<char *>(codec_embed.data()),
         codec_embed.size() * sizeof(float));
  if (!f)
    throw std::runtime_error("Truncated codec_embed: " + codec_embed_path);
}

void Qwen25OmniTalkerCausalLM::initialize() {
  CausalLM::initialize();
  mrope_cos.assign(static_cast<size_t>(MAX_SEQ_LEN) * HEAD_DIM, 1.0f);
  mrope_sin.assign(static_cast<size_t>(MAX_SEQ_LEN) * HEAD_DIM, 0.0f);
  talker_in_ = static_cast<float *>(
    malloc(sizeof(float) * static_cast<size_t>(INIT_SEQ_LEN) * EMBEDDING_SIZE));

  // End-to-end (Stage C) needs the Thinker + the codec embed table; Stage A
  // (replaying HF embeds) does not, so both are optional / lazy.
  if (!thinker_model_path.empty()) {
    json tcfg = LoadJsonFile(thinker_model_path + "/config.json");
    json tgen = json::object();
    const std::string tgen_path = thinker_model_path + "/generation_config.json";
    if (std::filesystem::exists(tgen_path))
      tgen = LoadJsonFile(tgen_path);
    json tnntr = LoadJsonFile(thinker_model_path + "/" + thinker_nntr_config);
    thinker = std::make_unique<ThinkerForCapture>(tcfg, tgen, tnntr);
    thinker->initialize();
    thinker->load_weight(thinker_model_path + "/" +
                         tnntr["model_file_name"].get<std::string>());
  }
}

std::vector<float *>
Qwen25OmniTalkerCausalLM::buildInferenceInputs(float *input_sample) {
  // The compiled graph's external-input ORDER differs by cache-placeholder
  // kind: UINT16 builds create explicit input layers inside the decoder
  // blocks -> [input0, mrope_cos, mrope_sin, caches...]; ENABLE_FP16 builds
  // create raw named tensors (createKVCachePlaceholders #ifdef) that the
  // graph realizes AFTER the explicit inputs -> [input0, caches...,
  // mrope_cos, mrope_sin]. Feeding mrope at the wrong slots maps the
  // zero-filled cache slabs onto the M-RoPE tables (cos==0 wipes Q/K —
  // wildly wrong codec ids), so disambiguate against the actual graph:
  // graph input #1 is mrope_cos (width == HEAD_DIM) in the first layout
  // and a cache slab (width == 2 * n_kv * HEAD_DIM) in the second.
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

  bool mrope_first = true;
  const auto in_dims = model->getInputDimension();
  if (in_dims.size() >= 2 &&
      in_dims[1].width() != static_cast<size_t>(HEAD_DIM))
    mrope_first = false;

  std::vector<float *> in;
  in.reserve(3 + caches.size());
  in.push_back(input_sample);
  if (mrope_first) {
    in.push_back(mrope_cos.data());
    in.push_back(mrope_sin.data());
  }
  for (const auto &c : caches)
    in.push_back(c.second);
  if (!mrope_first) {
    in.push_back(mrope_cos.data());
    in.push_back(mrope_sin.data());
  }
  return in;
}

void Qwen25OmniTalkerCausalLM::buildMRoPETables(
  const std::vector<std::array<int, 3>> &pos3d) {
  const unsigned int hd = HEAD_DIM;
  const unsigned int half = hd / 2;
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
  std::vector<int> axis(hd, 0);
  {
    unsigned int d = 0;
    for (size_t c = 0; c < sec2.size(); ++c)
      for (int n = 0; n < sec2[c] && d < hd; ++n, ++d)
        axis[d] = static_cast<int>(c % 3);
  }

  int max_pos = 0;
  for (const auto &pp : pos3d)
    max_pos = std::max({max_pos, pp[0], pp[1], pp[2]});

  for (unsigned int pidx = 0; pidx < MAX_SEQ_LEN; ++pidx) {
    int t, hh, ww;
    if (pidx < pos3d.size()) {
      t = pos3d[pidx][0];
      hh = pos3d[pidx][1];
      ww = pos3d[pidx][2];
    } else {
      const int val = max_pos + 1 + static_cast<int>(pidx - pos3d.size());
      t = hh = ww = val;
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

namespace {
std::vector<float> readF32WithHeader(const std::string &path, int &n, int &d) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Failed to open " + path);
  int32_t nn = 0, dd = 0;
  f.read(reinterpret_cast<char *>(&nn), sizeof(int32_t));
  f.read(reinterpret_cast<char *>(&dd), sizeof(int32_t));
  std::vector<float> v(static_cast<size_t>(nn) * dd);
  f.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(float));
  if (!f)
    throw std::runtime_error("Truncated " + path);
  n = nn;
  d = dd;
  return v;
}

std::vector<unsigned int> readI32WithHeader(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Failed to open " + path);
  int32_t n = 0;
  f.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
  std::vector<int32_t> tmp(n);
  f.read(reinterpret_cast<char *>(tmp.data()),
         static_cast<size_t>(n) * sizeof(int32_t));
  if (!f)
    throw std::runtime_error("Truncated " + path);
  return std::vector<unsigned int>(tmp.begin(), tmp.end());
}

void writeF32WithHeader(const std::string &path, const std::vector<float> &v,
                        int n, int d) {
  std::ofstream o(path, std::ios::binary);
  int32_t nn = n, dd = d;
  o.write(reinterpret_cast<const char *>(&nn), sizeof(int32_t));
  o.write(reinterpret_cast<const char *>(&dd), sizeof(int32_t));
  o.write(reinterpret_cast<const char *>(v.data()),
          v.size() * sizeof(float));
}
} // namespace

unsigned int Qwen25OmniTalkerCausalLM::argmaxSuppressBos(float *logits) const {
  logits[CODEC_BOS] = -std::numeric_limits<float>::infinity();
  return static_cast<unsigned int>(
    std::distance(logits, std::max_element(logits, logits + NUM_VOCAB)));
}

std::vector<unsigned int> Qwen25OmniTalkerCausalLM::talkerDecode(
  const std::vector<float> &prefill_embeds, int L0, int max_steps,
  bool stop_on_eos,
  const std::function<void(int, unsigned int, float *)> &gen_fn) {

  // sequential 3D positions (text-only talker -> 1-D RoPE)
  std::vector<std::array<int, 3>> pos3d;
  pos3d.reserve(INIT_SEQ_LEN);
  for (unsigned int i = 0; i < INIT_SEQ_LEN; ++i)
    pos3d.push_back(
      {static_cast<int>(i), static_cast<int>(i), static_cast<int>(i)});
  buildMRoPETables(pos3d);

  allocateAndBindKVCache();
  setKVCachePosition(0);

  std::vector<float *> label;
  std::vector<unsigned int> codes;
  const size_t emb = static_cast<size_t>(EMBEDDING_SIZE);

  // ---- prefill, processed ONE ROW AT A TIME (incremental) ----
  // A single batched prefill writes a correct KV cache but the last row's
  // output logits diverge from HF; the incremental path reproduces HF's first
  // codec token exactly. See [[omni-talker-batched-prefill-mha-bug]].
  unsigned int c = 0;
  for (int r = 0; r < L0; ++r) {
    std::memcpy(talker_in_, prefill_embeds.data() + static_cast<size_t>(r) * emb,
                emb * sizeof(float));
    std::vector<float *> in = buildInferenceInputs(talker_in_);
    std::vector<float *> out =
      model->incremental_inference(BATCH_SIZE, in, label, L0, r, r + 1, false);
    if (r == L0 - 1)
      c = argmaxSuppressBos(out[0]);
    for (auto o : out)
      delete[] o;
  }
  codes.push_back(c);

  // ---- generation ----
  for (int k = 0; k < max_steps; ++k) {
    gen_fn(k, c, talker_in_);
    std::vector<float *> in = buildInferenceInputs(talker_in_);
    std::vector<float *> out = model->incremental_inference(
      BATCH_SIZE, in, label, L0, L0 + k, L0 + k + 1);
    unsigned int cn = argmaxSuppressBos(out[0]);
    for (auto o : out)
      delete[] o;
    codes.push_back(cn);
    if (stop_on_eos && (cn == static_cast<unsigned int>(CODEC_EOS) ||
                        cn == static_cast<unsigned int>(CODEC_PAD)))
      break;
    c = cn;
  }
  return codes;
}

void Qwen25OmniTalkerCausalLM::runStageA(const std::string &dir,
                                         bool log_output) {
  int L0 = 0, dimp = 0, S = 0, dims = 0;
  std::vector<float> prefill =
    readF32WithHeader(dir + "/prefill.f32", L0, dimp);
  std::vector<float> steps;
  {
    std::ifstream probe(dir + "/steps.f32", std::ios::binary);
    if (probe.good())
      steps = readF32WithHeader(dir + "/steps.f32", S, dims);
  }
  if (dimp != EMBEDDING_SIZE)
    throw std::runtime_error("Stage A: prefill embed dim " +
                             std::to_string(dimp) + " != " +
                             std::to_string(EMBEDDING_SIZE));
  if (static_cast<unsigned int>(L0) > INIT_SEQ_LEN)
    throw std::runtime_error("Stage A: L0 > INIT_SEQ_LEN");
  if (log_output)
    std::cout << "[Stage A] prefill L0=" << L0 << " steps=" << S << std::endl;

  // replay the HF per-step inputs_embeds verbatim (ignore the prev code)
  auto gen_fn = [&](int k, unsigned int, float *out) {
    std::memcpy(out, steps.data() + static_cast<size_t>(k) * EMBEDDING_SIZE,
                static_cast<size_t>(EMBEDDING_SIZE) * sizeof(float));
  };
  std::vector<unsigned int> codes =
    talkerDecode(prefill, L0, S, /*stop_on_eos=*/false, gen_fn);

  std::cout << "CODES:";
  for (auto c : codes)
    std::cout << " " << c;
  std::cout << std::endl;
}

void Qwen25OmniTalkerCausalLM::runEndToEnd(const std::string &prompt,
                                           const std::string &dir,
                                           bool use_hf_ids, bool log_output) {
  if (!thinker)
    throw std::runtime_error(
      "end-to-end needs nntr_config[\"thinker_model_path\"]");
  loadCodecEmbed();
  const int EMB = EMBEDDING_SIZE;

  // ---- 1. prompt + reply token ids ----
  std::vector<unsigned int> prompt_ids, reply_ids, hf_codes;
  if (use_hf_ids) {
    prompt_ids = readI32WithHeader(dir + "/prompt_ids.i32");
    reply_ids = readI32WithHeader(dir + "/reply_ids.i32");
    std::ifstream probe(dir + "/codes.i32", std::ios::binary);
    if (probe.good())
      hf_codes = readI32WithHeader(dir + "/codes.i32");
  } else {
    const std::string full =
      "<|im_start|>system\nYou are Qwen, a virtual human developed by the Qwen "
      "Team, Alibaba Group, capable of perceiving auditory and visual inputs, "
      "as well as generating text and speech.<|im_end|>\n<|im_start|>user\n" +
      prompt + "<|im_end|>\n<|im_start|>assistant\n";
    // the tokenizer parses on a side thread since [round-13]; join before use
    ensureTokenizer();
    if (!tokenizer)
      throw std::runtime_error("end-to-end needs a tokenizer_file");
    auto enc = tokenizer->Encode(full);
    prompt_ids.assign(enc.begin(), enc.end());
    reply_ids = thinker->generateReply(prompt_ids, THINKER_MAX_NEW);
  }
  const int P = static_cast<int>(prompt_ids.size());
  const int G = static_cast<int>(reply_ids.size());
  const int L = P + G;
  const int L0 = P + 2;
  if (static_cast<unsigned int>(L0) > INIT_SEQ_LEN)
    throw std::runtime_error("end-to-end: L0 > INIT_SEQ_LEN");

  std::cout << "REPLY_IDS:";
  for (auto r : reply_ids)
    std::cout << " " << r;
  std::cout << std::endl;
  if (log_output)
    std::cout << "[Talker] prompt=" << P << " reply=" << G << " L0=" << L0
              << std::endl;

  // ---- 2. capture thinker per-token hidden + token-embed over [prompt+reply]
  std::vector<unsigned int> full(prompt_ids);
  full.insert(full.end(), reply_ids.begin(), reply_ids.end());
  std::vector<float> hidden, embed; // [L * EMB]
  thinker->captureActivations(full, hidden, embed);
  std::vector<float> spec; // [3 * EMB]: speaker_bos, text_eos, text_pad embeds
  thinker->captureEmbeds({static_cast<unsigned int>(SPEAKER_BOS),
                          static_cast<unsigned int>(TEXT_EOS),
                          static_cast<unsigned int>(TEXT_PAD)},
                         spec);
  const float *emb_spk = &spec[0];
  const float *emb_eos = &spec[static_cast<size_t>(EMB)];
  const float *emb_pad = &spec[static_cast<size_t>(2) * EMB];

  // ---- 3. assemble talker prefill inputs_embeds [L0, EMB] (HF lines 2322-2345)
  std::vector<float> prefill(static_cast<size_t>(L0) * EMB);
  for (int r = 0; r < P; ++r)
    for (int d = 0; d < EMB; ++d)
      prefill[static_cast<size_t>(r) * EMB + d] =
        hidden[static_cast<size_t>(r) * EMB + d] +
        embed[static_cast<size_t>(r) * EMB + d];
  for (int d = 0; d < EMB; ++d) {
    // row P: thinker_embed(speaker_bos) + codec_embed(codec_pad)
    prefill[static_cast<size_t>(P) * EMB + d] =
      emb_spk[d] + codecEmbed(CODEC_PAD)[d];
    // row P+1: (hidden[P] + embed[P]) + codec_embed(codec_bos)
    prefill[static_cast<size_t>(P + 1) * EMB + d] =
      hidden[static_cast<size_t>(P) * EMB + d] +
      embed[static_cast<size_t>(P) * EMB + d] + codecEmbed(CODEC_BOS)[d];
  }

  // ---- 4. assemble the thinker_reply_part stream [G+1, EMB] consumed 1/step
  // rows 0..G-2 = hidden+embed at positions P+1..L-1; then text_eos, text_pad.
  const int Sstream = G + 1;
  std::vector<float> stream(static_cast<size_t>(Sstream) * EMB);
  for (int j = 0; j <= G - 2; ++j)
    for (int d = 0; d < EMB; ++d)
      stream[static_cast<size_t>(j) * EMB + d] =
        hidden[static_cast<size_t>(P + 1 + j) * EMB + d] +
        embed[static_cast<size_t>(P + 1 + j) * EMB + d];
  for (int d = 0; d < EMB; ++d) {
    stream[static_cast<size_t>(G - 1) * EMB + d] = emb_eos[d];
    stream[static_cast<size_t>(G) * EMB + d] = emb_pad[d];
  }

  // ---- Stage B dumps: assembled prefill/steps vs HF (when ids came from a dump)
  if (!dir.empty()) {
    writeF32WithHeader(dir + "/assembled_prefill.f32", prefill, L0, EMB);
    if (!hf_codes.empty()) {
      std::vector<float> asteps(hf_codes.size() * EMB);
      for (size_t k = 0; k < hf_codes.size(); ++k) {
        const float *ce = codecEmbed(hf_codes[k]);
        const int j = std::min(static_cast<int>(k), Sstream - 1);
        const float *sr = &stream[static_cast<size_t>(j) * EMB];
        for (int d = 0; d < EMB; ++d)
          asteps[k * EMB + d] = ce[d] + sr[d];
      }
      writeF32WithHeader(dir + "/assembled_steps.f32", asteps,
                         static_cast<int>(hf_codes.size()), EMB);
    }
  }

  // ---- 5. drive the talker (feedback): step k = codec_embed(prev) + stream[k]
  auto gen_fn = [&](int k, unsigned int prev, float *out) {
    const float *ce = codecEmbed(prev);
    const int j = std::min(k, Sstream - 1);
    const float *sr = &stream[static_cast<size_t>(j) * EMB];
    for (int d = 0; d < EMB; ++d)
      out[d] = ce[d] + sr[d];
  };
  std::vector<unsigned int> codes =
    talkerDecode(prefill, L0, TALKER_MAX_NEW - 1, /*stop_on_eos=*/true, gen_fn);

  std::cout << "CODES:";
  for (auto c : codes)
    std::cout << " " << c;
  std::cout << std::endl;

  if (!token2wav_model_path.empty())
    speakCodes(codes, log_output);
}

void Qwen25OmniTalkerCausalLM::speakCodes(
  const std::vector<unsigned int> &codes, bool log_output) {
  // strip the trailing eos/pad the decode loop pushes before stopping
  std::vector<int32_t> ids;
  ids.reserve(codes.size());
  for (auto c : codes) {
    if (static_cast<int>(c) == CODEC_EOS || static_cast<int>(c) == CODEC_PAD)
      break;
    ids.push_back(static_cast<int32_t>(c));
  }
  if (ids.empty()) {
    std::cout << "[Token2Wav] no speech codes to synthesize" << std::endl;
    return;
  }

  if (!t2w) {
    t2w_cfg = LoadJsonFile(token2wav_model_path + "/config.json");
    t2w_gen = json::object();
    t2w_nntr = LoadJsonFile(token2wav_model_path + "/nntr_config.json");
    t2w = std::make_unique<Qwen25OmniToken2Wav>(t2w_cfg, t2w_gen, t2w_nntr);
    t2w->initialize();
    t2w->load_weight(token2wav_model_path + "/" +
                     t2w_nntr.value("model_file_name", std::string("dit.bin")));
  }

  std::vector<float> wav = t2w->speak(ids);
  Qwen25OmniBigVGAN::write_wav(speech_output, wav, 24000);
  if (log_output)
    std::cout << "[Token2Wav] wrote " << wav.size() << " samples ("
              << ids.size() << " codes) to " << speech_output << std::endl;
}

void Qwen25OmniTalkerCausalLM::run(const WSTR prompt, bool do_sample,
                                   const WSTR system_prompt,
                                   const WSTR tail_prompt, bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;
  if (!is_initialized)
    throw std::runtime_error("Talker model is not initialized.");

  static const std::string kStageA = "stageA:";
  static const std::string kStageBC = "stageBC:";
  if (prompt.rfind(kStageA, 0) == 0) {
    runStageA(prompt.substr(kStageA.size()), log_output);
  } else if (prompt.rfind(kStageBC, 0) == 0) {
    runEndToEnd("", prompt.substr(kStageBC.size()), /*use_hf_ids=*/true,
                log_output);
  } else {
    runEndToEnd(prompt, "", /*use_hf_ids=*/false, log_output);
  }
  has_run_ = true;
}

} // namespace causallm

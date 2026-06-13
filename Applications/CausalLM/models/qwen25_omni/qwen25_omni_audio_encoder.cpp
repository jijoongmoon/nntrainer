// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_audio_encoder.cpp
 * @date   12 June 2026
 * @brief  Qwen2.5-Omni Thinker audio encoder (Whisper-style, windowed).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <llm_util.hpp>
#include <mha_core.h>
#include <model.h>
#include <qwen25_omni_audio_encoder.h>
#include <whisper_mel.h>

namespace causallm {

void Qwen25OmniAudioEncoder::setupParameters(json &cfg, json &generation_cfg,
                                             json &nntr_cfg) {
  (void)generation_cfg;

  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");
  MEMORY_SWAP = nntr_cfg.value("fsu", false);
  head_weight_file = nntr_cfg.value(
    "audio_head_file_name", std::string("nntr_qwen2.5_omni_audio_head.bin"));

  // The converter emits a flat audio config (HF thinker_config.audio_config
  // fields); fall back to the nested form so the original HF config works.
  json audio = cfg;
  if (cfg.contains("thinker_config") &&
      cfg["thinker_config"].contains("audio_config"))
    audio = cfg["thinker_config"]["audio_config"];
  else if (cfg.contains("audio_config"))
    audio = cfg["audio_config"];

  DIM = audio.value("d_model", 1280);
  NUM_LAYERS = audio.value("encoder_layers", 32);
  NUM_HEADS = audio.value("encoder_attention_heads", 20);
  NUM_KEY_VALUE_HEADS = NUM_HEADS;
  GQA_SIZE = 1;
  HEAD_DIM = DIM / NUM_HEADS;
  INTERMEDIATE_SIZE = audio.value("encoder_ffn_dim", 5120);
  NUM_MEL = audio.value("num_mel_bins", 128);
  OUTPUT_DIM = audio.value("output_dim", 2048);
  CHUNK_MEL = 2 * audio.value("n_window", 100);
  CHUNK_FRAMES = CHUNK_MEL / 2;
  POOLED_FRAMES = CHUNK_FRAMES / 2;
  NORM_EPS = 1e-5; // nn.LayerNorm default used by the HF audio tower
  ROPE_THETA = 0;  // sinusoidal positions, no RoPE
  IS_CAUSAL = false;

  INIT_SEQ_LEN = CHUNK_FRAMES;
  MAX_SEQ_LEN = CHUNK_FRAMES;
  NUM_TO_GENERATE = 0;
}

Tensor Qwen25OmniAudioEncoder::createEncoderBlock(const int layer_id,
                                                  Tensor input) {
  const std::string prefix = "layer" + std::to_string(layer_id) + "_";

  // --- self attention ---
  LayerHandle attn_norm(createLayer(
    "layer_normalization",
    {withKey("name", prefix + "attention_norm"), withKey("axis", "3"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor normed = attn_norm(input);

  LayerHandle q_proj(createLayer(
    "fully_connected",
    {withKey("name", prefix + "wq"), withKey("unit", std::to_string(DIM)),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  // Whisper-style attention: k_proj carries NO bias
  LayerHandle k_proj(createLayer(
    "fully_connected",
    {withKey("name", prefix + "wk"), withKey("unit", std::to_string(DIM)),
     withKey("disable_bias", "true"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  LayerHandle v_proj(createLayer(
    "fully_connected",
    {withKey("name", prefix + "wv"), withKey("unit", std::to_string(DIM)),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor q = q_proj(normed);
  Tensor k = k_proj(normed);
  Tensor v = v_proj(normed);

  LayerHandle attention(createLayer(
    "mha_core",
    {withKey("name", prefix + "attention"),
     withKey("num_heads", std::to_string(NUM_HEADS)),
     withKey("num_heads_kv", std::to_string(NUM_HEADS)),
     withKey("max_timestep", std::to_string(CHUNK_FRAMES + 1)),
     withKey("is_causal", "false"), withKey("rope_theta", "0")}));
  Tensor a = attention({q, k, v});

  LayerHandle out_proj(createLayer(
    "fully_connected", {withKey("name", prefix + "attention_out"),
                        withKey("unit", std::to_string(DIM)),
                        withKey("disable_bias", "false"),
                        withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor att_out = out_proj(a);

  LayerHandle attn_res(
    createLayer("addition", {withKey("name", prefix + "attention_residual")}));
  Tensor residual = attn_res({input, att_out});

  // --- feed forward ---
  LayerHandle ffn_norm(createLayer(
    "layer_normalization",
    {withKey("name", prefix + "ffn_norm"), withKey("axis", "3"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor h = ffn_norm(residual);

  LayerHandle fc1(createLayer(
    "fully_connected", {withKey("name", prefix + "ffn_up"),
                        withKey("unit", std::to_string(INTERMEDIATE_SIZE)),
                        withKey("disable_bias", "false"),
                        withKey("weight_dtype", FC_LAYER_DTYPE)}));
  h = fc1(h);

  LayerHandle gelu(createLayer("activation",
                               {withKey("name", prefix + "ffn_gelu"),
                                withKey("activation", "gelu")}));
  h = gelu(h);

  LayerHandle fc2(createLayer(
    "fully_connected",
    {withKey("name", prefix + "ffn_down"), withKey("unit", std::to_string(DIM)),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  h = fc2(h);

  LayerHandle ffn_res(
    createLayer("addition", {withKey("name", prefix + "ffn_residual")}));
  return ffn_res({residual, h});
}

std::pair<Tensor, Tensor> Qwen25OmniAudioEncoder::constructModel() {

  // mel chunk, channel-major: [B, n_mels, 1, 200]
  Tensor x({BATCH_SIZE, NUM_MEL, 1, CHUNK_MEL}, "input0");

  LayerHandle conv1(createLayer(
    "conv1d", {withKey("name", "conv1"), withKey("filters", std::to_string(DIM)),
               withKey("kernel_size", "3"), withKey("stride", "1"),
               withKey("padding", "1")}));
  Tensor h = conv1(x);

  LayerHandle gelu1(createLayer(
    "activation",
    {withKey("name", "conv1_gelu"), withKey("activation", "gelu")}));
  h = gelu1(h);

  LayerHandle conv2(createLayer(
    "conv1d", {withKey("name", "conv2"), withKey("filters", std::to_string(DIM)),
               withKey("kernel_size", "3"), withKey("stride", "2"),
               withKey("padding", "1")}));
  h = conv2(h);

  LayerHandle gelu2(createLayer(
    "activation",
    {withKey("name", "conv2_gelu"), withKey("activation", "gelu")}));
  h = gelu2(h);

  // [B, DIM, 1, 100] -> [B, 1, 100, DIM]
  LayerHandle flatten(createLayer(
    "reshape",
    {withKey("name", "conv_flatten"),
     withKey("target_shape", "1:" + std::to_string(DIM) + ":" +
                               std::to_string(CHUNK_FRAMES))}));
  h = flatten(h);
  LayerHandle transpose(createLayer(
    "permute",
    {withKey("name", "conv_transpose"), withKey("direction", {1, 3, 2})}));
  h = transpose(h);

  // Sinusoidal positions restart at 0 for every window -> one fixed table,
  // baked into the .bin by the converter (it is absent from the checkpoint).
  LayerHandle pos_embed(createLayer(
    "weight",
    {withKey("name", "pos_embed/weights"),
     withKey("dim", "1:1:" + std::to_string(CHUNK_FRAMES) + ":" +
                      std::to_string(DIM)),
     withKey("tensor_dtype", "FP32"), withKey("weight_name", "pos_embed")}));
  Tensor pos = pos_embed(x);
  LayerHandle pos_add(createLayer("addition", {withKey("name", "pos_add")}));
  h = pos_add({h, pos});

  for (int i = 0; i < NUM_LAYERS; ++i)
    h = createEncoderBlock(i, h);

  return {x, h}; // [B, 1, CHUNK_FRAMES, DIM]
}

void Qwen25OmniAudioEncoder::initialize() {
  Transformer::initialize(); // compiles the per-chunk encoder graph

  // Head graph: ln_post + proj over host-pooled frames. Uniform row count, so
  // partial chunks slice cleanly via incremental_inference's [0, to) rows.
  head_model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  head_model->setProperty({withKey("batch_size", BATCH_SIZE),
                           withKey("epochs", "1"),
                           withKey("model_tensor_type", MODEL_TENSOR_TYPE)});

  Tensor hin({BATCH_SIZE, 1, POOLED_FRAMES, static_cast<unsigned int>(DIM)},
             "audio_head_input0");
  LayerHandle ln_post(createLayer(
    "layer_normalization",
    {withKey("name", "ln_post"), withKey("axis", "3"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor hh = ln_post(hin);
  LayerHandle proj(createLayer(
    "fully_connected",
    {withKey("name", "audio_proj"), withKey("unit", std::to_string(OUTPUT_DIM)),
     withKey("disable_bias", "false")}));
  hh = proj(hh);

  if (head_model->compile(hin, hh, ml::train::ExecutionMode::INFERENCE))
    throw std::invalid_argument("Audio head model compilation failed.");
}

void Qwen25OmniAudioEncoder::load_weight(const std::string &weight_path) {
  Transformer::load_weight(weight_path);

  std::filesystem::path head_path =
    std::filesystem::path(weight_path).parent_path() / head_weight_file;
  try {
    head_model->load(head_path.string(),
                     ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load audio head weights from " +
                             head_path.string() + ": " + e.what());
  }
}

void Qwen25OmniAudioEncoder::resetAttentionCache() {
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
      if (l.getType() == causallm::MHACoreLayer::type)
        l.setProperty({"cache_index=0"});
    };
  model->forEachLayer(fn, nullptr);
}

std::vector<float> Qwen25OmniAudioEncoder::encode(const float *mel,
                                                  unsigned int n_frames) {
  if (!is_initialized)
    throw std::runtime_error("Audio encoder is not initialized.");
  if (n_frames < 2 || n_frames % 2 != 0)
    throw std::invalid_argument(
      "mel frame count must be even and >= 2 (pad the waveform); got " +
      std::to_string(n_frames));

  std::vector<float> chunk_in(static_cast<size_t>(NUM_MEL) * CHUNK_MEL);
  std::vector<float> head_in(static_cast<size_t>(POOLED_FRAMES) * DIM);
  std::vector<float> result;

  for (unsigned int c0 = 0; c0 < n_frames; c0 += CHUNK_MEL) {
    const unsigned int L = std::min(CHUNK_MEL, n_frames - c0);
    const unsigned int L1 = (L - 1) / 2 + 1; // post-conv valid frames
    if (L1 < 2)
      break; // HF token formula yields 0 tokens for a <=2-frame tail
    const unsigned int tokens = (L1 - 2) / 2 + 1;

    std::fill(chunk_in.begin(), chunk_in.end(), 0.0f);
    for (unsigned int m = 0; m < NUM_MEL; ++m)
      std::memcpy(&chunk_in[static_cast<size_t>(m) * CHUNK_MEL],
                  &mel[static_cast<size_t>(m) * n_frames + c0],
                  L * sizeof(float));

    // Attention within a window only; restrict rows to the valid frames.
    resetAttentionCache();
    std::vector<float *> in = {chunk_in.data()};
    std::vector<float *> label;
    std::vector<float *> out =
      model->incremental_inference(BATCH_SIZE, in, label, L1, 0, L1, false);

    // AvgPool1d(kernel=2, stride=2): floor — an odd trailing frame is dropped
    for (unsigned int t = 0; t < tokens; ++t)
      for (unsigned int d = 0; d < DIM; ++d)
        head_in[static_cast<size_t>(t) * DIM + d] =
          0.5f * (out[0][(2 * static_cast<size_t>(t)) * DIM + d] +
                  out[0][(2 * static_cast<size_t>(t) + 1) * DIM + d]);

    std::vector<float *> hin = {head_in.data()};
    std::vector<float *> hout =
      head_model->incremental_inference(BATCH_SIZE, hin, label, tokens, 0,
                                        tokens, false);

    result.insert(result.end(), hout[0],
                  hout[0] + static_cast<size_t>(tokens) * OUTPUT_DIM);
  }
  return result;
}

void Qwen25OmniAudioEncoder::run(const WSTR prompt, bool do_sample,
                                 const WSTR system_prompt,
                                 const WSTR tail_prompt, bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;

  const std::string mel_path(prompt);
  std::vector<float> mel;
  unsigned int frames = 0;

  if (mel_path.size() > 4 &&
      mel_path.compare(mel_path.size() - 4, 4, ".wav") == 0) {
    std::vector<float> samples = whisper_mel::loadWav16kMono(mel_path);
    mel = whisper_mel::melSpectrogram(samples, frames);
    // drop the computed features next to the wav for inspection/debugging
    std::ofstream mf(mel_path + ".mel", std::ios::binary);
    const int32_t hdr[2] = {static_cast<int32_t>(NUM_MEL),
                            static_cast<int32_t>(frames)};
    mf.write(reinterpret_cast<const char *>(hdr), sizeof(hdr));
    mf.write(reinterpret_cast<const char *>(mel.data()),
             mel.size() * sizeof(float));
  } else {
    std::ifstream f(mel_path, std::ios::binary);
    if (!f.is_open())
      throw std::runtime_error("Failed to open mel feature file: " + mel_path);

    int32_t n_mels = 0, n_frames = 0;
    f.read(reinterpret_cast<char *>(&n_mels), sizeof(int32_t));
    f.read(reinterpret_cast<char *>(&n_frames), sizeof(int32_t));
    if (n_mels != static_cast<int32_t>(NUM_MEL) || n_frames <= 0)
      throw std::runtime_error("Invalid mel file header: n_mels=" +
                               std::to_string(n_mels) +
                               " n_frames=" + std::to_string(n_frames));

    mel.resize(static_cast<size_t>(n_mels) * n_frames);
    f.read(reinterpret_cast<char *>(mel.data()), mel.size() * sizeof(float));
    if (!f)
      throw std::runtime_error("Truncated mel feature file: " + mel_path);
    frames = static_cast<unsigned int>(n_frames);
  }

  std::vector<float> embd = encode(mel.data(), frames);
  const int32_t n_tokens = static_cast<int32_t>(embd.size() / OUTPUT_DIM);

  const std::string out_path = mel_path + ".embd";
  std::ofstream of(out_path, std::ios::binary);
  const int32_t out_dim = static_cast<int32_t>(OUTPUT_DIM);
  of.write(reinterpret_cast<const char *>(&n_tokens), sizeof(int32_t));
  of.write(reinterpret_cast<const char *>(&out_dim), sizeof(int32_t));
  of.write(reinterpret_cast<const char *>(embd.data()),
           embd.size() * sizeof(float));
  of.close();

  if (log_output) {
    std::cout << "audio tokens: " << n_tokens << " (dim " << OUTPUT_DIM
              << ") -> " << out_path << "\nfirst values:";
    std::cout << std::setprecision(7);
    for (int i = 0; i < 8 && i < static_cast<int>(embd.size()); ++i)
      std::cout << " " << embd[i];
    std::cout << std::endl;
  }
  has_run_ = true;
}

} // namespace causallm

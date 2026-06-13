// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_audio_causallm.cpp
 * @date   12 June 2026
 * @brief  Qwen2.5-Omni Thinker with audio input (speech in / text out).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <app_context.h>
#include <embedding_injection.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>
#include <qwen25_omni_audio_causallm.h>
#include <whisper_mel.h>

namespace causallm {

void Qwen25OmniAudioCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();

  const auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingInjectionLayer>);
  } catch (std::invalid_argument &e) {
    // already registered; ignore
  }
}

std::pair<Tensor, Tensor> Qwen25OmniAudioCausalLM::constructModel() {

  // Mirrors Transformer::constructModel + CausalLM's lm_head, with the
  // embedding_injection step inserted after embedding0. The injection layer
  // has no weights and its inputs are visited after the embedding subtree,
  // so the on-disk weight order is identical to the text-only model — the
  // same .bin loads unchanged.
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

  // Host-filled side input holding one DIM-wide row per prompt position.
  const std::string audio_shape = std::to_string(BATCH_SIZE) + ":1:" +
                                  std::to_string(INIT_SEQ_LEN) + ":" +
                                  std::to_string(DIM);
  LayerHandle audio_input(
    createLayer("input", {withKey("name", "aud_embd"),
                          withKey("input_shape", audio_shape)}));
  Tensor audio = audio_input(Tensor());

  LayerHandle inject(createLayer(
    "embedding_injection",
    {withKey("name", "embd_inject"),
     withKey("token_id", std::to_string(AUDIO_TOKEN_ID))}));
  h = inject({h, x, audio});

  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  LayerHandle out_norm(
    createLayer("rms_norm", {withKey("name", "output_norm"),
                             withKey("epsilon", std::to_string(NORM_EPS)),
                             withKey("packed", "false")}));
  h = out_norm(h);

  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";
  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };
  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));
  LayerHandle lmhead(createLayer(lmhead_type, lmhead_prop));
  Tensor y = lmhead(h);

  return {x, y};
}

void Qwen25OmniAudioCausalLM::initialize() {
  CausalLM::initialize();

  audio_buf.assign(static_cast<size_t>(INIT_SEQ_LEN) * DIM, 0.0f);

  if (audio_encoder_path.empty()) {
    std::cerr << "[Warning] nntr_config has no audio_encoder_path; only text "
                 "prompts will work."
              << std::endl;
    return;
  }

  audio_cfg = LoadJsonFile(audio_encoder_path + "/config.json");
  audio_generation_cfg = json::object();
  audio_nntr_cfg = LoadJsonFile(audio_encoder_path + "/nntr_config.json");

  audio_encoder = std::make_unique<Qwen25OmniAudioEncoder>(
    audio_cfg, audio_generation_cfg, audio_nntr_cfg);
  audio_encoder->initialize();
  audio_encoder->load_weight(
    audio_encoder_path + "/" +
    audio_nntr_cfg["model_file_name"].get<std::string>());
}

void Qwen25OmniAudioCausalLM::appendExtraInputs(
  std::vector<std::pair<std::string, float *>> &inputs) {
  inputs.emplace_back("aud_embd", audio_buf.data());
}

void Qwen25OmniAudioCausalLM::run(const WSTR prompt, bool do_sample,
                                  const WSTR system_prompt,
                                  const WSTR tail_prompt, bool log_output) {

  static const std::string kAudioPrefix = "audio:";
  if (prompt.rfind(kAudioPrefix, 0) != 0) {
    std::fill(audio_buf.begin(), audio_buf.end(), 0.0f);
    CausalLM::run(prompt, do_sample, system_prompt, tail_prompt, log_output);
    return;
  }

  if (!audio_encoder)
    throw std::runtime_error(
      "audio prompt given but audio_encoder_path is not configured");

  // "audio:<mel_file> <question>"
  const std::string rest = prompt.substr(kAudioPrefix.size());
  const auto space = rest.find(' ');
  if (space == std::string::npos)
    throw std::invalid_argument(
      "audio prompt syntax: audio:<mel_file> <question>");
  const std::string mel_path = rest.substr(0, space);
  const std::string question = rest.substr(space + 1);

  // --- encode the audio (raw wav or precomputed mel features) ---
  std::vector<float> mel;
  unsigned int n_frames = 0;
  if (mel_path.size() > 4 &&
      mel_path.compare(mel_path.size() - 4, 4, ".wav") == 0) {
    std::vector<float> samples = whisper_mel::loadWav16kMono(mel_path);
    mel = whisper_mel::melSpectrogram(samples, n_frames);
  } else {
    std::ifstream f(mel_path, std::ios::binary);
    if (!f.is_open())
      throw std::runtime_error("Failed to open mel feature file: " + mel_path);
    int32_t n_mels = 0, frames = 0;
    f.read(reinterpret_cast<char *>(&n_mels), sizeof(int32_t));
    f.read(reinterpret_cast<char *>(&frames), sizeof(int32_t));
    mel.resize(static_cast<size_t>(n_mels) * frames);
    f.read(reinterpret_cast<char *>(mel.data()), mel.size() * sizeof(float));
    if (!f || frames <= 0)
      throw std::runtime_error("Invalid mel feature file: " + mel_path);
    n_frames = static_cast<unsigned int>(frames);
  }

  std::vector<float> embd = audio_encoder->encode(mel.data(), n_frames);
  const size_t n_audio_tokens = embd.size() / DIM;
  if (log_output)
    std::cout << "[Audio] " << mel_path << " -> " << n_audio_tokens
              << " audio tokens" << std::endl;

  // --- build the chat prompt with expanded placeholders ---
  std::string audio_seg = "<|audio_bos|>";
  for (size_t i = 0; i < n_audio_tokens; ++i)
    audio_seg += "<|AUDIO|>";
  audio_seg += "<|audio_eos|>";

  const std::string full =
    "<|im_start|>system\nYou are Qwen, a virtual human developed by the Qwen "
    "Team, Alibaba Group, capable of perceiving auditory and visual inputs, "
    "as well as generating text and speech.<|im_end|>\n<|im_start|>user\n" +
    audio_seg + question + "<|im_end|>\n<|im_start|>assistant\n";

  // --- scatter encoder outputs to the placeholder positions ---
  std::fill(audio_buf.begin(), audio_buf.end(), 0.0f);
  auto ids = tokenizer->Encode(full);
  size_t scattered = 0;
  for (size_t i = 0; i < ids.size() && i < INIT_SEQ_LEN; ++i) {
    if (static_cast<int>(ids[i]) == AUDIO_TOKEN_ID) {
      if (scattered >= n_audio_tokens)
        throw std::runtime_error("more <|AUDIO|> tokens than audio features");
      std::memcpy(&audio_buf[i * static_cast<size_t>(DIM)],
                  &embd[scattered * static_cast<size_t>(DIM)],
                  DIM * sizeof(float));
      ++scattered;
    }
  }
  if (scattered != n_audio_tokens)
    throw std::runtime_error(
      "audio feature/token count mismatch: scattered " +
      std::to_string(scattered) + " of " + std::to_string(n_audio_tokens) +
      " (prompt truncated? INIT_SEQ_LEN=" + std::to_string(INIT_SEQ_LEN) +
      ")");

  // Base run re-tokenizes the same string, so ids land at the same positions.
  CausalLM::run(full, do_sample, "", "", log_output);
}

} // namespace causallm

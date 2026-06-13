// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_audio_causallm.h
 * @date   12 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Qwen2.5-Omni Thinker with audio input (speech in / text out).
 *
 * Extends the text-only Qwen25OmniCausalLM with the HF masked_scatter step:
 * an embedding_injection layer after embedding0 replaces every <|AUDIO|>
 * (id 151646) placeholder row with the matching row of a host-filled side
 * input ("aud_embd"), which run() fills from the standalone
 * Qwen25OmniAudioEncoder. For text+audio inputs the HF thinker uses plain
 * sequential positions (no M-RoPE divergence), so the decoder is otherwise
 * unchanged and loads the same .bin as the text model.
 *
 * run() prompt syntax:
 *   "audio:<mel_file> <question>"  — mel file as written by the audio
 *                                    encoder tooling; the chat template with
 *                                    expanded audio tokens is built here
 *   anything else                  — plain text chat (passthrough)
 *
 * nntr_config.json additions:
 *   "audio_encoder_path": directory of the converted audio encoder model
 */

#ifndef __QWEN25_OMNI_AUDIO_CAUSAL_LM_H__
#define __QWEN25_OMNI_AUDIO_CAUSAL_LM_H__

#include <qwen25_omni_audio_encoder.h>
#include <qwen25_omni_causallm.h>

namespace causallm {

/**
 * @brief Qwen25OmniAudioCausalLM class (text + audio in, text out)
 */
class Qwen25OmniAudioCausalLM : public Qwen25OmniCausalLM {

public:
  static constexpr const char *architectures = "Qwen25OmniAudioChat";

  Qwen25OmniAudioCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(flattenThinkerTextConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    Qwen25OmniCausalLM(cfg, generation_cfg, nntr_cfg) {
    AUDIO_TOKEN_ID = cfg.value("audio_token_index", 151646);
    audio_encoder_path = nntr_cfg.value("audio_encoder_path", std::string());
  }

  virtual ~Qwen25OmniAudioCausalLM() = default;

  /**
   * @brief Compile the decoder graph and bring up the audio encoder.
   */
  void initialize() override;

  /**
   * @brief Chat with optional audio input (see prompt syntax above).
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

protected:
  /**
   * @brief Decoder graph with the embedding_injection side input.
   */
  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief Base layers + embedding_injection.
   */
  void registerCustomLayers() override;

  /**
   * @brief Contribute the "aud_embd" side input buffer.
   */
  void appendExtraInputs(
    std::vector<std::pair<std::string, float *>> &inputs) override;

private:
  std::unique_ptr<Qwen25OmniAudioEncoder> audio_encoder;
  std::string audio_encoder_path;
  json audio_cfg, audio_generation_cfg, audio_nntr_cfg;

  std::vector<float> audio_buf; /**< [INIT_SEQ_LEN, DIM] side input */
  int AUDIO_TOKEN_ID = 151646;  /**< <|AUDIO|> placeholder id */
};

} // namespace causallm

#endif /* __QWEN25_OMNI_AUDIO_CAUSAL_LM_H__ */

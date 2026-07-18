// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_bigvgan.h
 * @date   15 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Qwen2.5-Omni Token2Wav BigVGAN vocoder (mel -> 24 kHz waveform).
 *
 * Deterministic conv vocoder:
 *   process_mel (HOST: exp -> amplitude-to-dB -> normalize -> clamp)
 *   -> conv_pre Conv1d(80->1536, k7, p3, +bias)
 *   -> 6 upsample stages i: conv1d_transpose(ch->ch/2, kernel UP_K[i],
 *        stride UP_R[i], pad (k-s)/2, +bias) then the MEAN of 3 AMPBlocks
 *        (addition of 3 -> scale 1/3)
 *   -> activation_post (antialiased_snake) -> conv_post Conv1d(24->1, k7, p3,
 *      NO bias) -> clamp[-1,1] (HOST)
 * 240x upsample -> 24 kHz. Validated against HF (Stage C: mel -> wav 4.4e-6).
 *
 * The graph is built imperatively in initialize() (height pinned to 1,
 * channel-major [B,C,1,T]); the input mel length is fixed at compile time
 * (mel_frames, default 128). run(prompt) reads a mel feature file and writes a
 * 24 kHz mono WAV; vocode() is the in-memory entry point for the end-to-end
 * Token2Wav pipeline.
 */

#ifndef __QWEN25_OMNI_BIGVGAN_H__
#define __QWEN25_OMNI_BIGVGAN_H__

#include <string>
#include <vector>

#include <transformer.h>

namespace causallm {

/**
 * @brief Qwen25OmniBigVGAN class (standalone Token2Wav vocoder)
 */
class Qwen25OmniBigVGAN : virtual public Transformer {
public:
  static constexpr const char *architectures = "Qwen25OmniBigVGAN";

  Qwen25OmniBigVGAN(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Qwen25OmniBigVGAN() = default;

  /** @brief Build the conv graph imperatively and compile it. */
  void initialize() override;

  /** @brief Load bigvgan.bin (DFS-from-output order). */
  void load_weight(const std::string &weight_path) override;

  /**
   * @brief Recompile the conv graph for a new mel length and reload weights.
   *        No-op if mel_frames already matches (weights are length-agnostic).
   */
  void ensure_frames(unsigned int n_frames);

  /**
   * @brief Vocode a mel file into a 24 kHz WAV.
   * @param prompt path to a mel feature file
   *   ([int32 mel_dim][int32 n_frames][fp32 mel[mel_dim][n_frames]]);
   *   n_frames must equal the compiled mel_frames. Writes "<path>.wav".
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

  /**
   * @brief In-memory vocode (end-to-end entry point).
   * @param mel mel_dim x n_frames, mel-bin major (natural-log domain);
   *   recompiles the graph if n_frames differs from the compiled length.
   * @return waveform samples in [-1, 1], length n_frames * 240
   */
  std::vector<float> vocode(const float *mel, unsigned int n_frames);

  /** @brief Write a 24 kHz mono 16-bit PCM WAV. */
  static void write_wav(const std::string &path, const std::vector<float> &wav,
                        unsigned int sample_rate);

protected:
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override;

private:
  /** @brief HF process_mel_spectrogram (host, per-element). in/out [mel*T]. */
  void process_mel(const float *mel, unsigned int n, std::vector<float> &out);

  /** @brief Create + compile + initialize the graph at MEL_FRAMES. */
  void build_and_init();

  std::string weight_path_; /**< bigvgan.bin path, for ensure_frames */

  unsigned int MEL_DIM = 80;
  unsigned int UP_INIT_CH = 1536;
  unsigned int MEL_FRAMES = 128;  /**< fixed input length (compile-time) */
  unsigned int SAMPLE_RATE = 24000;
};

} // namespace causallm

#endif /* __QWEN25_OMNI_BIGVGAN_H__ */

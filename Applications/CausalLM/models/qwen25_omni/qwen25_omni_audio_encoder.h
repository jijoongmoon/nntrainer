// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_audio_encoder.h
 * @date   12 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Qwen2.5-Omni Thinker audio encoder (Whisper-style, windowed).
 *
 * The Omni audio tower attends strictly within non-overlapping windows of
 * 2 * n_window (=200) mel frames; the sinusoidal positions also restart at 0
 * for every window. Each window is therefore a fully independent encoder
 * forward, so this class compiles a fixed-size per-chunk graph
 *   [B,128,1,200] conv1(k3 s1 p1)+gelu -> conv2(k3 s2 p1)+gelu
 *   -> [B,1,100,1280] +sinusoid pos-embed (baked into the .bin)
 *   -> 32 pre-LN blocks (q/v/out proj with bias, k proj without bias,
 *      bidirectional mha_core, GELU MLP)
 * and runs it once per 200-mel-frame chunk. Partial tail chunks are handled
 * by zero-padding the mel input and restricting attention to the valid
 * post-conv rows via incremental_inference's [0, to) row slicing.
 *
 * The per-audio tail (AvgPool1d(2,2) -> ln_post -> proj 1280->2048) halves
 * the sequence length, which a single graph cannot express under the shared
 * from/to row slicing; pooling is done host-side and ln_post+proj run as a
 * small second graph loaded from a separate .bin.
 *
 * run(prompt) treats @p prompt as the path to a mel feature file
 * ([int32 n_mels][int32 n_frames][fp32 mel[n_mels][n_frames]], n_frames
 * even) and writes "<path>.embd" ([int32 n_tokens][int32 2048][fp32 data]),
 * i.e. 25 embeddings per second of 16 kHz audio.
 */

#ifndef __QWEN25_OMNI_AUDIO_ENCODER_H__
#define __QWEN25_OMNI_AUDIO_ENCODER_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief Qwen25OmniAudioEncoder class (standalone audio tower)
 */
class Qwen25OmniAudioEncoder : virtual public Transformer {

public:
  static constexpr const char *architectures = "Qwen25OmniAudioEncoder";

  Qwen25OmniAudioEncoder(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Qwen25OmniAudioEncoder() = default;

  /**
   * @brief Compile the per-chunk encoder graph and the pooled head graph.
   */
  void initialize() override;

  /**
   * @brief Load encoder weights; the head .bin is resolved next to it.
   */
  void load_weight(const std::string &weight_path) override;

  /**
   * @brief Encode a mel feature file into audio token embeddings.
   * @param prompt path to the mel feature file (see file header note)
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

  /**
   * @brief Encode mel features in memory (Phase-B entry point).
   * @param mel n_mels x n_frames, mel-bin major; n_frames must be even
   * @return n_tokens x output_dim embeddings, row-major
   */
  std::vector<float> encode(const float *mel, unsigned int n_frames);

protected:
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief One pre-LN encoder block (LN -> QKV -> mha_core -> out + residual,
   *        LN -> fc1 -> gelu -> fc2 + residual)
   */
  Tensor createEncoderBlock(const int layer_id, Tensor input);

private:
  /**
   * @brief Rewind every mha_core's persistent cache_index before a chunk.
   */
  void resetAttentionCache();

  ModelHandle head_model;       /**< AvgPool tail: ln_post + proj */
  std::string head_weight_file; /**< head .bin filename (next to encoder) */

  unsigned int NUM_MEL = 128;      /**< mel bins */
  unsigned int CHUNK_MEL = 200;    /**< mel frames per attention window */
  unsigned int CHUNK_FRAMES = 100; /**< post-conv frames per window */
  unsigned int POOLED_FRAMES = 50; /**< post-pool frames per window */
  unsigned int OUTPUT_DIM = 2048;  /**< thinker hidden size */
};

} // namespace causallm

#endif /* __QWEN25_OMNI_AUDIO_ENCODER_H__ */

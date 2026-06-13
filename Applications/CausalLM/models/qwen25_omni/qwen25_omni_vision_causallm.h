// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_vision_causallm.h
 * @date   13 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Qwen2.5-Omni Thinker with image input (image + text in / text out).
 *
 * Extends the text decoder with: (1) embedding_injection that replaces each
 * <|IMAGE|> placeholder embedding with a vision-encoder output row, and
 * (2) M-RoPE — each attention layer applies the host-computed rotary cos/sin
 * (built from the 3D t/h/w position ids per get_rope_index) to q and k via
 * the mrope_apply layer, with mha_core running at rope_theta=0 (so the core
 * attention/KV-cache code is untouched for every other model).
 *
 * run() prompt: "image:<patch_file> <question>" — the patch file is the
 * flattened-patch tensor the vision tooling produces; this builds the chat
 * template with the expanded <|IMAGE|> tokens, encodes the image, scatters
 * its embeddings and fills the M-RoPE cos/sin tables.
 *
 * SCOPE: single image, single-window grid (<=112x112 px), matching the
 * standalone vision encoder. nntr_config: "vision_encoder_path".
 */

#ifndef __QWEN25_OMNI_VISION_CAUSAL_LM_H__
#define __QWEN25_OMNI_VISION_CAUSAL_LM_H__

#include <array>

#include <qwen25_omni_causallm.h>
#include <qwen25_omni_vision_encoder.h>

namespace causallm {

/**
 * @brief Qwen25OmniVisionCausalLM class (image + text in, text out)
 */
class Qwen25OmniVisionCausalLM : public Qwen25OmniCausalLM {

public:
  static constexpr const char *architectures = "Qwen25OmniVisionChat";

  Qwen25OmniVisionCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(flattenThinkerTextConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    Qwen25OmniCausalLM(cfg, generation_cfg, nntr_cfg) {
    IMAGE_TOKEN_ID = cfg.value("image_token_index", 151655);
    VIDEO_TOKEN_ID = cfg.value("video_token_index", 151656);
    VISION_START_ID = cfg.value("vision_start_token_id", 151652);
    VISION_END_ID = cfg.value("vision_end_token_id", 151653);
    POSITION_ID_PER_SECONDS = cfg.value("position_id_per_seconds", 25);
    VIDEO_SECOND_PER_GRID = nntr_cfg.value("video_second_per_grid", 1.0f);
    vision_encoder_path = nntr_cfg.value("vision_encoder_path", std::string());
    if (cfg.contains("rope_scaling") &&
        cfg["rope_scaling"].contains("mrope_section"))
      MROPE_SECTION =
        cfg["rope_scaling"]["mrope_section"].get<std::vector<int>>();
  }

  virtual ~Qwen25OmniVisionCausalLM() = default;

  void initialize() override;

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

protected:
  std::pair<Tensor, Tensor> constructModel() override;

  /** @brief Qwen2-style attention with M-RoPE on q/k and mha_core theta=0. */
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  void registerCustomLayers() override;

  /**
   * @brief Inputs in the compiled graph's order: input0, img_embd, mrope_cos,
   *        mrope_sin (creation order, right after input0), then the
   *        name-sorted KV caches.
   */
  std::vector<float *> buildInferenceInputs(float *input_sample) override;

private:
  /**
   * @brief Fill mrope_cos/mrope_sin from a 3D position table [3][len] using
   *        mrope_section, rope_theta, head_dim. Positions beyond @p len use
   *        sequential continuation (generated text tokens).
   */
  void buildMRoPETables(const std::vector<std::array<int, 3>> &pos3d);

  std::unique_ptr<Qwen25OmniVisionEncoder> vision_encoder;
  std::string vision_encoder_path;
  json vision_cfg, vision_gen_cfg, vision_nntr_cfg;

  Tensor mrope_cos_t, mrope_sin_t, img_embd_t; /**< symbolic side inputs */
  std::vector<float> mrope_cos, mrope_sin;     /**< [MAX_SEQ_LEN * HEAD_DIM] */
  std::vector<float> img_buf;                  /**< [INIT_SEQ_LEN * DIM] */

  int IMAGE_TOKEN_ID = 151655;
  int VIDEO_TOKEN_ID = 151656;
  int VISION_START_ID = 151652;
  int VISION_END_ID = 151653;
  int POSITION_ID_PER_SECONDS = 25;
  float VIDEO_SECOND_PER_GRID = 1.0f;
  std::vector<int> MROPE_SECTION = {16, 24, 24};
};

} // namespace causallm

#endif /* __QWEN25_OMNI_VISION_CAUSAL_LM_H__ */

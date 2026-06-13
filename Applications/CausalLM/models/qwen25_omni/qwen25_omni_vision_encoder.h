// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_vision_encoder.h
 * @date   13 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Qwen2.5-Omni Thinker vision encoder (Qwen2.5-VL-style ViT).
 *
 * Standalone image tower: flattened patches -> patch_embed (Conv3d-as-linear)
 * -> 32 pre-norm blocks (RMSNorm, separate q/k/v with bias, 2D-RoPE, full
 * bidirectional mha_core, proj, SwiGLU MLP with bias) -> patch merger
 * (RMSNorm + 2x2 spatial merge + MLP) -> out_hidden_size (2048) embeddings,
 * one per spatial_merge_size^2 patch block.
 *
 * SCOPE: this build targets images whose merged grid fits in a single window
 * (<= window_size px, i.e. <= 8x8 raw patches / 112x112 px for the 3B config).
 * In that regime the windowed and full-attention layers are identical and no
 * window reordering / block-diagonal masking is needed, so every layer is a
 * plain full-attention pass. Larger images need the windowed path (TODO).
 *
 * run(prompt) treats @p prompt as a patch feature file
 * ([int32 grid_h][int32 grid_w][fp32 patches[grid_h*grid_w][1176]]) and
 * writes "<path>.embd" ([int32 n_tokens][int32 2048][fp32 data]).
 */

#ifndef __QWEN25_OMNI_VISION_ENCODER_H__
#define __QWEN25_OMNI_VISION_ENCODER_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief Qwen25OmniVisionEncoder class (standalone, single-window)
 */
class Qwen25OmniVisionEncoder : virtual public Transformer {

public:
  static constexpr const char *architectures = "Qwen25OmniVisionEncoder";

  Qwen25OmniVisionEncoder(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Qwen25OmniVisionEncoder() = default;

  /** @brief Compile the main block graph and the merger-MLP head graph. */
  void initialize() override;

  /** @brief Load block weights; the merger-MLP head bin is loaded too. */
  void load_weight(const std::string &weight_path) override;

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

  /**
   * @brief Encode flattened patches (compiled GRID_H*GRID_W rows of PATCH_DIM)
   *        into merged*OUT_HIDDEN embeddings, row-major.
   */
  std::vector<float> encode(const float *patches);

  /** @brief raw patch grid the graph was compiled for */
  unsigned int gridH() const { return GRID_H; }
  unsigned int gridW() const { return GRID_W; }
  unsigned int gridT() const { return GRID_T; }
  unsigned int patchDim() const { return PATCH_DIM; }

protected:
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  std::pair<Tensor, Tensor> constructModel() override;

  void registerCustomLayers() override;

private:
  Tensor createBlock(int layer_id, Tensor input);

  ModelHandle head_model;       /**< merger MLP: reshape->fc->gelu->fc */
  std::string head_weight_file; /**< merger-MLP head bin filename */

  unsigned int PATCH_DIM;          /**< in_chans*temporal*patch*patch (1176) */
  unsigned int OUT_HIDDEN;         /**< merger output dim (2048) */
  unsigned int SPATIAL_MERGE;      /**< 2 */
  unsigned int GRID_H, GRID_W;     /**< raw patch grid for this run */
  unsigned int GRID_T;             /**< temporal patches (1 for images) */
  unsigned int MERGE_HIDDEN;       /**< DIM * SPATIAL_MERGE^2 (5120) */
  unsigned int WINDOW_SIZE;        /**< window size in px (112) */
  unsigned int PATCH_SIZE;         /**< patch size in px (14) */
  std::vector<int> FULLATT;        /**< full-attention block indexes */
};

} // namespace causallm

#endif /* __QWEN25_OMNI_VISION_ENCODER_H__ */

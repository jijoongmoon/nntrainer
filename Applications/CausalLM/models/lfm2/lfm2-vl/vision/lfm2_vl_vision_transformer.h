// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_vision_transformer.h
 * @date   13 May 2026
 * @brief  CLIP/SigLIP-style Vision Transformer encoder for nntrainer CausalLM.
 *         Targets the LFM2.5-VL vision tower (GGUF tensor naming v.*).
 *
 *         Architecture summary (256x256 image, patch16, 256 patches, no CLS):
 *
 *           [Image B,3,H,W]
 *                 |
 *           [Conv2D patch_embed]   (kernel=patch_size, stride=patch_size)
 *                 |
 *           [Reshape -> B,1,N,DIM] (N = num_patches)
 *                 |
 *           [+ position_embed]
 *                 |
 *           Encoder block * NUM_LAYERS (Pre-LN ViT):
 *                 LN1 -> Q/K/V (with bias) -> non-causal MHA -> O(+bias)
 *                       -> +residual
 *                 LN2 -> FC_up(+bias) -> GELU -> FC_down(+bias)
 *                       -> +residual
 *                 |
 *           [post_ln]
 *                 |
 *           [Image features B,N,DIM]
 */

#ifndef __LFM2_VL_VISION_TRANSFORMER_H__
#define __LFM2_VL_VISION_TRANSFORMER_H__

#include <transformer.h>

namespace causallm {

class Lfm2VlVisionTransformer : public Transformer {
public:
  static constexpr const char *architectures = "Lfm2VlVisionTransformer";

  Lfm2VlVisionTransformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::EMBEDDING) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  /**
   * @brief Fill in the LLM-shaped fields that base Transformer::setupParameters
   *        requires but a vision-tower config does not naturally carry
   *        (vocab_size, max_position_embeddings, rope_theta,
   *        tie_word_embeddings, rms_norm_eps, num_attention_heads). The values
   *        written here are immediately overwritten by
   *        Lfm2VlVisionTransformer::setupParameters() with the real ViT values.
   */
  static json &sanitizeConfig(json &cfg);

  ~Lfm2VlVisionTransformer() override = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Build the symbolic graph: image -> patch_embed + pos_embed ->
   *        N encoder blocks -> post_ln. Returns (input, output) for compile().
   */
  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief Single ViT encoder block (Pre-LN). Returns the block's output
   *        tensor which becomes the next block's input.
   */
  Tensor createEncoderBlock(int layer_id, Tensor input);

  /**
   * @brief Self-attention sub-graph (Q/K/V projections with bias,
   *        bidirectional mha_core, output projection with bias).
   */
  Tensor createSelfAttention(int layer_id, Tensor x);

  /**
   * @brief MLP sub-graph (FC_up + GELU + FC_down, all with bias).
   *        Named distinctly to avoid hiding the base Transformer's virtual
   *        createMlp(int, int, int, Tensor) overload.
   */
  Tensor createVitMlp(int layer_id, Tensor x);

  /**
   * @brief Run a forward pass on a preprocessed image tensor and dump the
   *        first batch's output features summary.
   */
  void run(const WSTR image_tensor_path, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

protected:
  unsigned int IMAGE_SIZE;   /**< image side length, e.g. 256 */
  unsigned int PATCH_SIZE;   /**< patch side length, e.g. 16 */
  unsigned int NUM_CHANNELS; /**< image channels, typically 3 */
  unsigned int NUM_PATCHES;  /**< (IMAGE_SIZE / PATCH_SIZE) ^ 2 */
};

} // namespace causallm

#endif /* __LFM2_VL_VISION_TRANSFORMER_H__ */

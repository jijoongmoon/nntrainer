// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen25_omni_causallm.h
 * @date   12 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Qwen2.5-Omni Thinker text model as a CausalLM (text-only path).
 * @note   Please refer to the following code :
 *  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
 *
 * Qwen2.5-Omni is a Thinker-Talker multimodal model. This class runs the
 * Thinker's text decoder only (text in / text out). The Thinker text model
 * uses the Qwen2.5 decoder architecture (Q/K/V projections with bias, GQA,
 * RMSNorm, SwiGLU), so the Qwen2 transformer graph is reused as-is.
 *
 * Two Omni-specific deltas are handled here:
 *  - The HF config nests the text model parameters under
 *    thinker_config.text_config; flattenThinkerTextConfig() merges them into
 *    the top level so the common Transformer/CausalLM setup can read them.
 *  - The text model declares M-RoPE (mrope_section). For pure-text inputs the
 *    temporal/height/width position ids are identical, which makes M-RoPE
 *    numerically equivalent to the standard 1-D RoPE used by mha_core.
 *
 * The audio/vision encoders, the Talker and the Token2Wav decoder are not
 * part of this model.
 */

#ifndef __QWEN25_OMNI_CAUSAL_LM_H__
#define __QWEN25_OMNI_CAUSAL_LM_H__

#include <qwen2_causallm.h>

namespace causallm {

/**
 * @brief Qwen25OmniCausalLM class (Thinker text decoder, text-only)
 */
class Qwen25OmniCausalLM : public Qwen2CausalLM {

public:
  static constexpr const char *architectures = "Qwen2_5OmniModel";

  Qwen25OmniCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(flattenThinkerTextConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    Qwen2CausalLM(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen25OmniCausalLM() = default;

  /**
   * @brief Merge thinker_config.text_config (and the thinker-level token ids)
   *        into the top level of @p cfg so that the common setupParameters()
   *        can consume the HF Qwen2.5-Omni config as-is.
   *        Idempotent: a config without thinker_config is returned unchanged.
   * @param cfg HF config.json contents; mutated in place
   * @return reference to @p cfg
   */
  static json &flattenThinkerTextConfig(json &cfg);
};

} // namespace causallm

#endif /* __QWEN25_OMNI_CAUSAL_LM_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   gemma3_causallm.h
 * @date   24 Dec 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __GEMMA3_CAUSAL_LM_H__
#define __GEMMA3_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Gemma3Transformer class
 * @note  Gemma3 differences from base CausalLM:
 *        - Post-attention norm (RMS norm after attention output, before residual)
 *        - Post-FFN norm (RMS norm after FFN output, before residual)
 *        - GeGLU FFN (GeLU activation instead of SiLU)
 *        - Q/K reshaped_rms_norm
 *        - Per-layer sliding window vs global attention
 *        - Attention logit softcapping
 *        - Embedding scaling by sqrt(hidden_size)
 */
class Gemma3Transformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "Gemma3Transformer";

  Gemma3Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {
    if (cfg.contains("layer_types")) {
      layer_types = cfg["layer_types"].get<std::vector<std::string>>();
    }
    EMBEDDING_SCALE = std::sqrt(static_cast<float>(cfg["hidden_size"]));
  }

  virtual ~Gemma3Transformer() = default;

protected:
  static json &sanitizeConfig(json &cfg);
  static json &sanitizeGenerationConfig(json &gen_cfg, const json &cfg);

  std::vector<std::string> layer_types;

public:
  /**
   * @brief Create attention using Symbolic Tensor API.
   * Gemma3 specifics: Q/K reshaped_rms_norm, per-layer sliding window,
   * softcapping, no bias.
   */
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                          int head_dim, Tensor query, Tensor key,
                          Tensor value) override;

  /**
   * @brief Create decoder block using Symbolic Tensor API.
   * Gemma3 specifics: post-attention norm and post-FFN norm (4 norms total).
   */
  Tensor createTransformerDecoderBlock(const int layer_id,
                                        Tensor input) override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Create MLP using Symbolic Tensor API.
   * Gemma3 specifics: GeGLU (GeLU activation instead of SiLU/SwiGLU).
   */
  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                    Tensor input) override;

  void registerCustomLayers() override;
};

/**
 * @brief Gemma3CausalLM class
 */
class Gemma3CausalLM : public Gemma3Transformer {

public:
  static constexpr const char *architectures = "Gemma3ForCausalLM";

  Gemma3CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    CausalLM(sanitizeConfig(cfg), sanitizeGenerationConfig(generation_cfg, cfg),
             nntr_cfg, ModelType::CAUSALLM),
    Gemma3Transformer(sanitizeConfig(cfg),
                      sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {
  }

  virtual ~Gemma3CausalLM() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override {
    CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);
    Gemma3Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  void registerCustomLayers() override;

private:
};
} // namespace causallm

#endif /* __GEMMA3_CAUSAL_LM_H__ */

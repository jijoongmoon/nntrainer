// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   embedding_gemma.h
 * @date   11 Jan 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This embedding_gemma.h constructs a class for Gemma3-based Embedding
 * model.
 */

#ifndef __EMBEDDING_GEMMA_H__
#define __EMBEDDING_GEMMA_H__

#include <gemma3_causallm.h>
#include <sentence_transformer.h>

namespace causallm {

/**
 * @brief EmbeddingGemma Class
 */
class EmbeddingGemma : public SentenceTransformer, public Gemma3Transformer {

public:
  static constexpr const char *architectures = "EmbeddingGemma";

  /**
   * @brief Construct a new EmbeddingGemma object
   * @param cfg Configuration for the model
   * @param generation_cfg Configuration for generation
   * @param nntr_cfg Configuration for nntrainer
   */
  EmbeddingGemma(json &cfg, json &generation_cfg, json &nntr_cfg) :
    CausalLM(
      Gemma3Transformer::sanitizeConfig(cfg),
      Gemma3Transformer::sanitizeGenerationConfig(generation_cfg, cfg),
      nntr_cfg, ModelType::EMBEDDING),
    SentenceTransformer(cfg, generation_cfg, nntr_cfg),
    Gemma3Transformer(cfg, generation_cfg, nntr_cfg) {}

  /**
   * @brief Destroy the EmbeddingGemma object
   */
  virtual ~EmbeddingGemma() = default;

  /**
   * @brief Setup parameters
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief register CustomLayers
   */
  void registerCustomLayers() override;
};

} // namespace causallm

#endif // __EMBEDDING_GEMMA_H__

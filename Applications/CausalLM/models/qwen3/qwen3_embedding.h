// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_embedding.h
 * @date   07 Jan 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This qwen3_embedding.h constructs a class for Qwen3-based Embedding
 * model.
 */

#ifndef __QWEN3_EMBEDDING_H__
#define __QWEN3_EMBEDDING_H__

#include <qwen3_causallm.h>
#include <sentence_transformer.h>

namespace causallm {

/**
 * @brief Qwen3Embedding Class
 */
class Qwen3Embedding : public SentenceTransformer, public Qwen3Transformer {

public:
  static constexpr const char *architectures = "Qwen3Embedding";

  /**
   * @brief Construct a new Qwen3Embedding object
   * @param cfg Configuration for the model
   * @param generation_cfg Configuration for generation
   * @param nntr_cfg Configuration for nntrainer
   */
  Qwen3Embedding(json &cfg, json &generation_cfg, json &nntr_cfg) :
    CausalLM(cfg, generation_cfg, nntr_cfg, ModelType::EMBEDDING),
    SentenceTransformer(cfg, generation_cfg, nntr_cfg),
    Qwen3Transformer(cfg, generation_cfg, nntr_cfg) {}

  /**
   * @brief Destroy the Qwen3Embedding object
   */
  virtual ~Qwen3Embedding() = default;

  /**
   * @brief register CustomLayers
   */
  void registerCustomLayers() override;
};

} // namespace causallm

#endif // __QWEN3_EMBEDDING_H__

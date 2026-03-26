// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   qwen2_embedding.h
 * @date   14 January 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This qwen2_embedding.h constructs a class for Qwen2-based Embedding
 * model.
 */
#ifndef __QWEN2_EMBEDDING_H__
#define __QWEN2_EMBEDDING_H__

#include <qwen2_causallm.h>
#include <sentence_transformer.h>

namespace causallm {

/**
 * @brief Qwen2Embedding class
 */
class Qwen2Embedding : public SentenceTransformer, public Qwen2Transformer {

public:
  Qwen2Embedding(json &cfg, json &generation_cfg, json &nntr_cfg) :
    CausalLM(cfg, generation_cfg, nntr_cfg, ModelType::EMBEDDING),
    SentenceTransformer(cfg, generation_cfg, nntr_cfg),
    Qwen2Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen2Embedding() {}
};

} // namespace causallm

#endif /* __QWEN2_EMBEDDING_H__ */

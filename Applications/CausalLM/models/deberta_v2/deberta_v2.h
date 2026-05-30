// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   deberta_v2.h
 * @date   14 January 2026
 * @brief  DeBERTa V2 encoder model for SentenceTransformer embeddings
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Please refer to the following code :
 * https://github.com/huggingface/transformers/blob/5c1c72b/src/transformers/models/deberta_v2/modeling_deberta_v2.py
 */

#ifndef __DEBERTA_V2_H__
#define __DEBERTA_V2_H__

#include <sentence_transformer.h>

namespace causallm {

/**
 * @brief DebertaV2 embedding model
 */
class DebertaV2 : public SentenceTransformer {

public:
  static constexpr const char *architectures = "DebertaV2";

  /**
   * @brief Construct a new DebertaV2 object
   * @param cfg Configuration for the model (config.json)
   * @param generation_cfg Configuration for the generation
   * @param nntr_cfg Configuration for nntrainer
   */
  DebertaV2(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::EMBEDDING),
    SentenceTransformer(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  /**
   * @brief Destroy the DebertaV2 object
   */
  virtual ~DebertaV2() = default;

  /**
   * @brief Encode the prompt and return the embedding output
   */
  std::vector<float *> encode(const WSTR prompt, const WSTR system_prompt = "",
                              const WSTR tail_prompt = "") override;

protected:
  /**
   * @brief Fill defaults absent from typical DeBERTa V2 config.json files.
   */
  static json &sanitizeConfig(json &cfg);

  /**
   * @brief Setup the parameters for the DeBERTa V2 model
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Construct DeBERTa V2 before SentenceTransformer pooling modules.
   */
  std::pair<Tensor, Tensor> constructTransformerModule() override;

  /**
   * @brief Create one DeBERTa V2 encoder layer
   */
  Tensor createDebertaLayer(const int layer_id, Tensor input,
                            Tensor rel_embeddings);

  /**
   * @brief Create DeBERTa V2 disentangled self-attention
   */
  Tensor createDebertaV2Attention(const int layer_id, Tensor input,
                                  Tensor rel_embeddings);

  /**
   * @brief register CustomLayers
   */
  void registerCustomLayers() override;

private:
  int MAX_RELATIVE_POSITIONS = 0;
  bool C2P = false;
  bool P2C = false;
  bool SHARE_ATT_KEY = true;
  bool RELATIVE_ATTENTION = true;
  bool NORM_REL_EBD = false;
  int POSITION_BUCKETS = -1;
};

} // namespace causallm

#endif /* __DEBERTA_V2_H__ */

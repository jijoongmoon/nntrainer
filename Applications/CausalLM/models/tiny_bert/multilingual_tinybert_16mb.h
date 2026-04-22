// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   multilingual_tinybert_16mb.h
 * @date   21 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This multilingual_tinybert_16mb.h constructs a class for
 *         a BERT-based encoder-only embedding model
 *         (multilingual-TinyBERT-16MB) built on top of the causallm
 *         Transformer base class.
 * @note   Please refer to the following code :
 *  https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/bert/modeling_bert.py
 */

#ifndef __MULTILINGUAL_TINYBERT_16MB_H__
#define __MULTILINGUAL_TINYBERT_16MB_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief BertTransformer class
 * @note  Base class for BERT-style encoder-only models.
 *        The structure is :
 *
 *          [Input] [PositionIds] [TokenTypeIds]
 *             |         |              |
 *        [WordEmb]  [PosEmb]     [TokenTypeEmb]
 *             \_________|_____________/
 *                       |
 *                  [LayerNorm]
 *                       |
 *             [Encoder Block] (repeated N times)
 *                       |
 *                   [Output]
 *
 *        Each encoder block uses post-norm :
 *          x = LayerNorm(x + SelfAttention(x))
 *          x = LayerNorm(x + FFN(x))
 */
class BertTransformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "BertTransformer";

  BertTransformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::EMBEDDING) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~BertTransformer() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void constructModel() override;

  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id,
                                std::string input_name) override;

  std::vector<LayerHandle> createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;

  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  void registerCustomLayers() override;

protected:
  /**
   * @brief Sanitize config to fill defaults that are not present in
   * typical BERT-style config.json files (e.g. rope_theta, rms_norm_eps).
   */
  static json &sanitizeConfig(json &cfg);

  /**
   * @brief Type-vocab size for token_type_ids (BERT default: 2)
   */
  unsigned int TYPE_VOCAB_SIZE = 2;

  /**
   * @brief Activation function used inside the FFN ("gelu" by default)
   */
  std::string HIDDEN_ACT = "gelu";
};

/**
 * @brief MultilingualTinyBert class
 * @note  Concrete runnable model for multilingual-TinyBERT-16MB.
 *        It inherits BertTransformer and provides the encode / run
 *        methods that feed three inputs (input_ids, position_ids,
 *        token_type_ids) into the underlying nntrainer model.
 */
class MultilingualTinyBert : public BertTransformer {

public:
  static constexpr const char *architectures = "BertModel";

  MultilingualTinyBert(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::EMBEDDING),
    BertTransformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~MultilingualTinyBert() = default;

  /**
   * @brief Run the model and print the embedding output
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

  /**
   * @brief Encode the prompt and return the embedding output
   */
  std::vector<float *> encode(const WSTR prompt, const WSTR system_prompt = "",
                              const WSTR tail_prompt = "");

  /**
   * @brief Attach (or detach) a BaseStreamer to intercept embedding
   *        output during the next call to run().
   *        Passing nullptr detaches any currently-attached streamer.
   */
  void setStreamer(::BaseStreamer *streamer) override {
    streamer_ = streamer;
  }

  /**
   * @brief Get the embedding output as a string
   * @param batch_idx Index of the batch item
   * @return Embedding result string
   */
  std::string getOutput(int batch_idx = 0) const override;

private:
  ::BaseStreamer *streamer_ = nullptr;
  std::vector<std::string> output_list;
};

} // namespace causallm

#endif /* __MULTILINGUAL_TINYBERT_16MB_H__ */

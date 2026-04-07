// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_5_causallm.h
 * @date   07 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author NNTrainer Authors
 * @bug    No known bugs except for NYI items
 * @note   Qwen3.5 hybrid architecture: alternating Full Attention and
 *         Gated Delta Net (linear attention) layers.
 * @ref    https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
 */

#ifndef __QWEN3_5_CAUSAL_LM_H__
#define __QWEN3_5_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Qwen3_5Transformer class
 * @note  Hybrid architecture with full_attention and linear_attention layers.
 *        - full_attention: Standard MHA with Q*2 sigmoid gate + Q/K norm
 *        - linear_attention: GatedDeltaNet with causal conv1d + delta rule
 *        Layer type is determined by full_attention_interval (default 4):
 *          layer_type[i] = (i+1) % interval ? "linear_attention" : "full_attention"
 */
class Qwen3_5Transformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "Qwen3_5Transformer";

  Qwen3_5Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen3_5Transformer() = default;

  /**
   * @brief Override createTransformerDecoderBlock to route between
   *        full_attention and linear_attention based on layer index
   */
  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id,
                                std::string input_name) override;

  /**
   * @brief Create full attention layers for Qwen3.5
   * @note  Q projection outputs 2x (half is sigmoid gate on output)
   *        Uses Q/K RMSNorm and partial RoPE
   */
  std::vector<LayerHandle>
  createAttention(const int layer_id, int seq_len, int n_heads, int head_dim,
                  std::string query_name, std::string key_name,
                  std::string value_name) override;

  /**
   * @brief Create GatedDeltaNet (linear attention) block
   * @param layer_id   Layer index
   * @param input_name Name of the input layer (after attention norm)
   * @return Vector of layer handles for the GatedDeltaNet block
   */
  virtual std::vector<LayerHandle>
  createGatedDeltaNet(const int layer_id, std::string input_name);

  void registerCustomLayers() override;

protected:
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /** Qwen3.5 specific parameters */
  unsigned int FULL_ATTENTION_INTERVAL = 4;
  int LINEAR_CONV_KERNEL_DIM = 4;
  int LINEAR_KEY_HEAD_DIM = 128;
  int LINEAR_VALUE_HEAD_DIM = 128;
  int LINEAR_NUM_KEY_HEADS = 16;
  int LINEAR_NUM_VALUE_HEADS = 16;
  float PARTIAL_ROTARY_FACTOR = 0.25f;

  /**
   * @brief Check if a layer uses full attention
   */
  bool isFullAttentionLayer(int layer_id) const {
    return ((layer_id + 1) % FULL_ATTENTION_INTERVAL) == 0;
  }
};

/**
 * @brief Qwen3_5CausalLM class
 */
class Qwen3_5CausalLM : public CausalLM, public Qwen3_5Transformer {

public:
  static constexpr const char *architectures = "Qwen3_5ForCausalLM";

  Qwen3_5CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    Qwen3_5Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen3_5CausalLM() = default;

  void registerCustomLayers() override;

private:
};

} // namespace causallm

#endif /* __QWEN3_5_CAUSAL_LM_H__ */

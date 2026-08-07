// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qwen3_5_moe_causallm.h
 * @date   30 June 2026
 * @brief  Qwen3.6-35B-A3B (Qwen3-Next, qwen3_5_moe) GDN-hybrid causal LM.
 * @author Claude Code
 * @bug    No known bugs except for NYI items
 *
 * Hybrid decoder: per-layer `layer_types` dispatch — `linear_attention` layers
 * use the GatedDeltaNet mixer (gated_delta_net), `full_attention` layers use the
 * Qwen3 attention path extended with an output gate + partial RoPE. Every layer's
 * FFN is an MoE block (routed top-k via qwen_moe) plus an always-on shared expert.
 */

#ifndef __QWEN3_5_MOE_CAUSAL_LM_H__
#define __QWEN3_5_MOE_CAUSAL_LM_H__

#include <climits>
#include <string>
#include <vector>

#include <qwen3_moe_causallm.h>

namespace causallm {

/**
 * @brief Qwen3_5MoeCausalLM — Qwen3-Next GDN-hybrid MoE causal LM
 */
class Qwen3_5MoeCausalLM : public Qwen3MoECausalLM {
public:
  static constexpr const char *architectures = "Qwen3_5MoeForCausalLM";

  Qwen3_5MoeCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    Qwen3MoECausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Qwen3_5MoeCausalLM() = default;

  /** dispatch on layer_types: GDN mixer vs gated full-attention */
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  /** routed top-k MoE (qwen_moe) + always-on shared expert */
  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                   Tensor input) override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override;

  /**
   * @brief No layer of this model is sliding-window.
   *
   * The base derives sliding-ness from SLIDING_WINDOW_PATTERN, which defaults
   * to 5 -- on a 40-layer model that would falsely mark layers sliding and then
   * mis-size the KV placeholders (kvRingCap) and the mha_core sliding_window
   * property, ending in a shape-mismatch throw in allocateAndBindKVCache. This
   * model's layer kinds come from `layer_types`, so the pattern is meaningless
   * here and the base contract requires overriding.
   */
  unsigned int getLayerSlidingWindow(int) const override { return UINT_MAX; }

private:
  std::vector<std::string> LAYER_TYPES; /**< per-layer "linear_attention"/"full_attention" */
  unsigned int N_EXPERTS;
  unsigned int N_EXPERTS_PER_TOK;
  unsigned int SHARED_EXPERT_INTERMEDIATE_SIZE;
  // GDN (linear_attention) dims
  unsigned int LINEAR_NUM_VALUE_HEADS;
  unsigned int LINEAR_NUM_KEY_HEADS;
  unsigned int LINEAR_KEY_HEAD_DIM;
  unsigned int LINEAR_VALUE_HEAD_DIM;
  unsigned int LINEAR_CONV_KERNEL_DIM;
  float PARTIAL_ROTARY_FACTOR; /**< full-attn partial RoPE (e.g. 0.25) */

  /** GatedDeltaNet mixer sub-graph (linear_attention layers) */
  Tensor createGatedDeltaNet(const int layer_id, Tensor input);
  /** Qwen3 attention + output gate + partial RoPE (full_attention layers) */
  Tensor createFullAttention(const int layer_id, int n_heads, int head_dim,
                             Tensor query, Tensor key, Tensor value);
};

} // namespace causallm

#endif /* __QWEN3_5_MOE_CAUSAL_LM_H__ */

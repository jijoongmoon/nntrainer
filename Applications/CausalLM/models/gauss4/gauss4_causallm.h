// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gauss4_causallm.h
 * @date   16 Jun 2026
 * @brief  Gauss4 CausalLM model.
 * @author Joonseok Oh <jrock.oh@samsung.com>
 *
 */

#ifndef __GAUSS4_CAUSAL_LM_H__
#define __GAUSS4_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Gauss4Transformer -- decoder stack (no LM head).
 */
class Gauss4Transformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "Gauss4Transformer";

  Gauss4Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Gauss4Transformer() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  std::pair<Tensor, Tensor> constructModel() override;

  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  /**
   * Normal attention (layers 0-19): Q, K, V from attention_norm output.
   * Includes gated-attn gate: gate_down(attn_norm) -> gate_up -> sigmoid.
   */
  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  /**
   * Shared-KV attention (layers 20-34): Q from attention_norm,
   * K/V from pre-computed shared rotation tensor (skip_prefill=true).
   */
  Tensor createSharedAttention(const int layer_id, Tensor query,
                               Tensor shared_kv);

  /**
   * LRA MLP: silu(linear_up + gate_down) * linear_up -> linear_fc2
   */
  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                   Tensor input) override;

  /**
   * PLE: ple_mix_method=1, ple_pre_mlp=true
   *   gate = sigmoid(PLE_input_gate(ple_input))  <- ffn_norm output
   *   mix  = gate + PLE_embedding(input0)        <- ADD (method=1)
   *   out  = PLE_post_norm(PLE_projection(mix))  <- rms_reverse_norm
   */
  Tensor createPerLayerEmbedding(const int layer_id, Tensor ple_input,
                                 Tensor input0, bool skip_prefill);

  void registerCustomLayers() override;

  /** Create KV-cache input placeholders for mha_core inputs 3 and 4. */
  std::pair<Tensor, Tensor>
  createGauss4KVCachePlaceholders(const int layer_id, unsigned int kv_width);

  /** @copydoc Transformer::getModelFeatures() — gauss4: PRE norm, gated-attn
   *  output (no q/k/v-norm), LRA-gate MLP (~swiglu), PLE (ADD-mix + reverse-norm),
   *  sliding+full KV-share tail (project-but-skip-prefill), tied lm_head, no
   *  softcap, uniform head_dim=128, decode on host initially. */
  nntrainer::ModelFeatures getModelFeatures() const override {
    nntrainer::ModelFeatures f;
    f.has_qk_norm = false;
    f.has_v_norm = false;
    f.mlp_kind = nntrainer::MlpKind::SWIGLU;
    f.norm_style = nntrainer::NormStyle::PRE;
    f.sliding_window = true;
    f.kv_share_skip_prefill = true;
    f.dual_head_dim = false;
    f.ple = true;
    f.attn_softcap = false;
    f.final_softcap = false;
    f.lmhead_kind = nntrainer::LmHeadKind::TIED;
    f.decode_gpu = true;       // full GPU: prefill + decode (like gemma4/qwen3)
    f.decode_rope_gpu = true;
    f.head_dim = 128;
    return f;
  }

protected:
  // From config.json (NOT nntr_config.json -- those values are STALE)
  bool USE_KV_SHARING = true;
  unsigned int NUM_SEQUENTIAL_LAYERS = 20;      // normal: layers 0..19
  unsigned int SLIDING_ATTENTION_KV_LAYER = 18; // config.json=19, -1 = 18
  unsigned int FULL_ATTENTION_KV_LAYER = 19;    // config.json=20, -1 = 19
  // Per-attention-type RoPE theta (config.json rope_parameters). Sliding layers
  // use sliding_attention.rope_theta (500000); full-attention layers (every 5th)
  // use full_attention.rope_theta (8000000). The CPU reference applied a single
  // (sliding) value to all layers; per-type application is the RoPE fix.
  double SLIDING_ATTENTION_ROPE_THETA = 500000.0;
  double FULL_ATTENTION_ROPE_THETA = 8000000.0;
  unsigned int LATENT_SIZE_PER_GATE = 256;
  unsigned int MLP_LRA_RANK = 512;
  unsigned int HIDDEN_SIZE_PER_LAYER_INPUT = 192;
  std::string PLE_ACT = "sigmoid";
  bool ENABLE_SKIP_PREFILL_OPT = true;

  // New in gauss4-e2b-0.1-rotated
  int PLE_MIX_METHOD = 1;  // 1 = sigmoid(gate) + emb (ADD)
  bool PLE_PRE_MLP = true; // gate input = ffn_norm output (pre-MLP)

  // Captured during constructModel() for KV-sharing layers 20-34
  // After rotation FC: rotation_sliding / rotation_full outputs
  Tensor kv_sharing_sliding_tensor;
  Tensor kv_sharing_full_tensor;

  void appendSkipPrefillIfNeeded(std::vector<std::string> &props,
                                 bool skip) const;
};

/**
 * @brief Gauss4CausalLM -- full causal LM with LM head and KV cache.
 */
class Gauss4CausalLM : public CausalLM, public Gauss4Transformer {

public:
  static constexpr const char *architectures = "Gauss4ForCausalLM";

  Gauss4CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    Gauss4Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Gauss4CausalLM() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override {
    CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);
    Gauss4Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  std::pair<Tensor, Tensor> constructModel() override;

  void registerCustomLayers() override;

protected:
  void allocateAndBindKVCache() override;
};

} // namespace causallm

#endif /* __GAUSS4_CAUSAL_LM_H__ */

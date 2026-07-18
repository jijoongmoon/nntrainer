// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dit_attention.h
 * @date   18 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Block-diagonal non-causal attention for the Qwen2.5-Omni Token2Wav
 *         DiT.
 *
 * The DiT runs a single full-sequence forward (no KV cache) with a per-layer
 * BOOLEAN block allow-mask (modeling:3039-3040, _create_block_diff:3476-3484):
 *   block_id[p] = p / block_size            (block_size = 24)
 *   keep(i, j) iff (block_id[j] - block_id[i]) in [-look_backward, look_ahead]
 * Only 3 of the 22 layers widen by one block (L0/L20 look_backward=1,
 * L10 look_ahead=1); the other 19 are strictly block-diagonal, so each layer
 * instantiates this with its own look props. Scores outside the mask get -inf
 * before the softmax (additive 0 / -inf equivalent).
 *
 * Head-0 q/k arrive already dit_rope'd upstream; all heads share the mask.
 * Non-causal FP32 SDPA, scale 1/sqrt(head_dim), no weights.
 *
 * Inputs: 0=q, 1=k, 2=v  ([B,1,seq,num_heads*head_dim]).
 */

#ifndef __DIT_ATTENTION_LAYER_H__
#define __DIT_ATTENTION_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <vector>

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

namespace causallm {

/**
 * @brief Qwen2.5-Omni DiT block-diagonal attention layer
 */
WIN_EXPORT class DiTAttentionLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT DiTAttentionLayer() :
    Layer(), num_heads(16), head_dim(64), block_size(24), look_ahead(0),
    look_backward(0) {}

  WIN_EXPORT ~DiTAttentionLayer() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void
  calcDerivative(nntrainer::RunLayerContext &context) override {}

  WIN_EXPORT bool supportBackwarding() const override { return false; };

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override{};

  WIN_EXPORT const std::string getType() const override {
    return DiTAttentionLayer::type;
  };

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "dit_attention";

private:
  unsigned int num_heads, head_dim, block_size;
  unsigned int look_ahead, look_backward;
};

} // namespace causallm

#endif /* __DIT_ATTENTION_LAYER_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   vision_attention.h
 * @date   13 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Non-causal bidirectional attention for the Qwen2.5 vision tower,
 *         with optional per-window (block-diagonal) masking.
 *
 * The encoder runs a single full-sequence forward (no KV cache), so this
 * layer computes masked scaled-dot-product attention directly in FP32:
 *   out[i] = softmax_j( q[i]·k[j]/sqrt(d) , j in mask(i) ) · v[j]
 * Window masking keeps HF's windowed attention without reordering patches:
 * token a attends b iff they share a window. The window id of each patch is
 * derived at finalize() from the (fixed) grid — patch p belongs to merged
 * unit p/merge^2 at (mh = g/llm_w, mw = g%llm_w), window
 * (mh/vmws, mw/vmws) with vmws = window_size/merge/patch_size. Full-attention
 * layers (fullatt_block_indexes) set is_full so every token shares window 0.
 *
 * Inputs: 0=q, 1=k, 2=v  ([B,1,seq,num_heads*head_dim], q/k already 2D-RoPE'd).
 */

#ifndef __VISION_ATTENTION_LAYER_H__
#define __VISION_ATTENTION_LAYER_H__

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
 * @brief Qwen2.5 vision windowed/full attention layer
 */
WIN_EXPORT class VisionAttentionLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT VisionAttentionLayer() :
    Layer(), num_heads(0), head_dim(0), grid_h(0), grid_w(0), grid_t(1),
    window_size(112), patch_size(14), spatial_merge_size(2), is_full(false) {}

  WIN_EXPORT ~VisionAttentionLayer() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override {}

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void
  calcDerivative(nntrainer::RunLayerContext &context) override {}

  WIN_EXPORT bool supportBackwarding() const override { return false; };

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override{};

  WIN_EXPORT const std::string getType() const override {
    return VisionAttentionLayer::type;
  };

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "vision_attention";

private:
  unsigned int num_heads, head_dim, grid_h, grid_w, grid_t, window_size,
    patch_size, spatial_merge_size;
  bool is_full;
  std::vector<int> window_id; /**< per-patch window id (empty if is_full) */

  void buildWindowIds();
};

} // namespace causallm

#endif /* __VISION_ATTENTION_LAYER_H__ */

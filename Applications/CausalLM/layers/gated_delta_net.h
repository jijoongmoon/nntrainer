// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gated_delta_net.h
 * @date   07 April 2026
 * @brief  Gated Delta Net layer for Qwen3.5 linear attention
 * @see    https://github.com/nntrainer/nntrainer
 * @author NNTrainer Authors
 * @bug    No known bugs except for NYI items
 * @note   Implements Gated Delta Rule linear attention as used in Qwen3.5.
 *         This layer combines:
 *         - Causal depthwise Conv1d on projected QKV
 *         - Gated delta rule recurrent computation
 *         - RMSNormGated output with SiLU gate
 *
 *         Data flow:
 *           input → in_proj_qkv → conv1d(depthwise, causal) → SiLU
 *                                  → split(Q, K, V)
 *           input → in_proj_z   (gate for output norm)
 *           input → in_proj_b   (beta: sigmoid update gate)
 *           input → in_proj_a   (alpha: decay gate via A_log, dt_bias)
 *                    ↓
 *           gated_delta_rule(Q, K, V, g, beta) → output
 *                    ↓
 *           RMSNormGated(output, z) → out_proj → final_output
 *
 * @ref    https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
 */

#ifndef __GATED_DELTA_NET_LAYER_H__
#define __GATED_DELTA_NET_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

#include <causallm_common_properties.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>

namespace causallm {

/**
 * @brief Gated Delta Net layer implementing linear attention with delta rule
 *
 * Weight layout (in order, matching GGUF tensor mapping):
 *   [0] in_proj_qkv.weight  [hidden_size, key_dim*2 + value_dim]
 *   [1] conv1d.weight        [conv_kernel_size, conv_dim] (depthwise)
 *   [2] A_log                [num_v_heads]
 *   [3] in_proj_b.weight     [hidden_size, num_v_heads] (beta/alpha proj)
 *   [4] in_proj_a.weight     [hidden_size, num_v_heads] (alpha/decay proj)
 *   [5] dt_bias              [num_v_heads]
 *   [6] norm.weight          [head_v_dim] (RMSNormGated gamma)
 *   [7] out_proj.weight      [value_dim, hidden_size]
 *   [8] in_proj_z.weight     [hidden_size, value_dim] (gate projection)
 *
 * Tensor layout (temporaries):
 *   [0] conv_state           [batch, conv_dim, conv_kernel_size-1]
 *   [1] recurrent_state      [batch, num_v_heads, key_head_dim, value_head_dim]
 */
WIN_EXPORT class GatedDeltaNetLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT GatedDeltaNetLayer() : Layer(), wt_idx({0}) {}
  WIN_EXPORT ~GatedDeltaNetLayer() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; };

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {};

  WIN_EXPORT const std::string getType() const override {
    return GatedDeltaNetLayer::type;
  };

  WIN_EXPORT void
  setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "gated_delta_net";

private:
  /**
   * @brief Apply causal depthwise conv1d
   * @param input  Input tensor [batch, conv_dim, seq_len]
   * @param weight Conv weight [conv_kernel_size, conv_dim]
   * @param state  Conv state [batch, conv_dim, kernel_size-1] (updated in-place)
   * @param output Output tensor [batch, conv_dim, seq_len]
   * @param seq_len Sequence length (for prefill) or 1 (for incremental)
   */
  void applyCausalConv1d(const nntrainer::Tensor &input,
                         const nntrainer::Tensor &weight,
                         nntrainer::Tensor &state,
                         nntrainer::Tensor &output,
                         unsigned int seq_len);

  /**
   * @brief Recurrent gated delta rule (token-by-token inference)
   */
  void recurrentGatedDeltaRule(const nntrainer::Tensor &query,
                               const nntrainer::Tensor &key,
                               const nntrainer::Tensor &value,
                               const nntrainer::Tensor &g,
                               const nntrainer::Tensor &beta,
                               nntrainer::Tensor &state,
                               nntrainer::Tensor &output);

  /**
   * @brief Chunk-based gated delta rule (prefill mode)
   */
  void chunkGatedDeltaRule(const nntrainer::Tensor &query,
                           const nntrainer::Tensor &key,
                           const nntrainer::Tensor &value,
                           const nntrainer::Tensor &g,
                           const nntrainer::Tensor &beta,
                           nntrainer::Tensor &state,
                           nntrainer::Tensor &output,
                           unsigned int chunk_size = 64);

  /**
   * @brief Apply RMSNormGated: RMSNorm(x) * SiLU(gate)
   */
  void applyRMSNormGated(const nntrainer::Tensor &input,
                          const nntrainer::Tensor &gate,
                          const nntrainer::Tensor &gamma,
                          nntrainer::Tensor &output, float eps);

  /**
   * @brief L2 normalize along the last dimension
   */
  void l2Normalize(nntrainer::Tensor &tensor, float eps = 1e-6f);

  std::array<unsigned int, 9> wt_idx;  /**< weight indices */
  std::array<unsigned int, 8> tensor_idx; /**< tensor indices for state/temps */

  /** Layer properties */
  unsigned int hidden_size;
  unsigned int num_v_heads;
  unsigned int num_k_heads;
  unsigned int head_k_dim;
  unsigned int head_v_dim;
  unsigned int key_dim;
  unsigned int value_dim;
  unsigned int conv_dim;
  unsigned int conv_kernel_size;
  float norm_eps;

  std::tuple<props::NumHeads, nntrainer::props::Epsilon, props::KeyHeadDim,
             props::ValueHeadDim, props::NumKeyHeads, props::ConvKernelSize>
    gdn_props;
};

} // namespace causallm

#endif /* __GATED_DELTA_NET_LAYER_H__ */

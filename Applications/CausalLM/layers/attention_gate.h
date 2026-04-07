// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   attention_gate.h
 * @date   07 April 2026
 * @brief  Sigmoid gating layer for attention output (Qwen3.5)
 * @see    https://github.com/nntrainer/nntrainer
 * @author NNTrainer Authors
 * @bug    No known bugs except for NYI items
 * @note   Implements: output = attn_output * sigmoid(gate)
 *         Used in Qwen3.5 full attention layers where Q projection
 *         outputs 2x size and the second half is used as a sigmoid gate.
 *
 *         Input[0]: attention output [batch, 1, seq, num_heads * head_dim]
 *         Input[1]: gate values     [batch, 1, seq, num_heads * head_dim]
 *         Output:   gated output    [batch, 1, seq, num_heads * head_dim]
 */

#ifndef __ATTENTION_GATE_LAYER_H__
#define __ATTENTION_GATE_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace causallm {

/**
 * @brief Sigmoid gate layer: output = input[0] * sigmoid(input[1])
 */
WIN_EXPORT class AttentionGateLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT AttentionGateLayer() : Layer() {}
  WIN_EXPORT ~AttentionGateLayer() {}

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
    return AttentionGateLayer::type;
  };

  WIN_EXPORT void
  setProperty(const std::vector<std::string> &values) override {
    NNTR_THROW_IF(!values.empty(), std::invalid_argument)
      << "[attention_gate] Unknown Layer Properties count " +
           std::to_string(values.size());
  };

  inline static const std::string type = "attention_gate";
};

} // namespace causallm

#endif /* __ATTENTION_GATE_LAYER_H__ */

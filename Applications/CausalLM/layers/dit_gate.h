// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dit_gate.h
 * @date   16 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Gated residual for the Qwen2.5-Omni Token2Wav DiT.
 *
 * out = residual + gate * x,  where `gate` is a 1024-wide column slice of
 * input 2 (the AdaLN linear output) at gate_off, broadcast over the sequence
 * (HF: h = h + gate_msa.unsqueeze(1) * attn, modeling:3038/3050). The
 * operation layers don't broadcast, hence this custom layer.
 *
 * Per-block: gate_off=2048 (gate_msa), gate_off=5120 (gate_mlp).
 *
 * Inputs: 0 = residual [B,1,seq,C], 1 = x [B,1,seq,C], 2 = cond [B,1,1,M]
 */

#ifndef __DIT_GATE_LAYER_H__
#define __DIT_GATE_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

namespace causallm {

/**
 * @brief Gated residual layer (residual + broadcast(gate) * x)
 */
WIN_EXPORT class DiTGateLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT DiTGateLayer() : Layer(), gate_off(0) {}
  WIN_EXPORT ~DiTGateLayer() {}

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
    return DiTGateLayer::type;
  };

  WIN_EXPORT void
  setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "dit_gate";

private:
  unsigned int gate_off;
};

} // namespace causallm

#endif /* __DIT_GATE_LAYER_H__ */

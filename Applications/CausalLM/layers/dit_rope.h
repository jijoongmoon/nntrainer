// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dit_rope.h
 * @date   16 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Adjacent-pair rotary embedding for the Qwen2.5-Omni Token2Wav DiT.
 *
 * The DiT applies RoPE to HEAD 0 ONLY (HF quirk, modeling:2965) using an
 * ADJACENT-PAIR rotation with INTERLEAVED cos/sin (modeling:2469-2470,
 * 2910-2916) — NOT the half-split convention of mrope_apply/vision_rope, so
 * neither can be reused.
 *
 * For the first head_dim channels (head 0):
 *   rotate_half(x)[2j] = -x[2j+1],  rotate_half(x)[2j+1] = x[2j]
 *   out = x*cos + rotate_half(x)*sin
 * cos/sin are interleaved-duplicate [..,f0,f0,f1,f1,..] over head_dim. All
 * other heads (1..num_heads-1) pass through unchanged.
 *
 * Inputs: 0 = x [B,1,seq,num_heads*head_dim], 1 = cos, 2 = sin
 *   (cos/sin [B,1,seq,head_dim] or [1,1,seq,head_dim], host-filled).
 */

#ifndef __DIT_ROPE_LAYER_H__
#define __DIT_ROPE_LAYER_H__

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
 * @brief Adjacent-pair head-0-only RoPE layer (DiT)
 */
WIN_EXPORT class DiTRoPELayer final : public nntrainer::Layer {
public:
  WIN_EXPORT DiTRoPELayer() : Layer() {}
  WIN_EXPORT ~DiTRoPELayer() {}

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
    return DiTRoPELayer::type;
  };

  WIN_EXPORT void
  setProperty(const std::vector<std::string> &values) override {}

  inline static const std::string type = "dit_rope";
};

} // namespace causallm

#endif /* __DIT_ROPE_LAYER_H__ */

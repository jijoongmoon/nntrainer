// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   mrope_apply.h
 * @date   13 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Apply precomputed rotary embedding (cos/sin) to a q or k tensor.
 *
 * Generalizes 1-D RoPE to Qwen2.5-Omni M-RoPE by delegating the position ->
 * angle mapping to the host: the host fills shared cos/sin side inputs
 * [B,1,max_seq,head_dim] (built from the 3D t/h/w position ids and
 * mrope_section), and this layer applies, per head,
 *   out = x*cos[gpos] + rotate_half(x)*sin[gpos],   gpos = from + row
 * so mha_core can run with rope_theta=0 (it then caches the already-rotated
 * key, leaving the core attention untouched for every other model).
 *
 * Inputs: 0 = x [B,1,seq,num_heads*head_dim], 1 = cos, 2 = sin
 * (cos/sin [B,1,max_seq,head_dim]; head_dim and num_heads are derived).
 */

#ifndef __MROPE_APPLY_LAYER_H__
#define __MROPE_APPLY_LAYER_H__

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
 * @brief M-RoPE apply layer (precomputed cos/sin from host)
 */
WIN_EXPORT class MRoPEApplyLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT MRoPEApplyLayer() : Layer() {}
  WIN_EXPORT ~MRoPEApplyLayer() {}

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
    return MRoPEApplyLayer::type;
  };

  WIN_EXPORT void
  setProperty(const std::vector<std::string> &values) override {}

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "mrope_apply";
};

} // namespace causallm

#endif /* __MROPE_APPLY_LAYER_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   vision_rope.h
 * @date   13 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Qwen2.5 vision 2D rotary position embedding (applied to q or k).
 *
 * Port of apply_rotary_pos_emb_vision + rot_pos_emb for a FIXED patch grid.
 * The per-patch (h,w) positions follow the spatial-merge-blocked patch order
 * the image processor emits; rotary dim = head_dim/2 split into an h-half and
 * a w-half, each using inv_freq over head_dim/4 frequencies. cos/sin tables
 * are computed once in finalize() from the grid properties (no baked weights).
 *
 * Input/output: [B, 1, seq, num_heads*head_dim]. For each head h and patch s
 *   out = x*cos[s] + rotate_half(x)*sin[s]
 * where cos[s]/sin[s] are head_dim-wide (the head_dim/2 freqs duplicated).
 */

#ifndef __VISION_ROPE_LAYER_H__
#define __VISION_ROPE_LAYER_H__

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
 * @brief Qwen2.5 vision 2D-RoPE layer (fixed grid)
 */
WIN_EXPORT class VisionRopeLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT VisionRopeLayer() :
    Layer(), grid_h(0), grid_w(0), grid_t(1), num_heads(0), head_dim(0),
    spatial_merge_size(2), rope_theta(10000.0f) {}

  WIN_EXPORT ~VisionRopeLayer() {}

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
    return VisionRopeLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   * @note grid_h, grid_w (raw patch grid), num_heads, head_dim,
   *       spatial_merge_size, rope_theta
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "vision_rope";

private:
  unsigned int grid_h, grid_w, grid_t, num_heads, head_dim, spatial_merge_size;
  float rope_theta;
  std::vector<float> cos_tbl, sin_tbl; /**< [seq * head_dim] */

  /** @brief build cos/sin tables from the grid (called in finalize) */
  void buildTables();
};

} // namespace causallm

#endif /* __VISION_ROPE_LAYER_H__ */

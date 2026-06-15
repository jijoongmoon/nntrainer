// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   scale_layer.h
 * @date   15 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Elementwise multiply-by-constant layer (no weights).
 *
 * out = scale * in. Used by the Qwen2.5-Omni BigVGAN to average the 3 parallel
 * AMPBlocks per upsample stage (addition of 3 -> scale 1/3); the operation
 * layers carry no scalar constant, so this fills that gap. Stateless, no
 * weights, so it does not affect the DFS-from-output weight load order.
 *
 *   property: scale=<float>  (default 1.0)
 *   Input/Output: 0 = x  (any shape, preserved)
 */

#ifndef __SCALE_LAYER_H__
#define __SCALE_LAYER_H__

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
 * @brief Elementwise multiply-by-constant layer
 */
WIN_EXPORT class ScaleLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT ScaleLayer() : Layer(), scale_value(1.0f) {}
  WIN_EXPORT ~ScaleLayer() {}

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
    return ScaleLayer::type;
  };

  WIN_EXPORT void
  setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "scale";

private:
  float scale_value;
};

} // namespace causallm

#endif /* __SCALE_LAYER_H__ */

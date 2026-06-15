// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   snake_beta.h
 * @date   15 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Snake-beta activation used by the Qwen2.5-Omni Token2Wav BigVGAN.
 *
 * Per-channel, parametric, stateless elementwise activation (BigVGAN /
 * "anti-aliased" snake). For channel c with learned log-domain parameters
 * alpha[c], beta[c]:
 *   a = exp(alpha[c]);  b = exp(beta[c])
 *   out = x + (1 / (b + 1e-9)) * sin(a * x)^2
 * matching HF SnakeBeta (modeling_qwen2_5_omni.py:3077-3091): the exp() is
 * applied unconditionally (no alpha_logscale flag), 1e-9 is added to exp(beta)
 * only, and there is no clamp. alpha/beta are stored raw (log-domain) in the
 * checkpoint and must NOT be pre-exp'd by the converter.
 *
 * In HF this op is always the inner activation of TorchActivation1d
 * (UpSample x2 -> SnakeBeta -> DownSample x2); this layer is just the inner
 * elementwise piece. The Kaiser-sinc up/down sampling lives in a separate
 * antialiased_snake layer that sandwiches this one.
 *
 * Input:  0 = x [B, C, 1, T]   (conv NCHW; channel axis 1)
 * Weights: alpha [C], beta [C]  (1-D, FP32, requested in that order so the
 *          DFS-from-output bin loads alpha then beta).
 */

#ifndef __SNAKE_BETA_LAYER_H__
#define __SNAKE_BETA_LAYER_H__

#include <array>

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
 * @brief Snake-beta activation layer (per-channel, log-domain alpha/beta)
 */
WIN_EXPORT class SnakeBetaLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT SnakeBetaLayer() : Layer() {}
  WIN_EXPORT ~SnakeBetaLayer() {}

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
    return SnakeBetaLayer::type;
  };

  WIN_EXPORT void
  setProperty(const std::vector<std::string> &values) override {}

  inline static const std::string type = "snake_beta";

private:
  std::array<unsigned int, 2> wt_idx; /**< 0 = alpha, 1 = beta */
};

} // namespace causallm

#endif /* __SNAKE_BETA_LAYER_H__ */

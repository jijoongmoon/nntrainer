// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   antialiased_snake.h
 * @date   15 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Anti-aliased snake-beta activation (HF TorchActivation1d) for the
 *         Qwen2.5-Omni Token2Wav BigVGAN.
 *
 * Wraps the per-channel SnakeBeta in a 2x Kaiser-sinc up/down anti-alias:
 *   y = DownSample1d( SnakeBeta( UpSample1d(x) ) )
 * (HF modeling_qwen2_5_omni.py:3190-3211). For ratio=2, kernel=12 both the up
 * and down depthwise filters are the SAME 12-tap symmetric Kaiser-sinc kernel
 * (cutoff 0.25, half_width 0.3); since HF broadcasts one [1,1,12] filter across
 * all channels (groups=C), the depthwise conv is just a per-channel FIR with a
 * fixed kernel, implemented directly here (no nntrainer grouped-conv needed).
 *
 *   UpSample1d:   replicate-pad(5,5) -> convT(stride2, k12) -> *2 -> [15:-15]  (T -> 2T)
 *   SnakeBeta:    x + (1/(exp(beta)+1e-9)) * sin(exp(alpha)*x)^2               (per-channel)
 *   DownSample1d: replicate-pad(5,6) -> conv(stride2, k12)                     (2T -> T)
 *
 * Input:  0 = x [B, C, 1, T]
 * Output: 0 = y [B, C, 1, T]   (T preserved)
 * Weights: alpha [C], beta [C]  (the inner SnakeBeta; requested alpha then beta).
 */

#ifndef __ANTIALIASED_SNAKE_LAYER_H__
#define __ANTIALIASED_SNAKE_LAYER_H__

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
 * @brief Anti-aliased snake-beta activation layer (HF TorchActivation1d)
 */
WIN_EXPORT class AntialiasedSnakeLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT AntialiasedSnakeLayer() : Layer() {}
  WIN_EXPORT ~AntialiasedSnakeLayer() {}

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
    return AntialiasedSnakeLayer::type;
  };

  WIN_EXPORT void
  setProperty(const std::vector<std::string> &values) override {}

  inline static const std::string type = "antialiased_snake";

private:
  std::array<unsigned int, 2> wt_idx; /**< 0 = alpha, 1 = beta */
};

} // namespace causallm

#endif /* __ANTIALIASED_SNAKE_LAYER_H__ */

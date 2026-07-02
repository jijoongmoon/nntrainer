// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dit_modulate.h
 * @date   16 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  AdaLN modulation for the Qwen2.5-Omni Token2Wav DiT.
 *
 * out = LayerNorm_noaffine(x) * (1 + scale) + shift,  per HF AdaLayerNormZero
 * (modeling:2842) / _Final (modeling:2861). The no-affine LayerNorm (eps=1e-6,
 * no gamma/beta) cannot use nntrainer's layer_normalization (forces affine).
 * `scale` and `shift` are 1024-wide column slices of input 1 (the AdaLN
 * linear output, SiLU(time)->Linear), selected by scale_off / shift_off; they
 * are [B,1,1,C] and broadcast over the sequence (broadcasting the operation
 * layers don't support, hence this custom layer).
 *
 * Per-block (attn): scale_off=1024, shift_off=0  (order shift_msa,scale_msa,..)
 * Per-block (ff):   scale_off=4096, shift_off=3072
 * Final norm_out:   scale_off=0,    shift_off=1024 (order scale,shift)
 *
 * Inputs: 0 = x [B,1,seq,C], 1 = cond [B,1,1,M]   (M >= max(scale_off,shift_off)+C)
 */

#ifndef __DIT_MODULATE_LAYER_H__
#define __DIT_MODULATE_LAYER_H__

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
 * @brief AdaLN modulation layer (no-affine LN + scale/shift from a cond slice)
 */
WIN_EXPORT class DiTModulateLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT DiTModulateLayer() : Layer(), scale_off(0), shift_off(0) {}
  WIN_EXPORT ~DiTModulateLayer() {}

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
    return DiTModulateLayer::type;
  };

  WIN_EXPORT void
  setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "dit_modulate";

private:
  unsigned int scale_off;
  unsigned int shift_off;
};

} // namespace causallm

#endif /* __DIT_MODULATE_LAYER_H__ */

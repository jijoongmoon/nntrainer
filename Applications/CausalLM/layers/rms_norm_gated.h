// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   rms_norm_gated.h
 * @date   07 April 2026
 * @brief  RMS Normalization with SiLU gating for Qwen3.5
 * @see    https://github.com/nntrainer/nntrainer
 * @author NNTrainer Authors
 * @bug    No known bugs except for NYI items
 * @note   Computes: RMSNorm(x) * SiLU(gate)
 *         Used inside GatedDeltaNet for output normalization.
 */

#ifndef __RMS_NORM_GATED_LAYER_H__
#define __RMS_NORM_GATED_LAYER_H__

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
 * @brief RMS Normalization with SiLU gating
 *
 * Takes two inputs:
 *   input[0]: hidden_states to normalize
 *   input[1]: gate values (for SiLU gating)
 *
 * Weight:
 *   [0] gamma (weight) [feature_size]
 *
 * Output = RMSNorm(input[0]) * weight * SiLU(input[1])
 */
WIN_EXPORT class RMSNormGatedLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT RMSNormGatedLayer() : Layer(), wt_idx({0}) {}
  WIN_EXPORT ~RMSNormGatedLayer() {}

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
    return RMSNormGatedLayer::type;
  };

  WIN_EXPORT void
  setProperty(const std::vector<std::string> &values) override {
    auto remain_props = loadProperties(values, rms_gated_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[rms_norm_gated] Unknown Layer Properties count " +
           std::to_string(values.size());
  };

  inline static const std::string type = "rms_norm_gated";

private:
  std::array<unsigned int, 1> wt_idx;
  std::tuple<props::RMS_NORM_GAMMA_INIT, nntrainer::props::Epsilon> rms_gated_props;
};

} // namespace causallm

#endif /* __RMS_NORM_GATED_LAYER_H__ */

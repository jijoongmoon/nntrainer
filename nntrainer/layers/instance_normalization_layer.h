// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 NNTrainer contributors
 *
 * @file   instance_normalization_layer.h
 * @date   15 Mar 2025
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1607.08022
 * @brief  Instance Normalization Layer Class for Neural Network
 *
 */

#ifndef __INSTANCE_NORMALIZATION_LAYER_H__
#define __INSTANCE_NORMALIZATION_LAYER_H__
#ifdef __cplusplus

#include <array>
#include <vector>

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   InstanceNormalizationLayer
 * @brief   Instance Normalization Layer
 *
 * Normalizes each (B, C) instance independently over spatial dims (H, W).
 * Equivalent to GroupNorm with num_groups == num_channels.
 */
class InstanceNormalizationLayer : public Layer {
public:
  InstanceNormalizationLayer();
  ~InstanceNormalizationLayer() {}

  InstanceNormalizationLayer(InstanceNormalizationLayer &&rhs) noexcept =
    default;
  InstanceNormalizationLayer &
  operator=(InstanceNormalizationLayer &&rhs) = default;

  void finalize(InitLayerContext &context) override;
  void forwarding(RunLayerContext &context, bool training) override;
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;
  void calcDerivative(RunLayerContext &context) override;
  void calcGradient(RunLayerContext &context) override;

  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  const std::string getType() const override {
    return InstanceNormalizationLayer::type;
  };

  bool supportBackwarding() const override { return true; }

  using Layer::setProperty;
  void setProperty(const std::vector<std::string> &values) override;

  void setBatch(RunLayerContext &context, unsigned int batch) override;

  static constexpr const char *type = "instance_normalization";

private:
  std::array<unsigned int, 5> wt_idx;
  std::tuple<props::Epsilon, props::GammaInitializer, props::BetaInitializer,
             props::WeightDecay, props::BiasDecay>
    in_props;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __INSTANCE_NORMALIZATION_LAYER_H__ */

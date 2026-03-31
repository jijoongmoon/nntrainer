// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 NNTrainer contributors
 *
 * @file   group_normalization_layer.h
 * @date   15 Mar 2025
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1803.08494
 * @brief  Group Normalization Layer Class for Neural Network
 *
 */

#ifndef __GROUP_NORMALIZATION_LAYER_H__
#define __GROUP_NORMALIZATION_LAYER_H__
#ifdef __cplusplus

#include <array>
#include <vector>

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   GroupNormalizationLayer
 * @brief   Group Normalization Layer
 *
 * Groups channels into num_groups and normalizes within each group.
 * Input shape: [B, C, H, W] where C must be divisible by num_groups.
 */
class GroupNormalizationLayer : public Layer {
public:
  GroupNormalizationLayer();
  ~GroupNormalizationLayer() {}

  GroupNormalizationLayer(GroupNormalizationLayer &&rhs) noexcept = default;
  GroupNormalizationLayer &operator=(GroupNormalizationLayer &&rhs) = default;

  void finalize(InitLayerContext &context) override;
  void forwarding(RunLayerContext &context, bool training) override;
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;
  void calcDerivative(RunLayerContext &context) override;
  void calcGradient(RunLayerContext &context) override;

  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  const std::string getType() const override {
    return GroupNormalizationLayer::type;
  };

  bool supportBackwarding() const override { return true; }

  using Layer::setProperty;
  void setProperty(const std::vector<std::string> &values) override;

  void setBatch(RunLayerContext &context, unsigned int batch) override;

  static constexpr const char *type = "group_normalization";

private:
  unsigned int num_groups;

  std::array<unsigned int, 5> wt_idx;
  std::tuple<props::Epsilon, props::GammaInitializer, props::BetaInitializer,
             props::WeightDecay, props::BiasDecay>
    gn_props;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __GROUP_NORMALIZATION_LAYER_H__ */

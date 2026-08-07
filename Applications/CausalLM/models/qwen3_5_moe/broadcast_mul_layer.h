// SPDX-License-Identifier: Apache-2.0
/**
 * @file   broadcast_mul_layer.h
 * @date   2 July 2026
 * @brief  Row-broadcast multiply: out[b,c,h,:] = in0[b,c,h,:] * in1[b,c,h,0]
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude Code
 * @bug    No known bugs except for NYI items
 *
 * The core 'multiply' op layer requires identical input dims in
 * incremental_forwarding, so a per-token scalar gate (e.g. the qwen3_5_moe
 * shared-expert gate sigmoid(Linear(hidden->1))) cannot be applied with it
 * without replicating the gate weight to full hidden width. This layer takes
 * {a [B,C,S,W], g [B,C,S,1]} and broadcasts g along the width.
 */

#ifndef __BROADCAST_MUL_LAYER_H__
#define __BROADCAST_MUL_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

namespace causallm {

/**
 * @brief Row-broadcast multiply layer (width-1 second input).
 */
WIN_EXPORT class BroadcastMulLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new BroadcastMul layer object
   */
  WIN_EXPORT BroadcastMulLayer() : Layer() {}

  /**
   * @brief Destroy the BroadcastMul layer object
   */
  WIN_EXPORT ~BroadcastMulLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  WIN_EXPORT bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return BroadcastMulLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override {
    NNTR_THROW_IF(!values.empty(), std::invalid_argument)
      << "[broadcast_mul] Unknown Layer Properties count " +
           std::to_string(values.size());
  };

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "broadcast_mul";
};

} // namespace causallm

#endif /* __BROADCAST_MUL_LAYER_H__ */

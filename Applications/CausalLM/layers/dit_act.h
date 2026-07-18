// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dit_act.h
 * @date   18 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Elementwise activation with a CUDA-stream drain for the DiT.
 *
 * Same math as the core activation layer (fn=tanh_gelu routes to
 * nntrainer::tanh_gelu, fn=swish is x*sigmoid(x)), plus a
 * cuda::drain_if_async() before reading: in the DiT's mixed cpu/cuda graph
 * under NNTR_CUDA_ASYNC=1 this host op may otherwise read an FC output whose
 * cuBLAS kernel is still in flight. The core activation layer has no such
 * hook, hence this app-side twin (used only between engine=cuda FCs).
 *
 * Inputs: 0 = x. Props: fn=tanh_gelu|swish.
 */

#ifndef __DIT_ACT_LAYER_H__
#define __DIT_ACT_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <string>

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

namespace causallm {

/**
 * @brief DiT activation layer (host math + CUDA drain)
 */
WIN_EXPORT class DiTActLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT DiTActLayer() : Layer(), fn("tanh_gelu") {}
  WIN_EXPORT ~DiTActLayer() {}

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
    return DiTActLayer::type;
  };

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "dit_act";

private:
  std::string fn;
};

} // namespace causallm

#endif /* __DIT_ACT_LAYER_H__ */

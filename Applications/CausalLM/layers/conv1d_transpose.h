// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv1d_transpose.h
 * @date   15 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Convolution 1D transpose (deconvolution) layer.
 *
 * Thin wrapper that delegates to the core Conv2DTransposeLayer with the height
 * axis pinned to 1, exactly as nntrainer's Conv1DLayer wraps Conv2DLayer. Used
 * by the Qwen2.5-Omni Token2Wav BigVGAN upsampler stages (ConvTranspose1d).
 *
 * Input:  0 = x [B, in_ch, 1, T]
 * Output: 0 = y [B, filters, 1, (T-1)*stride + dilation*(k-1)+1 - pad_l - pad_r]
 *
 * The converter must hand the PyTorch ConvTranspose1d weight [in, out, k] as
 * nntrainer's [out, in, 1, k] (transpose dims (0,1), then unsqueeze the height
 * axis); the core Conv2DTransposeLayer requests its kernel as [out,in,kh,kw].
 */

#ifndef __CONV1D_TRANSPOSE_LAYER_H__
#define __CONV1D_TRANSPOSE_LAYER_H__

#include <array>
#include <memory>
#include <tuple>

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <layer_impl.h>
#include <node_exporter.h>

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

namespace nntrainer {
class Conv2DTransposeLayer;
}

namespace causallm {

/**
 * @brief Convolution 1D transpose layer (wraps core Conv2DTransposeLayer)
 */
WIN_EXPORT class Conv1DTransposeLayer final : public nntrainer::LayerImpl {
public:
  WIN_EXPORT Conv1DTransposeLayer();
  WIN_EXPORT ~Conv1DTransposeLayer();

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  WIN_EXPORT const std::string getType() const override {
    return Conv1DTransposeLayer::type;
  };

  WIN_EXPORT bool supportBackwarding() const override { return true; }

  using nntrainer::Layer::setProperty;

  WIN_EXPORT void
  setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "conv1d_transpose";

private:
  std::tuple<nntrainer::props::FilterSize, nntrainer::props::KernelSize,
             nntrainer::props::Stride, nntrainer::props::Padding1D,
             nntrainer::props::Dilation>
    conv_props;

  std::unique_ptr<nntrainer::Conv2DTransposeLayer> conv2d_transpose_layer;
};

} // namespace causallm

#endif /* __CONV1D_TRANSPOSE_LAYER_H__ */

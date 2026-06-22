// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cuda_geglu_layer.h
 * @date   22 Jun 2026
 * @brief  GeGLU activation for the CUDA backend: out = gelu_tanh(gate) * up.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details geglu has no CPU/host class in the tree (only the OpenCL
 * GeGLULayerCl), but gemma4 builds it with engine=cuda. This is a minimal
 * host-on-UVM GeGLU (two inputs {gate, up} -> gelu_tanh(gate) * up) that runs
 * directly on the cudaMallocManaged (host-coherent) tensors. A GPU kernel is a
 * later optimization. Mirrors GeGLULayerCl's interface (Print/SkipPrefill
 * props, type "geglu").
 */

#ifndef __CUDA_GEGLU_LAYER_H__
#define __CUDA_GEGLU_LAYER_H__
#ifdef __cplusplus

#include <tuple>

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class CudaGeGLULayer
 * @brief GeGLU (gelu_tanh(gate) * up) on the CUDA backend (host-on-UVM).
 */
class CudaGeGLULayer final : public Layer {
public:
  /**
   * @brief Construct a new CudaGeGLULayer object
   */
  CudaGeGLULayer() : Layer(), geglu_props(props::Print(), props::SkipPrefill()) {}

  /**
   * @brief Destroy the CudaGeGLULayer object
   */
  ~CudaGeGLULayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return CudaGeGLULayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "geglu";

private:
  bool skip_prefill = false; /**< skip compute during prefill (Gemma4 KV-share) */
  std::tuple<props::Print, props::SkipPrefill> geglu_props;

  /**
   * @brief out = gelu_tanh(in1) * in2 over the first `rows` rows (host-on-UVM)
   */
  void gegluProcess(const Tensor &in1, const Tensor &in2, Tensor &out,
                    unsigned int rows);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CUDA_GEGLU_LAYER_H__ */

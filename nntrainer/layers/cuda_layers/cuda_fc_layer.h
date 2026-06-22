// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cuda_fc_layer.h
 * @date   22 Jun 2026
 * @brief  Fully Connected Layer for the NVIDIA CUDA backend (engine=cuda).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details Mirror of FullyConnectedLayerCl, but inherits LayerImpl directly
 * (LayerImplCl is an OpenCL-only thin wrapper). finalize() is identical to the
 * CL/CPU FC. forwarding() is a correctness floor: FP32 activation+weight go to
 * cuBLAS SGEMM on the UVM (cudaMallocManaged) data pointers; every other dtype
 * (FP16 / QINT4 / Q4_x / Q6_K) falls back to the host Tensor::dot, which is
 * correct on the host-coherent managed memory. A GPU dequant + dp4a path lands
 * in P3.
 */

#ifndef __CUDA_FC_LAYER_H__
#define __CUDA_FC_LAYER_H__
#ifdef __cplusplus

#include <array>
#include <tuple>

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   CudaFcLayer
 * @brief   fully connected layer (CUDA backend)
 */
class CudaFcLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  CudaFcLayer();

  /**
   * @brief     Destructor of Fully Connected Layer
   */
  ~CudaFcLayer() = default;

  /**
   *  @brief  Move constructor.
   */
  CudaFcLayer(CudaFcLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   */
  CudaFcLayer &operator=(CudaFcLayer &&rhs) = default;

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
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return CudaFcLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "fully_connected";

private:
  bool skip_prefill = false; /**< skip compute during prefill (Gemma4 KV-share) */
  std::tuple<props::Unit> fc_props; /**< fc layer properties : unit */
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CUDA_FC_LAYER_H__ */

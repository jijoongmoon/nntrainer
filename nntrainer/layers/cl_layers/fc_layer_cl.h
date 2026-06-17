// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   fc_layer_cl.h
 * @date   7 May 2024
 * @brief  This is Fully Connected Layer Class of Neural Network with OpenCl
 * implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __FC_LAYER_CL_H__
#define __FC_LAYER_CL_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl_cl.h>

namespace nntrainer {

/**
 * @class   FullyConnecedLayer
 * @brief   fully connected layer
 */
class FullyConnectedLayerCl : public LayerImplCl {
public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  FullyConnectedLayerCl();

  /**
   * @brief     Destructor of Fully Connected Layer
   */
  ~FullyConnectedLayerCl() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] FullyConnected &&
   */
  FullyConnectedLayerCl(FullyConnectedLayerCl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs FullyConnectedLayer to be moved.
   */
  FullyConnectedLayerCl &operator=(FullyConnectedLayerCl &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::read(std::ifstream &file, RunLayerContext &run_context,
   * ...)
   * @note after the base read, eagerly builds the v8c GPU weight entry
   *       (dotCl_v8c_prebuild_weight) so the first prefill does not pay the
   *       lazy per-weight nibble-permute + upload (~753ms across 182 FCs on
   *       Gemma2-2B). Skipped under FSU (the weight data may be streamed out
   *       again); no-op off the v8c path.
   */
  void read(std::ifstream &file, RunLayerContext &run_context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType defineWeightDataType, bool fsu,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1) override;

  /**
   * @copydoc Layer::read(ReadSource src, RunLayerContext &run_context, ...)
   */
  void read(ReadSource src, RunLayerContext &run_context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType defineWeightDataType, bool fsu,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1) override;

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
  const std::string getType() const override {
    return FullyConnectedLayerCl::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static bool registerClKernels([[maybe_unused]] ClContext &cl_context) {
    return true;
  };

  static constexpr const char *type = "fully_connected";

private:
  bool skip_prefill = false; /**< skip compute during prefill (Gemma4 KV-share) */
  std::tuple<props::Unit>
    fc_props; /**< fc layer properties : unit - number of output neurons */
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */

  static std::vector<ClContext::SharedPtrClKernel>
    layer_kernel_ptrs; /**< kernel list relevant with this layer */
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FC_LAYER_CL__ */

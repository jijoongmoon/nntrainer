// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   litert_graph.h
 * @date   08 Apr 2026
 * @brief  LiteRT-LM Graph Layer - executes entire LiteRT-LM model as a layer
 * @see    https://github.com/nntrainer/nntrainer
 * @author NNTrainer Contributors
 * @bug    No known bugs except for NYI items
 *
 * This layer wraps the entire LiteRT-LM inference engine as a single
 * nntrainer layer, following the same pattern as QNNGraph layer.
 * When forwarding() is called, it executes the complete LiteRT-LM
 * model (tokenization → prefill → decode → detokenization).
 */

#ifndef __LITERT_GRAPH_H__
#define __LITERT_GRAPH_H__

#include <string>
#include <variant>
#include <vector>

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class LiteRTGraph
 * @brief Layer that wraps entire LiteRT-LM model execution
 *
 * Similar to QNNGraph, this layer encapsulates the entire model
 * execution. LiteRT-LM handles tokenization, KV-cache, decode loop,
 * and sampling internally.
 */
class LiteRTGraph : public LayerImpl {
public:
  using BufferTypePtr =
      std::variant<std::monostate, uint8_t *, uint16_t *, float *>;

  LiteRTGraph();
  ~LiteRTGraph();

  inline static const std::string type = "litert_graph";

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return LiteRTGraph::type; }

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override {};

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ...)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @brief Read weights (no-op for LiteRT-LM, model is self-contained)
   */
  void read(std::ifstream &file, RunLayerContext &run_context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType defineWeightDataType, bool fsu = false,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1) override;

private:
  std::tuple<std::vector<props::TensorDimension>,
             std::vector<props::TensorDataType>, props::FilePath>
      graph_props;

  std::string model_path;
  bool is_engine_initialized;

  std::vector<TensorDim> t_dims;
  std::vector<unsigned int> tensor_idx;

  /// @todo Add LiteRT-LM engine and session handles when C++ API is integrated
  /// std::unique_ptr<litert::lm::Engine> engine_;
  /// std::unique_ptr<litert::lm::Engine::Session> session_;
};

} // namespace nntrainer

#endif /* __LITERT_GRAPH_H__ */

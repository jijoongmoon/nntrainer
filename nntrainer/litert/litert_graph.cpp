// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   litert_graph.cpp
 * @date   08 Apr 2026
 * @brief  LiteRT-LM Graph Layer implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author NNTrainer Contributors
 * @bug    No known bugs except for NYI items
 */

#include "litert_graph.h"
#include <chrono>
#include <iostream>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

LiteRTGraph::LiteRTGraph() :
  LayerImpl(),
  graph_props({}, {}, props::FilePath()),
  is_engine_initialized(false) {}

LiteRTGraph::~LiteRTGraph() {
  /// @todo Release LiteRT-LM engine/session handles
}

void LiteRTGraph::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, graph_props);
  LayerImpl::setProperty(remain_props);
}

void LiteRTGraph::finalize(InitLayerContext &context) {
  model_path = std::get<props::FilePath>(graph_props).get();

  auto &dims = std::get<std::vector<props::TensorDimension>>(graph_props);
  t_dims.assign(dims.begin(), dims.end());

  auto &t_dtype = std::get<std::vector<props::TensorDataType>>(graph_props);

  auto engine = context.getComputeEngineType();

  if (t_dtype.empty()) {
    t_dtype.resize(t_dims.size());
    for (auto &t : t_dims) {
      t_dtype.push_back(context.getActivationDataType());
    }
  }

  std::vector<TensorDim> out_dim;

  for (unsigned int i = 0; i < t_dims.size(); ++i) {
    t_dims[i].setFormat(context.getFormat());
    t_dims[i].setDataType(t_dtype[i]);
    out_dim.push_back(t_dims[i]);
  }

  context.setOutputDimensions(out_dim);
}

void LiteRTGraph::forwarding(RunLayerContext &context, bool training) {
  auto start = std::chrono::system_clock::now();

  /// @todo Implement LiteRT-LM inference execution
  /// Following the QNNGraph pattern:
  ///
  /// 1. Initialize engine if not done:
  ///    if (!is_engine_initialized) {
  ///      auto model_assets = ModelAssets::Create(model_path);
  ///      auto settings = EngineSettings::CreateDefault(
  ///          model_assets, litert::lm::Backend::GPU);
  ///      engine_ = EngineFactory::CreateAny(settings);
  ///      session_ = engine_->CreateSession(SessionConfig::CreateDefault());
  ///      is_engine_initialized = true;
  ///    }
  ///
  /// 2. Set up input tensors from context:
  ///    Tensor &input = context.getInput(0);
  ///    // Convert input tensor to LiteRT-LM input format
  ///
  /// 3. Execute inference:
  ///    auto responses = session_->GenerateContent({InputText(prompt)});
  ///    // OR for tensor-level:
  ///    session_->RunPrefill({input_data});
  ///    auto output = session_->RunDecode();
  ///
  /// 4. Copy results to output tensors:
  ///    Tensor &output = context.getOutput(0);
  ///    // Copy LiteRT-LM output to nntrainer tensor

  for (unsigned int i = 0; i < context.getNumOutputs(); ++i) {
    Tensor &input_ = context.getInput(i);
    Tensor &hidden_ = context.getOutput(i);
    if (!context.getInPlace()) {
      hidden_.copyData(input_);
    }
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  ml_logd("LiteRTGraph::forwarding elapsed: %f sec", elapsed_seconds.count());
}

void LiteRTGraph::read(std::ifstream &file, RunLayerContext &run_context,
                       bool opt_var, ml::train::ExecutionMode mode,
                       bool trainable, TensorDim::DataType defineWeightDataType,
                       bool fsu, size_t start_offset, bool read_from_offset,
                       int file_fd) {
  // No-op: LiteRT-LM model is self-contained in .litertlm file
}

void LiteRTGraph::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  exporter.saveResult(graph_props, method, this);
}

} // namespace nntrainer

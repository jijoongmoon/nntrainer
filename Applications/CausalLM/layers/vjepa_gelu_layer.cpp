// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   vjepa_gelu_layer.cpp
 * @date   22 May 2026
 * @brief  Token-parallel GELU activation (NEON gelu_v2 split over the pool).
 */

#include "vjepa_gelu_layer.h"

#include <cpu_backend.h>
#include <nntrainer_error.h>
#include <thread_manager.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void VjepaGeluLayer::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());
}

// gelu over [X, X+n) split across the ThreadManager pool (elementwise, so any
// split is valid). Matches the core activation's gelu_v2 exactly.
static void gelu_parallel(const float *X, float *Y, size_t n) {
  if (n == 0)
    return;
  auto &tm = nntrainer::ThreadManager::Global();
  const unsigned int nt = tm.getComputeThreadCount();
  if (nt <= 1 || n < 4096) {
    nntrainer::gelu_v2(static_cast<unsigned int>(n), X, Y);
    return;
  }
  tm.parallel_for(0, static_cast<size_t>(nt), [=](size_t t) {
    size_t s = (n * t) / nt;
    size_t e = (n * (t + 1)) / nt;
    if (e > s)
      nntrainer::gelu_v2(static_cast<unsigned int>(e - s), X + s, Y + s);
  });
}

void VjepaGeluLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  NNTR_THROW_IF(in.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[vjepa_gelu] only FP32 is supported";
  gelu_parallel(in.getData<float>(), out.getData<float>(), in.size());
}

void VjepaGeluLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  NNTR_THROW_IF(in.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[vjepa_gelu] only FP32 is supported";

  const nntrainer::TensorDim dim = in.getDim();
  const size_t width = dim.width();
  const size_t feature_len = dim.getFeatureLen();
  const size_t off = static_cast<size_t>(from) * width;
  const size_t n = static_cast<size_t>(to - from) * width;
  for (unsigned int b = 0; b < dim.batch(); ++b) {
    const float *x = in.getData<float>() + b * feature_len + off;
    float *y = out.getData<float>() + b * feature_len + off;
    gelu_parallel(x, y, n);
  }
}

void VjepaGeluLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("[vjepa_gelu] Training is not supported yet.");
}

void VjepaGeluLayer::setProperty(const std::vector<std::string> &values) {
  NNTR_THROW_IF(!values.empty(), std::invalid_argument)
    << "[vjepa_gelu] does not take properties";
}

void VjepaGeluLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

#ifdef PLUGGABLE
nntrainer::Layer *create_vjepa_gelu_layer() { return new VjepaGeluLayer(); }
void destroy_vjepa_gelu_layer(nntrainer::Layer *layer) { delete layer; }
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_vjepa_gelu_layer,
                                                   destroy_vjepa_gelu_layer};
}
#endif

} // namespace causallm

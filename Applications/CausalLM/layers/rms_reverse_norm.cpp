// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Joonseok Oh <jrock.oh@samsung.com>
 *
 * @file   rms_reverse_norm.cpp
 * @date   27 March 2026
 * @brief  This is Reverse RMS Norm Layer Class
 * @see    https://github.com/nntrainer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "rms_reverse_norm.h"

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_rmsnorm.h>
#include <cuda_stream_manager.h>
#endif

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum RMSReverseParams { weight, out_scale };

void RMSReverseNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  context.setOutputDimensions(dim);

  // Initialize weight and out_scale parameters
  auto weight_init = nntrainer::props::InitializerInfo::Enum::ONES;
  auto outscale_init = nntrainer::props::InitializerInfo::Enum::ONES;

  if (!std::get<props::RMS_REVERSE_NORM_WEIGHT_INIT>(rms_props).empty()) {
    weight_init =
      std::get<props::RMS_REVERSE_NORM_WEIGHT_INIT>(rms_props).get();
  }

  if (!std::get<props::RMS_REVERSE_NORM_OUTSCALE_INIT>(rms_props).empty()) {
    outscale_init =
      std::get<props::RMS_REVERSE_NORM_OUTSCALE_INIT>(rms_props).get();
  }

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty()) {
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();
  }

  // Request weight parameter (learnable multiplicative weight applied BEFORE
  // norm)
  nntrainer::TensorDim weight_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSReverseParams::weight] = context.requestWeight(
    weight_dim, weight_init, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
    "weight", true);

  // Request out_scale parameter (learnable scale applied AFTER norm)
  nntrainer::TensorDim outscale_dim(
    1, 1, 1, 1,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSReverseParams::out_scale] = context.requestWeight(
    outscale_dim, outscale_init, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
    "out_scale", true);
}

void RMSReverseNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {}

void RMSReverseNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &weight =
    context.getWeight(wt_idx[RMSReverseParams::weight]);
  nntrainer::Tensor &out_scale =
    context.getWeight(wt_idx[RMSReverseParams::out_scale]);

  ml::train::TensorDim in_dim = in.getDim();

  unsigned int step_size = to - from;
  bool is_prefill = !from || step_size > 1;
  if (skip_prefill && is_prefill)
    return;

  if (from) {
    // Normalize to 0-based while preserving step size for multi-token prefill
    to = to - from;
    from = 0;
  }

  // Whole-op dispatch (N4): the layer owns structure (step window, batch
  // walk, skip_prefill); ComputeOps owns the math. On a gpu graph this lands
  // in ClComputeOps::rms_reverse_norm (the SVM-gated GPU kernel, host-bounce
  // NAMED on a residency miss); on cpu/cuda it is the host FP32-temp math
  // that used to live in this body. No backend is named here and no raw
  // kernel wrapper is called from this Layer body.
  const unsigned int active_rows = (to - from) * in_dim.channel();
  const unsigned int rows_per_batch = in_dim.channel() * in_dim.height();
  nntrainer::ComputeOps *ops = in.getOps();
  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    ops->rms_reverse_norm(in, out, weight, out_scale, epsilon, active_rows,
                          b * rows_per_batch);
  }
}

void RMSReverseNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim output_dim =
    context.getOutput(SINGLE_INOUT_IDX).getDim();

  input_dim.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(SINGLE_INOUT_IDX, output_dim);
}

void RMSReverseNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // Training not implemented yet
  // std::throw_with_nested(std::runtime_error("Training is not supported
  // yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_reverse_norm_layer() {
  auto layer = new RMSReverseNormLayer();
  return layer;
}

void destroy_rms_reverse_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_rms_reverse_norm_layer, destroy_rms_reverse_norm_layer};
}

#endif

} // namespace causallm

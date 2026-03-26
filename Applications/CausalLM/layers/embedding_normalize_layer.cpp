// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   embedding_normalize_layer.cpp
 * @date   06 Jan 2026
 * @brief  This is Embedding Normalize Layer Class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <algorithm>
#include <cmath>
#include <embedding_normalize_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

EmbeddingNormalizeLayer::EmbeddingNormalizeLayer() : LayerImpl() {}

void EmbeddingNormalizeLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "EmbeddingNormalize layer takes only one input";

  const nntrainer::TensorDim &input_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];

  context.setOutputDimensions({input_dim});
}

void EmbeddingNormalizeLayer::forwarding(nntrainer::RunLayerContext &context,
                                         bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  // Copy input to output as we will modify output in-place
  output.copyData(input);
  // Normalize along the last dimension (dim=3)
  output.normalization_i(3);
}

void EmbeddingNormalizeLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  // Incremental forwarding for element-wise/row-wise normalization is typically
  // identical to forwarding if the input shape matches the processing chunk.
  // However, often incremental_forwarding is used when we process a chunk of
  // seq_len. BUT, EmbeddingNormalizeLayer usually comes AFTER Pooling, so
  // seq_len is likely 1. In that case, incremental_forwarding might not even be
  // called or acts same as forwarding. If we assume this layer is generic, we
  // should process 'from' to 'to'. But strictly, this layer is designed for
  // pooled output [batch, 1, 1, dim]. So 'from' and 'to' are likely 0 and 1.

  forwarding(context, training);
}

void EmbeddingNormalizeLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for EmbeddingNormalize layer is not supported");
}

void EmbeddingNormalizeLayer::calcGradient(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcGradient for EmbeddingNormalize layer is not supported");
}

void EmbeddingNormalizeLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
}

} // namespace causallm

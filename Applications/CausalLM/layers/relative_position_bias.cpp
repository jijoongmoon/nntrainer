// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   relative_position_bias.cpp
 * @date   24 March 2026
 * @brief  T5-style relative position bias layer implementation
 * @see    https://github.com/nnstreamer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#include <cmath>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <relative_position_bias.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

RelativePositionBiasLayer::RelativePositionBiasLayer() :
  LayerImpl(),
  rpb_props(nntrainer::props::NumHeads(), props::NumBuckets(),
            props::MaxDistance(), props::Bidirectional()),
  weight_idx(std::numeric_limits<unsigned>::max()),
  num_heads_(0),
  num_buckets_(0),
  max_distance_(0),
  bidirectional_(true) {}

void RelativePositionBiasLayer::finalize(
  nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "RelativePositionBias layer takes exactly one input (query tensor)";

  num_heads_ =
    std::get<nntrainer::props::NumHeads>(rpb_props).get();
  num_buckets_ = std::get<props::NumBuckets>(rpb_props).get();
  max_distance_ = std::get<props::MaxDistance>(rpb_props).get();
  bidirectional_ = std::get<props::Bidirectional>(rpb_props).get();

  NNTR_THROW_IF(num_heads_ == 0, std::invalid_argument)
    << "num_heads must be set for RelativePositionBias";

  const ml::train::TensorDim &input_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];
  const unsigned int batch_size = input_dim.batch();

  /** input query shape: (B, 1, seq_len, num_heads * head_dim)
   *  For the bias output we need seq_len to set the output dimension.
   *  In incremental mode seq_len may change, but finalize uses the max.
   */
  const unsigned int max_seq_len = input_dim.height();

  /** Output: (B, num_heads, seq_len, seq_len)
   *  NNTrainer 4D: batch=B, channel=num_heads, height=seq_len, width=seq_len
   */
  ml::train::TensorDim output_dim(
    batch_size, num_heads_, max_seq_len, max_seq_len,
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});

  /** Embedding weight: (1, 1, num_buckets, num_heads)
   *  Each bucket maps to a bias value per head.
   */
  ml::train::TensorDim weight_dim(
    1, 1, num_buckets_, num_heads_,
    {context.getFormat(), context.getWeightDataType()});

  weight_idx = context.requestWeight(
    weight_dim, nntrainer::Initializer::NONE,
    nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f,
    "relative_attention_bias", true);
}

int RelativePositionBiasLayer::relative_position_bucket(
  int relative_position) const {

  int ret = 0;
  int n;
  unsigned int effective_buckets = num_buckets_;

  if (bidirectional_) {
    effective_buckets /= 2;
    if (relative_position > 0) {
      ret += static_cast<int>(effective_buckets);
    }
    n = std::abs(relative_position);
  } else {
    n = std::max(-relative_position, 0);
  }

  /** Half the buckets are for exact increments in position */
  int max_exact = static_cast<int>(effective_buckets) / 2;
  if (n < max_exact) {
    return ret + n;
  }

  /** The other half of the buckets use logarithmic binning up to
   *  max_distance, clamped to [max_exact, effective_buckets - 1] */
  float log_ratio =
    std::log(static_cast<float>(n) / static_cast<float>(max_exact)) /
    std::log(static_cast<float>(max_distance_) /
             static_cast<float>(max_exact));

  int val_if_large = max_exact + static_cast<int>(
    log_ratio * static_cast<float>(effective_buckets - max_exact));
  val_if_large =
    std::min(val_if_large, static_cast<int>(effective_buckets) - 1);

  return ret + val_if_large;
}

void RelativePositionBiasLayer::compute_bias(
  unsigned int query_length, unsigned int key_length,
  const float *embedding_data, float *output_data) const {

  /** output layout: (num_heads, query_length, key_length)
   *  output_data[h * query_length * key_length + q * key_length + k]
   *
   *  For each (q, k) pair:
   *    relative_position = k - q  (memory_position - context_position)
   *    bucket = relative_position_bucket(relative_position)
   *    output[h][q][k] = embedding[bucket][h]
   */
  for (unsigned int q = 0; q < query_length; ++q) {
    for (unsigned int k = 0; k < key_length; ++k) {
      int rel_pos = static_cast<int>(k) - static_cast<int>(q);
      int bucket = relative_position_bucket(rel_pos);

      /** embedding layout: (num_buckets, num_heads)
       *  embedding_data[bucket * num_heads_ + h] */
      for (unsigned int h = 0; h < num_heads_; ++h) {
        output_data[h * query_length * key_length + q * key_length + k] =
          embedding_data[bucket * num_heads_ + h];
      }
    }
  }
}

void RelativePositionBiasLayer::forwarding(
  nntrainer::RunLayerContext &context, bool training) {

  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &weight = context.getWeight(weight_idx);
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);

  const unsigned int batch_size = input.batch();
  const unsigned int seq_len = input.height();
  const float *emb_data = weight.getData<float>();
  float *out_data = output.getData<float>();

  /** Compute bias for one batch (position bias is the same across batches) */
  const size_t per_batch = num_heads_ * seq_len * seq_len;
  compute_bias(seq_len, seq_len, emb_data, out_data);

  /** Copy to remaining batches */
  for (unsigned int b = 1; b < batch_size; ++b) {
    std::memcpy(out_data + b * per_batch, out_data,
                per_batch * sizeof(float));
  }
}

void RelativePositionBiasLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &weight = context.getWeight(weight_idx);

  const unsigned int query_length = to - from;
  const unsigned int key_length = to;
  const float *emb_data = weight.getData<float>();
  float *out_data = output.getData<float>();

  /** For incremental mode the output shape from finalize may be larger
   *  than needed. We write only the (query_length × key_length) region
   *  per head for batch 0. The caller (mha_core) reads the correct region.
   *
   *  output layout: (B, num_heads, query_length, key_length)
   *  But we only fill batch 0 since position bias is batch-invariant.
   *  mha_core handles per-batch extraction.
   */
  const size_t per_batch = num_heads_ * query_length * key_length;

  /** Compute using offset: context_position = from + i for query positions */
  for (unsigned int q = 0; q < query_length; ++q) {
    unsigned int context_pos = from + q;
    for (unsigned int k = 0; k < key_length; ++k) {
      int rel_pos = static_cast<int>(k) - static_cast<int>(context_pos);
      int bucket = relative_position_bucket(rel_pos);

      for (unsigned int h = 0; h < num_heads_; ++h) {
        out_data[h * query_length * key_length + q * key_length + k] =
          emb_data[bucket * num_heads_ + h];
      }
    }
  }

  /** Copy to remaining batches */
  const unsigned int batch_size = output.batch();
  for (unsigned int b = 1; b < batch_size; ++b) {
    std::memcpy(out_data + b * per_batch, out_data,
                per_batch * sizeof(float));
  }
}

void RelativePositionBiasLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for RelativePositionBias layer is not supported");
}

void RelativePositionBiasLayer::calcGradient(
  nntrainer::RunLayerContext &context) {}

void RelativePositionBiasLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, rpb_props);
  LayerImpl::setProperty(remain_props);
}

void RelativePositionBiasLayer::exportTo(
  nntrainer::Exporter &exporter,
  const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(rpb_props, method, this);
}

} // namespace causallm

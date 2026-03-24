// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   relative_position_bias.h
 * @date   24 March 2026
 * @brief  T5-style relative position bias layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @bug    No known bugs except for NYI items
 *
 * Implements the relative position bias computation from T5/mT5 models.
 * Computes bucketed relative distances between query and key positions,
 * then looks up learned bias values from an embedding table.
 * Output shape: (B, num_heads, query_len, key_len)
 */

#ifndef __RELATIVE_POSITION_BIAS_LAYER_H__
#define __RELATIVE_POSITION_BIAS_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <base_properties.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace props {

/**
 * @brief Number of relative position buckets
 */
class NumBuckets : public nntrainer::PositiveIntegerProperty {
public:
  NumBuckets(unsigned int value = 32) { set(value); };
  static constexpr const char *key = "num_buckets";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief Maximum relative distance before logarithmic binning
 */
class MaxDistance : public nntrainer::PositiveIntegerProperty {
public:
  MaxDistance(unsigned int value = 128) { set(value); };
  static constexpr const char *key = "max_distance";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief Whether attention is bidirectional (encoder) or unidirectional
 * (decoder)
 */
class Bidirectional : public nntrainer::Property<bool> {
public:
  Bidirectional(bool value = true) { set(value); };
  static constexpr const char *key = "bidirectional";
  using prop_tag = nntrainer::bool_prop_tag;
};

} // namespace props

/**
 * @class RelativePositionBiasLayer
 * @brief Computes T5-style relative position bias for attention.
 *
 * This layer fuses the entire relative position bias computation:
 *   1. Compute relative positions between query and key positions
 *   2. Map relative positions to bucket indices
 *   3. Look up bias values from a learned embedding table
 *   4. Output bias tensor shaped (B, num_heads, query_len, key_len)
 *
 * Properties:
 *   - num_heads: Number of attention heads
 *   - num_buckets: Number of relative position buckets (default: 32)
 *   - max_distance: Maximum relative distance (default: 128)
 *   - bidirectional: true for encoder, false for decoder (default: true)
 *
 * Input: Query tensor (used only for shape: batch, seq_len)
 * Output: Position bias tensor (B, num_heads, query_len, key_len)
 * Weight: Embedding table of shape (1, 1, num_buckets, num_heads)
 */
WIN_EXPORT class RelativePositionBiasLayer : public nntrainer::LayerImpl {
public:
  WIN_EXPORT RelativePositionBiasLayer();
  WIN_EXPORT ~RelativePositionBiasLayer() = default;
  WIN_EXPORT RelativePositionBiasLayer(RelativePositionBiasLayer &&rhs) noexcept
    = default;
  WIN_EXPORT RelativePositionBiasLayer &
  operator=(RelativePositionBiasLayer &&rhs) = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                              bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;
  WIN_EXPORT const std::string getType() const override {
    return RelativePositionBiasLayer::type;
  };
  WIN_EXPORT bool supportBackwarding() const override { return false; }

  using Layer::setProperty;
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "relative_position_bias";

private:
  /**
   * @brief Compute relative position bucket index.
   *
   * Maps a relative position (memory_pos - context_pos) to a bucket index
   * using the T5 bucketing scheme: exact positions for small distances,
   * logarithmic binning for larger distances.
   */
  int relative_position_bucket(int relative_position) const;

  /**
   * @brief Compute bias for given query/key lengths and write to output.
   */
  void compute_bias(unsigned int query_length, unsigned int key_length,
                    const float *embedding_data, float *output_data) const;

  std::tuple<nntrainer::props::NumHeads, props::NumBuckets, props::MaxDistance,
             props::Bidirectional>
    rpb_props;

  unsigned int weight_idx;
  unsigned int num_heads_;
  unsigned int num_buckets_;
  unsigned int max_distance_;
  bool bidirectional_;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __RELATIVE_POSITION_BIAS_LAYER_H__ */

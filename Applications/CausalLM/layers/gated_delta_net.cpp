// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gated_delta_net.cpp
 * @date   07 April 2026
 * @brief  Gated Delta Net layer implementation for Qwen3.5
 * @see    https://github.com/nntrainer/nntrainer
 * @author NNTrainer Authors
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <iostream>
#include <vector>

#include "gated_delta_net.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum GDNWeights {
  in_proj_qkv = 0,
  conv1d_weight = 1,
  A_log = 2,
  in_proj_b = 3,
  in_proj_a = 4,
  dt_bias = 5,
  norm_weight = 6,
  out_proj = 7,
  in_proj_z = 8,
};

enum GDNTensors {
  conv_state = 0,
  recurrent_state = 1,
};

void GatedDeltaNetLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gdn_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[gated_delta_net] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void GatedDeltaNetLayer::finalize(nntrainer::InitLayerContext &context) {
  /// @todo Read properties from config and set dimensions
  /// For now, derive from the Qwen3.5-2B config:
  ///   hidden_size=2048, linear_key_head_dim=128, linear_value_head_dim=128,
  ///   linear_num_key_heads=16, linear_num_value_heads=16,
  ///   conv_kernel_size=4

  auto &num_heads_prop = std::get<props::NumHeads>(gdn_props);
  auto &epsilon_prop = std::get<nntrainer::props::Epsilon>(gdn_props);

  const auto &input_dims = context.getInputDimensions();
  hidden_size = input_dims[0].width();

  /// @todo make these configurable via properties
  num_v_heads = num_heads_prop.get();
  head_v_dim = hidden_size / num_v_heads;
  num_k_heads = num_v_heads;
  head_k_dim = head_v_dim;
  key_dim = head_k_dim * num_k_heads;
  value_dim = head_v_dim * num_v_heads;
  conv_dim = key_dim * 2 + value_dim;
  conv_kernel_size = 4;
  norm_eps = epsilon_prop.get();

  ml::train::TensorDim::TensorType weight_type = {
    context.getFormat(), context.getWeightDataType()};
  ml::train::TensorDim::TensorType activation_type = {
    context.getFormat(), context.getActivationDataType()};

  // Output dimension same as input
  std::vector<nntrainer::TensorDim> out_dims = input_dims;
  context.setOutputDimensions(out_dims);

  // [0] in_proj_qkv: [hidden_size, key_dim*2+value_dim]
  nntrainer::TensorDim qkv_dim(1, 1, hidden_size, conv_dim, weight_type);
  wt_idx[GDNWeights::in_proj_qkv] = context.requestWeight(
    qkv_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "in_proj_qkv", false);

  // [1] conv1d_weight: [conv_kernel_size, conv_dim] (depthwise)
  nntrainer::TensorDim conv_w_dim(1, 1, conv_kernel_size, conv_dim,
                                  weight_type);
  wt_idx[GDNWeights::conv1d_weight] = context.requestWeight(
    conv_w_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "conv1d_weight", false);

  // [2] A_log: [num_v_heads]
  nntrainer::TensorDim a_log_dim(1, 1, 1, num_v_heads, weight_type);
  wt_idx[GDNWeights::A_log] = context.requestWeight(
    a_log_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "A_log", false);

  // [3] in_proj_b: [hidden_size, num_v_heads] (beta projection)
  nntrainer::TensorDim proj_b_dim(1, 1, hidden_size, num_v_heads, weight_type);
  wt_idx[GDNWeights::in_proj_b] = context.requestWeight(
    proj_b_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "in_proj_b", false);

  // [4] in_proj_a: [hidden_size, num_v_heads] (decay projection)
  nntrainer::TensorDim proj_a_dim(1, 1, hidden_size, num_v_heads, weight_type);
  wt_idx[GDNWeights::in_proj_a] = context.requestWeight(
    proj_a_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "in_proj_a", false);

  // [5] dt_bias: [num_v_heads]
  nntrainer::TensorDim dt_bias_dim(1, 1, 1, num_v_heads, weight_type);
  wt_idx[GDNWeights::dt_bias] = context.requestWeight(
    dt_bias_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "dt_bias", false);

  // [6] norm_weight: [head_v_dim] (RMSNormGated gamma)
  nntrainer::TensorDim norm_dim(1, 1, 1, head_v_dim, weight_type);
  wt_idx[GDNWeights::norm_weight] = context.requestWeight(
    norm_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "norm_weight", false);

  // [7] out_proj: [value_dim, hidden_size]
  nntrainer::TensorDim out_proj_dim(1, 1, value_dim, hidden_size, weight_type);
  wt_idx[GDNWeights::out_proj] = context.requestWeight(
    out_proj_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "out_proj", false);

  // [8] in_proj_z: [hidden_size, value_dim] (gate projection)
  nntrainer::TensorDim proj_z_dim(1, 1, hidden_size, value_dim, weight_type);
  wt_idx[GDNWeights::in_proj_z] = context.requestWeight(
    proj_z_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "in_proj_z", false);

  // Request tensors for state caching
  // conv_state: [batch, 1, conv_dim, conv_kernel_size-1]
  unsigned int batch = input_dims[0].batch();
  nntrainer::TensorDim conv_state_dim(batch, 1, conv_dim,
                                      conv_kernel_size - 1, activation_type);
  /// @todo request persistent tensors for state caching
}

void GatedDeltaNetLayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {
  /// @note forwarding is used for prefill mode (chunk-based delta rule)
  /// @todo implement chunk_gated_delta_rule for prefill
}

void GatedDeltaNetLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &w_qkv = context.getWeight(wt_idx[GDNWeights::in_proj_qkv]);
  nntrainer::Tensor &w_conv = context.getWeight(wt_idx[GDNWeights::conv1d_weight]);
  nntrainer::Tensor &w_a_log = context.getWeight(wt_idx[GDNWeights::A_log]);
  nntrainer::Tensor &w_proj_b = context.getWeight(wt_idx[GDNWeights::in_proj_b]);
  nntrainer::Tensor &w_proj_a = context.getWeight(wt_idx[GDNWeights::in_proj_a]);
  nntrainer::Tensor &w_dt_bias = context.getWeight(wt_idx[GDNWeights::dt_bias]);
  nntrainer::Tensor &w_norm = context.getWeight(wt_idx[GDNWeights::norm_weight]);
  nntrainer::Tensor &w_out = context.getWeight(wt_idx[GDNWeights::out_proj]);
  nntrainer::Tensor &w_z = context.getWeight(wt_idx[GDNWeights::in_proj_z]);

  unsigned int batch_size = input.getDim().batch();
  unsigned int seq_len = to - from;

  /// @todo Implement the full incremental forwarding:
  ///
  /// 1. mixed_qkv = input @ in_proj_qkv           [batch, seq, conv_dim]
  /// 2. z = input @ in_proj_z                      [batch, seq, value_dim]
  /// 3. b = input @ in_proj_b                      [batch, seq, num_v_heads]
  /// 4. a = input @ in_proj_a                      [batch, seq, num_v_heads]
  ///
  /// 5. mixed_qkv = causal_conv1d(mixed_qkv, conv1d_weight, conv_state)
  ///    Apply SiLU activation
  ///
  /// 6. Split mixed_qkv → Q[key_dim], K[key_dim], V[value_dim]
  /// 7. Reshape Q, K, V to [batch, seq, num_heads, head_dim]
  /// 8. L2 normalize Q and K
  ///
  /// 9. beta = sigmoid(b)
  /// 10. g = -exp(A_log) * softplus(a + dt_bias)
  ///
  /// 11. For each token (recurrent mode):
  ///     state = state * exp(g_t)
  ///     kv_mem = (state * k_t).sum(-2)
  ///     delta = (v_t - kv_mem) * beta_t
  ///     state = state + k_t * delta
  ///     output_t = (state * q_t).sum(-2)
  ///
  /// 12. output = RMSNormGated(output, z, norm_weight)
  /// 13. output = output @ out_proj

  (void)w_qkv;
  (void)w_conv;
  (void)w_a_log;
  (void)w_proj_b;
  (void)w_proj_a;
  (void)w_dt_bias;
  (void)w_norm;
  (void)w_out;
  (void)w_z;
  (void)batch_size;
  (void)seq_len;
}

void GatedDeltaNetLayer::recurrentGatedDeltaRule(
  const nntrainer::Tensor &query, const nntrainer::Tensor &key,
  const nntrainer::Tensor &value, const nntrainer::Tensor &g,
  const nntrainer::Tensor &beta, nntrainer::Tensor &state,
  nntrainer::Tensor &output) {

  /// @todo Implement recurrent gated delta rule
  ///
  /// For each timestep t:
  ///   scale = 1 / sqrt(k_head_dim)
  ///   q_t = query[:, :, t] * scale
  ///   k_t = key[:, :, t]
  ///   v_t = value[:, :, t]
  ///   g_t = exp(g[:, :, t])  // [batch, num_heads, 1, 1]
  ///   beta_t = beta[:, :, t] // [batch, num_heads, 1]
  ///
  ///   // Decay state
  ///   state *= g_t
  ///
  ///   // Delta rule update
  ///   kv_mem = (state * k_t.unsqueeze(-1)).sum(-2)  // Read from state
  ///   delta = (v_t - kv_mem) * beta_t               // Error correction
  ///   state += k_t.unsqueeze(-1) * delta.unsqueeze(-2)  // Write to state
  ///
  ///   // Read output
  ///   output[:, :, t] = (state * q_t.unsqueeze(-1)).sum(-2)
}

void GatedDeltaNetLayer::chunkGatedDeltaRule(
  const nntrainer::Tensor &query, const nntrainer::Tensor &key,
  const nntrainer::Tensor &value, const nntrainer::Tensor &g,
  const nntrainer::Tensor &beta, nntrainer::Tensor &state,
  nntrainer::Tensor &output, unsigned int chunk_size) {

  /// @todo Implement chunk-based gated delta rule for prefill
  /// This is an optimization over the recurrent version for processing
  /// multiple tokens at once during prompt prefill.
}

void GatedDeltaNetLayer::applyCausalConv1d(
  const nntrainer::Tensor &input, const nntrainer::Tensor &weight,
  nntrainer::Tensor &state, nntrainer::Tensor &output,
  unsigned int seq_len) {

  /// @todo Implement causal depthwise conv1d
  ///
  /// For incremental (seq_len=1):
  ///   1. Concatenate [state, input] along time axis
  ///   2. Update state with latest kernel_size-1 values
  ///   3. Apply depthwise conv1d (each channel independently)
  ///   4. Apply SiLU activation
  ///
  /// For prefill (seq_len>1):
  ///   1. Left-pad input with state (or zeros)
  ///   2. Apply depthwise conv1d across full sequence
  ///   3. Apply SiLU activation
  ///   4. Save last kernel_size-1 values as new state
}

void GatedDeltaNetLayer::applyRMSNormGated(
  const nntrainer::Tensor &input, const nntrainer::Tensor &gate,
  const nntrainer::Tensor &gamma, nntrainer::Tensor &output, float eps) {

  /// @todo Implement RMSNormGated: RMSNorm(x) * gamma * SiLU(gate)
  ///
  /// 1. variance = mean(input^2, dim=-1)
  /// 2. normalized = input * rsqrt(variance + eps)
  /// 3. output = gamma * normalized * SiLU(gate)
}

void GatedDeltaNetLayer::l2Normalize(nntrainer::Tensor &tensor, float eps) {
  /// @todo L2 normalize along last dimension
  /// norm = sqrt(sum(x^2, dim=-1))
  /// x = x / max(norm, eps)
}

void GatedDeltaNetLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_gated_delta_net_layer() {
  auto layer = new GatedDeltaNetLayer();
  return layer;
}

void destroy_gated_delta_net_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_gated_delta_net_layer, destroy_gated_delta_net_layer};
}

#endif

} // namespace causallm

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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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
  tmp_mixed_qkv = 2,
  tmp_z = 3,
  tmp_attn_out = 4,
  prefill_qkv = 5,   // [batch, 1, max_seq, conv_dim]
  prefill_z = 6,      // [batch, 1, max_seq, value_dim]
  prefill_out = 7,    // [batch, 1, max_seq, value_dim]
};

void GatedDeltaNetLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gdn_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[gated_delta_net] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void GatedDeltaNetLayer::finalize(nntrainer::InitLayerContext &context) {

  auto &num_heads_prop = std::get<props::NumHeads>(gdn_props);
  auto &epsilon_prop = std::get<nntrainer::props::Epsilon>(gdn_props);

  const auto &input_dims = context.getInputDimensions();
  hidden_size = input_dims[0].width();

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

  unsigned int batch = input_dims[0].batch();

  // ======================= Weights =======================

  // [0] in_proj_qkv: [hidden_size, conv_dim]
  wt_idx[GDNWeights::in_proj_qkv] = context.requestWeight(
    nntrainer::TensorDim(1, 1, hidden_size, conv_dim, weight_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "in_proj_qkv", false);

  // [1] conv1d_weight: [conv_kernel_size, conv_dim]
  wt_idx[GDNWeights::conv1d_weight] = context.requestWeight(
    nntrainer::TensorDim(1, 1, conv_kernel_size, conv_dim, weight_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "conv1d_weight", false);

  // [2] A_log: [num_v_heads]
  wt_idx[GDNWeights::A_log] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, num_v_heads, weight_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "A_log", false);

  // [3] in_proj_b: [hidden_size, num_v_heads]
  wt_idx[GDNWeights::in_proj_b] = context.requestWeight(
    nntrainer::TensorDim(1, 1, hidden_size, num_v_heads, weight_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "in_proj_b", false);

  // [4] in_proj_a: [hidden_size, num_v_heads]
  wt_idx[GDNWeights::in_proj_a] = context.requestWeight(
    nntrainer::TensorDim(1, 1, hidden_size, num_v_heads, weight_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "in_proj_a", false);

  // [5] dt_bias: [num_v_heads]
  wt_idx[GDNWeights::dt_bias] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, num_v_heads, weight_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "dt_bias", false);

  // [6] norm_weight: [head_v_dim]
  wt_idx[GDNWeights::norm_weight] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, head_v_dim, weight_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "norm_weight", false);

  // [7] out_proj: [value_dim, hidden_size]
  wt_idx[GDNWeights::out_proj] = context.requestWeight(
    nntrainer::TensorDim(1, 1, value_dim, hidden_size, weight_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "out_proj", false);

  // [8] in_proj_z: [hidden_size, value_dim]
  wt_idx[GDNWeights::in_proj_z] = context.requestWeight(
    nntrainer::TensorDim(1, 1, hidden_size, value_dim, weight_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "in_proj_z", false);

  // =================== Persistent State Tensors ===================

  // conv_state: [batch, 1, conv_dim, conv_kernel_size-1]
  tensor_idx[GDNTensors::conv_state] = context.requestTensor(
    nntrainer::TensorDim(batch, 1, conv_dim, conv_kernel_size - 1,
                         activation_type),
    "conv_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  // recurrent_state: [batch, num_v_heads, head_k_dim, head_v_dim]
  tensor_idx[GDNTensors::recurrent_state] = context.requestTensor(
    nntrainer::TensorDim(batch, num_v_heads, head_k_dim, head_v_dim,
                         activation_type),
    "recurrent_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  // =================== Temporary Tensors ===================

  // tmp_mixed_qkv: [batch, 1, 1, conv_dim] (for single token)
  tensor_idx[GDNTensors::tmp_mixed_qkv] = context.requestTensor(
    nntrainer::TensorDim(batch, 1, 1, conv_dim, activation_type),
    "tmp_mixed_qkv", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // tmp_z: [batch, 1, 1, value_dim] (gate projection output)
  tensor_idx[GDNTensors::tmp_z] = context.requestTensor(
    nntrainer::TensorDim(batch, 1, 1, value_dim, activation_type),
    "tmp_z", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // tmp_attn_out: [batch, 1, num_v_heads, head_v_dim] (delta rule output)
  tensor_idx[GDNTensors::tmp_attn_out] = context.requestTensor(
    nntrainer::TensorDim(batch, 1, num_v_heads, head_v_dim, activation_type),
    "tmp_attn_out", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // =================== Prefill Tensors (for forwarding) ===================
  // Use INIT_SEQ_LEN from input height as max prefill length
  unsigned int max_seq = input_dims[0].height();
  if (max_seq < 1)
    max_seq = 1;

  // prefill_qkv: [batch, 1, max_seq, conv_dim]
  tensor_idx[GDNTensors::prefill_qkv] = context.requestTensor(
    nntrainer::TensorDim(batch, 1, max_seq, conv_dim, activation_type),
    "prefill_qkv", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // prefill_z: [batch, 1, max_seq, value_dim]
  tensor_idx[GDNTensors::prefill_z] = context.requestTensor(
    nntrainer::TensorDim(batch, 1, max_seq, value_dim, activation_type),
    "prefill_z", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // prefill_out: [batch, 1, max_seq, value_dim]
  tensor_idx[GDNTensors::prefill_out] = context.requestTensor(
    nntrainer::TensorDim(batch, 1, max_seq, value_dim, activation_type),
    "prefill_out", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void GatedDeltaNetLayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  // Weights
  nntrainer::Tensor &w_qkv = context.getWeight(wt_idx[GDNWeights::in_proj_qkv]);
  nntrainer::Tensor &w_conv = context.getWeight(wt_idx[GDNWeights::conv1d_weight]);
  nntrainer::Tensor &w_a_log = context.getWeight(wt_idx[GDNWeights::A_log]);
  nntrainer::Tensor &w_proj_b = context.getWeight(wt_idx[GDNWeights::in_proj_b]);
  nntrainer::Tensor &w_proj_a = context.getWeight(wt_idx[GDNWeights::in_proj_a]);
  nntrainer::Tensor &w_dt_bias = context.getWeight(wt_idx[GDNWeights::dt_bias]);
  nntrainer::Tensor &w_norm = context.getWeight(wt_idx[GDNWeights::norm_weight]);
  nntrainer::Tensor &w_out = context.getWeight(wt_idx[GDNWeights::out_proj]);
  nntrainer::Tensor &w_z = context.getWeight(wt_idx[GDNWeights::in_proj_z]);

  nntrainer::Tensor &conv_st =
    context.getTensor(tensor_idx[GDNTensors::conv_state]);
  nntrainer::Tensor &rec_st =
    context.getTensor(tensor_idx[GDNTensors::recurrent_state]);

  unsigned int batch_size = input.getDim().batch();
  unsigned int seq_len = input.getDim().height();
  float scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));

  float *a_log_data = w_a_log.getAddress<float>(0);
  float *dt_bias_data = w_dt_bias.getAddress<float>(0);
  float *conv_w_data = w_conv.getAddress<float>(0);
  float *gamma_data = w_norm.getAddress<float>(0);
  float *proj_b_data = w_proj_b.getAddress<float>(0);
  float *proj_a_data = w_proj_a.getAddress<float>(0);

  // Get prefill tensors
  nntrainer::Tensor &pf_qkv =
    context.getTensor(tensor_idx[GDNTensors::prefill_qkv]);
  nntrainer::Tensor &pf_z =
    context.getTensor(tensor_idx[GDNTensors::prefill_z]);
  nntrainer::Tensor &pf_out =
    context.getTensor(tensor_idx[GDNTensors::prefill_out]);

  for (unsigned int b = 0; b < batch_size; ++b) {

    // ======= Step 1: Batch projections via Tensor dot() =======
    // input_seq: [1, 1, seq_len, hidden_size]
    ml::train::TensorDim seq_dim(1, 1, seq_len, hidden_size,
                                 input.getDim().getTensorType());
    nntrainer::Tensor in_seq = input.getSharedDataTensor(
      seq_dim, b * input.getDim().getFeatureLen(), true);

    // qkv_seq: [1, 1, seq_len, conv_dim]
    ml::train::TensorDim qkv_seq_dim(1, 1, seq_len, conv_dim,
                                      pf_qkv.getDim().getTensorType());
    nntrainer::Tensor qkv_seq = pf_qkv.getSharedDataTensor(
      qkv_seq_dim, b * seq_len * conv_dim, true);

    // z_seq: [1, 1, seq_len, value_dim]
    ml::train::TensorDim z_seq_dim(1, 1, seq_len, value_dim,
                                    pf_z.getDim().getTensorType());
    nntrainer::Tensor z_seq = pf_z.getSharedDataTensor(
      z_seq_dim, b * seq_len * value_dim, true);

    // out_seq: [1, 1, seq_len, value_dim]
    ml::train::TensorDim out_seq_dim(1, 1, seq_len, value_dim,
                                      pf_out.getDim().getTensorType());
    nntrainer::Tensor out_attn = pf_out.getSharedDataTensor(
      out_seq_dim, b * seq_len * value_dim, true);

    // mixed_qkv = input_seq @ w_qkv  [seq_len, hidden] x [hidden, conv_dim]
    in_seq.dot(w_qkv, qkv_seq, false, false);

    // z = input_seq @ w_z  [seq_len, hidden] x [hidden, value_dim]
    in_seq.dot(w_z, z_seq, false, false);

    float *in_data = in_seq.getAddress<float>(0);
    float *qkv_buf_data = qkv_seq.getAddress<float>(0);
    float *z_buf_data = z_seq.getAddress<float>(0);
    float *out_buf_data = out_attn.getAddress<float>(0);

    // ======= Step 2: Causal Conv1d over full sequence =======
    float *cs_data = conv_st.getAddress<float>(0) +
                     b * conv_dim * (conv_kernel_size - 1);

    for (unsigned int t = 0; t < seq_len; ++t) {
      float *qkv_t = qkv_buf_data + t * conv_dim;
      for (unsigned int c = 0; c < conv_dim; ++c) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < conv_kernel_size - 1; ++k) {
          sum += cs_data[c * (conv_kernel_size - 1) + k] *
                 conv_w_data[k * conv_dim + c];
        }
        sum += qkv_t[c] * conv_w_data[(conv_kernel_size - 1) * conv_dim + c];

        // Shift state
        for (unsigned int k = 0; k < conv_kernel_size - 2; ++k) {
          cs_data[c * (conv_kernel_size - 1) + k] =
            cs_data[c * (conv_kernel_size - 1) + k + 1];
        }
        cs_data[c * (conv_kernel_size - 1) + conv_kernel_size - 2] = qkv_t[c];

        // SiLU
        float sigmoid_val = 1.0f / (1.0f + std::exp(-sum));
        qkv_t[c] = sum * sigmoid_val;
      }
    }

    // ======= Step 3: Compute b, a for full sequence =======
    // b, a projections per token per head
    std::vector<float> beta_buf(seq_len * num_v_heads);
    std::vector<float> g_buf(seq_len * num_v_heads);

    for (unsigned int t = 0; t < seq_len; ++t) {
      float *in_t = in_data + t * hidden_size;
      for (unsigned int h = 0; h < num_v_heads; ++h) {
        float b_sum = 0.0f, a_sum = 0.0f;
        for (unsigned int d = 0; d < hidden_size; ++d) {
          b_sum += in_t[d] * proj_b_data[d * num_v_heads + h];
          a_sum += in_t[d] * proj_a_data[d * num_v_heads + h];
        }
        beta_buf[t * num_v_heads + h] = 1.0f / (1.0f + std::exp(-b_sum));
        float softplus = std::log(1.0f + std::exp(a_sum + dt_bias_data[h]));
        g_buf[t * num_v_heads + h] = -std::exp(a_log_data[h]) * softplus;
      }
    }

    // ======= Step 4: L2 normalize Q, K and apply scale =======
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (unsigned int t = 0; t < seq_len; ++t) {
      float *qkv_t = qkv_buf_data + t * conv_dim;
      for (unsigned int h = 0; h < num_v_heads; ++h) {
        float *q_h = qkv_t + h * head_k_dim;
        float *k_h = qkv_t + key_dim + h * head_k_dim;

        float q_norm = 0.0f, k_norm = 0.0f;
        for (unsigned int d = 0; d < head_k_dim; ++d) {
          q_norm += q_h[d] * q_h[d];
          k_norm += k_h[d] * k_h[d];
        }
        q_norm = std::max(std::sqrt(q_norm), 1e-6f);
        k_norm = std::max(std::sqrt(k_norm), 1e-6f);
        for (unsigned int d = 0; d < head_k_dim; ++d) {
          q_h[d] = q_h[d] / q_norm * scale;
          k_h[d] /= k_norm;
        }
      }
    }

    // ======= Step 5: Chunk-based delta rule =======
    // Process per head for better cache locality
    float *rs_data = rec_st.getAddress<float>(0) +
                     b * num_v_heads * head_k_dim * head_v_dim;

    for (unsigned int h = 0; h < num_v_heads; ++h) {
      float *state_h = rs_data + h * head_k_dim * head_v_dim;

      for (unsigned int t = 0; t < seq_len; ++t) {
        float *qkv_t = qkv_buf_data + t * conv_dim;
        float *q_h = qkv_t + h * head_k_dim;
        float *k_h = qkv_t + key_dim + h * head_k_dim;
        float *v_h = qkv_t + key_dim * 2 + h * head_v_dim;
        float g_exp = std::exp(g_buf[t * num_v_heads + h]);
        float beta_h = beta_buf[t * num_v_heads + h];

        // Decay
        for (unsigned int i = 0; i < head_k_dim * head_v_dim; ++i)
          state_h[i] *= g_exp;

        // kv_mem = state^T @ k
        float *out_h = out_buf_data + t * value_dim + h * head_v_dim;
        for (unsigned int j = 0; j < head_v_dim; ++j) {
          float kv_mem_j = 0.0f;
          for (unsigned int i = 0; i < head_k_dim; ++i)
            kv_mem_j += state_h[i * head_v_dim + j] * k_h[i];
          float delta_j = (v_h[j] - kv_mem_j) * beta_h;

          // Update state column j
          for (unsigned int i = 0; i < head_k_dim; ++i)
            state_h[i * head_v_dim + j] += k_h[i] * delta_j;

          // Read output
          float o_j = 0.0f;
          for (unsigned int i = 0; i < head_k_dim; ++i)
            o_j += state_h[i * head_v_dim + j] * q_h[i];
          out_h[j] = o_j;
        }
      }
    }

    // ======= Step 6: RMSNormGated for full sequence =======
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (unsigned int t = 0; t < seq_len; ++t) {
      float *z_t = z_buf_data + t * value_dim;
      for (unsigned int h = 0; h < num_v_heads; ++h) {
        float *o_h = out_buf_data + t * value_dim + h * head_v_dim;
        float *z_h = z_t + h * head_v_dim;

        float variance = 0.0f;
        for (unsigned int d = 0; d < head_v_dim; ++d)
          variance += o_h[d] * o_h[d];
        variance /= static_cast<float>(head_v_dim);
        float inv_rms = 1.0f / std::sqrt(variance + norm_eps);

        for (unsigned int d = 0; d < head_v_dim; ++d) {
          float normalized = o_h[d] * inv_rms * gamma_data[d];
          float silu_z = z_h[d] / (1.0f + std::exp(-z_h[d]));
          o_h[d] = normalized * silu_z;
        }
      }
    }

    // ======= Step 7: Output projection for full sequence =======
    // output = attn_out @ w_out  [seq_len, value_dim] x [value_dim, hidden_size]
    ml::train::TensorDim o_seq_dim(1, 1, seq_len, hidden_size,
                                    output.getDim().getTensorType());
    nntrainer::Tensor o_seq = output.getSharedDataTensor(
      o_seq_dim, b * output.getDim().getFeatureLen(), true);
    out_attn.dot(w_out, o_seq, false, false);
  }
}

void GatedDeltaNetLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  // Weights
  nntrainer::Tensor &w_qkv = context.getWeight(wt_idx[GDNWeights::in_proj_qkv]);
  nntrainer::Tensor &w_conv = context.getWeight(wt_idx[GDNWeights::conv1d_weight]);
  nntrainer::Tensor &w_a_log = context.getWeight(wt_idx[GDNWeights::A_log]);
  nntrainer::Tensor &w_proj_b = context.getWeight(wt_idx[GDNWeights::in_proj_b]);
  nntrainer::Tensor &w_proj_a = context.getWeight(wt_idx[GDNWeights::in_proj_a]);
  nntrainer::Tensor &w_dt_bias = context.getWeight(wt_idx[GDNWeights::dt_bias]);
  nntrainer::Tensor &w_norm = context.getWeight(wt_idx[GDNWeights::norm_weight]);
  nntrainer::Tensor &w_out = context.getWeight(wt_idx[GDNWeights::out_proj]);
  nntrainer::Tensor &w_z = context.getWeight(wt_idx[GDNWeights::in_proj_z]);

  // Persistent state tensors
  nntrainer::Tensor &conv_st =
    context.getTensor(tensor_idx[GDNTensors::conv_state]);
  nntrainer::Tensor &rec_st =
    context.getTensor(tensor_idx[GDNTensors::recurrent_state]);

  // Temporary tensors
  nntrainer::Tensor &tmp_qkv =
    context.getTensor(tensor_idx[GDNTensors::tmp_mixed_qkv]);
  nntrainer::Tensor &tmp_z_buf =
    context.getTensor(tensor_idx[GDNTensors::tmp_z]);
  nntrainer::Tensor &tmp_attn =
    context.getTensor(tensor_idx[GDNTensors::tmp_attn_out]);

  unsigned int batch_size = input.getDim().batch();
  unsigned int seq_len = to - from;

  float scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));

  // Pre-compute A_log exp values: A_exp[h] = exp(A_log[h])
  // A_log is [1,1,1,num_v_heads]
  float *a_log_data = w_a_log.getAddress<float>(0);
  float *dt_bias_data = w_dt_bias.getAddress<float>(0);

  // Process each batch and each token
  for (unsigned int b = 0; b < batch_size; ++b) {

    // Get batch slice dimensions
    ml::train::TensorDim in_step_dim = input.getDim();
    in_step_dim.batch(1);
    in_step_dim.height(1); // one token at a time

    ml::train::TensorDim out_step_dim = output.getDim();
    out_step_dim.batch(1);
    out_step_dim.height(1);

    for (unsigned int t = from; t < to; ++t) {
      // ===================== Step 1: Linear projections =====================
      // input_t: [1, 1, 1, hidden_size]
      nntrainer::Tensor input_t = input.getSharedDataTensor(
        in_step_dim,
        b * input.getDim().getFeatureLen() + t * hidden_size, true);

      // mixed_qkv = input_t @ w_qkv → [1, 1, 1, conv_dim]
      nntrainer::Tensor qkv_t = tmp_qkv.getSharedDataTensor(
        {1, 1, 1, conv_dim, tmp_qkv.getDim().getTensorType()}, 0, true);
      input_t.dot(w_qkv, qkv_t, false, false);

      // z = input_t @ w_z → [1, 1, 1, value_dim]
      nntrainer::Tensor z_t = tmp_z_buf.getSharedDataTensor(
        {1, 1, 1, value_dim, tmp_z_buf.getDim().getTensorType()}, 0, true);
      input_t.dot(w_z, z_t, false, false);

      // ===================== Step 2: Causal Conv1d =====================
      // conv_state for this batch: [1, 1, conv_dim, kernel_size-1]
      ml::train::TensorDim cs_dim(1, 1, conv_dim, conv_kernel_size - 1,
                                  conv_st.getDim().getTensorType());
      nntrainer::Tensor cs_batch = conv_st.getSharedDataTensor(
        cs_dim, b * conv_dim * (conv_kernel_size - 1), true);

      // Depthwise causal conv1d for single token:
      // For each channel c:
      //   out[c] = sum(conv_weight[k, c] * [state[c,k-1..], input[c]])
      // Then shift state left and append input
      float *qkv_data = qkv_t.getAddress<float>(0);
      float *conv_w_data = w_conv.getAddress<float>(0);
      float *cs_data = cs_batch.getAddress<float>(0);

      for (unsigned int c = 0; c < conv_dim; ++c) {
        float sum = 0.0f;
        // Multiply state entries with conv weights [0..kernel_size-2]
        for (unsigned int k = 0; k < conv_kernel_size - 1; ++k) {
          sum += cs_data[c * (conv_kernel_size - 1) + k] *
                 conv_w_data[k * conv_dim + c];
        }
        // Multiply current input with last conv weight
        sum += qkv_data[c] * conv_w_data[(conv_kernel_size - 1) * conv_dim + c];

        // Shift state left: state[c, 0..k-3] = state[c, 1..k-2]
        for (unsigned int k = 0; k < conv_kernel_size - 2; ++k) {
          cs_data[c * (conv_kernel_size - 1) + k] =
            cs_data[c * (conv_kernel_size - 1) + k + 1];
        }
        // Append current input to state
        cs_data[c * (conv_kernel_size - 1) + conv_kernel_size - 2] =
          qkv_data[c];

        // Apply SiLU activation: x * sigmoid(x)
        float sigmoid_val = 1.0f / (1.0f + std::exp(-sum));
        qkv_data[c] = sum * sigmoid_val;
      }

      // ===================== Step 3: Split Q, K, V =====================
      // qkv_data now has [key_dim, key_dim, value_dim] = [Q, K, V]
      // No need to copy - just use offsets into qkv_data

      // ===================== Step 4: Compute b, a projections =====================
      // b = input_t @ w_proj_b → [1, 1, 1, num_v_heads]
      // a = input_t @ w_proj_a → [1, 1, 1, num_v_heads]
      // We compute these scalars per head
      float b_vals[256], a_vals[256]; // max heads
      float *in_data = input_t.getAddress<float>(0);
      float *proj_b_data = w_proj_b.getAddress<float>(0);
      float *proj_a_data = w_proj_a.getAddress<float>(0);

      for (unsigned int h = 0; h < num_v_heads; ++h) {
        float b_sum = 0.0f, a_sum = 0.0f;
        for (unsigned int d = 0; d < hidden_size; ++d) {
          b_sum += in_data[d] * proj_b_data[d * num_v_heads + h];
          a_sum += in_data[d] * proj_a_data[d * num_v_heads + h];
        }
        // beta = sigmoid(b)
        b_vals[h] = 1.0f / (1.0f + std::exp(-b_sum));

        // g = -exp(A_log) * softplus(a + dt_bias)
        float a_plus_dt = a_sum + dt_bias_data[h];
        float softplus_val = std::log(1.0f + std::exp(a_plus_dt));
        a_vals[h] = -std::exp(a_log_data[h]) * softplus_val;
      }

      // ===================== Step 5: Recurrent Gated Delta Rule =====================
      // recurrent_state for this batch: [1, num_v_heads, head_k_dim, head_v_dim]
      ml::train::TensorDim rs_dim(1, num_v_heads, head_k_dim, head_v_dim,
                                  rec_st.getDim().getTensorType());
      nntrainer::Tensor rs_batch = rec_st.getSharedDataTensor(
        rs_dim, b * num_v_heads * head_k_dim * head_v_dim, true);
      float *rs_data = rs_batch.getAddress<float>(0);

      float *out_data = tmp_attn.getAddress<float>(0);

      for (unsigned int h = 0; h < num_v_heads; ++h) {
        float *state_h = rs_data + h * head_k_dim * head_v_dim;

        // Get Q, K, V pointers for this head
        float *q_h = qkv_data + h * head_k_dim;
        float *k_h = qkv_data + key_dim + h * head_k_dim;
        float *v_h = qkv_data + key_dim * 2 + h * head_v_dim;

        // L2 normalize Q and K
        float q_norm = 0.0f, k_norm = 0.0f;
        for (unsigned int d = 0; d < head_k_dim; ++d) {
          q_norm += q_h[d] * q_h[d];
          k_norm += k_h[d] * k_h[d];
        }
        q_norm = std::sqrt(q_norm);
        k_norm = std::sqrt(k_norm);
        if (q_norm < 1e-6f) q_norm = 1e-6f;
        if (k_norm < 1e-6f) k_norm = 1e-6f;
        for (unsigned int d = 0; d < head_k_dim; ++d) {
          q_h[d] /= q_norm;
          k_h[d] /= k_norm;
        }

        // Apply scale to query
        for (unsigned int d = 0; d < head_k_dim; ++d) {
          q_h[d] *= scale;
        }

        float g_exp = std::exp(a_vals[h]); // decay factor
        float beta_h = b_vals[h];

        // 1. Decay state: state *= exp(g)
        for (unsigned int i = 0; i < head_k_dim * head_v_dim; ++i) {
          state_h[i] *= g_exp;
        }

        // 2. kv_mem = (state * k).sum(dim=-2) → [head_v_dim]
        //    kv_mem[j] = sum_i(state[i][j] * k[i])
        float kv_mem[256]; // max head_v_dim
        for (unsigned int j = 0; j < head_v_dim; ++j) {
          float sum = 0.0f;
          for (unsigned int i = 0; i < head_k_dim; ++i) {
            sum += state_h[i * head_v_dim + j] * k_h[i];
          }
          kv_mem[j] = sum;
        }

        // 3. delta = (v - kv_mem) * beta → [head_v_dim]
        float delta[256]; // max head_v_dim
        for (unsigned int j = 0; j < head_v_dim; ++j) {
          delta[j] = (v_h[j] - kv_mem[j]) * beta_h;
        }

        // 4. state += k.unsqueeze(-1) * delta.unsqueeze(-2)
        //    state[i][j] += k[i] * delta[j]
        for (unsigned int i = 0; i < head_k_dim; ++i) {
          for (unsigned int j = 0; j < head_v_dim; ++j) {
            state_h[i * head_v_dim + j] += k_h[i] * delta[j];
          }
        }

        // 5. output = (state * q).sum(dim=-2) → [head_v_dim]
        //    out[j] = sum_i(state[i][j] * q[i])
        float *out_h = out_data + h * head_v_dim;
        for (unsigned int j = 0; j < head_v_dim; ++j) {
          float sum = 0.0f;
          for (unsigned int i = 0; i < head_k_dim; ++i) {
            sum += state_h[i * head_v_dim + j] * q_h[i];
          }
          out_h[j] = sum;
        }
      }

      // ===================== Step 6: RMSNormGated =====================
      // out_data: [num_v_heads * head_v_dim] = [value_dim]
      // z_t: [value_dim]
      // Apply per-head: RMSNorm(out_h) * gamma * SiLU(z_h)
      float *z_data = z_t.getAddress<float>(0);
      float *gamma_data = w_norm.getAddress<float>(0);

      for (unsigned int h = 0; h < num_v_heads; ++h) {
        float *o_h = out_data + h * head_v_dim;
        float *z_h = z_data + h * head_v_dim;

        // RMSNorm: variance = mean(x^2)
        float variance = 0.0f;
        for (unsigned int d = 0; d < head_v_dim; ++d) {
          variance += o_h[d] * o_h[d];
        }
        variance /= static_cast<float>(head_v_dim);
        float inv_rms = 1.0f / std::sqrt(variance + norm_eps);

        // output = gamma * (x * inv_rms) * SiLU(z)
        for (unsigned int d = 0; d < head_v_dim; ++d) {
          float normalized = o_h[d] * inv_rms * gamma_data[d];
          float z_val = z_h[d];
          float silu_z = z_val / (1.0f + std::exp(-z_val));
          o_h[d] = normalized * silu_z;
        }
      }

      // ===================== Step 7: Output projection =====================
      // output_t = attn_out @ w_out → [1, 1, 1, hidden_size]
      nntrainer::Tensor out_t = output.getSharedDataTensor(
        out_step_dim,
        b * output.getDim().getFeatureLen() + t * hidden_size, true);

      nntrainer::Tensor attn_flat = tmp_attn.getSharedDataTensor(
        {1, 1, 1, value_dim, tmp_attn.getDim().getTensorType()}, 0, true);
      attn_flat.dot(w_out, out_t, false, false);
    }
  }
}

void GatedDeltaNetLayer::recurrentGatedDeltaRule(
  const nntrainer::Tensor &query, const nntrainer::Tensor &key,
  const nntrainer::Tensor &value, const nntrainer::Tensor &g,
  const nntrainer::Tensor &beta, nntrainer::Tensor &state,
  nntrainer::Tensor &output) {
  // Standalone helper - logic is inlined in incremental_forwarding for now
}

void GatedDeltaNetLayer::chunkGatedDeltaRule(
  const nntrainer::Tensor &query, const nntrainer::Tensor &key,
  const nntrainer::Tensor &value, const nntrainer::Tensor &g,
  const nntrainer::Tensor &beta, nntrainer::Tensor &state,
  nntrainer::Tensor &output, unsigned int chunk_size) {
  /// @todo Implement chunk-based gated delta rule for efficient prefill
}

void GatedDeltaNetLayer::applyCausalConv1d(
  const nntrainer::Tensor &input, const nntrainer::Tensor &weight,
  nntrainer::Tensor &state, nntrainer::Tensor &output,
  unsigned int seq_len) {
  // Logic inlined in incremental_forwarding for now
}

void GatedDeltaNetLayer::applyRMSNormGated(
  const nntrainer::Tensor &input, const nntrainer::Tensor &gate,
  const nntrainer::Tensor &gamma, nntrainer::Tensor &output, float eps) {
  // Logic inlined in incremental_forwarding for now
}

void GatedDeltaNetLayer::l2Normalize(nntrainer::Tensor &tensor, float eps) {
  // Logic inlined in incremental_forwarding for now
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

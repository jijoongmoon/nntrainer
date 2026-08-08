
/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	moe_layer.cpp
 * @date	09 June 2025
 * @brief	This is a Mixture of Expert Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <acti_func.h>
#include <algorithm>
#include <cmath>
#include <compute_ops.h>
#include <cstdio>
#include <cstring>
#include <layer_prof.h>
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_fc_qint4.h>
#include <cuda_moe.h>
#include <cuda_stream_manager.h>
#endif
#include <node_exporter.h>
#include <qwen_moe_layer.h>
#include <stdexcept>
#include <thread_manager.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

MoELayer::MoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::MoEActivation()),
  expert_gate_proj_indices({}),
  expert_up_proj_indices({}),
  expert_down_proj_indices({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  expert_mask_idx(std::numeric_limits<unsigned>::max()) {}

void MoELayer::finalize(nntrainer::InitLayerContext &context) {

  // 1. Validate input/output dimensions
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "MoE layer only supports single input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  // 2. Set output dimensions (same as input)
  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] = in_dim;
  context.setOutputDimensions(output_dims);

  // 3. Get MoE properties
  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width(); // Feature dimension

  // activation function
  if (std::get<props::MoEActivation>(moe_props).empty()) {
    throw std::runtime_error("Activation type is not set for MoE layer");
  }
  switch (context.getActivationDataType()) {
  case ml::train::TensorDim::DataType::FP32:
    acti_func.setActiFunc<float>(
      std::get<props::MoEActivation>(moe_props).get());
    break;
#ifdef ENABLE_FP16
  case ml::train::TensorDim::DataType::FP16:
    // An FP16-activation MoE is not exotic: every QINT4-FP16 package is one.
    // Without this case finalize() threw at graph build for the whole family.
    acti_func.setActiFunc<_FP16>(
      std::get<props::MoEActivation>(moe_props).get());
    break;
#endif
  default:
    throw std::runtime_error("Unsupported activation data type for MoE layer");
  }

  // 4. Initialie gate layer (router)
  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  // 5. Initializer expert weights
  expert_gate_proj_indices.reserve(num_experts);
  expert_up_proj_indices.reserve(num_experts);
  expert_down_proj_indices.reserve(num_experts);

  nntrainer::TensorDim expert_gate_dim(
    1, is_nchw ? 1 : intermediate_size, is_nchw ? hidden_size : 1,
    is_nchw ? intermediate_size : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  nntrainer::TensorDim expert_down_dim(
    1, is_nchw ? 1 : hidden_size, is_nchw ? intermediate_size : 1,
    is_nchw ? hidden_size : intermediate_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  for (unsigned int i = 0; i < num_experts; ++i) {
    // Up projection
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, // Same dimensions as gate projection
      weight_initializer, weight_regularizer, weight_regularizer_constant,
      weight_decay, "expert_up_" + std::to_string(i), false));

    // Gate projection
    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), false));

    // Down projection
    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false));
  }

  // 6. Request intermediate tensors
  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  // Router logits :  [batch * seq, num_experts]
  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // Expert mask: [num_experts, batch*seq]
  expert_mask_idx =
    context.requestTensor({num_experts, 1, topk, total_tokens}, "expert_mask",
                          nntrainer::Initializer::ZEROS, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  // 7. Batched per-expert scratch, in the ACTIVATION dtype (the requestTensor
  // brace-init overload above defaults to FP32, which is right for the router
  // and wrong for these). Height is the worst case -- every token routed to a
  // single expert -- because the pool cannot grow mid-forward, and on a CUDA
  // run a late growth inside a graph capture is refused outright.
  const auto act_tt = in_dim.getTensorType();
  const nntrainer::TensorDim rows_hidden({1, 1, total_tokens, hidden_size},
                                         act_tt);
  const nntrainer::TensorDim rows_inter({1, 1, total_tokens, intermediate_size},
                                        act_tt);

  gathered_in_idx = context.requestTensor(
    rows_hidden, "moe_gathered_in", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  gate_out_idx = context.requestTensor(
    rows_inter, "moe_gate_out", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  up_out_idx = context.requestTensor(
    rows_inter, "moe_up_out", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  acti_out_idx = context.requestTensor(
    rows_inter, "moe_acti_out", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  expert_out_idx = context.requestTensor(
    rows_hidden, "moe_expert_out", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void MoELayer::forwarding(nntrainer::RunLayerContext &context, bool training) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits = context.getTensor(router_logits_idx);
  nntrainer::Tensor &expert_mask = context.getTensor(expert_mask_idx);

  const unsigned batch_size = input.batch();
  const unsigned seq_len = input.height();
  const unsigned hidden_size = input.width();
  const unsigned total_tokens = batch_size * seq_len;

  // reshape input: [B,1,S,H] -> [B*S,1,1,H]
  input.reshape({total_tokens, 1, 1, hidden_size});

  // reshape output: [B,1,S,H] -> [B*S,1,1,H]
  output.reshape({total_tokens, 1, 1, hidden_size});
  output.setZero();

  // routing
  nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
  // Routing is ALWAYS fp32: the gate weight and router_logits are FP32,
  // so an FP16 input makes HalfTensor::dot write FP16 bits into the FP32
  // logits buffer -- garbage top-k, no crash. Widen first on FP16 models.
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    input.dot(gate_weights, router_logits);
  } else {
    nntrainer::Tensor input32 =
      input.clone(ml::train::TensorDim::DataType::FP32);
    input32.dot(gate_weights, router_logits);
  }
  router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);
  auto topk_result = router_logits.topK(topk);
  auto topk_values = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);

  const uint32_t *indices_data = topk_indices.getData<uint32_t>();
  {
    auto &tm = nntrainer::ThreadManager::Global();
    size_t total_iters =
      static_cast<size_t>(total_tokens) * static_cast<size_t>(topk);
    tm.parallel_for(0, static_cast<size_t>(total_iters), [&](size_t idx) {
      int k = idx % topk;
      int i = idx / topk;
      expert_mask.setValue(indices_data[idx], 0, k, i, 1.0f);
    });
  }

  // Pre-compute expert token assignments for better cache locality
  std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
    num_experts);
  for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
    for (int k = 0; k < static_cast<int>(topk); ++k) {
      unsigned expert_idx = indices_data[i * topk + k];
      float weight = topk_values.getValue<float>(i, 0, 0, k);
      expert_assignments[expert_idx].emplace_back(i, weight);
    }
  }

  // Serial outer loop: the expert GEMV/GEMM parallelizes internally via
  // ThreadManager (dot() calls parallel_for), and nesting parallel_for
  // deadlocks because ThreadManager::parallelize() uses a non-recursive
  // execution_mutex_.
  for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
       ++expert_idx) {
    const auto &assignments = expert_assignments[expert_idx];
    if (assignments.empty())
      continue;

    // Use optimized expert forward computation without memory copies
    compute_expert_forward(
      input, output, assignments,
      context.getWeight(expert_gate_proj_indices[expert_idx]),
      context.getWeight(expert_up_proj_indices[expert_idx]),
      context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size);
  }

  // reshape output: [B*S,1,1,H] -> [B,1,S,H]
  output.reshape({batch_size, 1, seq_len, hidden_size});
}

inline void MoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  // Create tensor dimensions for single token processing
  nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size},
                                        input.getTensorType());

  // Create a temporary output tensor for this expert to avoid critical section
  nntrainer::Tensor expert_output(output.batch(), output.channel(),
                                  output.height(), output.width(),
                                  output.getTensorType());
  expert_output.setZero();

  // Process each token individually to avoid memory copies
  for (size_t i = 0; i < num_tokens; ++i) {
    const unsigned token_idx = token_assignments[i].first;
    const float weight = token_assignments[i].second;

    // Create shared tensor for input token (no memory copy)
    size_t token_offset = token_idx * hidden_size;
    nntrainer::Tensor token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);

    // Create intermediate tensors for this token
    nntrainer::Tensor gate_out(intermediate_dim);
    nntrainer::Tensor acti_out(intermediate_dim);
    nntrainer::Tensor up_out(intermediate_dim);

    // Gate projection using optimized dot operation
    token_input.dot(gate_proj, gate_out);

    // Up projection using optimized dot operation
    token_input.dot(up_proj, up_out);

    // dtype-generic SwiGLU. The free nntrainer::swiglu() is FP32-only and
    // reads its operands through an unchecked getData<float>(), which on an
    // FP16 tensor reinterprets the bits rather than converting them.
    acti_func.run_fn(gate_out, acti_out);
    acti_out.multiply_i(up_out);

    // Down projection using optimized dot operation
    nntrainer::Tensor token_expert_output(token_output_dim);
    acti_out.dot(down_proj, token_expert_output);

    // Apply weight and accumulate to expert's temporary output
    token_expert_output.multiply_i(weight);
    size_t output_offset = token_idx * hidden_size;
    nntrainer::Tensor token_output =
      expert_output.getSharedDataTensor(token_output_dim, output_offset, true);

    token_output.add_i(token_expert_output);
  }

  // Add expert's result to final output (no critical section in sequential
  // mode)
  output.add_i(expert_output);
}

inline void MoELayer::compute_expert_forward_no_critical(
  const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  // Create tensor dimensions for single token processing
  nntrainer::TensorDim token_input_dim({1, 1, 1, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, 1, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, 1, hidden_size},
                                        input.getTensorType());

  // Process each token individually to avoid memory copies
  for (size_t i = 0; i < num_tokens; ++i) {
    const unsigned token_idx = token_assignments[i].first;
    const float weight = token_assignments[i].second;

    // Create shared tensor for input token (no memory copy)
    size_t token_offset = token_idx * hidden_size;
    nntrainer::Tensor token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);

    // Create intermediate tensors for this token
    nntrainer::Tensor gate_out(intermediate_dim);
    nntrainer::Tensor acti_out(intermediate_dim);
    nntrainer::Tensor up_out(intermediate_dim);

    // Gate projection using optimized dot operation
    token_input.dot(gate_proj, gate_out);

    // Up projection using optimized dot operation
    token_input.dot(up_proj, up_out);

    // dtype-generic SwiGLU. The free nntrainer::swiglu() is FP32-only and
    // reads its operands through an unchecked getData<float>(), which on an
    // FP16 tensor reinterprets the bits rather than converting them.
    acti_func.run_fn(gate_out, acti_out);
    acti_out.multiply_i(up_out);

    // Down projection using optimized dot operation
    nntrainer::Tensor token_expert_output(token_output_dim);
    acti_out.dot(down_proj, token_expert_output);

    // Apply weight and accumulate to expert's output (no critical section
    // needed)
    token_expert_output.multiply_i(weight);
    size_t output_offset = token_idx * hidden_size;
    nntrainer::Tensor token_output =
      expert_output.getSharedDataTensor(token_output_dim, output_offset, true);

    token_output.add_i(token_expert_output);
  }
}

namespace {

/** @brief copy this expert's assigned rows into a contiguous [m, width] block */
template <typename T>
void gather_rows(const T *src, T *dst,
                 const std::vector<std::pair<unsigned, float>> &assignments,
                 unsigned int width) {
  for (size_t i = 0; i < assignments.size(); ++i)
    std::memcpy(dst + i * width,
                src + static_cast<size_t>(assignments[i].first) * width,
                static_cast<size_t>(width) * sizeof(T));
}

/**
 * @brief scatter [m, width] back to the token rows, scaled and accumulated
 * @note the accumulate is safe without an atomic only because one expert never
 * sees the same token twice -- topK returns distinct indices within a row.
 */
template <typename T>
void scatter_weighted_add(
  const T *src, T *dst,
  const std::vector<std::pair<unsigned, float>> &assignments,
  unsigned int width) {
  for (size_t i = 0; i < assignments.size(); ++i) {
    const float w = assignments[i].second;
    const T *s = src + i * width;
    T *d = dst + static_cast<size_t>(assignments[i].first) * width;
    for (unsigned int j = 0; j < width; ++j)
      d[j] = static_cast<T>(static_cast<float>(d[j]) +
                            static_cast<float>(s[j]) * w);
  }
}

/**
 * @brief NNTR_MOE_DBG=1: L2 norm + finite count of one tensor, first N calls.
 * @note the 35B has no MoE golden at its own geometry and on Orin the CPU
 * reference cannot run QINT4 experts at all, so this is the only way to see
 * whether a stage is zero / NaN / exploded without a reference to diff against.
 */
void moe_dbg(const char *tag, const nntrainer::Tensor &t) {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_MOE_DBG");
    return e != nullptr && e[0] == '1';
  }();
  if (!on)
    return;
  static int n = 0;
  if (n++ >= 24)
    return;
  const size_t len = t.size();
  double sq = 0.0, amax = 0.0;
  size_t bad = 0;
  for (size_t i = 0; i < len; ++i) {
    float v;
    if (t.getDataType() == ml::train::TensorDim::DataType::FP32)
      v = t.getData<float>()[i];
#ifdef ENABLE_FP16
    else if (t.getDataType() == ml::train::TensorDim::DataType::FP16)
      v = static_cast<float>(t.getData<_FP16>()[i]);
#endif
    else
      return;
    if (!std::isfinite(v)) {
      ++bad;
      continue;
    }
    sq += (double)v * v;
    if (std::fabs(v) > amax)
      amax = std::fabs(v);
  }
  std::fprintf(stderr, "[MOE-DBG] %-12s n=%zu l2=%.6g max=%.6g nonfinite=%zu\n",
               tag, len, std::sqrt(sq), amax, bad);
}

} // namespace

bool MoELayer::runGroupedMoE(
  nntrainer::RunLayerContext &context, const nntrainer::Tensor &input,
  nntrainer::Tensor &output,
  const std::vector<std::vector<std::pair<unsigned, float>>> &assign,
  unsigned int total_tokens, unsigned int hidden_size) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
  const unsigned int E = num_experts;

  // The pointer table is filled ONCE. Weight pointers are stable for the run
  // (the weight arena is allocate-once and FSU is off), and rebuilding it would
  // mean 3*E scale-buffer map lookups per layer per token.
  if (!moe_tbl_built) {
    if (!nntrainer::cuda::cuda_moe_new_ptr_table(3 * E, &moe_wptr, &moe_wsc))
      return false;
    auto fill = [&](unsigned int base,
                    const std::vector<unsigned int> &idx) -> bool {
      for (unsigned int e = 0; e < E; ++e) {
        auto &w = context.getWeight(idx[e]);
        const unsigned short *sc = nullptr;
        if (!nntrainer::cuda::cuda_fc_qs4cx_scales_to_uvm_fp16(
              w.getScale<float>(), w.getDim().width(), &sc))
          return false;
        moe_wptr[base + e] = w.getData<uint8_t>();
        moe_wsc[base + e] = sc;
      }
      return true;
    };
    // PROJECTION-MAJOR, and the layer requests up/gate/down in that order.
    // Swapping gate and up here yields silu(up)*gate: fluent garbage, no error.
    if (!fill(0, expert_up_proj_indices) ||
        !fill(E, expert_gate_proj_indices) ||
        !fill(2 * E, expert_down_proj_indices))
      return false;
    moe_tbl_built = true;
  }

  unsigned int A = 0, Wmax = 0;
  for (unsigned int e = 0; e < E; ++e) {
    const unsigned int m = static_cast<unsigned int>(assign[e].size());
    A += m;
    Wmax += (m + 63) / 64;
  }
  if (A == 0 || Wmax == 0)
    return false;

  nntrainer::cuda::MoePlan plan{};
  plan.wptr = moe_wptr;
  plan.wsc = moe_wsc;
  plan.off_up = 0;
  plan.off_gate = E;
  plan.off_down = 2 * E;
  if (!nntrainer::cuda::cuda_moe_plan_stage(A, total_tokens, topk, E, Wmax,
                                            &plan))
    return false;

  // Assignments expert-major (so a work item is one expert's contiguous rows),
  // plus the inverse token->slot map the combine step needs.
  for (size_t i = 0; i < (size_t)total_tokens * topk; ++i)
    plan.slots[i] = -1;
  std::vector<unsigned int> used(total_tokens, 0);
  unsigned int a = 0, w = 0;
  for (unsigned int e = 0; e < E; ++e) {
    const auto &v = assign[e];
    if (v.empty())
      continue;
    const unsigned int r0 = a;
    for (const auto &pr : v) {
      plan.rows[a] = static_cast<int>(pr.first);
      plan.wts[a] = pr.second;
      plan.slots[(size_t)pr.first * topk + used[pr.first]++] =
        static_cast<int>(a);
      ++a;
    }
    for (size_t t0 = 0; t0 < v.size(); t0 += 64) {
      plan.wl_e[w] = static_cast<int>(e);
      plan.wl_r0[w] = static_cast<int>(r0 + t0);
      plan.wl_n[w] = static_cast<int>(std::min<size_t>(64, v.size() - t0));
      ++w;
    }
  }

  // intermediate_size is not a member; it is the gate/up weight's OUTPUT width
  // (they are stored [in, out]), exactly as compute_expert_forward_batched
  // reads it off gate_proj.
  const unsigned int inter =
    context.getWeight(expert_gate_proj_indices[0]).getDim().width();
  if (!nntrainer::cuda::cuda_moe_expert_ffn_fp16(
        reinterpret_cast<const unsigned short *>(input.getData<_FP16>()),
        reinterpret_cast<unsigned short *>(output.getData<_FP16>()), plan, A, w,
        total_tokens, topk, hidden_size, inter))
    return false;
  // The whole FFN is issued undrained; the graph reads `output` next.
  nntrainer::cuda::StreamManager::Global().finish();
  return true;
#else
  (void)context;
  (void)input;
  (void)output;
  (void)assign;
  (void)total_tokens;
  (void)hidden_size;
  return false;
#endif
}

void MoELayer::compute_expert_forward_batched(
  nntrainer::ComputeOps *ops, nntrainer::RunLayerContext &context,
  const nntrainer::Tensor &input, nntrainer::Tensor &output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  nntrainer::Tensor &gate_proj, nntrainer::Tensor &up_proj,
  nntrainer::Tensor &down_proj, unsigned int hidden_size,
  const int *rows_dev, const float *wts_dev) {

  const unsigned int m = static_cast<unsigned int>(token_assignments.size());
  if (m == 0)
    return;
  // rows_dev/wts_dev non-null means the caller staged this expert's row list on
  // the device AND is holding a deferred-drain region open, so the gather, the
  // SwiGLU and the scatter must all stay on the device -- a host touch of any
  // of these buffers now would read work that has been issued and not drained.
  const bool dev_path = (rows_dev != nullptr && wts_dev != nullptr);

  const unsigned int intermediate_size = gate_proj.width();
  const auto tt = input.getTensorType();

  // {1,1,m,*} views of the pooled scratch. The row count MUST sit in height():
  // CudaComputeOps::fc reads M as batch*channel*height and HalfTensor::dot's
  // QS4CX arm reads it as height() alone, so the {m,1,1,*} shape this layer
  // reshapes its input to would compute M=1 on the host path and silently drop
  // m-1 rows.
  auto rows = [&](unsigned int idx, unsigned int width) {
    return context.getTensor(idx).getSharedDataTensor(
      nntrainer::TensorDim({1, 1, m, width}, tt), 0, true);
  };

  auto gathered = rows(gathered_in_idx, hidden_size);
  auto gate_out = rows(gate_out_idx, intermediate_size);
  auto up_out = rows(up_out_idx, intermediate_size);
  auto acti_out = rows(acti_out_idx, intermediate_size);
  auto expert_out = rows(expert_out_idx, hidden_size);

  const auto dt = input.getDataType();
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
  if (dev_path) {
    nntrainer::cuda::cuda_moe_gather_fp16(
      reinterpret_cast<const unsigned short *>(input.getData<_FP16>()),
      reinterpret_cast<unsigned short *>(gathered.getData<_FP16>()), rows_dev,
      m, hidden_size);
  } else
#endif
  {
    switch (dt) {
    case ml::train::TensorDim::DataType::FP32:
      gather_rows(input.getData<float>(), gathered.getData<float>(),
                  token_assignments, hidden_size);
      break;
#ifdef ENABLE_FP16
    case ml::train::TensorDim::DataType::FP16:
      gather_rows(input.getData<_FP16>(), gathered.getData<_FP16>(),
                  token_assignments, hidden_size);
      break;
#endif
    default:
      throw std::runtime_error("MoE: unsupported activation dtype for gather");
    }
  }

  if (!dev_path)
    moe_dbg("gathered", gathered);
  {
    nntrainer::LayerProfScope _p("  moe:3x fc", m == 1);
    ops->fc(gathered, gate_proj, gate_out);
    ops->fc(gathered, up_proj, up_out);
  }
  if (!dev_path) {
    moe_dbg("gate_out", gate_out);
    moe_dbg("up_out", up_out);
  }

  {
    nntrainer::LayerProfScope _p(dev_path ? "  moe:swiglu(dev)"
                                          : "  moe:swiglu(host)",
                                 m == 1);
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    if (dev_path) {
      nntrainer::cuda::cuda_moe_swiglu_fp16(
        reinterpret_cast<const unsigned short *>(gate_out.getData<_FP16>()),
        reinterpret_cast<const unsigned short *>(up_out.getData<_FP16>()),
        reinterpret_cast<unsigned short *>(acti_out.getData<_FP16>()),
        m * intermediate_size);
    } else
#endif
    {
      // dtype-generic SwiGLU, as in the per-token path: the free
      // nntrainer::swiglu() is FP32-only and reads its operands through an
      // unchecked getData<float>().
      acti_func.run_fn(gate_out, acti_out);
      acti_out.multiply_i(up_out);
    }
  }
  if (!dev_path)
    moe_dbg("swiglu", acti_out);

  {
    nntrainer::LayerProfScope _p("  moe:3x fc", m == 1);
    ops->fc(acti_out, down_proj, expert_out);
  }
  if (!dev_path)
    moe_dbg("expert_out", expert_out);

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
  if (dev_path) {
    nntrainer::cuda::cuda_moe_scatter_add_fp16(
      reinterpret_cast<const unsigned short *>(expert_out.getData<_FP16>()),
      reinterpret_cast<unsigned short *>(output.getData<_FP16>()), rows_dev,
      wts_dev, m, hidden_size);
    return;
  }
#endif
  switch (dt) {
  case ml::train::TensorDim::DataType::FP32:
    scatter_weighted_add(expert_out.getData<float>(), output.getData<float>(),
                         token_assignments, hidden_size);
    break;
#ifdef ENABLE_FP16
  case ml::train::TensorDim::DataType::FP16:
    scatter_weighted_add(expert_out.getData<_FP16>(), output.getData<_FP16>(),
                         token_assignments, hidden_size);
    break;
#endif
  default:
    throw std::runtime_error("MoE: unsupported activation dtype for scatter");
  }
}

void MoELayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                      unsigned int from, unsigned int to,
                                      bool training) {

  nntrainer::LayerProfScope _prof("qwen_moe(total)", (to - from) == 1);

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &router_logits_ = context.getTensor(router_logits_idx);
  nntrainer::Tensor &expert_mask = context.getTensor(expert_mask_idx);

  nntrainer::TensorDim input_step_dim = input_.getDim();
  nntrainer::TensorDim output_step_dim = output_.getDim();
  nntrainer::TensorDim router_logits_step_dim = router_logits_.getDim();

  input_step_dim.batch(1);
  output_step_dim.batch(1);
  router_logits_step_dim.batch(to - from);

  input_step_dim.height(to - from);
  output_step_dim.height(to - from);

  for (unsigned int b = 0; b < input_.batch(); ++b) {

    auto input = input_.getSharedDataTensor(
      input_step_dim, b * input_step_dim.getFeatureLen(), true);
    auto output = output_.getSharedDataTensor(
      output_step_dim, b * output_step_dim.getFeatureLen(), true);
    auto router_logits =
      router_logits_.getSharedDataTensor(router_logits_step_dim, 0, true);

    const unsigned batch_size = input.batch();
    const unsigned seq_len = input.height();
    const unsigned hidden_size = input.width();
    const unsigned total_tokens = batch_size * seq_len;

    // reshape input: [B,1,S,H] -> [B*S,1,1,H]
    input.reshape({total_tokens, 1, 1, hidden_size});

    // reshape output: [B,1,S,H] -> [B*S,1,1,H]
    output.reshape({total_tokens, 1, 1, hidden_size});
    output.setZero();
    // expert_mask is WRITE-ONLY on this path: nothing in this layer, and
    // nothing outside it (the tensor is layer-local, FORWARD_FUNC_LIFESPAN),
    // ever reads it. Zeroing it cost an 8 MB memset and filling it cost 32,768
    // virtual setValue() calls -- per layer, per chunk, 120 times over an
    // 8,353-token prefill -- to produce a buffer that is then discarded. The
    // non-incremental forwarding() still writes it; left alone because it is
    // not on any measured path and other models share the shape.

    // routing
    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);
    {
    nntrainer::LayerProfScope _pr("  moe:routing(host)", total_tokens == 1);
    nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
    // Routing is ALWAYS fp32: the gate weight and router_logits are FP32,
    // so an FP16 input makes HalfTensor::dot write FP16 bits into the FP32
    // logits buffer -- garbage top-k, no crash. Widen first on FP16 models.
    //
    // The router GEMM goes to the DEVICE when it can. On the host it is an
    // [T,H]x[H,E] sgemm -- 2.1 GMAC at T=4096 -- plus a 33 MB fp16->fp32
    // materialisation of the input, and it is the bulk of a routing block that
    // measured 13,637 ms of a 35 s prefill. Same arithmetic: the kernel widens
    // each fp16 element and accumulates in fp32, exactly as the clone+sgemm
    // does, so only the summation order differs.
    bool router_dev = false;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // NNTR_MOE_ROUTER_DEV=0 opts out (the host path stays the A/B reference).
    static const bool router_dev_on = []() {
      const char *e = std::getenv("NNTR_MOE_ROUTER_DEV");
      return !(e != nullptr && e[0] == '0');
    }();
    if (router_dev_on &&
        input.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        gate_weights.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const unsigned short *xp =
        reinterpret_cast<const unsigned short *>(input.getData<_FP16>());
      const float *wp = gate_weights.getData<float>();
      float *lp = router_logits.getData<float>();
      if (nntrainer::cuda::dev_accessible(xp) &&
          nntrainer::cuda::dev_accessible(wp) &&
          nntrainer::cuda::dev_accessible(lp)) {
        router_dev = nntrainer::cuda::cuda_moe_router_gemm_fp16(
          xp, wp, lp, total_tokens, hidden_size, num_experts);
        // The softmax/top-k below read these logits on the host.
        if (router_dev)
          nntrainer::cuda::StreamManager::Global().finish();
      }
    }
#endif
    if (!router_dev) {
      if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
        input.dot(gate_weights, router_logits);
      } else {
        nntrainer::Tensor input32 =
          input.clone(ml::train::TensorDim::DataType::FP32);
        input32.dot(gate_weights, router_logits);
      }
    }
    router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);
    auto topk_result = router_logits.topK(topk);
    auto topk_values = std::get<0>(topk_result);
    auto topk_indices = std::get<1>(topk_result);

    // norm_topk_prob
    topk_values.divide_i(topk_values.sum(3));

    const uint32_t *indices_data = topk_indices.getData<uint32_t>();
    // (the expert_mask fill that used to sit here was write-only -- see above)

    // Pre-compute expert token assignments for better performance
    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        unsigned expert_idx = indices_data[i * topk + k];
        float weight = topk_values.getValue<float>(i, 0, 0, k);
        expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }
    } // end routing scope

    // One expert, all of its tokens, three GEMMs. The dispatch table is taken
    // from the layer INPUT: that tensor carries the node's ContextData, while a
    // getSharedDataTensor view does not reliably carry it, so resolving
    // getOps() off `input` here would silently pick the host table.
    //
    // This also retires the old per-expert full-size temporaries, which
    // allocated {total_tokens,1,1,hidden} for EVERY non-empty expert -- at
    // prefill that is 256 x 4 MiB = ~1 GiB of alloc/free per layer per forward,
    // almost all of it zero.
    nntrainer::ComputeOps *ops = input_.getOps();

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // NNTR_CUDA_MOE_GROUPED=1: one grid over ALL routed experts instead of one
    // ops->fc per expert per projection. Opt-in until it is A/B'd against the
    // per-expert path below, which is the measured one.
    // OFF BY DEFAULT, and the reason is measured, not stylistic.
    //
    // The grouped kernel calls dp4a directly. The per-expert path calls
    // ops->fc, which at M >= 32 reaches the cuBLAS int8-IMMA arm -- the Tensor
    // Cores. That arm is worth ~3x on this hardware (gemma4, same build:
    // prefill 3,937 TPS with it vs 1,290 forced to dp4a), and no amount of
    // grouping recovers 3x. Measured on an 8,353-token prefill at chunk 4096,
    // i.e. m_e = 128, where the 64-row tile is TWO FULL TILES and the grouped
    // kernel is at its best possible shape:
    //   per-expert (ops->fc -> IMMA)  prefill 228.0 TPS  qwen_moe 23,998 ms
    //   grouped    (dp4a direct)      prefill 195.8 TPS  qwen_moe 30,001 ms
    // It also loses badly at decode (m_e = 1, tile 98% padding).
    //
    // So the grouping win is real but smaller than the Tensor-Core loss it
    // pays for. Kept because it becomes the right answer the moment it can
    // reach IMMA itself -- which needs the expert weights laid out as one
    // contiguous [E,N,K] tensor so a single cuBLAS batched call can cover
    // them, the way vLLM's fused MoE does. NNTR_CUDA_MOE_GROUPED=1 to measure.
    static const int grouped_env = []() {
      const char *e = std::getenv("NNTR_CUDA_MOE_GROUPED");
      return e ? (e[0] == '0' ? 0 : 1) : -1;
    }();
    const bool grouped_on = (grouped_env == 1);
    static const bool moe_dbg_gate = std::getenv("NNTR_MOE_DBG") != nullptr;
    if (grouped_on && !moe_dbg_gate &&
        input.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        output.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        runGroupedMoE(context, input, output, expert_assignments, total_tokens,
                      hidden_size)) {
      output.reshape({batch_size, 1, seq_len, hidden_size});
      return;
    }
#endif

    // Stage EVERY expert's row list once, in one buffer with per-expert
    // offsets, rather than reusing one buffer per expert: the drain is deferred
    // below, so expert i's gather may not have run when the host would have
    // overwritten the rows for expert i+1.
    const int *rows_all = nullptr;
    const float *wts_all = nullptr;
    std::vector<size_t> expert_off(num_experts, 0);
    bool moe_dev = false;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // NNTR_MOE_DBG reads these buffers on the host, which a deferred drain
    // makes meaningless -- it forces the host path so the counters stay true.
    static const bool moe_dbg_on = std::getenv("NNTR_MOE_DBG") != nullptr;
    if (!moe_dbg_on &&
        input.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        output.getDataType() == ml::train::TensorDim::DataType::FP16) {
      int *rp = nullptr;
      float *wp = nullptr;
      if (nntrainer::cuda::cuda_moe_stage(total_tokens * topk, &rp, &wp) &&
          nntrainer::cuda::dev_accessible(input.getData<_FP16>()) &&
          nntrainer::cuda::dev_accessible(output.getData<_FP16>()) &&
          nntrainer::cuda::dev_accessible(rp)) {
        size_t off = 0;
        for (unsigned int e = 0; e < num_experts; ++e) {
          expert_off[e] = off;
          for (const auto &a : expert_assignments[e]) {
            rp[off] = static_cast<int>(a.first);
            wp[off] = a.second;
            ++off;
          }
        }
        rows_all = rp;
        wts_all = wp;
        moe_dev = true;
      }
    }
    // One drain for the whole expert loop instead of one per fc. On integrated
    // that is 24 full cudaStreamSynchronize per layer per token at decode, and
    // 768 per layer per chunk at prefill; a 1,341-token prefill measured 61,440
    // of them, ~92% of this layer's time against ~8% of actual GEMM.
    if (moe_dev)
      nntrainer::cuda::StreamManager::Global().pushDeferDrain();
#endif

    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      const auto &assignments = expert_assignments[expert_idx];
      if (assignments.empty())
        continue;

      compute_expert_forward_batched(
        ops, context, input, output, assignments,
        context.getWeight(expert_gate_proj_indices[expert_idx]),
        context.getWeight(expert_up_proj_indices[expert_idx]),
        context.getWeight(expert_down_proj_indices[expert_idx]), hidden_size,
        moe_dev ? rows_all + expert_off[expert_idx] : nullptr,
        moe_dev ? wts_all + expert_off[expert_idx] : nullptr);
    }

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    if (moe_dev) {
      auto &sm = nntrainer::cuda::StreamManager::Global();
      sm.popDeferDrain();
      sm.finish(); // the region's single drain; `output` is host-read next
    }
#endif

    // reshape output: [B*S,1,1,H] -> [B,1,S,H]
    output.reshape({batch_size, 1, seq_len, hidden_size});
  }
}

void MoELayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void MoELayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // MoE layer does not support derivative calculation
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void MoELayer::calcGradient(nntrainer::RunLayerContext &context) {
  // MoE layer does not support gradient calculation
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void MoELayer::exportTo(nntrainer::Exporter &exporter,
                        const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this); // Save MoE specific properties
}

} // namespace causallm

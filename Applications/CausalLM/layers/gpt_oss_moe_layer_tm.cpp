/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
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
 * @file	gpt_oss_moe_layer_tm.cpp
 * @date	22 March 2026
 * @brief	GPT-OSS MoE layer using ThreadManager for async expert loading
 *              with look-ahead prefetch
 * @see		https://github.com/nnstreamer/
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * Key design:
 *   - Expert weight loading: ThreadManager::submit() → I/O worker
 *   - Expert GEMM compute: ThreadManager::parallel_for() → compute workers
 *   - Look-ahead: While computing expert[i], prefetch expert[i+1] weights
 *   - Async eviction: submit(deactivate) → I/O worker (non-blocking)
 *   - No OpenMP dependency
 */

#include <acti_func.h>
#include <algorithm>
#include <cmath>
#include <gpt_oss_moe_layer_tm.h>
#include <node_exporter.h>
#include <stdexcept>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

TMGptOssMoELayer::TMGptOssMoELayer() :
  LayerImpl(),
  num_experts(0),
  topk(0),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit()),
  expert_gate_proj_indices({}),
  expert_gate_bias_indices({}),
  expert_up_proj_indices({}),
  expert_up_bias_indices({}),
  expert_down_proj_indices({}),
  expert_down_bias_indices({}),
  gate_idx(std::numeric_limits<unsigned>::max()),
  gate_bias_idx(std::numeric_limits<unsigned>::max()),
  loaded_expert_deque({}),
  need_load({}),
  router_logits_idx(std::numeric_limits<unsigned>::max()),
  expert_mask_idx(std::numeric_limits<unsigned>::max()) {}

void TMGptOssMoELayer::finalize(nntrainer::InitLayerContext &context) {

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

  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const bool is_nchw = context.getFormat() == nntrainer::Tformat::NCHW;
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[SINGLE_INOUT_IDX] = in_dim;
  context.setOutputDimensions(output_dims);

  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();
  const unsigned int hidden_size = in_dim.width();

  // Gate (router) weight
  nntrainer::TensorDim gate_dim(
    1, is_nchw ? 1 : num_experts, is_nchw ? hidden_size : 1,
    is_nchw ? num_experts : hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32),
    is_nchw ? 0b0011 : 0b0101);

  gate_idx = context.requestWeight(
    gate_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gate", true);

  nntrainer::TensorDim gate_bias_dim(
    1, 1, 1, num_experts,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getActivationDataType()));
  gate_bias_idx =
    context.requestWeight(gate_bias_dim, weight_initializer, weight_regularizer,
                          1.0f, weight_decay, "gate_bias", false);

  // Expert weights (virtual tensors - lazy loaded)
  expert_gate_proj_indices.reserve(num_experts);
  expert_up_proj_indices.reserve(num_experts);
  expert_down_proj_indices.reserve(num_experts);
  expert_gate_bias_indices.reserve(num_experts);
  expert_up_bias_indices.reserve(num_experts);
  expert_down_bias_indices.reserve(num_experts);

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

  nntrainer::TensorDim expert_gate_bias_dim(
    1, 1, 1, intermediate_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getActivationDataType()),
    is_nchw ? 0b0011 : 0b0101);

  nntrainer::TensorDim expert_down_bias_dim(
    1, 1, 1, hidden_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getActivationDataType()),
    is_nchw ? 0b0011 : 0b0101);

  for (unsigned int i = 0; i < num_experts; ++i) {
    expert_up_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_up_" + std::to_string(i), false, true));

    expert_up_bias_indices.push_back(context.requestWeight(
      expert_gate_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_up_bias_" + std::to_string(i), false, true));

    expert_gate_proj_indices.push_back(context.requestWeight(
      expert_gate_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_" + std::to_string(i), false, true));

    expert_gate_bias_indices.push_back(context.requestWeight(
      expert_gate_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_gate_bias_" + std::to_string(i), false, true));

    expert_down_proj_indices.push_back(context.requestWeight(
      expert_down_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_" + std::to_string(i), false, true));

    expert_down_bias_indices.push_back(context.requestWeight(
      expert_down_bias_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay,
      "expert_down_bias_" + std::to_string(i), false, true));

    need_load.push_back(true);
  }

  // Intermediate tensors
  const unsigned batch_size = in_dim.batch();
  const unsigned seq_len = in_dim.height();
  const unsigned total_tokens = batch_size * seq_len;

  router_logits_idx =
    context.requestTensor({total_tokens, 1, 1, num_experts}, "router_logits",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  expert_mask_idx =
    context.requestTensor({num_experts, 1, topk, total_tokens}, "expert_mask",
                          nntrainer::Initializer::ZEROS, false,
                          nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
}

void TMGptOssMoELayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {}

nntrainer::CompletionToken
TMGptOssMoELayer::asyncActivateExpert(nntrainer::RunLayerContext &context,
                                       int expert_idx) {
  auto &tm = nntrainer::ThreadManager::Global();
  return tm.submit([&context, this, expert_idx] {
    context.getWeight(expert_gate_proj_indices[expert_idx]).activate();
    context.getWeight(expert_up_proj_indices[expert_idx]).activate();
    context.getWeight(expert_down_proj_indices[expert_idx]).activate();
    context.getWeight(expert_gate_bias_indices[expert_idx]).activate();
    context.getWeight(expert_up_bias_indices[expert_idx]).activate();
    context.getWeight(expert_down_bias_indices[expert_idx]).activate();
  });
}

nntrainer::CompletionToken
TMGptOssMoELayer::asyncDeactivateExpert(nntrainer::RunLayerContext &context,
                                         int expert_idx) {
  auto &tm = nntrainer::ThreadManager::Global();
  return tm.submit([&context, this, expert_idx] {
    context.getWeight(expert_gate_proj_indices[expert_idx]).deactivate();
    context.getWeight(expert_up_proj_indices[expert_idx]).deactivate();
    context.getWeight(expert_down_proj_indices[expert_idx]).deactivate();
    context.getWeight(expert_gate_bias_indices[expert_idx]).deactivate();
    context.getWeight(expert_up_bias_indices[expert_idx]).deactivate();
    context.getWeight(expert_down_bias_indices[expert_idx]).deactivate();
  });
}

void TMGptOssMoELayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &router_logits_ = context.getTensor(router_logits_idx);

  nntrainer::TensorDim input_step_dim = input_.getDim();
  nntrainer::TensorDim output_step_dim = output_.getDim();
  nntrainer::TensorDim router_logits_step_dim = router_logits_.getDim();

  input_step_dim.batch(1);
  output_step_dim.batch(1);
  router_logits_step_dim.batch(to - from);

  input_step_dim.height(to - from);
  output_step_dim.height(to - from);

  auto &tm = nntrainer::ThreadManager::Global();

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

    input.reshape({total_tokens, 1, 1, hidden_size});
    output.reshape({total_tokens, 1, 1, hidden_size});

    // ─── Step 1: Routing ──────────────────────────────────
    nntrainer::Tensor &gate_weights = context.getWeight(gate_idx);
    input.dot(gate_weights, router_logits);
    router_logits.apply(nntrainer::ActiFunc::softmax<float>, router_logits);

    // Get extra top-k for LRU prediction
    auto extra_topk_result = router_logits.topK(topk + 3);
    auto extra_topk_values = std::get<0>(extra_topk_result);
    auto extra_topk_indices = std::get<1>(extra_topk_result);
    std::deque<int> extra_top_k = {};
    extra_topk_values.divide_i(extra_topk_values.sum(3));
    const uint32_t *extra_indices_data =
      extra_topk_indices.getData<uint32_t>();

    for (int i = static_cast<int>(total_tokens) - 1; i >= 0; --i) {
      for (int k = 0; k < static_cast<int>(topk + 3); ++k) {
        unsigned expert_idx = extra_indices_data[i * topk + k];
        extra_top_k.push_back(expert_idx);
      }
    }

    auto topk_result = router_logits.topK(topk);
    auto topk_values = std::get<0>(topk_result);
    auto topk_indices = std::get<1>(topk_result);
    topk_values.divide_i(topk_values.sum(3));

    // ─── Step 2: Build expert assignments ─────────────────
    const uint32_t *indices_data = topk_indices.getData<uint32_t>();
    std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments(
      num_experts);

    for (int i = 0; i < static_cast<int>(total_tokens); ++i) {
      for (int k = 0; k < static_cast<int>(topk); ++k) {
        unsigned expert_idx = indices_data[i * topk + k];
        float weight = topk_values.getValue<float>(i, 0, 0, k);
        expert_assignments[expert_idx].emplace_back(i, weight);
      }
    }

    // Collect active experts (those with assigned tokens)
    std::vector<int> active_experts;
    for (int expert_idx = 0; expert_idx < static_cast<int>(num_experts);
         ++expert_idx) {
      if (!expert_assignments[expert_idx].empty())
        active_experts.push_back(expert_idx);
    }

    // Pre-allocate per-expert output tensors
    std::vector<nntrainer::Tensor> expert_outputs(num_experts);
    for (int expert_idx : active_experts) {
      expert_outputs[expert_idx] = nntrainer::Tensor(
        total_tokens, 1, 1, hidden_size, output.getTensorType());
    }

    // ─── Step 3: Look-ahead expert loading pipeline ───────
    //
    // Pipeline: for each active expert i:
    //   1. wait for expert[i] weights to finish loading
    //   2. submit(load expert[i+1]) — prefetch on I/O worker
    //   3. parallel_for(compute expert[i]) — GEMM on compute workers
    //   4. submit(deactivate expert[i]) — async eviction if cache full
    //
    // This overlaps disk I/O with GEMM compute.

    // Track async load tokens per expert
    std::unordered_map<int, nntrainer::CompletionToken> load_tokens;

    // Track async eviction tokens (must complete before reuse)
    std::vector<nntrainer::CompletionToken> eviction_tokens;

    // Phase 3a: Kick off loading for first 2 experts (look-ahead = 1)
    const size_t n_active = active_experts.size();
    const size_t lookahead = std::min(static_cast<size_t>(2), n_active);

    for (size_t i = 0; i < lookahead; ++i) {
      int eidx = active_experts[i];
      if (need_load[eidx]) {
        load_tokens[eidx] = asyncActivateExpert(context, eidx);
      }
      // else: already cached, no token needed (will skip wait)
    }

    // Phase 3b: Process each active expert sequentially with pipeline
    for (size_t i = 0; i < n_active; ++i) {
      int eidx = active_experts[i];

      // Wait for current expert's weights to load
      if (load_tokens.count(eidx)) {
        load_tokens[eidx].wait();
        load_tokens.erase(eidx);
        need_load[eidx] = false;
        loaded_expert_deque.push_back(eidx);
        iteration_map[eidx] = --loaded_expert_deque.end();
      }

      // Prefetch next expert (look-ahead)
      if (i + lookahead < n_active) {
        int next_eidx = active_experts[i + lookahead];
        if (need_load[next_eidx]) {
          load_tokens[next_eidx] = asyncActivateExpert(context, next_eidx);
        }
      }

      // Compute expert GEMM (on compute workers via parallel_for internally)
      compute_expert_forward(
        input, expert_outputs[eidx], expert_assignments[eidx],
        context.getWeight(expert_gate_proj_indices[eidx]),
        context.getWeight(expert_up_proj_indices[eidx]),
        context.getWeight(expert_down_proj_indices[eidx]),
        context.getWeight(expert_gate_bias_indices[eidx]),
        context.getWeight(expert_up_bias_indices[eidx]),
        context.getWeight(expert_down_bias_indices[eidx]), hidden_size);
    }

    // ─── Step 4: LRU update with prediction ──────────────
    for (int i = extra_top_k.size() - 1; i >= 0; i--) {
      if (iteration_map.find(extra_top_k[i]) != iteration_map.end()) {
        loaded_expert_deque.erase(iteration_map[extra_top_k[i]]);
        loaded_expert_deque.push_back(extra_top_k[i]);
        iteration_map[extra_top_k[i]] = --loaded_expert_deque.end();
      }
    }

    // ─── Step 5: Async eviction (on I/O worker) ──────────
    while (loaded_expert_deque.size() > max_cached_experts) {
      int target_idx = loaded_expert_deque.front();
      loaded_expert_deque.pop_front();
      iteration_map.erase(target_idx);
      need_load[target_idx] = true;

      // Deactivate on I/O worker — does NOT block compute
      eviction_tokens.push_back(asyncDeactivateExpert(context, target_idx));
    }

    // Wait for all evictions to complete before combining outputs
    // (evictions write to swap device; we must ensure consistency)
    for (auto &token : eviction_tokens)
      token.wait();

    // ─── Step 6: Combine expert outputs ──────────────────
    int init = 0;
    for (int expert_idx : active_experts) {
      if (!init) {
        output.copyData(expert_outputs[expert_idx]);
        ++init;
      } else {
        output.add_i(expert_outputs[expert_idx]);
      }
    }

    output.reshape({batch_size, 1, seq_len, hidden_size});
  }
}

inline void TMGptOssMoELayer::compute_expert_forward(
  const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
  const std::vector<std::pair<unsigned, float>> &token_assignments,
  const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
  const nntrainer::Tensor &down_proj, const nntrainer::Tensor &gate_bias,
  const nntrainer::Tensor &up_bias, const nntrainer::Tensor &down_bias,
  unsigned int hidden_size) {

  const unsigned intermediate_size = gate_proj.width();
  const unsigned num_tokens = token_assignments.size();

  if (num_tokens == 0)
    return;

  nntrainer::TensorDim token_input_dim({1, 1, num_tokens, hidden_size},
                                       input.getTensorType());
  nntrainer::TensorDim intermediate_dim({1, 1, num_tokens, intermediate_size},
                                        input.getTensorType());
  nntrainer::TensorDim token_output_dim({1, 1, num_tokens, hidden_size},
                                        input.getTensorType());
  nntrainer::TensorDim out_step_dim({1, 1, 1, hidden_size},
                                    input.getTensorType());
  nntrainer::TensorDim step_dim({1, 1, 1, intermediate_size},
                                input.getTensorType());

  nntrainer::Tensor gate_out(intermediate_dim);
  nntrainer::Tensor acti_out(intermediate_dim);
  nntrainer::Tensor up_out(intermediate_dim);
  nntrainer::Tensor token_input(token_input_dim);
  nntrainer::Tensor token_expert_output(token_output_dim);

  unsigned token_idx = token_assignments[0].first;
  float weight = token_assignments[0].second;

  auto &tm = nntrainer::ThreadManager::Global();

  if (num_tokens > 1) {
    // Prefill: copy selected tokens into contiguous batch using parallel_for
    tm.parallel_for(0, num_tokens, [&](size_t i) {
      const unsigned tidx = token_assignments[i].first;
      nntrainer::Tensor src_view = input.getSharedDataTensor(
        {1, 1, 1, hidden_size}, tidx * hidden_size, true);
      nntrainer::Tensor dst_view = token_input.getSharedDataTensor(
        {1, 1, 1, hidden_size}, i * hidden_size, true);
      dst_view.copyData(src_view);
    });
  } else {
    size_t token_offset = token_idx * hidden_size;
    token_input =
      input.getSharedDataTensor(token_input_dim, token_offset, true);
  }

  // Gate projection
  token_input.dot(gate_proj, gate_out);
  gate_out.add(gate_bias, gate_out);
  nntrainer::clamp(gate_out.getData(), gate_out.getData(),
                   num_tokens * intermediate_size,
                   std::numeric_limits<float>::lowest(), limit);

  // Up projection
  token_input.dot(up_proj, up_out);
  up_out.add_i(up_bias);
  nntrainer::clamp(up_out.getData(), up_out.getData(),
                   num_tokens * intermediate_size, -limit, limit);

  // SWiGLU activation using parallel_for for multi-token
  up_out.add_i(1);
  if (num_tokens > 2) {
    tm.parallel_for(0, num_tokens, [&](size_t i) {
      const unsigned offset = acti_out.getIndex(0, 0, i, 0);
      nntrainer::swiglu(acti_out.width(), acti_out.getData<float>() + offset,
                        gate_out.getData<float>() + offset,
                        up_out.getData<float>() + offset, alpha);
    });
  } else {
    for (size_t i = 0; i < num_tokens; ++i) {
      const unsigned offset = acti_out.getIndex(0, 0, i, 0);
      nntrainer::swiglu(acti_out.width(), acti_out.getData<float>() + offset,
                        gate_out.getData<float>() + offset,
                        up_out.getData<float>() + offset, alpha);
    }
  }

  // Down projection
  acti_out.dot(down_proj, token_expert_output);
  token_expert_output.add_i(down_bias);

  // Accumulate weighted expert output using parallel_for for multi-token
  if (num_tokens > 2) {
    tm.parallel_for(0, num_tokens, [&](size_t i) {
      unsigned tidx = token_assignments[i].first;
      float w = token_assignments[i].second;
      size_t output_offset = tidx * hidden_size;
      nntrainer::Tensor token_output =
        expert_output.getSharedDataTensor(out_step_dim, output_offset, true);
      nntrainer::Tensor target = token_expert_output.getSharedDataTensor(
        out_step_dim, i * hidden_size, true);
      target.multiply_i(w);
      token_output.add(target, token_output);
    });
  } else {
    for (size_t i = 0; i < num_tokens; ++i) {
      token_idx = token_assignments[i].first;
      weight = token_assignments[i].second;
      size_t output_offset = token_idx * hidden_size;
      nntrainer::Tensor token_output =
        expert_output.getSharedDataTensor(out_step_dim, output_offset, true);
      nntrainer::Tensor target = token_expert_output.getSharedDataTensor(
        out_step_dim, i * hidden_size, true);
      target.multiply_i(weight);
      token_output.add(target, token_output);
    }
  }
}

void TMGptOssMoELayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void TMGptOssMoELayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("MoE layer does not support derivative calculation");
}

void TMGptOssMoELayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("MoE layer does not support gradient calculation");
}

void TMGptOssMoELayer::exportTo(nntrainer::Exporter &exporter,
                                 const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this);
}

} // namespace causallm

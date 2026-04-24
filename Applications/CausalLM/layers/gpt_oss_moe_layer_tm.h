// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   gpt_oss_moe_layer_tm.h
 * @date   22 March 2026
 * @brief  GPT-OSS MoE layer using ThreadManager for async expert loading
 *         with look-ahead prefetch (replaces OpenMP + manual mutex version)
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @todo   This layer does not support backwarding yet.
 */

#ifndef __GPT_OSS_MOE_LAYER_TM_H__
#define __GPT_OSS_MOE_LAYER_TM_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <causallm_common_properties.h>
#include <common_properties.h>
#include <completion_token.h>
#include <layer_impl.h>
#include <list>
#include <thread_manager.h>

namespace causallm {

/**
 * @class   TMGptOssMoELayer
 * @brief   Mixture of Expert Layer using ThreadManager
 *
 * Key differences from CachedSlimGptOssMoELayer:
 * - Expert weight loading via ThreadManager::submit() (I/O worker)
 *   instead of OpenMP threads doing disk I/O
 * - Look-ahead prefetch: pre-load next expert weights while computing
 *   current expert on compute workers
 * - Expert GEMM via ThreadManager::parallel_for() (spin-wait barrier)
 *   instead of OpenMP parallel for
 * - Async eviction via submit() instead of blocking deactivate on
 *   compute threads
 * - No mutex for LRU cache: I/O operations are serialized on I/O worker
 */
class TMGptOssMoELayer : public nntrainer::LayerImpl {
public:
  TMGptOssMoELayer();
  ~TMGptOssMoELayer() = default;

  TMGptOssMoELayer(TMGptOssMoELayer &&rhs) noexcept = default;
  TMGptOssMoELayer &operator=(TMGptOssMoELayer &&rhs) = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  void calcGradient(nntrainer::RunLayerContext &context) override;
  void setProperty(const std::vector<std::string> &values) override;
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  const std::string getType() const override { return TMGptOssMoELayer::type; };
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "gpt_oss_moe_tm";

private:
  unsigned int num_experts;
  unsigned int topk;
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit>
    moe_props;

  // weight indices
  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_gate_bias_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_up_bias_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  std::vector<unsigned int> expert_down_bias_indices;
  unsigned int gate_idx;
  unsigned int gate_bias_idx;

  // LRU cache state
  static constexpr unsigned int max_cached_experts = 16;
  std::list<int> loaded_expert_deque;
  std::unordered_map<int, std::list<int>::iterator> iteration_map;
  std::vector<bool> need_load;

  // Intermediate tensor indices
  unsigned int router_logits_idx;
  unsigned int expert_mask_idx;

  float alpha = 1.702;
  float limit = 7.0;

  /**
   * @brief Activate (load) expert weights via I/O worker
   * @param context Run layer context
   * @param expert_idx Expert index to load
   * @return CompletionToken for tracking load completion
   */
  nntrainer::CompletionToken
  asyncActivateExpert(nntrainer::RunLayerContext &context, int expert_idx);

  /**
   * @brief Deactivate (unload) expert weights via I/O worker
   * @param context Run layer context
   * @param expert_idx Expert index to unload
   * @return CompletionToken for tracking unload completion
   */
  nntrainer::CompletionToken
  asyncDeactivateExpert(nntrainer::RunLayerContext &context, int expert_idx);

  /**
   * @brief Expert forward computation
   */
  inline void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, const nntrainer::Tensor &gate_bias,
    const nntrainer::Tensor &up_bias, const nntrainer::Tensor &down_bias,
    unsigned int hidden_size);
};

} // namespace causallm

#endif /** __cplusplus */
#endif /** __GPT_OSS_MOE_LAYER_TM_H__ */

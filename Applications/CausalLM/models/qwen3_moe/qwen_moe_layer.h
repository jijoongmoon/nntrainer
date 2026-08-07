// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   moe_layer.h
 * @date   09 June 2025
 * @brief  This is Mixture of Expert Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This file is part of the Mixture of Expert Layer implementation.
 *         It does not support shared experts.
 *         This layer is implemented based on the LLama-MoE.
 *         For more information, please refer to the following link:
 *         https://arxiv.org/pdf/2406.16554
 * @todo   This layer does not support backwarding yet.
 */

#ifndef __MOE_LAYER_H__
#define __MOE_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

#include <acti_func.h>
#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {
class ComputeOps;
}

namespace causallm {

/**
 * @class   MoELayer
 * @brief   Mixture of Expert Layer
 */
class WIN_EXPORT MoELayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Mixture of Expert Layer
   */
  MoELayer();

  /**
   * @brief     Destructor of Mixture of Expert Layer
   */
  ~MoELayer() = default;

  /**
   * @brief  Move constructor.
   *  @param[in] MoELayer &&
   */
  MoELayer(MoELayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs MoELayer to be moved.
   */
  MoELayer &operator=(MoELayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ml::train::ExportMethods
   * &methods)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return MoELayer::type; };

  /**
   * @brief Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "qwen_moe"; /**< type of the layer */

private:
  unsigned int num_experts;      /**< number of experts */
  unsigned int topk;             /**< number of experts per token, i.e., topk */
  nntrainer::ActiFunc acti_func; /**< activation function for the expert */
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::MoEActivation>
    moe_props;

  // weight indeices
  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  unsigned int gate_idx;

  // Intermediate tensor indices
  unsigned int router_logits_idx;
  unsigned int expert_mask_idx;

  // Batched per-expert scratch. These MUST come from the tensor pool rather
  // than the stack: on a CUDA run cuda::dev_accessible() is false for ordinary
  // heap memory, so a stack-allocated intermediate makes every QS4CX arm of
  // CudaComputeOps::fc decline (and, for the first projection, would hand a
  // kernel an output pointer the device cannot write). Sized for the worst
  // case, which is every token routed to one expert.
  // Per-layer device weight-pointer table for the grouped MoE kernel, filled
  // once (weight pointers are stable for the run) and owned for the process
  // lifetime. Per LAYER, not shared: each layer has its own experts.
  const unsigned char **moe_wptr = nullptr;
  const unsigned short **moe_wsc = nullptr;
  bool moe_tbl_built = false;

  unsigned int gathered_in_idx;
  unsigned int gate_out_idx;
  unsigned int up_out_idx;
  unsigned int acti_out_idx;
  unsigned int expert_out_idx;

  /**
   * @brief expert forward computation without memory copies
   * @param input Input tensor (reshaped to [total_tokens, 1, 1, hidden_size])
   * @param output Output tensor to accumulate results
   * @param token_assignments Vector of (token_index, weight) pairs for this
   * expert
   * @param gate_proj Gate projection weight tensor
   * @param up_proj Up projection weight tensor
   * @param down_proj Down projection weight tensor
   * @param hidden_size Hidden dimension size
   */
  inline void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);

  /**
   * @brief expert forward computation without critical section
   * @param input Input tensor (reshaped to [total_tokens, 1, 1, hidden_size])
   * @param expert_output Expert-specific output tensor
   * @param token_assignments Vector of (token_index, weight) pairs for this
   * expert
   * @param gate_proj Gate projection weight tensor
   * @param up_proj Up projection weight tensor
   * @param down_proj Down projection weight tensor
   * @param hidden_size Hidden dimension size
   */
  inline void compute_expert_forward_no_critical(
    const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);

  /**
   * @brief one expert, all of its tokens, three GEMMs
   *
   * Gathers this expert's assigned rows into a contiguous {1,1,m,H} buffer,
   * runs ONE projection per GEMM through the supplied ComputeOps, and scatters
   * the weighted result back. The per-token variants above issue 3*m GEMVs
   * instead, which at prefill is 24,576 dispatches per layer against 768 here
   * -- and on a CUDA run every one of those dispatches would also have to
   * re-enter the QS4CX dispatch ladder.
   *
   * @param ops dispatch table taken from the layer INPUT (which carries the
   * node's ContextData); views do not reliably carry it.
   * @param context run context, for the pooled scratch tensors
   * @param input  input reshaped to [total_tokens, 1, 1, hidden_size]
   * @param output output reshaped to [total_tokens, 1, 1, hidden_size]
   * @param token_assignments (token index, routing weight) for this expert
   */
  /**
   * @brief The whole expert FFN for one layer in ~7 launches, all experts in
   *        one grid. Returns false (having written nothing) if anything is
   *        unavailable, so the caller falls through to the per-expert path.
   */
  bool runGroupedMoE(
    nntrainer::RunLayerContext &context, const nntrainer::Tensor &input,
    nntrainer::Tensor &output,
    const std::vector<std::vector<std::pair<unsigned, float>>> &assign,
    unsigned int total_tokens, unsigned int hidden_size);

  void compute_expert_forward_batched(
    nntrainer::ComputeOps *ops, nntrainer::RunLayerContext &context,
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    nntrainer::Tensor &gate_proj, nntrainer::Tensor &up_proj,
    nntrainer::Tensor &down_proj, unsigned int hidden_size,
    const int *rows_dev = nullptr, const float *wts_dev = nullptr);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __MOE_LAYER_H__ */

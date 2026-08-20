// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   multiply_layer.cpp
 * @date   10 Oct 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is multiply layer class (operation layer)
 *
 */

#include <multiply_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_stream_manager.h>
#endif
#endif

namespace nntrainer {

void MultiplyLayer::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(multiply_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(multiply_props).get();
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void MultiplyLayer::forwarding_operation(const Tensor &input0,
                                         const Tensor &input1, Tensor &hidden) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#ifdef ENABLE_FP16
  // Device path: one stream-ordered eltwise kernel. Bit-identical to the
  // host loop (an fp16 x fp16 product is exact in fp32; one rn round), and it
  // removes both the full-stream drain below and the single-threaded host
  // multiply over the whole chunk (10 nodes/chunk on the 35B's attention
  // gate). Size-gated: at decode (m=1, ~4K elements) the launch+sync
  // machinery costs more than the host loop it replaces -- the un-gated
  // version measured ~-2.5 ms/token of decode. NNTR_EW_DEV=0 kills the path.
  static const bool g_ew_dev = []() {
    const char *e = std::getenv("NNTR_EW_DEV");
    return e == nullptr || e[0] != '0';
  }();
  // Under CUDA-graph capture the size gate must yield: the host loop below
  // reads producer outputs that have NOT run (capture records, not executes)
  // and leaves this node OUT of the graph entirely — the replayed forward
  // then reuses the capture-time garbage every token (the 35B M2B decode
  // corruption, 2026-08-21). A ~4K-element kernel is ~10 us; correctness of
  // the capture beats the eager-decode micro-cost, and eager decode still
  // takes the host loop as before.
  const bool in_capture =
    nntrainer::cuda::StreamManager::Global().isCapturing();
  if (g_ew_dev && (input0.size() >= 32768 || in_capture) &&
      input0.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      input1.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      hidden.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      input0.size() == input1.size() && input0.size() == hidden.size() &&
      input0.getContiguous() && input1.getContiguous() &&
      hidden.getContiguous()) {
    const auto *ap =
      reinterpret_cast<const unsigned short *>(input0.getData<_FP16>());
    const auto *bp =
      reinterpret_cast<const unsigned short *>(input1.getData<_FP16>());
    auto *op = reinterpret_cast<unsigned short *>(hidden.getData<_FP16>());
    if (nntrainer::cuda::dev_accessible(ap) &&
        nntrainer::cuda::dev_accessible(bp) &&
        nntrainer::cuda::dev_accessible(op) &&
        nntrainer::cuda::cuda_mul_fp16(ap, bp, op,
                                       (unsigned int)input0.size()))
      return;
  }
#endif
  // Tensor::multiply is a pure host element loop with no ComputeOps dispatch:
  // on a CUDA graph both operands are device-written (attention output, gate
  // sigmoid) and the only ordering is the producers' per-op drains. Inside a
  // deferred-drain or async region this supplies the missing drain; in plain
  // sync mode it is a no-op.
  nntrainer::cuda::drain_if_async();
#endif
  input0.multiply(input1, hidden);
}

void MultiplyLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  BinaryOperationLayer::incremental_forwarding(context, from, to, training);
}

void MultiplyLayer::calcDerivative(RunLayerContext &context) {
  context.getOutgoingDerivative(0).copy(
    context.getIncomingDerivative(SINGLE_INOUT_IDX)
      .multiply(context.getInput(1)));

  context.getOutgoingDerivative(1).copy(
    context.getIncomingDerivative(SINGLE_INOUT_IDX)
      .multiply(context.getInput(0)));
}

void MultiplyLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, multiply_props);
  if (!remain_props.empty()) {
    std::string msg = "[MultiplyLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */

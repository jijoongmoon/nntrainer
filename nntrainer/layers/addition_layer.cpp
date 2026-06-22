// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   addition_layer.cpp
 * @date   30 July 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Addition Layer Class for Neural Network
 *
 */

#include <addition_layer.h>
#include <cstdio>
#include <cstdlib>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <blas_kernel_interface.h>
#endif
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_elementwise.h>
#include <cuda_runtime.h>
#endif
#include <layer_context.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void AdditionLayer::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(add_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(add_props).get();
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void AdditionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  /** @todo check possibility for in-place of addition layer */
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    const Tensor &input_ = context.getInput(idx);
    if (!idx) {
      hidden_.copy(input_);
    } else {
      hidden_.add_i(input_);
    }
  }
}

void AdditionLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  bool is_prefill = !from;
  if (skip_prefill && is_prefill)
    return;

  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  TensorDim hidden_dim = hidden_.getDim();
  TensorDim hidden_step_dim = hidden_dim;

  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // GPU residual add (the common 2-input fp16 device case): out = in0 + in1
    // in one kernel, keeping the residual on-device. Opt-in (NNTR_CUDA_ELTWISE).
    if (context.getNumInputs() == 2 &&
        hidden_.getDataType() == ml::train::TensorDim::DataType::FP16) {
      static const bool gpu = std::getenv("NNTR_CUDA_ELTWISE") != nullptr;
      if (gpu) {
        const Tensor &i0 = context.getInput(0);
        const Tensor &i1 = context.getInput(1);
        TensorDim sd0 = i0.getDim(), sd1 = i1.getDim();
        sd0.batch(1);
        sd0.height(to - from);
        sd1.batch(1);
        sd1.height(to - from);
        Tensor s0 =
          i0.getSharedDataTensor(sd0, b * i0.getDim().getFeatureLen(), true);
        Tensor s1 =
          i1.getSharedDataTensor(sd1, b * i1.getDim().getFeatureLen(), true);
        auto *a = reinterpret_cast<const unsigned short *>(s0.getData<_FP16>());
        auto *bb = reinterpret_cast<const unsigned short *>(s1.getData<_FP16>());
        auto *o =
          reinterpret_cast<unsigned short *>(hidden_step.getData<_FP16>());
        cudaPointerAttributes pa{};
        bool dev = cudaPointerGetAttributes(&pa, a) == cudaSuccess &&
                   (pa.type == cudaMemoryTypeManaged ||
                    pa.type == cudaMemoryTypeDevice);
        cudaGetLastError();
        if (dev &&
            cuda::cuda_add_fp16(a, bb, o, (unsigned int)hidden_step.size()))
          continue;
      }
    }
#endif

    /** @todo check possibility for in-place of addition layer */
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      TensorDim input_dim = input_.getDim();

      TensorDim input_step_dim = input_dim;
      input_step_dim.batch(1);
      input_step_dim.height(to - from);

      Tensor input_step = input_.getSharedDataTensor(
        input_step_dim, b * input_dim.getFeatureLen(), true);
      if (!idx) {
        hidden_step.copy(input_step);
      } else {
        hidden_step.add_i(input_step);
      }
    }
  }

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // Paper §3.6 chain-residency hand-off: publish the residual_add
  // output to a GPU TensorBacking so a downstream RMSNorm consumer
  // (specifically fused_rmsnorm_quant_resident_fp32) can read its
  // input from cl_mem and skip the per-call host upload. Gated by
  // NNTR_RESIDUAL_PUBLISH=1; default off keeps the original
  // pure-CPU contract intact. GPU-only: requires the OpenCL backend.
  static const bool publish_on =
    std::getenv("NNTR_RESIDUAL_PUBLISH") != nullptr;
  if (publish_on &&
      hidden_.getDataType() == ml::train::TensorDim::DataType::FP32 &&
      hidden_.batch() == 1) {
    publish_host_fp32_to_backing(hidden_, hidden_.getName());
    static int trip = 0;
    if (!trip && std::getenv("NNTR_RESIDUAL_PUBLISH_TRIP") != nullptr) {
      trip = 1;
      std::fprintf(stderr,
                   "[RESIDUAL-PUBLISH] first publish: name=%s host_ptr=%p\n",
                   hidden_.getName().c_str(),
                   (void *)hidden_.getData<uint8_t>());
      std::fflush(stderr);
    }
  }
#endif
}

void AdditionLayer::calcDerivative(RunLayerContext &context) {

  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    /**
     * TODO: replace this with tensor assignment during optimization.
     * Tensor assignment needs to make sure that the previous connected layers
     * are not inplace
     */
    context.getOutgoingDerivative(idx).copy(
      context.getIncomingDerivative(SINGLE_INOUT_IDX));
  }
}

void AdditionLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, add_props);
  if (!remain_props.empty()) {
    std::string msg = "[AdditionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void AdditionLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  for (size_t i = 0; i < context.getNumInputs(); ++i) {
    context.updateInput(i, input_dimensions[0]);
  }
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

} /* namespace nntrainer */

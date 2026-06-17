// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file   addition_layer_cl.cpp
 * @date   28 May 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yash Singh yash.singh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief	 This is Addition Layer Class Class for Neural Network with OpenCl
 * implementation
 */

#include <addition_layer_cl.h>
#include <attention_kernels.h>
#include <blas_kernel_interface.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void AdditionLayerCL::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(add_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(add_props).get();
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void AdditionLayerCL::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  /** @todo check possibility for in-place of addition layer */
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    const Tensor &input_ = context.getInput(idx);
    if (!idx) {
      hidden_.copy(input_);
    } else {
      add_i_cl(hidden_, input_);
    }
  }
}

void AdditionLayerCL::incremental_forwarding(RunLayerContext &context,
                                             unsigned int from, unsigned int to,
                                             bool training) {
  if (skip_prefill && from == 0)
    return;
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  TensorDim hidden_dim = hidden_.getDim();
  TensorDim hidden_step_dim = hidden_dim;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  // FP32 fast-path bypasses Tensor::copy and add_i_cl which both
  // misbehave in this code path (see addition_layer_cl debug notes,
  // commit message for [v8c] Addition CL: bypass copy+add_cl). The
  // result is bit-equivalent to AdditionLayer (CPU) for FP32 inputs.
  // FP16 falls through to the original CL path.
  const bool fp32_fast =
    hidden_.getDataType() == ml::train::TensorDim::DataType::FP32;

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    /** @todo check possibility for in-place of addition layer */
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      TensorDim input_dim = input_.getDim();

      TensorDim input_step_dim = input_dim;
      input_step_dim.batch(1);
      input_step_dim.height(to - from);

      Tensor input_step = input_.getSharedDataTensor(
        input_step_dim, b * input_dim.getFeatureLen(), true);
      if (fp32_fast && hidden_step.size() == input_step.size() &&
          input_step.getDataType() ==
            ml::train::TensorDim::DataType::FP32) {
        const size_t n = hidden_step.size();
        if (!idx) {
          std::memcpy(hidden_step.getData<uint8_t>(),
                      input_step.getData<uint8_t>(), n * sizeof(float));
        } else {
          float *out = hidden_step.getData<float>();
          const float *in = input_step.getData<float>();
          for (size_t k = 0; k < n; ++k) out[k] += in[k];
        }
      } else if (!idx) {
        // First residual operand: copy input -> hidden.
        // Planner-decided STATIC residency first: when either side is
        // GPU_CLMEM (the residual stream on the live Gemma2 path), the copy
        // runs with each side bound to its static plane (cl_mem sub-buffer /
        // SVM) -- cl_mem->cl_mem is a plain buffer copy, mixed is a kernel
        // with mixed args (e.g. layer0's SVM embedding -> cl_mem residual).
        // Under SVM residency (NNTR_SVM_RESIDENT) this MUST stay on the GPU —
        // a host Tensor::copy reads/writes the SVM buffer on the host, which
        // is not coherent when the per-op maps are skipped (=> <pad> garbage).
        // Use the GPU fp16 SVM copy so the whole residual add chains
        // device-resident; fall back to the host copy otherwise (FP32 /
        // non-SVM).
#ifdef ENABLE_FP16
        if (nntrainer::clmem_residual_op_cl(hidden_step, input_step,
                                            /*accumulate=*/false))
          continue;
        const bool svm16 =
          hidden_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
          input_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
          hidden_step.getMemoryData() && hidden_step.getMemoryData()->isSVM() &&
          input_step.getMemoryData() && input_step.getMemoryData()->isSVM() &&
          hidden_step.size() == input_step.size();
        // drain=false (NNTR_ADD_DRAIN=1 restores): every consumer of the
        // residual copy is a GPU kernel on the in-order queue (the accumulate
        // below, then the norm); the trailing clFinish here measured
        // ~50ms/prefill of GPU idle (scatter_copy_f16 -> v8c_add_h2h). The
        // skip branch clFlush keeps the submission point.
        static const bool add_drain = std::getenv("NNTR_ADD_DRAIN") != nullptr;
        if (svm16 &&
            nntrainer::gpu_copy_f16_cl(
              reinterpret_cast<const uint16_t *>(input_step.getData<_FP16>()),
              reinterpret_cast<uint16_t *>(hidden_step.getData<_FP16>()),
              (unsigned int)hidden_step.size(), /*svm=*/true,
              /*in_clmem=*/nullptr, /*out_clmem=*/nullptr,
              /*drain=*/add_drain)) {
          // GPU copy done.
        } else
#endif
          hidden_step.copy(input_step);
      } else {
#ifdef ENABLE_FP16
        // Static residency: accumulate with each side on its static plane.
        if (nntrainer::clmem_residual_op_cl(hidden_step, input_step,
                                            /*accumulate=*/true))
          continue;
#endif
        add_i_cl(hidden_step, input_step);
      }
    }
  }

  // Paper §3.6 chain-residency hand-off: publish the residual_add
  // output to a GPU TensorBacking so a downstream RMSNorm consumer
  // (specifically fused_rmsnorm_quant_resident_fp32) can read its
  // input from cl_mem and skip the per-call host upload. Gated by
  // NNTR_RESIDUAL_PUBLISH=1; default off keeps the original
  // pure-CPU contract intact.
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
}

void AdditionLayerCL::calcDerivative(RunLayerContext &context) {

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

void AdditionLayerCL::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, add_props);
  if (!remain_props.empty()) {
    std::string msg = "[AdditionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   reshaped_rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cpu_backend.h>
#include <reshaped_rms_norm.h>

#include <blas_kernels.h>
#include <memory_data.h>

#include "_layer_prof.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ReshapedRMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  feature_size = std::get<props::FeatureSize>(rms_props);
  use_gamma = std::get<props::UseGamma>(rms_props).get();

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  NNTR_THROW_IF(dim[0].width() % feature_size != 0, std::invalid_argument)
    << "feature size must be a divisor of width";

  if (use_gamma) {
    nntrainer::TensorDim gamma_dim(
      1, 1, 1, feature_size,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()));
    wt_idx[RMSParams::gamma] = context.requestWeight(
      gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);
  }
}

void ReshapedRMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {
  // incremental_forwarding() is being phased out as the inference entry point,
  // so run the full-sequence reshaped RMS norm through forwarding(). The shared
  // worker (incremental_forwarding) handles both the host and the GPU-resident
  // (isSVM) paths over the whole input [0, height).
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  incremental_forwarding(context, 0, in.height(), training);
}

void ReshapedRMSNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  causallm::LayerProfScope _prof("rms_norm", (to - from) == 1);
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  bool is_prefill = !from;
  if (skip_prefill && is_prefill)
    return;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  // set reshaped dim to (1, 1, -1, feature_size)
  ml::train::TensorDim step_reshaped_dim = in_step_dim;

  step_reshaped_dim.width(feature_size);
  step_reshaped_dim.height(in_step_dim.height() *
                           (in_dim.width() / feature_size));

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    // reshape in_step
    // reshape out_step
    in_step.reshape(step_reshaped_dim);
    out_step.reshape(step_reshaped_dim);

    // GPU-resident path: when the activation, output and gamma are all SVM
    // (the graph runs on the SVM pool), normalize each feature_size chunk on
    // the GPU SVM-direct -- no host round-trip. The GPU rmsnorm kernel folds
    // gamma in, so the separate multiply_i(gamma) is skipped on this path.
    const auto in_md = in_step.getMemoryData();
    const auto out_md = out_step.getMemoryData();
    const auto gamma_md = gamma.getMemoryData();
    const bool use_svm = in_md && in_md->isSVM() && out_md &&
                         out_md->isSVM() && gamma_md && gamma_md->isSVM();
    // NNTR_DEVRES Step 3 diagnostic: why is rmsnorm host? Log which operand is
    // not SVM (in/out/gamma). One-shot per process; gated by the trace flag.
    {
      static const bool devres_trace =
        std::getenv("NNTR_DEVRES_TRACE") != nullptr;
      static bool logged = false;
      if (devres_trace && !logged) {
        logged = true;
        std::fprintf(stderr,
                     "[devres] rmsnorm use_svm=%d  in.svm=%d out.svm=%d "
                     "gamma.svm=%d (in_md=%d out_md=%d gamma_md=%d)\n",
                     (int)use_svm, in_md ? (int)in_md->isSVM() : -1,
                     out_md ? (int)out_md->isSVM() : -1,
                     gamma_md ? (int)gamma_md->isSVM() : -1, in_md ? 1 : 0,
                     out_md ? 1 : 0, gamma_md ? 1 : 0);
      }
    }
    const unsigned int n_rows = in_step.getDim().height();
    bool gpu_done = false;
    if (use_svm) {
      if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
        nntrainer::rmsnorm_cl(in_step.getData<float>(), gamma.getData<float>(),
                              out_step.getData<float>(), epsilon, n_rows,
                              feature_size, /*use_svm=*/true);
        gpu_done = true;
#ifdef ENABLE_FP16
      } else if (in_step.getDataType() ==
                 ml::train::TensorDim::DataType::FP16) {
        nntrainer::rmsnorm_cl_fp16(
          in_step.getData<_FP16>(), gamma.getData<_FP16>(),
          out_step.getData<_FP16>(), epsilon, n_rows, feature_size,
          /*use_svm=*/true);
        gpu_done = true;
#endif
      }
    }

    if (!gpu_done) {
      if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
        ///@todo rms_norm_wrt_width_something() should be refactored to
        /// nntrainer::Tensor operation.
        // fp32_intrinsic (not fp16_intrinsic) even under ENABLE_FP16 -- the
        // fp16 variant overflows (upstream RMSNorm FP16 overflow fix).
        nntrainer::rms_norm_wrt_width_fp32_intrinsic(
          in_step.getData<float>(), out_step.getData<float>(),
          in_step.getDim().height(), in_step.getDim().width(), epsilon);
      } else if (in_step.getDataType() ==
                 ml::train::TensorDim::DataType::FP16) {
        // FP16 path: fall back to the generic Tensor::multiply/average/
        // inv_sqrt_i sequence. Avoids a separate intrinsic; still inside
        // the FP16 residual stream.
        auto t = in_step.multiply(in_step).average(3).add(epsilon);
        t.inv_sqrt_i();
        in_step.multiply(t, out_step);
      } else {
        throw std::invalid_argument(
          "Error: not yet implemented for this data type");
      }
      out_step.multiply_i(gamma);
    }

    // NNTR_DEVRES Step 1: flag the rmsnorm output device-resident iff the GPU
    // wrote it (gpu_done); clear it on the host path so a stale bit from a prior
    // token can't falsely mark a host-written buffer (clear-on-write). out_step
    // shares out's MemoryData (slice view), so this rides the producer->consumer
    // edge to the downstream FC (QKV / gate-up). Gated; off => bit untouched.
    static const bool devres_rms = std::getenv("NNTR_DEVRES") != nullptr;
    if (devres_rms && out_md) {
      if (gpu_done)
        out_md->setDeviceValid(true, out_step.getData<uint8_t>());
      else
        out_md->setDeviceValid(false);
    }

    // reshape again out_step
    out_step.reshape(out_step_dim);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << std::endl;
#endif
  }
}

void ReshapedRMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void ReshapedRMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new ReshapedRMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm

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

#if defined(ENABLE_OPENCL)
// OpenCL GPU rmsnorm kernels (rmsnorm_cl / rmsnorm_cl_fp16 / cl_svm_*). Guarded so
// the FP32 CPU build (enable-opencl=false) compiles host-only. [T12]
#include <blas_kernels.h>
#endif
#include <memory_data.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_rmsnorm.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

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
  // gamma weight is only requested in finalize() when use_gamma is true
  // (e.g. Gemma4 v_norm sets use_gamma=false). When false, wt_idx[gamma] is
  // left at UINT_MAX, so reading it would index out of bounds -- the norm is
  // gamma-free (identity scale) in that case.
  nntrainer::Tensor *gamma =
    use_gamma ? &context.getWeight(wt_idx[RMSParams::gamma]) : nullptr;

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
    const auto gamma_md = gamma ? gamma->getMemoryData() : nullptr;
    // The GPU rmsnorm kernel folds gamma in, so the SVM-direct path requires a
    // gamma weight; fall back to the host path when use_gamma is false.
    const bool use_svm = gamma && in_md && in_md->isSVM() && out_md &&
                         out_md->isSVM() && gamma_md && gamma_md->isSVM();
    // S2 gamma-free GPU v_norm (S1.2, default-ON): Gemma4 v_norm/PLE-norm set
    // use_gamma=false, so there is no gamma weight and the path above falls to
    // the host (cl_queue_finish + 2x blocking SVM map + FP32 intrinsic + unmap,
    // ~0.18 ms/call x ~30 calls/token = ~15 full queue drains/token). Route it
    // to the gamma-free coop kernel (rmsnorm_cl_fp16_coop_ng) when in/out are
    // SVM FP16: the kernel does the sum-of-squares in fp32 (overflow-safe like
    // the host intrinsic) and skips the gamma fold. DEFAULT-ON now (kill-switch
    // NNTR_VNORM_HOST): with the S1.1 engine key the gamma-free norms are
    // GPU-context layers, so the host fallback's CPU Tensor ops (clone/
    // multiply_i) would crash on gpu-allocated tensors -- the GPU path is
    // required, not optional. Coupled with S1.1 this measured 18.9 -> 23 TPS.
    static const bool vnorm_host = std::getenv("NNTR_VNORM_HOST") != nullptr;
    const bool use_svm_ng =
      !gamma && !vnorm_host && in_md && in_md->isSVM() && out_md &&
      out_md->isSVM() &&
      in_step.getDataType() == ml::train::TensorDim::DataType::FP16;
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
    if (std::getenv("NNTR_RMSN_TRACE")) {
      std::fprintf(stderr,
                   "[RMSN] %-26s use_svm=%d use_svm_ng=%d in.clmem=%d "
                   "out.clmem=%d use_gamma=%d fs=%u rows=%u\n",
                   context.getName().c_str(), (int)use_svm, (int)use_svm_ng,
                   (int)in_step.isClMem(), (int)out_step.isClMem(),
                   (int)use_gamma, feature_size, n_rows);
      std::fflush(stderr);
    }
    bool gpu_done = false;
#if defined(ENABLE_OPENCL)
    // GPU-resident SVM rmsnorm dispatch. Without OpenCL gpu_done stays false and
    // the host RMSNorm loop below runs. [T12]
    if (use_svm) {
      if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
        nntrainer::rmsnorm_cl(in_step.getData<float>(), gamma->getData<float>(),
                              out_step.getData<float>(), epsilon, n_rows,
                              feature_size, /*use_svm=*/true);
        gpu_done = true;
#ifdef ENABLE_FP16
      } else if (in_step.getDataType() ==
                 ml::train::TensorDim::DataType::FP16) {
        // Bind cl_mem sub-buffers when the operands are GPU_CLMEM-resident
        // (Gemma4 q/k/projection norms consume v8c FC outputs that are direct-
        // to-cl_mem). Without this the SVM path reads a stale host shadow of
        // the cl_mem plane -> garbage (gemma2 has no per-head norm so never hit
        // this). gamma stays SVM, matching rms_norm_gpu.
        void *in_cl = in_step.isClMem() ? in_step.getClMem() : nullptr;
        void *out_cl = out_step.isClMem() ? out_step.getClMem() : nullptr;
        nntrainer::rmsnorm_cl_fp16(
          in_step.getData<_FP16>(), gamma->getData<_FP16>(),
          out_step.getData<_FP16>(), epsilon, n_rows, feature_size,
          /*use_svm=*/true, out_cl, in_cl);
        gpu_done = true;
#endif
      }
#ifdef ENABLE_FP16
    } else if (use_svm_ng) {
      // Gamma-free GPU v_norm: same cl_mem/SVM operand binding as above, gamma
      // is null so the wrapper dispatches the _ng kernel (no gamma fold).
      void *in_cl = in_step.isClMem() ? in_step.getClMem() : nullptr;
      void *out_cl = out_step.isClMem() ? out_step.getClMem() : nullptr;
      nntrainer::rmsnorm_cl_fp16(in_step.getData<_FP16>(), /*gamma=*/nullptr,
                                 out_step.getData<_FP16>(), epsilon, n_rows,
                                 feature_size,
                                 /*use_svm=*/true, out_cl, in_cl);
      gpu_done = true;
#endif
    }
#endif // ENABLE_OPENCL [T12]

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // CUDA per-head norm: each feature_size chunk is one rmsnorm row, so reuse
    // cuda_rmsnorm_fp16 with rows=n_rows, width=feature_size (gamma null for the
    // gamma-free v_norm). Keeps q/k/v_norm on the device. Opt-in NNTR_CUDA_QKNORM.
    // The host fallback below reads the GPU-produced UVM in_step/gamma on the CPU
    // (rms_norm_wrt_width_fp32_intrinsic); sync first in async mode (no-op in
    // default sync mode). Placed before the dtype-read entry, off the SVM
    // GPU->GPU and cl_queue_finish-guarded paths.
    nntrainer::cuda::StreamManager::Global().finishIfAsync();
    if (!gpu_done &&
        in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      static const bool gpu = std::getenv("NNTR_CUDA_QKNORM") != nullptr;
      if (gpu) {
        auto *ip =
          reinterpret_cast<const unsigned short *>(in_step.getData<_FP16>());
        auto *op =
          reinterpret_cast<unsigned short *>(out_step.getData<_FP16>());
        const unsigned short *gp =
          gamma ? reinterpret_cast<const unsigned short *>(gamma->getData<_FP16>())
                : nullptr;
        auto dev_ok = [](const void *p) {
          if (!p)
            return true;
          return nntrainer::cuda::dev_accessible(p);
        };
        if (dev_ok(ip) && dev_ok(op) && dev_ok(gp) &&
            nntrainer::cuda::cuda_rmsnorm_fp16(ip, gp, op, epsilon, n_rows,
                                               feature_size))
          gpu_done = true;
      }
    }
#endif

    if (!gpu_done) {
#ifdef ENABLE_FP16
      // Coherent SVM hand-off for a GPU-produced FP16 input read on the host
      // (Gemma4 v_norm, use_gamma=false, is the only norm that reaches this
      // fallback): the v8c FC async-maps its SVM output for a next-GPU consumer,
      // so a host reader must drain + blocking-map first or it reads a stale
      // shadow -> garbage V -> garbage attention (the prompt-independent bug).
      // Map out for the host write too, and unmap it after so the next consumer
      // (mha_core) sees the result.
      const bool fp16_svm =
        in_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        in_md && in_md->isSVM() && out_md && out_md->isSVM();
#if defined(ENABLE_OPENCL)
      // SVM map/drain is OpenCL-only (fp16_svm is always false without SVM). [T12]
      if (fp16_svm) {
        nntrainer::cl_queue_finish();
        nntrainer::cl_svm_map_force(in_step.getData<_FP16>(),
                                    (size_t)in_step.size() * sizeof(_FP16),
                                    /*read_only=*/true);
        nntrainer::cl_svm_map_force(out_step.getData<_FP16>(),
                                    (size_t)out_step.size() * sizeof(_FP16),
                                    /*read_only=*/false);
      }
#endif
#endif
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
        // FP16 path: the sum-of-squares MUST be computed in FP32. Doing
        // in_step.multiply(in_step) in FP16 squares each element in FP16, so
        // any per-row element |x|>~256 (sqrt(65504)) overflows to +Inf ->
        // average->Inf -> inv_sqrt=0 -> the whole row is zeroed (Gemma4 v_norm
        // row 0, |x|=674, was zeroed -> all-zero attention -> NaN cascade).
        // Normalize via the same FP32 intrinsic the FP32 branch uses by
        // converting in to FP32, then cast the normalized result back to FP16.
        // This stays inside the residual stream (out is still FP16) and reuses
        // the proven-correct FP32 RMSNorm (no overflow).
        nntrainer::Tensor in_f32 =
          in_step.clone(ml::train::TensorDim::DataType::FP32);
        nntrainer::Tensor out_f32(in_f32.getDim());
        nntrainer::rms_norm_wrt_width_fp32_intrinsic(
          in_f32.getData<float>(), out_f32.getData<float>(),
          in_step.getDim().height(), in_step.getDim().width(), epsilon);
        out_step.copyData(out_f32);
      } else {
        throw std::invalid_argument(
          "Error: not yet implemented for this data type");
      }
      if (gamma)
        out_step.multiply_i(*gamma);
#ifdef ENABLE_FP16
      // Hand the host-written output back to the device for the next consumer.
#if defined(ENABLE_OPENCL)
      // OpenCL-only SVM hand-back. [T12]
      if (fp16_svm)
        nntrainer::cl_svm_unmap_force(out_step.getData<_FP16>());
#endif
#endif
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

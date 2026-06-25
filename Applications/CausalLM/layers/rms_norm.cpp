// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cpu_backend.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "rms_norm.h"

#include <blas_kernel_interface.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void RMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);
}

void RMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {}

void RMSNormLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
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

  unsigned int b_size = in_dim.batch();

  // Chain-robustification path: GPU rmsnorm + readback to host so any
  // downstream CPU op sees the same bits as a GPU consumer would. This
  // collapses the per-op rmsnorm CPU/GPU split into a single source of
  // truth (GPU compute), which is the prerequisite for combining with
  // GPU mha without the mixed-numerics drift documented in
  // [gpu-mha-correct-drift-amplifies]. Gated by NNTR_RMSNORM_GPU=1
  // AND requires NNTR_RESIDENT_RMSNORM=1 (which gates the actual GPU
  // dispatch). Output backing is published; host data is overwritten
  // with the GPU result via clEnqueueReadBuffer so CPU consumers stay
  // bit-equal to GPU consumers.
  // Fused RMSNorm + v8c-quant path. Always runs when env is on and dtype
  // matches — produces 4 GPU buffers in the pool keyed by (output_name |
  // ptr:out_host) so v8c FC can find them and skip its own activation
  // quant (paper §3.6 #2). If the consumer wire is also on, the CPU
  // RMSNorm below is redundant and can be skipped.
  bool fused_ok = false;
  if (b_size == 1 &&
      in.getDataType() == ml::train::TensorDim::DataType::FP32 &&
      gamma.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const auto &d = in_step_dim;
    const void *out_host = out.getData<uint8_t>();
    fused_ok = nntrainer::fused_rmsnorm_quant_resident_fp32(
      in, gamma, epsilon, d.batch() * d.channel() * d.height(), d.width(),
      out.getName(), out_host);
  }
  // NOTE: Even when fused-consume is wired up, the CPU RMSNorm result
  // turns out to be read by SOMETHING downstream we haven't fully
  // mapped (skipping it produces drift-amplified garbage). Most
  // likely candidates: a TensorPool reuse where rmsnorm's `out`
  // memory aliases a later layer's buffer that gets re-read before
  // being overwritten, or a debug/profile path that reads it. Until
  // that producer/consumer map is verified, leave the CPU RMSNorm on
  // unconditionally. CPU cost is ~5 ms/prefill so the perf headroom
  // is small anyway; the real win lies in the host-upload elimination
  // (resident-input mode in fused_rmsnorm_quant_resident_fp32, fires
  // when the producer of this layer's input publishes a backing).
  (void)fused_ok;

  bool gpu_handled = false;
  if (b_size == 1 && std::getenv("NNTR_RMSNORM_GPU") != nullptr) {
    const auto &d = in_step_dim;
    bool ok = false;
    const auto dt = in.getDataType();
    if (dt == ml::train::TensorDim::DataType::FP16 &&
        gamma.getDataType() == ml::train::TensorDim::DataType::FP16) {
      ok = nntrainer::rmsnorm_resident_fp16(
        in, gamma, epsilon, d.batch(), d.channel(), d.height(), d.width(),
        out.getName(), out);
    } else if (dt == ml::train::TensorDim::DataType::FP32 &&
               gamma.getDataType() == ml::train::TensorDim::DataType::FP32 &&
               d.width() % 4 == 0) {
      // FP32 path (Qwen3 residual-stream dtype). Signature only takes
      // H, W; batch/channel are folded into the dispatch elsewhere.
      ok = nntrainer::rmsnorm_resident_fp32(
        in, gamma, epsilon, d.height(), d.width(), out.getName(), out);
    }
    if (ok) {
      // Read the GPU result back into the host buffer. Subsequent CPU
      // ops (e.g., FC quant when residency is off, RoPE, residual add)
      // see GPU-computed values.
      nntrainer::readback_backing_to_host(out);
      gpu_handled = true;
      static int trip = 0;
      if (!trip && std::getenv("NNTR_RMSNORM_GPU_TRIP") != nullptr) {
        trip = 1;
        std::fprintf(stderr,
                     "[RMS-GPU] first trip: %s dtype=%d B=%u C=%u H=%u W=%u\n",
                     out.getName().c_str(), (int)dt, d.batch(), d.channel(),
                     d.height(), d.width());
        std::fflush(stderr);
      }
    }
  }

  if (!gpu_handled) {
    for (unsigned int b = 0; b < b_size; ++b) {
      nntrainer::Tensor in_step =
        in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
      nntrainer::Tensor out_step =
        out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

      if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
        auto t = in_step.multiply(in_step).average(3).add(epsilon);
        t.inv_sqrt_i();
        in_step.multiply(t, out_step);
#ifdef ENABLE_FP16
      } else if (in_step.getDataType() ==
                 ml::train::TensorDim::DataType::FP16) {
        // FP16 path: the sum-of-squares MUST be accumulated in FP32. Doing it
        // in fp16 (in_step.multiply(in_step).average(3)) loses precision and
        // overflows on a large residual (|x| up to ~1700 in gemma4 ->
        // x^2 > fp16 max 65504 -> +Inf -> wrong scale -> exploded norm). On x86
        // Tensor::average(fp16) happens to accumulate in fp32, but the aarch64
        // NEON fp16 path accumulates in fp16 -> garbage on Jetson/Orin (the norm
        // exploded to +-1389 there while reshaped_rms_norm/CudaRMSNormLayer,
        // which already use fp32, stayed correct). Explicit fp32 reduction here
        // makes the host norm correct on every arch. gamma is applied below.
        const unsigned int rows =
          in_step_dim.channel() * in_step_dim.height();
        const unsigned int W = in_step_dim.width();
        const _FP16 *xi = in_step.getData<_FP16>();
        _FP16 *yi = out_step.getData<_FP16>();
        for (unsigned int r = 0; r < rows; ++r) {
          const _FP16 *xr = xi + (size_t)r * W;
          _FP16 *yr = yi + (size_t)r * W;
          float ss = 0.f;
          for (unsigned int k = 0; k < W; ++k) {
            float v = static_cast<float>(xr[k]);
            ss += v * v;
          }
          float inv = 1.0f / std::sqrt(ss / static_cast<float>(W) + epsilon);
          if (std::getenv("NNTR_RMS_PROBE")) {
            static int _n = 0;
            if (_n++ < 6)
              std::fprintf(stderr,
                           "[RMS-HOST-FP16] rows=%u W=%u ss=%.3g inv=%.3g\n",
                           rows, W, ss, inv);
          }
          for (unsigned int k = 0; k < W; ++k)
            yr[k] = static_cast<_FP16>(static_cast<float>(xr[k]) * inv);
        }
#endif
      } else {
        throw std::invalid_argument(
          "Error: not yet implemented for this data type");
      }
      out_step.multiply_i(gamma);

#ifdef DEBUG
      std::cout << context.getName() << " \n input:" << in_step
                << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
    }
  }

  // Segment A.2 hand-off: publish the CPU-computed RMSNorm output to a
  // TensorBacking so that downstream FC layers with NNTR_RESIDENT_FC=1
  // can read GPU-resident data and skip the host→device upload. Bit-
  // exact w.r.t. baseline because the math runs on CPU exactly as
  // before; this just makes the result reachable by GPU consumers.
  // Env-gated so the original CPU-only path remains intact when off.
  if (std::getenv("NNTR_SEGA_RMSNORM_PUBLISH") != nullptr && b_size == 1 &&
      out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    nntrainer::publish_host_fp32_to_backing(out, out.getName());
  }
}

void RMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void RMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new RMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm

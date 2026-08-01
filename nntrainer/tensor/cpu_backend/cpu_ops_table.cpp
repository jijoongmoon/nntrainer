// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cpu_ops_table.cpp
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Unified CPU backend ComputeOps subclass.
 *
 * Single concrete ComputeOps subclass for ALL CPU targets (ARM /
 * x86 / fallback). The nntrainer::sgemm etc. functions are arch-
 * specialized — each arch_compute_backend.cpp defines its own body
 * — so a single forwarding wrapper is enough; build-time arch
 * dispatch picks the right symbol at link time.
 */

#include "cpu_ops_table.h"

#include <cmath>
#include <stdexcept>

#include <acti_func.h>
#include <tensor.h>

namespace nntrainer {

ComputeOps *get_cpu_ops() {
  static CpuComputeOps instance;
  return &instance;
}

namespace {
// gelu (tanh approximation, gelu_pytorch_tanh) -- same constants as the OpenCL
// geglu_cl / CUDA geglu kernels, so the host path is numerically consistent.
inline float gelu_tanh(float x) {
  const float k = 0.7978845608028654f; // sqrt(2/pi)
  return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}
// silu (numerically stable: x/(1+exp(-x)) == x*sigmoid(x)) -- matches the
// OpenCL swiglu_cl kernel exactly (avoids the x*exp(x)/(1+exp(x)) overflow).
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
// sigmoid -- matches the OpenCL sigmoid_glu/sigmoid_add kernels and the CUDA
// ELTWISE_SRC form (1/(1+exp(-x))) so the three backends agree token-for-token.
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
} // namespace

// out = gelu_tanh(in1) * in2 over rows [row_offset, row_offset+active_rows).
// row_offset is 0 on every current caller (the live token is at the buffer
// base for the host/SVM/UVM paths); the offset is honored for generality.
void CpuComputeOps::geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                          unsigned int active_rows, unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
  const auto dt = in1.getDataType();

  if (dt == ml::train::TensorDim::DataType::FP32) {
    const float *a = in1.getData<float>() + elem_off;
    const float *b = in2.getData<float>() + elem_off;
    float *o = out.getData<float>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = gelu_tanh(a[i]) * b[i];
#ifdef ENABLE_FP16
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
    const _FP16 *a = in1.getData<_FP16>() + elem_off;
    const _FP16 *b = in2.getData<_FP16>() + elem_off;
    _FP16 *o = out.getData<_FP16>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = static_cast<_FP16>(gelu_tanh((float)a[i]) * (float)b[i]);
#endif
  } else {
    throw std::invalid_argument("CpuComputeOps::geglu: unsupported data type");
  }
}

// out = silu(in1) * in2 over rows [row_offset, row_offset+active_rows).
void CpuComputeOps::swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                           unsigned int active_rows, unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
  const auto dt = in1.getDataType();

  if (dt == ml::train::TensorDim::DataType::FP32) {
    const float *a = in1.getData<float>() + elem_off;
    const float *b = in2.getData<float>() + elem_off;
    float *o = out.getData<float>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = silu(a[i]) * b[i];
#ifdef ENABLE_FP16
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
    const _FP16 *a = in1.getData<_FP16>() + elem_off;
    const _FP16 *b = in2.getData<_FP16>() + elem_off;
    _FP16 *o = out.getData<_FP16>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = static_cast<_FP16>(silu((float)a[i]) * (float)b[i]);
#endif
  } else {
    throw std::invalid_argument("CpuComputeOps::swiglu: unsupported data type");
  }
}

// out = sigmoid(in1) * in2 over rows [row_offset, row_offset+active_rows).
// A sigmoid-gated attention output gate is one example. FP32 accumulation
// (upcast fp16 -> float) so the LRA-MLP intermediates do not overflow fp16.
void CpuComputeOps::sigmoid_glu(const Tensor &in1, const Tensor &in2,
                                Tensor &out, unsigned int active_rows,
                                unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
  const auto dt = in1.getDataType();

  if (dt == ml::train::TensorDim::DataType::FP32) {
    const float *a = in1.getData<float>() + elem_off;
    const float *b = in2.getData<float>() + elem_off;
    float *o = out.getData<float>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = sigmoidf(a[i]) * b[i];
#ifdef ENABLE_FP16
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
    const _FP16 *a = in1.getData<_FP16>() + elem_off;
    const _FP16 *b = in2.getData<_FP16>() + elem_off;
    _FP16 *o = out.getData<_FP16>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = static_cast<_FP16>(sigmoidf((float)a[i]) * (float)b[i]);
#endif
  } else {
    throw std::invalid_argument(
      "CpuComputeOps::sigmoid_glu: unsupported data type");
  }
}

// out = sigmoid(in1) + in2 over rows [row_offset, row_offset+active_rows).
// A per-layer-embedding (PLE) mix path (method=1) is one example. FP32
// accumulation as above.
void CpuComputeOps::sigmoid_add(const Tensor &in1, const Tensor &in2,
                                Tensor &out, unsigned int active_rows,
                                unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
  const auto dt = in1.getDataType();

  if (dt == ml::train::TensorDim::DataType::FP32) {
    const float *a = in1.getData<float>() + elem_off;
    const float *b = in2.getData<float>() + elem_off;
    float *o = out.getData<float>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = sigmoidf(a[i]) + b[i];
#ifdef ENABLE_FP16
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
    const _FP16 *a = in1.getData<_FP16>() + elem_off;
    const _FP16 *b = in2.getData<_FP16>() + elem_off;
    _FP16 *o = out.getData<_FP16>() + elem_off;
    for (size_t i = 0; i < n; ++i)
      o[i] = static_cast<_FP16>(sigmoidf((float)a[i]) + (float)b[i]);
#endif
  } else {
    throw std::invalid_argument(
      "CpuComputeOps::sigmoid_add: unsupported data type");
  }
}

// out = out_scale * normalize(in * weight) over the live rows — the
// PLE post_norm (RMSReverseNormLayer) host math, moved VERBATIM out of the
// layer body (N4). ReverseRMSNorm order: x * weight -> normalize ->
// multiply by out_scale; the per-feature weight sits INSIDE the RMS
// denominator, so this is not rmsnorm*gamma. The FP32 path folds the weight
// into `in` in place (the layer's original math — the reverse-norm input has
// no other consumer). The FP16 path computes in an FP32 temporary and upcasts
// weight/out_scale first: multiplying the FP32 temp by the FP16 tensors
// directly would reinterpret their bytes as FP32 (Tensor::multiply ->
// apply_broadcast reads getData<float>()) -> garbage.
void CpuComputeOps::rms_reverse_norm(Tensor &in, Tensor &out,
                                     const Tensor &weight,
                                     const Tensor &out_scale, float epsilon,
                                     unsigned int active_rows,
                                     unsigned int row_offset) {
  // Rebuild the layer's per-step window as a flattened-row view (the layer
  // passed {1, C, to-from, W} views; every consumer has C==1, for which
  // {1, 1, active_rows, W} at the same element offset is the identical
  // memory region and the identical per-width-row math).
  ml::train::TensorDim in_step_dim = in.getDim();
  in_step_dim.batch(1);
  in_step_dim.channel(1);
  in_step_dim.height(active_rows);
  ml::train::TensorDim out_step_dim = out.getDim();
  out_step_dim.batch(1);
  out_step_dim.channel(1);
  out_step_dim.height(active_rows);

  const size_t elem_off = (size_t)row_offset * in.width();
  Tensor in_step = in.getSharedDataTensor(in_step_dim, elem_off, true);
  Tensor out_step = out.getSharedDataTensor(out_step_dim, elem_off, true);

  if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    // ReverseRMSNorm order: x * weight → normalize → multiply by out_scale

    // Step 1: Multiply input by weight (BEFORE normalization)
    in_step.multiply_i(weight);

    // Step 2: Compute RMS normalization
    // rsqrt(average(x^2) + eps)
    auto t = in_step.multiply(in_step).average(3).add(epsilon);
    t.inv_sqrt_i();

    // Step 3: Apply normalization
    in_step.multiply(t, out_step);

    // Step 4: Apply output scale (AFTER normalization)
    out_step.multiply_i(out_scale);
  } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    // Host path (ARM CPU / non-GPU-resident): FP32-temp compute.
    ml::train::TensorDim instep_dim = in_step_dim;
    ml::train::TensorDim outstep_dim = out_step_dim;

    instep_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    outstep_dim.setDataType(ml::train::TensorDim::DataType::FP32);

    Tensor in_step32(instep_dim, true);
    Tensor out_step32(outstep_dim, true);

    in_step32.copyData(in_step);

    // weight/out_scale share the activation dtype (packed=false), which is
    // FP16 on an enable-fp16 build. Upcast the scale tensors to FP32 so every
    // multiply below is dtype-matched (see the function comment).
    ml::train::TensorDim weight32_dim = weight.getDim();
    weight32_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor weight32(weight32_dim, true);
    weight32.copyData(weight);

    ml::train::TensorDim out_scale32_dim = out_scale.getDim();
    out_scale32_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor out_scale32(out_scale32_dim, true);
    out_scale32.copyData(out_scale);

    // ReverseRMSNorm order: x * weight → normalize → multiply by out_scale

    // Step 1: Multiply input by weight (BEFORE normalization)
    in_step32.multiply_i(weight32);

    // Step 2: Compute RMS normalization
    auto t = in_step32.multiply(in_step32).average(3).add(epsilon);
    t.inv_sqrt_i();

    // Step 3: Apply normalization
    in_step32.multiply(t, out_step32);

    // Step 4: Apply output scale (AFTER normalization)
    out_step32.multiply_i(out_scale32);

    out_step.copyData(out_step32);
#else
    throw std::invalid_argument("Error: enable-fp16 is not set");
#endif
  } else {
    // Unreachable from the layer (activations are FP32/FP16 only); loud like
    // the sibling whole-ops rather than the layer's old silent no-op.
    throw std::invalid_argument(
      "CpuComputeOps::rms_reverse_norm: unsupported data type");
  }
}

// hidden = input (copy) or hidden += input (add) on the host buffer. Mirrors
// the core AdditionLayer's per-input copy()/add_i() (correct for host and UVM).
void CpuComputeOps::residual_op(Tensor &hidden, const Tensor &input,
                                bool accumulate) {
  if (accumulate)
    hidden.add_i(input);
  else
    hidden.copy(input);
}

// output = input * weight. Host Tensor::dot (CPU/UVM FC matmul). The CL/CUDA
// quantized GEMM paths override this in their ComputeOps subclasses.
void CpuComputeOps::fc(Tensor &input, Tensor &weight, Tensor &output) {
  input.dot(weight, output, false, false);
}

// Fused activation epilogue on the host: build the SAME ActiFunc the standalone
// ActivationLayer would (so the fused result is value-identical), and run it in
// place when the activation supports it (relu/sigmoid/tanh) or via a temp input
// copy otherwise — mirroring ActivationLayer::run_fn(input, output) exactly.
void CpuComputeOps::apply_activation(Tensor &out, int act_type) {
  const auto at = static_cast<ActivationType>(act_type);
  if (at == ActivationType::ACT_NONE)
    return;
  ActiFunc f;
  if (out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    f.setActiFunc<_FP16>(at);
#else
    throw std::invalid_argument("apply_activation: fp16 needs enable-fp16");
#endif
  } else {
    f.setActiFunc<float>(at);
  }
  if (f.supportInPlace()) {
    f.run_fn(out, out);
  } else {
    Tensor in_copy = out.clone();
    f.run_fn(in_copy, out);
  }
}

} // namespace nntrainer

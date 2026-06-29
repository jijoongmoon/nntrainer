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

// hidden = input (copy) or hidden += input (add) on the host buffer. Mirrors the
// core AdditionLayer's per-input copy()/add_i() (correct for host and UVM).
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

} // namespace nntrainer

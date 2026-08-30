// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cl_compute_ops.cpp
 * @date   25 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  OpenCL ComputeOps subclass — provides accelerated quantized
 *         GEMM/GEMV variants on top of the existing nntrainer
 *         OpenCL kernels in cl_operations/blas_kernels.cpp.
 *
 * Two families live here. The accelerator-specific ops (Q4_0 batch / accel,
 * INT4 batch / accel) are overridden with their supports_*() predicates
 * returning true; callers rely on supports_*() to decide whether to use this
 * path or fall back to a CPU ops table — exactly the contract
 * float_tensor.cpp's dispatch sites already follow. The whole-ops a
 * backend-neutral layer calls unconditionally (fc, apply_activation) have no
 * supports_*() escape hatch, so every op a layer registered on the gpu engine
 * reaches must resolve here: the base ComputeOps default throws, and a layer
 * has nowhere to catch that. Those are implemented below.
 *
 * This file is what unblocks GPU dispatch end-to-end:
 *   ClContext (Engine-registered) -> ContextData -> ClComputeOps
 *   -> nntrainer::gemm_q4_0_async_cl(...) -> OpenCL kernel queue.
 */

#include <cstring>
#include <stdexcept>

#include <blas_kernel_interface.h> // add_i_cl, dotCl
#include <blas_kernels.h>
#include <common_properties.h> // ActivationType, the act_type int encoding
#include <compute_ops.h>
#include <geglu_cl_op.h>
#include <gelu_cl_op.h>
#include <layernorm_cl_op.h>
#include <swiglu_cl_op.h>
#include <tensor.h>

namespace nntrainer {

class ClComputeOps : public ComputeOps {
public:
  /**
   * @brief Fully connected matmul on the OpenCL GEMM.
   *
   * This is the call FullyConnectedLayerCl made directly before the layer
   * became backend-neutral, so a gpu-engine graph keeps the same kernel and
   * the same numerics. dotCl() writes the result (it does not accumulate),
   * which is why no zero-fill precedes it.
   */
  void fc(Tensor &input, Tensor &weight, Tensor &output) override {
    dotCl(input, weight, output);
  }

  /**
   * @brief Activation epilogue, run on the host.
   *
   * A gpu-engine Tensor is host-allocated (the kernels stage it into device
   * buffers per call), so the CPU table operates on exactly the same memory
   * and yields the same values a standalone ActivationLayer would. There is no
   * whole-op activation kernel yet; this exists so a fused epilogue on the gpu
   * engine computes rather than throwing, and it can be replaced by a kernel
   * without touching a caller.
   */
  void apply_activation(Tensor &out, int act_type) override {
    get_cpu_ops()->apply_activation(out, act_type);
  }

  // ── Accelerator-only Q4_0 / INT4 GEMM/GEMV ────────────────
  bool supports_gemm_q4_0_batch_fp32() const override { return true; }
  void gemm_q4_0_batch_fp32(std::vector<void *> matAdata, float *matBdata,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> N,
                            unsigned int K) override {
    nntrainer::gemm_q4_0_async_cl(matAdata, matBdata, matCdata, M, N, K);
  }

  bool supports_gemm_q4_0_accel_fp32() const override { return true; }
  void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata, float *matCdata,
                            unsigned int M, unsigned int N,
                            unsigned int K) override {
    nntrainer::gemm_q4_0_cl(matAdata, matBdata, matCdata, M, N, K);
  }

  bool supports_gemv_int4_batch_fp32() const override { return true; }
  void gemv_int4_batch_fp32(std::vector<void *> weights,
                            std::vector<uint16_t *> scales, float *input,
                            std::vector<float *> outputs, unsigned int K,
                            std::vector<unsigned int> Ns,
                            unsigned int group_size) override {
    nntrainer::gemv_int4_async_cl(weights, scales, input, outputs, K, Ns,
                                  group_size);
  }

  bool supports_gemm_int4_batch_fp32() const override { return true; }
  void gemm_int4_batch_fp32(float *input, std::vector<void *> weights,
                            std::vector<uint16_t *> scales,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> Ns, unsigned int K,
                            unsigned int group_size) override {
    nntrainer::gemm_int4_async_cl(input, weights, scales, matCdata, M, Ns, K,
                                  group_size);
  }

  bool supports_gemv_int4_accel_fp32() const override { return true; }
  void gemv_int4_accel_fp32(char *weight, uint16_t *scale, float *input,
                            float *output, unsigned int K, unsigned int N,
                            unsigned int group_size) override {
    nntrainer::gemv_int4_cl(weight, scale, input, output, K, N, group_size);
  }

  bool supports_sgemm_int4_accel_fp32() const override { return true; }
  void sgemm_int4_accel_fp32(float *input, char *weight, uint16_t *scale,
                             float *output, unsigned int M, unsigned int N,
                             unsigned int K, unsigned int group_size) override {
    nntrainer::sgemm_int4_cl(input, weight, scale, output, M, N, K, group_size);
  }

  // ── Whole-ops (Tensor level) ──────────────────────────────────
  // The gated pairs: (gate, up) -> out, element-wise, one kernel each.
  void geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
             unsigned int active_rows, unsigned int row_offset) override {
    nntrainer::geglu_cl_op(in1, in2, out, active_rows, row_offset);
  }
  void swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
              unsigned int active_rows, unsigned int row_offset) override {
    nntrainer::swiglu_cl_op(in1, in2, out, active_rows, row_offset);
  }

  // LayerNorm over the last axis. The neutral LayerNormalizationLayer owns
  // the axis contract and only dispatches here when its property matches, so
  // this op never sees a property.
  void layer_norm(const Tensor &in, Tensor &out, const Tensor &gamma,
                  const Tensor &beta, float epsilon, unsigned int active_rows,
                  unsigned int row_offset) override {
    nntrainer::layernorm_cl_op(in, out, gamma, beta, epsilon, active_rows,
                               row_offset);
  }

  // Element-wise activation. Only gelu and tanh_gelu have OpenCL kernels;
  // every other mode throws rather than quietly running a host loop, because a
  // tensor on this context may live in device memory the host has unmapped,
  // where a host loop is not merely slower but wrong. Which mode a backend can
  // serve is a backend question, so the mapping lives here and not in a Layer.
  void activation(const Tensor &in, Tensor &out, int act_type,
                  unsigned int active_rows, unsigned int row_offset) override {
    switch (static_cast<ActivationType>(act_type)) {
    case ActivationType::ACT_GELU:
      nntrainer::gelu_cl_op(in, out, /*mode=*/0, active_rows, row_offset);
      return;
    case ActivationType::ACT_TANH_GELU:
      nntrainer::gelu_cl_op(in, out, /*mode=*/1, active_rows, row_offset);
      return;
    default:
      throw std::invalid_argument(
        "ClComputeOps::activation: only gelu and tanh_gelu are accelerated on "
        "this backend; use the cpu engine for the other activations");
    }
  }

  bool supports_activation(int act_type) const override {
    const auto type = static_cast<ActivationType>(act_type);
    return type == ActivationType::ACT_GELU ||
           type == ActivationType::ACT_TANH_GELU;
  }

  // One residual-add operand, for the neutral AdditionLayer. FP32 same-size
  // operands take a host copy/add: the FP32 addition kernel reads its result
  // back into the caller's pointer, which is the very read into shared memory
  // that does not land (see the FP32 GEMM read-back), and both operands are
  // host-addressable here anyway.
  void residual_op(Tensor &hidden, const Tensor &input,
                   bool accumulate) override {
    const auto fp32 = ml::train::TensorDim::DataType::FP32;
    if (hidden.getDataType() == fp32 && input.getDataType() == fp32 &&
        hidden.size() == input.size()) {
      const size_t n = hidden.size();
      float *out = hidden.getData<float>();
      const float *in = input.getData<float>();
      if (!accumulate) {
        std::memcpy(out, in, n * sizeof(float));
      } else {
        for (size_t i = 0; i < n; ++i)
          out[i] += in[i];
      }
      return;
    }

    if (!accumulate) {
      hidden.copy(input);
    } else {
      nntrainer::add_i_cl(hidden, input);
    }
  }

  // Tensor::copy() reaches the table with no supports_*() guard, so without
  // these a copy of a tensor on this context would throw "not implemented" --
  // which the residual copy above does on the FP16 path. A host loop is
  // correct for host pointers and for host-coherent shared memory; moving the
  // copy onto a kernel is a residency refinement, not a correctness one.
  void scopy_fp32(const unsigned int N, const float *X, const unsigned int incX,
                  float *Y, const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }

#ifdef ENABLE_FP16
  void scopy_fp16(const unsigned int N, const _FP16 *X, const unsigned int incX,
                  _FP16 *Y, const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }
  // Mixed precision: an FP32 source feeding an FP16 graph, or an FP16 result
  // read back as FP32, both route here on this backend.
  void scopy_fp32_to_fp16(const unsigned int N, const float *X,
                          const unsigned int incX, _FP16 *Y,
                          const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = static_cast<_FP16>(X[i * incX]);
  }
  void scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, float *Y,
                          const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = static_cast<float>(X[i * incX]);
  }
#endif
};

ComputeOps *get_cl_ops() {
  static ClComputeOps instance;
  return &instance;
}

} // namespace nntrainer

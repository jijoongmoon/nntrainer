// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_compute_ops.h
 * @date    29 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA ComputeOps subclass: the op table the cuda context installs
 *          via ContextData::setComputeOps(), so the backend-neutral layers
 *          dispatch device work through getOps() instead of calling CUDA
 *          kernels directly. Inherits CpuComputeOps (not the abstract
 *          ComputeOps base): engine=cuda tensors are Unified Memory
 *          (host-coherent), so every op the CUDA backend does not accelerate
 *          runs correctly via the CPU implementation over the managed
 *          buffers -- but ONLY while the pool is managed. On a discrete GPU
 *          the context auto-arms the device-only activation pool, and there an
 *          inherited host body dereferences cudaMalloc memory.
 *
 *          INVARIANT this table enforces: every whole-op a layer dispatches
 *          through ComputeOps either has a device implementation here, or
 *          refuses at a NAMED guard (host_math_gate) -- it never silently runs
 *          host math on a device-only pointer. Each override therefore ends in
 *          host_math_gate() before delegating to the inherited body.
 */

#ifndef __CUDA_COMPUTE_OPS_H__
#define __CUDA_COMPUTE_OPS_H__

#include <cpu_ops_table.h>

namespace nntrainer {

/**
 * @brief CUDA op table: CpuComputeOps plus device overrides for the
 *        element-wise decode kernels. The host bodies stay correct on the
 *        UVM buffers, so an override is only added where a device kernel
 *        exists; everything else inherits.
 */
class CudaComputeOps : public CpuComputeOps {
public:
  /**
   * @brief SwiGLU whole-op: device-resident fp16 one-kernel fast path
   *        (cuda_swiglu_fp16) under the residency gates, else the inherited
   *        host body.
   */
  void swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
              unsigned int active_rows, unsigned int row_offset) override;

  /**
   * @brief GeGLU whole-op: out = gelu_tanh(gate) * up. Opt-in
   *        (NNTR_CUDA_GEGLU) device-resident fp16 kernel (cuda_geglu_fp16),
   *        else drain-then-host fallback.
   */
  void geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
             unsigned int active_rows, unsigned int row_offset) override;

  /**
   * @brief Fused sigmoid-GLU whole-op: fp16 device kernel
   *        (cuda_sigmoid_glu_fp16) on device-accessible tensors, else
   *        drain-then-host fallback. Kill-switch NNTR_CUDA_SIGMOID_GATE=0.
   */
  void sigmoid_glu(const Tensor &in1, const Tensor &in2, Tensor &out,
                   unsigned int active_rows, unsigned int row_offset) override;

  /**
   * @brief Fused sigmoid-add whole-op: fp16 device kernel
   *        (cuda_sigmoid_add_fp16) on device-accessible tensors, else
   *        drain-then-host fallback. Kill-switch NNTR_CUDA_SIGMOID_GATE=0.
   */
  void sigmoid_add(const Tensor &in1, const Tensor &in2, Tensor &out,
                   unsigned int active_rows, unsigned int row_offset) override;

  /**
   * @brief Scalar multiply whole-op: opt-in (NNTR_CUDA_ELTWISE) fp16 device
   *        kernel (cuda_scalar_mul_fp16), else drain-then-host fallback.
   */
  void scalar_mul(const Tensor &in, Tensor &out, float scale) override;

  /**
   * @brief Logit soft-capping whole-op: fp16 device kernel
   *        (cuda_softcap_fp16) on device-accessible logits, else the
   *        inherited host body. Carries the terminal pipeline drain for the
   *        selective-sync path (first host-read point of the logits).
   */
  void softcap(const Tensor &in, Tensor &out, float cap, int act_type) override;

  /**
   * @brief RMSNorm whole-op: block-per-row fp16 device kernel
   *        (cuda_rmsnorm_fp16, FP32 sum-of-squares) for decode-sized row
   *        counts on device-accessible tensors, else this backend's own
   *        fused host fallback (also FP32-accumulated) after the async
   *        coherence drain. Deliberately does NOT delegate to the inherited
   *        CpuComputeOps::rms_norm: the fallback here is the fused
   *        normalize*gamma loop this backend has always run, kept
   *        bit-for-bit.
   */
  void rms_norm(const Tensor &in, Tensor &out, const Tensor &gamma,
                float epsilon, unsigned int active_rows,
                unsigned int row_offset) override;

  /**
   * @brief Reverse-RMSNorm (per-layer-embedding post_norm) whole-op:
   *        y = (x*w / rms(x*w)) * out_scale, the per-feature weight folded
   *        INSIDE the denominator. FP16-ONLY device kernel + guard:
   *        cuda_rms_reverse_norm_fp16 (FP32 sum-of-squares) is the only kernel
   *        that exists, so an FP32 ACTIVATION has no device path here and
   *        takes the named guard / inherited host body. An FP32
   *        weight/out_scale does NOT decline: it is bound through the cached
   *        fp16 converter (the FP32-gamma case rms_norm had to be repaired
   *        for -- a norm weight is unquantized, so a quantized package pairs
   *        an FP16 activation with an FP32 weight).
   *        Kill-switch NNTR_CUDA_RMS_REVERSE_NORM=0.
   *
   *        Without this override the op inherited CpuComputeOps' host
   *        FP32-temp math, which on the device-only activation pool faulted
   *        inside avx2::vcvt_f16_f32 -- the gap this table's invariant exists
   *        to make impossible.
   */
  void rms_reverse_norm(Tensor &in, Tensor &out, const Tensor &weight,
                        const Tensor &out_scale, float epsilon,
                        unsigned int active_rows,
                        unsigned int row_offset) override;

  /**
   * @brief One residual-add operand: hidden = input, or hidden += input.
   *        The accumulate form is a fp16 device kernel (cuda_add_fp16) on
   *        device-accessible operands -- the inherited host body reaches
   *        Tensor::add_i -> ele_add_fp16, host math this table does not
   *        override. The copy form is gated on WHICH branch Tensor::copy will
   *        take: the matching-shape branch routes through scopy_fp16, which IS
   *        device-aware here, so it only drains; the mismatch branch host-reads
   *        the source and swaps in a host backing store, so it is refused by
   *        name. Kill-switch NNTR_CUDA_ELTWISE=0 (then the named guard refuses
   *        on a device-only pool rather than faulting).
   */
  void residual_op(Tensor &hidden, const Tensor &input,
                   bool accumulate) override;

  /**
   * @brief Fused activation epilogue. No device kernel: the LLM stack never
   *        sets a fused activation on an FC, so this exists to hold the
   *        invariant -- run the inherited host ActiFunc on a host-reachable
   *        output, refuse by name on a device-only one.
   */
  void apply_activation(Tensor &out, int act_type) override;

  /**
   * @brief FC GEMM whole-op: output = input * weight. QS4CX weight -> fused
   *        dequant-GEMM on device, consuming the PLAIN nibble payload in
   *        place (single weight copy, no UVM duplicate), else the inherited
   *        host dot after the async coherence drain. QINT4 never reaches
   *        here: layer_context coerces it to QS4CX at init.
   */
  void fc(Tensor &input, Tensor &weight, Tensor &output) override;

  /**
   * @brief Load-time QS4CX weight prefetch to device (opt-in,
   *        NNTR_CUDA_WPREFETCH >= 2), executed through the op-table prebuild
   *        seam from FullyConnectedLayerCl::read() inside the parallel load
   *        worker. Creates derived device residency only; never invalidates
   *        the host payload.
   */
  void fc_prebuild_weight(Tensor &w) override;

  // ── Copy ops (device-only aware) ───────────────────────────────────────
  // Under the device-only activation pool (NNTR_CUDA_DEV_ACT) an activation is
  // real device memory; Tensor::copy() -> the CpuComputeOps host loop would
  // fault on it. Route contiguous device-only copies through a stream-ordered
  // cudaMemcpyAsync; host / host-coherent UVM keep the CPU path.
  void scopy_fp32(const unsigned int N, const float *X, const unsigned int incX,
                  float *Y, const unsigned int incY) override;
#ifdef ENABLE_FP16
  void scopy_fp16(const unsigned int N, const _FP16 *X, const unsigned int incX,
                  _FP16 *Y, const unsigned int incY) override;
  // Converting copies with a device-only endpoint: stage through host temps
  // (synchronous; these do not occur inside graph capture today).
  void scopy_fp32_to_fp16(const unsigned int N, const float *X,
                          const unsigned int incX, _FP16 *Y,
                          const unsigned int incY) override;
  void scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, float *Y,
                          const unsigned int incY) override;
#endif
};

} // namespace nntrainer

#endif // __CUDA_COMPUTE_OPS_H__

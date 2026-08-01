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

#include <cstdint>

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
   *        take AND on the element type: only a matching-shape copy of a dtype
   *        whose ITensor::copy(const void *) lands exclusively in ops THIS
   *        table overrides is device-aware, so only that drains; the mismatch
   *        branch (host read + backing-store swap) and every other dtype are
   *        refused by name. Kill-switch NNTR_CUDA_ELTWISE=0 (then the named
   *        guard refuses on a device-only pool rather than faulting).
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
  //
  // The SAME invariant as the whole-ops above applies to this family, and it
  // applies to ALL of it: an op left un-overridden here inherits the
  // CpuComputeOps host element loop, which dereferences X and Y directly. The
  // family splits three ways:
  //
  //   * byte-identical moves (scopy_fp32/fp16/u8/s8) -> device_copy(), a
  //     stream-ordered cudaMemcpyAsync, dtype-agnostic and exact;
  //   * fp32<->fp16 conversion -> staged through host temps (these are on live
  //     paths: logits readback, activation dtype bridging);
  //   * every other CONVERSION (int4/int8/int16 <-> float, float -> narrow) ->
  //     NAMED REFUSAL. There is no device kernel for them, and the host bodies
  //     they would have to reproduce have irregular extents that differ per
  //     arch (e.g. scopy_int4_to_float32 reads N BYTES and writes 2N floats
  //     while ignoring incX/incY; scopy_int8_to_fp32_* index X by incY and Y by
  //     incX). Re-deriving that bug-compatibly for a staged copy, on paths no
  //     in-tree consumer can reach with a device pointer, would be untested
  //     code that can silently corrupt; the refusal cannot -- it fires only
  //     where the inherited host body would have faulted on the next
  //     instruction.
  void scopy_fp32(const unsigned int N, const float *X, const unsigned int incX,
                  float *Y, const unsigned int incY) override;
  /**
   * @brief Byte-identical uint8 move. Consumer: Uint4QTensor::copy(const void*)
   *        (the packed-nibble payload). Device-aware via device_copy().
   */
  void scopy_u8(const unsigned int N, const uint8_t *X, const unsigned int incX,
                uint8_t *Y, const unsigned int incY) override;
  /**
   * @brief Byte-identical int8 move. Consumers: CharTensor::copy(const void*)
   *        and Int4QTensor::copy(const void*). Device-aware via device_copy().
   *        This is the op that made a QINT8 residual copy pass the old
   *        shape-only gate and then land in the host loop on device memory.
   */
  void scopy_s8(const unsigned int N, const int8_t *X, const unsigned int incX,
                int8_t *Y, const unsigned int incY) override;

  // Converting copies with no device kernel -- named refusal (see the block
  // comment above). FP32 half of the family.
  void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                             const unsigned int incX, float *Y,
                             const unsigned int incY) override;
  void scopy_int8_to_fp32_u(const unsigned int N, const uint8_t *X,
                            const unsigned int incX, float *Y,
                            const unsigned int incY) override;
  void scopy_int8_to_fp32_s(const unsigned int N, const int8_t *X,
                            const unsigned int incX, float *Y,
                            const unsigned int incY) override;
  void copy_s16_fp32(const unsigned int N, const int16_t *X,
                     float *Y) override;
  void copy_u16_fp32(const unsigned int N, const uint16_t *X,
                     float *Y) override;
  void copy_fp32_u32(const unsigned int N, const float *X,
                     uint32_t *Y) override;
  void copy_fp32_u16(const unsigned int N, const float *X,
                     uint16_t *Y) override;
  void copy_fp32_u8(const unsigned int N, const float *X, uint8_t *Y) override;
  void copy_fp32_s16(const unsigned int N, const float *X,
                     int16_t *Y) override;
  void copy_fp32_s8(const unsigned int N, const float *X, int8_t *Y) override;
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
  // Converting copies with no device kernel -- named refusal. FP16 half.
  void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                             const unsigned int incX, _FP16 *Y,
                             const unsigned int incY) override;
  void scopy_int8_to_float16_u(const unsigned int N, const uint8_t *X,
                               const unsigned int incX, _FP16 *Y,
                               const unsigned int incY) override;
  void scopy_int8_to_float16_s(const unsigned int N, const int8_t *X,
                               const unsigned int incX, _FP16 *Y,
                               const unsigned int incY) override;
#endif
};

} // namespace nntrainer

#endif // __CUDA_COMPUTE_OPS_H__

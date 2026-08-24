// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cuda_compute_ops.cpp
 * @date   22 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  CUDA ComputeOps subclass (mirror of ClComputeOps). P1 provides only
 *         the host-side copy ops so Tensor::copy() works on engine=cuda tensors
 *         (their memory is Unified/managed, hence host-addressable). The
 *         accelerator quantized GEMM/GEMV predicates are left at the base
 *         default (false), so float_tensor.cpp falls back to the CPU path until
 *         the CUDA kernels land in P3 (cuda_operations/).
 */

#include <env_compat.h>
#include <common_properties.h> // ActivationType (the act_type int encoding)
#include <compute_ops.h>
#include <cpu_ops_table.h>
#include <nntrainer_log.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <tensor.h>

#include <cuda_stream_manager.h>
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_gelu.h>
#include <cuda_layernorm.h>
#include <cuda_runtime.h>
#include <fp16.h>
#include <map>
#include <mutex>
#include <utility>
#include <vector>
#endif

namespace nntrainer {

// CudaComputeOps derives from CpuComputeOps (not the abstract ComputeOps base):
// engine=cuda tensors are Unified Memory (host-coherent), so every standard op
// runs correctly via the CPU implementations; this class only overrides the
// host-side copy ops for now. Inheriting CpuComputeOps means get_cuda_ops() can
// be installed without throwing on the un-accelerated ops (prereq for the CUDA
// op kernels in a later phase).
class CudaComputeOps : public CpuComputeOps {
public:
  // Plain elementwise copy (Y = X). Tensor::copy() calls this unconditionally
  // (no supports_*() guard); correct for host and (host-coherent) managed
  // pointers. Under the device-only pools (NNTR_CUDA_DEV_ACT / KV_DEV) either
  // endpoint may be cudaMalloc memory the host loop below would fault on --
  // device_copy() routes contiguous same-type copies through a stream-ordered
  // cudaMemcpyAsync (legal inside graph capture, ordered against the
  // producing kernels on the same stream); a copy the host reads next (D2H)
  // drains first. Strided device copies do not occur in the forward path --
  // fail loudly rather than fault.
  static bool device_copy(const void *X, void *Y, size_t bytes,
                          bool contiguous) {
    if (!(cuda::dev_only(X) || cuda::dev_only(Y)))
      return false;
    if (!contiguous)
      throw std::runtime_error(
        "CudaComputeOps: strided copy on device-only memory is unsupported");
    auto &sm = cuda::StreamManager::Global();
    if (cudaMemcpyAsync(Y, X, bytes, cudaMemcpyDefault, sm.GetStream()) !=
        cudaSuccess) {
      cudaGetLastError();
      throw std::runtime_error(
        "CudaComputeOps: device copy (cudaMemcpyAsync) failed");
    }
    if (!cuda::dev_only(Y))
      sm.finish(); // D2H: the host consumes the destination immediately
    return true;
  }

  void scopy_fp32(const unsigned int N, const float *X, const unsigned int incX,
                  float *Y, const unsigned int incY) override {
    if (device_copy(X, Y, (size_t)N * sizeof(float), incX == 1 && incY == 1))
      return;
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }

#ifdef ENABLE_FP16
  void scopy_fp16(const unsigned int N, const _FP16 *X, const unsigned int incX,
                  _FP16 *Y, const unsigned int incY) override {
    if (device_copy(X, Y, (size_t)N * sizeof(_FP16), incX == 1 && incY == 1))
      return;
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }
  // Converting copies with a device-only endpoint: stage through host temps
  // (synchronous; these do not occur inside graph capture today).
  void scopy_fp32_to_fp16(const unsigned int N, const float *X,
                          const unsigned int incX, _FP16 *Y,
                          const unsigned int incY) override {
    if (cuda::dev_only(X) || cuda::dev_only(Y)) {
      if (incX != 1 || incY != 1)
        throw std::runtime_error(
          "CudaComputeOps: strided converting copy on device-only memory");
      cuda::StreamManager::Global().finish();
      std::vector<float> xs;
      const float *xp = X;
      if (cuda::dev_only(X)) {
        xs.resize(N);
        cuda::copy_any(xs.data(), X, (size_t)N * sizeof(float));
        xp = xs.data();
      }
      std::vector<_FP16> ys(N);
      for (unsigned int i = 0; i < N; ++i)
        ys[i] = static_cast<_FP16>(xp[i]);
      if (cuda::dev_only(Y))
        cuda::copy_any(Y, ys.data(), (size_t)N * sizeof(_FP16));
      else
        std::memcpy(Y, ys.data(), (size_t)N * sizeof(_FP16));
      return;
    }
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = static_cast<_FP16>(X[i * incX]);
  }
  void scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, float *Y,
                          const unsigned int incY) override {
    if (cuda::dev_only(X) || cuda::dev_only(Y)) {
      if (incX != 1 || incY != 1)
        throw std::runtime_error(
          "CudaComputeOps: strided converting copy on device-only memory");
      cuda::StreamManager::Global().finish();
      std::vector<_FP16> xs;
      const _FP16 *xp = X;
      if (cuda::dev_only(X)) {
        xs.resize(N);
        cuda::copy_any(xs.data(), X, (size_t)N * sizeof(_FP16));
        xp = xs.data();
      }
      std::vector<float> ys(N);
      for (unsigned int i = 0; i < N; ++i)
        ys[i] = static_cast<float>(xp[i]);
      if (cuda::dev_only(Y))
        cuda::copy_any(Y, ys.data(), (size_t)N * sizeof(float));
      else
        std::memcpy(Y, ys.data(), (size_t)N * sizeof(float));
      return;
    }
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = static_cast<float>(X[i * incX]);
  }
#endif

  // ── Whole-op (Tensor-level) ───────────────────────────────────────────────
  // GeGLU: out = gelu_tanh(gate) * up. Device-resident fp16 kernel (opt-in via
  // NNTR_CUDA_GEGLU until the whole decode chain is on-GPU); otherwise the host
  // gelu loop on the host-coherent UVM tensors (CpuComputeOps::geglu). Matches
  // the former forked GeGLU layer's math byte-for-byte.
  void geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
             unsigned int active_rows, unsigned int row_offset) override {
    const unsigned int dim2 = in1.width();
    const size_t elem_off = (size_t)row_offset * dim2;
    const size_t n = (size_t)active_rows * dim2;
    const auto dt = in1.getDataType();

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // GPU geglu (device-resident fp16): one kernel instead of the host loop, so
    // the FFN/PLE activation stays on the device. NNTR_CUDA_ASYNC governs the
    // drain.
    if (dt == ml::train::TensorDim::DataType::FP16) {
      static const bool gpu = nntr_env_on("NNTR_CUDA_GEGLU");
      if (gpu && n > 0) {
        auto *a =
          reinterpret_cast<const unsigned short *>(in1.getData<_FP16>() +
                                                   elem_off);
        auto *b =
          reinterpret_cast<const unsigned short *>(in2.getData<_FP16>() +
                                                   elem_off);
        auto *o = reinterpret_cast<unsigned short *>(out.getData<_FP16>() +
                                                     elem_off);
        const bool dev = nntrainer::cuda::dev_accessible(a);
        if (dev && cuda::cuda_geglu_fp16(a, b, o, (unsigned int)n))
          return;
      }
    }
#endif

    // Host gelu fallback: sync first so the host read of GPU-produced gate/up is
    // coherent under NNTR_CUDA_ASYNC (no-op in sync mode).
    cuda::StreamManager::Global().finishIfAsync();
    CpuComputeOps::geglu(in1, in2, out, active_rows, row_offset);
  }

  // Fused sigmoid gates on cuda (mirror of geglu above). The cuda activation
  // pool is device-resident, so the DEVICE kernel is the primary path (the
  // base CpuComputeOps host loop faults on it -- that was the CUDA-engine
  // SIGSEGV in runDecode). Host loop only for genuinely host tensors.
  // Kill-switch: NNTR_CUDA_SIGMOID_GATE=0.
  void sigmoid_glu(const Tensor &in1, const Tensor &in2, Tensor &out,
                   unsigned int active_rows,
                   unsigned int row_offset) override {
    const unsigned int dim2 = in1.width();
    const size_t elem_off = (size_t)row_offset * dim2;
    const size_t n = (size_t)active_rows * dim2;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    if (in1.getDataType() == ml::train::TensorDim::DataType::FP16 && n > 0) {
      static const bool gpu = []() {
        const char *e = std::getenv("NNTR_CUDA_SIGMOID_GATE");
        return !(e && e[0] == '0');
      }();
      if (gpu) {
        auto *a = reinterpret_cast<const unsigned short *>(
          in1.getData<_FP16>() + elem_off);
        auto *b = reinterpret_cast<const unsigned short *>(
          in2.getData<_FP16>() + elem_off);
        auto *o =
          reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
        if (nntrainer::cuda::dev_accessible(a) &&
            cuda::cuda_sigmoid_glu_fp16(a, b, o, (unsigned int)n))
          return;
      }
    }
#endif
    cuda::StreamManager::Global().finishIfAsync();
    CpuComputeOps::sigmoid_glu(in1, in2, out, active_rows, row_offset);
  }
  void sigmoid_add(const Tensor &in1, const Tensor &in2, Tensor &out,
                   unsigned int active_rows,
                   unsigned int row_offset) override {
    const unsigned int dim2 = in1.width();
    const size_t elem_off = (size_t)row_offset * dim2;
    const size_t n = (size_t)active_rows * dim2;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    if (in1.getDataType() == ml::train::TensorDim::DataType::FP16 && n > 0) {
      static const bool gpu = []() {
        const char *e = std::getenv("NNTR_CUDA_SIGMOID_GATE");
        return !(e && e[0] == '0');
      }();
      if (gpu) {
        auto *a = reinterpret_cast<const unsigned short *>(
          in1.getData<_FP16>() + elem_off);
        auto *b = reinterpret_cast<const unsigned short *>(
          in2.getData<_FP16>() + elem_off);
        auto *o =
          reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
        if (nntrainer::cuda::dev_accessible(a) &&
            cuda::cuda_sigmoid_add_fp16(a, b, o, (unsigned int)n))
          return;
      }
    }
#endif
    cuda::StreamManager::Global().finishIfAsync();
    CpuComputeOps::sigmoid_add(in1, in2, out, active_rows, row_offset);
  }

  // LayerNorm: out = (x-mean)*rsqrt(var+eps)*gamma + beta per row over width.
  // Device fp16 kernel for all-FP16 in/gamma/beta/out within the row gate;
  // everything else (FP32, every mixed activation/weight dtype combo, and
  // rows > gate) runs the INHERITED host loop CpuComputeOps::layer_norm over
  // the host-coherent UVM tensors — i.e. UNACCELERATED rather than
  // "CUDA support". cuda_layernorm_fp32 exists and is
  // covered by unittest_cuda_kernels_layernorm, but is deliberately not routed
  // from here yet: it has had no in-graph validation, unlike the fp16 path.
  //
  // The row gate is a CUDA-specific PERFORMANCE POLICY and belongs here, in the
  // op, never in the Layer (a Layer branching on backend behaviour is exactly
  // the fork smell this collapse removes). Rationale: the kernel syncs per
  // call, so for a wide prefill norm (rows = seq_len) the multi-threaded host
  // loop over UVM wins; gating by rows gives the decode speedup with no prefill
  // regression (same tradeoff as CudaRMSNormLayer). ClComputeOps gets NO
  // equivalent gate and that is correct, not an oversight — it has no host
  // fallback to fall back to. Replaces the former forked LayerNorm layer.
  void layer_norm(const Tensor &in, Tensor &out, const Tensor &gamma,
                  const Tensor &beta, float epsilon, unsigned int active_rows,
                  unsigned int row_offset) override {
    const unsigned int width = in.width();
    const size_t elem_off = (size_t)row_offset * width;

    if (std::getenv("NNTR_CUDA_DBG")) {
      static int _n = 0;
      if (_n++ < 3)
        std::fprintf(stderr,
                     "[CUDA-DBG] CudaComputeOps::layer_norm rows=%u width=%u\n",
                     active_rows, width);
    }

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    using DT = ml::train::TensorDim::DataType;
    // NNTR_LAYERNORM_CUDA_OFF: unset => 32-row decode-only cap, "a"/"all" =>
    // uncapped, anything else => off. CudaContext::initialize() sets "all" on
    // discrete GPUs next to the RMSNorm cap raise.
    static const int gpu_max_rows = []() {
      const char *e = std::getenv("NNTR_LAYERNORM_CUDA_OFF");
      if (e && e[0] == 'a')
        return 1 << 30; // "all"
      if (e)
        return 0; // off
      return 32;  // decode-only default
    }();
    if (in.getDataType() == DT::FP16 && gamma.getDataType() == DT::FP16 &&
        beta.getDataType() == DT::FP16 && out.getDataType() == DT::FP16 &&
        (int)active_rows <= gpu_max_rows && active_rows > 0) {
      auto *xi = reinterpret_cast<const unsigned short *>(in.getData<_FP16>() +
                                                          elem_off);
      auto *gi =
        reinterpret_cast<const unsigned short *>(gamma.getData<_FP16>());
      auto *bi = reinterpret_cast<const unsigned short *>(beta.getData<_FP16>());
      auto *yi =
        reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
      if (nntrainer::cuda::dev_accessible(xi) &&
          nntrainer::cuda::dev_accessible(gi) &&
          nntrainer::cuda::dev_accessible(bi) &&
          nntrainer::cuda::dev_accessible(yi) &&
          cuda::cuda_layernorm_fp16(xi, gi, bi, yi, epsilon, active_rows, width))
        return;
    }
#endif
    // Host layernorm fallback (UNACCELERATED): sync first so the host read of a
    // GPU-produced input is coherent under NNTR_CUDA_ASYNC (no-op in sync mode).
    cuda::StreamManager::Global().finishIfAsync();
    CpuComputeOps::layer_norm(in, out, gamma, beta, epsilon, active_rows,
                              row_offset);
  }

  // Element-wise activation. Device fp16 GELU/tanh-GELU kernel; every other
  // mode and every other dtype runs the INHERITED host ActiFunc over UVM —
  // UNACCELERATED, so say so rather than claiming CUDA support.
  // Note there is NO row gate here and there should not be one: this is a flat
  // 1-D elementwise map with no host-wins crossover (mirrors the ungated
  // swiglu/geglu CUDA fast paths). Replaces the former CudaActivationLayer
  // fork; the ActivationType -> mode mapping (its getGeluMode) lives here now,
  // because it is a backend concern.
  void activation(const Tensor &in, Tensor &out, int act_type,
                  unsigned int active_rows, unsigned int row_offset) override {
    const auto at = static_cast<ActivationType>(act_type);
    const unsigned int width = in.width();
    const size_t elem_off = (size_t)row_offset * width;
    const size_t n = (size_t)active_rows * width;
    const bool is_gelu =
      (at == ActivationType::ACT_GELU || at == ActivationType::ACT_TANH_GELU);

    if (std::getenv("NNTR_CUDA_DBG")) {
      static int _n = 0;
      if (_n++ < 3)
        std::fprintf(stderr,
                     "[CUDA-DBG] CudaComputeOps::activation n=%zu act=%d\n", n,
                     act_type);
    }

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    using DT = ml::train::TensorDim::DataType;
    if (is_gelu && n > 0 && in.getDataType() == DT::FP16 &&
        out.getDataType() == DT::FP16) {
      const int mode = (at == ActivationType::ACT_TANH_GELU) ? 1 : 0;
      auto *xi = reinterpret_cast<const unsigned short *>(in.getData<_FP16>() +
                                                          elem_off);
      auto *yi =
        reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
      if (nntrainer::cuda::dev_accessible(xi) &&
          nntrainer::cuda::dev_accessible(yi) &&
          cuda::cuda_gelu_fp16(xi, yi, mode, (unsigned int)n))
        return;
    }
#endif

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // Under a device-only activation pool (NNTR_CUDA_DEV_ACT) the host loop
    // below would FAULT on a non-UVM pointer. Fail loudly instead: the caller
    // must either use an accelerated mode/dtype or turn the pool off.
    if (n > 0 && (cuda::dev_only(in.getData<uint8_t>()) ||
                  cuda::dev_only(out.getData<uint8_t>())))
      throw std::runtime_error(
        "CudaComputeOps::activation: this activation mode/dtype has no device "
        "kernel and the tensors are device-only (NNTR_CUDA_DEV_ACT); the host "
        "path would fault");
#endif

    cuda::StreamManager::Global().finishIfAsync();
    CpuComputeOps::activation(in, out, act_type, active_rows, row_offset);
  }
};

ComputeOps *get_cuda_ops() {
  static CudaComputeOps instance;
  return &instance;
}

} // namespace nntrainer

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

#include <compute_ops.h>
#include <cpu_ops_table.h>

#include <cstdlib>

#include <tensor.h>

#include <cuda_stream_manager.h>
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#endif

namespace nntrainer {

// CudaComputeOps derives from CpuComputeOps (not the abstract ComputeOps base):
// engine=cuda tensors are Unified Memory (host-coherent), so every standard op
// runs correctly via the CPU implementations; this class only overrides the
// host-side copy ops for now. Inheriting CpuComputeOps means get_cuda_ops() can
// be installed without throwing on the un-accelerated ops (prereq for the CUDA
// op kernels in a later phase). [T6]
class CudaComputeOps : public CpuComputeOps {
public:
  // Plain elementwise copy (Y = X). Tensor::copy() calls this unconditionally
  // (no supports_*() guard); correct for host and (host-coherent) managed
  // pointers. A device-kernel copy is a later residency refinement.
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

  // ── Whole-op (Tensor-level) ───────────────────────────────────────────────
  // GeGLU: out = gelu_tanh(gate) * up. Device-resident fp16 kernel (opt-in via
  // NNTR_CUDA_GEGLU until the whole decode chain is on-GPU); otherwise the host
  // gelu loop on the host-coherent UVM tensors (CpuComputeOps::geglu). Matches
  // the former CudaGeGLULayer::gegluProcess byte-for-byte. [T7]
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
      static const bool gpu = std::getenv("NNTR_CUDA_GEGLU") != nullptr;
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
};

ComputeOps *get_cuda_ops() {
  static CudaComputeOps instance;
  return &instance;
}

} // namespace nntrainer

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
};

ComputeOps *get_cuda_ops() {
  static CudaComputeOps instance;
  return &instance;
}

} // namespace nntrainer

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

};

ComputeOps *get_cuda_ops() {
  static CudaComputeOps instance;
  return &instance;
}

} // namespace nntrainer

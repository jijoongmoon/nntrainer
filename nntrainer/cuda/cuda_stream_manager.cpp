// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_stream_manager.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA stream/dispatch management implementation.
 */

#include "cuda_stream_manager.h"
#include "cuda_common.h"
#include "cuda_context_manager.h"
#include "cuda_kernel.h"

#include <cstdlib>

namespace nntrainer::cuda {

void StreamManager::initialize() noexcept {
  // make sure the device + primary context exist before creating a stream
  ContextManager::Global().EnsureCurrent();
  if (!cudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate"))
    stream_ = nullptr;
}

bool StreamManager::EnqueueWriteBuffer(void *dst_dev, size_t size,
                                       const void *src_host, bool async) {
  if (!cudaCheck(cudaMemcpyAsync(dst_dev, src_host, size,
                                 cudaMemcpyHostToDevice, stream_),
                 "cudaMemcpyAsync H2D"))
    return false;
  if (!async)
    return cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  return true;
}

bool StreamManager::EnqueueReadBuffer(const void *src_dev, size_t size,
                                      void *dst_host, bool async) {
  if (!cudaCheck(cudaMemcpyAsync(dst_host, src_dev, size,
                                 cudaMemcpyDeviceToHost, stream_),
                 "cudaMemcpyAsync D2H"))
    return false;
  if (!async)
    return cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  return true;
}

bool StreamManager::DispatchCommand(Kernel &kernel, const int (&grid)[3],
                                    const int (&block)[3],
                                    unsigned int shared_bytes) {
  if (!kernel.valid()) {
    ml_loge("[CUDA] DispatchCommand: invalid kernel");
    return false;
  }
  ContextManager::Global().EnsureCurrent();
  auto params = kernel.getKernelParams();
  CUresult r = cuLaunchKernel(
    kernel.GetFunction(), (unsigned)grid[0], (unsigned)grid[1],
    (unsigned)grid[2], (unsigned)block[0], (unsigned)block[1],
    (unsigned)block[2], shared_bytes, reinterpret_cast<CUstream>(stream_),
    params.empty() ? nullptr : params.data(), nullptr);
  return cuCheck(r, "cuLaunchKernel");
}

void StreamManager::finish() {
  if (stream_)
    cudaStreamSynchronize(stream_);
}

static bool cuda_async_mode() {
  static const bool async = []() {
    const char *e = std::getenv("NNTR_CUDA_ASYNC");
    return e != nullptr && e[0] == '1';
  }();
  return async;
}

void StreamManager::maybeFinish() {
  if (!cuda_async_mode())
    finish();
}

void StreamManager::finishIfAsync() {
  if (cuda_async_mode())
    finish();
}

StreamManager::~StreamManager() {
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

} // namespace nntrainer::cuda

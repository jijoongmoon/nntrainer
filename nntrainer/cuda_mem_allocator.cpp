// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_mem_allocator.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA Unified Memory allocator implementation.
 */

#include <cuda_mem_allocator.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_set>

#include <cuda_runtime.h>

#include <cuda_context_manager.h>

namespace nntrainer {

namespace {
// Host-fallback ownership set (pointers from MemAllocator::alloc, not
// cudaMallocManaged). ContextManager is a global singleton so no per-instance
// state is needed; keep this hidden in the .cpp.
std::mutex host_owned_mtx;
std::unordered_set<void *> host_owned;
} // namespace

CudaMemAllocator::CudaMemAllocator() {
  // bring up the device + primary context once (idempotent)
  cuda::ContextManager::Global();
}

void CudaMemAllocator::track_host_owned(void *ptr) {
  std::lock_guard<std::mutex> lk(host_owned_mtx);
  host_owned.insert(ptr);
}

bool CudaMemAllocator::consume_host_owned(void *ptr) {
  std::lock_guard<std::mutex> lk(host_owned_mtx);
  return host_owned.erase(ptr) > 0;
}

void CudaMemAllocator::alloc(void **ptr, size_t size, size_t alignment) {
  static const bool dbg = std::getenv("NNTR_UVM_DEBUG") != nullptr;
  if (size > 0) {
    void *managed = nullptr;
    if (cudaMallocManaged(&managed, size) == cudaSuccess && managed != nullptr) {
      *ptr = managed;
      if (dbg)
        fprintf(stderr, "[UVMDBG] cudaMallocManaged %zu bytes -> %p OK\n", size,
                managed);
      return;
    }
    if (dbg)
      fprintf(stderr, "[UVMDBG] cudaMallocManaged %zu bytes FAILED -> host\n",
              size);
    // a failed managed alloc leaves the runtime error state set; clear it so a
    // subsequent real CUDA op does not see this benign fallback as an error.
    cudaGetLastError();
  }
  // size==0 or managed alloc failed -> host buffer (correctness > speed). The
  // matching free() consults host_owned to pick std::free vs cudaFree.
  MemAllocator::alloc(ptr, size, alignment);
  track_host_owned(*ptr);
}

void CudaMemAllocator::free(void *ptr) {
  if (ptr == nullptr)
    return;
  if (consume_host_owned(ptr)) {
    MemAllocator::free(ptr);
    return;
  }
  cudaFree(ptr);
}

} // namespace nntrainer

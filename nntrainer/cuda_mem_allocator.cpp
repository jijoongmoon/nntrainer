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

CudaMemAllocator::CudaMemAllocator(bool device_only) :
  device_only_(device_only) {
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
    void *dptr = nullptr;
    // device_only -> real device memory (cudaMalloc, NOT host-addressable);
    // else UVM (cudaMallocManaged, host-coherent). The activation pool uses
    // device_only so the CPU never migrates its pages (the async thrash);
    // weights stay UVM (host writes them at load).
    //
    // INTEGRATED GPU (Tegra/Jetson Orin): host and device share one physical
    // memory pool, so a device-only cudaMalloc gives ZERO bandwidth benefit
    // yet makes the buffer host-INCOHERENT -- the host RoPE/short-attention/
    // norm fallbacks then dereference a device pointer and fault/garbage. Force
    // the managed (host-coherent) pool there regardless of the device_only_
    // request, so the same binary stays coherent on Orin even if a launch
    // script leaks NNTR_CUDA_DEV_ACT. The dev()/dev_ok() GPU gates still pass
    // because they accept Managed too. Discrete GPUs are unaffected.
    const bool integrated = cuda::ContextManager::Global().isIntegrated();
    const bool dev_only = device_only_ && !integrated;
    const cudaError_t e = dev_only ? cudaMalloc(&dptr, size)
                                   : cudaMallocManaged(&dptr, size);
    if (e == cudaSuccess && dptr != nullptr) {
      if (!dev_only && !integrated) {
        // Optionally pin the managed pages to the device. Opt-in
        // (NNTR_CUDA_UVM_DEVICE); a partial, weaker alternative to device_only
        // (kept for A/B). Meaningless for real device memory AND for an
        // integrated GPU (no distinct device location to migrate to; can
        // mis-pin on some Tegra drivers), so it is skipped when integrated.
        static const bool pin_device =
          std::getenv("NNTR_CUDA_UVM_DEVICE") != nullptr;
        if (pin_device) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12020
          int dev = 0;
          cudaGetDevice(&dev);
          cudaMemLocation loc;
          loc.type = cudaMemLocationTypeDevice;
          loc.id = dev;
          cudaMemAdvise(dptr, size, cudaMemAdviseSetPreferredLocation, loc);
          cudaMemAdvise(dptr, size, cudaMemAdviseSetAccessedBy, loc);
          cudaGetLastError(); // clear any benign advise error
#endif
        }
      }
      *ptr = dptr;
      if (dbg)
        fprintf(stderr, "[UVMDBG] %s %zu bytes -> %p OK\n",
                dev_only ? "cudaMalloc" : "cudaMallocManaged", size, dptr);
      return;
    }
    if (dbg)
      fprintf(stderr, "[UVMDBG] %s %zu bytes FAILED -> host\n",
              dev_only ? "cudaMalloc" : "cudaMallocManaged", size);
    // a failed device alloc leaves the runtime error state set; clear it so a
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

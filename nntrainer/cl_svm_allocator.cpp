// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cl_svm_allocator.cpp
 * @date    27 Apr 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Implementation of OpenCL SVM-backed MemAllocator subclass.
 */

#include <env_compat.h>
#include <cl_buffer_pool.h>
#include <cl_svm_allocator.h>
#include <cstdlib>
#include <memory_pool.h>
#include <mutex>
#include <opencl_context_manager.h>
#include <unordered_set>

namespace nntrainer {

namespace {
// Inline static state — no per-instance state needed since
// ContextManager is itself a global singleton. Keeping this hidden
// in a .cpp namespace keeps the header lean.
std::mutex host_owned_mtx;
std::unordered_set<void *> host_owned;
} // namespace

ClSVMAllocator::ClSVMAllocator(opencl::ContextManager &ctx) : ctx_(ctx) {}

std::shared_ptr<MemoryPool>
ClSVMAllocator::makePool(const std::shared_ptr<MemAllocator> &self,
                         const std::string &pool_name) {
  // NNTR_GPU_CLMEM_POOL: back the activation plane with a device cl_mem pool
  // (ClBufferPool) so activations stay GPU-resident as plain cl_mem; default
  // OFF => the SVM-backed MemoryPool. Same condition as the old TensorPool
  // getName()=="gpu-svm" && env check — now owned by the allocator. [Mem M2]
  if (nntr_env_on("NNTR_GPU_CLMEM_POOL")) {
    // [weight plane skip] No WEIGHT tensor ever binds its per-offset cl_mem:
    // the v8c FCs read their own packed backing and the norms are host-read
    // (runtime residency dump: 566/566 gauss4 weights classify SVM), yet the
    // shadow plane is fully committed by the Windows/WDDM driver (~1373MB in
    // the process working set for gauss4). Give the never-consumed plane only
    // to the pools that use it (activations); the weight pool keeps the plain
    // SVM MemoryPool. NNTR_CLMEM_WEIGHT_PLANE=1 restores the old behavior.
    static const bool force_weight_plane = []() {
      const char *e = std::getenv("NNTR_CLMEM_WEIGHT_PLANE");
      return e != nullptr && e[0] == '1';
    }();
    // x86 only for now: the never-bound claim is field-verified on Intel
    // (Windows residency dump + Linux A/B: no RSS/golden change). Adreno also
    // auto-sets NNTR_GPU_CLMEM_POOL and its binding pattern is unverified on
    // device -- keep the old plane there until a device A/B clears it.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    const bool skip_weight_plane =
      pool_name == "weight_pool" && !force_weight_plane;
#else
    const bool skip_weight_plane = false;
    (void)force_weight_plane;
#endif
    if (!skip_weight_plane)
      return std::make_shared<ClBufferPool>(self);
  }
  return std::make_shared<MemoryPool>(self);
}

void ClSVMAllocator::track_host_owned(void *ptr) {
  std::lock_guard<std::mutex> lk(host_owned_mtx);
  host_owned.insert(ptr);
}

bool ClSVMAllocator::consume_host_owned(void *ptr) {
  std::lock_guard<std::mutex> lk(host_owned_mtx);
  return host_owned.erase(ptr) > 0;
}

void ClSVMAllocator::alloc(void **ptr, size_t size, size_t alignment) {
  void *svm = ctx_.createSVMRegion(size);
  if (svm != nullptr) {
    *ptr = svm;
    return;
  }
  // SVM not available (driver lacks support, or capacity exhausted) —
  // fall back to a host buffer so the layer still runs (correctness >
  // speed). Caller is unaware; only the matching free() needs to pick
  // the right release path.
  MemAllocator::alloc(ptr, size, alignment);
  track_host_owned(*ptr);
}

void ClSVMAllocator::free(void *ptr) {
  if (ptr == nullptr)
    return;
  if (consume_host_owned(ptr)) {
    MemAllocator::free(ptr);
    return;
  }
  ctx_.releaseSVMRegion(ptr);
}

} // namespace nntrainer

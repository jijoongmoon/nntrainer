// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_mem_allocator.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   MemAllocator subclass that routes MemoryPool allocations through CUDA
 *          Unified Memory (cudaMallocManaged), the direct analogue of the
 *          OpenCL SVM path (ClSVMAllocator): one pointer that is both host-
 *          addressable and device-accessible, so engine=cuda tensors are
 *          device-resident without a separate copy step. CudaContext installs an
 *          instance at engine init. Falls back to host memory if the managed
 *          allocation fails, so a missing-UVM device degrades rather than fails.
 */

#ifndef __CUDA_MEM_ALLOCATOR_H__
#define __CUDA_MEM_ALLOCATOR_H__

#include <string>

#include <mem_allocator.h>

namespace nntrainer {

/**
 * @brief CUDA Unified Memory (managed) allocator.
 */
class CudaMemAllocator : public MemAllocator {
public:
  /**
   * @brief Ensure the CUDA device + primary context exist before any
   *        managed allocation.
   *
   * @param device_only when true, alloc() uses cudaMalloc (real device memory,
   *        NOT host-addressable) instead of cudaMallocManaged (UVM). Used for
   *        the activation pool so the CPU never touches it -> no host<->device
   *        page migration under async (the UVM thrash). Weights keep UVM (false)
   *        because the host writes them at load. The OpenCL cl_mem analog.
   */
  explicit CudaMemAllocator(bool device_only = false);

  /**
   * @copydoc MemAllocator::alloc
   *
   * Tries cudaMallocManaged first; on failure (or size==0) falls back to
   * MemAllocator::alloc and records the pointer as host-owned so free() picks
   * the right release path.
   */
  void alloc(void **ptr, size_t size, size_t alignment) override;

  /**
   * @copydoc MemAllocator::free
   *
   * cudaFree for managed pointers, std::free for host-fallback pointers.
   */
  void free(void *ptr) override;

  std::string getName() override { return device_only_ ? "cuda-dev" : "cuda-uvm"; }

private:
  bool device_only_;
  void track_host_owned(void *ptr);
  bool consume_host_owned(void *ptr);
};

} // namespace nntrainer

#endif // __CUDA_MEM_ALLOCATOR_H__

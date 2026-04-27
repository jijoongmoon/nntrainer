// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    mem_allocator.h
 * @date    13 Jan 2025
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is memory allocator for memory pool
 *
 */
#ifndef __MEM_ALLOCATOR_H__
#define __MEM_ALLOCATOR_H__

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nntrainer {

/**
 * @brief MemAllocator, Memory allocator class
 */
class MemAllocator {
public:
  MemAllocator() = default;
  virtual ~MemAllocator() = default;
  virtual void alloc(void **ptr, size_t size, size_t alignment);
  virtual void free(void *ptr);
  virtual std::string getName() { return "cpu"; };
};

/**
 * @brief Page-aligned host memory allocator.
 *
 * Uses std::aligned_alloc on POSIX and _aligned_malloc on Windows so the
 * returned block satisfies the requested alignment. Replaces the ad-hoc
 * ALIGNED_ALLOC/ALIGNED_FREE macros previously inlined in MemoryPool.
 */
class CpuMemAllocator : public MemAllocator {
public:
  CpuMemAllocator() = default;
  ~CpuMemAllocator() override = default;
  void alloc(void **ptr, size_t size, size_t alignment) override;
  void free(void *ptr) override;
  std::string getName() override { return "cpu"; }
};

/**
 * @brief Hexagon NPU shared-memory allocator backed by libcdsprpc.
 *
 * Available on Android+NPU builds only. Captures the rpcmem heap_id and
 * flags so MemoryPool does not need to know rpcmem-specific arguments.
 * Outside of Android+NPU builds the class is not defined and tryCreate()
 * returns nullptr — callers should fall back to CpuMemAllocator.
 */
class RpcMemAllocator : public MemAllocator {
public:
#if defined(__ANDROID__) && ENABLE_NPU
  /// Default heap id and flags used by the previous in-pool implementation.
  static constexpr int kDefaultHeapId = 25;        // RPCMEM_HEAP_ID_SYSTEM
  static constexpr uint32_t kDefaultFlags = 1;     // RPCMEM_DEFAULT_FLAGS

  explicit RpcMemAllocator(int heap_id = kDefaultHeapId,
                           uint32_t flags = kDefaultFlags);
  ~RpcMemAllocator() override;
  void alloc(void **ptr, size_t size, size_t alignment) override;
  void free(void *ptr) override;
  std::string getName() override { return "npu"; }

private:
  using RpcMemAllocFn = void *(*)(int, uint32_t, int);
  using RpcMemFreeFn = void (*)(void *);

  void *library_handle{nullptr};
  RpcMemAllocFn rpcmem_alloc_fn{nullptr};
  RpcMemFreeFn rpcmem_free_fn{nullptr};
  int heap_id;
  uint32_t flags;
#endif
};

/**
 * @brief Try to construct an RpcMemAllocator on the current platform.
 *
 * @return shared_ptr to an allocator on Android+NPU builds, nullptr
 * otherwise. Always declared so callers do not need their own #ifdef.
 */
std::shared_ptr<MemAllocator> tryCreateRpcMemAllocator();

} // namespace nntrainer

#endif

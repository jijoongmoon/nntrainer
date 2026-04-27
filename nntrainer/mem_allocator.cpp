// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    mem_allocator.cpp
 * @date    13 Jan 2025
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is memory allocator for memory pool
 *
 */

#include <cstdlib>
#include <limits>
#include <mem_allocator.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <numeric>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <sysinfoapi.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#if defined(__ANDROID__) && ENABLE_NPU
#include <dynamic_library_loader.h>
#endif

namespace nntrainer {

void MemAllocator::alloc(void **ptr, size_t size, size_t alignment) {
  if (size == 0)
    ml_loge("cannot allocate size = 0");

  *ptr = std::calloc(size, 1);
};

void MemAllocator::free(void *ptr) { std::free(ptr); };

namespace {

/// System page size — used as the default alignment when callers pass 0.
size_t systemPageSize() {
#if defined(_WIN32)
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  return sys_info.dwPageSize;
#else
  return static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
#endif
}

} // namespace

void CpuMemAllocator::alloc(void **ptr, size_t size, size_t alignment) {
  if (size == 0) {
    ml_loge("cannot allocate size = 0");
    *ptr = nullptr;
    return;
  }

  size_t align = alignment != 0 ? alignment : systemPageSize();

#if defined(_WIN32)
  *ptr = _aligned_malloc(size, align);
#else
  // std::aligned_alloc requires size to be a multiple of alignment.
  size_t rounded = ((size + align - 1) / align) * align;
  *ptr = std::aligned_alloc(align, rounded);
#endif
}

void CpuMemAllocator::free(void *ptr) {
  if (ptr == nullptr)
    return;
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

#if defined(__ANDROID__) && ENABLE_NPU

namespace {
constexpr int kDLNow = 0x0001;
constexpr int kDLLocal = 0x0002;
} // namespace

RpcMemAllocator::RpcMemAllocator(int heap_id_in, uint32_t flags_in) :
  heap_id(heap_id_in), flags(flags_in) {
  library_handle =
    DynamicLibraryLoader::loadLibrary("libcdsprpc.so", kDLNow | kDLLocal);
  if (library_handle == nullptr) {
    throw std::runtime_error(
      "[RpcMemAllocator] failed to load libcdsprpc.so: " +
      std::string(DynamicLibraryLoader::getLastError()));
  }

  rpcmem_alloc_fn = reinterpret_cast<RpcMemAllocFn>(
    DynamicLibraryLoader::loadSymbol(library_handle, "rpcmem_alloc"));
  rpcmem_free_fn = reinterpret_cast<RpcMemFreeFn>(
    DynamicLibraryLoader::loadSymbol(library_handle, "rpcmem_free"));

  if (rpcmem_alloc_fn == nullptr || rpcmem_free_fn == nullptr) {
    DynamicLibraryLoader::freeLibrary(library_handle);
    library_handle = nullptr;
    throw std::runtime_error(
      "[RpcMemAllocator] failed to resolve rpcmem_alloc/rpcmem_free");
  }
}

RpcMemAllocator::~RpcMemAllocator() {
  if (library_handle != nullptr) {
    DynamicLibraryLoader::freeLibrary(library_handle);
    library_handle = nullptr;
  }
}

void RpcMemAllocator::alloc(void **ptr, size_t size,
                            size_t /*alignment*/) {
  if (size == 0) {
    ml_loge("cannot allocate size = 0");
    *ptr = nullptr;
    return;
  }
  *ptr = rpcmem_alloc_fn(heap_id, flags, static_cast<int>(size));
}

void RpcMemAllocator::free(void *ptr) {
  if (ptr == nullptr)
    return;
  rpcmem_free_fn(ptr);
}

std::shared_ptr<MemAllocator> tryCreateRpcMemAllocator() {
  try {
    return std::make_shared<RpcMemAllocator>();
  } catch (const std::exception &e) {
    ml_loge("%s", e.what());
    return nullptr;
  }
}

#else

std::shared_ptr<MemAllocator> tryCreateRpcMemAllocator() { return nullptr; }

#endif

} // namespace nntrainer

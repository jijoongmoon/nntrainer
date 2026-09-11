// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cl_buffer_pool.cpp
 * @date   24 August 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Implementation of the device cl_mem plane.
 */

#include "cl_buffer_pool.h"

#include <CL/cl.h>

#include <cstdio>
#include <cstdlib>

#include <cl_context.h>
#include <engine.h>
#include <nntrainer_log.h>
#include <opencl_loader.h>

namespace nntrainer {

namespace {

/** @brief the OpenCL context this process' gpu Context owns */
ClContext *clContext() {
  return static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
}

} // namespace

ClBufferPool::~ClBufferPool() { ClBufferPool::deallocate(); }

void ClBufferPool::recordPlannerLayout() {
  /** Record where the planner put each token. Tokens that share an offset have
   *  disjoint lifetimes, so one buffer sized to the largest of them backs all
   *  of them -- the device-side expression of the planner's reuse. */
  const auto &offsets = getMemoryOffset();
  const auto &sizes = getMemorySize();

  token_offset_.assign(offsets.begin(), offsets.end());
  offset_size_.clear();
  for (size_t i = 0; i < offsets.size(); ++i) {
    const size_t bytes = (i < sizes.size()) ? sizes[i] : 0;
    auto it = offset_size_.find(offsets[i]);
    if (it == offset_size_.end() || bytes > it->second)
      offset_size_[offsets[i]] = bytes;
  }
}

bool ClBufferPool::sharedSliceNeeded(size_t offset) const {
  return shared_slice_skipped_.find(offset) == shared_slice_skipped_.end();
}

void ClBufferPool::noteDeviceOnlyTokens(
  const std::vector<unsigned int> &tokens) {
  std::lock_guard<std::mutex> lk(device_mtx_);
  device_only_tokens_.assign(tokens.begin(), tokens.end());
}

void ClBufferPool::allocate() {
  /** The base MemoryPool::allocate() requests ONE contiguous clSVMAlloc of the
   *  whole plane. When that plane exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE,
   *  clSVMAlloc returns null and ClSVMAllocator falls back to plain host
   *  memory -- which the device cannot use as SVM at all: map/unmap return
   *  CL_INVALID_VALUE and every kernel reads zeros. The fallback is silent by
   *  design (correctness over speed for a host-only run), so on a GPU run it
   *  surfaces only as output collapsing to a single repeated token.
   *
   *  The per-offset cl_mem buffers below already dodge this cap; the SVM plane
   *  has to as well. Above the cap, take the per-offset (shared-objects)
   *  allocateFSU() path so every SVM buffer is a single tensor wide -- far
   *  under the cap -- and stays REAL SVM. A plane under the cap keeps the
   *  single-buffer path and is byte-identical to before.
   *
   *  A process with no registered gpu Context reports no cap and keeps the
   *  single-buffer path, which is the same answer it gave before: on such a
   *  run the SVM allocator is already the host fallback and there is nothing
   *  to protect. */
  cl_ulong plane_cap = 0;
  if (auto *cc = clContext())
    opencl::clGetDeviceInfo(cc->context_inst_.GetDeviceId(),
                            CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(plane_cap),
                            &plane_cap, nullptr);
  const bool over_cap =
    plane_cap > 0 && static_cast<cl_ulong>(size()) > plane_cap;

  /** NNTR_CLMEM_SKIP_SHARED -- what to do about the tensors that live on the
   *  device plane and nowhere else.
   *
   *   1           leave their shared slice unallocated,
   *   2           allocate per offset but skip nothing (the control arm that
   *               isolates the per-offset allocation shape from the skip),
   *   0 (default) the pre-existing single-buffer plane.
   *
   *  The skip is OFF by default because its premise -- that a GPU_CLMEM tensor
   *  is one no host code touches -- is a property of the GRAPH, not of the
   *  pool. A model whose attention lowers an input back to the host
   *  (clmem_lower_cl -> clEnqueueReadBuffer into Tensor::getData()) has no
   *  shared-plane destination for that read-back once the slice is gone: 3/3
   *  runs died with "clmem_lower_cl: buffer read-back failed for
   *  layer0_attention:input0", while =0 gave rc=0 and the byte-identical
   *  reference output. A graph that never takes that path is unaffected either
   *  way and can opt in with =1. Making this safe by default needs either a
   *  narrower qualification rule (exclude the offsets a host read-back path
   *  touches) or a lazy fallback in clmem_lower_cl that reads into a temporary
   *  host buffer when the tensor has no shared plane.
   *
   *  Measured on an Adreno 840 device: the shared plane holds 132.5 MiB that
   *  the host touches 17.9 MiB of, because every offset the planner classified
   *  GPU_CLMEM gets a device cl_mem AND keeps its slice of the shared plane. On
   *  this driver an SVM allocation is a kernel-side mapping, so those bytes are
   *  charged to the GPU whether or not anything reads them.
   *
   *  Skipping needs the per-offset path: a hole cannot be left in one
   *  contiguous allocation. */
  const int skip_mode = [] {
    const char *e = std::getenv("NNTR_CLMEM_SKIP_SHARED");
    return (e != nullptr && e[0] != 0) ? std::atoi(e) : 0;
  }();

  std::lock_guard<std::mutex> lk(device_mtx_);
  recordPlannerLayout();

  if (skip_mode == 1 && !device_only_tokens_.empty()) {
    /** An offset qualifies only if EVERY token the planner placed there is
     *  device-only. Tokens at one offset alias one region; one host-addressable
     *  tensor among them means the region has to exist on the shared plane. */
    std::unordered_set<size_t> device_only_offsets;
    for (unsigned int tok : device_only_tokens_) {
      const size_t i = tok - 1;
      if (i < token_offset_.size())
        device_only_offsets.insert(token_offset_[i]);
    }
    std::unordered_set<size_t> host_offsets;
    {
      std::unordered_set<unsigned int> dev_tok(device_only_tokens_.begin(),
                                               device_only_tokens_.end());
      for (size_t i = 0; i < token_offset_.size(); ++i)
        if (dev_tok.find(static_cast<unsigned int>(i + 1)) == dev_tok.end())
          host_offsets.insert(token_offset_[i]);
    }

    /** Create each candidate's device buffer HERE, before the shared plane is
     *  allocated, and skip the slice only for the ones that succeeded. Doing it
     *  the other way round -- skip first, ask the driver later -- leaves a
     *  tensor with no memory on either plane when clCreateBuffer says no. */
    for (size_t off : device_only_offsets) {
      if (host_offsets.find(off) != host_offsets.end())
        continue;
      if (createDeviceBufferLocked(off) != nullptr)
        shared_slice_skipped_.insert(off);
    }

    /** getMemory() refuses a pool that allocated nothing, and a graph whose
     *  every offset is device-only would produce exactly that. Give the
     *  largest offset its slice back rather than trip that check. */
    if (!shared_slice_skipped_.empty() &&
        shared_slice_skipped_.size() == offset_size_.size()) {
      size_t biggest = *shared_slice_skipped_.begin();
      for (size_t off : shared_slice_skipped_)
        if (offset_size_[off] > offset_size_[biggest])
          biggest = off;
      shared_slice_skipped_.erase(biggest);
    }

    size_t skipped_bytes = 0;
    for (size_t off : shared_slice_skipped_)
      skipped_bytes += offset_size_[off];
    /** stderr, not ml_logi: on Android ml_logi is __android_log_print, so this
     *  line would land in logcat and never in a run's captured output -- and a
     *  saving nobody can see in the run that made it is a claim, not a
     *  measurement. Same reason KVCacheManager prints its kv-share banner
     *  here. */
    std::fprintf(stderr,
                 "[clmempool] %zu of %zu planner offsets are device-only; "
                 "%.1f MB of shared plane not allocated\n",
                 shared_slice_skipped_.size(), offset_size_.size(),
                 skipped_bytes / 1048576.0);
    std::fflush(stderr);
  }

  if (over_cap) {
    ml_logi("ClBufferPool: the %.1f MB SVM plane exceeds the device maximum "
            "allocation %.1f MB; allocating it per offset (shared objects) so "
            "every buffer stays real SVM",
            size() / 1048576.0, plane_cap / 1048576.0);
    MemoryPool::allocateFSU();
  } else if (skip_mode != 0) {
    MemoryPool::allocateFSU();
  } else {
    MemoryPool::allocate();
  }

  /** Re-read the layout: allocateFSU()/allocate() do not change it, but
   *  recordPlannerLayout() before them is what the skip decision needed, and
   *  running it again keeps the post-condition ("the maps describe the plane
   *  that now exists") true from either path. */
  recordPlannerLayout();
}

void *ClBufferPool::createDeviceBufferLocked(size_t offset) {
  auto hit = offset_buffer_.find(offset);
  if (hit != offset_buffer_.end())
    return hit->second;

  auto sit = offset_size_.find(offset);
  if (sit == offset_size_.end() || sit->second == 0)
    return nullptr;
  const size_t bytes = sit->second;

  auto *cc = clContext();
  cl_device_id dev = cc->context_inst_.GetDeviceId();

  /** A single allocation larger than the device can hold is not a device
   *  buffer at all. Report it and leave the tensor on the shared plane, which
   *  is where a buffer this size (a host-dequantized weight table) belongs
   *  anyway -- placing it nowhere would be worse than placing it there. */
  cl_ulong max_alloc = 0;
  opencl::clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc),
                          &max_alloc, nullptr);
  if (max_alloc > 0 && static_cast<cl_ulong>(bytes) > max_alloc) {
    ml_logw("ClBufferPool: %.1f MB exceeds the device maximum allocation "
            "%.1f MB; the tensor stays on the shared plane",
            bytes / 1048576.0, max_alloc / 1048576.0);
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  opencl::ClMemAcctScope _acct("act:device_plane");
  cl_mem buf = opencl::clCreateBufferT(cc->context_inst_.GetContext(),
                                       CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err != CL_SUCCESS || buf == nullptr) {
    ml_logw("ClBufferPool: clCreateBuffer for %zu bytes failed with %d; the "
            "tensor stays on the shared plane",
            bytes, err);
    return nullptr;
  }

  /** Match the shared plane's zero-initialisation: a producer writes only the
   *  rows it has, and an element-wise consumer reads the padded rows too. The
   *  in-order queue orders this fill ahead of every kernel, since allocation
   *  precedes the first forward. */
  const cl_uchar zero = 0;
  if (opencl::clEnqueueFillBuffer(cc->command_queue_inst_.GetCommandQueue(),
                                  buf, &zero, sizeof(zero), 0, bytes, 0,
                                  nullptr, nullptr) != CL_SUCCESS) {
    opencl::clReleaseMemObjectT(buf);
    ml_logw("ClBufferPool: zero-filling a %zu byte device buffer failed; the "
            "tensor stays on the shared plane",
            bytes);
    return nullptr;
  }

  offset_buffer_[offset] = static_cast<void *>(buf);
  return offset_buffer_[offset];
}

void *ClBufferPool::deviceMemory(unsigned int idx) {
  std::lock_guard<std::mutex> lk(device_mtx_);

  const size_t i = idx - 1;
  if (i >= token_offset_.size())
    return nullptr;

  return createDeviceBufferLocked(token_offset_[i]);
}

void ClBufferPool::deallocate() {
  {
    std::lock_guard<std::mutex> lk(device_mtx_);
    for (auto &entry : offset_buffer_)
      if (entry.second != nullptr)
        opencl::clReleaseMemObjectT(static_cast<cl_mem>(entry.second));
    offset_buffer_.clear();
    offset_size_.clear();
    token_offset_.clear();
    shared_slice_skipped_.clear();
    device_only_tokens_.clear();
  }
  MemoryPool::deallocate();
}

} // namespace nntrainer

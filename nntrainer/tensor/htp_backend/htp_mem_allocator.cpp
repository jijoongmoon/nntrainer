// SPDX-License-Identifier: Apache-2.0
#ifdef ENABLE_HEXKL

#include <htp_mem_allocator.h>

#include <nntrainer_log.h>
#include <remote.h>
#include <sdkl.h>
#include <stdexcept>
#include <string>

namespace nntrainer {

void HtpMemAllocator::alloc(void **ptr, size_t size, size_t /*alignment*/) {
  // sdkl_npu_alloc returns NPU-accessible, DMA-aligned memory; the SDK
  // ignores caller-supplied alignment (its own page alignment applies).
  int err = sdkl_npu_alloc(size, ptr);
  if (err != 0 || *ptr == nullptr)
    throw std::runtime_error(
      "HtpMemAllocator: sdkl_npu_alloc failed (size=" + std::to_string(size) +
      ", err=" + std::to_string(err) + ")");
  // T1 verification log: count must match #HTP FC weights (~197 for
  // Qwen3-0.6B).
  ml_logd("htp-npu alloc %zu bytes -> %p", size, *ptr);
}

void HtpMemAllocator::free(void *ptr) {
  if (ptr)
    sdkl_npu_free(ptr);
}

} // namespace nntrainer
#endif // ENABLE_HEXKL

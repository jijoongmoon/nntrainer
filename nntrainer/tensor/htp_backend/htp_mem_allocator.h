// SPDX-License-Identifier: Apache-2.0
#ifndef __HTP_MEM_ALLOCATOR_H__
#define __HTP_MEM_ALLOCATOR_H__
#ifdef ENABLE_HEXKL

#include <mem_allocator.h>

namespace nntrainer {

/**
 * @brief MemAllocator backed by sdkl_npu_alloc so that tensors of
 *        engine="htp" layers live in NPU-accessible memory. Mirrors
 *        QNNRpcManager / ClSVMAllocator.
 */
class HtpMemAllocator : public MemAllocator {
public:
  void alloc(void **ptr, size_t size, size_t alignment) override;
  void free(void *ptr) override;
  std::string getName() override { return "htp-npu"; }
};

} // namespace nntrainer
#endif // ENABLE_HEXKL
#endif // __HTP_MEM_ALLOCATOR_H__

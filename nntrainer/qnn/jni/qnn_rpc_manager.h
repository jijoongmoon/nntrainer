// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    qnn_rpc_manager.h
 * @date    06 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains qnn rpc memory manager
 */
#ifndef __QNN_RPC_MANAGER_H__
#define __QNN_RPC_MANAGER_H__
#include "Log/Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnTypes.h"
#include "Utils/DynamicLoadUtil.hpp"
#include "rpc_mem.h"
#include <cstddef>
#include <dlfcn.h>
#include <map>
#include <mem_allocator.h>
#include <set>
#include <vector>

namespace nntrainer {

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
  const QnnInterface_t ***providerList, uint32_t *numProviders);

/** @brief Manages QNN RPC shared memory allocation via libcdsprpc. */
class QNNRpcManager : public MemAllocator {
public:
  QNNRpcManager();
  ~QNNRpcManager();

  void alloc(void **ptr, size_t size, size_t alignment) override;
  void free(void *ptr) override;

  std::string getName() override { return "qnn"; }

  // rpcmem/ION is CPU-mapped (host-addressable inherits the base default true),
  // but the DSP can only use it after registerQnnTensor() builds a
  // QNN_MEM_TYPE_ION handle -> needsRegister()=true. Not unified memory:
  // device-visible inherits the base default false (pre-register), so isSVM()
  // derives false, matching the old getName()!="gpu-svm".
  bool needsRegister() const override { return true; }

  /**
   * @brief Did THIS allocator produce @p ptr (rpcmem/ION)?
   * @note  The register leg of the host<->rpcmem residency bridge uses this to
   *        tell an already-DSP-shareable buffer (register directly, zero-copy)
   *        from a foreign host buffer (must be staged into rpcmem first). This
   *        is the ownership half of the needsRegister() capability predicate --
   *        no getName()=="qnn" string test. [multi-hw M6 register marker]
   */
  bool owns(const void *ptr) const {
    return qnnMemPtrMap_.count(const_cast<void *>(ptr)) != 0;
  }

  void setQnnInterfaceAndContext(void *context);

  void registerQnnTensor(void *ptr, Qnn_Tensor_t &qnnTensor,
                         Qnn_ContextHandle_t &context);
  void deRegisterQnnTensor();

  bool findMatchingPtr(void *ptr, Qnn_ContextHandle_t &context,
                       Qnn_Tensor_t &qnnTensor);

private:
  QNN_INTERFACE_VER_TYPE qnnInterface_;

  // memHandle set, to check if the ptr is allocted by rpcmem_alloc
  std::set<void *> qnnMemPtrMap_;

  std::map<void *,
           std::pair<Qnn_ContextHandle_t, std::pair<int, Qnn_MemHandle_t>>>
    ptrToFdAndMemHandleMap_;
};

} // namespace nntrainer
#endif

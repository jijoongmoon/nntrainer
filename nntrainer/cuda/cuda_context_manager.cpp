// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_context_manager.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA device/context management implementation.
 */

#include "cuda_context_manager.h"
#include "cuda_common.h"

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace nntrainer::cuda {

void ContextManager::initialize() noexcept { initialized_ok_ = CreateDefaultGPUDevice(); }

bool ContextManager::CreateDefaultGPUDevice() {
  if (!cuCheck(cuInit(0), "cuInit"))
    return false;

  int count = 0;
  if (!cuCheck(cuDeviceGetCount(&count), "cuDeviceGetCount") || count == 0) {
    ml_loge("[CUDA] no CUDA-capable device found");
    return false;
  }

  device_ordinal_ = 0;
  if (const char *e = getenv("NNTR_CUDA_DEVICE"))
    device_ordinal_ = atoi(e);
  if (device_ordinal_ < 0 || device_ordinal_ >= count)
    device_ordinal_ = 0;

  // bind the Runtime API to this device (creates/uses its primary context) ...
  if (!cudaCheck(cudaSetDevice(device_ordinal_), "cudaSetDevice"))
    return false;

  // ... and retain the SAME primary context for the Driver API so module loads
  // and kernel launches share allocations made through the Runtime API.
  if (!cuCheck(cuDeviceGet(&device_, device_ordinal_), "cuDeviceGet"))
    return false;
  if (!cuCheck(cuDevicePrimaryCtxRetain(&context_, device_),
               "cuDevicePrimaryCtxRetain"))
    return false;
  if (!cuCheck(cuCtxSetCurrent(context_), "cuCtxSetCurrent"))
    return false;

  cudaDeviceProp prop{};
  if (cudaCheck(cudaGetDeviceProperties(&prop, device_ordinal_),
                "cudaGetDeviceProperties")) {
    device_name_ = prop.name;
    cc_major_ = prop.major;
    cc_minor_ = prop.minor;
    // Integrated GPU (Tegra/Jetson Orin): host+device share one physical memory
    // pool. prop.integrated is 1 there, 0 on discrete GPUs (RTX4070). This bit
    // gates every "is this discrete VRAM?" residency assumption (device-only
    // activation pool, KV mirror copies, MemAdvise device-pin) so the same
    // binary stays coherent on both -- see isIntegrated().
    integrated_ = prop.integrated != 0;
  }
  cudaDriverGetVersion(&driver_version_);

  ml_logi("[CUDA] device %d: %s (sm_%d%d, driver %d, %.1f GiB, %s)",
          device_ordinal_, device_name_.c_str(), cc_major_, cc_minor_,
          driver_version_, prop.totalGlobalMem / 1073741824.0,
          integrated_ ? "integrated" : "discrete");

  // NNTR_CUDA_DBG: a VISIBLE (stderr, logger-independent) dump of the residency
  // facts the GPU-vs-host dispatch gates depend on. On Tegra/Orin the critical
  // unknown is whether cudaMallocManaged memory reports as cudaMemoryTypeManaged
  // (==2) -- if it instead reports Host(1)/Unregistered(0), every dev()/dev_ok()
  // gate (mha_core RoPE/V-copy, cuda_attention) fails and the GPU ops silently
  // fall to the host => deterministic garbage + low GPU% + slow. This self-probe
  // prints the actual type so that hypothesis is confirmable in one run.
  if (std::getenv("NNTR_CUDA_DBG") != nullptr) {
    int cma = 0, pma = 0;
    cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess,
                           device_ordinal_);
    cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess,
                           device_ordinal_);
    int mtype = -2, dtype = -2;
    void *mp = nullptr, *dp = nullptr;
    if (cudaMallocManaged(&mp, 256) == cudaSuccess && mp) {
      cudaPointerAttributes a{};
      if (cudaPointerGetAttributes(&a, mp) == cudaSuccess)
        mtype = (int)a.type;
      cudaFree(mp);
    }
    if (cudaMalloc(&dp, 256) == cudaSuccess && dp) {
      cudaPointerAttributes a{};
      if (cudaPointerGetAttributes(&a, dp) == cudaSuccess)
        dtype = (int)a.type;
      cudaFree(dp);
    }
    cudaGetLastError();
    std::fprintf(stderr,
                 "[CUDA-DBG] %s sm_%d%d integrated=%d concurrentManagedAccess=%d "
                 "pageableMemoryAccess=%d | cudaPointerGetAttributes.type: "
                 "managed=%d device=%d (expect managed==2 device==2; "
                 "type enum 0=unreg 1=host 2=device... NOTE managed reports as "
                 "type 2/Device OR 3 depending on driver -- gates accept 2&3)\n",
                 device_name_.c_str(), cc_major_, cc_minor_, (int)integrated_,
                 cma, pma, mtype, dtype);
    std::fflush(stderr);
  }
  return true;
}

void ContextManager::EnsureCurrent() {
  if (context_)
    cuCtxSetCurrent(context_);
}

std::string ContextManager::GetDeviceSignature() const {
  return device_name_ + "|drv" + std::to_string(driver_version_) + "|sm_" +
         std::to_string(cc_major_) + std::to_string(cc_minor_);
}

std::string ContextManager::GetComputeArch() const {
  return "compute_" + std::to_string(cc_major_) + std::to_string(cc_minor_);
}

ContextManager::~ContextManager() {
  if (context_) {
    cuDevicePrimaryCtxRelease(device_);
    context_ = nullptr;
  }
}

} // namespace nntrainer::cuda

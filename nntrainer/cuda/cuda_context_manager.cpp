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
  }
  cudaDriverGetVersion(&driver_version_);

  ml_logi("[CUDA] device %d: %s (sm_%d%d, driver %d, %.1f GiB)", device_ordinal_,
          device_name_.c_str(), cc_major_, cc_minor_, driver_version_,
          prop.totalGlobalMem / 1073741824.0);
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

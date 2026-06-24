// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_context_manager.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA wrapper for context/device management. Peer of
 *          nntrainer::opencl::ContextManager. Retains the device PRIMARY context
 *          so the Driver API (cuModuleLoad/cuLaunchKernel) and the Runtime API
 *          (cudaMalloc/cudaMemcpy) share one context.
 */

#ifndef __CUDA_CONTEXT_MANAGER_H__
#define __CUDA_CONTEXT_MANAGER_H__

#include <string>

#include <cuda.h>

#include "singleton.h"

namespace nntrainer::cuda {

/**
 * @class ContextManager
 * @brief Singleton wrapper around the selected CUDA device + primary context.
 */
class ContextManager : public Singleton<ContextManager> {
public:
  /**
   * @brief true if the device + primary context were created successfully
   */
  bool isAvailable() const { return initialized_ok_; }

  /**
   * @brief Get the active primary CUDA context
   */
  CUcontext GetContext() const { return context_; }

  /**
   * @brief Get the active CUDA device handle
   */
  CUdevice GetDevice() const { return device_; }

  /**
   * @brief Get the active device ordinal (cudaSetDevice index)
   */
  int GetDeviceOrdinal() const { return device_ordinal_; }

  /**
   * @brief Get the active device name (e.g. "NVIDIA GeForce RTX 4070 Laptop GPU")
   */
  const std::string &GetDeviceName() const { return device_name_; }

  /**
   * @brief true if the device is an INTEGRATED GPU (Tegra/Jetson Orin etc.)
   *        where host and device share one physical memory pool. On such
   *        devices the discrete-GPU residency tricks (device-only cudaMalloc
   *        activation pool, KV mirror copies, MemAdvise device-pin) give no
   *        bandwidth benefit and BREAK host-coherence -- callers gate those
   *        off when this returns true. Read from cudaDevAttrIntegrated once.
   */
  bool isIntegrated() const { return integrated_; }

  /**
   * @brief Stable signature used to key the on-disk PTX cache so a module built
   *        for a different GPU / driver / arch is never loaded.
   * @return "<name>|drv<driver>|sm_<cc>"
   */
  std::string GetDeviceSignature() const;

  /**
   * @brief NVRTC --gpu-architecture target for this device, e.g. "compute_89".
   */
  std::string GetComputeArch() const;

  /**
   * @brief Make the primary context current on the calling thread. Cheap; safe
   *        to call before any Driver-API op (module load / kernel launch).
   */
  void EnsureCurrent();

  /**
   * @brief Release the primary context.
   */
  ~ContextManager() override;

protected:
  /**
   * @brief Singleton hook: create device + primary context once.
   */
  void initialize() noexcept override;

private:
  bool CreateDefaultGPUDevice();

  CUdevice device_{0};
  int device_ordinal_{0};
  CUcontext context_{nullptr};
  std::string device_name_;
  int cc_major_{0};
  int cc_minor_{0};
  int driver_version_{0};
  bool integrated_{false};
  bool initialized_ok_{false};
};

} // namespace nntrainer::cuda

#endif // __CUDA_CONTEXT_MANAGER_H__

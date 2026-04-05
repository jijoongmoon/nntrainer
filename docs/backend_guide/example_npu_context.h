// SPDX-License-Identifier: Apache-2.0
/**
 * @file   example_npu_context.h
 * @brief  Example: How to add a new compute backend (NPU) to nntrainer.
 *
 * This file demonstrates the minimal steps to create a new hardware backend
 * that integrates with nntrainer's ComputeOps dispatch system. Once
 * registered, all tensor operations (dot, add, multiply, etc.) will
 * automatically dispatch to your backend's implementations.
 *
 * To add a real backend (CUDA, HMX, SLSI, etc.), follow this pattern:
 *
 *   1. Define your ComputeOps table (function pointers)
 *   2. Create a Context class
 *   3. Register it as an Engine plugin (static or .so)
 *   4. Layers using engine="your_backend" will get your ops table
 *
 * See BACKEND_GUIDE.md for the full documentation.
 */

#ifndef __EXAMPLE_NPU_CONTEXT_H__
#define __EXAMPLE_NPU_CONTEXT_H__

#include <context.h>
#include <context_data.h>
#include <singleton.h>

namespace example_npu {

/**
 * @class ExampleNpuContext
 * @brief Demonstrates a new hardware backend for nntrainer.
 *
 * This context:
 * - Provides an NPU-accelerated ops table (ComputeOps)
 * - Registers NPU-specific layers (optional)
 * - Inherits from Context + Singleton for Engine integration
 */
class ExampleNpuContext : public nntrainer::Context,
                          public nntrainer::Singleton<ExampleNpuContext> {
  friend class nntrainer::Singleton<ExampleNpuContext>;

public:
  /**
   * @brief Get the backend name used for engine="example_npu"
   */
  std::string getName() override { return "example_npu"; }

  /**
   * @brief Create a layer object (optional — for NPU-specific layers)
   *
   * If your backend reuses CPU layers (just accelerates tensor ops),
   * you can skip this and let the CPU context handle layer creation.
   */
  PtrType<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &props = {}) override;

private:
  ExampleNpuContext();

  /**
   * @brief Called once by Singleton::Global() via std::call_once
   */
  void initialize() noexcept override;

  /**
   * @brief Register NPU-specific layers (if any)
   */
  void add_default_object();
};

} // namespace example_npu

#endif // __EXAMPLE_NPU_CONTEXT_H__

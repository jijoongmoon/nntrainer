// SPDX-License-Identifier: Apache-2.0
/**
 * @file   example_npu_context.h
 * @brief  Example: How to add a new NPU compute backend to nntrainer.
 *
 * This demonstrates the complete pattern for adding a new hardware backend:
 *
 *   1. Subclass ContextData to hold vendor-specific runtime data
 *   2. Define a ComputeOps table with your accelerated functions
 *   3. Create a Context class (lifecycle, registration)
 *   4. NPU layers access vendor data via as<T>() type-safe cast
 *
 * This example shows BOTH dispatch models:
 *   - Op-level: ComputeOps table (sgemm, ele_add, etc.)
 *   - Graph-level: ContextData subclass (for whole-graph NPU execution)
 */

#ifndef __EXAMPLE_NPU_CONTEXT_H__
#define __EXAMPLE_NPU_CONTEXT_H__

#include <context.h>
#include <context_data.h>
#include <singleton.h>

namespace example_npu {

// =========================================================================
// Step 1: Vendor-specific runtime data (subclass ContextData)
// =========================================================================

/**
 * @brief NPU runtime state — graph handles, sessions, device memory.
 *
 * This holds everything the NPU needs to execute operations.
 * Equivalent to QNNVar in the QNN backend.
 */
struct NpuRuntimeData {
  void *device_handle = nullptr;
  void *session_handle = nullptr;
  // ... graph handles, compiled models, etc.

  bool initialize(const std::string &model_path) {
    // npu_driver_init(&device_handle);
    // npu_create_session(device_handle, &session_handle);
    return true;
  }

  void execute_graph(const float *input, float *output, size_t size) {
    // npu_execute(session_handle, input, output, size);
    // For this example, just copy:
    for (size_t i = 0; i < size; i++)
      output[i] = input[i];
  }
};

/**
 * @brief ContextData subclass for NPU backend.
 *
 * Extends ContextData with NPU-specific runtime data.
 * Layers access this via: context.getContextData()->as<NpuBackendData>()
 *
 * This is the same pattern as QNNBackendVar in the QNN backend.
 */
class NpuBackendData : public nntrainer::ContextData {
public:
  /// Identify this backend type (for debugging and error messages)
  const char *getType() const override { return "example_npu"; }

  /// Access NPU runtime data
  std::shared_ptr<NpuRuntimeData> &getNpuData() { return npu_data; }

private:
  std::shared_ptr<NpuRuntimeData> npu_data =
    std::make_shared<NpuRuntimeData>();
};

// =========================================================================
// Step 2: Context class
// =========================================================================

/**
 * @class ExampleNpuContext
 * @brief NPU backend context — manages lifecycle and registration.
 *
 * Follows the same pattern as AppContext (CPU) and ClContext (GPU):
 *   initialize() → set ops table + vendor data + register layers
 */
class ExampleNpuContext : public nntrainer::Context,
                          public nntrainer::Singleton<ExampleNpuContext> {
  friend class nntrainer::Singleton<ExampleNpuContext>;

public:
  std::string getName() override { return "example_npu"; }

  /// Create NPU-specific layers (optional)
  PtrType<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &props = {}) override;

private:
  /// Constructor: creates NpuBackendData (ContextData subclass)
  ExampleNpuContext()
    : Context(std::make_shared<NpuBackendData>()) {}

  /// Called once by Singleton::Global()
  void initialize() noexcept override;
};

} // namespace example_npu

#endif // __EXAMPLE_NPU_CONTEXT_H__

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_context.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Additive NVIDIA CUDA application context. Peer of ClContext: manages
 *          the per-engine layer factory and the NVRTC module/kernel cache, and
 *          is registered under engine string "cuda". The OpenCL ClContext is
 *          left untouched; selection is at runtime via the layer engine= prop.
 */

#ifndef __CUDA_CONTEXT_H__
#define __CUDA_CONTEXT_H__

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <context.h>
#include <layer.h>
#include <layer_devel.h>
#include <mem_allocator.h>

#include <cuda_buffer_manager.h>
#include <cuda_context_manager.h>
#include <cuda_kernel.h>
#include <cuda_module.h>
#include <cuda_stream_manager.h>

#include "singleton.h"

namespace nntrainer {

extern std::mutex cuda_factory_mutex;

/**
 * @class CudaContext
 * @brief NVIDIA CUDA support for app context (mirror of ClContext).
 */
class CudaContext : public Context, public Singleton<CudaContext> {
public:
  using SharedPtrCudaKernel = std::shared_ptr<cuda::Kernel>;

  /** kernel-name(+options) -> resolved kernel pointer */
  using CudaKernelMap = std::unordered_map<std::string, SharedPtrCudaKernel>;

  // global runtime singletons (device/context, stream, device-buffer pools)
  cuda::ContextManager &context_inst_ = cuda::ContextManager::Global();
  cuda::StreamManager &stream_inst_ = cuda::StreamManager::Global();
  CudaBufferManager &cudabuffInstance = CudaBufferManager::Global();

  /**
   * @brief   Default constructor
   */
  CudaContext() : Context(std::make_shared<ContextData>()) {}

  /**
   * @brief destructor
   */
  ~CudaContext() override = default;

  /**
   * @brief Get the process-wide instance (out-of-line override of
   *        Singleton<T>::Global(), intentionally-leaked heap instance --
   *        never destroyed). ~CudaContext is itself trivial, but its member
   *        singleton references (context_inst_/stream_inst_/cudabuffInstance)
   *        are only initialized once here; leaking keeps the CUDA singleton
   *        family (context, ContextManager, StreamManager, BlasManager) on one
   *        never-destroy convention (field crash fix). ClContext::Global() is
   *        different: a function-local static destroyed at exit.
   */
  static CudaContext &Global();

  /**
   * @brief Factory register function (function-pointer overload)
   */
  template <typename T>
  const int registerFactory(const PtrFactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1) {
    FactoryType<T> f = factory;
    return registerFactory(f, key, int_key);
  }

  /**
   * @brief Factory register function (std::function overload)
   */
  template <typename T>
  const int registerFactory(const FactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1);

  /**
   * @copydoc Context::runDecode
   * @brief CUDA override of the decode/prefill step: capture the step into a
   *        CUDA graph once and replay it, instead of re-issuing every launch.
   *        The decode graph defaults on for a discrete part (initialize()
   *        fills NNTR_CUDA_GRAPH / NNTR_CUDA_M2B) and is only taken for a
   *        model that declares graph-replay feed nodes; the prefill graph
   *        defaults on for an integrated part. With both off this is a plain
   *        eager walk == the base.
   */
  std::vector<std::shared_ptr<const Tensor>>
  runDecode(NeuralNetwork &nn, unsigned int from, unsigned int to,
            const std::vector<std::shared_ptr<const Tensor>> &input,
            const std::vector<std::shared_ptr<const Tensor>> &label) override;

  /**
   * @brief Create an Object from the integer key
   */
  template <typename T>
  PtrType<T> createObject(const int int_key,
                          const PropsType &props = {}) const {
    static_assert(isSupported<T>::value,
                  "given type is not supported for cuda context");
    auto &index = std::get<IndexType<T>>(factory_map);
    auto &int_map = std::get<IntIndexType>(index);

    const auto &entry = int_map.find(int_key);
    if (entry == int_map.end()) {
      std::stringstream ss;
      ss << "Int Key is not found for the object. Key: " << int_key;
      throw exception::not_supported(ss.str().c_str());
    }
    return createObject<T>(entry->second, props);
  }

  /**
   * @brief Create an Object from the string key
   */
  template <typename T>
  PtrType<T> createObject(const std::string &key,
                          const PropsType &props = {}) const {
    auto &index = std::get<IndexType<T>>(factory_map);
    auto &str_map = std::get<StrIndexType<T>>(index);

    std::string lower_key;
    lower_key.resize(key.size());
    std::transform(key.begin(), key.end(), lower_key.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    const auto &entry = str_map.find(lower_key);
    if (entry == str_map.end()) {
      std::stringstream ss;
      ss << "Key is not found for the object. Key: " << lower_key;
      throw exception::not_supported(ss.str().c_str());
    }
    return entry->second(props);
  }

  /**
   * @copydoc Context::registerLayerFactory
   */
  int registerLayerFactory(PtrFactoryType<nntrainer::Layer> factory,
                           const std::string &key = "",
                           const int int_key = -1) override {
    return registerFactory<nntrainer::Layer>(factory, key, int_key);
  }

  /**
   * @brief Create a Layer object from the string key
   */
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(type, properties);
  }

  /**
   * @brief Create a Layer object from the integer key
   */
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(int_key, properties);
  }

  /**
   * @brief Compile (NVRTC) and cache a CUDA kernel by name. Mirrors
   *        ClContext::registerClKernel. The owning module is kept alive in
   *        cuda_module_map for the lifetime of the context.
   * @param kernel_source   full .cu source string
   * @param kernel_name     __global__ function name to resolve
   * @param compile_options extra NVRTC options (arch added automatically)
   * @return shared_ptr<cuda::Kernel> or nullptr on failure
   */
  const SharedPtrCudaKernel
  registerCudaKernel(const std::string &kernel_source,
                     const std::string &kernel_name,
                     const std::string &compile_options = {});

  /**
   * @brief Get the name of the context
   */
  std::string getName() override { return "cuda"; }

  /**
   * @copydoc Context::residencyEngine
   * @brief The CUDA backend's tensors live on the CUDA residency plane, so
   *        this context declares CUDA -- the same override ClContext makes for
   *        GPU. Without it toLayerComputeEngine("cuda") resolves this very
   *        context and reads the host-residency base, so every engine=cuda
   *        layer reports a CPU plane and LayerNode::isComputeEngineCUDA() can
   *        never be true.
   * @return ml::train::LayerComputeEngine::CUDA
   */
  ml::train::LayerComputeEngine residencyEngine() const override {
    return ml::train::LayerComputeEngine::CUDA;
  }

  /**
   * @copydoc Context::caps
   * @brief CUDA override: a snapshot of the device this context selected
   *        (the NNTR_CUDA_DEVICE ordinal), taken on the first call from the
   *        properties cuda::ContextManager read when it opened the device. It
   *        never throws; without a usable device only the fields that need no
   *        device (backend "cuda", image_v8c off) are set.
   */
  const DeviceCaps &caps() const override;

  /**
   * @brief Set the Mem Allocator object
   */
  void setMemAllocator(std::shared_ptr<MemAllocator> mem) {
    getContextData()->setMemAllocator(mem);
  }

private:
  /**
   * @brief Singleton hook: bring up the CUDA runtime + register default layers.
   */
  void initialize() noexcept override;

  void add_default_object();

  /// true once the CUDA device/context/stream were created
  bool cuda_initialized = false;

  // Device capability snapshot behind caps(). Mutable because caps() is a
  // const query whose fill is deferred to first use; the once_flag makes the
  // fill (and its log line) happen once even with concurrent callers.
  mutable DeviceCaps device_caps_;
  mutable std::once_flag caps_probed_;

  FactoryMap<nntrainer::Layer> factory_map;

  template <typename Args, typename T> struct isSupportedHelper;

  /// name(+options) -> kernel
  inline static CudaKernelMap cuda_kernel_map;
  /// source-hash(+options) -> owning module (kept alive for the kernels)
  inline static std::unordered_map<std::string, std::shared_ptr<cuda::Module>>
    cuda_module_map;

  template <typename T, typename... Args>
  struct isSupportedHelper<T, CudaContext::FactoryMap<Args...>> {
    static constexpr bool value =
      (std::is_same_v<std::decay_t<T>, std::decay_t<Args>> || ...);
  };

  template <typename T>
  struct isSupported : isSupportedHelper<T, decltype(factory_map)> {};

  /**
   * @brief Bring up device + primary context + stream once.
   * @return true if the CUDA runtime is usable
   */
  bool cudaInit() {
    if (cuda_initialized)
      return true;
    if (!context_inst_.isAvailable())
      return false;
    // StreamManager::Global() (member init above) already created the stream.
    cudabuffInstance.initBuffers();
    cuda_initialized = true;
    return cuda_initialized;
  }
};

/**
 * @copydoc const int CudaContext::registerFactory
 */
extern template const int CudaContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

/**
 * @brief Model-teardown hook for the decode CUDA-graph cache: drop the cached
 *        graphExec and the cached output tensors (whose shared_ptrs otherwise
 *        pin a destroyed model's buffers until the next prefill boundary).
 *        Pure reset -- call only when no run is in flight (the API's
 *        unload/destroy path). No driver call unless a graph exists.
 */
void cuda_reset_decode_graph_cache();

} // namespace nntrainer

#endif // __CUDA_CONTEXT_H__

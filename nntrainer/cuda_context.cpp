// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_context.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   NVIDIA CUDA application context implementation (mirror of ClContext).
 */

#include <cuda_context.h>

#include <mutex>

#include <addition_layer.h>
#include <compute_ops.h>
#include <cuda_fc_layer.h>
#include <cuda_geglu_layer.h>
#include <cuda_mem_allocator.h>

namespace nntrainer {

std::mutex cuda_factory_mutex;

void CudaContext::initialize() noexcept {
  try {
    if (!cudaInit()) {
      ml_loge("Error: CudaContext::initialize() failed (no usable CUDA device)");
      return;
    }

    // Probe device capabilities once (log-only; the ExecPlan resolver consumes
    // this later, docs/ARCHITECTURE_REFACTOR.md §10 T1). Truth from the existing
    // cuda::ContextManager queries — adds no decision site, so byte-identical.
    caps_.backend = "cuda";
    caps_.device_name = context_inst_.GetDeviceName();
    caps_.arch = context_inst_.GetComputeArch();
    caps_.integrated = context_inst_.isIntegrated();
    caps_.unified_memory = true; // cudaMallocManaged (UVM) is the default pool
    ml_logi("[CudaContext] %s", caps_.toString().c_str());

    add_default_object();

    // Unified-Memory allocator: MemoryPool buffers for engine=cuda tensors are
    // cudaMallocManaged -> host-addressable AND device-accessible (the SVM
    // analogue), so a tensor on this context is device-resident with no
    // separate copy step. Falls back to host memory if UVM is unavailable.
    setMemAllocator(std::make_shared<CudaMemAllocator>());

    // ComputeOps = the CPU ops table. Unlike OpenCL cl_mem (not host-
    // addressable, hence ClComputeOps), engine=cuda tensors are Unified Memory
    // (cudaMallocManaged) = host-coherent, so every standard tensor op
    // (sgemv/sgemm/scopy/dot/elementwise) runs correctly on the managed buffers
    // via the CPU backend. GPU-accelerated ops are layered on top by the CUDA
    // layer classes (CudaFcLayer's cuBLAS path); anything not yet ported simply
    // falls back to a correct CPU computation on the same UVM pointer.
    getContextData()->setComputeOps(get_cpu_ops());

  } catch (std::exception &e) {
    ml_loge("cuda_context: initialization failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("cuda_context: initialization failed due to unknown reason");
  }
}

void CudaContext::add_default_object() {
  // P2: register the CUDA FC layer. More CUDA layers (rmsnorm, swiglu/geglu,
  // attention, ...) are added here as they land, mirroring
  // ClContext::add_default_object().
  registerFactory(nntrainer::createLayer<CudaFcLayer>, CudaFcLayer::type,
                  ml::train::LayerType::LAYER_FC);
  // addition: the core CPU AdditionLayer is pure host Tensor ops -> correct on
  // the host-coherent UVM tensors (do NOT use the OpenCL AdditionLayerCL).
  registerFactory(nntrainer::createLayer<AdditionLayer>, AdditionLayer::type,
                  ml::train::LayerType::LAYER_ADDITION);
  // geglu: no host class exists in the tree (only OpenCL GeGLULayerCl), so
  // provide a host-on-UVM gelu_tanh(gate)*up implementation.
  registerFactory(nntrainer::createLayer<CudaGeGLULayer>, CudaGeGLULayer::type);
}

template <typename T>
const int CudaContext::registerFactory(const FactoryType<T> factory,
                                       const std::string &key,
                                       const int int_key) {
  static_assert(isSupported<T>::value,
                "cuda_context: given type is not supported for current context");

  auto &index = std::get<IndexType<T>>(factory_map);
  auto &str_map = std::get<StrIndexType<T>>(index);
  auto &int_map = std::get<IntIndexType>(index);

  std::string assigned_key = key == "" ? factory({})->getType() : key;

  std::transform(assigned_key.begin(), assigned_key.end(), assigned_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::lock_guard<std::mutex> lock(cuda_factory_mutex);
  if (str_map.find(assigned_key) != str_map.end()) {
    std::stringstream ss;
    ss << "cuda_context: cannot register factory with already taken key: "
       << key;
    throw std::invalid_argument(ss.str().c_str());
  }

  if (int_key != -1 && int_map.find(int_key) != int_map.end()) {
    std::stringstream ss;
    ss << "cuda_context: cannot register factory with already taken int key: "
       << int_key;
    throw std::invalid_argument(ss.str().c_str());
  }

  int assigned_int_key = int_key == -1 ? str_map.size() + 1 : int_key;

  str_map[assigned_key] = factory;
  int_map[assigned_int_key] = assigned_key;

  ml_logd("cuda_context: factory has registered with key: %s, int_key: %d",
          assigned_key.c_str(), assigned_int_key);

  return assigned_int_key;
}

const CudaContext::SharedPtrCudaKernel
CudaContext::registerCudaKernel(const std::string &kernel_source,
                                const std::string &kernel_name,
                                const std::string &compile_options) {
  // hot path: a single key + lookup, no copy of the (multi-KB) source string.
  const std::string kkey = kernel_name + compile_options;
  auto it = cuda_kernel_map.find(kkey);
  if (it != cuda_kernel_map.end())
    return it->second;

  // owning module cache: kernels sharing one (source, options) reuse the
  // compiled+loaded CUmodule (and its on-disk PTX cache, see cuda_module.cpp).
  const std::string mkey =
    std::to_string(cuda::Module::GetKernelHash(kernel_source, compile_options));
  std::shared_ptr<cuda::Module> module;
  auto mit = cuda_module_map.find(mkey);
  if (mit != cuda_module_map.end()) {
    module = mit->second;
  } else {
    module = std::make_shared<cuda::Module>();
    if (!module->CreateModuleFromSource(kernel_source, kernel_name,
                                        compile_options)) {
      ml_loge("Failed to compile CUDA module for kernel %s",
              kernel_name.c_str());
      return nullptr;
    }
    cuda_module_map.emplace(mkey, module);
  }

  SharedPtrCudaKernel kernelPtr = std::make_shared<cuda::Kernel>();
  if (!kernelPtr->CreateKernelFromModule(*module, kernel_name)) {
    ml_loge("Failed to resolve CUDA kernel %s", kernel_name.c_str());
    return nullptr;
  }
  cuda_kernel_map.emplace(kkey, kernelPtr);
  return cuda_kernel_map[kkey];
}

/**
 * @copydoc const int CudaContext::registerFactory
 */
template const int CudaContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

// Non-template seam (Context::registerLayerFactory override): forwards to the
// per-class registerFactory<Layer> here in the same TU so the explicit
// instantiation is used and no template crosses the .so boundary. [T3]
int CudaContext::registerLayerFactory(PtrFactoryType<nntrainer::Layer> factory,
                                      const std::string &key,
                                      const int int_key) {
  return registerFactory<nntrainer::Layer>(factory, key, int_key);
}

} // namespace nntrainer

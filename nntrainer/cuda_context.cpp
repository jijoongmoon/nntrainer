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

#include <env_compat.h>
#include <cuda_context.h>

#include <mutex>

#include <activation_layer.h>
#include <addition_layer.h>
#include <compute_ops.h>
#include <cuda_mem_allocator.h>
#include <cuda_rmsnorm_layer.h>
#include <layer_normalization_layer.h>

#include <cstdlib>

namespace nntrainer {

std::mutex cuda_factory_mutex;

CudaContext &CudaContext::Global() {
  // Out-of-line + intentionally leaked (see header note): matches the
  // never-destroy convention adopted for the whole GPU-context singleton
  // family (ClContext::Global(), cuda::ContextManager/StreamManager/
  // BlasManager::Global()) after the 2026-07-20 shared+cuda exit crash.
  static CudaContext *instance = new CudaContext();
  instance->initializeOnce();
  return *instance;
}

void CudaContext::initialize() noexcept {
  try {
    // [r20 fresh-init tax] On a dual-backend build this runs at the FIRST
    // Engine::Global() touch of ANY run — including engine=cpu/gpu — and
    // cudaInit()'s cuInit wakes a runtime-PM-suspended dGPU over PCIe
    // (measured: nvidia-smi-alone D3cold wake 2.27s on RTX 5060 = the whole
    // "fresh intel init +2.4s" constant; waking the card first drops a fresh
    // intel init from 3451 to 1133 ms). Defer the bring-up when CUDA is not
    // the active engine: explicit NNTR_ENGINE != cuda, or NNTR_ENGINE unset
    // on an OpenCL-enabled build (where the engine default is "gpu",
    // mirroring causallm_engine()). Non-cuda runs never legitimately touch
    // this context (prewarm/StreamManager gate on the engine string).
    // NNTR_CUDA_EAGER_INIT=1 restores the old eager behavior.
    {
      const char *eng = std::getenv("NNTR_ENGINE");
      const char *eager = std::getenv("NNTR_CUDA_EAGER_INIT");
      const bool eager_on = eager && eager[0] == '1';
#if defined(ENABLE_OPENCL)
      const bool cuda_active = eng && std::string(eng) == "cuda";
#else
      const bool cuda_active = !eng || std::string(eng) == "cuda";
#endif
      if (!cuda_active && !eager_on) {
        ml_logi("[CudaContext] bring-up deferred (engine=%s)",
                eng ? eng : "(unset; OpenCL default)");
        return;
      }
    }
    if (!cudaInit()) {
      ml_loge("Error: CudaContext::initialize() failed (no usable CUDA device)");
      return;
    }

    const bool integrated = context_inst_.isIntegrated();
    ml_logi("[CudaContext] device=\"%s\" arch=%s integrated=%d "
            "concurrentManagedAccess=%d",
            context_inst_.GetDeviceName().c_str(),
            context_inst_.GetComputeArch().c_str(), (int)integrated,
            (int)context_inst_.concurrentManagedAccess());

    // Hardware-derived defaults. The device kernels this backend adds are
    // individually switchable, which is useful while bringing a new part up
    // but is a bad deal for a user: nobody should have to export a list of
    // flags to get the backend they asked for. So the context fills in the
    // profile that is right for the device it just probed, with
    // setenv(..., overwrite=0) so an explicit setting from the environment
    // always wins (including "=0", which every consumer treats as off -- see
    // nntr_env_on()).
    if (!integrated && context_inst_.concurrentManagedAccess()) {
      // Discrete-GPU profile: let work queue up instead of draining after
      // every op. This is only legal when the driver reports concurrent
      // managed access -- without it (notably the Windows WDDM model) a host
      // touch of managed memory with kernels in flight is an access violation
      // rather than a race, so an integrated or WDDM device keeps the
      // conservative profile.
      setenv("NNTR_CUDA_ASYNC", "1", 0);
      // The row cap reads "=all" as RAISE, not disable: the device norm kernel
      // synchronizes per call, so on a wide (prefill-shaped) row window the
      // multi-threaded host loop wins and the default caps the device path at
      // 32 rows. On a discrete part the launch is cheap enough that uncapping
      // wins everywhere.
      setenv("NNTR_RMSNORM_CUDA_OFF", "all", 0);
    }

    add_default_object();

    // Unified-Memory allocator: MemoryPool buffers for engine=cuda tensors are
    // cudaMallocManaged -> host-addressable AND device-accessible (the SVM
    // analogue), so a tensor on this context is device-resident with no
    // separate copy step. Falls back to host memory if UVM is unavailable.
    setMemAllocator(std::make_shared<CudaMemAllocator>());

    // ComputeOps = the CUDA op table. CudaComputeOps derives from CpuComputeOps
    // rather than from the abstract base, because engine=cuda tensors are
    // Unified Memory and therefore host-coherent: every op the CUDA table has
    // not overridden yet still computes the right answer by running the CPU
    // implementation over the managed buffer. That is what lets the table be
    // filled in one op at a time instead of having to cover the whole surface
    // before anything can run. A neutral Layer calling
    // in.getOps()->layer_norm(...) lands here with no #ifdef anywhere in
    // nntrainer/layers.
    getContextData()->setComputeOps(get_cuda_ops());

  } catch (std::exception &e) {
    ml_loge("cuda_context: initialization failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("cuda_context: initialization failed due to unknown reason");
  }
}

void CudaContext::add_default_object() {
  // RMS normalization is the one CUDA-specific Layer class here, and it exists
  // for a numerical reason rather than a performance one: the host FP16 path
  // squares the row in FP16, so a residual element of |x| ~ 1700 -- which real
  // transformer blocks do produce -- overflows the sum of squares to +Inf and
  // zeroes the row. This class accumulates in FP32 and hands the row window to
  // a device kernel. It registers under the same type string as the OpenCL
  // RMSNormLayerCl, so a graph moves between backends by changing engine= and
  // nothing else.
  registerFactory(nntrainer::createLayer<CudaRMSNormLayer>,
                  CudaRMSNormLayer::type, ml::train::LayerType::LAYER_RMSNORM);

  // Everything below is a BACKEND-NEUTRAL core class, registered
  // unchanged -- literally the same objects the CPU context registers. They
  // reach the device through the CUDA op table (CudaComputeOps), not through a
  // per-backend Layer fork, which is the entire point of the Tensor-level
  // whole-op surface.
  //
  // addition: host Tensor ops, correct on the host-coherent managed buffers;
  // its residual_op dispatch is where the residual stream can stay in place.
  registerFactory(nntrainer::createLayer<AdditionLayer>, AdditionLayer::type,
                  ml::train::LayerType::LAYER_ADDITION);
  // layer normalization / activation: dispatch to CudaComputeOps::layer_norm and
  // ::activation once those entries exist; until then they run the inherited
  // host implementation over the managed buffer, which is correct.
  registerFactory(nntrainer::createLayer<LayerNormalizationLayer>,
                  LayerNormalizationLayer::type,
                  ml::train::LayerType::LAYER_LAYER_NORMALIZATION);
  registerFactory(nntrainer::createLayer<ActivationLayer>, ActivationLayer::type,
                  ml::train::LayerType::LAYER_ACTIVATION);
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


} // namespace nntrainer

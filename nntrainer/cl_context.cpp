// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    cl_context.h
 * @date    23 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @author  Niket Agarwal <niket.a@samsung.com>
 * @author  Thummala Pallavi <t.pallavi@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains app context related functions and classes that
 * manages the global configuration of the current OpenCL environment. It also
 * creates the OpenCL command queue and context.
 */

#include <addition_layer.h>
#include <attention_kernels.h>
#include <blas_kernel_interface.h>
#include <cl_context.h>
#include <cl_kernels/cl_kernels.h>
#include <cl_svm_allocator.h>
#include <compute_ops.h>
#include <concat_cl.h>
#include <cstdlib>
#include <env_compat.h>
#include <fc_layer_cl.h>
#include <geglu_cl_op.h>
#include <geglu_layer.h>
#include <mutex>
#include <opencl_command_queue_manager.h>
#include <opencl_context_manager.h>
#include <reshape_cl.h>
#include <rmsnorm_layer_cl.h>
#include <scalar_multiply_gpu.h>
#include <sigmoid_add_cl_op.h>
#include <sigmoid_add_layer.h>
#include <sigmoid_glu_cl_op.h>
#include <sigmoid_glu_layer.h>
#include <string>
#include <swiglu_cl_op.h>
#include <swiglu_layer.h>
#include <tie_word_embedding.h>
#include <transpose_cl.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <system_error>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace nntrainer {
#if KERNEL_CACHE
static constexpr bool KERNEL_CACHE_ENABLED = true;
#else
static constexpr bool KERNEL_CACHE_ENABLED = false;
#endif
std::mutex cl_factory_mutex;

std::vector<std::byte> readBinaryFile(const std::string &path) {
  // reading binary
  std::ifstream fs(path, std::ios::binary | std::ios::in);

  if (fs.good()) {
    fs.seekg(0, std::ios::end);
    size_t binary_size = fs.tellg();
    fs.seekg(0, std::ios::beg);

    std::vector<std::byte> data(binary_size);
    fs.read(reinterpret_cast<char *>(data.data()), binary_size);
    return data;
  } else {
    return {};
  }
}

/**
 * @brief Directory the kernel binary cache lives in.
 *
 * Resolution order, first writable candidate wins:
 *   1. NNTR_KERNEL_CACHE_DIR -- explicit override. This is the integration
 *      point for a packaged application, which knows its own private storage
 *      (an Android app would pass its Context.getCacheDir() here) and cannot
 *      rely on the process working directory being writable at all.
 *   2. TMPDIR, when set, as <TMPDIR>/nntrainer_opencl_kernels.
 *   3. The legacy opencl-kernel-path value, which is relative to the working
 *      directory. Kept last for compatibility.
 *
 * The legacy path is still READ even when it is not the resolved directory, so
 * caches written by an earlier version are not orphaned.
 */
static const std::string &kernelCacheDir() {
  static const std::string dir = []() -> std::string {
    if (const char *e = std::getenv("NNTR_KERNEL_CACHE_DIR"); e && *e)
      return std::string(e);
    if (const char *t = std::getenv("TMPDIR"); t && *t)
      return std::string(t) + "/" + opencl::Program::DEFAULT_KERNEL_PATH;
    return opencl::Program::DEFAULT_KERNEL_PATH;
  }();
  return dir;
}

bool writeBinaryFile(const std::string &path,
                     const std::vector<std::byte> &data) {
  // Write to a private temporary and rename into place. A cache entry must
  // never be observable half-written: two builders of the same program -- two
  // threads of the build pool below, or simply two processes started together
  // -- otherwise interleave their writes into one file, and a reader can pick
  // up the torn result. The rename is atomic within the directory, so a reader
  // sees either no entry or a complete one.
  std::error_code ec;
  const std::string tmp_path =
    path + ".tmp" +
    std::to_string(static_cast<unsigned long long>(
      std::hash<std::thread::id>{}(std::this_thread::get_id())));
  {
    std::ofstream fs(tmp_path, std::ios::out | std::ios::binary);
    if (!fs) {
      ml_loge("Failed to open file for writing: %s", tmp_path.c_str());
      return false;
    }
    fs.write(reinterpret_cast<const char *>(data.data()), data.size());
    if (!fs.good()) {
      fs.close();
      std::filesystem::remove(tmp_path, ec);
      ml_loge("Failed to write kernel cache entry: %s", tmp_path.c_str());
      return false;
    }
  }
  std::filesystem::rename(tmp_path, path, ec);
  if (ec) {
    std::filesystem::remove(tmp_path, ec);
    ml_loge("Failed to publish kernel cache entry %s (%s)", path.c_str(),
            ec.message().c_str());
    return false;
  }
  return true;
}

namespace {

/**
 * @brief Thread count for the eager kernel/program build pool.
 *
 * Half the hardware threads by default: the builds are CPU-bound compiler
 * runs inside the OpenCL driver, and the rest of initialization is still
 * making progress alongside them. NNTR_CL_BUILD_THREADS overrides; a value of
 * 1 or less restores the serial path exactly.
 *
 * @param ntasks number of independent build tasks
 * @return number of threads to run them on (never more than ntasks)
 */
unsigned clBuildThreads(size_t ntasks) {
  if (ntasks <= 1)
    return 1;
  long v = -1;
  if (const char *e = std::getenv("NNTR_CL_BUILD_THREADS"); e && *e)
    v = std::atol(e);
  unsigned n;
  if (v >= 1) {
    n = static_cast<unsigned>(v);
  } else {
    const unsigned hw = std::thread::hardware_concurrency();
    n = hw >= 4 ? hw / 2 : 1;
  }
  return static_cast<unsigned>(std::min<size_t>(n, ntasks));
}

/**
 * @brief Run independent kernel-build tasks on a small pool, then join.
 *
 * Legality: clCreateProgramWithSource/clCreateProgramWithBinary/
 * clBuildProgram are thread-safe per the OpenCL specification, and no kernel
 * object is shared between tasks (clSetKernelArg on one kernel object is the
 * unsafe call, and it happens later, on the dispatch path). The state this
 * file itself shares between tasks -- the kernel map and the binary cache
 * directory -- is guarded in registerClKernel() and writeBinaryFile() above.
 *
 * A task that throws is logged and skipped rather than terminating the
 * process: a failed kernel registration is already a recoverable condition on
 * the serial path, where it is caught in initialize().
 *
 * @param tasks build tasks, consumed
 */
void runClBuildTasks(std::vector<std::function<void()>> &&tasks) {
  const unsigned n = clBuildThreads(tasks.size());
  if (n <= 1) {
    for (auto &t : tasks)
      t();
    return;
  }
  std::atomic<size_t> next{0};
  auto worker = [&tasks, &next]() {
    for (size_t i; (i = next.fetch_add(1)) < tasks.size();) {
      try {
        tasks[i]();
      } catch (const std::exception &e) {
        ml_loge("cl_context: kernel build task failed: %s", e.what());
      } catch (...) {
        ml_loge("cl_context: kernel build task failed for an unknown reason");
      }
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(n - 1);
  for (unsigned i = 0; i + 1 < n; ++i)
    pool.emplace_back(worker);
  worker(); // the calling thread takes tasks too
  for (auto &th : pool)
    th.join();
}

} // namespace

void ClContext::initialize() noexcept {
  try {
    if (!clInit()) {
      ml_loge("Error: ClContext::initialize() failed");
      return;
    }

    // Probe device capabilities once (log-only: no decision site reads this
    // yet). Values come from the existing DeviceInfo queries.
    if (const auto *di = context_inst_.getDeviceInfo()) {
      caps_.backend = "gpu";
      caps_.device_name = di->getDeviceName();
      // CL_DEVICE_NAME is stored sized to include the query's trailing NUL; an
      // embedded NUL would truncate the %s log line, so strip trailing NUL/ws.
      while (!caps_.device_name.empty()) {
        const char c = caps_.device_name.back();
        if (c == '\0' || c == ' ' || c == '\n' || c == '\r' || c == '\t')
          caps_.device_name.pop_back();
        else
          break;
      }
      caps_.vendor_id = di->getDeviceVendorId();
      caps_.compute_units = di->getDeviceMaxComputeUnits();
      caps_.max_alloc_bytes = di->getDeviceMaxMemAllocSize();
      caps_.unified_memory = di->getDeviceSVMCapabilities() != 0;
      caps_.svm_fine_grain =
        (di->getDeviceSVMCapabilities() & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) != 0;
      caps_.subgroups = di->getDeviceExtensions().find("cl_intel_subgroups") !=
                        std::string::npos;
      // cl_intel_subgroups is advertised by every Intel GPU since Gen9
      // (including non-DPAS Xe-LPG parts), so it cannot gate a DPAS/XMX
      // matrix-engine kernel. The matrix-multiply-accumulate extension is
      // DPAS-specific, so it is the real capability signal.
      caps_.dpas =
        di->getDeviceExtensions().find(
          "cl_intel_subgroup_matrix_multiply_accumulate") != std::string::npos;
      // image_v8c: whether the device should prefer an image2d-based path over
      // a cl_mem buffer path. No clean device query distinguishes the two
      // (both report CL_DEVICE_IMAGE_SUPPORT); the practical split is that
      // Intel NEO's compiler rejects integer-coordinate read_imageui kernels.
      // Keyed off vendor_id -- a stable, queryable, vendor-wide attribute (the
      // quirk is a compiler trait, not a per-model one), not the brittle
      // device_name. Intel => buffer; others keep the image default.
      caps_.image_v8c = (caps_.vendor_id != DeviceCaps::VENDOR_INTEL);
      cl_bool host_unified = CL_FALSE;
      caps_.integrated =
        (clGetDeviceInfo(context_inst_.GetDeviceId(),
                         CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(host_unified),
                         &host_unified, nullptr) == CL_SUCCESS) &&
        (host_unified == CL_TRUE);
      ml_logi("[ClContext] %s", caps_.toString().c_str());

      // Decide the SVM coherence drain HERE - after device enumeration, where
      // the caps are known - and push it into the queue manager, rather than
      // letting the queue manager resolve it lazily on first use (it can be
      // reached before any device exists, and would then have to guess).
      //
      // An in-order queue on a device without fine-grain SVM does not keep a
      // coarse-grain SVM allocation coherent across a kernel->kernel handoff,
      // so the consuming dispatch needs a host-side drain first. Fine-grain
      // devices are coherent on their own. Scoped to Intel because a
      // coarse-grain Adreno is coherent in practice and would only lose
      // throughput. NNTR_XE3_SYNC still overrides both ways.
      const bool svm_drain =
        (caps_.vendor_id == DeviceCaps::VENDOR_INTEL) && !caps_.svm_fine_grain;
      opencl::CommandQueueManager::Global().setSvmCoherenceDrain(svm_drain);

      // HW-derived env DEFAULT. setenv(..., overwrite=0) means an
      // explicitly-set env ALWAYS wins (and =0 still disables for A/B), so
      // this is a default layer, not a mandate. Vendor_id is a stable
      // vendor-wide attribute (not a brittle device_name match).
      //
      // Apply ONLY when OpenCL is the ACTIVE compute engine: on a
      // multi-backend build another engine's run still initializes this CL
      // context for the kernels it links, but its defaults must not leak into
      // that run. Skip unless NNTR_ENGINE is unset (OpenCL is the default) or
      // explicitly "gpu".
      const char *active_engine = std::getenv("NNTR_ENGINE");
      const bool opencl_is_active =
        (active_engine == nullptr) || std::string(active_engine) == "gpu";
      constexpr uint32_t ADRENO_VENDOR_ID = 0x5143;
      if (opencl_is_active && caps_.vendor_id == DeviceCaps::VENDOR_INTEL) {
        // XMX/DPAS GEMM default. Gated on the DPAS-specific extension
        // (caps_.dpas), NOT the generic cl_intel_subgroups (present on every
        // Intel GPU since Gen9, including matrix-engine-less Xe-LPG parts —
        // defaulting XMX there ropes the device into software-emulated DPAS
        // at a fraction of the dp4a fallback's speed).
        if (caps_.dpas)
          setenv("NNTR_FC_XMX", "1", 0);
        // The NNTR_DETERMINISTIC=1 contract on
        // Windows pins the minimal reproducibility pair (no cl_mem pool
        // offsets + post-FC drain, measured 9/9 identical det256) at the
        // CONSUMER sites — tensor_pool.cpp / cl_svm_allocator.cpp /
        // blas_kernel_interface.cpp / attention_kernels.cpp — because runner
        // and API bundles set NNTR_GPU_CLMEM_POOL=1 explicitly, which an
        // overwrite=0 env layer here cannot beat. No setenv for it here.
        // GPU attention: with no host NEON on these hosts the GPU MHA path
        // wins outright, so default it on.
        setenv("NNTR_MHA_GPU", "1", 0);
        setenv("NNTR_GPU_CLMEM_POOL", "1",
               0); // cl_mem device residency sub-pool
        // Some Intel in-order-queue drivers do not give kernel->kernel
        // coarse-grain SVM coherence for the v8c int8 FC GEMM (its SVM output
        // is read stale by the next kernel), and the global NNTR_XE3_SYNC
        // drain misses it. Drain after the FC GEMM instead
        // (blas_kernel_interface.cpp) -- needed for small-M prefill
        // coherence at ~negligible prefill cost. Override NNTR_XE3_FC_SYNC=0.
        //
        // Windows (WDDM) default-OFF: an extensive battery (cold-boot goldens,
        // token-class A/B, long-context summarize, all with FC_SYNC=0) found
        // no coherence failure attributable to skipping the drain there, and
        // the drain costs ~15-25% decode on that stack. Linux keeps default-ON
        // (the stale read reproduces there). Explicit env wins either way.
#ifdef _WIN32
        setenv("NNTR_XE3_FC_SYNC", "0", 0);
#else
        setenv("NNTR_XE3_FC_SYNC", "1", 0);
#endif
      } else if (opencl_is_active && caps_.vendor_id == ADRENO_VENDOR_ID) {
        setenv("NNTR_MHA_GPU", "1", 0);     // GPU attention
        setenv("NNTR_KV_IMG_ATTN", "1", 0); // image2d KV/attention path
        setenv("NNTR_GPU_CLMEM_POOL", "1",
               0); // cl_mem device residency sub-pool
      }
    }

    if (KERNEL_CACHE_ENABLED) {
      // Non-throwing overload on purpose: the cache directory is relative to
      // the working directory by default (see the opencl-kernel-path option),
      // so it can be uncreatable in a read-only or sandboxed CWD. That must
      // cost the cache, not the context -- the throwing overload would land in
      // the catch below and skip the whole registration block, leaving a
      // half-initialized ClContext.
      std::error_code cache_dir_ec;
      std::filesystem::create_directories(kernelCacheDir(), cache_dir_ec);
      if (cache_dir_ec)
        ml_logw("Kernel cache directory %s unusable (%s); kernels will compile "
                "from source",
                kernelCacheDir().c_str(), cache_dir_ec.message().c_str());
      else
        ml_logi("Kernel binary cache directory: %s", kernelCacheDir().c_str());
    }

    // SVM-backed allocator so MemoryPool buffers are device-visible
    // without an explicit copy. Falls back to host memory inside
    // ClSVMAllocator when the driver lacks SVM support.
    //
    // Installed BEFORE the kernel build / factory registration below
    // (matching AppContext::initialize), and not after: every throw inside
    // those is swallowed by the catch below, so with the old order a single
    // bad registration left this context with a null MemAllocator and crashed
    // the TensorPool ctor for EVERY model. The allocator does not depend on
    // any registration, so ordering it first bounds a registration failure to
    // the registrations.
    setMemAllocator(
      std::make_shared<ClSVMAllocator>(opencl::ContextManager::Global()));

    // Build every eager program from one task list on a small pool. Serially
    // these are the dominant cost of a cold start: measured on an Intel Xe3
    // iGPU with an empty binary cache, the 52 programs registered here took
    // 6.41-6.64 s of clBuildProgram, 72.8% of the time from process start to
    // the first generated token. Nothing about the resulting programs
    // changes -- only how many are built at once -- and the layer factories
    // are still registered serially, in their original order, because the
    // integer keys they are assigned depend on it.
    // NNTR_CL_BUILD_THREADS=1 restores the serial path.
    //
    // The whole block sits AFTER setMemAllocator() for the reason stated
    // there: every throw in here is swallowed by the catch below, and a
    // registration failure must not be able to leave this context with a null
    // MemAllocator.
    std::vector<std::function<void()>> build_tasks;
    DefaultKernelResults helper_results;
    collectAttentionKernelTasks(build_tasks);
    collectDefaultObjectKernelTasks(build_tasks, helper_results);
    collectBlasKernelTasks(build_tasks);
    runClBuildTasks(std::move(build_tasks));
    blas_kernels_initialized = true;
    attention_kernels_initialized = true;
    registerDefaultFactories(helper_results);

    // Install the OpenCL ComputeOps subclass so tensors created from
    // this Context dispatch their accelerator-only ops (Q4_0/INT4
    // batch & accel GEMM/GEMV) to the existing OpenCL kernels in
    // cl_operations/blas_kernels.cpp instead of throwing or silently
    // taking the CPU path. CPU-only ops on a CL-attached tensor still
    // throw via base default — by design, those stay on a CPU context.
    getContextData()->setComputeOps(get_cl_ops());

  } catch (std::exception &e) {
    ml_loge("cl_context: registering layers failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("cl_context: registering layer failed due to unknown reason");
  }
};

void ClContext::collectDefaultObjectKernelTasks(
  std::vector<std::function<void()>> &out, DefaultKernelResults &results) {
  // Every helper registers its own kernel set and touches nothing beyond
  // registerClKernel() and its own translation-unit statics, so the helpers
  // are independent tasks. `results` is filled here and read afterwards by
  // registerDefaultFactories(), so it must outlive the pool join.
  // Concat registers the most kernels of the set, so it goes first: the pool
  // should not end on it.
  //
  // FC and Addition are absent on purpose: both are backend-neutral now and
  // dispatch through the op table (ClComputeOps::fc / ::residual_op), so they
  // have no per-layer kernel set to build and nothing to gate on.
  out.push_back([this, &results]() {
    results.concat = ConcatLayerCl::registerClKernels(*this);
  });
  out.push_back([this, &results]() {
    results.rmsnorm = RMSNormLayerCl::registerClKernels(*this);
  });
  out.push_back(
    [this, &results]() { results.swiglu = registerSwiGLUClKernels(*this); });
  out.push_back(
    [this, &results]() { results.geglu = registerGeGLUClKernels(*this); });
  out.push_back([this, &results]() {
    results.reshape = ReshapeLayerCl::registerClKernels(*this);
  });
  out.push_back([this, &results]() {
    results.transpose = TransposeLayerCl::registerClKernels(*this);
  });
}

void ClContext::registerDefaultFactories(const DefaultKernelResults &results) {
  // The quantized FC layer is backend-neutral now: the GEMM dispatches
  // through the op table (ClComputeOps::fc), so there is no per-layer kernel
  // registration to gate on.
  registerFactory(nntrainer::createLayer<FullyConnectedLayerCl>,
                  FullyConnectedLayerCl::type, ml::train::LayerType::LAYER_FC);

  // The core AdditionLayer is backend-neutral: its per-input copy/add
  // dispatches via ComputeOps::residual_op (the GPU residency body lives in
  // ClComputeOps::residual_op). createLayer("addition", {engine=gpu}) routes
  // here; the former AdditionLayerCL fork is deleted.
  registerFactory(nntrainer::createLayer<AdditionLayer>, AdditionLayer::type,
                  ml::train::LayerType::LAYER_ADDITION);

  if (results.swiglu) {
    // createLayer("swiglu", {engine=gpu}) routes to the backend-neutral
    // SwiGLULayer, which dispatches via ClComputeOps::swiglu. (CPU/CUDA use
    // the same neutral class registered on their contexts.)
    registerFactory(nntrainer::createLayer<SwiGLULayer>, SwiGLULayer::type,
                    ml::train::LayerType::LAYER_SWIGLU);
  }

  if (results.geglu) {
    // No dedicated LayerType enum for GeGLU; register by type string only
    // (int_key auto-assigned). createLayer("geglu", {engine=gpu}) routes to
    // the backend-neutral GeGLULayer, which dispatches via
    // ClComputeOps::geglu.
    registerFactory(nntrainer::createLayer<GeGLULayer>, GeGLULayer::type);
    // scalar_multiply GPU variant: the OpenCL-resident class for the
    // "scalar_multiply" type on the gpu context (the CPU class stays on the
    // cpu/cuda contexts).
    registerFactory(nntrainer::createLayer<ScalarMultiplyLayerGPU>,
                    ScalarMultiplyLayerGPU::type);
    // tie_word_embedding: the lm_head (GPU Q6_K/Q4_0 GEMV on the gpu
    // context, host loop otherwise). Same class on cpu/gpu/cuda.
    registerFactory(nntrainer::createLayer<TieWordEmbedding>,
                    TieWordEmbedding::type);
  }

  if (results.reshape) {
    registerFactory(nntrainer::createLayer<ReshapeLayerCl>,
                    ReshapeLayerCl::type, ml::train::LayerType::LAYER_RESHAPE);
  }

  if (results.rmsnorm) {
    registerFactory(nntrainer::createLayer<RMSNormLayerCl>,
                    RMSNormLayerCl::type, ml::train::LayerType::LAYER_RMSNORM);
  }

  if (results.concat) {
    registerFactory(nntrainer::createLayer<ConcatLayerCl>, ConcatLayerCl::type,
                    ml::train::LayerType::LAYER_CONCAT);
  }

  if (results.transpose) {
    registerFactory(nntrainer::createLayer<TransposeLayerCl>,
                    TransposeLayerCl::type,
                    ml::train::LayerType::LAYER_TRANSPOSE);
  }

  // Fused sigmoid gates: sigmoid_glu (attn output gate = sigmoid(gate)*x) and
  // sigmoid_add (PLE mix = sigmoid(gate)+emb). Registered LAST with EXPLICIT
  // high int_keys: the auto int_key (str_map.size()+1) is fragile -- inserting
  // these mid-list shifted later auto-keys so scalar_multiply's auto-key
  // collided with addition's explicit int_key (7), corrupting int_map and
  // aborting ClContext init BEFORE setMemAllocator (null gpu allocator ->
  // TensorPool ctor crash for every model, gemma4 included). Gated on the CL
  // kernels registering (mirrors GeGLU); backend-neutral layer ->
  // ClComputeOps::sigmoid_glu/sigmoid_add.
  if (registerSigmoidGluClKernels(*this)) {
    registerFactory(nntrainer::createLayer<SigmoidGluLayer>,
                    SigmoidGluLayer::type, /*int_key=*/9001);
  }
  if (registerSigmoidAddClKernels(*this)) {
    registerFactory(nntrainer::createLayer<SigmoidAddLayer>,
                    SigmoidAddLayer::type, /*int_key=*/9002);
  }
}

template <typename T>
const int ClContext::registerFactory(const FactoryType<T> factory,
                                     const std::string &key,
                                     const int int_key) {
  static_assert(isSupported<T>::value,
                "cl_context: given type is not supported for current context");

  auto &index = std::get<IndexType<T>>(factory_map);
  auto &str_map = std::get<StrIndexType<T>>(index);
  auto &int_map = std::get<IntIndexType>(index);

  std::string assigned_key = key == "" ? factory({})->getType() : key;

  std::transform(assigned_key.begin(), assigned_key.end(), assigned_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::lock_guard<std::mutex> lock(cl_factory_mutex);
  if (str_map.find(assigned_key) != str_map.end()) {
    std::stringstream ss;
    ss << "cl_context: cannot register factory with already taken key: " << key;
    throw std::invalid_argument(ss.str().c_str());
  }

  // Auto keys come from max(existing) + 1 and are therefore independent of
  // where in add_default_object() the registration sits; an explicit key that
  // is already bound throws here, naming both colliding types.
  const int assigned_int_key =
    resolveIntKey(int_map, int_key, assigned_key, "cl_context");

  str_map[assigned_key] = factory;
  int_map[assigned_int_key] = assigned_key;

  ml_logd("cl_context: factory has registered with key: %s, int_key: %d",
          assigned_key.c_str(), assigned_int_key);

  return assigned_int_key;
}

void ClContext::collectBlasKernelTasks(
  std::vector<std::function<void()>> &out) {
  // The kernel sources are namespace-scope constants generated into the
  // cl_kernels headers, so a task can capture the source by address and keep
  // the list a plain transcription of the serial one.
  auto add = [this, &out](const std::string &src, const char *name) {
    const std::string *psrc = &src;
    out.push_back([this, psrc, name]() { registerClKernel(*psrc, name); });
  };

  add(sgemv_kernel, "sgemv_cl");
  add(sgemv_no_trans_kernel, "sgemv_cl_noTrans");
  add(dot_kernel, "dot_cl");
  add(sgemm_no_trans_kernel, "sgemm_cl_noTrans");
  add(sgemm_trans_a_kernel, "sgemm_cl_transA");
  add(sgemm_trans_b_kernel, "sgemm_cl_transB");
  add(sgemm_trans_ab_kernel, "sgemm_cl_transAB");
  add(addition_kernel, "addition_cl");
  add(sscal_kernel, "sscal_cl");
  add(q6_k_sgemv_kernel, "kernel_mul_mv_q6_K_f32");

  // register Q4_0 kernels
  add(convert_block_q4_0_kernel, "kernel_convert_block_q4_0_noshuffle");
  add(restore_block_q4_0_kernel, "kernel_restore_block_q4_0");
  add(transpose_16bit_kernel, "kernel_transpose_16");
  add(transpose_32bit_16bit_kernel, "kernel_transpose_32_16");
  add(q4_0_ab_bi_8x4_kernel, "kernel_mul_mat_Ab_Bi_8x4");

  // register INT4 computation kernels
  add(int4_gemv_kernel, "fully_connected_gpu_int4_gemv");
  add(int4_quantize_input_kernel, "quantize_input_int4");
  add(int4_quantize_input_kernel, "quantize_input_int4_pad");

#ifdef ENABLE_FP16
  add(hgemv_kernel, "sgemv_cl_fp16");
  add(hgemv_no_trans_kernel, "sgemv_cl_noTrans_fp16");
  add(dot_fp16_kernel, "dot_cl_fp16");
  add(hgemm_no_trans_kernel, "sgemm_cl_noTrans_fp16");
  add(hgemm_trans_a_kernel, "sgemm_cl_transA_fp16");
  add(hgemm_trans_b_kernel, "sgemm_cl_transB_fp16");
  add(hgemm_trans_ab_kernel, "sgemm_cl_transAB_fp16");
  add(addition_fp16_kernel, "addition_cl_fp16");
  add(hscal_kernel, "sscal_cl_fp16");
#endif
}

void ClContext::initBlasClKernels() {
  if (blas_kernels_initialized) {
    ml_logi(
      "ClContext: Default blas kernels already registered and initialized");
    return;
  }

  std::vector<std::function<void()>> tasks;
  collectBlasKernelTasks(tasks);
  runClBuildTasks(std::move(tasks));
  blas_kernels_initialized = true;
}

void ClContext::collectAttentionKernelTasks(
  std::vector<std::function<void()>> &out) {
  out.push_back(
    [this]() { registerClKernel(rotary_emb_kernel, "rotary_emb_cl"); });

#ifdef ENABLE_FP16
  out.push_back([this]() {
    registerClKernel(rotary_emb_fp16_kernel, "rotary_emb_cl_fp16");
  });

  // Pre-build the prefill-critical PROGRAMS at context init (model load) so
  // their one-time build/binary-load cost does not land inside the first
  // timed prefill. One kernel per program suffices: the program cache in
  // clCreateKernel makes sibling kernels of the same source free. Skipped
  // under NNTR_V8C_BUF (Intel buffer path) where these programs use
  // different compile options -- the hot path builds them on first use as
  // before.
  //
  // These go through the same task list as everything else, so the prewarm is
  // built on the pool rather than serially after it.
  if (std::getenv("NNTR_V8C_BUF") == nullptr) {
    out.push_back([this]() {
      registerClKernel(two_conv_attention_kernel, "softmax_row_f16");
    });
    out.push_back([this]() {
      registerClKernel(int8_int4_gemm_v8c_kernel, "v8c_act_quant_f16_par");
    });
    // rope/scatter program (file-local source in attention_kernels.cpp).
    out.push_back([this]() { attention_prewarm_programs(*this); });
    // Remaining first-use builds profiling shows inside the first prefill as
    // one-time idle outliers: the fp16 norm program and the Q6_K lm_head
    // GEMV program.
    out.push_back([this]() {
      registerClKernel(rmsnorm_fp16_kernel, "rmsnorm_cl_fp16_coop");
    });
    out.push_back([this]() {
      registerClKernel(q6_k_sgemv_kernel, "kernel_mul_mv_q6_K_f32");
    });
    // v8c output-residency program (copy_h2h/add_h2h, file-local source in
    // blas_kernel_interface.cpp) -- the rmsnorm->copy_h2h and gemm->copy_h2h
    // first-call outliers.
    out.push_back([this]() { v8c_prewarm_programs(*this); });
  }
#endif

  // Programs that would otherwise be built on their first dispatch, i.e.
  // inside the first prefill and the first decode step. Collected by the
  // translation unit that owns their sources AND the compile options its
  // dispatch passes, because a prewarm with the wrong options builds a
  // program the hot path never looks up: it pays the compile twice and
  // removes nothing from the critical path.
  //
  // Appended to the SAME task list rather than run serially after it, so the
  // build pool covers them too.
  v8c_collect_lazy_program_tasks(*this, out);
}

void ClContext::initAttentionClKernels() {
  if (attention_kernels_initialized) {
    ml_logi("ClContext: Default attention kernels already registered and "
            "initialized");
    return;
  }

  std::vector<std::function<void()>> tasks;
  collectAttentionKernelTasks(tasks);
  runClBuildTasks(std::move(tasks));
  attention_kernels_initialized = true;
}

const ClContext::SharedPtrClKernel
ClContext::registerClKernel(const std::string &kernel_string,
                            const std::string &kernel_name,
                            const std::string &compile_options) {
  // The eager bring-up registers kernels from a build pool, so the map is
  // guarded. The lock is never held across a program build -- those take
  // hundreds of milliseconds on a cold cache -- so two threads may build the
  // same key concurrently; emplace below keeps the first result and the
  // duplicate is dropped, which is correct because the two objects are
  // interchangeable. Uncontended locking here costs tens of nanoseconds, and
  // only on the registration path.
  //
  // The parameters are const references, NOT by value: the old by-value
  // signature copied the multi-10KB kernel source on every cached lookup,
  // measured at ~12ms per call on Adreno/Android (~36ms of host issue tax per
  // layer in the attention path alone, with the GPU idle for exactly that).
  static std::mutex kernel_map_mutex;
  const std::string key = kernel_name + compile_options;

  // Kernel ring-rotation: hand out K rotating CLONES of each kernel instead
  // of one process-global singleton. Every dispatcher re-binds args on the
  // object this returns; with a singleton that re-bind can hit an object
  // whose previous enqueue the driver has not locked in yet — a measured
  // token-altering hazard on this stack (see the v8c_probe_copy precedent
  // in blas_kernel_interface.cpp; a per-FC flush only reduces its
  // frequency). Rotation guarantees the re-bound object is the one enqueued
  // K calls ago. Cost: K-1 extra clCreateKernel per kernel (program cache
  // makes the clones cheap), zero per-call overhead. Call sites that cache
  // the returned pointer statically keep singleton behavior (documented
  // gap).
  //
  // The ring is a correctness fix, so it is UNCONDITIONAL: it does not sit
  // under the NNTR_DETERMINISTIC opt-out (=0 relaxes only the ordering /
  // math-mode half of the contract). NNTR_CL_KERNEL_RING=K remains the
  // explicit diagnostic override; =1 reproduces the legacy singleton
  // deliberately (bisection aid), which is the one way to re-enable the
  // known-wrong re-bind behavior.
  //
  // The ring map has its own mutex, and like the kernel map below it is
  // never held across clCreateKernel: the first clone of a cold kernel
  // builds its program, which takes hundreds of milliseconds. Two threads
  // racing to create the same clone just leave the ring one entry longer,
  // which is harmless.
  {
    static const int ring_k = []() {
      const char *r = std::getenv("NNTR_CL_KERNEL_RING");
      if (r)
        return std::max(1, std::atoi(r));
      return 8;
    }();
    if (ring_k > 1) {
      static std::mutex ring_mutex;
      static std::unordered_map<
        std::string, std::pair<std::vector<SharedPtrClKernel>, size_t>>
        ring_map;
      {
        const std::lock_guard<std::mutex> lock(ring_mutex);
        auto &slot = ring_map[key];
        if ((int)slot.first.size() >= ring_k) {
          slot.second = (slot.second + 1) % slot.first.size();
          return slot.first[slot.second];
        }
      }
      std::string ks = kernel_string, kn = kernel_name, co = compile_options;
      SharedPtrClKernel kp = std::make_shared<opencl::Kernel>();
      if (clCreateKernel(ks, kn, co, kp)) {
        const std::lock_guard<std::mutex> lock(ring_mutex);
        ring_map[key].first.push_back(kp);
        return ring_map[key].first.back();
      }
      // clone creation failed: fall through to the singleton path below
    }
  }

  // check if created before
  {
    const std::lock_guard<std::mutex> lock(kernel_map_mutex);
    auto it = ocl_kernel_map.find(key);
    if (it != ocl_kernel_map.end())
      return it->second;
  }

  // creating shared_ptr for kernel object (cold path: copies are fine here,
  // clCreateKernel takes mutable refs)
  std::string ks = kernel_string, kn = kernel_name, co = compile_options;
  SharedPtrClKernel kernelPtr = std::make_shared<opencl::Kernel>();
  if (!clCreateKernel(ks, kn, co, kernelPtr)) {
    ml_loge("Failed to register kernel %s", kernel_name.c_str());
    return nullptr;
  }
  // add to map
  const std::lock_guard<std::mutex> lock(kernel_map_mutex);
  return ocl_kernel_map.emplace(key, kernelPtr).first->second;
}

bool ClContext::clCreateKernel(std::string &kernel_string,
                               std::string &kernel_name,
                               std::string &compile_options,
                               const SharedPtrClKernel &kernel_ptr_) {

  ml_logi("Kernel initializing: %s", kernel_name.c_str());

  bool result = false;

  opencl::Program program;

  // In-memory program cache: kernels that share one source+options reuse the
  // built cl_program. Without this every kernel re-did its own binary-file
  // read + clCreateProgramWithBinary (~300ms for the large sources on
  // Adreno 840) -- e.g. 3 kernels of one program paid ~0.9s, all inside the
  // first timed run (mis-read as a per-call issue tax).
  static std::unordered_map<std::string, opencl::Program> program_cache;
  static std::mutex program_cache_mtx;
  const std::string pc_key =
    std::to_string(program.GetKernelHash(kernel_string, "")) + "|" +
    compile_options;
  {
    std::lock_guard<std::mutex> lk(program_cache_mtx);
    auto it = program_cache.find(pc_key);
    if (it != program_cache.end())
      return kernel_ptr_->CreateKernelFromProgram(it->second, kernel_name);
  }

  // On-disk kernel binary cache. The cache key folds in the per-kernel
  // compile_options AND the device signature (name + driver version): a stored
  // binary is only a valid substitute for the exact source it was built from
  // with the exact options it was built with, and only on the same GPU/driver,
  // so kernels that share one source but differ in compile options (e.g. an
  // fp16 variant selected by a -D) must not collide on one cache entry, and a
  // binary from another device or a driver update must never be loaded as-is.
  // clCreateProgramWithBinary still validates and can reject a stale binary,
  // so a load failure falls back to a source compile (and re-caches).
  static const std::string device_sig =
    opencl::ContextManager::Global().GetDeviceSignature();
  const std::string binary_file_name =
    std::to_string(program.GetKernelHash(kernel_string,
                                         compile_options + "|" + device_sig)) +
    ".cl.bin";
  std::string binary_file_path = kernelCacheDir() + "/" + binary_file_name;
  auto binary_data = KERNEL_CACHE_ENABLED ? readBinaryFile(binary_file_path)
                                          : std::vector<std::byte>();
  if (KERNEL_CACHE_ENABLED && binary_data.empty() &&
      kernelCacheDir() != opencl::Program::DEFAULT_KERNEL_PATH) {
    // Fall back to the legacy working-directory location so a cache written
    // before the directory was resolvable is still used (read-only: new
    // entries go to the resolved directory).
    binary_data = readBinaryFile(opencl::Program::DEFAULT_KERNEL_PATH + "/" +
                                 binary_file_name);
  }

  bool loaded_from_binary = false;
  if (KERNEL_CACHE_ENABLED && !binary_data.empty()) {
    ml_logi("Using cached version of kernel: %s at path %s",
            kernel_name.c_str(), binary_file_path.c_str());
    loaded_from_binary = program.CreateCLProgramWithBinary(
      opencl::ContextManager::Global().GetContext(),
      opencl::ContextManager::Global().GetDeviceId(), binary_data,
      binary_file_path, "");
    // A binary is device- and driver-specific: one written by another GPU or
    // before a driver update is rejected here. That is recoverable -- rebuild
    // from source (and re-cache) rather than failing the kernel.
    if (!loaded_from_binary)
      ml_logw("Cached kernel binary %s rejected (stale device/driver?); "
              "recompiling from source",
              binary_file_path.c_str());
  }

  if (loaded_from_binary) {
    result = true;
  } else {
    ml_logi("Binary for kernel %s not found, compiling from source...",
            kernel_name.c_str());
    result =
      program.CreateCLProgram(opencl::ContextManager::Global().GetContext(),
                              opencl::ContextManager::Global().GetDeviceId(),
                              kernel_string, compile_options);

    if (KERNEL_CACHE_ENABLED && result) {
      // Persisting the binary is best effort: the freshly compiled program is
      // already usable, so a read-only or full cache directory must not fail
      // the kernel (and with it every GPU code path that needs it).
      auto binary = program.GetProgramBinary(
        opencl::ContextManager::Global().GetDeviceId());
      if (binary.empty()) {
        ml_logw("Failed retrieving binary for kernel %s; skipping cache write",
                kernel_name.c_str());
      } else if (!writeBinaryFile(binary_file_path, binary)) {
        ml_logw("Failed writing kernel cache %s; continuing",
                binary_file_path.c_str());
      }
    }
  }

  if (!result) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lk(program_cache_mtx);
    program_cache.emplace(pc_key, program);
  }
  result = kernel_ptr_->CreateKernelFromProgram(program, kernel_name);

  return result;
}

/**
 * @copydoc const int ClContext::registerFactory
 */
template const int ClContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

// Non-template seam (Context::registerLayerFactory override): forwards to the
// per-class registerFactory<Layer> here in the same TU so the explicit
// instantiation is used and no template crosses the .so boundary.
int ClContext::registerLayerFactory(PtrFactoryType<nntrainer::Layer> factory,
                                    const std::string &key, const int int_key) {
  return registerFactory<nntrainer::Layer>(factory, key, int_key);
}

} // namespace nntrainer

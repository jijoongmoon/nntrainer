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

#include <addition_layer_cl.h>
#include <cl_context.h>
#include <cl_kernels/cl_kernels.h>
#include <cl_svm_allocator.h>
#include <compute_ops.h>
#include <concat_cl.h>
#include <fc_layer_cl.h>
#include <opencl_context_manager.h>
#include <reshape_cl.h>
#include <rmsnorm_layer_cl.h>
#include <swiglu_cl.h>
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

    // Build every eager program from one task list on a small pool. Serially
    // these are the dominant cost of a cold start: measured on an Intel Xe3
    // iGPU with an empty binary cache, the 52 programs registered here took
    // 6.41-6.64 s of clBuildProgram, 72.8% of the time from process start to
    // the first generated token. Nothing about the resulting programs
    // changes -- only how many are built at once -- and the layer factories
    // below are still registered serially, in their original order, because
    // the integer keys they are assigned depend on it.
    // NNTR_CL_BUILD_THREADS=1 restores the serial path.
    std::vector<std::function<void()>> build_tasks;
    DefaultKernelResults helper_results;
    collectAttentionKernelTasks(build_tasks);
    collectDefaultObjectKernelTasks(build_tasks, helper_results);
    collectBlasKernelTasks(build_tasks);
    runClBuildTasks(std::move(build_tasks));
    blas_kernels_initialized = true;
    attention_kernels_initialized = true;
    registerDefaultFactories(helper_results);
    // SVM-backed allocator so MemoryPool buffers are device-visible
    // without an explicit copy. Falls back to host memory inside
    // ClSVMAllocator when the driver lacks SVM support.
    setMemAllocator(
      std::make_shared<ClSVMAllocator>(opencl::ContextManager::Global()));

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
  out.push_back([this, &results]() {
    results.concat = ConcatLayerCl::registerClKernels(*this);
  });
  out.push_back([this, &results]() {
    results.rmsnorm = RMSNormLayerCl::registerClKernels(*this);
  });
  out.push_back([this, &results]() {
    results.swiglu = SwiGLULayerCl::registerClKernels(*this);
  });
  out.push_back([this, &results]() {
    results.reshape = ReshapeLayerCl::registerClKernels(*this);
  });
  out.push_back([this, &results]() {
    results.fully_connected = FullyConnectedLayerCl::registerClKernels(*this);
  });
  out.push_back([this, &results]() {
    results.addition = AdditionLayerCL::registerClKernels(*this);
  });
  out.push_back([this, &results]() {
    results.transpose = TransposeLayerCl::registerClKernels(*this);
  });
}

void ClContext::registerDefaultFactories(const DefaultKernelResults &results) {
  if (results.fully_connected) {
    registerFactory(nntrainer::createLayer<FullyConnectedLayerCl>,
                    FullyConnectedLayerCl::type,
                    ml::train::LayerType::LAYER_FC);
  }

  if (results.addition) {
    registerFactory(nntrainer::createLayer<AdditionLayerCL>,
                    AdditionLayerCL::type,
                    ml::train::LayerType::LAYER_ADDITION);
  }

  if (results.swiglu) {
    registerFactory(nntrainer::createLayer<SwiGLULayerCl>, SwiGLULayerCl::type,
                    ml::train::LayerType::LAYER_SWIGLU);
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

  if (int_key != -1 && int_map.find(int_key) != int_map.end()) {
    std::stringstream ss;
    ss << "cl_context: cannot register factory with already taken int key: "
       << int_key;
    throw std::invalid_argument(ss.str().c_str());
  }

  int assigned_int_key = int_key == -1 ? str_map.size() + 1 : int_key;

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
#endif
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
ClContext::registerClKernel(std::string kernel_string, std::string kernel_name,
                            std::string compile_options) {
  // The eager bring-up registers kernels from a build pool, so the map is
  // guarded. The lock is never held across a program build -- those take
  // hundreds of milliseconds on a cold cache -- so two threads may build the
  // same key concurrently; emplace below keeps the first result and the
  // duplicate is dropped, which is correct because the two objects are
  // interchangeable. Uncontended locking here costs tens of nanoseconds, and
  // only on the registration path.
  static std::mutex kernel_map_mutex;
  const std::string key = kernel_name + compile_options;

  // check if created before
  {
    const std::lock_guard<std::mutex> lock(kernel_map_mutex);
    auto it = ocl_kernel_map.find(key);
    if (it != ocl_kernel_map.end())
      return it->second;
  }

  // creating shared_ptr for kernel object
  SharedPtrClKernel kernelPtr = std::make_shared<opencl::Kernel>();
  if (!clCreateKernel(kernel_string, kernel_name, compile_options, kernelPtr)) {
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

  // Cache key: the source AND the compile options. A stored binary is only a
  // valid substitute for the exact source it was built from with the exact
  // options it was built with, so kernels that share one source but differ in
  // compile options (e.g. an fp16 variant selected by a -D) must not collide
  // on one cache entry.
  const std::string binary_file_name =
    std::to_string(program.GetKernelHash(kernel_string, compile_options)) +
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

  result = kernel_ptr_->CreateKernelFromProgram(program, kernel_name);

  return result;
}

/**
 * @copydoc const int ClContext::registerFactory
 */
template const int ClContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

} // namespace nntrainer

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    context.h
 * @date    10 Dec 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains app context related functions and classes that
 * manages the global configuration of the current environment.
 */

#ifndef __CONTEXT_H__
#define __CONTEXT_H__

#include <algorithm>
#include <cstdint>
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
#include <context_data.h>
#include <layer.h>
#include <layer_devel.h>
#include <mem_allocator.h>
#include <optimizer.h>
#include <optimizer_devel.h>

#include <nntrainer_log.h>

namespace nntrainer {

// ContextData lives in its own header so that layer_context.h / layer_node.h
// can pull it in without triggering the context.h → layer_devel.h cycle.

/**
 * @struct DeviceCaps
 * @brief Read-only snapshot of device capabilities, probed ONCE per backend at
 *        Context init from real device queries (clGetDeviceInfo /
 *        cudaGetDeviceProperties via the per-backend ContextManagers) rather
 *        than from NNTR_* env flags. Currently LOG-ONLY — no decision site
 *        reads it yet; it is the input the ExecPlan resolver will consume (see
 *        docs/ARCHITECTURE_REFACTOR.md §10 T1/T4). Fields describe attributes
 *        (what the device can do), never identity (who it is); unknown values
 *        stay at the defaults below.
 */
struct DeviceCaps {
  std::string backend = "cpu";  /**< "cpu" / "gpu" (OpenCL) / "cuda" */
  std::string device_name = ""; /**< human-readable device name */
  std::string arch = "";        /**< backend arch tag, e.g. "compute_120" */
  uint32_t vendor_id = 0;       /**< OpenCL CL_DEVICE_VENDOR_ID; 0 = n/a */
  bool integrated = true;       /**< host+device share one physical pool
                                     (host-coherent); CPU = true */
  bool unified_memory = false;  /**< single-pointer SVM/UVM available */
  bool subgroups = false;       /**< OpenCL cl_intel_subgroups (XMX/DPAS) */
  uint32_t compute_units = 0;   /**< OpenCL CL_DEVICE_MAX_COMPUTE_UNITS */
  uint64_t max_alloc_bytes = 0; /**< per-alloc cap (CL MAX_MEM_ALLOC_SIZE);
                                     0 = unknown/unbounded */

  /**
   * @brief One-line human-readable dump for the init-time log.
   */
  std::string toString() const {
    std::ostringstream os;
    os << "DeviceCaps{backend=" << backend << ", device=\"" << device_name
       << "\", arch=" << (arch.empty() ? "-" : arch) << ", vendor_id=0x"
       << std::hex << vendor_id << std::dec << ", integrated=" << integrated
       << ", unified_memory=" << unified_memory << ", subgroups=" << subgroups
       << ", compute_units=" << compute_units
       << ", max_alloc_bytes=" << max_alloc_bytes << "}";
    return os.str();
  }
};

/**
 * @class Context contains user-dependent configuration for  support
 * @brief  support for app context
 */

class Context {
public:
  using PropsType = std::vector<std::string>;

  template <typename T> using PtrType = std::unique_ptr<T>;

  template <typename T>
  using FactoryType = std::function<PtrType<T>(const PropsType &)>;

  template <typename T>
  using PtrFactoryType = PtrType<T> (*)(const PropsType &);

  template <typename T>
  using StrIndexType = std::unordered_map<std::string, FactoryType<T>>;

  /** integer to string key */
  using IntIndexType = std::unordered_map<int, std::string>;

  /**
   * This type contains tuple of
   * 1) integer -> string index
   * 2) string -> factory index
   */
  template <typename T>
  using IndexType = std::tuple<StrIndexType<T>, IntIndexType>;

  template <typename... Ts> using FactoryMap = std::tuple<IndexType<Ts>...>;

  /**
   * @brief   Default constructor
   */
  Context(std::shared_ptr<ContextData> data_ = nullptr) : data(data_) {}

  /**
   * @brief   Destructor
   */
  virtual ~Context() = default;

  /**
   *
   * @brief Initialization of Context.
   *
   * @return status &
   */
  virtual int init() { return 0; };

  /**
   * @brief Create an Layer Object from the type (string)
   *
   * @param type type of layer
   * @param props property
   * @return PtrType<nntrainer::Layer> unique pointer to the object
   */
  virtual PtrType<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &props = {}) {
    ml_logw(
      "[Warning] Implement createLayerObject for the concrete context class to "
      "properly create the layer");
    return nullptr;
  };

  /**
   * @brief Create an Layer Object from the integer key
   *
   * @param int_key integer key
   * @param props property
   * @return PtrType<nntrainer::Layer> unique pointer to the object
   */
  virtual PtrType<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &props = {}) {
    ml_logw(
      "[Warning] Implement createLayerObject for the concrete context class to "
      "properly create the layer");
    return nullptr;
  };

  /**
   * @brief Create an Optimizer Object from the type (string)
   *
   * @param type type of optimizer
   * @param props property
   * @return PtrType<nntrainer::Optimizer> unique pointer to the object
   */
  virtual PtrType<nntrainer::Optimizer>
  createOptimizerObject(const std::string &type,
                        const std::vector<std::string> &props = {}) {
    return nullptr;
  };

  /**
   * @brief Create an Layer Object from the integer key
   *
   * @param int_key integer key
   * @param props property
   * @return PtrType<nntrainer::Optimizer> unique pointer to the object
   */
  virtual PtrType<nntrainer::Optimizer>
  createOptimizerObject(const int int_key,
                        const std::vector<std::string> &properties = {}) {
    return nullptr;
  };

  /**
   * @brief Create an LearningRateScheduler Object from the type (stirng)
   *
   * @param type type of optimizer
   * @param props property
   * @return PtrType<ml::train::LearningRateScheduler> unique pointer to the
   * object
   */
  virtual PtrType<ml::train::LearningRateScheduler>
  createLearningRateSchedulerObject(
    const std::string &type, const std::vector<std::string> &propeties = {}) {
    return nullptr;
  }

  /**
   * @brief Create an LearningRateScheduler Object from the integer key
   *
   * @param int_key integer key
   * @param props property
   * @return PtrType<ml::train::LearningRateScheduler> unique pointer to the
   * object
   */
  virtual std::unique_ptr<ml::train::LearningRateScheduler>
  createLearningRateSchedulerObject(
    const int int_key, const std::vector<std::string> &propeties = {}) {
    return nullptr;
  }

  /**
   * @brief getter of context name
   *
   * @return string name of the context
   */
  virtual std::string getName() = 0;

  std::shared_ptr<ContextData> getContextData() { return data; }

  std::shared_ptr<MemAllocator> getMemAllocator() {
    return getContextData()->getMemAllocator();
  };

  /**
   * @brief load weight and graph for the specific context
   *
   * @return return 0 for success
   */
  virtual int load(const std::string &file_path) { return 0; };

  /**
   * @brief Read-only device capability snapshot for this backend, probed once
   *        at init. The base returns CPU caps (host-coherent, no accelerator);
   *        ClContext / CudaContext override with a probed snapshot. LOG-ONLY for
   *        now (docs/ARCHITECTURE_REFACTOR.md §10 T1) — no decision site reads
   *        it yet, so adding/overriding it is byte-identical.
   *
   * @return const DeviceCaps& capabilities of the device backing this context
   */
  virtual const DeviceCaps &caps() const {
    static const DeviceCaps cpu_caps; // backend="cpu", integrated=true, defaults
    return cpu_caps;
  }

private:
  /**
   * @brief map of context
   */
  static inline std::unordered_map<std::string, Context *> ContextMap;

  std::shared_ptr<ContextData> data = nullptr;
};

using CreateContextFunc = nntrainer::Context *(*)();
using DestroyContextFunc = void (*)(nntrainer::Context *);

/**
 * @brief  Context Pluggable struct that enables pluggable layer
 *
 */
typedef struct {
  CreateContextFunc createfunc;   /**< create layer function */
  DestroyContextFunc destroyfunc; /**< destory function */
} ContextPluggable;

/**
 * @brief pluggable Context must have this structure defined
 */
extern "C" ContextPluggable ml_train_context_pluggable;

} // namespace nntrainer

#endif /* __CONTEXT_H__ */

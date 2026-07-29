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
   * @brief   Next int_key that no registration in @a int_map can already hold.
   *
   * Every Context used to derive an auto-assigned int_key as
   * `str_map.size() + 1`, i.e. from a COUNT. That makes the key a function of
   * how many registrations precede it, so inserting one registration mid-list
   * walks every later auto key up by one until one lands on a key that was
   * requested EXPLICITLY (an ml::train::LayerType enum value). Nothing checked
   * the auto key, so `int_map[k] = type` silently rebound an existing entry,
   * and the damage surfaced later - as a throw from an unrelated registration,
   * or as a lookup returning the wrong layer type. The only defence was a
   * comment asking contributors to append new registrations at the end.
   *
   * Deriving the key from max(existing) + 1 instead makes it independent of
   * insertion position: it cannot collide with anything already registered,
   * in any order. Explicit keys are still checked (see resolveIntKey).
   *
   * @param int_map the context's int -> type-string index
   * @return an int_key not present in @a int_map
   */
  static int nextAutoIntKey(const IntIndexType &int_map) {
    int next = static_cast<int>(int_map.size()) + 1;
    for (const auto &entry : int_map) {
      if (entry.first >= next)
        next = entry.first + 1;
    }
    return next;
  }

  /**
   * @brief   Resolve and validate the int_key a factory registration binds.
   *
   * @note Failing here is deliberate and must stay loud. The collision this
   *       replaces used to abort a Context's initialize() from inside
   *       add_default_object(), which is wrapped in a catch-and-log; the
   *       context then finished initialisation with NO MemAllocator installed
   *       and every model crashed later in TensorPool with a null allocator -
   *       a symptom with no visible connection to the registration that caused
   *       it. Contexts now install their allocator BEFORE registering, so a
   *       registration failure can only ever be this message. Both colliding
   *       type names are named so the message identifies the mistake by
   *       itself.
   *
   * @param int_map      the context's int -> type-string index
   * @param int_key      requested key, or -1 to auto-assign
   * @param assigned_key the (lowercased) type string being registered
   * @param ctx_name     context name, used as the message prefix
   * @return the int_key to bind
   * @throw std::invalid_argument if the requested key is already bound
   */
  static int resolveIntKey(const IntIndexType &int_map, const int int_key,
                           const std::string &assigned_key,
                           const char *ctx_name) {
    const int assigned_int_key =
      int_key == -1 ? nextAutoIntKey(int_map) : int_key;

    auto taken = int_map.find(assigned_int_key);
    if (taken != int_map.end()) {
      std::stringstream ss;
      ss << ctx_name << ": cannot register factory '" << assigned_key
         << "' with int_key " << assigned_int_key
         << ": that key is already registered to '" << taken->second
         << "'. Explicit int_keys must be unique within one context; pass "
            "int_key = -1 to have a free key assigned instead.";
      throw std::invalid_argument(ss.str());
    }

    return assigned_int_key;
  }

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

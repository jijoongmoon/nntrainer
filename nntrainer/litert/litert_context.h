// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   litert_context.h
 * @date   08 Apr 2026
 * @brief  LiteRT-LM context for on-device LLM inference via GPU2 backend
 * @see    https://github.com/nntrainer/nntrainer
 * @author NNTrainer Contributors
 * @bug    No known bugs except for NYI items
 */

#ifndef __LITERT_CONTEXT_H__
#define __LITERT_CONTEXT_H__

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <context.h>
#include <layer.h>
#include <layer_devel.h>
#include <nntrainer_log.h>
#include <singleton.h>

#ifdef ENABLE_LITERT_LM
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#endif

namespace nntrainer {

extern std::mutex litert_factory_mutex;

/**
 * @class LiteRTContext
 * @brief Context for LiteRT-LM backend (gpu2 compute engine)
 *
 * This context manages LiteRT-LM engine lifecycle and provides
 * layer factories for LiteRT-LM graph execution. It follows the
 * same plugin architecture as QNNContext.
 */
class LiteRTContext : public Context, public Singleton<LiteRTContext> {
public:
  /**
   * @brief Default constructor
   */
  LiteRTContext() : Context(std::make_shared<ContextData>()) {}

  /**
   * @brief Destructor - releases LiteRT-LM resources
   */
  ~LiteRTContext() { release(); }

  /**
   * @brief Initialize LiteRT-LM backend
   * @return 0 on success, -1 on failure
   */
  int init() override;

  /**
   * @brief Factory register function for layer types
   *
   * @tparam T object to create (Layer)
   * @param factory factory function
   * @param key string key to access the factory
   * @param int_key integer key to access the factory
   * @return const int unique integer value for the factory
   */
  template <typename T>
  const int registerFactory(const PtrFactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1) {
    FactoryType<T> f = factory;
    return registerFactory(f, key, int_key);
  }

  /**
   * @brief Factory register function for layer types
   */
  template <typename T>
  const int registerFactory(const FactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1);

  /**
   * @brief Create a Layer object from string type
   */
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(type, properties);
  }

  /**
   * @brief Create a Layer object from integer key
   */
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(int_key, properties);
  }

  /**
   * @brief Create an object from integer key
   */
  template <typename T>
  PtrType<T> createObject(const int int_key,
                          const PropsType &props = {}) const {
    static_assert(isSupported<T>::value,
                  "given type is not supported for LiteRT context");
    auto &index = std::get<IndexType<T>>(factory_map);
    auto &int_map = std::get<IntIndexType>(index);

    const auto &entry = int_map.find(int_key);
    if (entry == int_map.end()) {
      std::stringstream ss;
      ss << "Int Key is not found for the object. Key: " << int_key;
      throw std::invalid_argument(ss.str().c_str());
    }
    return createObject<T>(entry->second, props);
  }

  /**
   * @brief Create an object from string key
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
      throw std::invalid_argument(ss.str().c_str());
    }
    return entry->second(props);
  }

  /**
   * @brief Get context name
   * @return "gpu2"
   */
  std::string getName() override { return "gpu2"; }

  /**
   * @brief Load a LiteRT-LM model file
   * @param file_path path to .litertlm model file
   * @return 0 on success
   */
  int load(const std::string &file_path) override;

#ifdef ENABLE_LITERT_LM
  /**
   * @brief Get LiteRT-LM engine instance
   * @return pointer to engine (nullptr if not initialized)
   */
  litert::lm::Engine *getEngine() { return engine_.get(); }

  /**
   * @brief Create a new session for inference
   * @return unique_ptr to session
   */
  std::unique_ptr<litert::lm::Engine::Session> createSession();
#endif

private:
  void initialize() noexcept override;

  void release();

  FactoryMap<nntrainer::Layer> factory_map;

#ifdef ENABLE_LITERT_LM
  std::unique_ptr<litert::lm::Engine> engine_;
  std::string loaded_model_path_;
#endif

  template <typename Args, typename T> struct isSupportedHelper;

  template <typename T, typename... Args>
  struct isSupportedHelper<T, LiteRTContext::FactoryMap<Args...>> {
    static constexpr bool value =
        (std::is_same_v<std::decay_t<T>, std::decay_t<Args>> || ...);
  };

  template <typename T>
  struct isSupported : isSupportedHelper<T, decltype(factory_map)> {};
};

/**
 * @copydoc const int LiteRTContext::registerFactory
 */
extern template const int LiteRTContext::registerFactory<nntrainer::Layer>(
    const FactoryType<nntrainer::Layer> factory, const std::string &key,
    const int int_key);

} // namespace nntrainer

#endif /* __LITERT_CONTEXT_H__ */

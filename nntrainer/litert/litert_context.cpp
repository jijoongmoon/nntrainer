// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   litert_context.cpp
 * @date   08 Apr 2026
 * @brief  LiteRT-LM context implementation for on-device LLM inference
 * @see    https://github.com/nntrainer/nntrainer
 * @author NNTrainer Contributors
 * @bug    No known bugs except for NYI items
 */

#include "litert_context.h"
#include "litert_graph.h"
#include <iostream>
#include <tensor_layer.h>

namespace nntrainer {

std::mutex litert_factory_mutex;

void LiteRTContext::initialize() noexcept {
  try {
    init();
    ml_logi("litert init done");

    // Register LiteRT-LM layers
    registerFactory(nntrainer::createLayer<LiteRTGraph>, LiteRTGraph::type, -1);
    registerFactory(nntrainer::createLayer<TensorLayer>, TensorLayer::type,
                    ml::train::LayerType::LAYER_TENSOR);

    ml_logi("litert registerFactory done");
  } catch (std::exception &e) {
    ml_loge("registering litert layers failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("registering litert layer failed due to unknown reason");
  }
}

int LiteRTContext::init() {
  ml_logi("LiteRTContext::init - initializing LiteRT-LM backend");

  /// @todo Initialize LiteRT-LM runtime here
  /// This will be populated when LiteRT-LM C++ API is integrated:
  /// - Verify LiteRT-LM shared libraries are available
  /// - Initialize GPU accelerator if available
  /// - Set up logging

  return 0;
}

int LiteRTContext::load(const std::string &file_path) {
  ml_logi("LiteRTContext::load - loading model: %s", file_path.c_str());

  /// @todo Load .litertlm model file via LiteRT-LM API:
  /// auto model_assets = ModelAssets::Create(file_path);
  /// auto settings = EngineSettings::CreateDefault(model_assets, Backend::GPU);
  /// engine_ = EngineFactory::CreateAny(settings);

  return 0;
}

void LiteRTContext::release() {
  ml_logi("LiteRTContext::release - cleaning up LiteRT-LM resources");

  /// @todo Release LiteRT-LM engine and session resources
}

template <typename T>
const int LiteRTContext::registerFactory(const FactoryType<T> factory,
                                         const std::string &key,
                                         const int int_key) {
  static_assert(isSupported<T>::value,
                "litert_context: given type is not supported");

  auto &index = std::get<IndexType<T>>(factory_map);
  auto &str_map = std::get<StrIndexType<T>>(index);
  auto &int_map = std::get<IntIndexType>(index);

  std::string assigned_key = key == "" ? factory({})->getType() : key;
  std::transform(assigned_key.begin(), assigned_key.end(), assigned_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::lock_guard<std::mutex> lock(litert_factory_mutex);
  if (str_map.find(assigned_key) != str_map.end()) {
    std::stringstream ss;
    ss << "litert_context: cannot register factory with already taken key: "
       << key;
    throw std::invalid_argument(ss.str().c_str());
  }

  if (int_key != -1 && int_map.find(int_key) != int_map.end()) {
    std::stringstream ss;
    ss << "litert_context: cannot register factory with already taken int key: "
       << int_key;
    throw std::invalid_argument(ss.str().c_str());
  }

  int assigned_int_key = int_key == -1 ? str_map.size() + 1 : int_key;

  str_map[assigned_key] = factory;
  int_map[assigned_int_key] = assigned_key;

  ml_logd("litert_context: factory registered with key: %s, int_key: %d",
          assigned_key.c_str(), assigned_int_key);

  return assigned_int_key;
}

template const int LiteRTContext::registerFactory<nntrainer::Layer>(
    const FactoryType<nntrainer::Layer> factory, const std::string &key,
    const int int_key);

#ifdef PLUGGABLE

nntrainer::Context *create_litert_context() {
  nntrainer::LiteRTContext *litert_context = new nntrainer::LiteRTContext();
  litert_context->Global();
  return litert_context;
}

void destroy_litert_context(nntrainer::Context *ct) { delete ct; }

extern "C" {
nntrainer::ContextPluggable ml_train_context_pluggable{
    create_litert_context, destroy_litert_context};
}

#endif

} // namespace nntrainer

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   causal_lm.h
 * @date   22 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  CausalLM Factory to support registration and creation of various
 * CausalLM models
 */

#ifndef __CAUSALLM_FACTORY_H__
#define __CAUSALLM_FACTORY_H__

#include <ostream>
#include <transformer.h>
#include <unordered_map>
#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "QuickAI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(fmt, ...) fprintf(stdout, fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif


namespace causallm {

/**
 * @brief Factory class
 */
class Factory {
public:
  using Creator =
    std::function<std::unique_ptr<Transformer>(json &, json &, json &)>;

  static Factory &Instance() {
    static Factory factory;
    return factory;
  }

  void registerModel(const std::string &key, Creator creator) {
    LOGD("[DEBUG] key in Model Factory %s, %p", key.c_str(),(void*)&creator);
    creators[key] = creator;
  }

  std::unique_ptr<Transformer> create(const std::string &key, json &cfg,
                                      json &generation_cfg,
                                      json &nntr_cfg) const {
    auto it = creators.find(key);
    if (it != creators.end()) {
      LOGD("[DEBUG] key in Model Factory %s found : %p", key.c_str(), (void*)&(it->second));
      return (it->second)(cfg, generation_cfg, nntr_cfg);
    }
    LOGD("[DEBUG] key in Model Factory %s Not found", key.c_str());              
    return nullptr;
  }

  void printRegistered(std::ostream &os) const {
    for (const auto &pair : creators) {
      os << "\n\t" << pair.first;
    }
  }

private:
  std::unordered_map<std::string, Creator> creators;
};

} // namespace causallm

#endif

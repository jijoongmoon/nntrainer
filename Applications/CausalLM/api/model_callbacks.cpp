// SPDX-License-Identifier: Apache-2.0
#include "model_callbacks.h"

ModelCallbackRegistry &ModelCallbackRegistry::instance() {
  static ModelCallbackRegistry reg;
  return reg;
}

void ModelCallbackRegistry::register_for(const std::string &architecture,
                                         ModelCallbacks cb) {
  by_arch_[architecture] = std::move(cb);
}

const ModelCallbacks *
ModelCallbackRegistry::lookup(const std::string &architecture) const {
  auto it = by_arch_.find(architecture);
  return it != by_arch_.end() ? &it->second : nullptr;
}

bool ModelCallbackRegistry::any_requires_htp() const {
  for (const auto &[arch, cb] : by_arch_) {
    if (cb.requires_htp)
      return true;
  }
  return false;
}

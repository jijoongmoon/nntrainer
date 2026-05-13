//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <stdexcept>

#include "BackendExtensions.hpp"
#include "dlwrap.hpp"
#include "PAL/DynamicLoading.hpp"
#include "qualla/detail/Log.hpp"

BackendExtensions::BackendExtensions(BackendExtensionsConfigs backendExtensionsConfig,
                                     void* backendLibHandle,
                                     bool debug_qnn,
                                     QnnLog_Callback_t registeredLogCallback,
                                     QnnLog_Level_t qnnLogLevel)
    : m_backendInterface(nullptr), m_destroyBackendInterfaceFn(nullptr) {
  QNN_DEBUG("DEBUG: backendExtensionsConfig.sharedLibraryPath=%s\n",
            backendExtensionsConfig.sharedLibraryPath.c_str());
  if (backendExtensionsConfig.sharedLibraryPath.empty()) {
    throw std::runtime_error("Empty backend extensions library path.");
  }

  QNN_DEBUG("DEBUG: backendExtensionsConfig.configFilePath=%s\n",
            backendExtensionsConfig.configFilePath.c_str());
  if (backendExtensionsConfig.configFilePath.empty()) {
    throw std::runtime_error("Empty backend extensions config path.");
  }

  void* libHandle =
      pal::dynamicloading::dlOpen(backendExtensionsConfig.sharedLibraryPath.c_str(),
                                  pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_LOCAL);
  if (nullptr == libHandle) {
    const char* msg = pal::dynamicloading::dlError();
    QNN_ERROR("Unable to load backend extensions lib: [%s]. dlerror(): [%s]",
              backendExtensionsConfig.sharedLibraryPath.c_str(),
              msg ? msg : "Unknown error");
    throw std::runtime_error("Unable to open backend extension library.");
  }

  auto createBackendInterfaceFn =
      reinterpret_cast<qnn::tools::netrun::CreateBackendInterfaceFnType_t>(
          pal::dynamicloading::dlSym(libHandle, "createBackendInterface"));
  if (nullptr == createBackendInterfaceFn) {
    throw std::runtime_error("Unable to resolve createBackendInterface.");
  }

  m_destroyBackendInterfaceFn =
      reinterpret_cast<qnn::tools::netrun::DestroyBackendInterfaceFnType_t>(
          pal::dynamicloading::dlSym(libHandle, "destroyBackendInterface"));
  if (nullptr == m_destroyBackendInterfaceFn) {
    throw std::runtime_error("Unable to resolve destroyBackendInterface.");
  }

  m_backendInterface = createBackendInterfaceFn();
  if (nullptr == m_backendInterface) {
    throw std::runtime_error("Unable to load backend extensions interface.");
  }

  if (debug_qnn) {
    if (!(m_backendInterface->setupLogging(registeredLogCallback, qnnLogLevel))) {
      throw std::runtime_error("Unable to initialize logging in backend extensions.");
    }
  }

  if (!m_backendInterface->initialize(backendLibHandle)) {
    throw std::runtime_error("Unable to initialize backend extensions interface.");
  }

  if (!m_backendInterface->loadConfig(backendExtensionsConfig.configFilePath)) {
    throw std::runtime_error("Unable to load backend extensions config.");
  }
}

BackendExtensions::~BackendExtensions() { m_destroyBackendInterfaceFn(m_backendInterface); }

qnn::tools::netrun::IBackend* BackendExtensions::interface() { return m_backendInterface; }

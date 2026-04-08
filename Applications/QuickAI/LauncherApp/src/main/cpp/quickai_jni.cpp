// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quickai_jni.cpp
 * @brief   JNI shim that forwards calls from Kotlin's NativeCausalLm to
 *          the handle-based C entry points declared in
 *          Applications/CausalLM/api/causal_lm_api.h.
 *
 * This file contains no business logic — only JNI marshalling:
 *   jstring   <-> const char*
 *   jlong     <-> CausalLmHandle
 *   ErrorCode +  struct PerformanceMetrics -> Kotlin data classes.
 *
 * Higher-level concerns (per-model threading, FIFO queue, Gemma4
 * routing) live in Kotlin (ModelWorker / ModelRegistry).
 */

#include <android/log.h>
#include <cstddef>
#include <jni.h>
#include <string>

#include "causal_lm_api.h"

#define LOG_TAG "quickai_jni"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// ---------------------------------------------------------------------------
// Cached global references to the Kotlin result data-classes so we can
// construct them back in JNI. Looked up once in JNI_OnLoad.
// ---------------------------------------------------------------------------

struct JniCache {
  jclass loadResultCls = nullptr;   // NativeCausalLm$LoadResult
  jmethodID loadResultCtor = nullptr;

  jclass runResultCls = nullptr;    // NativeCausalLm$RunResult
  jmethodID runResultCtor = nullptr;

  jclass metricsResultCls = nullptr; // NativeCausalLm$MetricsResult
  jmethodID metricsResultCtor = nullptr;
};

JniCache g_cache;

jclass find_global(JNIEnv *env, const char *name) {
  jclass local = env->FindClass(name);
  if (local == nullptr) {
    LOGE("FindClass failed: %s", name);
    return nullptr;
  }
  auto *global = reinterpret_cast<jclass>(env->NewGlobalRef(local));
  env->DeleteLocalRef(local);
  return global;
}

} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void * /*reserved*/) {
  JNIEnv *env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK ||
      env == nullptr) {
    return JNI_ERR;
  }

  g_cache.loadResultCls =
    find_global(env, "com/example/QuickAI/service/NativeCausalLm$LoadResult");
  if (g_cache.loadResultCls != nullptr) {
    g_cache.loadResultCtor =
      env->GetMethodID(g_cache.loadResultCls, "<init>", "(IJ)V");
  }

  g_cache.runResultCls =
    find_global(env, "com/example/QuickAI/service/NativeCausalLm$RunResult");
  if (g_cache.runResultCls != nullptr) {
    g_cache.runResultCtor = env->GetMethodID(
      g_cache.runResultCls, "<init>", "(ILjava/lang/String;)V");
  }

  g_cache.metricsResultCls = find_global(
    env, "com/example/QuickAI/service/NativeCausalLm$MetricsResult");
  if (g_cache.metricsResultCls != nullptr) {
    g_cache.metricsResultCtor = env->GetMethodID(
      g_cache.metricsResultCls, "<init>", "(IIDIDDDJ)V");
  }

  return JNI_VERSION_1_6;
}

// ---------------------------------------------------------------------------
// setOptions
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_QuickAI_service_NativeCausalLm_setOptionsNative(
  JNIEnv * /*env*/, jobject /*thiz*/, jboolean useChatTemplate,
  jboolean debugMode, jboolean verbose) {
  Config cfg;
  cfg.use_chat_template = (useChatTemplate == JNI_TRUE);
  cfg.debug_mode = (debugMode == JNI_TRUE);
  cfg.verbose = (verbose == JNI_TRUE);
  return static_cast<jint>(setOptions(cfg));
}

// ---------------------------------------------------------------------------
// loadModelHandle
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jobject JNICALL
Java_com_example_QuickAI_service_NativeCausalLm_loadModelHandleNative(
  JNIEnv *env, jobject /*thiz*/, jint backendOrdinal, jint modelOrdinal,
  jint quantOrdinal) {
  CausalLmHandle handle = nullptr;
  ErrorCode ec =
    loadModelHandle(static_cast<BackendType>(backendOrdinal),
                    static_cast<ModelType>(modelOrdinal),
                    static_cast<ModelQuantizationType>(quantOrdinal), &handle);

  if (g_cache.loadResultCls == nullptr || g_cache.loadResultCtor == nullptr) {
    return nullptr;
  }
  return env->NewObject(g_cache.loadResultCls, g_cache.loadResultCtor,
                        static_cast<jint>(ec),
                        reinterpret_cast<jlong>(handle));
}

// ---------------------------------------------------------------------------
// runModelHandle
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jobject JNICALL
Java_com_example_QuickAI_service_NativeCausalLm_runModelHandleNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong, jstring promptJ) {
  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);

  const char *prompt = env->GetStringUTFChars(promptJ, nullptr);
  if (prompt == nullptr) {
    return env->NewObject(g_cache.runResultCls, g_cache.runResultCtor,
                          static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER),
                          nullptr);
  }

  const char *output = nullptr;
  ErrorCode ec = runModelHandle(handle, prompt, &output);

  env->ReleaseStringUTFChars(promptJ, prompt);

  jstring outJ = nullptr;
  if (ec == CAUSAL_LM_ERROR_NONE && output != nullptr) {
    outJ = env->NewStringUTF(output);
  }

  return env->NewObject(g_cache.runResultCls, g_cache.runResultCtor,
                        static_cast<jint>(ec), outJ);
}

// ---------------------------------------------------------------------------
// getPerformanceMetricsHandle
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jobject JNICALL
Java_com_example_QuickAI_service_NativeCausalLm_getPerformanceMetricsHandleNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong) {
  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);

  PerformanceMetrics m{};
  ErrorCode ec = getPerformanceMetricsHandle(handle, &m);

  if (g_cache.metricsResultCls == nullptr ||
      g_cache.metricsResultCtor == nullptr) {
    return nullptr;
  }
  return env->NewObject(
    g_cache.metricsResultCls, g_cache.metricsResultCtor, static_cast<jint>(ec),
    static_cast<jint>(m.prefill_tokens), static_cast<jdouble>(m.prefill_duration_ms),
    static_cast<jint>(m.generation_tokens),
    static_cast<jdouble>(m.generation_duration_ms),
    static_cast<jdouble>(m.total_duration_ms),
    static_cast<jdouble>(m.initialization_duration_ms),
    static_cast<jlong>(m.peak_memory_kb));
}

// ---------------------------------------------------------------------------
// destroyModelHandle
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_QuickAI_service_NativeCausalLm_destroyModelHandleNative(
  JNIEnv * /*env*/, jobject /*thiz*/, jlong handleJlong) {
  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  return static_cast<jint>(destroyModelHandle(handle));
}

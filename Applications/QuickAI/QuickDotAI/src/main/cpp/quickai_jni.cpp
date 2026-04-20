// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quickai_jni.cpp
 * @brief   JNI shim that forwards calls from Kotlin's
 *          com.example.quickdotai.NativeCausalLm object to the
 *          handle-based C entry points declared in
 *          Applications/CausalLM/api/quick_dot_ai_api.h.
 *
 * This file contains no business logic — only JNI marshalling:
 *   jstring   <-> const char*
 *   jlong     <-> CausalLmHandle
 *   ErrorCode +  struct PerformanceMetrics -> Kotlin data classes.
 *
 * Higher-level concerns (per-model threading, FIFO queue, Gemma4
 * routing) live one level up in NativeQuickDotAI.kt and in the host
 * app's worker (for QuickAIService that is ModelWorker / ModelRegistry;
 * SampleTestAPP drives NativeQuickDotAI directly from its own
 * background dispatcher).
 */

#include <android/log.h>
#include <cerrno>
#include <cstddef>
#include <jni.h>
#include <string>
#include <unistd.h>

#include "quick_dot_ai_api.h"

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
    find_global(env, "com/example/quickdotai/NativeCausalLm$LoadResult");
  if (g_cache.loadResultCls != nullptr) {
    g_cache.loadResultCtor =
      env->GetMethodID(g_cache.loadResultCls, "<init>", "(IJ)V");
  }

  g_cache.runResultCls =
    find_global(env, "com/example/quickdotai/NativeCausalLm$RunResult");
  if (g_cache.runResultCls != nullptr) {
    g_cache.runResultCtor = env->GetMethodID(
      g_cache.runResultCls, "<init>", "(ILjava/lang/String;)V");
  }

  g_cache.metricsResultCls = find_global(
    env, "com/example/quickdotai/NativeCausalLm$MetricsResult");
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
Java_com_example_quickdotai_NativeCausalLm_setOptionsNative(
  JNIEnv * /*env*/, jobject /*thiz*/, jboolean useChatTemplate,
  jboolean debugMode, jboolean verbose) {
  Config cfg;
  cfg.use_chat_template = (useChatTemplate == JNI_TRUE);
  cfg.debug_mode = (debugMode == JNI_TRUE);
  cfg.verbose = (verbose == JNI_TRUE);
  return static_cast<jint>(setOptions(cfg));
}

// ---------------------------------------------------------------------------
// chdir
//
// The C API in quick_dot_ai_api.cpp hardcodes the model discovery prefix to
// the relative path "./models/<model>-<quant>" (see resolve_model_path()),
// which ties model lookup to the process's current working directory.
// Android apps start with cwd="/" so the only way to point the loader at
// an app-owned directory is to chdir(2) the process before calling
// loadModelHandle. This helper exposes that chdir to Kotlin as a thin
// wrapper — returning 0 on success or the POSIX errno on failure, which
// NativeQuickDotAI surfaces as CAUSAL_LM_ERROR_MODEL_LOAD_FAILED.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_chdirNative(
  JNIEnv *env, jobject /*thiz*/, jstring pathJ) {
  if (pathJ == nullptr) {
    return EINVAL;
  }
  const char *path = env->GetStringUTFChars(pathJ, nullptr);
  if (path == nullptr) {
    return ENOMEM;
  }
  int rc = chdir(path);
  int err = (rc == 0) ? 0 : errno;
  env->ReleaseStringUTFChars(pathJ, path);
  return static_cast<jint>(err);
}

// ---------------------------------------------------------------------------
// loadModelHandle
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jobject JNICALL
Java_com_example_quickdotai_NativeCausalLm_loadModelHandleNative(
  JNIEnv *env, jobject /*thiz*/, jint backendOrdinal, jint modelOrdinal,
  jint quantOrdinal, jstring nativeLibDirJ) {
  const char *native_lib_dir = nullptr;
  if (nativeLibDirJ != nullptr) {
    native_lib_dir = env->GetStringUTFChars(nativeLibDirJ, nullptr);
  }

  CausalLmHandle handle = nullptr;
  ErrorCode ec =
    loadModelHandle(static_cast<BackendType>(backendOrdinal),
                    static_cast<ModelType>(modelOrdinal),
                    static_cast<ModelQuantizationType>(quantOrdinal),
                    native_lib_dir, &handle);

  if (native_lib_dir != nullptr && nativeLibDirJ != nullptr) {
    env->ReleaseStringUTFChars(nativeLibDirJ, native_lib_dir);
  }

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
Java_com_example_quickdotai_NativeCausalLm_runModelHandleNative(
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
Java_com_example_quickdotai_NativeCausalLm_getPerformanceMetricsHandleNative(
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
// unloadModelHandle
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_unloadModelHandleNative(
  JNIEnv * /*env*/, jobject /*thiz*/, jlong handleJlong) {
  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  return static_cast<jint>(unloadModelHandle(handle));
}

// ---------------------------------------------------------------------------
// destroyModelHandle
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_destroyModelHandleNative(
  JNIEnv * /*env*/, jobject /*thiz*/, jlong handleJlong) {
  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  return static_cast<jint>(destroyModelHandle(handle));
}

// ---------------------------------------------------------------------------
// runModelHandleStreaming
//
// Forwards deltas from the native quick_dot_ai_api streaming callback to a
// Kotlin NativeStreamListener.onDelta(String). See AsyncAndStreaming.md §4
// at the repo root for the design rationale — in particular, the
// callback fires on the SAME thread that invoked this JNI entry point
// (the ModelWorker thread), which means we do NOT need AttachCurrentThread:
// the JNIEnv* captured here is still valid throughout every callback.
// ---------------------------------------------------------------------------
namespace {
struct StreamCtx {
  JNIEnv *env;
  jobject listener;   // local ref owned by the JNI entry frame
  jmethodID onDelta;  // Ljava/lang/String;)V
};

int stream_trampoline(const char *delta, void *user_data) {
  auto *ctx = static_cast<StreamCtx *>(user_data);
  if (ctx == nullptr || ctx->env == nullptr || ctx->listener == nullptr ||
      ctx->onDelta == nullptr) {
    return 1; // cancel
  }
  jstring js = ctx->env->NewStringUTF(delta != nullptr ? delta : "");
  if (js == nullptr) {
    // OOM or pending exception; clear and ask the native runner to stop.
    if (ctx->env->ExceptionCheck()) {
      ctx->env->ExceptionClear();
    }
    return 1;
  }
  ctx->env->CallVoidMethod(ctx->listener, ctx->onDelta, js);
  ctx->env->DeleteLocalRef(js);
  if (ctx->env->ExceptionCheck()) {
    // Surface Kotlin-side errors as cancellation; the Kotlin override
    // in NativeQuickDotAI.runStreaming will catch the exception on the
    // JNI call's return and report it through StreamSink.onError.
    ctx->env->ExceptionDescribe();
    ctx->env->ExceptionClear();
    return 1;
  }
  return 0;
}
} // namespace

extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_runModelHandleStreamingNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong, jstring promptJ,
  jobject listenerObj) {
  if (promptJ == nullptr || listenerObj == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Resolve the onDelta(String)V method id per-call. We can't cache
  // this globally because NativeStreamListener is a `fun interface`
  // and the concrete class of `listenerObj` varies call-to-call.
  jclass listenerCls = env->GetObjectClass(listenerObj);
  if (listenerCls == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }
  jmethodID onDelta =
    env->GetMethodID(listenerCls, "onDelta", "(Ljava/lang/String;)V");
  env->DeleteLocalRef(listenerCls);
  if (onDelta == nullptr) {
    if (env->ExceptionCheck()) {
      env->ExceptionClear();
    }
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  const char *prompt = env->GetStringUTFChars(promptJ, nullptr);
  if (prompt == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  StreamCtx ctx{env, listenerObj, onDelta};
  ErrorCode ec =
    runModelHandleStreaming(handle, prompt, &stream_trampoline, &ctx);

  env->ReleaseStringUTFChars(promptJ, prompt);
  return static_cast<jint>(ec);
}

// ---------------------------------------------------------------------------
// runMultimodalHandleStreaming
//
// Multimodal streaming inference that accepts preprocessed image patches
// (as FloatArray) and a text prompt. The pixel values are converted from
// jfloatArray to native float* and passed to the C API.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_quickdotai_NativeCausalLm_runMultimodalHandleStreamingNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong, jstring promptJ,
  jfloatArray pixelValuesJ, jint numPatches, jint originalHeight, jint originalWidth,
  jobject listenerObj) {
  if (promptJ == nullptr || pixelValuesJ == nullptr || listenerObj == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Resolve the onDelta(String)V method id
  jclass listenerCls = env->GetObjectClass(listenerObj);
  if (listenerCls == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }
  jmethodID onDelta =
    env->GetMethodID(listenerCls, "onDelta", "(Ljava/lang/String;)V");
  env->DeleteLocalRef(listenerCls);
  if (onDelta == nullptr) {
    if (env->ExceptionCheck()) {
      env->ExceptionClear();
    }
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  const char *prompt = env->GetStringUTFChars(promptJ, nullptr);
  if (prompt == nullptr) {
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  // Get float* from FloatArray
  float *pixels = env->GetFloatArrayElements(pixelValuesJ, nullptr);
  if (pixels == nullptr) {
    env->ReleaseStringUTFChars(promptJ, prompt);
    return static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER);
  }

  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  (void)handle;
  
  // TODO: runMultimodalHandleStreaming not yet implemented in libquick_dot_ai_api.so
  // StreamCtx ctx{env, listenerObj, onDelta};
  // ErrorCode ec = runMultimodalHandleStreaming(
  //   handle, prompt, pixels, numPatches, originalHeight, originalWidth,
  //   &stream_trampoline, &ctx);

  // Release resources
  env->ReleaseFloatArrayElements(pixelValuesJ, pixels, JNI_ABORT);
  env->ReleaseStringUTFChars(promptJ, prompt);
  
  return static_cast<jint>(CAUSAL_LM_ERROR_UNSUPPORTED);
}

// ---------------------------------------------------------------------------
// runMultimodalHandle
//
// Blocking multimodal inference that returns the complete output.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jobject JNICALL
Java_com_example_quickdotai_NativeCausalLm_runMultimodalHandleNative(
  JNIEnv *env, jobject /*thiz*/, jlong handleJlong, jstring promptJ,
  jfloatArray pixelValuesJ, jint numPatches, jint originalHeight, jint originalWidth) {
  if (promptJ == nullptr || pixelValuesJ == nullptr) {
    return env->NewObject(g_cache.runResultCls, g_cache.runResultCtor,
                          static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER),
                          nullptr);
  }

  const char *prompt = env->GetStringUTFChars(promptJ, nullptr);
  if (prompt == nullptr) {
    return env->NewObject(g_cache.runResultCls, g_cache.runResultCtor,
                          static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER),
                          nullptr);
  }

  // Get float* from FloatArray
  float *pixels = env->GetFloatArrayElements(pixelValuesJ, nullptr);
  if (pixels == nullptr) {
    env->ReleaseStringUTFChars(promptJ, prompt);
    return env->NewObject(g_cache.runResultCls, g_cache.runResultCtor,
                          static_cast<jint>(CAUSAL_LM_ERROR_INVALID_PARAMETER),
                          nullptr);
  }

  auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);
  (void)handle;
  const char *output = nullptr;
  (void)output;
  
  // TODO: runMultimodalHandle not yet implemented in libquick_dot_ai_api.so
  // ErrorCode ec = runMultimodalHandle(
  //   handle, prompt, pixels, numPatches, originalHeight, originalWidth, &output);

  // Release resources
  env->ReleaseFloatArrayElements(pixelValuesJ, pixels, JNI_ABORT);
  env->ReleaseStringUTFChars(promptJ, prompt);

  return env->NewObject(g_cache.runResultCls, g_cache.runResultCtor,
                        static_cast<jint>(CAUSAL_LM_ERROR_UNSUPPORTED), nullptr);
}

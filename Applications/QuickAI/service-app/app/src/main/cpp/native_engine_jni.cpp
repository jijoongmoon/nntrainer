// SPDX-License-Identifier: Apache-2.0
/**
 * JNI bridge between NativeEngine.kt and libcausallm_api.so
 */

#include <jni.h>
#include <string>
#include "causal_lm_api.h"

extern "C" {

JNIEXPORT jint JNICALL
Java_com_quickai_service_NativeEngine_nativeSetOptions(
    JNIEnv *env, jobject, jboolean useChatTemplate, jboolean debugMode,
    jboolean verbose) {
  Config config;
  config.use_chat_template = useChatTemplate;
  config.debug_mode = debugMode;
  config.verbose = verbose;
  return (jint)setOptions(config);
}

JNIEXPORT jint JNICALL
Java_com_quickai_service_NativeEngine_nativeLoadModel(JNIEnv *env, jobject,
                                                       jint backend,
                                                       jint modelType,
                                                       jint quantType) {
  return (jint)loadModel((BackendType)backend, (ModelType)modelType,
                         (ModelQuantizationType)quantType);
}

JNIEXPORT jstring JNICALL
Java_com_quickai_service_NativeEngine_nativeRunModel(JNIEnv *env, jobject,
                                                      jstring prompt) {
  const char *promptStr = env->GetStringUTFChars(prompt, nullptr);
  const char *outputText = nullptr;

  ErrorCode err = runModel(promptStr, &outputText);
  env->ReleaseStringUTFChars(prompt, promptStr);

  if (err != CAUSAL_LM_ERROR_NONE || outputText == nullptr) {
    return nullptr;
  }

  return env->NewStringUTF(outputText);
}

JNIEXPORT jint JNICALL
Java_com_quickai_service_NativeEngine_nativeUnloadModel(JNIEnv *env, jobject) {
  return (jint)unloadModel();
}

JNIEXPORT jint JNICALL
Java_com_quickai_service_NativeEngine_nativeSetModelBasePath(JNIEnv *env,
                                                              jobject,
                                                              jstring basePath) {
  const char *path = env->GetStringUTFChars(basePath, nullptr);
  ErrorCode err = setModelBasePath(path);
  env->ReleaseStringUTFChars(basePath, path);
  return (jint)err;
}

JNIEXPORT jint JNICALL
Java_com_quickai_service_NativeEngine_nativeGetLoadedBackend(JNIEnv *env,
                                                              jobject) {
  return (jint)getLoadedBackend();
}

JNIEXPORT jdoubleArray JNICALL
Java_com_quickai_service_NativeEngine_nativeGetPerformanceMetrics(JNIEnv *env,
                                                                   jobject) {
  PerformanceMetrics metrics;
  ErrorCode err = getPerformanceMetrics(&metrics);

  if (err != CAUSAL_LM_ERROR_NONE) {
    return nullptr;
  }

  jdoubleArray result = env->NewDoubleArray(7);
  double values[7] = {
      (double)metrics.prefill_tokens,
      metrics.prefill_duration_ms,
      (double)metrics.generation_tokens,
      metrics.generation_duration_ms,
      metrics.total_duration_ms,
      metrics.initialization_duration_ms,
      (double)metrics.peak_memory_kb};
  env->SetDoubleArrayRegion(result, 0, 7, values);
  return result;
}

} // extern "C"

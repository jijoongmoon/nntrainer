// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    NativeCausalLm.kt
 * @brief   JNI bindings for libcausallm_api.so (handle-based API only).
 *
 * All methods here are 1:1 with the handle-based entry points added to
 * causal_lm_api.h. Higher-level serialization/FIFO/registry logic lives in
 * ModelRegistry / ModelWorker / NativeCausalLmBackend — this file is only
 * the JNI glue.
 */
package com.example.QuickAI.service

/**
 * @brief Low-level JNI bridge to libcausallm_api.so.
 *
 * Loaded libraries (must be present in jniLibs/<abi>/ of the hosting APK):
 *   - libquickai_jni.so     (this JNI shim; produced by src/main/cpp)
 *   - libcausallm_api.so    (the C API lib built from Applications/CausalLM)
 *   - libcausallm_core.so   (transitive)
 *   - libnntrainer.so       (transitive)
 *   - libccapi-nntrainer.so (transitive)
 *
 * Any non-zero `errorCode` value corresponds to `ErrorCode` in
 * causal_lm_api.h — see `QuickAiError.fromNativeCode` for the Kotlin
 * mapping.
 */
object NativeCausalLm {

    @Volatile
    private var loaded: Boolean = false

    /**
     * @brief Must be called once before any other method. Swallows
     * UnsatisfiedLinkError so the service can still come up (and return
     * MODEL_LOAD_FAILED to clients) when the native lib is missing, e.g.
     * during emulator development without the prebuilt .so files.
     */
    @Synchronized
    fun ensureLoaded(): Boolean {
        if (loaded) return true
        return try {
            // quickai_jni dlopens libcausallm_api.so as part of its JNI_OnLoad
            System.loadLibrary("quickai_jni")
            loaded = true
            true
        } catch (t: UnsatisfiedLinkError) {
            android.util.Log.e(TAG, "Failed to load libquickai_jni.so: ${t.message}")
            false
        }
    }

    /**
     * @brief Result of a loadModel call. [handle] is an opaque pointer
     * (packed in a long) that must be passed back to [runModelHandle],
     * [getPerformanceMetricsHandle] and [destroyModelHandle].
     */
    data class LoadResult(val errorCode: Int, val handle: Long)

    /**
     * @brief Result of a runModel call.
     */
    data class RunResult(val errorCode: Int, val output: String?)

    /**
     * @brief Result of a metrics call.
     */
    data class MetricsResult(
        val errorCode: Int,
        val prefillTokens: Int,
        val prefillDurationMs: Double,
        val generationTokens: Int,
        val generationDurationMs: Double,
        val totalDurationMs: Double,
        val initializationDurationMs: Double,
        val peakMemoryKb: Long
    )

    /** Forwards to `setOptions` in causal_lm_api.h. */
    external fun setOptionsNative(
        useChatTemplate: Boolean,
        debugMode: Boolean,
        verbose: Boolean
    ): Int

    /** Forwards to `loadModelHandle` in causal_lm_api.h. */
    external fun loadModelHandleNative(
        backendOrdinal: Int,
        modelOrdinal: Int,
        quantOrdinal: Int
    ): LoadResult

    /** Forwards to `runModelHandle` in causal_lm_api.h. */
    external fun runModelHandleNative(
        handle: Long,
        prompt: String
    ): RunResult

    /** Forwards to `getPerformanceMetricsHandle` in causal_lm_api.h. */
    external fun getPerformanceMetricsHandleNative(handle: Long): MetricsResult

    /** Forwards to `destroyModelHandle` in causal_lm_api.h. */
    external fun destroyModelHandleNative(handle: Long): Int

    private const val TAG = "NativeCausalLm"
}

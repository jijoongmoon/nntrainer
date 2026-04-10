// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    NativeCausalLm.kt
 * @brief   JNI bindings for libcausallm_api.so (handle-based API only).
 *
 * All methods here are 1:1 with the handle-based entry points added to
 * quick_dot_ai_api.h. Higher-level lifecycle (serialization, registry,
 * threading) lives in [NativeQuickDotAI] and in the host app — this
 * file is only the JNI glue.
 */
package com.example.quickdotai

/**
 * @brief Low-level JNI bridge to libcausallm_api.so.
 *
 * Loaded libraries (all bundled into the QuickDotAI AAR under
 * jniLibs/arm64-v8a/):
 *   - libquickai_jni.so     (JNI shim produced by src/main/cpp)
 *   - libcausallm_api.so    (the C API lib built from Applications/CausalLM)
 *   - libcausallm_core.so   (transitive)
 *   - libnntrainer.so       (transitive)
 *   - libccapi-nntrainer.so (transitive)
 *
 * Any non-zero `errorCode` value corresponds to `ErrorCode` in
 * quick_dot_ai_api.h — see [QuickAiError.fromNativeCode] for the Kotlin
 * mapping.
 *
 * @hide
 *
 * Implementation detail: this object is `public` rather than `internal`
 * because Kotlin's `internal`-visibility name mangling (`$modulename`
 * suffix) would interfere with JNI symbol resolution — the JNI entry
 * points in quickai_jni.cpp use the unmangled `Java_com_example_quickdotai_
 * NativeCausalLm_<method>` names. Treat it as implementation detail and
 * always go through [NativeQuickDotAI].
 */
object NativeCausalLm {

    @Volatile
    private var loaded: Boolean = false

    /**
     * @brief Must be called once before any other method. Swallows
     * UnsatisfiedLinkError so callers can still return a clean
     * MODEL_LOAD_FAILED error to their own clients when the native lib
     * is missing (e.g. during emulator development without the
     * prebuilt .so files).
     */
    @Synchronized
    fun ensureLoaded(): Boolean {
        if (loaded) return true
        return try {
            // quickai_jni dlopens libcausallm_api.so as part of its JNI_OnLoad.
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
     * (packed in a long) that must be passed back to [runModelHandleNative],
     * [getPerformanceMetricsHandleNative] and [destroyModelHandleNative].
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

    /** Forwards to `setOptions` in quick_dot_ai_api.h. */
    external fun setOptionsNative(
        useChatTemplate: Boolean,
        debugMode: Boolean,
        verbose: Boolean
    ): Int

    /**
     * @brief Thin wrapper around POSIX `chdir(2)`.
     *
     * The native C API in quick_dot_ai_api.cpp builds its model paths as
     * `./models/<name>-<quant>` (see `resolve_model_path`), so the
     * loader's behaviour depends on the process's current working
     * directory. Android apps launch with cwd="/" which is not writable,
     * so the host code must chdir the process to an app-owned directory
     * (typically `Context.getExternalFilesDir(null)`) before calling
     * [loadModelHandleNative]. [NativeQuickDotAI] does this
     * automatically when the caller supplies a [LoadModelRequest.modelPath].
     *
     * @return 0 on success, or the POSIX errno value on failure.
     */
    external fun chdirNative(path: String): Int

    /** Forwards to `loadModelHandle` in quick_dot_ai_api.h. */
    external fun loadModelHandleNative(
        backendOrdinal: Int,
        modelOrdinal: Int,
        quantOrdinal: Int
    ): LoadResult

    /** Forwards to `runModelHandle` in quick_dot_ai_api.h. */
    external fun runModelHandleNative(
        handle: Long,
        prompt: String
    ): RunResult

    /**
     * @brief Listener invoked by the JNI trampoline once per decoded
     * delta during [runModelHandleStreamingNative].
     *
     * The method is called **on the same thread that invoked
     * runModelHandleStreamingNative** — the JNI bridge does NOT attach
     * any new thread to the JVM — so implementations must be
     * non-blocking (deltas arrive back-to-back at decode speed).
     */
    fun interface NativeStreamListener {
        fun onDelta(text: String)
    }

    /**
     * @brief Forwards to `runModelHandleStreaming` in quick_dot_ai_api.h.
     *
     * Blocking: returns only when generation finishes, EOS is emitted,
     * NUM_TO_GENERATE is reached, the listener throws, or an error
     * occurs. [listener] is invoked synchronously from the same thread
     * for every decoded delta; if it throws, the JNI bridge catches
     * the exception, asks the native runner to cancel at the next
     * token boundary, and propagates a non-zero ErrorCode back here.
     * Terminal events (onDone / onError) are synthesized on the Kotlin
     * side from the return value — see [NativeQuickDotAI.runStreaming].
     *
     * @return An `ErrorCode` int; 0 on clean completion.
     */
    external fun runModelHandleStreamingNative(
        handle: Long,
        prompt: String,
        listener: NativeStreamListener
    ): Int

    /** Forwards to `getPerformanceMetricsHandle` in quick_dot_ai_api.h. */
    external fun getPerformanceMetricsHandleNative(handle: Long): MetricsResult

    /** Forwards to `destroyModelHandle` in quick_dot_ai_api.h. */
    external fun destroyModelHandleNative(handle: Long): Int

    private const val TAG = "NativeCausalLm"
}

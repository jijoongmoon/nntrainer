// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    NativeQuickDotAI.kt
 * @brief   QuickDotAI implementation backed by the handle-based
 *          quick_dot_ai_api.h (routed through libquickai_jni.so → JNI →
 *          libcausallm_api.so).
 */
package com.example.quickdotai

import android.util.Log
import java.io.File

/**
 * @brief Kotlin wrapper around a single `CausalLmHandle` in native code.
 *
 * Non-thread-safe by design — the host app must drive a single instance
 * from a single worker thread.
 */
class NativeQuickDotAI : QuickDotAI {

    override val kind: String = "native"

    override var architecture: String? = null
        private set

    private var handle: Long = 0L
    private var loaded: Boolean = false

    override fun load(req: LoadModelRequest): BackendResult<Unit> {
        Log.i(
            TAG,
            "load() entered: model=${req.model} backend=${req.backend} " +
                "quant=${req.quantization}"
        )
        if (loaded) {
            Log.i(TAG, "load(): already loaded, returning Ok")
            return BackendResult.Ok(Unit)
        }

        if (!NativeCausalLm.ensureLoaded()) {
            Log.e(TAG, "load(): native libs unavailable on this device")
            return BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                "libquickai_jni.so / libcausallm_api.so not available on this device"
            )
        }

        val nativeModelOrdinal = mapModelId(req.model)
            ?: run {
                Log.e(TAG, "load(): model ${req.model} has no native ordinal")
                return BackendResult.Err(
                    QuickAiError.UNSUPPORTED,
                    "Model ${req.model} is not supported by NativeQuickDotAI"
                )
            }

        // Point the native loader at the right directory. The C API
        // builds its model paths as "./models/<name>-<quant>" relative
        // to the process's cwd, so when a caller (e.g. SampleTestAPP)
        // passes an absolute [modelPath] we chdir the process to its
        // grandparent — for `.../files/models/qwen3-0.6b-w4a32` that is
        // `.../files/`, which makes `./models/qwen3-0.6b-w4a32` resolve
        // to the caller's intended directory. See NativeCausalLm.chdirNative.
        req.modelPath?.takeIf { it.isNotBlank() }?.let { modelPathStr ->
            val nativeCwd = File(modelPathStr).parentFile?.parentFile
            if (nativeCwd == null) {
                Log.e(
                    TAG,
                    "load(): modelPath=$modelPathStr has no grandparent; " +
                        "cannot derive native cwd"
                )
                return BackendResult.Err(
                    QuickAiError.INVALID_PARAMETER,
                    "modelPath must be an absolute path with at least two " +
                        "parent directories (e.g. .../files/models/qwen3-0.6b-w4a32)"
                )
            }
            if (!nativeCwd.exists()) {
                Log.e(TAG, "load(): native cwd does not exist: $nativeCwd")
                return BackendResult.Err(
                    QuickAiError.MODEL_LOAD_FAILED,
                    "Native working directory $nativeCwd does not exist. " +
                        "Push the model to $modelPathStr first."
                )
            }
            Log.i(TAG, "load(): chdir -> ${nativeCwd.absolutePath}")
            val chdirErr = NativeCausalLm.chdirNative(nativeCwd.absolutePath)
            if (chdirErr != 0) {
                Log.e(TAG, "load(): chdir failed, errno=$chdirErr")
                return BackendResult.Err(
                    QuickAiError.MODEL_LOAD_FAILED,
                    "chdir to ${nativeCwd.absolutePath} failed (errno=$chdirErr)"
                )
            }
        }

        return try {
            Log.i(
                TAG,
                "load(): calling loadModelHandleNative(backend=${req.backend.ordinal}, " +
                    "model=$nativeModelOrdinal, quant=${req.quantization.ordinal})"
            )
            val result = NativeCausalLm.loadModelHandleNative(
                backendOrdinal = mapBackend(req.backend),
                modelOrdinal = nativeModelOrdinal,
                quantOrdinal = mapQuant(req.quantization)
            )
            Log.i(
                TAG,
                "load(): loadModelHandleNative returned " +
                    "errorCode=${result.errorCode} handle=0x${result.handle.toString(16)}"
            )
            if (result.errorCode != 0 || result.handle == 0L) {
                Log.e(
                    TAG,
                    "load(): loadModelHandle FAILED — errorCode=${result.errorCode} " +
                        "(this is the NATIVE engine; for Gemma4 you want LiteRTLm " +
                        "— check that ModelId.GEMMA4 was NOT requested)"
                )
                BackendResult.Err(
                    QuickAiError.fromNativeCode(result.errorCode),
                    "loadModelHandle failed (errorCode=${result.errorCode})"
                )
            } else {
                handle = result.handle
                loaded = true
                // Architecture is resolved native-side; we report the
                // model key until we add a dedicated native getter.
                architecture = req.model.name
                Log.i(TAG, "load(): SUCCESS, handle=0x${handle.toString(16)}")
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "load(): loadModelHandleNative threw", t)
            BackendResult.Err(QuickAiError.MODEL_LOAD_FAILED, t.message)
        }
    }

    override fun run(prompt: String): BackendResult<String> {
        if (!loaded || handle == 0L) {
            return BackendResult.Err(QuickAiError.NOT_INITIALIZED)
        }
        return try {
            val r = NativeCausalLm.runModelHandleNative(handle, prompt)
            if (r.errorCode != 0) {
                BackendResult.Err(QuickAiError.fromNativeCode(r.errorCode))
            } else {
                BackendResult.Ok(r.output.orEmpty())
            }
        } catch (t: Throwable) {
            Log.e(TAG, "runModelHandleNative threw", t)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    /**
     * @brief Streaming override that forwards deltas from the native
     * `runModelHandleStreaming` entry point into [sink].
     *
     * Threading: this method runs on the caller thread; the native
     * callback is invoked synchronously on the same thread for every
     * delta, so no JNI AttachCurrentThread is needed. Terminal events
     * (onDone / onError) are synthesized from the native return value
     * because the C API reports completion through its return code
     * rather than through the streamer vtable.
     */
    override fun runStreaming(
        prompt: String,
        sink: StreamSink
    ): BackendResult<Unit> {
        if (!loaded || handle == 0L) {
            Log.e(TAG, "runStreaming(): called before load()")
            val err = BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "NativeQuickDotAI has not been loaded yet"
            )
            sink.onError(err.error, err.message)
            return err
        }

        Log.i(TAG, "runStreaming(): prompt length=${prompt.length}")
        return try {
            val errorCode = NativeCausalLm.runModelHandleStreamingNative(
                handle,
                prompt
            ) { delta ->
                // Called on the caller thread (this one). Forward
                // straight to the sink — the contract is that sink
                // implementations are non-blocking.
                sink.onDelta(delta)
            }
            if (errorCode != 0) {
                val err = QuickAiError.fromNativeCode(errorCode)
                Log.e(
                    TAG,
                    "runStreaming(): runModelHandleStreaming failed " +
                        "errorCode=$errorCode (${err.name})"
                )
                sink.onError(err, "runModelHandleStreaming failed (errorCode=$errorCode)")
                BackendResult.Err(err, "runModelHandleStreaming failed (errorCode=$errorCode)")
            } else {
                Log.i(TAG, "runStreaming(): native runner returned NONE, signalling onDone")
                sink.onDone()
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "runStreaming(): runModelHandleStreamingNative threw", t)
            sink.onError(QuickAiError.INFERENCE_FAILED, t.message)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    override fun metrics(): BackendResult<PerformanceMetrics> {
        if (!loaded || handle == 0L) {
            return BackendResult.Err(QuickAiError.NOT_INITIALIZED)
        }
        return try {
            val m = NativeCausalLm.getPerformanceMetricsHandleNative(handle)
            if (m.errorCode != 0) {
                BackendResult.Err(QuickAiError.fromNativeCode(m.errorCode))
            } else {
                BackendResult.Ok(
                    PerformanceMetrics(
                        prefillTokens = m.prefillTokens,
                        prefillDurationMs = m.prefillDurationMs,
                        generationTokens = m.generationTokens,
                        generationDurationMs = m.generationDurationMs,
                        totalDurationMs = m.totalDurationMs,
                        initializationDurationMs = m.initializationDurationMs,
                        peakMemoryKb = m.peakMemoryKb
                    )
                )
            }
        } catch (t: Throwable) {
            Log.e(TAG, "getPerformanceMetricsHandleNative threw", t)
            BackendResult.Err(QuickAiError.UNKNOWN, t.message)
        }
    }

    override fun unload(): BackendResult<Unit> {
        if (!loaded || handle == 0L) {
            return BackendResult.Ok(Unit)
        }
        return try {
            val ec = NativeCausalLm.unloadModelHandleNative(handle)
            loaded = false
            if (ec != 0) {
                BackendResult.Err(QuickAiError.fromNativeCode(ec))
            } else {
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            Log.w(TAG, "unloadModelHandleNative threw", t)
            BackendResult.Err(QuickAiError.UNKNOWN, t.message)
        }
    }

    // --- chat session (dummy) --------------------------------------------

    private var activeSession: NativeChatSession? = null

    override fun openChatSession(
        config: QuickAiChatSessionConfig?
    ): BackendResult<QuickAiChatSession> {
        if (!loaded || handle == 0L) {
            return BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "NativeQuickDotAI has not been loaded yet"
            )
        }
        if (activeSession != null) {
            return BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "A chat session is already active (${activeSession!!.sessionId}). " +
                    "Close it before opening a new one."
            )
        }
        val session = NativeChatSession()
        activeSession = session
        Log.i(TAG, "openChatSession(): created dummy session ${session.sessionId}")
        return BackendResult.Ok(session)
    }

    override fun close() {
        activeSession?.close()
        activeSession = null
        if (handle != 0L) {
            try {
                NativeCausalLm.destroyModelHandleNative(handle)
            } catch (t: Throwable) {
                Log.w(TAG, "destroyModelHandleNative threw", t)
            }
            handle = 0L
        }
        loaded = false
    }

    // --- enum → native-ordinal mapping ---------------------------------

    /**
     * @brief Maps a ModelId to the C enum ordinal in quick_dot_ai_api.h.
     * Returns null for values that are Kotlin-only (e.g. GEMMA4) —
     * those are routed to [LiteRTLm] instead and never reach this
     * engine.
     */
    private fun mapModelId(m: ModelId): Int? = when (m) {
        ModelId.QWEN3_0_6B -> 0 // CAUSAL_LM_MODEL_QWEN3_0_6B
        ModelId.GEMMA4 -> null
    }

    private fun mapBackend(b: BackendType): Int = when (b) {
        BackendType.CPU -> 0
        BackendType.GPU -> 1
        BackendType.NPU -> 2
    }

    private fun mapQuant(q: QuantizationType): Int = when (q) {
        QuantizationType.UNKNOWN -> 0
        QuantizationType.W4A32 -> 1
        QuantizationType.W16A16 -> 2
        QuantizationType.W8A16 -> 3
        QuantizationType.W32A32 -> 4
    }

    companion object {
        private const val TAG = "NativeQuickDotAI"
    }
}

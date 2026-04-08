// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    NativeCausalLmBackend.kt
 * @brief   Backend implementation that forwards to the handle-based
 *          causal_lm_api.h via JNI.
 */
package com.example.QuickAI.service.backend

import android.util.Log
import com.example.QuickAI.service.BackendType
import com.example.QuickAI.service.LoadModelRequest
import com.example.QuickAI.service.ModelId
import com.example.QuickAI.service.NativeCausalLm
import com.example.QuickAI.service.PerformanceMetrics
import com.example.QuickAI.service.QuantizationType
import com.example.QuickAI.service.QuickAiError

/**
 * @brief Kotlin wrapper around a single `CausalLmHandle` in native code.
 *
 * Non-thread-safe by design — one ModelWorker owns one backend instance and
 * calls into it from one thread.
 */
class NativeCausalLmBackend : Backend {

    override val kind: String = "native"

    override var architecture: String? = null
        private set

    private var handle: Long = 0L
    private var loaded: Boolean = false

    override fun load(req: LoadModelRequest): BackendResult<Unit> {
        if (loaded) return BackendResult.Ok(Unit)

        if (!NativeCausalLm.ensureLoaded()) {
            return BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                "libquickai_jni.so / libcausallm_api.so not available on this device"
            )
        }

        val nativeModelOrdinal = mapModelId(req.model)
            ?: return BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "Model ${req.model} is not supported by the native backend"
            )

        return try {
            val result = NativeCausalLm.loadModelHandleNative(
                backendOrdinal = mapBackend(req.backend),
                modelOrdinal = nativeModelOrdinal,
                quantOrdinal = mapQuant(req.quantization)
            )
            if (result.errorCode != 0 || result.handle == 0L) {
                BackendResult.Err(
                    QuickAiError.fromNativeCode(result.errorCode),
                    "loadModelHandle failed"
                )
            } else {
                handle = result.handle
                loaded = true
                // Architecture is resolved native-side; we report the model
                // key until we add a dedicated native getter.
                architecture = req.model.name
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "loadModelHandleNative threw", t)
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

    override fun close() {
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
     * @brief Maps a ModelId to the C enum ordinal in causal_lm_api.h.
     * Returns null for values that are Kotlin-only (e.g. GEMMA4) — those
     * are never routed to this backend.
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
        private const val TAG = "NativeCausalLmBackend"
    }
}

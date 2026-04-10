// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    Types.kt
 * @brief   Value types shared by the QuickDotAI interface and its
 *          implementations.
 *
 * The enums mirror the C enums in Applications/CausalLM/api/quick_dot_ai_api.h.
 * Kotlin-only values (ModelId.GEMMA4) are additionally defined so the host
 * app can route to LiteRT-LM without crossing the JNI boundary.
 *
 * Every public class in this file carries `@Serializable` so host apps
 * that want to JSON-ify requests/responses (for example QuickAIService's
 * REST layer) can do so without redefining the types — the AAR exposes
 * kotlinx-serialization-json as an `api` dependency for that purpose.
 */
package com.example.quickdotai

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * @brief Compute backend. Mirrors BackendType in quick_dot_ai_api.h.
 */
@Serializable
enum class BackendType {
    CPU,
    GPU,
    NPU
}

/**
 * @brief Model identifier.
 *
 * The first entries mirror the C enum `ModelType` in quick_dot_ai_api.h.
 * [GEMMA4] is a Kotlin-only value that triggers the [LiteRTLm] code
 * path; it never crosses the JNI boundary.
 */
@Serializable
enum class ModelId {
    QWEN3_0_6B,
    GEMMA4
}

/**
 * @brief Quantization type. Mirrors ModelQuantizationType in
 *        quick_dot_ai_api.h.
 */
@Serializable
enum class QuantizationType {
    UNKNOWN,
    W4A32,
    W16A16,
    W8A16,
    W32A32
}

/**
 * @brief Error code. Mirrors ErrorCode in quick_dot_ai_api.h plus a few
 *        Kotlin-level additions for out-of-band conditions.
 */
@Serializable
enum class QuickAiError(val code: Int) {
    NONE(0),
    INVALID_PARAMETER(1),
    MODEL_LOAD_FAILED(2),
    INFERENCE_FAILED(3),
    NOT_INITIALIZED(4),
    INFERENCE_NOT_RUN(5),
    UNKNOWN(99),

    // Kotlin-only conditions returned by higher layers (QuickAIService
    // worker / dispatcher). The AAR itself only surfaces the native
    // codes and NOT_INITIALIZED, but it is convenient for the host app
    // to have the full enum in one place.
    QUEUE_FULL(100),
    MODEL_NOT_FOUND(101),
    UNSUPPORTED(102),
    BAD_REQUEST(103);

    companion object {
        fun fromNativeCode(code: Int): QuickAiError =
            entries.firstOrNull { it.code == code } ?: UNKNOWN
    }
}

/**
 * @brief Descriptor passed to [QuickDotAI.load].
 *
 * [modelPath] is required by [LiteRTLm] (which takes an explicit path
 * to a `.litertlm` file) and ignored by [NativeQuickDotAI] (which
 * discovers its model assets through the native C API's internal
 * model-directory resolution).
 */
@Serializable
data class LoadModelRequest(
    val backend: BackendType = BackendType.GPU,
    val model: ModelId,
    val quantization: QuantizationType = QuantizationType.W4A32,
    @SerialName("model_path") val modelPath: String? = null
) {
    /**
     * Canonical key shared across the stack: one worker/handle per
     * (model, quantization) pair.
     */
    val modelKey: String get() = "${model.name}:${quantization.name}"
}

/**
 * @brief Performance metrics for the most recent run.
 *
 * Not every engine fills every field:
 *  - [NativeQuickDotAI] fills prefill_* / generation_* / peak_memory_kb
 *    from the C API's PerformanceMetrics struct.
 *  - [LiteRTLm] currently only fills [initializationDurationMs] and
 *    [totalDurationMs] because the LiteRT-LM Kotlin API does not expose
 *    token-level counters in the release we target.
 */
@Serializable
data class PerformanceMetrics(
    @SerialName("prefill_tokens") val prefillTokens: Int = 0,
    @SerialName("prefill_duration_ms") val prefillDurationMs: Double = 0.0,
    @SerialName("generation_tokens") val generationTokens: Int = 0,
    @SerialName("generation_duration_ms") val generationDurationMs: Double = 0.0,
    @SerialName("total_duration_ms") val totalDurationMs: Double = 0.0,
    @SerialName("initialization_duration_ms") val initializationDurationMs: Double = 0.0,
    @SerialName("peak_memory_kb") val peakMemoryKb: Long = 0
)

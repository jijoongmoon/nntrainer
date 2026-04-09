// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    Protocol.kt
 * @brief   Wire-level DTOs, enums and helpers shared between the HTTP
 *          server (inside QuickAIService) and any client.
 *
 * The enums mirror the C enums in causal_lm_api.h. Kotlin-only values like
 * GEMMA4 are additionally defined so the service can route those requests
 * to LiteRT-LM without ever crossing the JNI boundary.
 */
package com.example.QuickAI.service

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * @brief Compute backend. Mirrors BackendType in causal_lm_api.h.
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
 * The first entries mirror the C enum `ModelType` in causal_lm_api.h.
 * GEMMA4 is a Kotlin-only value that triggers the LiteRT-LM code path
 * inside the service; it is never sent over JNI.
 */
@Serializable
enum class ModelId {
    QWEN3_0_6B,
    GEMMA4
}

/**
 * @brief Quantization type. Mirrors ModelQuantizationType in causal_lm_api.h.
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
 * @brief Error code. Mirrors ErrorCode in causal_lm_api.h plus a few
 * Kotlin-level additions for transport errors.
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

    // Kotlin-only
    QUEUE_FULL(100),
    MODEL_NOT_FOUND(101),
    UNSUPPORTED(102),
    BAD_REQUEST(103);

    companion object {
        fun fromNativeCode(code: Int): QuickAiError =
            values().firstOrNull { it.code == code } ?: UNKNOWN
    }
}

/* -------------------------------------------------------------------- */
/* setOptions                                                           */
/* -------------------------------------------------------------------- */

@Serializable
data class SetOptionsRequest(
    @SerialName("use_chat_template") val useChatTemplate: Boolean = false,
    @SerialName("debug_mode") val debugMode: Boolean = false,
    val verbose: Boolean = false
)

@Serializable
data class SetOptionsResponse(
    @SerialName("error_code") val errorCode: Int
)

/* -------------------------------------------------------------------- */
/* loadModel                                                            */
/* -------------------------------------------------------------------- */

@Serializable
data class LoadModelRequest(
    val backend: BackendType = BackendType.GPU,
    val model: ModelId,
    val quantization: QuantizationType = QuantizationType.W4A32,
    /**
     * Absolute path to the on-device model asset. Required for the
     * LiteRT-LM backend (Gemma4) which takes an explicit `.litertlm` file
     * path. Optional (ignored) for the native causal_lm_api backend, which
     * performs its own model-directory discovery.
     */
    @SerialName("model_path") val modelPath: String? = null
) {
    /** Canonical key used by ModelRegistry — one worker per key. */
    val modelKey: String get() = "${model.name}:${quantization.name}"
}

@Serializable
data class LoadModelResponse(
    @SerialName("model_id") val modelId: String,
    val architecture: String? = null,
    @SerialName("error_code") val errorCode: Int = 0,
    val message: String? = null
)

/* -------------------------------------------------------------------- */
/* runModel                                                             */
/* -------------------------------------------------------------------- */

@Serializable
data class RunModelRequest(
    val prompt: String
)

@Serializable
data class RunModelResponse(
    val output: String? = null,
    @SerialName("error_code") val errorCode: Int = 0,
    val message: String? = null
)

/* -------------------------------------------------------------------- */
/* runModel streaming (NDJSON frames — see Architecture.md §5.1)         */
/* -------------------------------------------------------------------- */

/**
 * @brief One frame of the NDJSON response emitted by
 * `POST /v1/models/{id}/run_stream`.
 *
 * Each frame is serialised onto its own line, terminated by `\n`. The
 * recognised [type] values are:
 *  - `"delta"` — a chunk of newly-generated text; [text] is non-null.
 *  - `"done"`  — the stream completed successfully; [durationMs] may be set.
 *  - `"error"` — the stream terminated with an error; [errorCode] and
 *                [message] are set.
 *
 * A stream MUST end with exactly one `done` or `error` frame.
 */
@Serializable
data class StreamFrame(
    val type: String,
    val text: String? = null,
    @SerialName("duration_ms") val durationMs: Long? = null,
    @SerialName("error_code") val errorCode: Int? = null,
    val message: String? = null
)

/* -------------------------------------------------------------------- */
/* getPerformanceMetrics                                                */
/* -------------------------------------------------------------------- */

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

@Serializable
data class PerformanceMetricsResponse(
    val metrics: PerformanceMetrics? = null,
    @SerialName("error_code") val errorCode: Int = 0,
    val message: String? = null
)

/* -------------------------------------------------------------------- */
/* list / misc                                                          */
/* -------------------------------------------------------------------- */

@Serializable
data class LoadedModelInfo(
    @SerialName("model_id") val modelId: String,
    val architecture: String?,
    @SerialName("backend_kind") val backendKind: String // "native" or "litert-lm"
)

@Serializable
data class ListModelsResponse(
    val models: List<LoadedModelInfo>
)

@Serializable
data class HealthResponse(
    val status: String = "ok",
    val port: Int
)

/**
 * @brief Response to POST /v1/connect.
 *
 * The connect endpoint is intentionally minimal: it performs no model
 * work and only confirms that the REST surface is reachable and ready
 * to accept further requests. Clients use this as an explicit
 * "handshake" separate from the periodic /v1/health liveness probe.
 */
@Serializable
data class ConnectResponse(
    val connected: Boolean = true,
    val port: Int,
    val message: String = "connected"
)

@Serializable
data class ErrorResponse(
    @SerialName("error_code") val errorCode: Int,
    val message: String
)

/**
 * @brief Default TCP port for the REST server.
 *
 * 3453 is the user-requested port. The server falls back to an ephemeral
 * port if 3453 is already in use and publishes the actual bound port via
 * QuickAIPortProvider / /v1/health.
 */
const val DEFAULT_QUICKAI_PORT: Int = 3453

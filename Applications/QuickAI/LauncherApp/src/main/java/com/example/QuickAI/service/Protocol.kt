// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    Protocol.kt
 * @brief   REST wire-level DTOs and helpers for QuickAIService.
 *
 * The fundamental value types (BackendType, ModelId, QuantizationType,
 * QuickAiError, LoadModelRequest, PerformanceMetrics) live in the
 * :QuickDotAI AAR so both the service and third-party apps can share
 * them. This file only defines the REST-specific request/response DTOs
 * and JSON-level helpers that are private to the HTTP layer.
 */
package com.example.QuickAI.service

import com.example.quickdotai.PerformanceMetrics
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/* -------------------------------------------------------------------- */
/* Re-export shared types from the QuickDotAI AAR so existing unqualified
 * references inside com.example.QuickAI.service keep compiling without
 * a pile of churn in every file.                                       */
/* -------------------------------------------------------------------- */
typealias BackendType = com.example.quickdotai.BackendType
typealias ModelId = com.example.quickdotai.ModelId
typealias QuantizationType = com.example.quickdotai.QuantizationType
typealias QuickAiError = com.example.quickdotai.QuickAiError
typealias LoadModelRequest = com.example.quickdotai.LoadModelRequest
typealias PerformanceMetrics = com.example.quickdotai.PerformanceMetrics
typealias QuickAiChatSamplingConfig = com.example.quickdotai.QuickAiChatSamplingConfig
typealias QuickAiChatTemplateKwargs = com.example.quickdotai.QuickAiChatTemplateKwargs
typealias QuickAiChatSessionConfig = com.example.quickdotai.QuickAiChatSessionConfig

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

/* -------------------------------------------------------------------- */
/* Chat session DTOs                                                    */
/* -------------------------------------------------------------------- */

/**
 * @brief One content part in a chat message, as carried over the wire.
 *
 * Image bytes are Base64-encoded so they survive JSON serialization.
 * The dispatcher converts these to [com.example.quickdotai.PromptPart]
 * at the boundary.
 */
@Serializable
data class ChatContentPartDto(
    val type: String, // "text", "image_file", "image_bytes"
    val text: String? = null,
    @SerialName("absolute_path") val absolutePath: String? = null,
    @SerialName("image_base64") val imageBase64: String? = null
)

@Serializable
data class ChatMessageDto(
    val role: String, // "system", "user", "assistant"
    val parts: List<ChatContentPartDto>
)

@Serializable
data class ChatOpenRequest(
    val config: QuickAiChatSessionConfig? = null
)

@Serializable
data class ChatOpenResponse(
    @SerialName("session_id") val sessionId: String,
    @SerialName("error_code") val errorCode: Int = 0,
    val message: String? = null
)

@Serializable
data class ChatRunRequest(
    @SerialName("session_id") val sessionId: String,
    val messages: List<ChatMessageDto>
)

@Serializable
data class ChatRunResponse(
    val content: String? = null,
    val metrics: PerformanceMetrics? = null,
    @SerialName("error_code") val errorCode: Int = 0,
    val message: String? = null
)

@Serializable
data class ChatSessionIdRequest(
    @SerialName("session_id") val sessionId: String
)

@Serializable
data class ChatRebuildRequest(
    @SerialName("session_id") val sessionId: String,
    val messages: List<ChatMessageDto>
)

@Serializable
data class ChatGenericResponse(
    @SerialName("error_code") val errorCode: Int = 0,
    val message: String? = null
)

/**
 * @brief Default TCP port for the REST server.
 *
 * 3453 is the user-requested port. The server falls back to an ephemeral
 * port if 3453 is already in use and publishes the actual bound port via
 * QuickAIPortProvider / /v1/health.
 */
const val DEFAULT_QUICKAI_PORT: Int = 3453

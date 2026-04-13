// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    Models.kt
 * @brief   Wire-format data classes used by QuickAiClient to talk to
 *          QuickAIService. These are intentionally a duplicate of the
 *          service-side Protocol.kt — ClientApp must be independently
 *          buildable and installable, so we do not cross-link the two
 *          modules.
 *
 * The shapes must stay in sync with
 * LauncherApp/.../service/Protocol.kt.
 */
package com.example.clientapp.api

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
enum class BackendType { CPU, GPU, NPU }

@Serializable
enum class ModelId { QWEN3_0_6B, GEMMA4 }

@Serializable
enum class QuantizationType { UNKNOWN, W4A32, W16A16, W8A16, W32A32 }

@Serializable
data class SetOptionsRequest(
    @SerialName("use_chat_template") val useChatTemplate: Boolean = false,
    @SerialName("debug_mode") val debugMode: Boolean = false,
    val verbose: Boolean = false
)

@Serializable
data class SetOptionsResponse(
    @SerialName("error_code") val errorCode: Int = 0
)

@Serializable
data class LoadModelRequest(
    val backend: BackendType = BackendType.CPU,
    val model: ModelId,
    val quantization: QuantizationType = QuantizationType.W4A32,
    /**
     * Absolute path to the on-device model asset. Required for Gemma4
     * (LiteRT-LM); ignored for native causal_lm_api models.
     */
    @SerialName("model_path") val modelPath: String? = null,
    @SerialName("max_num_tokens") val maxNumTokens: Int? = null
)

@Serializable
data class LoadModelResponse(
    @SerialName("model_id") val modelId: String? = null,
    val architecture: String? = null,
    @SerialName("error_code") val errorCode: Int = 0,
    val message: String? = null
)

@Serializable
data class RunModelRequest(val prompt: String)

@Serializable
data class RunModelResponse(
    val output: String? = null,
    @SerialName("error_code") val errorCode: Int = 0,
    val message: String? = null
)

/**
 * @brief One line (frame) of the NDJSON stream emitted by
 * `POST /v1/models/{id}/run_stream`. Keep in sync with the service-side
 * [com.example.QuickAI.service.StreamFrame].
 */
@Serializable
data class StreamFrame(
    val type: String,
    val text: String? = null,
    @SerialName("duration_ms") val durationMs: Long? = null,
    @SerialName("error_code") val errorCode: Int? = null,
    val message: String? = null
)

/**
 * @brief Higher-level sealed class that [QuickAiClient.runModelStreaming]
 * hands to its caller for each decoded frame. UI code pattern-matches on
 * this instead of poking at the raw [StreamFrame] fields.
 */
sealed class StreamChunk {
    data class Delta(val text: String) : StreamChunk()
    data class Done(val durationMs: Long?) : StreamChunk()
    data class Error(val errorCode: Int, val message: String) : StreamChunk()
}

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

@Serializable
data class LoadedModelInfo(
    @SerialName("model_id") val modelId: String,
    val architecture: String? = null,
    @SerialName("backend_kind") val backendKind: String
)

@Serializable
data class ListModelsResponse(val models: List<LoadedModelInfo>)

@Serializable
data class HealthResponse(val status: String, val port: Int)

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

@Serializable
data class ChatSamplingConfig(
    val temperature: Double? = null,
    @SerialName("top_k") val topK: Int? = null,
    @SerialName("top_p") val topP: Double? = null,
    @SerialName("min_p") val minP: Double? = null,
    @SerialName("max_tokens") val maxTokens: Int? = null,
    val seed: Int? = null
)

@Serializable
data class ChatTemplateKwargs(
    @SerialName("enable_thinking") val enableThinking: Boolean? = null
)

@Serializable
data class ChatSessionConfig(
    val sampling: ChatSamplingConfig? = null,
    @SerialName("chat_template_kwargs") val chatTemplateKwargs: ChatTemplateKwargs? = null
)

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
    val config: ChatSessionConfig? = null
)

@Serializable
data class ChatOpenResponse(
    @SerialName("session_id") val sessionId: String? = null,
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

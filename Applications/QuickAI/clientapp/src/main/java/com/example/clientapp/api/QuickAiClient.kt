// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    QuickAiClient.kt
 * @brief   OkHttp-based REST client for the QuickAIService REST surface.
 *
 * The service endpoint is fixed at 127.0.0.1:3453 (see Architecture.md
 * §2.3). Each call corresponds directly to a handle-based entry point of
 * causal_lm_api.h on the service side.
 */
package com.example.clientapp.api

import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

/**
 * @brief High-level result wrapper so callers never have to deal with raw
 * IOException or JSON parsing failures in UI code.
 */
sealed class ApiResult<out T> {
    data class Ok<T>(val value: T) : ApiResult<T>()
    data class Err(val errorCode: Int, val message: String) : ApiResult<Nothing>()
}

class QuickAiClient(
    baseUrl: String = "http://127.0.0.1:3453"
) {
    private val json = Json {
        encodeDefaults = true
        ignoreUnknownKeys = true
    }

    private val http = OkHttpClient.Builder()
        // Model inference is slow on-device. Give the server plenty of
        // time before giving up on a run.
        .callTimeout(10, TimeUnit.MINUTES)
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.MINUTES)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val base: String = baseUrl.trimEnd('/')

    suspend fun health(): ApiResult<HealthResponse> =
        get("/v1/health", HealthResponse.serializer())

    suspend fun connect(): ApiResult<ConnectResponse> =
        postEmpty("/v1/connect", ConnectResponse.serializer())

    suspend fun listModels(): ApiResult<ListModelsResponse> =
        get("/v1/models", ListModelsResponse.serializer())

    suspend fun setOptions(req: SetOptionsRequest): ApiResult<SetOptionsResponse> =
        postJson("/v1/options", req, SetOptionsRequest.serializer(),
                 SetOptionsResponse.serializer())

    suspend fun loadModel(req: LoadModelRequest): ApiResult<LoadModelResponse> =
        postJson("/v1/models", req, LoadModelRequest.serializer(),
                 LoadModelResponse.serializer())

    suspend fun runModel(
        modelId: String,
        req: RunModelRequest
    ): ApiResult<RunModelResponse> =
        postJson(
            "/v1/models/$modelId/run",
            req,
            RunModelRequest.serializer(),
            RunModelResponse.serializer()
        )

    /**
     * @brief Streaming counterpart of [runModel].
     *
     * Opens `POST /v1/models/{id}/run_stream`, reads the NDJSON body
     * line-by-line via OkHttp's BufferedSource, decodes each line into a
     * [StreamFrame], and dispatches a higher-level [StreamChunk] to
     * [onChunk]. The coroutine returns [ApiResult.Ok] on a clean `done`
     * frame or [ApiResult.Err] if the transport itself fails (network
     * error, HTTP != 2xx, malformed frame).
     *
     * A server-side error frame IS reported through [onChunk] as
     * [StreamChunk.Error] and ALSO returned as [ApiResult.Err] so the
     * caller doesn't have to remember to check both places.
     *
     * The call is blocking for the duration of the stream — wrap it in a
     * coroutine scope that can be cancelled if the user navigates away.
     * The suspend [onChunk] is invoked on Dispatchers.IO; switch to the
     * main thread inside it if you need to touch UI state.
     */
    suspend fun runModelStreaming(
        modelId: String,
        req: RunModelRequest,
        onChunk: suspend (StreamChunk) -> Unit
    ): ApiResult<Unit> = withContext(Dispatchers.IO) {
        val body = json.encodeToString(RunModelRequest.serializer(), req)
            .toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
            .url("$base/v1/models/$modelId/run_stream")
            .post(body)
            .build()
        try {
            http.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    val bodyString = response.body?.string().orEmpty()
                    val err = runCatching {
                        json.decodeFromString(ErrorResponse.serializer(), bodyString)
                    }.getOrNull()
                    return@withContext ApiResult.Err(
                        errorCode = err?.errorCode ?: response.code,
                        message = err?.message ?: "HTTP ${response.code}"
                    )
                }
                val source = response.body?.source()
                    ?: return@withContext ApiResult.Err(-1, "empty response body")

                var terminalError: ApiResult.Err? = null
                // Read until EOF. readUtf8LineStrict() throws on EOF so
                // we break explicitly when exhausted() becomes true.
                while (!source.exhausted()) {
                    val line = try {
                        source.readUtf8LineStrict()
                    } catch (io: IOException) {
                        return@withContext ApiResult.Err(
                            -1,
                            "stream read error: ${io.message}"
                        )
                    }
                    if (line.isEmpty()) continue
                    val frame = try {
                        json.decodeFromString(StreamFrame.serializer(), line)
                    } catch (t: Throwable) {
                        return@withContext ApiResult.Err(-1, "bad frame: ${t.message}")
                    }
                    when (frame.type) {
                        "delta" -> {
                            val text = frame.text ?: continue
                            onChunk(StreamChunk.Delta(text))
                        }
                        "done" -> {
                            onChunk(StreamChunk.Done(frame.durationMs))
                            return@withContext ApiResult.Ok(Unit)
                        }
                        "error" -> {
                            val code = frame.errorCode ?: -1
                            val msg = frame.message ?: "stream error"
                            onChunk(StreamChunk.Error(code, msg))
                            terminalError = ApiResult.Err(code, msg)
                            // Keep draining in case there's a trailing
                            // newline before EOF, but record the error.
                        }
                        else -> {
                            // Unknown frame type — ignore forward-
                            // compatibly.
                        }
                    }
                }
                // Stream ended without an explicit done/error frame.
                terminalError ?: ApiResult.Err(-1, "stream ended without done frame")
            }
        } catch (io: IOException) {
            ApiResult.Err(-1, "network error: ${io.message}")
        }
    }

    // --- Chat session methods ---

    suspend fun chatOpen(
        modelId: String,
        config: ChatSessionConfig? = null
    ): ApiResult<ChatOpenResponse> =
        postJson(
            "/v1/models/$modelId/chat/open",
            ChatOpenRequest(config),
            ChatOpenRequest.serializer(),
            ChatOpenResponse.serializer()
        )

    suspend fun chatRun(
        modelId: String,
        sessionId: String,
        messages: List<ChatMessageDto>
    ): ApiResult<ChatRunResponse> =
        postJson(
            "/v1/models/$modelId/chat/run",
            ChatRunRequest(sessionId, messages),
            ChatRunRequest.serializer(),
            ChatRunResponse.serializer()
        )

    suspend fun chatRunStreaming(
        modelId: String,
        sessionId: String,
        messages: List<ChatMessageDto>,
        onChunk: suspend (StreamChunk) -> Unit
    ): ApiResult<Unit> = withContext(Dispatchers.IO) {
        val reqBody = json.encodeToString(
            ChatRunRequest.serializer(),
            ChatRunRequest(sessionId, messages)
        ).toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
            .url("$base/v1/models/$modelId/chat/run_stream")
            .post(reqBody)
            .build()
        try {
            http.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    val bodyStr = response.body?.string().orEmpty()
                    val err = runCatching {
                        json.decodeFromString(ErrorResponse.serializer(), bodyStr)
                    }.getOrNull()
                    return@withContext ApiResult.Err(
                        errorCode = err?.errorCode ?: response.code,
                        message = err?.message ?: "HTTP ${response.code}"
                    )
                }
                val source = response.body?.source()
                    ?: return@withContext ApiResult.Err(-1, "empty response body")

                var terminalError: ApiResult.Err? = null
                while (!source.exhausted()) {
                    val line = try {
                        source.readUtf8LineStrict()
                    } catch (io: IOException) {
                        return@withContext ApiResult.Err(-1, "stream read error: ${io.message}")
                    }
                    if (line.isEmpty()) continue
                    val frame = try {
                        json.decodeFromString(StreamFrame.serializer(), line)
                    } catch (t: Throwable) {
                        return@withContext ApiResult.Err(-1, "bad frame: ${t.message}")
                    }
                    when (frame.type) {
                        "delta" -> onChunk(StreamChunk.Delta(frame.text ?: continue))
                        "done" -> {
                            onChunk(StreamChunk.Done(frame.durationMs))
                            return@withContext ApiResult.Ok(Unit)
                        }
                        "error" -> {
                            val code = frame.errorCode ?: -1
                            val msg = frame.message ?: "stream error"
                            onChunk(StreamChunk.Error(code, msg))
                            terminalError = ApiResult.Err(code, msg)
                        }
                    }
                }
                terminalError ?: ApiResult.Err(-1, "stream ended without done frame")
            }
        } catch (io: IOException) {
            ApiResult.Err(-1, "network error: ${io.message}")
        }
    }

    suspend fun chatCancel(
        modelId: String,
        sessionId: String
    ): ApiResult<ChatGenericResponse> =
        postJson(
            "/v1/models/$modelId/chat/cancel",
            ChatSessionIdRequest(sessionId),
            ChatSessionIdRequest.serializer(),
            ChatGenericResponse.serializer()
        )

    suspend fun chatRebuild(
        modelId: String,
        sessionId: String,
        messages: List<ChatMessageDto>
    ): ApiResult<ChatGenericResponse> =
        postJson(
            "/v1/models/$modelId/chat/rebuild",
            ChatRebuildRequest(sessionId, messages),
            ChatRebuildRequest.serializer(),
            ChatGenericResponse.serializer()
        )

    suspend fun chatClose(
        modelId: String,
        sessionId: String
    ): ApiResult<ChatGenericResponse> =
        postJson(
            "/v1/models/$modelId/chat/close",
            ChatSessionIdRequest(sessionId),
            ChatSessionIdRequest.serializer(),
            ChatGenericResponse.serializer()
        )

    // --- existing methods ---

    suspend fun getMetrics(modelId: String): ApiResult<PerformanceMetricsResponse> =
        get("/v1/models/$modelId/metrics", PerformanceMetricsResponse.serializer())

    suspend fun unloadModel(modelId: String): ApiResult<SetOptionsResponse> =
        delete("/v1/models/$modelId", SetOptionsResponse.serializer())

    // --- generic verbs -------------------------------------------------

    private suspend fun <T> get(
        path: String,
        serializer: kotlinx.serialization.KSerializer<T>
    ): ApiResult<T> = withContext(Dispatchers.IO) {
        execute(Request.Builder().url(base + path).get().build(), serializer)
    }

    private suspend fun <T> delete(
        path: String,
        serializer: kotlinx.serialization.KSerializer<T>
    ): ApiResult<T> = withContext(Dispatchers.IO) {
        execute(Request.Builder().url(base + path).delete().build(), serializer)
    }

    private suspend fun <T> postEmpty(
        path: String,
        serializer: kotlinx.serialization.KSerializer<T>
    ): ApiResult<T> = withContext(Dispatchers.IO) {
        val body = "".toRequestBody("application/json".toMediaType())
        execute(
            Request.Builder().url(base + path).post(body).build(),
            serializer
        )
    }

    private suspend fun <Req, Res> postJson(
        path: String,
        req: Req,
        reqSerializer: kotlinx.serialization.KSerializer<Req>,
        resSerializer: kotlinx.serialization.KSerializer<Res>
    ): ApiResult<Res> = withContext(Dispatchers.IO) {
        val body = json.encodeToString(reqSerializer, req)
            .toRequestBody("application/json".toMediaType())
        execute(
            Request.Builder().url(base + path).post(body).build(),
            resSerializer
        )
    }

    private fun <T> execute(
        request: Request,
        serializer: kotlinx.serialization.KSerializer<T>
    ): ApiResult<T> {
        return try {
            http.newCall(request).execute().use { response ->
                val bodyString = response.body?.string().orEmpty()
                if (!response.isSuccessful) {
                    val err = runCatching {
                        json.decodeFromString(ErrorResponse.serializer(), bodyString)
                    }.getOrNull()
                    ApiResult.Err(
                        errorCode = err?.errorCode ?: response.code,
                        message = err?.message ?: "HTTP ${response.code}"
                    )
                } else {
                    try {
                        ApiResult.Ok(json.decodeFromString(serializer, bodyString))
                    } catch (t: Throwable) {
                        ApiResult.Err(-1, "bad response: ${t.message}")
                    }
                }
            }
        } catch (io: IOException) {
            ApiResult.Err(-1, "network error: ${io.message}")
        }
    }
}

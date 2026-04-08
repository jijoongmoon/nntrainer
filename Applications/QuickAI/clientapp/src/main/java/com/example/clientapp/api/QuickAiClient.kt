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

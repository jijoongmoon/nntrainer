// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    RequestDispatcher.kt
 * @brief   Parses path + JSON bodies and routes to ModelRegistry /
 *          ModelWorker, then builds a JSON response. Transport-agnostic —
 *          HttpServer is the thin adapter that turns NanoHTTPD sessions
 *          into [Request] objects.
 */
package com.example.QuickAI.service

import android.util.Log
import com.example.QuickAI.service.backend.BackendResult
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * @brief A parsed incoming HTTP request.
 */
sealed class Request {
    object Health : Request()
    object Connect : Request()
    object ListModels : Request()
    data class SetOptions(val body: SetOptionsRequest) : Request()
    data class LoadModel(val body: LoadModelRequest) : Request()
    data class RunModel(val modelId: String, val body: RunModelRequest) : Request()
    data class GetMetrics(val modelId: String) : Request()
    data class UnloadModel(val modelId: String) : Request()
}

/**
 * @brief Result of dispatching a request — a JSON body plus HTTP status.
 */
data class Response(
    val status: Int,
    val jsonBody: String
)

/**
 * @brief The routing engine. Holds a reference to the one ModelRegistry
 * for the process and exposes a single [dispatch] entry point.
 */
class RequestDispatcher(
    private val registry: ModelRegistry,
    private val port: Int
) {
    private val json = Json {
        encodeDefaults = true
        ignoreUnknownKeys = true
        prettyPrint = false
    }

    /**
     * @brief Blocking per-request timeout. REST is synchronous in this
     * iteration (Architecture.md §10) so the HTTP thread waits up to this
     * long for the ModelWorker to finish.
     */
    private val requestTimeoutMs: Long = 5 * 60 * 1000L // 5 minutes

    fun dispatch(request: Request): Response = try {
        when (request) {
            Request.Health -> okJson(HealthResponse(status = "ok", port = port))

            Request.Connect -> okJson(
                ConnectResponse(connected = true, port = port, message = "connected")
            )

            Request.ListModels -> okJson(ListModelsResponse(registry.list()))

            is Request.SetOptions -> handleSetOptions(request.body)

            is Request.LoadModel -> handleLoadModel(request.body)

            is Request.RunModel -> handleRunModel(request.modelId, request.body)

            is Request.GetMetrics -> handleGetMetrics(request.modelId)

            is Request.UnloadModel -> handleUnload(request.modelId)
        }
    } catch (t: Throwable) {
        Log.e(TAG, "dispatch threw", t)
        errorJson(500, QuickAiError.UNKNOWN, t.message ?: "internal error")
    }

    /* ---------------- setOptions ---------------- */

    private fun handleSetOptions(body: SetOptionsRequest): Response {
        // Options live in native globals so we must make sure the native
        // library is loaded before we try to set them. If not available
        // (e.g. during UI-only development) we silently accept — real
        // model loads will fail anyway.
        if (NativeCausalLm.ensureLoaded()) {
            try {
                val ec = NativeCausalLm.setOptionsNative(
                    body.useChatTemplate, body.debugMode, body.verbose
                )
                return okJson(SetOptionsResponse(errorCode = ec))
            } catch (t: Throwable) {
                Log.e(TAG, "setOptionsNative threw", t)
            }
        }
        return okJson(SetOptionsResponse(errorCode = QuickAiError.NOT_INITIALIZED.code))
    }

    /* ---------------- loadModel ---------------- */

    private fun handleLoadModel(body: LoadModelRequest): Response {
        return when (val r = registry.getOrLoad(body)) {
            is BackendResult.Ok -> okJson(
                LoadModelResponse(
                    modelId = r.value.modelId,
                    architecture = r.value.architecture,
                    errorCode = 0
                )
            )
            is BackendResult.Err -> errorJson(
                httpStatusFor(r.error),
                r.error,
                r.message ?: "load failed"
            )
        }
    }

    /* ---------------- runModel ---------------- */

    private fun handleRunModel(modelId: String, body: RunModelRequest): Response {
        val worker = registry.get(modelId)
            ?: return errorJson(404, QuickAiError.MODEL_NOT_FOUND, "no such model: $modelId")

        val latch = CountDownLatch(1)
        var outcome: BackendResult<String> = BackendResult.Err(QuickAiError.UNKNOWN)

        val accepted = worker.submitRun(body.prompt) { res ->
            outcome = res
            latch.countDown()
        }
        if (!accepted) {
            return errorJson(503, QuickAiError.QUEUE_FULL, "worker queue full")
        }
        if (!latch.await(requestTimeoutMs, TimeUnit.MILLISECONDS)) {
            return errorJson(504, QuickAiError.UNKNOWN, "worker timeout")
        }

        return when (val r = outcome) {
            is BackendResult.Ok -> okJson(
                RunModelResponse(output = r.value, errorCode = 0)
            )
            is BackendResult.Err -> errorJson(
                httpStatusFor(r.error),
                r.error,
                r.message ?: "inference failed"
            )
        }
    }

    /* ---------------- metrics ---------------- */

    private fun handleGetMetrics(modelId: String): Response {
        val worker = registry.get(modelId)
            ?: return errorJson(404, QuickAiError.MODEL_NOT_FOUND, "no such model: $modelId")

        val latch = CountDownLatch(1)
        var outcome: BackendResult<PerformanceMetrics> =
            BackendResult.Err(QuickAiError.UNKNOWN)

        val accepted = worker.submitMetrics { res ->
            outcome = res
            latch.countDown()
        }
        if (!accepted) {
            return errorJson(503, QuickAiError.QUEUE_FULL, "worker queue full")
        }
        if (!latch.await(30_000, TimeUnit.MILLISECONDS)) {
            return errorJson(504, QuickAiError.UNKNOWN, "worker timeout")
        }

        return when (val r = outcome) {
            is BackendResult.Ok -> okJson(
                PerformanceMetricsResponse(metrics = r.value, errorCode = 0)
            )
            is BackendResult.Err -> errorJson(
                httpStatusFor(r.error),
                r.error,
                r.message ?: "metrics failed"
            )
        }
    }

    /* ---------------- unload ---------------- */

    private fun handleUnload(modelId: String): Response {
        val removed = registry.unload(modelId)
        return if (removed) {
            okJson(SetOptionsResponse(errorCode = 0))
        } else {
            errorJson(404, QuickAiError.MODEL_NOT_FOUND, "no such model: $modelId")
        }
    }

    /* ---------------- helpers ---------------- */

    private inline fun <reified T> okJson(body: T): Response =
        Response(200, json.encodeToString(body))

    private fun errorJson(status: Int, error: QuickAiError, message: String): Response =
        Response(
            status,
            json.encodeToString(
                ErrorResponse(errorCode = error.code, message = message)
            )
        )

    private fun httpStatusFor(error: QuickAiError): Int = when (error) {
        QuickAiError.NONE -> 200
        QuickAiError.INVALID_PARAMETER,
        QuickAiError.BAD_REQUEST -> 400
        QuickAiError.MODEL_NOT_FOUND -> 404
        QuickAiError.QUEUE_FULL -> 503
        QuickAiError.UNSUPPORTED -> 501
        QuickAiError.NOT_INITIALIZED,
        QuickAiError.INFERENCE_NOT_RUN,
        QuickAiError.MODEL_LOAD_FAILED,
        QuickAiError.INFERENCE_FAILED,
        QuickAiError.UNKNOWN -> 500
    }

    companion object {
        private const val TAG = "RequestDispatcher"
    }
}

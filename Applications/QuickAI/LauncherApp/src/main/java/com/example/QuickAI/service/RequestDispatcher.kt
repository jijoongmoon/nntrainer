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
import com.example.quickdotai.BackendResult
import com.example.quickdotai.NativeCausalLm
import java.io.InputStream
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
    data class RunModelStream(val modelId: String, val body: RunModelRequest) : Request()
    data class GetMetrics(val modelId: String) : Request()
    data class UnloadModel(val modelId: String) : Request()

    // Chat session requests
    data class ChatOpen(val modelId: String, val body: ChatOpenRequest) : Request()
    data class ChatRun(val modelId: String, val body: ChatRunRequest) : Request()
    data class ChatRunStream(val modelId: String, val body: ChatRunRequest) : Request()
    data class ChatCancel(val modelId: String, val body: ChatSessionIdRequest) : Request()
    data class ChatRebuild(val modelId: String, val body: ChatRebuildRequest) : Request()
    data class ChatClose(val modelId: String, val body: ChatSessionIdRequest) : Request()
}

/**
 * @brief Result of dispatching a request.
 *
 * Two shapes:
 *  - [Json] — a fully buffered JSON response, written with a Content-Length.
 *  - [Chunked] — an open-ended stream the server should write with HTTP
 *    chunked transfer encoding. Used by `POST /v1/models/{id}/run_stream`
 *    (Architecture.md §5.1). The [body] InputStream blocks on its backing
 *    queue until the backend emits bytes or closes the stream with EOF.
 */
sealed class Response {
    abstract val status: Int

    data class Json(
        override val status: Int,
        val jsonBody: String
    ) : Response()

    data class Chunked(
        override val status: Int,
        val contentType: String,
        val body: InputStream
    ) : Response()
}

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

            is Request.RunModelStream -> handleRunStream(request.modelId, request.body)

            is Request.GetMetrics -> handleGetMetrics(request.modelId)

            is Request.UnloadModel -> handleUnload(request.modelId)

            is Request.ChatOpen -> handleChatOpen(request.modelId, request.body)
            is Request.ChatRun -> handleChatRun(request.modelId, request.body)
            is Request.ChatRunStream -> handleChatRunStream(request.modelId, request.body)
            is Request.ChatCancel -> handleChatCancel(request.modelId, request.body)
            is Request.ChatRebuild -> handleChatRebuild(request.modelId, request.body)
            is Request.ChatClose -> handleChatClose(request.modelId, request.body)
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

    /* ---------------- runModel (streaming) ---------------- */

    /**
     * @brief Entry point for `POST /v1/models/{id}/run_stream`.
     *
     * Unlike the blocking [handleRunModel] path, this handler does NOT
     * wait for inference to finish. Instead it:
     *  1. Creates a [ChunkedStreamSink] whose [InputStream] end is wired
     *     into the HTTP chunked response, and
     *  2. Enqueues a [Job.RunStream] on the target worker that drives the
     *     backend and writes deltas into the sink.
     *
     * The HTTP server thread returns the [Response.Chunked] immediately
     * and NanoHTTPD streams bytes from the sink's InputStream on its own
     * writer thread. Backpressure is provided by the sink's bounded
     * LinkedBlockingQueue. See Architecture.md §5.1.
     */
    private fun handleRunStream(modelId: String, body: RunModelRequest): Response {
        val worker = registry.get(modelId)
            ?: return errorJson(404, QuickAiError.MODEL_NOT_FOUND, "no such model: $modelId")

        val sink = ChunkedStreamSink()
        val accepted = worker.submitRunStream(body.prompt, sink)
        if (!accepted) {
            // Queue full — the sink was never handed to the worker so we
            // can fall back to a normal JSON 503.
            return errorJson(503, QuickAiError.QUEUE_FULL, "worker queue full")
        }
        return Response.Chunked(
            status = 200,
            contentType = "application/x-ndjson",
            body = sink.inputStream
        )
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

    /* ---------------- chat: open ---------------- */

    private fun handleChatOpen(modelId: String, body: ChatOpenRequest): Response {
        val worker = registry.get(modelId)
            ?: return errorJson(404, QuickAiError.MODEL_NOT_FOUND, "no such model: $modelId")

        val latch = CountDownLatch(1)
        var outcome: BackendResult<String> = BackendResult.Err(QuickAiError.UNKNOWN)

        val accepted = worker.submitChatOpen(body.config) { res ->
            outcome = res
            latch.countDown()
        }
        if (!accepted) return errorJson(503, QuickAiError.QUEUE_FULL, "worker queue full")
        if (!latch.await(requestTimeoutMs, TimeUnit.MILLISECONDS)) {
            return errorJson(504, QuickAiError.UNKNOWN, "worker timeout")
        }
        return when (val r = outcome) {
            is BackendResult.Ok -> okJson(
                ChatOpenResponse(sessionId = r.value, errorCode = 0)
            )
            is BackendResult.Err -> errorJson(
                httpStatusFor(r.error), r.error, r.message ?: "chat open failed"
            )
        }
    }

    /* ---------------- chat: run ---------------- */

    private fun handleChatRun(modelId: String, body: ChatRunRequest): Response {
        val worker = registry.get(modelId)
            ?: return errorJson(404, QuickAiError.MODEL_NOT_FOUND, "no such model: $modelId")

        val messages = try {
            body.messages.map { it.toDomain() }
        } catch (t: Throwable) {
            return errorJson(400, QuickAiError.BAD_REQUEST, "bad message: ${t.message}")
        }

        val latch = CountDownLatch(1)
        var outcome: BackendResult<com.example.quickdotai.QuickAiChatResult> =
            BackendResult.Err(QuickAiError.UNKNOWN)

        val accepted = worker.submitChatRun(body.sessionId, messages) { res ->
            outcome = res
            latch.countDown()
        }
        if (!accepted) return errorJson(503, QuickAiError.QUEUE_FULL, "worker queue full")
        if (!latch.await(requestTimeoutMs, TimeUnit.MILLISECONDS)) {
            return errorJson(504, QuickAiError.UNKNOWN, "worker timeout")
        }
        return when (val r = outcome) {
            is BackendResult.Ok -> okJson(
                ChatRunResponse(
                    content = r.value.content,
                    metrics = r.value.metrics,
                    errorCode = 0
                )
            )
            is BackendResult.Err -> errorJson(
                httpStatusFor(r.error), r.error, r.message ?: "chat run failed"
            )
        }
    }

    /* ---------------- chat: run_stream ---------------- */

    private fun handleChatRunStream(modelId: String, body: ChatRunRequest): Response {
        val worker = registry.get(modelId)
            ?: return errorJson(404, QuickAiError.MODEL_NOT_FOUND, "no such model: $modelId")

        val messages = try {
            body.messages.map { it.toDomain() }
        } catch (t: Throwable) {
            return errorJson(400, QuickAiError.BAD_REQUEST, "bad message: ${t.message}")
        }

        val sink = ChunkedStreamSink()
        val accepted = worker.submitChatRunStream(body.sessionId, messages, sink)
        if (!accepted) return errorJson(503, QuickAiError.QUEUE_FULL, "worker queue full")
        return Response.Chunked(
            status = 200,
            contentType = "application/x-ndjson",
            body = sink.inputStream
        )
    }

    /* ---------------- chat: cancel ---------------- */

    private fun handleChatCancel(modelId: String, body: ChatSessionIdRequest): Response {
        val worker = registry.get(modelId)
            ?: return errorJson(404, QuickAiError.MODEL_NOT_FOUND, "no such model: $modelId")

        val latch = CountDownLatch(1)
        var outcome: BackendResult<Unit> = BackendResult.Err(QuickAiError.UNKNOWN)

        val accepted = worker.submitChatCancel(body.sessionId) { res ->
            outcome = res
            latch.countDown()
        }
        if (!accepted) return errorJson(503, QuickAiError.QUEUE_FULL, "worker queue full")
        if (!latch.await(requestTimeoutMs, TimeUnit.MILLISECONDS)) {
            return errorJson(504, QuickAiError.UNKNOWN, "worker timeout")
        }
        return when (val r = outcome) {
            is BackendResult.Ok -> okJson(ChatGenericResponse(errorCode = 0))
            is BackendResult.Err -> errorJson(
                httpStatusFor(r.error), r.error, r.message ?: "chat cancel failed"
            )
        }
    }

    /* ---------------- chat: rebuild ---------------- */

    private fun handleChatRebuild(modelId: String, body: ChatRebuildRequest): Response {
        val worker = registry.get(modelId)
            ?: return errorJson(404, QuickAiError.MODEL_NOT_FOUND, "no such model: $modelId")

        val messages = try {
            body.messages.map { it.toDomain() }
        } catch (t: Throwable) {
            return errorJson(400, QuickAiError.BAD_REQUEST, "bad message: ${t.message}")
        }

        val latch = CountDownLatch(1)
        var outcome: BackendResult<Unit> = BackendResult.Err(QuickAiError.UNKNOWN)

        val accepted = worker.submitChatRebuild(body.sessionId, messages) { res ->
            outcome = res
            latch.countDown()
        }
        if (!accepted) return errorJson(503, QuickAiError.QUEUE_FULL, "worker queue full")
        if (!latch.await(requestTimeoutMs, TimeUnit.MILLISECONDS)) {
            return errorJson(504, QuickAiError.UNKNOWN, "worker timeout")
        }
        return when (val r = outcome) {
            is BackendResult.Ok -> okJson(ChatGenericResponse(errorCode = 0))
            is BackendResult.Err -> errorJson(
                httpStatusFor(r.error), r.error, r.message ?: "chat rebuild failed"
            )
        }
    }

    /* ---------------- chat: close ---------------- */

    private fun handleChatClose(modelId: String, body: ChatSessionIdRequest): Response {
        val worker = registry.get(modelId)
            ?: return errorJson(404, QuickAiError.MODEL_NOT_FOUND, "no such model: $modelId")

        val latch = CountDownLatch(1)
        var outcome: BackendResult<Unit> = BackendResult.Err(QuickAiError.UNKNOWN)

        val accepted = worker.submitChatClose(body.sessionId) { res ->
            outcome = res
            latch.countDown()
        }
        if (!accepted) return errorJson(503, QuickAiError.QUEUE_FULL, "worker queue full")
        if (!latch.await(requestTimeoutMs, TimeUnit.MILLISECONDS)) {
            return errorJson(504, QuickAiError.UNKNOWN, "worker timeout")
        }
        return when (val r = outcome) {
            is BackendResult.Ok -> okJson(ChatGenericResponse(errorCode = 0))
            is BackendResult.Err -> errorJson(
                httpStatusFor(r.error), r.error, r.message ?: "chat close failed"
            )
        }
    }

    /* ---------------- helpers ---------------- */

    /**
     * @brief Convert a wire-format [ChatMessageDto] to the domain
     * [com.example.quickdotai.QuickAiChatMessage].
     */
    private fun ChatMessageDto.toDomain(): com.example.quickdotai.QuickAiChatMessage {
        val chatRole = when (role.lowercase()) {
            "system" -> com.example.quickdotai.QuickAiChatRole.SYSTEM
            "user" -> com.example.quickdotai.QuickAiChatRole.USER
            "assistant" -> com.example.quickdotai.QuickAiChatRole.ASSISTANT
            else -> throw IllegalArgumentException("unknown role: $role")
        }
        val domainParts = parts.map { p ->
            when (p.type) {
                "text" -> com.example.quickdotai.PromptPart.Text(
                    p.text ?: throw IllegalArgumentException("text part missing 'text'")
                )
                "image_file" -> com.example.quickdotai.PromptPart.ImageFile(
                    p.absolutePath
                        ?: throw IllegalArgumentException("image_file part missing 'absolute_path'")
                )
                "image_bytes" -> {
                    val b64 = p.imageBase64
                        ?: throw IllegalArgumentException("image_bytes part missing 'image_base64'")
                    com.example.quickdotai.PromptPart.ImageBytes(
                        android.util.Base64.decode(b64, android.util.Base64.DEFAULT)
                    )
                }
                else -> throw IllegalArgumentException("unknown part type: ${p.type}")
            }
        }
        return com.example.quickdotai.QuickAiChatMessage(
            role = chatRole,
            parts = domainParts
        )
    }

    private inline fun <reified T> okJson(body: T): Response =
        Response.Json(200, json.encodeToString(body))

    private fun errorJson(status: Int, error: QuickAiError, message: String): Response =
        Response.Json(
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

// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    HttpServer.kt
 * @brief   NanoHTTPD-backed loopback REST server embedded in
 *          QuickAIService. Binds to 127.0.0.1 only — it is not reachable
 *          from the network.
 *
 * The server is a thin adapter: it parses the path / method / body of
 * each NanoHTTPD session, constructs a [Request], hands it to
 * [RequestDispatcher], and writes the returned [Response] back.
 */
package com.example.QuickAI.service

import android.util.Log
import fi.iki.elonen.NanoHTTPD
import java.net.BindException
import java.net.InetAddress
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json

class HttpServer(
    private val hostname: String = "127.0.0.1",
    private val preferredPort: Int = DEFAULT_QUICKAI_PORT,
    private val registry: ModelRegistry
) {
    private var nano: NanoHTTPD? = null
    private var dispatcher: RequestDispatcher? = null

    @Volatile
    var boundPort: Int = -1
        private set

    private val json = Json {
        encodeDefaults = true
        ignoreUnknownKeys = true
    }

    /**
     * @brief Bind and start the server.
     *
     * Tries [preferredPort] first; if it's already in use, falls back to
     * an OS-picked ephemeral port. Returns the actually-bound port on
     * success or -1 on failure.
     */
    @Synchronized
    fun start(): Int {
        if (nano != null) return boundPort

        val tryPorts = intArrayOf(preferredPort, 0) // 0 = ephemeral
        for (port in tryPorts) {
            val candidate = try {
                LoopbackNanoHttpd(hostname, port, this::handleSession)
                    .also { it.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false) }
            } catch (e: BindException) {
                Log.w(TAG, "Port $port already in use, trying next", e)
                null
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to start NanoHTTPD on $hostname:$port", t)
                null
            }
            if (candidate != null) {
                nano = candidate
                boundPort = candidate.listeningPort
                dispatcher = RequestDispatcher(registry, boundPort)
                Log.i(TAG, "QuickAI HTTP server listening on $hostname:$boundPort")
                return boundPort
            }
        }
        return -1
    }

    @Synchronized
    fun stop() {
        try {
            nano?.stop()
        } catch (t: Throwable) {
            Log.w(TAG, "NanoHTTPD stop threw", t)
        }
        nano = null
        dispatcher = null
        boundPort = -1
    }

    // --- request routing ----------------------------------------------

    private fun handleSession(session: NanoHTTPD.IHTTPSession): NanoHTTPD.Response {
        val dispatcher = this.dispatcher
            ?: return badRequest("server not initialized")

        return try {
            val req = parseRequest(session)
                ?: return badRequest("unsupported path: ${session.method} ${session.uri}")
            when (val resp = dispatcher.dispatch(req)) {
                is Response.Json -> NanoHTTPD.newFixedLengthResponse(
                    statusOf(resp.status),
                    "application/json",
                    resp.jsonBody
                )
                is Response.Chunked -> NanoHTTPD.newChunkedResponse(
                    statusOf(resp.status),
                    resp.contentType,
                    resp.body
                )
            }
        } catch (t: Throwable) {
            Log.e(TAG, "session handling threw", t)
            NanoHTTPD.newFixedLengthResponse(
                NanoHTTPD.Response.Status.INTERNAL_ERROR,
                "application/json",
                """{"error_code":99,"message":"${t.message?.escapeJson() ?: ""}"}"""
            )
        }
    }

    private fun parseRequest(session: NanoHTTPD.IHTTPSession): Request? {
        val method = session.method
        val uri = session.uri.trimEnd('/')

        // GET /v1/health
        if (method == NanoHTTPD.Method.GET && uri == "/v1/health") {
            return Request.Health
        }
        // POST /v1/connect — explicit, client-initiated handshake. Body
        // is ignored; semantics are "confirm you can reach me".
        if (method == NanoHTTPD.Method.POST && uri == "/v1/connect") {
            return Request.Connect
        }
        // GET /v1/models
        if (method == NanoHTTPD.Method.GET && uri == "/v1/models") {
            return Request.ListModels
        }
        // POST /v1/options
        if (method == NanoHTTPD.Method.POST && uri == "/v1/options") {
            val body = readBody(session)
            return Request.SetOptions(json.decodeFromString<SetOptionsRequest>(body))
        }
        // POST /v1/models   (load)
        if (method == NanoHTTPD.Method.POST && uri == "/v1/models") {
            val body = readBody(session)
            return Request.LoadModel(json.decodeFromString<LoadModelRequest>(body))
        }
        // POST /v1/models/{id}/run
        Regex("""^/v1/models/([^/]+)/run$""").matchEntire(uri)?.let { m ->
            if (method == NanoHTTPD.Method.POST) {
                val body = readBody(session)
                return Request.RunModel(
                    modelId = m.groupValues[1],
                    body = json.decodeFromString<RunModelRequest>(body)
                )
            }
        }
        // POST /v1/models/{id}/run_stream — NDJSON chunked stream
        Regex("""^/v1/models/([^/]+)/run_stream$""").matchEntire(uri)?.let { m ->
            if (method == NanoHTTPD.Method.POST) {
                val body = readBody(session)
                return Request.RunModelStream(
                    modelId = m.groupValues[1],
                    body = json.decodeFromString<RunModelRequest>(body)
                )
            }
        }
        // GET /v1/models/{id}/metrics
        Regex("""^/v1/models/([^/]+)/metrics$""").matchEntire(uri)?.let { m ->
            if (method == NanoHTTPD.Method.GET) {
                return Request.GetMetrics(m.groupValues[1])
            }
        }
        // DELETE /v1/models/{id}
        Regex("""^/v1/models/([^/]+)$""").matchEntire(uri)?.let { m ->
            if (method == NanoHTTPD.Method.DELETE) {
                return Request.UnloadModel(m.groupValues[1])
            }
        }

        // --- Chat session endpoints ---

        // POST /v1/models/{id}/chat/open
        Regex("""^/v1/models/([^/]+)/chat/open$""").matchEntire(uri)?.let { m ->
            if (method == NanoHTTPD.Method.POST) {
                val body = readBody(session)
                val req = if (body.isBlank()) ChatOpenRequest()
                    else json.decodeFromString<ChatOpenRequest>(body)
                return Request.ChatOpen(m.groupValues[1], req)
            }
        }
        // POST /v1/models/{id}/chat/run
        Regex("""^/v1/models/([^/]+)/chat/run$""").matchEntire(uri)?.let { m ->
            if (method == NanoHTTPD.Method.POST) {
                val body = readBody(session)
                return Request.ChatRun(
                    m.groupValues[1],
                    json.decodeFromString<ChatRunRequest>(body)
                )
            }
        }
        // POST /v1/models/{id}/chat/run_stream
        Regex("""^/v1/models/([^/]+)/chat/run_stream$""").matchEntire(uri)?.let { m ->
            if (method == NanoHTTPD.Method.POST) {
                val body = readBody(session)
                return Request.ChatRunStream(
                    m.groupValues[1],
                    json.decodeFromString<ChatRunRequest>(body)
                )
            }
        }
        // POST /v1/models/{id}/chat/cancel
        Regex("""^/v1/models/([^/]+)/chat/cancel$""").matchEntire(uri)?.let { m ->
            if (method == NanoHTTPD.Method.POST) {
                val body = readBody(session)
                return Request.ChatCancel(
                    m.groupValues[1],
                    json.decodeFromString<ChatSessionIdRequest>(body)
                )
            }
        }
        // POST /v1/models/{id}/chat/rebuild
        Regex("""^/v1/models/([^/]+)/chat/rebuild$""").matchEntire(uri)?.let { m ->
            if (method == NanoHTTPD.Method.POST) {
                val body = readBody(session)
                return Request.ChatRebuild(
                    m.groupValues[1],
                    json.decodeFromString<ChatRebuildRequest>(body)
                )
            }
        }
        // POST /v1/models/{id}/chat/close
        Regex("""^/v1/models/([^/]+)/chat/close$""").matchEntire(uri)?.let { m ->
            if (method == NanoHTTPD.Method.POST) {
                val body = readBody(session)
                return Request.ChatClose(
                    m.groupValues[1],
                    json.decodeFromString<ChatSessionIdRequest>(body)
                )
            }
        }

        return null
    }

    private fun readBody(session: NanoHTTPD.IHTTPSession): String {
        // NanoHTTPD requires us to call parseBody for POST/PUT to make the
        // raw body available via getInputStream/getBody. We funnel the raw
        // bytes through a map.
        val map = hashMapOf<String, String>()
        try {
            session.parseBody(map)
        } catch (t: Throwable) {
            Log.w(TAG, "parseBody threw", t)
        }
        return map["postData"] ?: ""
    }

    private fun statusOf(code: Int): NanoHTTPD.Response.IStatus =
        NanoHTTPD.Response.Status.values().firstOrNull { it.requestStatus == code }
            ?: NanoHTTPD.Response.Status.INTERNAL_ERROR

    private fun badRequest(message: String): NanoHTTPD.Response =
        NanoHTTPD.newFixedLengthResponse(
            NanoHTTPD.Response.Status.BAD_REQUEST,
            "application/json",
            """{"error_code":${QuickAiError.BAD_REQUEST.code},"message":"${message.escapeJson()}"}"""
        )

    private fun String.escapeJson(): String =
        replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")

    /**
     * @brief NanoHTTPD subclass that binds to a specific interface and
     * routes every request through the supplied [handler] lambda.
     */
    private class LoopbackNanoHttpd(
        hostname: String,
        port: Int,
        private val handler: (IHTTPSession) -> Response
    ) : NanoHTTPD(hostname, port) {
        override fun serve(session: IHTTPSession): Response = handler(session)
    }

    companion object {
        private const val TAG = "QuickAiHttpServer"
    }
}

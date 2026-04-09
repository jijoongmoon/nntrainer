// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    Backend.kt
 * @brief   Abstraction over a loaded model instance.
 *
 * Two implementations exist:
 *  - NativeCausalLmBackend: all non-Gemma4 models, routed through JNI to
 *    libcausallm_api.so.
 *  - LiteRtLmBackend: Gemma4 only, routed through the LiteRT-LM Kotlin API.
 *
 * A Backend instance is owned by exactly one ModelWorker and therefore
 * accessed by exactly one thread at a time — implementations do NOT need
 * to be internally thread-safe.
 */
package com.example.QuickAI.service.backend

import com.example.QuickAI.service.LoadModelRequest
import com.example.QuickAI.service.PerformanceMetrics
import com.example.QuickAI.service.QuickAiError

/**
 * @brief Outcome of a backend call.
 */
sealed class BackendResult<out T> {
    data class Ok<T>(val value: T) : BackendResult<T>()
    data class Err(val error: QuickAiError, val message: String? = null) : BackendResult<Nothing>()
}

/**
 * @brief Where a backend pushes streamed output during [Backend.runStreaming].
 *
 * The contract is:
 *  - zero or more [onDelta] calls carrying newly-generated text, followed by
 *  - exactly one terminal call — either [onDone] on success or [onError] on
 *    failure.
 *
 * Implementations (see ChunkedStreamSink) are expected to be thread-safe
 * because LiteRT-LM may invoke the backend's MessageCallback from its own
 * internal thread, not the ModelWorker thread.
 *
 * See Architecture.md §5.1.
 */
interface StreamSink {
    fun onDelta(text: String)
    fun onDone()
    fun onError(error: QuickAiError, message: String?)
}

/**
 * @brief Common interface implemented by every backend.
 */
interface Backend {
    /** @return a short identifier like "native" or "litert-lm". */
    val kind: String

    /** @return the architecture string reported by the engine, if any. */
    val architecture: String?

    /**
     * @brief Load the model described by [req]. Must be called exactly
     * once in the worker thread before any [run] call.
     */
    fun load(req: LoadModelRequest): BackendResult<Unit>

    /**
     * @brief Run inference on a prompt.
     */
    fun run(prompt: String): BackendResult<String>

    /**
     * @brief Streaming variant of [run]. The default implementation simply
     * calls [run] and emits the whole string as a single delta, so backends
     * without a native streaming path (the current native causal_lm_api
     * backend) automatically work through /v1/models/{id}/run_stream —
     * they just emit one big chunk instead of many small ones.
     *
     * Streaming-capable backends (LiteRtLmBackend) override this to push
     * progressive deltas through [sink] as tokens arrive.
     *
     * Contract: on return, exactly one of [StreamSink.onDone] or
     * [StreamSink.onError] MUST have been delivered. The returned
     * BackendResult mirrors the terminal state for the caller's
     * convenience.
     */
    fun runStreaming(prompt: String, sink: StreamSink): BackendResult<Unit> {
        return when (val r = run(prompt)) {
            is BackendResult.Ok -> {
                if (r.value.isNotEmpty()) sink.onDelta(r.value)
                sink.onDone()
                BackendResult.Ok(Unit)
            }
            is BackendResult.Err -> {
                sink.onError(r.error, r.message)
                r
            }
        }
    }

    /**
     * @brief Fetch metrics from the last run.
     */
    fun metrics(): BackendResult<PerformanceMetrics>

    /**
     * @brief Release all resources. Called by ModelWorker on shutdown.
     */
    fun close()
}

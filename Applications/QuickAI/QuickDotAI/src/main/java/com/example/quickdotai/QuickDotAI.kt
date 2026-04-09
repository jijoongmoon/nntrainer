// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    QuickDotAI.kt
 * @brief   Public surface of the QuickDotAI AAR.
 *
 * QuickDotAI is a thin abstraction over a single loaded on-device
 * language model. Two concrete implementations are shipped in this AAR:
 *
 *  - [NativeQuickDotAI] — routes non-Gemma models through JNI to
 *    libcausallm_api.so, the handle-based C API built from
 *    Applications/CausalLM.
 *  - [LiteRTLm]         — routes Gemma-family models through the
 *    LiteRT-LM Kotlin API.
 *
 * Both implementations satisfy the same [QuickDotAI] contract so a host
 * app can pick an engine once at load time and then drive it through a
 * single interface for run / runStreaming / metrics / close.
 *
 * Threading: a [QuickDotAI] instance is NOT internally thread-safe. The
 * expectation is that the host app owns exactly one instance per loaded
 * model and drives it from a single worker thread — the same contract
 * that QuickAIService's ModelWorker implements, and the one the sample
 * app (SampleTestAPP) follows from its background dispatcher.
 */
package com.example.quickdotai

/**
 * @brief Outcome of a QuickDotAI call.
 *
 * Every public method returns a [BackendResult] so errors never
 * propagate out as exceptions across the AAR boundary. [Ok] carries the
 * successful value; [Err] carries a [QuickAiError] code and an optional
 * human-readable message.
 */
sealed class BackendResult<out T> {
    data class Ok<T>(val value: T) : BackendResult<T>()
    data class Err(
        val error: QuickAiError,
        val message: String? = null
    ) : BackendResult<Nothing>()
}

/**
 * @brief Where a [QuickDotAI] implementation pushes streamed output
 *        during [QuickDotAI.runStreaming].
 *
 * The contract is:
 *  - zero or more [onDelta] calls carrying newly-generated text,
 *    followed by
 *  - exactly one terminal call — either [onDone] on success or
 *    [onError] on failure.
 *
 * Implementations may be invoked from an implementation-internal
 * thread (LiteRT-LM for example dispatches MessageCallback on its own
 * worker thread). Host code that wants to marshal events back to the UI
 * thread must do that bridging itself — the AAR does not assume any
 * particular threading model on the consumer side.
 */
interface StreamSink {
    fun onDelta(text: String)
    fun onDone()
    fun onError(error: QuickAiError, message: String?)
}

/**
 * @brief Common interface implemented by every QuickDotAI engine.
 *
 * Lifecycle: [load] exactly once, then any number of [run] /
 * [runStreaming] / [metrics] calls, then [close] exactly once. Calling
 * [run] before [load] returns a [BackendResult.Err] with
 * [QuickAiError.NOT_INITIALIZED].
 */
interface QuickDotAI {
    /** @return a short identifier like "native" or "litert-lm". */
    val kind: String

    /** @return the architecture string reported by the engine, if any. */
    val architecture: String?

    /**
     * @brief Load the model described by [req]. Must be called exactly
     * once before any [run] or [runStreaming] call.
     */
    fun load(req: LoadModelRequest): BackendResult<Unit>

    /**
     * @brief Blocking inference on a single prompt.
     *
     * Returns the full decoded generation on success, or a
     * [BackendResult.Err] on failure.
     */
    fun run(prompt: String): BackendResult<String>

    /**
     * @brief Streaming variant of [run].
     *
     * Default implementation simply calls [run] and emits the whole
     * string as a single delta, so engines without a native streaming
     * path still work — they just emit one big chunk instead of many
     * small ones.
     *
     * Streaming-capable engines (both [LiteRTLm] and [NativeQuickDotAI]
     * in this AAR) override this to push progressive deltas through
     * [sink] as tokens are decoded.
     *
     * Contract: on return, exactly one of [StreamSink.onDone] or
     * [StreamSink.onError] MUST have been delivered. The returned
     * [BackendResult] mirrors the terminal state for the caller's
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
     * @brief Fetch performance metrics for the most recent run.
     */
    fun metrics(): BackendResult<PerformanceMetrics>

    /**
     * @brief Release all resources. Idempotent — safe to call more
     * than once.
     */
    fun close()
}

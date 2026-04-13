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
 * @brief A live chat session created by [QuickDotAI.openChatSession].
 *
 * The session accumulates conversation history internally. Callers
 * send new [QuickAiChatMessage]s and the backend appends them plus
 * the assistant reply to the running history. Multiple sessions may
 * coexist on the same loaded model.
 *
 * Threading: a session is NOT internally thread-safe. The host must
 * drive it from a single worker thread — the same contract as the
 * owning [QuickDotAI] instance. [cancel] is the only method safe to
 * call from an external thread.
 */
interface QuickAiChatSession : AutoCloseable {
    /** Unique identifier for this session. */
    val sessionId: String

    /**
     * @brief Append [messages] to the conversation, run inference, and
     * return the assistant's reply. The new user messages AND the
     * assistant reply are persisted in the session's internal history.
     */
    fun run(messages: List<QuickAiChatMessage>): BackendResult<QuickAiChatResult>

    /**
     * @brief Streaming variant of [run]. Deltas are pushed through
     * [sink]; the full assistant reply is returned on completion and
     * also appended to the internal history.
     */
    fun runStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult>

    /**
     * @brief Cancel an in-flight [run] or [runStreaming]. Safe to call
     * from any thread.
     */
    fun cancel()

    /**
     * @brief Replace the entire conversation history and rebuild
     * internal engine state. Use this after history edits, sampling
     * changes, or to recover from a failed/cancelled turn.
     */
    fun rebuild(messages: List<QuickAiChatMessage>): BackendResult<Unit>

    /**
     * @brief Close the session, releasing its resources (conversation
     * handle, cached images, etc.). Idempotent.
     */
    override fun close()
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
     * @brief Multimodal variant of [run] — accepts a sequence of
     * [PromptPart]s that may interleave text and image inputs.
     *
     * The default implementation returns [QuickAiError.UNSUPPORTED]
     * because not every engine can handle non-text inputs. Concrete
     * implementations backed by multimodal-capable models (currently
     * [LiteRTLm] with a multimodal Gemma loaded through a non-null
     * [LoadModelRequest.visionBackend]) override this to do the real
     * work. [NativeQuickDotAI] inherits the UNSUPPORTED default, so
     * consumers get a clear error message instead of a silent failure
     * when they aim an image prompt at the text-only native engine.
     *
     * Contract:
     *  - [parts] must be non-empty; an empty list returns
     *    [QuickAiError.INVALID_PARAMETER].
     *  - Parts may appear in any order. The canonical Gemma-4 /
     *    Gemma3n convention is one or more image parts followed by a
     *    single trailing text instruction.
     *  - Must be called only after a successful [load]; calling it
     *    before [load] returns [QuickAiError.NOT_INITIALIZED].
     *
     * Example:
     * ```
     * val reply = engine.runMultimodal(listOf(
     *     PromptPart.ImageFile("/sdcard/photo.jpg"),
     *     PromptPart.Text("What is happening in this picture?"),
     * ))
     * ```
     */
    fun runMultimodal(parts: List<PromptPart>): BackendResult<String> =
        BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "runMultimodal is not supported by engine '$kind'. " +
                "Load a multimodal-capable model (e.g. GEMMA4) with " +
                "LoadModelRequest.visionBackend set to a non-null value."
        )

    /**
     * @brief Streaming variant of [runMultimodal].
     *
     * The default implementation returns [QuickAiError.UNSUPPORTED]
     * and delivers a single terminal [StreamSink.onError] before
     * returning, so callers can rely on the same StreamSink contract
     * as text-only streaming regardless of which engine they targeted.
     */
    fun runMultimodalStreaming(
        parts: List<PromptPart>,
        sink: StreamSink
    ): BackendResult<Unit> {
        val err = BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "runMultimodalStreaming is not supported by engine '$kind'. " +
                "Load a multimodal-capable model (e.g. GEMMA4) with " +
                "LoadModelRequest.visionBackend set to a non-null value."
        )
        sink.onError(err.error, err.message)
        return err
    }

    /**
     * @brief Unload the model weights without destroying the engine.
     *
     * After a successful unload the engine is in a "not initialized" state
     * — subsequent [run] / [runStreaming] / [metrics] calls will return
     * [QuickAiError.NOT_INITIALIZED]. The instance can still be [close]d
     * normally (and must be, to release any remaining resources).
     *
     * Implementations that do not support partial unload may treat this as
     * a full [close] or return [BackendResult.Ok] as a no-op.
     */
    fun unload(): BackendResult<Unit>

    /**
     * @brief Fetch performance metrics for the most recent run.
     */
    fun metrics(): BackendResult<PerformanceMetrics>

    /**
     * @brief Open a new structured chat session on this engine.
     *
     * Multiple sessions may coexist on the same loaded model. Each
     * session maintains its own conversation history and image cache.
     * The default implementation returns [QuickAiError.UNSUPPORTED];
     * concrete engines override this to provide session management.
     */
    fun openChatSession(
        config: QuickAiChatSessionConfig? = null
    ): BackendResult<QuickAiChatSession> =
        BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "openChatSession is not supported by engine '$kind'."
        )

    /**
     * @brief Release all resources. Idempotent — safe to call more
     * than once.
     */
    fun close()
}

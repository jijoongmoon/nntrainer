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
 *
 * **Chat session lifecycle:** [openChatSession] → [chatRun] /
 * [chatRunStreaming] / [chatCancel] / [chatRebuild] → [closeChatSession].
 * Only one session may be active at a time. While a chat session is
 * active, the flat [run] / [runStreaming] / [runMultimodal] APIs are
 * unavailable (their internal Conversation is released to the session).
 */
interface QuickDotAI {
    /** @return a short identifier like "native" or "litert-lm". */
    val kind: String

    /** @return the architecture string reported by the engine, if any. */
    val architecture: String?

    /**
     * @return the sessionId of the currently active chat session, or
     * null if no session is open.
     */
    val chatSessionId: String?
        get() = null

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

    // ----- Chat session API ------------------------------------------------
    // All chat operations go through this interface so the app never needs
    // to interact with chat session classes directly.

    /**
     * @brief Open a new structured chat session on this engine.
     *
     * Only **one** session may be active at a time (LiteRT-LM allows a
     * single Conversation per Engine). If a session is already open,
     * this method returns [QuickAiError.BAD_REQUEST]. Returns the
     * session ID on success.
     */
    fun openChatSession(
        config: QuickAiChatSessionConfig? = null
    ): BackendResult<String> =
        BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "openChatSession is not supported by engine '$kind'."
        )

    /**
     * @brief Close the active chat session, releasing its resources
     * (conversation handle, cached images, etc.). After closing, the
     * flat [run] / [runStreaming] APIs become usable again.
     */
    fun closeChatSession(): BackendResult<Unit> =
        BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "closeChatSession is not supported by engine '$kind'."
        )

    /**
     * @brief Send structured chat [messages] and return the assistant
     * reply. Messages and the reply are accumulated in the session's
     * internal history. Requires an active session opened via
     * [openChatSession].
     */
    fun chatRun(
        messages: List<QuickAiChatMessage>
    ): BackendResult<QuickAiChatResult> =
        BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "chatRun is not supported by engine '$kind'."
        )

    /**
     * @brief Streaming variant of [chatRun]. Deltas are pushed through
     * [sink]; the full assistant reply is returned on completion and
     * also appended to the internal history.
     */
    fun chatRunStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult> {
        val err = BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "chatRunStreaming is not supported by engine '$kind'."
        )
        sink.onError(err.error, err.message)
        return err
    }

    /**
     * @brief Cancel an in-flight [chatRun] or [chatRunStreaming].
     * Safe to call from any thread. No-op if no generation is running.
     */
    fun chatCancel() { /* no-op by default */ }

    /**
     * @brief Replace the active session's entire conversation history
     * and rebuild internal engine state. Use this after history edits,
     * sampling changes, or to recover from a failed/cancelled turn.
     */
    fun chatRebuild(
        messages: List<QuickAiChatMessage>
    ): BackendResult<Unit> =
        BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "chatRebuild is not supported by engine '$kind'."
        )

    /**
     * @brief Release all resources. Idempotent — safe to call more
     * than once.
     */
    fun close()
}

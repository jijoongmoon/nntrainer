// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    NativeChatSession.kt
 * @brief   Chat session helper for the native causal_lm backend.
 *
 * The native engine now manages its own KV cache and chat template, so
 * this wrapper no longer tracks conversation history in Kotlin.
 * It simply validates input, extracts the trailing USER turn, and
 * forwards it to the native handle. The native engine is responsible
 * for prompt formatting and state retention across turns.
 *
 * Lifecycle:
 *   openChatSession() -> run()/runStreaming() -> closeChatSession()
 *
 * Rebuild semantics (option D):
 *   rebuild() is a no-op at the Kotlin layer. The native engine handles
 *   KV cache resets internally. The system instruction supplied at
 *   session creation remains in effect.
 */
package com.example.quickdotai

import android.util.Log
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

internal class NativeChatSession(
    private val handleProvider: () -> Long,
    private val config: QuickAiChatSessionConfig? = null,
    val sessionId: String = UUID.randomUUID().toString()
) {

    private val cancelRequested = AtomicBoolean(false)

    @Volatile
    private var closed = false

    private var lastRunDurationMs: Double = 0.0

    init {
        config?.systemInstruction?.takeIf { it.isNotBlank() }?.let { sys ->
            Log.i(TAG, "NativeChatSession($sessionId): system instruction configured (${sys.length} chars)")
        }
    }

    fun run(
        messages: List<QuickAiChatMessage>
    ): BackendResult<QuickAiChatResult> {
        if (closed) return errClosed()

        val prep = prepareTurn(messages) ?: return lastPrepError
            ?: BackendResult.Err(QuickAiError.INVALID_PARAMETER, "invalid chat input")

        cancelRequested.set(false)

        val handle = handleProvider()
        if (handle == 0L) {
            return BackendResult.Err(QuickAiError.NOT_INITIALIZED, "Native handle is not available")
        }

        val prompt = extractText(prep.lastUser)

        return try {
            val startNs = System.nanoTime()
            val result = NativeCausalLm.runModelHandleNative(handle, prompt)
            lastRunDurationMs = (System.nanoTime() - startNs) / 1_000_000.0

            if (result.errorCode != 0) {
                Log.e(TAG, "run($sessionId): inference failed with errorCode=${result.errorCode}")
                BackendResult.Err(QuickAiError.fromNativeCode(result.errorCode))
            } else {
                val output = result.output.orEmpty()
                Log.i(TAG, "run($sessionId): completed in ${lastRunDurationMs.toLong()} ms, output length=${output.length}")
                BackendResult.Ok(
                    QuickAiChatResult(
                        content = output,
                        metrics = PerformanceMetrics(totalDurationMs = lastRunDurationMs)
                    )
                )
            }
        } catch (t: Throwable) {
            Log.e(TAG, "run($sessionId): threw exception", t)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    fun runStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult> {
        if (closed) {
            val err = errClosed()
            sink.onError(err.error, err.message)
            return err
        }

        val prep = prepareTurn(messages)
        if (prep == null) {
            val err = lastPrepError ?: BackendResult.Err(QuickAiError.INVALID_PARAMETER, "invalid chat input")
            sink.onError(err.error, err.message)
            return err
        }

        cancelRequested.set(false)

        val handle = handleProvider()
        if (handle == 0L) {
            val err = BackendResult.Err(QuickAiError.NOT_INITIALIZED, "Native handle is not available")
            sink.onError(err.error, err.message)
            return err
        }

        val prompt = extractText(prep.lastUser)
        val accumulated = StringBuilder()
        val startNs = System.nanoTime()

        return try {
            val errorCode = NativeCausalLm.runModelHandleStreamingNative(handle, prompt) { delta ->
                if (cancelRequested.get()) return@runModelHandleStreamingNative
                accumulated.append(delta)
                sink.onDelta(delta)
            }

            lastRunDurationMs = (System.nanoTime() - startNs) / 1_000_000.0

            if (errorCode != 0) {
                val err = QuickAiError.fromNativeCode(errorCode)
                Log.e(TAG, "runStreaming($sessionId): failed with errorCode=$errorCode")
                sink.onError(err, "Inference failed (errorCode=$errorCode)")
                BackendResult.Err(err, "Inference failed (errorCode=$errorCode)")
            } else {
                val output = accumulated.toString()
                Log.i(TAG, "runStreaming($sessionId): completed in ${lastRunDurationMs.toLong()} ms")
                sink.onDone()
                BackendResult.Ok(
                    QuickAiChatResult(
                        content = output,
                        metrics = PerformanceMetrics(totalDurationMs = lastRunDurationMs)
                    )
                )
            }
        } catch (t: Throwable) {
            Log.e(TAG, "runStreaming($sessionId): threw exception", t)
            sink.onError(QuickAiError.INFERENCE_FAILED, t.message)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    fun cancel() {
        if (closed) return
        cancelRequested.set(true)
        val handle = handleProvider()
        if (handle != 0L) {
            Log.i(TAG, "cancel($sessionId): requesting stop for handle=0x${handle.toString(16)}")
            NativeCausalLm.cancelModelHandleNative(handle)
        } else {
            Log.w(TAG, "cancel($sessionId): no valid handle to cancel")
        }
    }

    fun rebuild(
        messages: List<QuickAiChatMessage>
    ): BackendResult<Unit> {
        if (closed) return errClosed()

        Log.i(TAG, "rebuild($sessionId): no-op at Kotlin layer — native engine manages KV cache")

        // The native engine owns the KV cache. There is no local history to clear.
        // System instruction remains in [config] and is still in effect.
        // Callers who need a hard reset can close/open a new session instead.

        return BackendResult.Ok(Unit)
    }

    fun close() {
        if (closed) return
        closed = true
        Log.i(TAG, "close($sessionId): session closed")
    }

    private data class TurnPrep(val lastUser: QuickAiChatMessage)

    @Volatile
    private var lastPrepError: BackendResult.Err? = null

    private fun prepareTurn(messages: List<QuickAiChatMessage>): TurnPrep? {
        lastPrepError = null

        if (messages.isEmpty()) {
            lastPrepError = BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "messages list is empty"
            )
            return null
        }

        if (messages.last().role != QuickAiChatRole.USER) {
            lastPrepError = BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "last message must have role USER to trigger inference (got ${messages.last().role})"
            )
            return null
        }

        return TurnPrep(lastUser = messages.last())
    }

    private fun extractText(msg: QuickAiChatMessage): String =
        msg.parts.filterIsInstance<PromptPart.Text>().joinToString("") { it.text }

    private fun errClosed(): BackendResult.Err = BackendResult.Err(
        QuickAiError.NOT_INITIALIZED,
        "Chat session $sessionId is closed"
    )

    companion object {
        private const val TAG = "NativeChatSession"
    }
}

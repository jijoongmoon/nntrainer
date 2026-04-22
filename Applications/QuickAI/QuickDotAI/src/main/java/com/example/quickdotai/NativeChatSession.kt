// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    NativeChatSession.kt
 * @brief   Chat session helper for the native causal_lm backend.
 *
 * The native engine does not support KV cache reuse across turns, so this
 * wrapper manages conversation history in Kotlin and rebuilds the full
 * prompt on each turn. This means each turn is O(all tokens) rather than
 * O(new tokens), but it enables structured chat without native modifications.
 *
 * History ownership:
 * This wrapper tracks the full conversation history in [history]. Each
 * inference turn appends the user message and assistant response to the
 * history. The full history is rendered through a chat template before
 * being sent to the native engine.
 */
package com.example.quickdotai

import android.util.Log
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

internal class NativeChatSession(
    private val handleProvider: () -> Long,
    private val architectureProvider: () -> String?,
    private val config: QuickAiChatSessionConfig? = null,
    val sessionId: String = UUID.randomUUID().toString()
) {

    private val history: MutableList<QuickAiChatMessage> = mutableListOf()

    private val cancelRequested = AtomicBoolean(false)

    @Volatile
    private var closed = false

    private var lastRunDurationMs: Double = 0.0

    init {
        config?.systemInstruction?.takeIf { it.isNotBlank() }?.let { sys ->
            history.add(QuickAiChatMessage(role = QuickAiChatRole.SYSTEM, parts = listOf(PromptPart.Text(sys))))
            Log.i(TAG, "NativeChatSession($sessionId): added system instruction (${sys.length} chars)")
        }
    }

    fun run(
        messages: List<QuickAiChatMessage>
    ): BackendResult<QuickAiChatResult> {
        if (closed) return errClosed()

        val prep = prepareTurn(messages) ?: return lastPrepError
            ?: BackendResult.Err(QuickAiError.INVALID_PARAMETER, "invalid chat input")

        cancelRequested.set(false)

        val prompt = buildPromptFromHistory(prep.newMessages)
        Log.i(TAG, "run($sessionId): prompt length=${prompt.length}, history size=${history.size}")

        val handle = handleProvider()
        if (handle == 0L) {
            return BackendResult.Err(QuickAiError.NOT_INITIALIZED, "Native handle is not available")
        }

        return try {
            val startNs = System.nanoTime()
            val result = NativeCausalLm.runModelHandleNative(handle, prompt)
            lastRunDurationMs = (System.nanoTime() - startNs) / 1_000_000.0

            if (result.errorCode != 0) {
                Log.e(TAG, "run($sessionId): inference failed with errorCode=${result.errorCode}")
                BackendResult.Err(QuickAiError.fromNativeCode(result.errorCode))
            } else {
                val output = result.output.orEmpty()
                history.add(QuickAiChatMessage(role = QuickAiChatRole.ASSISTANT, parts = listOf(PromptPart.Text(output))))
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

        val prompt = buildPromptFromHistory(prep.newMessages)
        Log.i(TAG, "runStreaming($sessionId): prompt length=${prompt.length}, history size=${history.size}")

        val handle = handleProvider()
        if (handle == 0L) {
            val err = BackendResult.Err(QuickAiError.NOT_INITIALIZED, "Native handle is not available")
            sink.onError(err.error, err.message)
            return err
        }

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
                history.add(QuickAiChatMessage(role = QuickAiChatRole.ASSISTANT, parts = listOf(PromptPart.Text(output))))
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

        Log.i(TAG, "rebuild($sessionId): clearing history and seeding with ${messages.size} message(s)")

        history.clear()

        config?.systemInstruction?.takeIf { it.isNotBlank() }?.let { sys ->
            history.add(QuickAiChatMessage(role = QuickAiChatRole.SYSTEM, parts = listOf(PromptPart.Text(sys))))
        }

        for (msg in messages) {
            history.add(msg)
        }

        return BackendResult.Ok(Unit)
    }

    fun close() {
        if (closed) return
        closed = true
        history.clear()
        Log.i(TAG, "close($sessionId): session closed")
    }

    private data class TurnPrep(
        val newMessages: List<QuickAiChatMessage>
    )

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

        return TurnPrep(newMessages = messages)
    }

    private fun buildPromptFromHistory(newMessages: List<QuickAiChatMessage>): String {
        for (msg in newMessages) {
            history.add(msg)
        }

        val architecture = architectureProvider() ?: "Qwen3ForCausalLM"
        return applyChatTemplate(history, architecture)
    }

    private fun applyChatTemplate(
        messages: List<QuickAiChatMessage>,
        architecture: String
    ): String {
        val result = StringBuilder()

        when {
            architecture.contains("Qwen", ignoreCase = true) -> {
                for (msg in messages) {
                    result.append("<|im_start|>${msg.role.name.lowercase()}\n")
                    result.append(extractText(msg))
                    result.append("<|im_end|>\n")
                }
                result.append("<|im_start|>assistant\n")
            }
            architecture.contains("Llama", ignoreCase = true) -> {
                var inInst = false
                for (msg in messages) {
                    when (msg.role) {
                        QuickAiChatRole.SYSTEM -> {
                            result.append("<<SYS>>\n${extractText(msg)}\n<</SYS>>\n\n")
                        }
                        QuickAiChatRole.USER -> {
                            result.append("[INST] ${extractText(msg)} [/INST]")
                            inInst = true
                        }
                        QuickAiChatRole.ASSISTANT -> {
                            result.append("${extractText(msg)}\n")
                            inInst = false
                        }
                    }
                }
            }
            architecture.contains("Gemma", ignoreCase = true) -> {
                for (msg in messages) {
                    when (msg.role) {
                        QuickAiChatRole.USER -> {
                            result.append("<start_of_turn>user\n${extractText(msg)}<end_of_turn>\n")
                        }
                        QuickAiChatRole.ASSISTANT -> {
                            result.append("<start_of_turn>model\n${extractText(msg)}<end_of_turn>\n")
                        }
                        QuickAiChatRole.SYSTEM -> {
                            result.append("${extractText(msg)}\n\n")
                        }
                    }
                }
                result.append("<start_of_turn>model\n")
            }
            else -> {
                for (msg in messages) {
                    result.append("[${msg.role.name}]: ${extractText(msg)}\n")
                }
                result.append("[ASSISTANT]: ")
            }
        }

        return result.toString()
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

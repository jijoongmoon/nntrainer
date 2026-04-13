// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    LiteRTLmChatSession.kt
 * @brief   QuickAiChatSession backed by a LiteRT-LM Conversation.
 *
 * Each session owns its own LiteRT-LM [Conversation] and an [ImageStore]
 * that caches image bytes keyed by SHA-256 hash. When the same image
 * arrives via a different temporary file path, the hash matches and
 * the conversation history stays consistent.
 *
 * History is accumulated internally — callers send only the new
 * messages for each turn, and the session appends them plus the
 * assistant reply.
 */
package com.example.quickdotai

import android.util.Log
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import java.io.File
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * @param engine      the parent LiteRT-LM engine (kept for rebuild)
 * @param config      session-level sampling + template config
 * @param visionEnabled whether the engine was loaded with a vision backend
 * @param onSessionClosed callback fired once when [close] is invoked.
 *        Used by [LiteRTLm] to clear its `activeSession` and restore
 *        the flat-API Conversation. Skipped when the parent engine is
 *        tearing down (it nulls out `activeSession` before calling
 *        close so the callback sees nothing to do).
 */
class LiteRTLmChatSession(
    private val engine: Engine,
    private val config: QuickAiChatSessionConfig?,
    private val visionEnabled: Boolean,
    private val onSessionClosed: (() -> Unit)? = null,
    override val sessionId: String = UUID.randomUUID().toString()
) : QuickAiChatSession {

    private var conversation: Conversation? = engine.createConversation()
    val imageStore = ImageStore()

    /** Accumulated conversation history. */
    private val history = mutableListOf<QuickAiChatMessage>()

    /** Signals an in-flight cancel request. */
    private val cancelRequested = AtomicBoolean(false)

    @Volatile
    private var closed = false

    private var lastRunDurationMs: Double = 0.0

    // ----- run (blocking) ------------------------------------------------

    override fun run(
        messages: List<QuickAiChatMessage>
    ): BackendResult<QuickAiChatResult> {
        val c = conversation ?: return errClosed()
        if (closed) return errClosed()
        cancelRequested.set(false)

        if (messages.isEmpty()) {
            return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "messages list is empty"
            )
        }

        // Build the LiteRT-LM contents from the last user message.
        // The conversation object already holds prior context.
        val lastUserMsg = messages.lastOrNull { it.role == QuickAiChatRole.USER }
            ?: return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "no USER message found in the provided messages"
            )

        // Append all new messages to history (they become part of the
        // session state regardless of inference outcome).
        history.addAll(messages)

        return try {
            val startNs = System.nanoTime()
            val response = if (hasImages(lastUserMsg)) {
                if (!visionEnabled) {
                    return BackendResult.Err(
                        QuickAiError.UNSUPPORTED,
                        "Engine loaded in text-only mode — cannot process images"
                    )
                }
                val contents = toChatContents(lastUserMsg)
                c.sendMessage(contents)
            } else {
                val text = extractText(lastUserMsg)
                c.sendMessage(text)
            }
            lastRunDurationMs = (System.nanoTime() - startNs) / 1_000_000.0

            val output = response.toString()
            Log.i(TAG, "run($sessionId): completed in ${lastRunDurationMs.toLong()} ms")

            // Append assistant reply to history
            val assistantMsg = QuickAiChatMessage(
                role = QuickAiChatRole.ASSISTANT,
                parts = listOf(PromptPart.Text(output))
            )
            history.add(assistantMsg)

            BackendResult.Ok(
                QuickAiChatResult(
                    content = output,
                    metrics = PerformanceMetrics(totalDurationMs = lastRunDurationMs)
                )
            )
        } catch (t: Throwable) {
            Log.e(TAG, "run($sessionId): inference failed", t)
            BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                t.message ?: "chat inference failed"
            )
        }
    }

    // ----- runStreaming ---------------------------------------------------

    override fun runStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult> {
        val c = conversation ?: run {
            val err = errClosed()
            sink.onError(err.error, err.message)
            return err
        }
        if (closed) {
            val err = errClosed()
            sink.onError(err.error, err.message)
            return err
        }
        cancelRequested.set(false)

        if (messages.isEmpty()) {
            val err = BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "messages list is empty"
            )
            sink.onError(err.error, err.message)
            return err
        }

        val lastUserMsg = messages.lastOrNull { it.role == QuickAiChatRole.USER }
        if (lastUserMsg == null) {
            val err = BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "no USER message found"
            )
            sink.onError(err.error, err.message)
            return err
        }

        history.addAll(messages)

        val latch = CountDownLatch(1)
        val accumulated = StringBuilder()
        var terminalError: BackendResult.Err? = null
        val startNs = System.nanoTime()

        val callback = object : MessageCallback {
            override fun onMessage(message: Message) {
                if (cancelRequested.get()) return
                try {
                    val full = message.toString()
                    val delta = if (full.startsWith(accumulated.toString())) {
                        full.substring(accumulated.length)
                    } else {
                        full
                    }
                    if (delta.isNotEmpty()) {
                        accumulated.append(delta)
                        sink.onDelta(delta)
                    }
                } catch (t: Throwable) {
                    Log.w(TAG, "runStreaming($sessionId): onMessage threw", t)
                }
            }

            override fun onDone() {
                lastRunDurationMs = (System.nanoTime() - startNs) / 1_000_000.0
                Log.i(
                    TAG,
                    "runStreaming($sessionId): onDone after " +
                        "${lastRunDurationMs.toLong()} ms"
                )
                try { sink.onDone() } finally { latch.countDown() }
            }

            override fun onError(throwable: Throwable) {
                Log.e(TAG, "runStreaming($sessionId): onError", throwable)
                val err = BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    throwable.message ?: "chat streaming failed"
                )
                terminalError = err
                try { sink.onError(err.error, err.message) } finally { latch.countDown() }
            }
        }

        return try {
            if (hasImages(lastUserMsg)) {
                if (!visionEnabled) {
                    val err = BackendResult.Err(
                        QuickAiError.UNSUPPORTED,
                        "Engine loaded in text-only mode"
                    )
                    sink.onError(err.error, err.message)
                    return err
                }
                val contents = toChatContents(lastUserMsg)
                c.sendMessageAsync(contents, callback)
            } else {
                val text = extractText(lastUserMsg)
                c.sendMessageAsync(text, callback)
            }

            val finished = latch.await(5, TimeUnit.MINUTES)
            if (!finished) {
                Log.e(TAG, "runStreaming($sessionId): timed out")
                val err = BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    "chat streaming timeout"
                )
                sink.onError(err.error, err.message)
                return err
            }

            if (terminalError != null) {
                return terminalError!!
            }

            val output = accumulated.toString()
            history.add(
                QuickAiChatMessage(
                    role = QuickAiChatRole.ASSISTANT,
                    parts = listOf(PromptPart.Text(output))
                )
            )
            BackendResult.Ok(
                QuickAiChatResult(
                    content = output,
                    metrics = PerformanceMetrics(totalDurationMs = lastRunDurationMs)
                )
            )
        } catch (t: Throwable) {
            Log.e(TAG, "runStreaming($sessionId): threw", t)
            val err = BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                t.message ?: "chat streaming failed"
            )
            sink.onError(err.error, err.message)
            err
        }
    }

    // ----- cancel --------------------------------------------------------

    override fun cancel() {
        cancelRequested.set(true)
        Log.i(TAG, "cancel($sessionId): cancel requested")
    }

    // ----- rebuild -------------------------------------------------------

    override fun rebuild(
        messages: List<QuickAiChatMessage>
    ): BackendResult<Unit> {
        if (closed) return errClosed()

        Log.i(
            TAG,
            "rebuild($sessionId): replacing history " +
                "(${history.size} → ${messages.size} messages)"
        )

        // Close old conversation
        try {
            conversation?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "rebuild($sessionId): conversation.close() threw", t)
        }

        // Create a fresh conversation
        conversation = try {
            engine.createConversation()
        } catch (t: Throwable) {
            Log.e(TAG, "rebuild($sessionId): createConversation failed", t)
            return BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                "Failed to create new conversation: ${t.message}"
            )
        }

        // Update history
        history.clear()
        history.addAll(messages)

        // Prune images not referenced by the new history
        val referencedHashes = collectImageHashes(messages)
        imageStore.retainOnly(referencedHashes)

        Log.i(TAG, "rebuild($sessionId): done, new history size=${history.size}")
        return BackendResult.Ok(Unit)
    }

    // ----- close ---------------------------------------------------------

    override fun close() {
        if (closed) return
        closed = true
        Log.i(TAG, "close($sessionId)")
        try {
            conversation?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "close($sessionId): conversation.close() threw", t)
        }
        conversation = null
        imageStore.clear()
        history.clear()
        try {
            onSessionClosed?.invoke()
        } catch (t: Throwable) {
            Log.w(TAG, "close($sessionId): onSessionClosed threw", t)
        }
    }

    // ----- helpers -------------------------------------------------------

    private fun hasImages(msg: QuickAiChatMessage): Boolean =
        msg.parts.any { it is PromptPart.ImageFile || it is PromptPart.ImageBytes }

    private fun extractText(msg: QuickAiChatMessage): String =
        msg.parts.filterIsInstance<PromptPart.Text>().joinToString("") { it.text }

    /**
     * Convert a user message's parts into a LiteRT-LM [Contents].
     * Images are stored in [imageStore] along the way.
     */
    private fun toChatContents(msg: QuickAiChatMessage): Contents {
        val mapped: List<Content> = msg.parts.map { p ->
            when (p) {
                is PromptPart.Text -> Content.Text(p.text)
                is PromptPart.ImageFile -> {
                    val f = File(p.absolutePath)
                    require(f.exists() && f.canRead()) {
                        "Image file not readable: ${p.absolutePath}"
                    }
                    // Store in ImageStore for stable hash-based identity
                    imageStore.store(p.absolutePath)
                    Content.ImageFile(p.absolutePath)
                }
                is PromptPart.ImageBytes -> {
                    require(p.bytes.isNotEmpty()) {
                        "Image bytes are empty"
                    }
                    imageStore.store(p.bytes)
                    Content.ImageBytes(p.bytes)
                }
            }
        }
        return Contents.of(mapped)
    }

    /**
     * Collect all image hashes referenced in a list of messages.
     * Used by [rebuild] to determine which cached images to keep.
     */
    private fun collectImageHashes(messages: List<QuickAiChatMessage>): Set<String> {
        val hashes = mutableSetOf<String>()
        for (msg in messages) {
            for (part in msg.parts) {
                when (part) {
                    is PromptPart.ImageFile -> {
                        val f = File(part.absolutePath)
                        if (f.exists() && f.canRead()) {
                            hashes.add(ImageStore.sha256Hex(f.readBytes()))
                        }
                    }
                    is PromptPart.ImageBytes -> {
                        if (part.bytes.isNotEmpty()) {
                            hashes.add(ImageStore.sha256Hex(part.bytes))
                        }
                    }
                    is PromptPart.Text -> { /* no image */ }
                }
            }
        }
        return hashes
    }

    private fun errClosed(): BackendResult.Err = BackendResult.Err(
        QuickAiError.NOT_INITIALIZED,
        "Chat session $sessionId is closed"
    )

    companion object {
        private const val TAG = "LiteRTLmChatSession"
    }
}

// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    NativeChatSession.kt
 * @brief   Internal chat session helper for the native causal_lm backend.
 *
 * The native engine does not yet support structured chat / multi-turn
 * conversation. This stub returns [QuickAiError.UNSUPPORTED] for all
 * inference operations so that callers get a clear error message rather
 * than a silent failure. The session object is still constructable so
 * that the host service layer can manage it uniformly.
 */
package com.example.quickdotai

import java.util.UUID

internal class NativeChatSession(
    val sessionId: String = UUID.randomUUID().toString()
) {

    @Volatile
    private var closed = false

    fun run(
        messages: List<QuickAiChatMessage>
    ): BackendResult<QuickAiChatResult> {
        if (closed) return errClosed()
        return BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "Structured chat is not supported by the native backend"
        )
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
        val err = BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "Structured chat streaming is not supported by the native backend"
        )
        sink.onError(err.error, err.message)
        return err
    }

    fun cancel() {
        // no-op — nothing to cancel
    }

    fun rebuild(
        messages: List<QuickAiChatMessage>
    ): BackendResult<Unit> {
        if (closed) return errClosed()
        return BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "Structured chat rebuild is not supported by the native backend"
        )
    }

    fun close() {
        closed = true
    }

    private fun errClosed(): BackendResult.Err = BackendResult.Err(
        QuickAiError.NOT_INITIALIZED,
        "Chat session $sessionId is closed"
    )
}

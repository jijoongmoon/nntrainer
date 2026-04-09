// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    ChunkedStreamSink.kt
 * @brief   StreamSink implementation that exposes a queue-backed
 *          InputStream suitable for NanoHTTPD.newChunkedResponse().
 *
 * Producer side:  backend threads (ModelWorker + LiteRT-LM MessageCallback)
 *                 call onDelta/onDone/onError.
 * Consumer side:  the NanoHTTPD writer thread reads from [inputStream] and
 *                 copies bytes to the HTTP socket until EOF.
 *
 * The two sides are decoupled by a LinkedBlockingQueue<ByteArray>, with an
 * empty ByteArray serving as the EOF sentinel. See Architecture.md §5.1.
 */
package com.example.QuickAI.service

import android.util.Log
import com.example.quickdotai.QuickAiError
import com.example.quickdotai.StreamSink
import java.io.InputStream
import java.util.concurrent.LinkedBlockingQueue
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * @brief A StreamSink that serialises every event as an NDJSON line
 * ([StreamFrame] + `\n`) and queues it for a downstream InputStream.
 *
 * Thread safety: all public methods are safe to call concurrently with
 * [InputStream.read] on [inputStream]. In practice the producer is a single
 * thread (the backend / MessageCallback) and the consumer is a single
 * NanoHTTPD writer thread, so contention is limited to the blocking queue.
 *
 * Lifecycle: the sink becomes "closed" after the first terminal event
 * ([onDone] or [onError]). Additional events after that are dropped with a
 * warning log, never duplicated on the wire.
 */
class ChunkedStreamSink : StreamSink {

    private val json = Json {
        encodeDefaults = false
        ignoreUnknownKeys = true
    }

    // Deltas + the final done/error frame land here. The reader blocks on
    // queue.take() until either bytes or the EOF sentinel show up.
    private val queue = LinkedBlockingQueue<ByteArray>()

    // Empty byte array is the EOF sentinel — reader treats it as "no more
    // data will ever come" and returns -1 from read().
    private val eofSentinel = ByteArray(0)

    @Volatile
    private var terminated: Boolean = false

    /**
     * @brief The read end of the sink. Hand this to NanoHTTPD via
     * `newChunkedResponse(status, mimeType, inputStream)` — the framework
     * will drain it on its own writer thread.
     */
    val inputStream: InputStream = object : InputStream() {
        private var current: ByteArray = ByteArray(0)
        private var pos: Int = 0
        private var eof: Boolean = false

        override fun read(): Int {
            if (eof) return -1
            if (!ensureBuffer()) return -1
            return current[pos++].toInt() and 0xFF
        }

        override fun read(b: ByteArray, off: Int, len: Int): Int {
            if (eof) return -1
            if (len <= 0) return 0
            if (!ensureBuffer()) return -1
            val remaining = current.size - pos
            val n = if (remaining < len) remaining else len
            System.arraycopy(current, pos, b, off, n)
            pos += n
            return n
        }

        /**
         * Refill [current] if it's been fully consumed. Blocks on the
         * queue until new bytes or EOF arrive. Returns false on EOF.
         */
        private fun ensureBuffer(): Boolean {
            while (pos >= current.size) {
                val next = try {
                    queue.take()
                } catch (ie: InterruptedException) {
                    Thread.currentThread().interrupt()
                    eof = true
                    return false
                }
                if (next === eofSentinel || next.isEmpty()) {
                    eof = true
                    return false
                }
                current = next
                pos = 0
            }
            return true
        }

        override fun close() {
            eof = true
        }
    }

    override fun onDelta(text: String) {
        if (terminated) {
            Log.w(TAG, "onDelta after terminal event, dropping ${text.length} chars")
            return
        }
        if (text.isEmpty()) return
        queue.put(encodeFrame(StreamFrame(type = "delta", text = text)))
    }

    override fun onDone() {
        if (terminated) return
        terminated = true
        queue.put(encodeFrame(StreamFrame(type = "done")))
        queue.put(eofSentinel)
    }

    override fun onError(error: QuickAiError, message: String?) {
        if (terminated) {
            Log.w(TAG, "onError after terminal event: ${error.name} $message")
            return
        }
        terminated = true
        queue.put(
            encodeFrame(
                StreamFrame(
                    type = "error",
                    errorCode = error.code,
                    message = message ?: error.name
                )
            )
        )
        queue.put(eofSentinel)
    }

    private fun encodeFrame(frame: StreamFrame): ByteArray {
        val line = json.encodeToString(frame) + "\n"
        return line.toByteArray(Charsets.UTF_8)
    }

    companion object {
        private const val TAG = "ChunkedStreamSink"
    }
}

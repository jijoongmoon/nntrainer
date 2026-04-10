// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    LiteRTLm.kt
 * @brief   QuickDotAI implementation backed by the LiteRT-LM Kotlin API
 *          (https://github.com/google-ai-edge/LiteRT-LM).
 *
 * LiteRTLm is the QuickDotAI-level routing target for ModelId.GEMMA4
 * and is typically selected inside the host app's registry. It
 * implements the same [QuickDotAI] contract as [NativeQuickDotAI], so
 * consumers never need to branch on the concrete implementation.
 *
 * See how-to-use-litert-lm-guide.md at the repo root for the canonical
 * LiteRT-LM Kotlin API surface this code is written against.
 */
package com.example.quickdotai

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend as LlmBackend
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * @brief LiteRT-LM-backed QuickDotAI implementation for Gemma-family
 *        models.
 *
 * Non-thread-safe — the host app must drive a single instance from a
 * single worker thread.
 *
 * @param appContext application context, used only to resolve the
 *        test-mode fallback model path via [Context.getExternalFilesDir]
 *        when [LoadModelRequest.modelPath] is null. Third-party apps
 *        are encouraged to always pass an explicit [LoadModelRequest.modelPath]
 *        and therefore never hit the fallback.
 */
class LiteRTLm(
    private val appContext: Context
) : QuickDotAI {

    override val kind: String = "litert-lm"

    override var architecture: String? = "Gemma4ForCausalLM"
        private set

    // LiteRT-LM's Engine is AutoCloseable — we hold on to it (and a
    // single reusable Conversation) for the entire lifetime of the
    // LiteRTLm instance. Closed in [close].
    private var engine: Engine? = null
    private var conversation: Conversation? = null

    // Simple wall-clock metrics. LiteRT-LM's Kotlin API does not expose
    // token-level prefill/generation timings in the release we target,
    // so we record initialization and last-run durations ourselves and
    // leave the token counts at 0 for now.
    private var initializationDurationMs: Double = 0.0
    private var lastRunDurationMs: Double = 0.0

    override fun load(req: LoadModelRequest): BackendResult<Unit> {
        Log.i(
            TAG,
            "load() entered: model=${req.model} backend=${req.backend} " +
                "quant=${req.quantization} modelPath=${req.modelPath}"
        )

        // TEST-MODE fallback: during bring-up we want `load(GEMMA4)` to
        // Just Work even if the caller forgets to pass model_path. Fall
        // back to a known-good on-device path inside the host app's
        // external files dir so the pipeline can be end-to-end verified.
        //
        // /data/local/tmp is NOT app-readable on user builds: it carries
        // the shell_data_file SELinux context and the untrusted_app
        // domain is denied read access. getExternalFilesDir() (a)
        // requires no runtime permissions, (b) is always writable by
        // adb, and (c) is already created by the framework for us.
        val modelPath = req.modelPath?.takeIf { it.isNotBlank() }
            ?: run {
                val fallback = testModelFile().absolutePath
                Log.w(
                    TAG,
                    "load(): model_path not provided, falling back to " +
                        "test path: $fallback"
                )
                fallback
            }

        Log.i(TAG, "load(): resolved modelPath=$modelPath")

        val modelFile = File(modelPath)
        if (!modelFile.exists()) {
            val parentDir = testModelFile().parentFile?.absolutePath ?: "<unknown>"
            val hint = "push it with: adb push $TEST_GEMMA4_FILE_NAME $parentDir/"
            Log.e(TAG, "load(): model file not found at $modelPath — $hint")
            return BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                "model file not found at $modelPath. $hint"
            )
        }
        Log.i(
            TAG,
            "load(): model file exists, size=${modelFile.length()} bytes, " +
                "canRead=${modelFile.canRead()}"
        )

        val llmBackend: LlmBackend = when (req.backend) {
            BackendType.CPU -> LlmBackend.CPU()
            BackendType.GPU -> LlmBackend.GPU()
            // LiteRT-LM's NPU backend wants the dir holding the vendor
            // native .so files. For an app-bundled setup that is simply
            // the APK's nativeLibraryDir, but we don't have a Context
            // here — fall back to CPU until the caller wires one in.
            BackendType.NPU -> LlmBackend.CPU()
        }
        Log.i(
            TAG,
            "load(): mapped compute backend ${req.backend} -> " +
                llmBackend::class.java.simpleName
        )

        val engineConfig = EngineConfig(
            modelPath = modelPath,
            backend = llmBackend
        )
        Log.i(TAG, "load(): EngineConfig built, constructing Engine…")

        return try {
            val startNs = System.nanoTime()
            val e = Engine(engineConfig)
            Log.i(TAG, "load(): Engine() constructed, calling initialize()…")
            e.initialize()
            Log.i(
                TAG,
                "load(): Engine.initialize() returned after " +
                    "${(System.nanoTime() - startNs) / 1_000_000} ms"
            )

            val c = e.createConversation()
            Log.i(TAG, "load(): Engine.createConversation() returned")

            initializationDurationMs = (System.nanoTime() - startNs) / 1_000_000.0
            engine = e
            conversation = c
            Log.i(
                TAG,
                "load(): SUCCESS, total init duration=${initializationDurationMs} ms"
            )
            BackendResult.Ok(Unit)
        } catch (t: Throwable) {
            Log.e(TAG, "load(): LiteRT-LM engine load failed", t)
            // On partial success, make sure we don't leak a half-initialised
            // engine into the caller's registry.
            closeQuietly()
            BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                t.message ?: "LiteRT-LM engine initialization failed"
            )
        }
    }

    override fun run(prompt: String): BackendResult<String> {
        val c = conversation
            ?: run {
                Log.e(TAG, "run(): called before load() — conversation is null")
                return BackendResult.Err(
                    QuickAiError.NOT_INITIALIZED,
                    "LiteRTLm has not been loaded yet"
                )
            }

        Log.i(TAG, "run(): sending prompt of length ${prompt.length}")
        return try {
            val startNs = System.nanoTime()
            // Blocking synchronous send; callers are expected to drive
            // us from a background thread. Streaming is handled in
            // runStreaming() via sendMessageAsync().
            val message = c.sendMessage(prompt)
            lastRunDurationMs = (System.nanoTime() - startNs) / 1_000_000.0
            val output = message.toString()
            Log.i(
                TAG,
                "run(): sendMessage returned in ${lastRunDurationMs.toLong()} ms, " +
                    "output length=${output.length}"
            )
            BackendResult.Ok(output)
        } catch (t: Throwable) {
            Log.e(TAG, "run(): LiteRT-LM sendMessage failed", t)
            BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                t.message ?: "LiteRT-LM inference failed"
            )
        }
    }

    /**
     * @brief Streaming override that drives LiteRT-LM's asynchronous
     * `sendMessageAsync(prompt, MessageCallback)` and forwards each
     * incremental `onMessage` to [sink] as a delta.
     *
     * The caller's thread blocks on a [CountDownLatch] until LiteRT-LM
     * invokes either `onDone` or `onError`, so FIFO ordering across
     * streaming and non-streaming jobs is preserved — we do not return
     * to the caller while the model is still decoding tokens.
     *
     * LiteRT-LM's `onMessage(Message)` contract is not explicit about
     * whether each Message is a delta or a running accumulation. We
     * therefore keep a private StringBuilder of everything we've
     * emitted so far and only forward the new suffix when the incoming
     * message starts with it — otherwise (defensive fallback) we
     * forward the raw text. This handles both shapes without
     * double-emitting tokens.
     */
    override fun runStreaming(
        prompt: String,
        sink: StreamSink
    ): BackendResult<Unit> {
        val c = conversation
            ?: run {
                Log.e(TAG, "runStreaming(): called before load()")
                val err = BackendResult.Err(
                    QuickAiError.NOT_INITIALIZED,
                    "LiteRTLm has not been loaded yet"
                )
                sink.onError(err.error, err.message)
                return err
            }

        Log.i(TAG, "runStreaming(): prompt length=${prompt.length}")

        val latch = CountDownLatch(1)
        val accumulated = StringBuilder()
        // Outcome is published from the callback thread and read on the
        // caller thread after latch.await() returns.
        var terminalError: BackendResult.Err? = null
        val startNs = System.nanoTime()

        val callback = object : MessageCallback {
            override fun onMessage(message: Message) {
                try {
                    val full = message.toString()
                    // Defensive delta extraction — if the callback emits
                    // accumulated snapshots we forward only the suffix;
                    // if it already emits per-token deltas, `full` will
                    // not start with `accumulated` and we forward the raw
                    // text and accumulate it.
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
                    Log.w(TAG, "runStreaming(): onMessage threw", t)
                }
            }

            override fun onDone() {
                lastRunDurationMs = (System.nanoTime() - startNs) / 1_000_000.0
                Log.i(
                    TAG,
                    "runStreaming(): onDone after ${lastRunDurationMs.toLong()} ms, " +
                        "total chars=${accumulated.length}"
                )
                try {
                    sink.onDone()
                } finally {
                    latch.countDown()
                }
            }

            override fun onError(throwable: Throwable) {
                Log.e(TAG, "runStreaming(): onError from LiteRT-LM", throwable)
                val err = BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    throwable.message ?: "LiteRT-LM streaming inference failed"
                )
                terminalError = err
                try {
                    sink.onError(err.error, err.message)
                } finally {
                    latch.countDown()
                }
            }
        }

        return try {
            c.sendMessageAsync(prompt, callback)
            // Wait up to 5 minutes — the same envelope the host
            // QuickAIService uses for blocking runs. If the callback
            // never fires we surface a timeout so the caller thread
            // isn't parked forever.
            val finished = latch.await(5, TimeUnit.MINUTES)
            if (!finished) {
                Log.e(TAG, "runStreaming(): timed out waiting for onDone/onError")
                val err = BackendResult.Err(
                    QuickAiError.INFERENCE_FAILED,
                    "LiteRT-LM streaming timeout"
                )
                sink.onError(err.error, err.message)
                return err
            }
            terminalError ?: BackendResult.Ok(Unit)
        } catch (t: Throwable) {
            Log.e(TAG, "runStreaming(): sendMessageAsync threw", t)
            val err = BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                t.message ?: "LiteRT-LM streaming inference failed"
            )
            sink.onError(err.error, err.message)
            err
        }
    }

    override fun unload(): BackendResult<Unit> {
        Log.i(TAG, "unload() invoked")
        closeQuietly()
        return BackendResult.Ok(Unit)
    }

    override fun metrics(): BackendResult<PerformanceMetrics> {
        if (conversation == null) {
            return BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "LiteRTLm has not been loaded yet"
            )
        }
        // LiteRT-LM does not currently expose token-level counters
        // through its Kotlin API, so most fields stay at 0. We still
        // publish the wall-clock timings we measured ourselves so
        // callers can at least see the load + last-run durations.
        return BackendResult.Ok(
            PerformanceMetrics(
                initializationDurationMs = initializationDurationMs,
                totalDurationMs = lastRunDurationMs
            )
        )
    }

    override fun close() {
        Log.i(TAG, "close() invoked")
        closeQuietly()
    }

    private fun closeQuietly() {
        try {
            conversation?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "conversation.close() threw", t)
        }
        conversation = null
        try {
            engine?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "engine.close() threw", t)
        }
        engine = null
    }

    /**
     * @brief Build the test-mode fallback model file handle, rooted in
     * the host app's external files dir (app-private, no permissions).
     * Creates the parent directory so `adb push` can write directly
     * without a separate `mkdir -p`.
     */
    private fun testModelFile(): File {
        val externalFiles = appContext.getExternalFilesDir(null)
            ?: appContext.filesDir
        val dir = File(externalFiles, TEST_GEMMA4_REL_DIR)
        if (!dir.exists()) dir.mkdirs()
        return File(dir, TEST_GEMMA4_FILE_NAME)
    }

    companion object {
        private const val TAG = "LiteRTLm"

        /**
         * @brief TEST ONLY — path components of the Gemma-4 E2B-IT
         * `.litertlm` model, relative to the host app's external files
         * dir. The absolute path is resolved at runtime via
         * [testModelFile] so it always reflects the actual host package.
         *
         * Push the file with adb before installing the app (note that
         * the app id depends on which host is using the AAR):
         *   adb push gemma-4-E2B-it.litertlm \
         *       /sdcard/Android/data/<app-id>/files/models/gemma-4-E2B-it/
         */
        const val TEST_GEMMA4_REL_DIR: String = "models/gemma-4-E2B-it"
        const val TEST_GEMMA4_FILE_NAME: String = "gemma-4-E2B-it.litertlm"
    }
}

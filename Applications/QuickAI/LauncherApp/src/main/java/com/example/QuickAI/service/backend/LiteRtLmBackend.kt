// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    LiteRtLmBackend.kt
 * @brief   Backend implementation that runs Gemma-family models through
 *          the LiteRT-LM Kotlin API
 *          (https://github.com/google-ai-edge/LiteRT-LM).
 *
 * This backend is the Kotlin-level routing target for ModelId.GEMMA4 and
 * is selected inside ModelRegistry.createBackendFor(). It implements the
 * same Backend interface as NativeCausalLmBackend, so ModelWorker and the
 * rest of the REST surface are unaware of which engine is in use.
 *
 * See how-to-use-litert-lm-guide.md at the repo root for the canonical
 * LiteRT-LM Kotlin API surface this code is written against.
 */
package com.example.QuickAI.service.backend

import android.util.Log
import com.example.QuickAI.service.BackendType as QuickAiBackendType
import com.example.QuickAI.service.LoadModelRequest
import com.example.QuickAI.service.PerformanceMetrics
import com.example.QuickAI.service.QuickAiError
import com.google.ai.edge.litertlm.Backend as LlmBackend
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig

/**
 * @brief LiteRT-LM-backed implementation for Gemma4.
 *
 * Non-thread-safe by design — one ModelWorker owns one backend instance
 * and drives it from its dedicated worker thread (Architecture.md §2.6).
 */
class LiteRtLmBackend : Backend {

    override val kind: String = "litert-lm"

    override var architecture: String? = "Gemma4ForCausalLM"
        private set

    // LiteRT-LM's Engine is AutoCloseable — we hold on to it (and a single
    // reusable Conversation) for the entire lifetime of the ModelWorker.
    // Closed in [close] when the worker shuts down.
    private var engine: Engine? = null
    private var conversation: Conversation? = null

    // Simple wall-clock metrics. LiteRT-LM's Kotlin API does not expose
    // token-level prefill/generation timings in the release we target, so
    // we record initialization and last-run durations ourselves and leave
    // the token counts at 0 for now.
    private var initializationDurationMs: Double = 0.0
    private var lastRunDurationMs: Double = 0.0

    override fun load(req: LoadModelRequest): BackendResult<Unit> {
        // TEST HARDCODING (see gemma-model-path.md): during bring-up we
        // want `POST /v1/models` with model=GEMMA4 to Just Work even if
        // the client forgets to pass model_path. Fall back to the known
        // good on-device path so we can end-to-end-verify the LiteRT-LM
        // pipeline. Remove once the client UI reliably supplies a path.
        val modelPath = req.modelPath?.takeIf { it.isNotBlank() }
            ?: TEST_GEMMA4_MODEL_PATH.also {
                Log.w(
                    TAG,
                    "loadModel: model_path not provided, falling back to test " +
                        "hardcoded path: $it"
                )
            }

        val llmBackend: LlmBackend = when (req.backend) {
            QuickAiBackendType.CPU -> LlmBackend.CPU()
            QuickAiBackendType.GPU -> LlmBackend.GPU()
            // LiteRT-LM's NPU backend wants the dir holding the vendor
            // native .so files. For an app-bundled setup that is simply
            // the APK's nativeLibraryDir, but we don't have a Context
            // here — fall back to CPU until the caller wires one in.
            QuickAiBackendType.NPU -> LlmBackend.CPU()
        }

        val engineConfig = EngineConfig(
            modelPath = modelPath,
            backend = llmBackend
        )

        return try {
            val startNs = System.nanoTime()
            val e = Engine(engineConfig)
            e.initialize()
            val c = e.createConversation()
            initializationDurationMs = (System.nanoTime() - startNs) / 1_000_000.0
            engine = e
            conversation = c
            BackendResult.Ok(Unit)
        } catch (t: Throwable) {
            Log.w(TAG, "LiteRT-LM engine load failed", t)
            // On partial success, make sure we don't leak a half-initialised
            // engine into the registry.
            closeQuietly()
            BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                t.message ?: "LiteRT-LM engine initialization failed"
            )
        }
    }

    override fun run(prompt: String): BackendResult<String> {
        val c = conversation
            ?: return BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "LiteRT-LM backend has not been loaded yet"
            )

        return try {
            val startNs = System.nanoTime()
            // Blocking synchronous send; the ModelWorker thread that calls
            // us is already a background thread, so this is safe. Streaming
            // via sendMessageAsync().collect { ... } is a later iteration
            // (Architecture.md §10 — out of scope for the current REST
            // endpoints, which are blocking request/response).
            val message = c.sendMessage(prompt)
            lastRunDurationMs = (System.nanoTime() - startNs) / 1_000_000.0
            BackendResult.Ok(message.toString())
        } catch (t: Throwable) {
            Log.w(TAG, "LiteRT-LM sendMessage failed", t)
            BackendResult.Err(
                QuickAiError.INFERENCE_FAILED,
                t.message ?: "LiteRT-LM inference failed"
            )
        }
    }

    override fun metrics(): BackendResult<PerformanceMetrics> {
        if (conversation == null) {
            return BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "LiteRT-LM backend has not been loaded yet"
            )
        }
        // LiteRT-LM does not currently expose token-level counters through
        // its Kotlin API, so most fields stay at 0. We still publish the
        // wall-clock timings we measured ourselves so clients can at least
        // see the load + last-run durations.
        return BackendResult.Ok(
            PerformanceMetrics(
                initializationDurationMs = initializationDurationMs,
                totalDurationMs = lastRunDurationMs
            )
        )
    }

    override fun close() {
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

    companion object {
        private const val TAG = "LiteRtLmBackend"

        /**
         * @brief TEST ONLY — absolute on-device path to the Gemma-4 E2B-IT
         * `.litertlm` model used for LiteRT-LM bring-up.
         *
         * Kept in sync with gemma-model-path.md at the repo root. Push the
         * file with adb before installing the app:
         *   adb push gemma-4-E2B-it.litertlm \
         *       /data/local/tmp/Quick.AI/models/gemma-4-E2B-it/
         */
        const val TEST_GEMMA4_MODEL_PATH: String =
            "/data/local/tmp/Quick.AI/models/gemma-4-E2B-it/gemma-4-E2B-it.litertlm"
    }
}

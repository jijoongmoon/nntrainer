// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    ModelRegistry.kt
 * @brief   Process-wide map from model_id → ModelWorker.
 *
 * The registry is the entry point used by RequestDispatcher. It:
 *  - creates a fresh worker when a client calls loadModel,
 *  - returns the existing worker on repeated loads (idempotent),
 *  - routes Gemma4 loads to the LiteRT-LM backend and everything else to
 *    the native causal_lm_api backend (Architecture.md §4),
 *  - guarantees there is at most one worker per (model, quantization)
 *    pair — i.e. same model requests share a FIFO, different models run
 *    on different threads in parallel (Architecture.md §2.6).
 */
package com.example.QuickAI.service

import android.content.Context
import android.util.Log
import com.example.quickdotai.BackendResult
import com.example.quickdotai.LiteRTLm
import com.example.quickdotai.NativeQuickDotAI
import com.example.quickdotai.QuickDotAI
import java.util.concurrent.ConcurrentHashMap

/**
 * @param appContext application context used by backends that need to
 *        resolve on-device paths (LiteRT-LM fallback model path in
 *        particular). Stored as the application context so it outlives
 *        any single activity.
 */
class ModelRegistry(
    private val appContext: Context
) {

    private val workers = ConcurrentHashMap<String, ModelWorker>()

    // A single lock serializes load/unload transitions so we don't race
    // when two clients simultaneously request the same model for the
    // first time. Run requests do NOT take this lock.
    private val transitionLock = Any()

    /**
     * @brief Look up an existing worker or create one.
     *
     * If a worker for [req.modelKey] already exists, it is returned
     * unchanged (idempotent load). Otherwise a new worker is created,
     * started, and the load result is returned.
     */
    fun getOrLoad(req: LoadModelRequest): BackendResult<ModelWorker> {
        Log.i(TAG, "getOrLoad(${req.modelKey}) entered")
        workers[req.modelKey]?.let {
            Log.i(TAG, "getOrLoad(${req.modelKey}): cache hit, returning existing worker")
            return BackendResult.Ok(it)
        }

        synchronized(transitionLock) {
            // Double-check after acquiring the lock to avoid creating two
            // workers in a load/load race.
            workers[req.modelKey]?.let {
                Log.i(TAG, "getOrLoad(${req.modelKey}): cache hit after lock")
                return BackendResult.Ok(it)
            }

            val backend = createBackendFor(req)
            Log.i(
                TAG,
                "getOrLoad(${req.modelKey}): created backend " +
                    "${backend::class.java.simpleName} (kind=${backend.kind})"
            )
            val worker = ModelWorker(
                modelId = req.modelKey,
                loadRequest = req,
                backend = backend
            )
            Log.i(TAG, "getOrLoad(${req.modelKey}): starting worker and loading model…")
            val loadOutcome = worker.start()
            return when (loadOutcome) {
                is BackendResult.Ok -> {
                    workers[req.modelKey] = worker
                    Log.i(TAG, "getOrLoad(${req.modelKey}): LOAD SUCCESS")
                    BackendResult.Ok(worker)
                }
                is BackendResult.Err -> {
                    Log.e(
                        TAG,
                        "getOrLoad(${req.modelKey}): LOAD FAILED — " +
                            "error=${loadOutcome.error} message=${loadOutcome.message}"
                    )
                    worker.shutdown()
                    loadOutcome
                }
            }
        }
    }

    /**
     * @brief Find an already-loaded worker by its model id string.
     */
    fun get(modelId: String): ModelWorker? = workers[modelId]

    /**
     * @brief Unload and shut down a worker. Idempotent.
     */
    fun unload(modelId: String): Boolean {
        synchronized(transitionLock) {
            val w = workers.remove(modelId) ?: return false
            w.shutdown()
            return true
        }
    }

    /**
     * @brief Shut down every worker. Called from QuickAIService.onDestroy.
     */
    fun shutdownAll() {
        synchronized(transitionLock) {
            val snapshot = workers.values.toList()
            workers.clear()
            snapshot.forEach {
                try {
                    it.shutdown()
                } catch (t: Throwable) {
                    Log.w(TAG, "Error shutting down worker ${it.modelId}", t)
                }
            }
        }
    }

    /**
     * @brief Snapshot of currently-loaded models for the `/v1/models` GET.
     */
    fun list(): List<LoadedModelInfo> =
        workers.values.map {
            LoadedModelInfo(
                modelId = it.modelId,
                architecture = it.architecture,
                backendKind = it.backendKind
            )
        }

    /**
     * @brief Choose a backend implementation based on the model id.
     *
     * This is the Kotlin-level routing point called out in
     * Architecture.md §4 — Gemma4 goes to LiteRT-LM, everything else to
     * the native engine.
     */
    private fun createBackendFor(req: LoadModelRequest): QuickDotAI = when (req.model) {
        ModelId.GEMMA4 -> LiteRTLm(appContext)
        else -> NativeQuickDotAI()
    }

    companion object {
        private const val TAG = "ModelRegistry"
    }
}

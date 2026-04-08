// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    Backend.kt
 * @brief   Abstraction over a loaded model instance.
 *
 * Two implementations exist:
 *  - NativeCausalLmBackend: all non-Gemma4 models, routed through JNI to
 *    libcausallm_api.so.
 *  - LiteRtLmBackend: Gemma4 only, routed through the LiteRT-LM Kotlin API.
 *
 * A Backend instance is owned by exactly one ModelWorker and therefore
 * accessed by exactly one thread at a time — implementations do NOT need
 * to be internally thread-safe.
 */
package com.example.QuickAI.service.backend

import com.example.QuickAI.service.LoadModelRequest
import com.example.QuickAI.service.PerformanceMetrics
import com.example.QuickAI.service.QuickAiError

/**
 * @brief Outcome of a backend call.
 */
sealed class BackendResult<out T> {
    data class Ok<T>(val value: T) : BackendResult<T>()
    data class Err(val error: QuickAiError, val message: String? = null) : BackendResult<Nothing>()
}

/**
 * @brief Common interface implemented by every backend.
 */
interface Backend {
    /** @return a short identifier like "native" or "litert-lm". */
    val kind: String

    /** @return the architecture string reported by the engine, if any. */
    val architecture: String?

    /**
     * @brief Load the model described by [req]. Must be called exactly
     * once in the worker thread before any [run] call.
     */
    fun load(req: LoadModelRequest): BackendResult<Unit>

    /**
     * @brief Run inference on a prompt.
     */
    fun run(prompt: String): BackendResult<String>

    /**
     * @brief Fetch metrics from the last run.
     */
    fun metrics(): BackendResult<PerformanceMetrics>

    /**
     * @brief Release all resources. Called by ModelWorker on shutdown.
     */
    fun close()
}

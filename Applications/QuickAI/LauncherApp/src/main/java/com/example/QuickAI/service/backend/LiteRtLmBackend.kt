// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    LiteRtLmBackend.kt
 * @brief   Backend implementation that runs Gemma-family models through
 *          the LiteRT-LM Kotlin API
 *          (https://github.com/google-ai-edge/LiteRT-LM).
 *
 * This backend is intentionally a stub in the first iteration — the
 * control flow is wired end-to-end (ModelRegistry → ModelWorker → here)
 * but the actual model execution returns UNSUPPORTED until the LiteRT-LM
 * AAR is added to the project. Only the body of `load()` / `run()` will
 * change at that point; no other file in QuickAI will need to move.
 */
package com.example.QuickAI.service.backend

import com.example.QuickAI.service.LoadModelRequest
import com.example.QuickAI.service.PerformanceMetrics
import com.example.QuickAI.service.QuickAiError

/**
 * @brief LiteRT-LM-backed implementation for Gemma4.
 *
 * Non-thread-safe by design — one ModelWorker owns one backend instance.
 */
class LiteRtLmBackend : Backend {

    override val kind: String = "litert-lm"

    override var architecture: String? = "Gemma4ForCausalLM"
        private set

    // TODO(litert-lm): replace with an actual LlmInference handle once the
    // LiteRT-LM artifact is integrated. See Architecture.md §2.7 / §10.
    private var loaded: Boolean = false

    override fun load(req: LoadModelRequest): BackendResult<Unit> {
        // Stub: pretend load succeeds so the wiring can be exercised in
        // integration tests. Actual weight loading comes with the LiteRT-LM
        // integration (Architecture.md §10).
        loaded = true
        return BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "LiteRT-LM backend is a stub; integrate LiteRT-LM AAR to enable Gemma4"
        )
    }

    override fun run(prompt: String): BackendResult<String> {
        return BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "LiteRT-LM backend is a stub; integrate LiteRT-LM AAR to enable Gemma4"
        )
    }

    override fun metrics(): BackendResult<PerformanceMetrics> {
        return BackendResult.Err(
            QuickAiError.UNSUPPORTED,
            "LiteRT-LM backend is a stub; no metrics available"
        )
    }

    override fun close() {
        loaded = false
    }
}

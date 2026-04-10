// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    MainActivity.kt
 * @brief   Standalone sample that drives the :QuickDotAI AAR directly —
 *          no QuickAIService, no REST, no remote process.
 *
 * The user picks a (ModelId, BackendType, QuantizationType) triple, types
 * a prompt, and taps "Run (streaming)". MainActivity:
 *
 *  1. Instantiates [LiteRTLm] for GEMMA4 and [NativeQuickDotAI] for
 *     every other model, both against a single-thread Executor so all
 *     calls touching a given engine are serialised on the same worker
 *     thread (the interface is not internally thread-safe).
 *  2. Calls [QuickDotAI.load] once per chosen (model, quant) pair.
 *  3. Drives [QuickDotAI.runStreaming] with an in-memory StreamSink that
 *     appends each delta to the output TextView on the main thread.
 *
 * This exists as the end-to-end proof that the AAR is genuinely reusable
 * from a third-party app — LauncherApp keeps the HTTP plumbing, but
 * SampleTestAPP shows a client that needs none of it. See Architecture.md
 * §2.9 for the big-picture view.
 */
package com.example.sampletestapp

import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.quickdotai.BackendResult
import com.example.quickdotai.BackendType
import com.example.quickdotai.LiteRTLm
import com.example.quickdotai.LoadModelRequest
import com.example.quickdotai.ModelId
import com.example.quickdotai.NativeQuickDotAI
import com.example.quickdotai.QuantizationType
import com.example.quickdotai.QuickAiError
import com.example.quickdotai.QuickDotAI
import com.example.quickdotai.StreamSink
import java.util.concurrent.Executor
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    /**
     * @brief Single-thread executor that every [QuickDotAI] call is
     * dispatched on. [QuickDotAI] implementations are NOT thread-safe —
     * pinning every load / run / metrics / close to one thread mirrors
     * what QuickAIService's ModelWorker does internally.
     */
    private val engineExecutor: Executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "SampleTestAPP-Engine").apply { isDaemon = true }
    }

    /** Posts Runnables back to the Looper thread for UI updates. */
    private val mainHandler by lazy { android.os.Handler(mainLooper) }

    /**
     * The currently-loaded engine, if any. Guarded by only being touched
     * from [engineExecutor]. [loadedKey] is only read/written from the
     * same executor.
     */
    @Volatile
    private var engine: QuickDotAI? = null

    @Volatile
    private var loadedKey: String? = null

    // --- UI state -----------------------------------------------------

    private lateinit var modelSpinner: Spinner
    private lateinit var backendSpinner: Spinner
    private lateinit var quantSpinner: Spinner
    private lateinit var modelPathField: EditText
    private lateinit var promptField: EditText
    private lateinit var statusView: TextView
    private lateinit var outputView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val pad = (resources.displayMetrics.density * 16).toInt()
        val scrollRoot = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT)
            isFillViewport = true
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
            gravity = Gravity.TOP
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }

        val title = TextView(this).apply {
            text = "QuickDotAI AAR sample (in-process)"
            textSize = 20f
        }
        root.addView(title)

        val subtitle = TextView(this).apply {
            text = "No service, no REST — runs the AAR directly."
            textSize = 13f
            setPadding(0, 0, 0, pad / 2)
        }
        root.addView(subtitle)

        // --- Model / backend / quantization selectors ----------------
        root.addView(labelView("Model"))
        modelSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                ModelId.values().map { it.name }
            )
            // Default to GEMMA4 so the LiteRTLm path is exercised with a
            // single Load click — the more interesting bring-up case.
            setSelection(ModelId.values().indexOf(ModelId.GEMMA4))
        }
        root.addView(modelSpinner)

        root.addView(labelView("Compute backend"))
        backendSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                BackendType.values().map { it.name }
            )
            setSelection(BackendType.values().indexOf(BackendType.GPU))
        }
        root.addView(backendSpinner)

        root.addView(labelView("Quantization"))
        quantSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                QuantizationType.values().map { it.name }
            )
            setSelection(QuantizationType.values().indexOf(QuantizationType.W4A32))
        }
        root.addView(quantSpinner)

        root.addView(labelView("Model path"))
        modelPathField = EditText(this).apply {
            hint = "Absolute path to the model file/dir"
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        root.addView(modelPathField)

        // Keep the model-path field in sync with the current (model,
        // quantization) selection so a single Load click works without
        // the user having to type a path manually. The user can still
        // edit the field after the fact — a subsequent spinner change
        // will just overwrite it with the new default.
        val pathSyncListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?, view: View?, position: Int, id: Long
            ) {
                modelPathField.setText(defaultModelPathFor(selectedModelId(), selectedQuant()))
            }
            override fun onNothingSelected(parent: AdapterView<*>?) { /* no-op */ }
        }
        modelSpinner.onItemSelectedListener = pathSyncListener
        quantSpinner.onItemSelectedListener = pathSyncListener
        // Seed the field synchronously for the initial selection (the
        // spinner listeners fire asynchronously after setContentView).
        modelPathField.setText(defaultModelPathFor(selectedModelId(), selectedQuant()))

        val loadBtn = Button(this).apply {
            text = "Load model"
            setOnClickListener { onLoadClicked() }
        }
        root.addView(loadBtn)

        root.addView(labelView("Prompt"))
        promptField = EditText(this).apply {
            hint = "Type a prompt…"
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        root.addView(promptField)

        val runBtn = Button(this).apply {
            text = "Run (streaming)"
            setOnClickListener { onRunClicked() }
        }
        root.addView(runBtn)

        val metricsBtn = Button(this).apply {
            text = "Fetch metrics"
            setOnClickListener { onMetricsClicked() }
        }
        root.addView(metricsBtn)

        val unloadBtn = Button(this).apply {
            text = "Unload"
            setOnClickListener { onUnloadClicked() }
        }
        root.addView(unloadBtn)

        statusView = TextView(this).apply {
            text = "Idle."
            setPadding(0, pad / 2, 0, 0)
        }
        root.addView(statusView)

        outputView = TextView(this).apply {
            text = ""
            setPadding(0, pad / 2, 0, 0)
            setTextIsSelectable(true)
        }
        root.addView(outputView)

        scrollRoot.addView(root)
        setContentView(scrollRoot)
    }

    override fun onDestroy() {
        // Fire-and-forget close on the engine thread so we don't leak the
        // native model handle / LiteRT-LM Engine on config changes.
        val e = engine
        engine = null
        loadedKey = null
        if (e != null) {
            engineExecutor.execute {
                try {
                    e.close()
                } catch (_: Throwable) { /* best effort */ }
            }
        }
        super.onDestroy()
    }

    // --- button handlers ----------------------------------------------

    private fun onLoadClicked() {
        val model = ModelId.valueOf(modelSpinner.selectedItem as String)
        val backend = BackendType.valueOf(backendSpinner.selectedItem as String)
        val quant = QuantizationType.valueOf(quantSpinner.selectedItem as String)
        val modelPath = modelPathField.text.toString().trim().ifEmpty { null }
        val req = LoadModelRequest(
            backend = backend,
            model = model,
            quantization = quant,
            modelPath = modelPath
        )
        setStatus("Loading ${req.modelKey}…")
        outputView.text = ""

        engineExecutor.execute {
            // If a different model is already loaded, swap it out so the
            // sample stays simple (one engine at a time).
            if (loadedKey != null && loadedKey != req.modelKey) {
                try {
                    engine?.close()
                } catch (_: Throwable) { /* best effort */ }
                engine = null
                loadedKey = null
            }
            if (engine != null && loadedKey == req.modelKey) {
                setStatus("Already loaded: ${req.modelKey}")
                return@execute
            }

            val newEngine: QuickDotAI = when (req.model) {
                ModelId.GEMMA4 -> LiteRTLm(applicationContext)
                else -> NativeQuickDotAI()
            }
            when (val r = newEngine.load(req)) {
                is BackendResult.Ok -> {
                    engine = newEngine
                    loadedKey = req.modelKey
                    setStatus(
                        "Loaded ${req.modelKey} " +
                            "(${newEngine.kind}, arch=${newEngine.architecture ?: "?"})"
                    )
                }
                is BackendResult.Err -> {
                    try {
                        newEngine.close()
                    } catch (_: Throwable) { /* best effort */ }
                    setStatus("Load failed: [${r.error.name}] ${r.message ?: ""}")
                }
            }
        }
    }

    private fun onRunClicked() {
        val prompt = promptField.text.toString()
        if (prompt.isBlank()) {
            setStatus("Prompt is empty.")
            return
        }
        // Clear any previous output before we start streaming.
        mainHandler.post { outputView.text = "" }
        setStatus("Running…")

        engineExecutor.execute {
            val e = engine
            if (e == null) {
                setStatus("No model loaded — tap Load first.")
                return@execute
            }
            val sink = object : StreamSink {
                override fun onDelta(text: String) {
                    mainHandler.post { outputView.append(text) }
                }
                override fun onDone() {
                    setStatus("Done.")
                }
                override fun onError(error: QuickAiError, message: String?) {
                    setStatus("Run failed: [${error.name}] ${message ?: ""}")
                }
            }
            // runStreaming blocks until the backend finishes; that's
            // fine because we're already off the main thread.
            try {
                e.runStreaming(prompt, sink)
            } catch (t: Throwable) {
                setStatus("Run threw: ${t.message}")
            }
        }
    }

    private fun onMetricsClicked() {
        engineExecutor.execute {
            val e = engine
            if (e == null) {
                setStatus("No model loaded.")
                return@execute
            }
            when (val r = e.metrics()) {
                is BackendResult.Ok -> {
                    val m = r.value
                    val text = buildString {
                        append("prefill: ").append(m.prefillTokens).append(" toks / ")
                            .append(m.prefillDurationMs).append(" ms\n")
                        append("gen:     ").append(m.generationTokens).append(" toks / ")
                            .append(m.generationDurationMs).append(" ms\n")
                        append("total:   ").append(m.totalDurationMs).append(" ms\n")
                        append("init:    ").append(m.initializationDurationMs).append(" ms\n")
                        append("peak:    ").append(m.peakMemoryKb).append(" KB")
                    }
                    mainHandler.post { outputView.text = text }
                    setStatus("Metrics fetched.")
                }
                is BackendResult.Err ->
                    setStatus("Metrics failed: [${r.error.name}] ${r.message ?: ""}")
            }
        }
    }

    private fun onUnloadClicked() {
        engineExecutor.execute {
            val e = engine
            if (e == null) {
                setStatus("Nothing to unload.")
                return@execute
            }
            when (val r = e.unload()) {
                is BackendResult.Ok -> {
                    // Keep the engine instance alive so onDestroy can still
                    // call close(), but clear loadedKey so a subsequent Load
                    // tap creates a fresh engine.
                    loadedKey = null
                    setStatus("Unloaded.")
                }
                is BackendResult.Err ->
                    setStatus("Unload failed: [${r.error.name}] ${r.message ?: ""}")
            }
        }
    }

    // --- helpers -------------------------------------------------------

    private fun labelView(text: String): TextView {
        val pad = (resources.displayMetrics.density * 6).toInt()
        return TextView(this).apply {
            this.text = text
            textSize = 13f
            setPadding(0, pad, 0, 0)
        }
    }

    private fun setStatus(text: String) {
        mainHandler.post { statusView.text = text }
    }

    /**
     * @brief Reads the current value of the model spinner.
     *
     * Guarded with a try/catch because AdapterView listeners can fire
     * once before the adapter is fully wired up on some API levels.
     */
    private fun selectedModelId(): ModelId = try {
        ModelId.valueOf(modelSpinner.selectedItem as String)
    } catch (_: Throwable) {
        ModelId.GEMMA4
    }

    private fun selectedQuant(): QuantizationType = try {
        QuantizationType.valueOf(quantSpinner.selectedItem as String)
    } catch (_: Throwable) {
        QuantizationType.W4A32
    }

    /**
     * @brief Builds the default on-device model path for the given
     * (model, quantization) pair rooted in this app's external files
     * dir, so the path lines up with the native C API's hardcoded
     * `./models/<name>-<quant>` prefix (resolve_model_path() in
     * quick_dot_ai_api.cpp).
     *
     * Layout — e.g. for QWEN3_0_6B + W4A32:
     *   /sdcard/Android/data/com.example.sampletestapp/files/
     *       models/qwen3-0.6b-w4a32
     *
     * For GEMMA4 the path is the `.litertlm` model file (LiteRT-LM
     * takes an explicit file path), not a directory.
     */
    private fun defaultModelPathFor(model: ModelId, quant: QuantizationType): String {
        val externalFiles = applicationContext.getExternalFilesDir(null)
        val base = externalFiles?.absolutePath
            ?: "/sdcard/Android/data/$packageName/files"
        return when (model) {
            ModelId.GEMMA4 ->
                "$base/models/gemma-4-E2B-it/gemma-4-E2B-it.litertlm"
            ModelId.QWEN3_0_6B ->
                "$base/models/qwen3-0.6b${quantizationSuffix(quant)}"
        }
    }

    /**
     * @brief Mirror of `get_quantization_suffix` in
     * Applications/CausalLM/api/quick_dot_ai_api.cpp — kept in sync so the
     * default path stays valid for whichever quant the user picks.
     */
    private fun quantizationSuffix(quant: QuantizationType): String = when (quant) {
        QuantizationType.W4A32 -> "-w4a32"
        QuantizationType.W16A16 -> "-w16a16"
        QuantizationType.W8A16 -> "-w8a16"
        QuantizationType.W32A32 -> "-w32a32"
        QuantizationType.UNKNOWN -> "-w4a32"
    }
}

// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    MainActivity.kt
 * @brief   Minimal ClientApp UI that exercises the QuickAIService REST
 *          endpoint. Lets the user choose a model, load it, run a
 *          prompt, and view the output + metrics. See Architecture.md
 *          §2.9.
 *
 * UI is built programmatically to keep the sample free of XML layout
 * boilerplate.
 */
package com.example.clientapp

import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.example.clientapp.api.ApiResult
import com.example.clientapp.api.LoadModelRequest
import com.example.clientapp.api.ModelId
import com.example.clientapp.api.QuantizationType
import com.example.clientapp.api.QuickAiClient
import com.example.clientapp.api.RunModelRequest
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private val client = QuickAiClient()

    private lateinit var connectionView: TextView
    private lateinit var modelSpinner: Spinner
    private lateinit var quantSpinner: Spinner
    private lateinit var modelPathField: EditText
    private lateinit var promptField: EditText
    private lateinit var outputView: TextView
    private lateinit var statusView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val pad = (resources.displayMetrics.density * 16).toInt()
        // A vertical ScrollView wraps everything so long Gemma4 outputs
        // (which can easily exceed the visible area) remain readable.
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

        // --- Connection indicator bar (always at the very top) ---------
        // This view is driven by the periodic /v1/health poll started in
        // onCreate() below. It reports the live reachability of the
        // QuickAIService REST endpoint on 127.0.0.1:3453 independently of
        // whatever operation the user most recently issued.
        connectionView = TextView(this).apply {
            text = "● Connecting…"
            textSize = 16f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(COLOR_UNKNOWN)
            setPadding(0, 0, 0, pad / 2)
        }
        root.addView(connectionView)

        val title = TextView(this).apply {
            text = "QuickAI Client"
            textSize = 22f
        }
        root.addView(title)

        // Explicit, user-initiated handshake. Distinct from the periodic
        // health poll above: this hits POST /v1/connect and reports the
        // result in statusView so the user can see exactly what happened
        // without fighting the background polling loop.
        val connectBtn = Button(this).apply {
            text = "Connect"
            setOnClickListener { onConnectClicked() }
        }
        root.addView(connectBtn)

        modelSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                ModelId.values().map { it.name }
            )
        }
        root.addView(modelSpinner)

        quantSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                QuantizationType.values().map { it.name }
            )
            setSelection(
                QuantizationType.values().indexOf(QuantizationType.W4A32)
            )
        }
        root.addView(quantSpinner)

        modelPathField = EditText(this).apply {
            hint = "Model path (required for GEMMA4 / LiteRT-LM)"
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
            // TEST HARDCODING: prefill with the known-good Gemma4 path
            // from gemma-model-path.md so the LiteRT-LM flow works with a
            // single Load click. The user can still edit/clear it.
            setText(TEST_GEMMA4_MODEL_PATH)
        }
        root.addView(modelPathField)

        val loadBtn = Button(this).apply {
            text = "Load model"
            setOnClickListener { onLoadClicked() }
        }
        root.addView(loadBtn)

        promptField = EditText(this).apply {
            hint = "Enter a prompt…"
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        root.addView(promptField)

        val runBtn = Button(this).apply {
            text = "Run"
            setOnClickListener { onRunClicked() }
        }
        root.addView(runBtn)

        val metricsBtn = Button(this).apply {
            text = "Fetch metrics"
            setOnClickListener { onMetricsClicked() }
        }
        root.addView(metricsBtn)

        statusView = TextView(this).apply { text = "Service not contacted yet." }
        root.addView(statusView)

        outputView = TextView(this).apply {
            text = ""
            setPadding(0, pad, 0, 0)
            // Allow the user to long-press to copy the generated output
            // — handy for inspecting Gemma4 responses.
            setTextIsSelectable(true)
        }
        root.addView(outputView)

        scrollRoot.addView(root)
        setContentView(scrollRoot)

        // Periodic health poll. Runs only while the activity is at least
        // STARTED, so it automatically pauses in the background and
        // resumes on foreground — no manual start/stop bookkeeping.
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                while (true) {
                    updateConnectionStatus()
                    delay(HEALTH_POLL_INTERVAL_MS)
                }
            }
        }
    }

    /**
     * @brief Fire one /v1/health request and paint [connectionView] with
     * the result. Called from the polling loop in onCreate().
     */
    private suspend fun updateConnectionStatus() {
        when (val r = client.health()) {
            is ApiResult.Ok -> {
                connectionView.text = "● Connected (port ${r.value.port})"
                connectionView.setTextColor(COLOR_CONNECTED)
            }
            is ApiResult.Err -> {
                connectionView.text = "● Disconnected — ${r.message}"
                connectionView.setTextColor(COLOR_DISCONNECTED)
            }
        }
    }

    private fun onConnectClicked() {
        statusView.text = "Connecting…"
        lifecycleScope.launch {
            when (val r = client.connect()) {
                is ApiResult.Ok ->
                    statusView.text =
                        "Connected: ${r.value.message} (port ${r.value.port})"
                is ApiResult.Err ->
                    statusView.text = "Connect failed: [${r.errorCode}] ${r.message}"
            }
        }
    }

    private fun onLoadClicked() {
        val model = ModelId.valueOf(modelSpinner.selectedItem as String)
        val quant = QuantizationType.valueOf(quantSpinner.selectedItem as String)
        val modelPath = modelPathField.text.toString().trim().ifEmpty { null }
        statusView.text = "Loading $model [$quant]…"
        lifecycleScope.launch {
            val result = client.loadModel(
                LoadModelRequest(
                    model = model,
                    quantization = quant,
                    modelPath = modelPath
                )
            )
            statusView.text = when (result) {
                is ApiResult.Ok ->
                    "Loaded ${result.value.modelId} (${result.value.architecture ?: "?"})"
                is ApiResult.Err ->
                    "Load failed: [${result.errorCode}] ${result.message}"
            }
        }
    }

    private fun onRunClicked() {
        val model = ModelId.valueOf(modelSpinner.selectedItem as String)
        val quant = QuantizationType.valueOf(quantSpinner.selectedItem as String)
        val modelId = "${model.name}:${quant.name}"
        val prompt = promptField.text.toString()
        if (prompt.isBlank()) {
            statusView.text = "Prompt is empty."
            return
        }
        statusView.text = "Running on $modelId…"
        outputView.text = ""
        lifecycleScope.launch {
            val result = client.runModel(modelId, RunModelRequest(prompt))
            when (result) {
                is ApiResult.Ok -> {
                    statusView.text = "Done."
                    outputView.text = result.value.output.orEmpty()
                }
                is ApiResult.Err -> {
                    statusView.text = "Run failed: [${result.errorCode}] ${result.message}"
                }
            }
        }
    }

    private fun onMetricsClicked() {
        val model = ModelId.valueOf(modelSpinner.selectedItem as String)
        val quant = QuantizationType.valueOf(quantSpinner.selectedItem as String)
        val modelId = "${model.name}:${quant.name}"
        lifecycleScope.launch {
            when (val r = client.getMetrics(modelId)) {
                is ApiResult.Ok -> outputView.text = buildString {
                    val m = r.value.metrics
                    if (m == null) {
                        append("no metrics")
                    } else {
                        append("prefill: ").append(m.prefillTokens).append(" toks / ")
                            .append(m.prefillDurationMs).append(" ms\n")
                        append("gen:     ").append(m.generationTokens).append(" toks / ")
                            .append(m.generationDurationMs).append(" ms\n")
                        append("total:   ").append(m.totalDurationMs).append(" ms\n")
                        append("init:    ").append(m.initializationDurationMs).append(" ms\n")
                        append("peak:    ").append(m.peakMemoryKb).append(" KB")
                    }
                }
                is ApiResult.Err ->
                    statusView.text = "Metrics failed: [${r.errorCode}] ${r.message}"
            }
        }
    }

    private companion object {
        /** How often the connection indicator re-probes /v1/health. */
        const val HEALTH_POLL_INTERVAL_MS: Long = 3_000L

        // Material-ish greens / reds. Deliberately inline so we don't need
        // a colors.xml just for two swatches.
        val COLOR_CONNECTED = Color.parseColor("#2E7D32")
        val COLOR_DISCONNECTED = Color.parseColor("#C62828")
        val COLOR_UNKNOWN = Color.parseColor("#757575")

        /**
         * TEST ONLY — absolute on-device path to the Gemma-4 E2B-IT
         * `.litertlm` model, kept in sync with gemma-model-path.md at the
         * repo root and the server-side constant in LiteRtLmBackend.kt.
         * Prefilled into the model-path field so the LiteRT-LM flow works
         * with a single Load click during bring-up.
         */
        const val TEST_GEMMA4_MODEL_PATH: String =
            "/data/local/tmp/Quick.AI/models/gemma-4-E2B-it/gemma-4-E2B-it.litertlm"
    }
}

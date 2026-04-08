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

import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.clientapp.api.ApiResult
import com.example.clientapp.api.LoadModelRequest
import com.example.clientapp.api.ModelId
import com.example.clientapp.api.QuantizationType
import com.example.clientapp.api.QuickAiClient
import com.example.clientapp.api.RunModelRequest
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private val client = QuickAiClient()

    private lateinit var modelSpinner: Spinner
    private lateinit var quantSpinner: Spinner
    private lateinit var promptField: EditText
    private lateinit var outputView: TextView
    private lateinit var statusView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val pad = (resources.displayMetrics.density * 16).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
            gravity = Gravity.TOP
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT)
        }

        val title = TextView(this).apply {
            text = "QuickAI Client"
            textSize = 22f
        }
        root.addView(title)

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
        }
        root.addView(outputView)

        setContentView(root)

        // Health probe on startup.
        lifecycleScope.launch {
            when (val r = client.health()) {
                is ApiResult.Ok -> statusView.text = "Service OK on port ${r.value.port}"
                is ApiResult.Err -> statusView.text =
                    "Service not reachable: ${r.message} (is LauncherApp running?)"
            }
        }
    }

    private fun onLoadClicked() {
        val model = ModelId.valueOf(modelSpinner.selectedItem as String)
        val quant = QuantizationType.valueOf(quantSpinner.selectedItem as String)
        statusView.text = "Loading $model [$quant]…"
        lifecycleScope.launch {
            val result = client.loadModel(
                LoadModelRequest(model = model, quantization = quant)
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
}

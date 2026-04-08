package com.quickai.client.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.gson.GsonBuilder
import kotlinx.coroutines.launch

class ApiTestFragment : Fragment() {

    private lateinit var logText: TextView
    private val gson = GsonBuilder().setPrettyPrinting().create()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, saved: Bundle?): View {
        val layout = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }

        layout.addView(TextView(requireContext()).apply {
            text = "API Test"
            textSize = 18f
        })

        // Individual endpoint buttons
        val buttons = listOf(
            "GET /v1/health" to { testHealth() },
            "GET /v1/models" to { testModels() },
            "GET /v1/metrics" to { testMetrics() },
            "POST /v1/engine/unload" to { testUnload() },
            "Run Full Scenario" to { testFullScenario() },
        )
        for ((label, action) in buttons) {
            layout.addView(Button(requireContext()).apply {
                text = label
                setOnClickListener { action() }
            })
        }

        // Log output
        val scrollView = ScrollView(requireContext()).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
            )
        }
        logText = TextView(requireContext()).apply {
            text = ""
            textSize = 12f
            setTextIsSelectable(true)
            typeface = android.graphics.Typeface.MONOSPACE
        }
        scrollView.addView(logText)
        layout.addView(scrollView)

        // Clear button
        layout.addView(Button(requireContext()).apply {
            text = "Clear Log"
            setOnClickListener { logText.text = "" }
        })

        return layout
    }

    private fun log(msg: String) {
        logText.append("$msg\n")
    }

    private fun testHealth() {
        val client = (activity as MainActivity).client
        lifecycleScope.launch {
            log(">>> GET /v1/health")
            val result = client.health()
            result.onSuccess { log("<<< ${gson.toJson(it)}") }
                .onFailure { log("<<< ERROR: ${it.message}") }
        }
    }

    private fun testModels() {
        val client = (activity as MainActivity).client
        lifecycleScope.launch {
            log(">>> GET /v1/models")
            val result = client.getModels()
            result.onSuccess { log("<<< ${gson.toJson(it)}") }
                .onFailure { log("<<< ERROR: ${it.message}") }
        }
    }

    private fun testMetrics() {
        val client = (activity as MainActivity).client
        lifecycleScope.launch {
            log(">>> GET /v1/metrics")
            val result = client.getMetrics()
            result.onSuccess { log("<<< ${gson.toJson(it)}") }
                .onFailure { log("<<< ERROR: ${it.message}") }
        }
    }

    private fun testUnload() {
        val client = (activity as MainActivity).client
        lifecycleScope.launch {
            log(">>> POST /v1/engine/unload")
            val result = client.unloadModel()
            result.onSuccess { log("<<< ${gson.toJson(it)}") }
                .onFailure { log("<<< ERROR: ${it.message}") }
        }
    }

    private fun testFullScenario() {
        val client = (activity as MainActivity).client
        lifecycleScope.launch {
            log("=== Full Scenario Test ===")

            // 1. Health
            log("\n[1/5] Health check")
            client.health().onSuccess { log("  OK: ${it.status}") }
                .onFailure { log("  FAIL: ${it.message}"); return@launch }

            // 2. Models
            log("[2/5] Get models")
            client.getModels().onSuccess { log("  Found ${it.models.size} models") }
                .onFailure { log("  FAIL: ${it.message}"); return@launch }

            // 3. Load
            log("[3/5] Load model (cpu, qwen3-0.6b)")
            client.loadModel("cpu", "qwen3-0.6b", 1)
                .onSuccess { log("  Loaded: ${it.status}") }
                .onFailure { log("  FAIL: ${it.message}"); return@launch }

            // 4. Generate
            log("[4/5] Generate")
            client.generate("What is 2+2?")
                .onSuccess {
                    log("  Output: ${it.text.take(100)}...")
                    it.metrics?.let { m ->
                        log("  Prefill: ${m.prefill_tokens} tok, Decode: ${m.generation_tokens} tok")
                    }
                }
                .onFailure { log("  FAIL: ${it.message}"); return@launch }

            // 5. Unload
            log("[5/5] Unload")
            client.unloadModel()
                .onSuccess { log("  Unloaded: ${it.status}") }
                .onFailure { log("  FAIL: ${it.message}") }

            log("\n=== Scenario Complete ===")
        }
    }
}

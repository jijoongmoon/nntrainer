package com.quickai.client.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class BenchmarkFragment : Fragment() {

    private lateinit var resultText: TextView
    private lateinit var promptEdit: EditText

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, saved: Bundle?): View {
        val layout = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }

        layout.addView(TextView(requireContext()).apply {
            text = "Benchmark"
            textSize = 18f
        })

        promptEdit = EditText(requireContext()).apply {
            hint = "Benchmark prompt (default: 'Explain quantum computing')"
            setPadding(0, 16, 0, 16)
        }
        layout.addView(promptEdit)

        val runBtn = Button(requireContext()).apply {
            text = "Run Benchmark"
            setOnClickListener { runBenchmark() }
        }
        layout.addView(runBtn)

        val scrollView = ScrollView(requireContext()).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
            )
        }
        resultText = TextView(requireContext()).apply {
            text = "Load a model first, then run benchmark.\n"
            textSize = 13f
            setTextIsSelectable(true)
        }
        scrollView.addView(resultText)
        layout.addView(scrollView)

        return layout
    }

    private fun runBenchmark() {
        val prompt = promptEdit.text.toString().ifEmpty { "Explain quantum computing in simple terms." }
        val client = (activity as MainActivity).client

        resultText.append("\n--- Benchmark Start ---\n")
        resultText.append("Prompt: $prompt\n")

        lifecycleScope.launch {
            val startTime = System.currentTimeMillis()
            val result = client.generate(prompt, useChatTemplate = true)
            val endTime = System.currentTimeMillis()

            result.onSuccess { resp ->
                val outputLen = resp.text.length
                resultText.append("Output length: $outputLen chars\n")
                resultText.append("E2E time: ${endTime - startTime}ms\n")

                resp.metrics?.let { m ->
                    val prefillTokS = if (m.prefill_duration_ms > 0)
                        m.prefill_tokens / (m.prefill_duration_ms / 1000.0) else 0.0
                    val decodeTokS = if (m.generation_duration_ms > 0)
                        m.generation_tokens / (m.generation_duration_ms / 1000.0) else 0.0

                    resultText.append("\nPrefill:\n")
                    resultText.append("  Tokens: ${m.prefill_tokens}\n")
                    resultText.append("  Duration: ${m.prefill_duration_ms.toInt()}ms\n")
                    resultText.append("  Speed: ${"%.1f".format(prefillTokS)} tok/s\n")
                    resultText.append("\nDecode:\n")
                    resultText.append("  Tokens: ${m.generation_tokens}\n")
                    resultText.append("  Duration: ${m.generation_duration_ms.toInt()}ms\n")
                    resultText.append("  Speed: ${"%.1f".format(decodeTokS)} tok/s\n")
                    resultText.append("\nTotal: ${m.total_duration_ms.toInt()}ms\n")
                    resultText.append("Peak memory: ${m.peak_memory_kb / 1024}MB\n")
                }
                resultText.append("--- Benchmark End ---\n")
            }.onFailure { e ->
                resultText.append("Error: ${e.message}\n")
            }
        }
    }
}

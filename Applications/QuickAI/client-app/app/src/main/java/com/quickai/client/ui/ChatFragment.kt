package com.quickai.client.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class ChatFragment : Fragment() {

    private lateinit var outputText: TextView
    private lateinit var inputEdit: EditText
    private lateinit var sendButton: Button
    private lateinit var metricsText: TextView

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, saved: Bundle?): View {
        val layout = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }

        // Output area (scrollable)
        val scrollView = ScrollView(requireContext()).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
            )
        }
        outputText = TextView(requireContext()).apply {
            text = "Load a model first, then start chatting.\n"
            textSize = 14f
            setTextIsSelectable(true)
        }
        scrollView.addView(outputText)
        layout.addView(scrollView)

        // Metrics
        metricsText = TextView(requireContext()).apply {
            text = ""
            textSize = 11f
            setPadding(0, 8, 0, 8)
        }
        layout.addView(metricsText)

        // Input area
        val inputRow = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        inputEdit = EditText(requireContext()).apply {
            hint = "Enter prompt..."
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        inputRow.addView(inputEdit)

        sendButton = Button(requireContext()).apply {
            text = "Send"
            setOnClickListener { onSend() }
        }
        inputRow.addView(sendButton)
        layout.addView(inputRow)

        return layout
    }

    private fun onSend() {
        val prompt = inputEdit.text.toString().trim()
        if (prompt.isEmpty()) return

        val client = (activity as MainActivity).client
        outputText.append("\n[User] $prompt\n")
        inputEdit.setText("")
        sendButton.isEnabled = false

        lifecycleScope.launch {
            val result = client.generate(prompt)
            result.onSuccess { resp ->
                outputText.append("[AI] ${resp.text}\n")
                resp.metrics?.let { m ->
                    metricsText.text = "Prefill: ${m.prefill_tokens} tok / ${m.prefill_duration_ms.toInt()}ms | " +
                        "Decode: ${m.generation_tokens} tok / ${m.generation_duration_ms.toInt()}ms | " +
                        "Total: ${m.total_duration_ms.toInt()}ms"
                }
            }.onFailure { e ->
                outputText.append("[Error] ${e.message}\n")
            }
            sendButton.isEnabled = true
        }
    }
}

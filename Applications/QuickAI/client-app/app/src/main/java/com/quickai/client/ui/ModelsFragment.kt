package com.quickai.client.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class ModelsFragment : Fragment() {

    private lateinit var modelList: LinearLayout
    private lateinit var statusText: TextView
    private lateinit var backendSpinner: Spinner

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, saved: Bundle?): View {
        val layout = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }

        statusText = TextView(requireContext()).apply {
            text = "Models"
            textSize = 18f
        }
        layout.addView(statusText)

        // Backend selector
        val backendRow = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, 16, 0, 16)
        }
        backendRow.addView(TextView(requireContext()).apply { text = "Backend: "; textSize = 14f })
        backendSpinner = Spinner(requireContext()).apply {
            adapter = ArrayAdapter(
                requireContext(),
                android.R.layout.simple_spinner_dropdown_item,
                listOf("cpu", "gpu", "npu", "gpu2")
            )
        }
        backendRow.addView(backendSpinner)
        layout.addView(backendRow)

        // Refresh button
        val refreshBtn = Button(requireContext()).apply {
            text = "Refresh Models"
            setOnClickListener { refreshModels() }
        }
        layout.addView(refreshBtn)

        // Unload button
        val unloadBtn = Button(requireContext()).apply {
            text = "Unload Current Model"
            setOnClickListener { unloadModel() }
        }
        layout.addView(unloadBtn)

        // Model list
        val scrollView = ScrollView(requireContext()).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
            )
        }
        modelList = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
        }
        scrollView.addView(modelList)
        layout.addView(scrollView)

        refreshModels()
        return layout
    }

    private fun refreshModels() {
        val client = (activity as MainActivity).client
        lifecycleScope.launch {
            val result = client.getModels()
            result.onSuccess { resp ->
                modelList.removeAllViews()
                for (model in resp.models) {
                    val row = LinearLayout(requireContext()).apply {
                        orientation = LinearLayout.HORIZONTAL
                        setPadding(0, 12, 0, 12)
                    }
                    row.addView(TextView(requireContext()).apply {
                        text = "${model.name}\n(${model.backends.joinToString(", ")})"
                        textSize = 14f
                        layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                    })
                    row.addView(Button(requireContext()).apply {
                        text = "Load"
                        setOnClickListener { loadModel(model.id) }
                    })
                    modelList.addView(row)
                }
                statusText.text = "Models (${resp.models.size})"
            }.onFailure { e ->
                statusText.text = "Error: ${e.message}"
            }
        }
    }

    private fun loadModel(modelId: String) {
        val backend = backendSpinner.selectedItem.toString()
        val client = (activity as MainActivity).client
        statusText.text = "Loading $modelId on $backend..."

        lifecycleScope.launch {
            val result = client.loadModel(backend, modelId)
            result.onSuccess {
                statusText.text = "Loaded: $modelId on $backend"
            }.onFailure { e ->
                statusText.text = "Load failed: ${e.message}"
            }
        }
    }

    private fun unloadModel() {
        val client = (activity as MainActivity).client
        lifecycleScope.launch {
            val result = client.unloadModel()
            result.onSuccess { statusText.text = "Model unloaded" }
                .onFailure { e -> statusText.text = "Unload failed: ${e.message}" }
        }
    }
}

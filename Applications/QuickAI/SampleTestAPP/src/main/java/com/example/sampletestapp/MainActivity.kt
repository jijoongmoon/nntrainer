// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    MainActivity.kt
 * @brief   Standalone sample that drives the :QuickDotAI AAR directly —
 *          no QuickAIService, no REST, no remote process.
 *
 * The user picks a (ModelId, BackendType, QuantizationType) triple, types
 * a prompt, optionally picks an image via the system photo picker, and
 * taps "Run (streaming)". MainActivity:
 *
 *  1. Instantiates [LiteRTLm] for GEMMA4 and [NativeQuickDotAI] for
 *     every other model, both against a single-thread Executor so all
 *     calls touching a given engine are serialised on the same worker
 *     thread (the interface is not internally thread-safe).
 *  2. Calls [QuickDotAI.load] once per chosen (model, quant) pair.
 *     For GEMMA4 it auto-populates [LoadModelRequest.visionBackend] so
 *     the multimodal path is armed from load time.
 *  3. Drives [QuickDotAI.runStreaming] (text-only) or
 *     [QuickDotAI.runMultimodalStreaming] (when an image is selected)
 *     with an in-memory StreamSink that appends each delta to the
 *     output TextView on the main thread.
 *
 * This exists as the end-to-end proof that the AAR is genuinely reusable
 * from a third-party app — LauncherApp keeps the HTTP plumbing, but
 * SampleTestAPP shows a client that needs none of it. See Architecture.md
 * §2.9 for the big-picture view.
 */
package com.example.sampletestapp

import android.net.Uri
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
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.quickdotai.BackendResult
import com.example.quickdotai.BackendType
import com.example.quickdotai.LiteRTLm
import com.example.quickdotai.LoadModelRequest
import com.example.quickdotai.ModelId
import com.example.quickdotai.NativeQuickDotAI
import com.example.quickdotai.PromptPart
import com.example.quickdotai.QuickAiChatMessage
import com.example.quickdotai.QuickAiChatRole
import com.example.quickdotai.QuickAiChatSamplingConfig
import com.example.quickdotai.QuickAiChatSessionConfig
import com.example.quickdotai.QuickAiChatTemplateKwargs
import com.example.quickdotai.QuantizationType
import com.example.quickdotai.QuickAiError
import com.example.quickdotai.QuickDotAI
import com.example.quickdotai.StreamSink
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

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
    private lateinit var imageStatusView: TextView
    private lateinit var statusView: TextView
    private lateinit var outputView: TextView

    // Chat session UI
    private lateinit var chatSessionStatusView: TextView
    private lateinit var chatSystemPromptField: EditText
    private lateinit var chatTemperatureField: EditText
    private lateinit var chatTopKField: EditText
    private lateinit var chatTopPField: EditText
    private lateinit var chatSeedField: EditText
    private lateinit var chatEnableThinkingSpinner: Spinner
    private lateinit var chatPromptField: EditText

    // OpenAI-style messages input UI
    private lateinit var openAIMessagesField: EditText

    /**
     * @brief Raw bytes of the most recently picked image, or null if
     * no image is currently selected.
     *
     * Written by the image-reader thread spawned in
     * [readImageBytesAsync] and by [onClearImageClicked]; read by the
     * engine thread inside [onRunClicked]. `@Volatile` is enough —
     * we never need a read-modify-write cycle on this field.
     */
    @Volatile
    private var selectedImageBytes: ByteArray? = null

    /**
     * @brief ActivityResult launcher for the Android system photo picker.
     *
     * `PickVisualMedia` does NOT require any runtime permissions — the
     * system photo picker runs in a separate process and grants the
     * caller a one-shot, URI-level read grant that stays valid for as
     * long as we hold the returned [Uri]. We immediately drain the
     * bytes on a background thread so we do not depend on that grant
     * beyond the call to [readImageBytesAsync].
     *
     * Registered as a property so the contract is wired up BEFORE the
     * activity reaches STARTED; calling [ActivityResultContracts] in
     * onCreate after super.onCreate would also work, but the property
     * form is the idiomatic one-liner.
     */
    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        if (uri == null) {
            setStatus("Image pick cancelled.")
            return@registerForActivityResult
        }
        readImageBytesAsync(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val pad = (resources.displayMetrics.density * 16).toInt()
        val scrollRoot = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT)
            isFillViewport = true
            fitsSystemWindows = true
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

        // --- Image picker (multimodal, GEMMA4 only) -------------------
        // Not hidden for non-GEMMA4 models on purpose — tapping Run
        // with a selected image against a text-only engine exercises
        // the UNSUPPORTED default in QuickDotAI.runMultimodal, which
        // is a useful smoke test of that error path too.
        root.addView(labelView("Image input (for GEMMA4 multimodal)"))
        imageStatusView = TextView(this).apply {
            text = "Image: none"
            textSize = 13f
        }
        root.addView(imageStatusView)

        val pickImageBtn = Button(this).apply {
            text = "Pick image"
            setOnClickListener { onPickImageClicked() }
        }
        root.addView(pickImageBtn)

        val clearImageBtn = Button(this).apply {
            text = "Clear image"
            setOnClickListener { onClearImageClicked() }
        }
        root.addView(clearImageBtn)

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

        // ============================================================
        // Chat Session Test Section
        // ============================================================
        root.addView(dividerView())

        val chatTitle = TextView(this).apply {
            text = "Chat Session Test"
            textSize = 18f
            setPadding(0, pad / 2, 0, 0)
        }
        root.addView(chatTitle)

        val chatSubtitle = TextView(this).apply {
            text = "Tests openChatSession / run / cancel / rebuild / close"
            textSize = 12f
            setPadding(0, 0, 0, pad / 4)
        }
        root.addView(chatSubtitle)

        chatSessionStatusView = TextView(this).apply {
            text = "Session: none"
            textSize = 13f
            setPadding(0, 0, 0, pad / 4)
        }
        root.addView(chatSessionStatusView)

        // System prompt (maps to ConversationConfig.systemInstruction)
        root.addView(labelView("System prompt (optional)"))
        chatSystemPromptField = EditText(this).apply {
            hint = "e.g. You are a helpful assistant."
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        root.addView(chatSystemPromptField)

        // Sampling config (all optional; specifying any of
        // temperature/topK/topP forces the wrapper to supply all three,
        // falling back to temperature=1.0 / topK=40 / topP=0.95 for
        // unspecified fields — see LiteRTLmChatSession.buildSamplerConfig).
        root.addView(labelView("Temperature (optional, ≥ 0)"))
        chatTemperatureField = EditText(this).apply {
            hint = "e.g. 0.7"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or
                android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        root.addView(chatTemperatureField)

        root.addView(labelView("topK (optional, > 0)"))
        chatTopKField = EditText(this).apply {
            hint = "e.g. 40"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        root.addView(chatTopKField)

        root.addView(labelView("topP (optional, 0..1)"))
        chatTopPField = EditText(this).apply {
            hint = "e.g. 0.95"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or
                android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        root.addView(chatTopPField)

        root.addView(labelView("seed (optional, 0 = nondeterministic)"))
        chatSeedField = EditText(this).apply {
            hint = "e.g. 42"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or
                android.text.InputType.TYPE_NUMBER_FLAG_SIGNED
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        root.addView(chatSeedField)

        root.addView(labelView("enable_thinking"))
        chatEnableThinkingSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                listOf("default (null)", "true", "false")
            )
        }
        root.addView(chatEnableThinkingSpinner)

        val chatOpenBtn = Button(this).apply {
            text = "Open Chat Session"
            setOnClickListener { onChatOpenClicked() }
        }
        root.addView(chatOpenBtn)

        root.addView(labelView("Chat message"))
        chatPromptField = EditText(this).apply {
            hint = "Type a chat message…"
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        root.addView(chatPromptField)

        val chatRunBtn = Button(this).apply {
            text = "Chat Send (streaming)"
            setOnClickListener { onChatRunStreamingClicked() }
        }
        root.addView(chatRunBtn)

        val chatRunBlockingBtn = Button(this).apply {
            text = "Chat Send (blocking)"
            setOnClickListener { onChatRunBlockingClicked() }
        }
        root.addView(chatRunBlockingBtn)

        val chatCancelBtn = Button(this).apply {
            text = "Chat Cancel"
            setOnClickListener { onChatCancelClicked() }
        }
        root.addView(chatCancelBtn)

        val chatRebuildBtn = Button(this).apply {
            text = "Chat Rebuild (clear history)"
            setOnClickListener { onChatRebuildClicked() }
        }
        root.addView(chatRebuildBtn)

        val chatCloseBtn = Button(this).apply {
            text = "Chat Close Session"
            setOnClickListener { onChatCloseClicked() }
        }
        root.addView(chatCloseBtn)

        // ============================================================
        // OpenAI-Style Messages Input Section
        // ============================================================
        root.addView(dividerView())

        val openAITitle = TextView(this).apply {
            text = "OpenAI-Style Messages Input"
            textSize = 18f
            setPadding(0, pad / 2, 0, 0)
        }
        root.addView(openAITitle)

        val openAISubtitle = TextView(this).apply {
            text = "Parse OpenAI-style JSON messages array and run chat"
            textSize = 12f
            setPadding(0, 0, 0, pad / 4)
        }
        root.addView(openAISubtitle)

        root.addView(labelView("Messages JSON (OpenAI format)"))
        openAIMessagesField = EditText(this).apply {
            hint = """[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]"""
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
            minLines = 4
            maxLines = 10
            setPadding(pad / 2, pad / 2, pad / 2, pad / 2)
            textSize = 12f
            // Pre-fill with a role-interleaved example that exercises
            // LiteRT-LM's native chat template (multiple SYSTEM turns).
            setText("""[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi! How can I help?"},
  {"role": "system", "content": "Answer in one short sentence."},
  {"role": "user", "content": "Write a short joke about saving RAM."}
]""")
        }
        root.addView(openAIMessagesField)

        val openAIRunBtn = Button(this).apply {
            text = "Run OpenAI Messages (streaming)"
            setOnClickListener { onOpenAIMessagesRunClicked() }
        }
        root.addView(openAIRunBtn)

        val openAIRunBlockingBtn = Button(this).apply {
            text = "Run OpenAI Messages (blocking)"
            setOnClickListener { onOpenAIMessagesRunBlockingClicked() }
        }
        root.addView(openAIRunBlockingBtn)

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
        // engine.close() internally closes any active chat session.
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
        val req = buildLoadRequest()
        setStatus(
            "Loading ${req.modelKey}… " +
                "(vision=${req.visionBackend?.name ?: "off"})"
        )
        outputView.text = ""
        engineExecutor.execute { loadModelInternal(req) }
    }

    /**
     * @brief Build a [LoadModelRequest] from the current spinner / text
     * field values. Must be called on the main thread.
     */
    private fun buildLoadRequest(): LoadModelRequest {
        val model = selectedModelId()
        val backend = BackendType.valueOf(backendSpinner.selectedItem as String)
        val quant = selectedQuant()
        val modelPath = modelPathField.text.toString().trim().ifEmpty { null }
        val visionBackend = if (model == ModelId.GEMMA4) backend else null
        val nativeLibDir = applicationContext.applicationInfo.nativeLibraryDir
        return LoadModelRequest(
            backend = backend,
            model = model,
            quantization = quant,
            modelPath = modelPath,
            visionBackend = visionBackend,
            nativeLibDir = nativeLibDir,
        )
    }

    /**
     * @brief Core model loading logic. Must be called from [engineExecutor].
     * Returns the loaded [QuickDotAI] engine, or null on failure.
     */
    private fun loadModelInternal(req: LoadModelRequest): QuickDotAI? {
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
            return engine
        }

        val newEngine: QuickDotAI = when (req.model) {
            ModelId.GEMMA4 -> LiteRTLm(applicationContext)
            else -> NativeQuickDotAI()
        }
        return when (val r = newEngine.load(req)) {
            is BackendResult.Ok -> {
                engine = newEngine
                loadedKey = req.modelKey
                setStatus(
                    "Loaded ${req.modelKey} " +
                        "(${newEngine.kind}, arch=${newEngine.architecture ?: "?"})"
                )
                newEngine
            }
            is BackendResult.Err -> {
                try {
                    newEngine.close()
                } catch (_: Throwable) { /* best effort */ }
                setStatus("Load failed: [${r.error.name}] ${r.message ?: ""}")
                null
            }
        }
    }

    private fun onRunClicked() {
        val prompt = promptField.text.toString()
        if (prompt.isBlank()) {
            setStatus("Prompt is empty.")
            return
        }
        // Snapshot the image bytes on the main thread so the engine
        // thread sees a stable reference even if the user taps "Clear
        // image" mid-run.
        val imgBytes = selectedImageBytes

        // Clear any previous output before we start streaming.
        mainHandler.post { outputView.text = "" }
        setStatus(
            if (imgBytes != null) "Running multimodal (${imgBytes.size}B image)…"
            else "Running…"
        )

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
            // Both streaming variants block until the backend finishes
            // or errors out; that's fine because we're already off the
            // main thread.
            try {
                if (imgBytes != null) {
                    // Canonical Gemma-4 / Gemma3n convention: image
                    // part(s) first, then a trailing text instruction.
                    val parts = listOf(
                        PromptPart.ImageBytes(imgBytes),
                        PromptPart.Text(prompt),
                    )
                    e.runMultimodalStreaming(parts, sink)
                } else {
                    e.runStreaming(prompt, sink)
                }
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

    // --- chat session handlers -------------------------------------------

    private fun onChatOpenClicked() {
        // Capture all UI values on the main thread — no latches needed.
        val req = buildLoadRequest()
        val systemPrompt = chatSystemPromptField.text.toString().trim().ifEmpty { null }
        val temperature = chatTemperatureField.text.toString().trim()
            .ifEmpty { null }?.toDoubleOrNull()
        val topK = chatTopKField.text.toString().trim()
            .ifEmpty { null }?.toIntOrNull()
        val topP = chatTopPField.text.toString().trim()
            .ifEmpty { null }?.toDoubleOrNull()
        val seed = chatSeedField.text.toString().trim()
            .ifEmpty { null }?.toIntOrNull()
        val thinkingIdx = chatEnableThinkingSpinner.selectedItemPosition

        setStatus("Opening chat session…")
        engineExecutor.execute {
            // Auto-load the model if not loaded yet.
            val e = loadModelInternal(req)
            if (e == null) {
                setStatus("Cannot open chat session — model load failed.")
                return@execute
            }

            // Close existing session if any.
            if (e.chatSessionId != null) {
                try { e.closeChatSession() } catch (_: Throwable) {}
            }

            // Build config from captured UI values. Leave sampling null
            // when every sampling field is empty so LiteRT-LM uses its
            // own engine/model default.
            val sampling = if (temperature != null || topK != null ||
                topP != null || seed != null) {
                QuickAiChatSamplingConfig(
                    temperature = temperature,
                    topK = topK,
                    topP = topP,
                    seed = seed
                )
            } else null

            val templateKwargs = when (thinkingIdx) {
                1 -> QuickAiChatTemplateKwargs(enableThinking = true)
                2 -> QuickAiChatTemplateKwargs(enableThinking = false)
                else -> null
            }

            val config = if (systemPrompt != null || sampling != null || templateKwargs != null) {
                QuickAiChatSessionConfig(
                    systemInstruction = systemPrompt,
                    sampling = sampling,
                    chatTemplateKwargs = templateKwargs
                )
            } else null

            when (val r = e.openChatSession(config)) {
                is BackendResult.Ok -> {
                    val sid = r.value
                    mainHandler.post {
                        chatSessionStatusView.text =
                            "Session: ${sid.take(8)}… (active)"
                    }
                    setStatus("Chat session opened: ${sid.take(8)}…")
                }
                is BackendResult.Err -> {
                    setStatus("Chat open failed: [${r.error.name}] ${r.message ?: ""}")
                }
            }
        }
    }

    private fun onChatRunStreamingClicked() {
        val prompt = chatPromptField.text.toString()
        if (prompt.isBlank()) {
            setStatus("Chat message is empty.")
            return
        }

        val imgBytes = selectedImageBytes
        outputView.text = ""
        setStatus("Chat streaming…")

        engineExecutor.execute {
            val e = engine
            if (e == null || e.chatSessionId == null) {
                setStatus("No chat session — tap Open Chat Session first.")
                return@execute
            }
            val sink = object : StreamSink {
                override fun onDelta(text: String) {
                    mainHandler.post { outputView.append(text) }
                }
                override fun onDone() {
                    setStatus("Chat done.")
                }
                override fun onError(error: QuickAiError, message: String?) {
                    setStatus("Chat error: [${error.name}] ${message ?: ""}")
                }
            }
            val parts = buildChatParts(prompt, imgBytes)
            val messages = listOf(
                QuickAiChatMessage(role = QuickAiChatRole.USER, parts = parts)
            )
            try {
                when (val r = e.chatRunStreaming(messages, sink)) {
                    is BackendResult.Ok -> {
                        setStatus(
                            "Chat done. (${r.value.metrics?.totalDurationMs?.toLong() ?: "?"} ms)"
                        )
                    }
                    is BackendResult.Err -> {
                        // Error already surfaced via sink.onError
                    }
                }
            } catch (t: Throwable) {
                setStatus("Chat threw: ${t.message}")
            }
        }
    }

    private fun onChatRunBlockingClicked() {
        val prompt = chatPromptField.text.toString()
        if (prompt.isBlank()) {
            setStatus("Chat message is empty.")
            return
        }

        val imgBytes = selectedImageBytes
        outputView.text = ""
        setStatus("Chat running (blocking)…")

        engineExecutor.execute {
            val e = engine
            if (e == null || e.chatSessionId == null) {
                setStatus("No chat session — tap Open Chat Session first.")
                return@execute
            }
            val parts = buildChatParts(prompt, imgBytes)
            val messages = listOf(
                QuickAiChatMessage(role = QuickAiChatRole.USER, parts = parts)
            )
            try {
                when (val r = e.chatRun(messages)) {
                    is BackendResult.Ok -> {
                        mainHandler.post { outputView.text = r.value.content }
                        setStatus(
                            "Chat done. (${r.value.metrics?.totalDurationMs?.toLong() ?: "?"} ms)"
                        )
                    }
                    is BackendResult.Err -> {
                        setStatus("Chat failed: [${r.error.name}] ${r.message ?: ""}")
                    }
                }
            } catch (t: Throwable) {
                setStatus("Chat threw: ${t.message}")
            }
        }
    }

    private fun onChatCancelClicked() {
        val e = engine
        if (e == null || e.chatSessionId == null) {
            setStatus("No active chat session.")
            return
        }
        // chatCancel() is thread-safe — can be called from main thread
        e.chatCancel()
        setStatus("Chat cancel requested.")
    }

    private fun onChatRebuildClicked() {
        setStatus("Rebuilding chat (clear history)…")
        engineExecutor.execute {
            val e = engine
            if (e == null || e.chatSessionId == null) {
                setStatus("No active chat session.")
                return@execute
            }
            when (val r = e.chatRebuild(emptyList())) {
                is BackendResult.Ok -> {
                    setStatus("Chat history cleared. Session still active.")
                }
                is BackendResult.Err -> {
                    setStatus("Chat rebuild failed: [${r.error.name}] ${r.message ?: ""}")
                }
            }
        }
    }

    private fun onChatCloseClicked() {
        setStatus("Closing chat session…")
        engineExecutor.execute {
            val e = engine
            if (e == null || e.chatSessionId == null) {
                setStatus("No active chat session.")
                return@execute
            }
            when (val r = e.closeChatSession()) {
                is BackendResult.Ok -> {
                    mainHandler.post {
                        chatSessionStatusView.text = "Session: none"
                    }
                    setStatus("Chat session closed.")
                }
                is BackendResult.Err -> {
                    setStatus("Chat close failed: [${r.error.name}] ${r.message ?: ""}")
                }
            }
        }
    }

    /**
     * @brief Build [PromptPart] list for a chat message. If image bytes
     * are available, include them (image first, then text — Gemma convention).
     */
    private fun buildChatParts(prompt: String, imgBytes: ByteArray?): List<PromptPart> {
        return if (imgBytes != null) {
            listOf(PromptPart.ImageBytes(imgBytes), PromptPart.Text(prompt))
        } else {
            listOf(PromptPart.Text(prompt))
        }
    }

    // --- OpenAI-style messages handlers -----------------------------------

    /**
     * @brief Parse OpenAI-style JSON messages array into QuickDotAI types.
     *
     * Input format:
     * ```json
     * [
     *   {"role": "system", "content": "You are a helpful assistant."},
     *   {"role": "user", "content": "Hello!"},
     *   {"role": "assistant", "content": "Hi there!"},
     *   {"role": "system", "content": "Now answer briefly."},
     *   {"role": "user", "content": "How are you?"}
     * ]
     * ```
     *
     * Every entry — including multiple [QuickAiChatRole.SYSTEM] turns —
     * is preserved in order. Role-interleaved input is forwarded to
     * LiteRT-LM's native chat template through
     * [com.example.quickdotai.QuickDotAI.chatRun] /
     * [com.example.quickdotai.QuickDotAI.chatRunStreaming] ; the
     * wrapper does no role folding or merging.
     *
     * @param jsonString JSON array of message objects
     * @return parsed messages in input order, or null on parse error
     */
    private fun parseOpenAIMessages(jsonString: String): List<QuickAiChatMessage>? {
        return try {
            val json = Json { ignoreUnknownKeys = true; isLenient = true }
            val jsonArray = json.parseToJsonElement(jsonString).jsonArray

            val messages = mutableListOf<QuickAiChatMessage>()

            for (element in jsonArray) {
                val obj = element.jsonObject
                val role = obj["role"]?.jsonPrimitive?.content?.lowercase() ?: continue
                val content = obj["content"]?.jsonPrimitive?.content ?: ""

                val quickRole = when (role) {
                    "system" -> QuickAiChatRole.SYSTEM
                    "user" -> QuickAiChatRole.USER
                    "assistant" -> QuickAiChatRole.ASSISTANT
                    else -> continue
                }
                messages.add(
                    QuickAiChatMessage(
                        role = quickRole,
                        parts = listOf(PromptPart.Text(content))
                    )
                )
            }

            messages
        } catch (t: Throwable) {
            null
        }
    }

    /**
     * @brief Run OpenAI-style messages with streaming output.
     * Parses the JSON, opens a session with system prompt, and runs chat.
     */
    private fun onOpenAIMessagesRunClicked() {
        val jsonText = openAIMessagesField.text.toString().trim()
        if (jsonText.isBlank()) {
            setStatus("Messages JSON is empty.")
            return
        }

        val messages = parseOpenAIMessages(jsonText)
        if (messages == null) {
            setStatus("Failed to parse messages JSON. Check format.")
            return
        }

        if (messages.isEmpty()) {
            setStatus("No messages found in JSON.")
            return
        }
        if (messages.last().role != QuickAiChatRole.USER) {
            setStatus("Last message must be role=\"user\" to trigger inference.")
            return
        }

        outputView.text = ""
        setStatus("Opening session and running OpenAI messages…")

        val req = buildLoadRequest()
        engineExecutor.execute {
            // Load model if needed
            val e = loadModelInternal(req)
            if (e == null) {
                setStatus("Model load failed.")
                return@execute
            }

            // No session-level systemInstruction — every role from the
            // JSON (including SYSTEM turns) is forwarded in-order via
            // chatRunStreaming so LiteRT-LM's native chat template
            // renders the full role-annotated array.
            //
            // If a chat session is already active (e.g. the user pressed
            // "Open Chat Session" first to configure enableThinking /
            // sampling), reuse it — chatRunStreaming will rebuild the
            // Conversation for the multi-role bundle internally while
            // preserving the session's original config. Opening a fresh
            // config=null session here would discard those settings.
            val config: QuickAiChatSessionConfig? = null

            if (e.chatSessionId == null) {
                when (val openResult = e.openChatSession(config)) {
                    is BackendResult.Err -> {
                        setStatus("Failed to open session: ${openResult.message}")
                        return@execute
                    }
                    is BackendResult.Ok -> {
                        mainHandler.post {
                            chatSessionStatusView.text = "Session: ${openResult.value.take(8)}… (active)"
                        }
                    }
                }
            }

            // Run streaming
            val sink = object : StreamSink {
                override fun onDelta(text: String) {
                    mainHandler.post { outputView.append(text) }
                }
                override fun onDone() {
                    setStatus("OpenAI messages chat done.")
                }
                override fun onError(error: QuickAiError, message: String?) {
                    setStatus("Chat error: [${error.name}] ${message ?: ""}")
                }
            }

            try {
                when (val r = e.chatRunStreaming(messages, sink)) {
                    is BackendResult.Ok -> {
                        setStatus("Done. (${r.value.metrics?.totalDurationMs?.toLong() ?: "?"} ms)")
                    }
                    is BackendResult.Err -> {
                        // Error already surfaced via sink.onError
                    }
                }
            } catch (t: Throwable) {
                setStatus("Chat threw: ${t.message}")
            }
        }
    }

    /**
     * @brief Run OpenAI-style messages with blocking output.
     * Parses the JSON, opens a session with system prompt, and runs chat.
     */
    private fun onOpenAIMessagesRunBlockingClicked() {
        val jsonText = openAIMessagesField.text.toString().trim()
        if (jsonText.isBlank()) {
            setStatus("Messages JSON is empty.")
            return
        }

        val messages = parseOpenAIMessages(jsonText)
        if (messages == null) {
            setStatus("Failed to parse messages JSON. Check format.")
            return
        }

        if (messages.isEmpty()) {
            setStatus("No messages found in JSON.")
            return
        }
        if (messages.last().role != QuickAiChatRole.USER) {
            setStatus("Last message must be role=\"user\" to trigger inference.")
            return
        }

        outputView.text = ""
        setStatus("Opening session and running OpenAI messages (blocking)…")

        val req = buildLoadRequest()
        engineExecutor.execute {
            // Load model if needed
            val e = loadModelInternal(req)
            if (e == null) {
                setStatus("Model load failed.")
                return@execute
            }

            // No session-level systemInstruction — every role from the
            // JSON (including SYSTEM turns) is forwarded in-order via
            // chatRun so LiteRT-LM's native chat template renders the
            // full role-annotated array.
            //
            // If a chat session is already active (e.g. the user pressed
            // "Open Chat Session" first to configure enableThinking /
            // sampling), reuse it — chatRun will rebuild the Conversation
            // for the multi-role bundle internally while preserving the
            // session's original config. Opening a fresh config=null
            // session here would discard those settings.
            val config: QuickAiChatSessionConfig? = null

            if (e.chatSessionId == null) {
                when (val openResult = e.openChatSession(config)) {
                    is BackendResult.Err -> {
                        setStatus("Failed to open session: ${openResult.message}")
                        return@execute
                    }
                    is BackendResult.Ok -> {
                        mainHandler.post {
                            chatSessionStatusView.text = "Session: ${openResult.value.take(8)}… (active)"
                        }
                    }
                }
            }

            // Run blocking
            try {
                when (val r = e.chatRun(messages)) {
                    is BackendResult.Ok -> {
                        mainHandler.post { outputView.text = r.value.content }
                        setStatus("Done. (${r.value.metrics?.totalDurationMs?.toLong() ?: "?"} ms)")
                    }
                    is BackendResult.Err -> {
                        setStatus("Chat failed: [${r.error.name}] ${r.message ?: ""}")
                    }
                }
            } catch (t: Throwable) {
                setStatus("Chat threw: ${t.message}")
            }
        }
    }

    // --- image picker handlers ----------------------------------------

    /**
     * @brief Launches the system photo picker (no runtime permissions
     * required).
     *
     * Uses `PickVisualMedia.ImageOnly` to filter out videos. The
     * result is dispatched to [imagePickerLauncher]'s callback, which
     * drains the URI on a background thread via [readImageBytesAsync].
     */
    private fun onPickImageClicked() {
        imagePickerLauncher.launch(
            PickVisualMediaRequest(
                ActivityResultContracts.PickVisualMedia.ImageOnly
            )
        )
        setStatus("Opening photo picker…")
    }

    /**
     * @brief Forgets the currently-selected image so the next Run tap
     * falls back to the text-only [QuickDotAI.runStreaming] path.
     */
    private fun onClearImageClicked() {
        selectedImageBytes = null
        mainHandler.post { imageStatusView.text = "Image: none" }
        setStatus("Image cleared.")
    }

    /**
     * @brief Reads the picked image URI into a byte array on a
     * throwaway background thread.
     *
     * We deliberately do NOT reuse [engineExecutor] here so a long-
     * running inference does not block the image read and leave the
     * user staring at a stale "Image: none" label. The thread is a
     * one-shot daemon — it terminates as soon as the read completes.
     *
     * The bytes are the raw file contents (JPEG / PNG / …) — exactly
     * what LiteRT-LM's `Content.ImageBytes` expects on the far side
     * of [PromptPart.ImageBytes].
     */
    private fun readImageBytesAsync(uri: Uri) {
        setStatus("Reading image…")
        Thread({
            try {
                val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
                if (bytes == null || bytes.isEmpty()) {
                    setStatus("Image read failed or empty.")
                    return@Thread
                }
                selectedImageBytes = bytes
                mainHandler.post {
                    imageStatusView.text = "Image: ${bytes.size} bytes selected"
                }
                setStatus("Image loaded (${bytes.size} bytes). Tap Run to send.")
            } catch (t: Throwable) {
                setStatus("Failed to read image: ${t.message}")
            }
        }, "SampleTestAPP-ImageRead").apply { isDaemon = true }.start()
    }

    // --- helpers -------------------------------------------------------

    private fun dividerView(): View {
        val pad = (resources.displayMetrics.density * 12).toInt()
        return View(this).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, 2).also {
                it.topMargin = pad
                it.bottomMargin = pad / 2
            }
            setBackgroundColor(0xFF888888.toInt())
        }
    }

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
            ModelId.GAUSS3_8_QNN ->
                "$base/models/gauss-3.8b-qnn"
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

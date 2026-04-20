// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    MainActivity.kt
 * @brief   Standalone sample that drives the :QuickDotAI AAR directly —
 *          no QuickAIService, no REST, no remote process.
 *
 * Engine wiring (unchanged from the original sample):
 *
 *  1. Instantiates [LiteRTLm] for GEMMA4 and [NativeQuickDotAI] for every
 *     other model, both against a single-thread Executor so all calls
 *     touching a given engine are serialised on the same worker thread
 *     (the [QuickDotAI] interface is not internally thread-safe).
 *  2. Calls [QuickDotAI.load] once per chosen (model, quant) pair. For
 *     GEMMA4 it auto-populates [LoadModelRequest.visionBackend] so the
 *     multimodal path is armed from load time.
 *  3. Drives [QuickDotAI.runStreaming] (text-only) or
 *     [QuickDotAI.runMultimodalStreaming] (when an image is selected)
 *     with an in-memory StreamSink that appends each delta to the output
 *     view on the main thread.
 *
 * UI (M3 Expressive redesign — see Applications/QuickAI/QuickDotAI/QuickDotAI.html
 * design bundle): the screen is rebuilt as a tabbed Material 3 interface
 * with a custom top bar, hero status pill, collapsible Model section
 * with chip-group backend / quantization pickers, and a terminal-styled
 * output panel shared across the Run / Chat / OpenAI tabs. A light/dark
 * toggle in the top bar swaps the full M3 token set at runtime.
 */
package com.example.sampletestapp

import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.PorterDuff
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.net.Uri
import android.os.Bundle
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.PopupMenu
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.NestedScrollView
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import com.example.quickdotai.BackendResult
import com.example.quickdotai.BackendType
import com.example.quickdotai.LiteRTLm
import com.example.quickdotai.LoadModelRequest
import com.example.quickdotai.ModelId
import com.example.quickdotai.NativeQuickDotAI
import com.example.quickdotai.PerformanceMetrics
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

/* ────────────────────────────────────────────────────────────────────────
 * M3 Expressive token set — mirrors the LIGHT / DARK objects defined in
 * the QuickDotAI.html design bundle. Holding both palettes in a single
 * data class lets us swap them atomically when the user taps the dark-
 * mode toggle in the top bar.
 * ──────────────────────────────────────────────────────────────────────── */
private data class M3Tokens(
    val bg: Int, val surface: Int, val surfaceDim: Int,
    val surfaceContainer: Int, val surfaceContainerHigh: Int,
    val outline: Int, val outlineVariant: Int,
    val onSurface: Int, val onSurfaceVar: Int,
    val primary: Int, val onPrimary: Int,
    val primaryContainer: Int, val onPrimaryContainer: Int,
    val secondary: Int, val secondaryContainer: Int,
    val tertiary: Int, val tertiaryContainer: Int,
    val error: Int, val errorContainer: Int,
    val success: Int, val successContainer: Int,
    val codeBg: Int, val codeFg: Int,
)

private val LIGHT = M3Tokens(
    bg = 0xFFFBF8FF.toInt(),
    surface = 0xFFFFFFFF.toInt(),
    surfaceDim = 0xFFF2EEF8.toInt(),
    surfaceContainer = 0xFFF4EFFA.toInt(),
    surfaceContainerHigh = 0xFFEDE7F6.toInt(),
    outline = 0xFFCAC4D0.toInt(),
    outlineVariant = 0xFFE7E0EC.toInt(),
    onSurface = 0xFF1C1B1F.toInt(),
    onSurfaceVar = 0xFF49454F.toInt(),
    primary = 0xFF5B3EBE.toInt(),
    onPrimary = 0xFFFFFFFF.toInt(),
    primaryContainer = 0xFFE9DDFF.toInt(),
    onPrimaryContainer = 0xFF21005D.toInt(),
    secondary = 0xFF625B71.toInt(),
    secondaryContainer = 0xFFE8DEF8.toInt(),
    tertiary = 0xFF7D5260.toInt(),
    tertiaryContainer = 0xFFFFD8E4.toInt(),
    error = 0xFFB3261E.toInt(),
    errorContainer = 0xFFF9DEDC.toInt(),
    success = 0xFF146C2E.toInt(),
    successContainer = 0xFFD5F5DF.toInt(),
    codeBg = 0xFF0F0B1E.toInt(),
    codeFg = 0xFFEDE7F6.toInt(),
)

private val DARK = M3Tokens(
    bg = 0xFF121019.toInt(),
    surface = 0xFF1B1823.toInt(),
    surfaceDim = 0xFF100E17.toInt(),
    surfaceContainer = 0xFF211E2B.toInt(),
    surfaceContainerHigh = 0xFF2B2834.toInt(),
    outline = 0xFF4A4458.toInt(),
    outlineVariant = 0xFF2D2A37.toInt(),
    onSurface = 0xFFE6E0E9.toInt(),
    onSurfaceVar = 0xFFCAC4D0.toInt(),
    primary = 0xFFCFBCFF.toInt(),
    onPrimary = 0xFF371E73.toInt(),
    primaryContainer = 0xFF4A3A8C.toInt(),
    onPrimaryContainer = 0xFFE9DDFF.toInt(),
    secondary = 0xFFCCC2DC.toInt(),
    secondaryContainer = 0xFF4A4458.toInt(),
    tertiary = 0xFFEFB8C8.toInt(),
    tertiaryContainer = 0xFF633B48.toInt(),
    error = 0xFFF2B8B5.toInt(),
    errorContainer = 0xFF601410.toInt(),
    success = 0xFF6EDB88.toInt(),
    successContainer = 0xFF124F24.toInt(),
    codeBg = 0xFF06040F.toInt(),
    codeFg = 0xFFCFBCFF.toInt(),
)

class MainActivity : AppCompatActivity() {

    /* ───── Engine plumbing (unchanged from the original sample) ───── */

    private val engineExecutor: Executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "SampleTestAPP-Engine").apply { isDaemon = true }
    }

    private val mainHandler by lazy { android.os.Handler(mainLooper) }

    @Volatile private var engine: QuickDotAI? = null
    @Volatile private var loadedKey: String? = null
    @Volatile private var selectedImageBytes: ByteArray? = null

    /* ───── UI state (preserved across light/dark theme rebuilds) ───── */

    private var darkMode = false
    private var selectedTab: String = "run"             // run | chat | openai | metrics
    private var modelExpanded = true
    private var samplingExpanded = false

    private var selectedModel: ModelId = ModelId.GEMMA4
    private var selectedBackend: BackendType = BackendType.GPU
    private var selectedQuant: QuantizationType = QuantizationType.W4A32

    private var modelPathText: String = ""
    private var promptText: String = "What is rainbow?"

    private var systemPromptText: String = ""
    private var temperatureText: String = ""
    private var topKText: String = ""
    private var topPText: String = ""
    private var seedText: String = ""
    private var thinkingChoice: String = "default"      // default | true | false
    private var chatPromptText: String = ""
    private var openAiJsonText: String = """[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi! How can I help?"},
  {"role": "system", "content": "Answer in one short sentence."},
  {"role": "user", "content": "Write a short joke about saving RAM."}
]"""

    private var statusText: String = "Idle."
    private var outputText: String = ""
    private var streaming: Boolean = false
    private var loadStatus: String = "idle"             // idle | loading | loaded
    private var loadedLabel: String = ""
    private var sessionIdText: String? = null
    private var lastMetrics: PerformanceMetrics? = null

    private var mainScrollY = 0
    private var outputScrollY = 0

    /* ───── UI refs (re-wired on every rebuildUi() call) ───── */

    private lateinit var rootHost: FrameLayout
    private lateinit var mainScrollView: NestedScrollView
    private lateinit var outputScrollView: NestedScrollView
    private lateinit var statusView: TextView
    private lateinit var outputView: TextView
    private lateinit var modelPathField: EditText
    private lateinit var promptField: EditText
    private lateinit var imageStatusView: TextView
    private lateinit var chatSystemPromptField: EditText
    private lateinit var chatTemperatureField: EditText
    private lateinit var chatTopKField: EditText
    private lateinit var chatTopPField: EditText
    private lateinit var chatSeedField: EditText
    private lateinit var chatPromptField: EditText
    private lateinit var openAIMessagesField: EditText
    private lateinit var chatSessionStatusView: TextView

    /**
     * @brief ActivityResult launcher for the Android system photo picker.
     * Uses [ActivityResultContracts.PickVisualMedia] which does NOT
     * require any runtime permissions — the system photo picker grants a
     * one-shot URI read grant. We immediately drain the bytes on a
     * background thread so we do not depend on that grant beyond
     * [readImageBytesAsync].
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
        // Seed the path field so a single Load tap works without typing.
        modelPathText = defaultModelPathFor(selectedModel, selectedQuant)
        rebuildUi()
    }

    /**
     * @brief Tear down and re-inflate the entire view tree using the
     * current [darkMode] palette. EditText contents are preserved by
     * round-tripping through the `*Text` state vars, which are kept in
     * sync via [TextWatcher]s installed in the field builders below.
     */
    private fun rebuildUi() {
        val tokens = if (darkMode) DARK else LIGHT
        // Snapshot any in-flight EditText contents into state vars so the
        // theme rebuild does not lose typed input.
        if (::promptField.isInitialized) promptText = promptField.text.toString()
        if (::modelPathField.isInitialized) modelPathText = modelPathField.text.toString()
        if (::chatSystemPromptField.isInitialized) systemPromptText = chatSystemPromptField.text.toString()
        if (::chatTemperatureField.isInitialized) temperatureText = chatTemperatureField.text.toString()
        if (::chatTopKField.isInitialized) topKText = chatTopKField.text.toString()
        if (::chatTopPField.isInitialized) topPText = chatTopPField.text.toString()
        if (::chatSeedField.isInitialized) seedText = chatSeedField.text.toString()
        if (::chatPromptField.isInitialized) chatPromptText = chatPromptField.text.toString()
        if (::openAIMessagesField.isInitialized) openAiJsonText = openAIMessagesField.text.toString()

        // Save scroll positions before rebuilding
        if (::mainScrollView.isInitialized) mainScrollY = mainScrollView.scrollY
        if (::outputScrollView.isInitialized) outputScrollY = outputScrollView.scrollY

        rootHost = FrameLayout(this).apply {
            layoutParams = ViewGroup.LayoutParams(MATCH_PARENT, MATCH_PARENT)
            setBackgroundColor(tokens.bg)
            fitsSystemWindows = true
        }
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT)
        }
        rootHost.addView(column)

        column.addView(buildTopBar(tokens))
        column.addView(buildHeroCard(tokens))
        column.addView(buildTabBar(tokens))

        // Main scrolling content area — wraps the model section, the
        // active tab body, and (for non-metrics tabs) the shared output
        // panel.
        mainScrollView = NestedScrollView(this).apply {
            isFillViewport = false
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f)
            overScrollMode = View.OVER_SCROLL_NEVER
        }
        val scrollColumn = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), 0, dp(12), dp(120))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        mainScrollView.addView(scrollColumn)
        column.addView(mainScrollView)

        scrollColumn.addView(buildModelSection(tokens))
        spacer(scrollColumn, 10)

        scrollColumn.addView(when (selectedTab) {
            "run"     -> buildRunTab(tokens)
            "chat"    -> buildChatTab(tokens)
            "openai"  -> buildOpenAiTab(tokens)
            "metrics" -> buildMetricsTab(tokens)
            else      -> buildRunTab(tokens)
        })

        if (selectedTab != "metrics") {
            spacer(scrollColumn, 10)
            scrollColumn.addView(buildOutputPanel(tokens))
        }

        setContentView(rootHost)

        // Restore scroll positions after UI is built
        mainScrollView.post { mainScrollView.scrollTo(0, mainScrollY) }
        if (::outputScrollView.isInitialized) {
            outputScrollView.post { outputScrollView.scrollTo(0, outputScrollY) }
        }
    }

    /* ════════════════════════════════════════════════════════════════
     * Component builders
     * ════════════════════════════════════════════════════════════════ */

    private fun buildTopBar(t: M3Tokens): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(16), dp(14), dp(16), dp(10))
        }
        // Brand mark — gradient-filled rounded square with a "✦" glyph,
        // approximating the hero icon in the design.
        val brand = TextView(this).apply {
            text = "✦"
            setTextColor(Color.WHITE)
            textSize = 18f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            background = gradient(t.primary, t.tertiary, 12)
            layoutParams = LinearLayout.LayoutParams(dp(40), dp(40))
        }
        row.addView(brand)

        spacerH(row, 12)

        val titleColumn = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        }
        val title = TextView(this).apply {
            text = "QuickDotAI"
            setTextColor(t.onSurface)
            textSize = 18f
            typeface = Typeface.DEFAULT_BOLD
        }
        val sub = TextView(this).apply {
            val tail = if (loadStatus == "loaded") loadedLabel else "no model"
            val tailColor = if (loadStatus == "loaded") t.success else t.onSurfaceVar
            val full = "In-process AAR  ·  $tail"
            text = full
            textSize = 11f
            // Approximate the React design's two-tone subtitle by using
            // the success color when a model is loaded, otherwise neutral.
            setTextColor(tailColor)
        }
        titleColumn.addView(title)
        titleColumn.addView(sub)
        row.addView(titleColumn)

        // Light/dark toggle button.
        val toggle = TextView(this).apply {
            text = if (darkMode) "☀" else "☾"
            textSize = 16f
            gravity = Gravity.CENTER
            setTextColor(t.onSurfaceVar)
            background = solid(t.surfaceContainer, 20)
            layoutParams = LinearLayout.LayoutParams(dp(40), dp(40))
            setOnClickListener {
                darkMode = !darkMode
                rebuildUi()
            }
        }
        row.addView(toggle)
        return row
    }

    private fun buildHeroCard(t: M3Tokens): View {
        val tone = statusTone()
        val (bgColor, dotColor, fgColor) = when (tone) {
            "error"    -> Triple(t.errorContainer,   t.error,   t.error)
            "success"  -> Triple(t.successContainer, t.success, t.success)
            "progress" -> Triple(t.primaryContainer, t.primary, t.onPrimaryContainer)
            else       -> Triple(t.surfaceContainer, t.outline, t.onSurfaceVar)
        }
        val outer = FrameLayout(this).apply {
            setPadding(dp(12), 0, dp(12), dp(12))
        }
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = solid(bgColor, 20)
            layoutParams = FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        // Status indicator dot.
        val dot = View(this).apply {
            background = circle(dotColor)
            layoutParams = LinearLayout.LayoutParams(dp(10), dp(10))
        }
        card.addView(dot)
        spacerH(card, 10)

        statusView = TextView(this).apply {
            text = statusText
            setTextColor(fgColor)
            textSize = 13f
            typeface = Typeface.MONOSPACE
            ellipsize = android.text.TextUtils.TruncateAt.END
            maxLines = 1
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        }
        card.addView(statusView)

        if (streaming) {
            val stop = TextView(this).apply {
                text = "■  Stop"
                setTextColor(Color.WHITE)
                textSize = 12f
                typeface = Typeface.DEFAULT_BOLD
                background = solid(t.error, 100)
                setPadding(dp(12), dp(6), dp(12), dp(6))
                setOnClickListener { onCancelClicked() }
            }
            card.addView(stop)
        }
        outer.addView(card)
        return outer
    }

    private fun buildTabBar(t: M3Tokens): View {
        val outer = FrameLayout(this).apply {
            setPadding(dp(12), 0, dp(12), dp(10))
        }
        val pill = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            background = solid(t.surfaceContainer, 100)
            setPadding(dp(4), dp(4), dp(4), dp(4))
            layoutParams = FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        val tabs = listOf("run" to "▶ Run", "chat" to "⌬ Chat",
                          "openai" to "{ } OpenAI", "metrics" to "▤ Metrics")
        for ((idx, entry) in tabs.withIndex()) {
            val (key, label) = entry
            val active = key == selectedTab
            val tab = TextView(this).apply {
                text = label
                gravity = Gravity.CENTER
                textSize = 13f
                typeface = if (active) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
                setTextColor(if (active) t.onPrimary else t.onSurfaceVar)
                background = if (active) solid(t.primary, 100) else null
                setPadding(dp(6), dp(10), dp(6), dp(10))
                layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f).also {
                    if (idx > 0) it.leftMargin = dp(4)
                }
                setOnClickListener {
                    selectedTab = key
                    rebuildUi()
                }
            }
            pill.addView(tab)
        }
        outer.addView(pill)
        return outer
    }

    private fun buildModelSection(t: M3Tokens): View {
        val subtitle = if (loadStatus == "loaded")
            "$loadedLabel  ·  ${selectedBackend.name}"
        else
            "${selectedModel.name}  ·  ${selectedBackend.name}  ·  ${selectedQuant.name}"
        val statusDotColor = when (loadStatus) {
            "loaded"  -> t.success
            "loading" -> t.primary
            else      -> t.outline
        }
        val card = collapsibleCard(
            t = t,
            iconGlyph = "▦",
            iconBg = t.primaryContainer,
            iconFg = t.onPrimaryContainer,
            title = "Model",
            subtitle = subtitle,
            rightAdornment = View(this).apply {
                background = circle(statusDotColor)
                layoutParams = LinearLayout.LayoutParams(dp(10), dp(10)).also {
                    it.rightMargin = dp(4)
                }
            },
            expanded = modelExpanded,
            onToggle = {
                modelExpanded = !modelExpanded
                rebuildUi()
            }
        ) { body ->
            // MODEL select.
            body.addView(labelView(t, "MODEL"))
            val modelRow = TextView(this).apply {
                text = badgePlusLabel(selectedModel)
                setTextColor(t.onSurface)
                textSize = 14f
                typeface = Typeface.MONOSPACE
                background = solid(t.surfaceContainer, 12)
                setPadding(dp(14), dp(12), dp(14), dp(12))
                gravity = Gravity.CENTER_VERTICAL
                setCompoundDrawablesWithIntrinsicBounds(null, null, null, null)
                setOnClickListener {
                    val popup = PopupMenu(this@MainActivity, this)
                    ModelId.values().forEachIndexed { i, m -> popup.menu.add(0, i, i, badgePlusLabel(m)) }
                    popup.setOnMenuItemClickListener { item ->
                        selectedModel = ModelId.values()[item.itemId]
                        modelPathText = defaultModelPathFor(selectedModel, selectedQuant)
                        rebuildUi(); true
                    }
                    popup.show()
                }
            }
            body.addView(modelRow)
            spacer(body, 12)

            // BACKEND chip group.
            body.addView(labelView(t, "COMPUTE BACKEND"))
            body.addView(chipRow(t, BackendType.values().map { it.name },
                selectedBackend.name) { picked ->
                selectedBackend = BackendType.valueOf(picked)
                rebuildUi()
            })
            spacer(body, 12)

            // QUANT chip group.
            body.addView(labelView(t, "QUANTIZATION"))
            body.addView(chipRow(t, QuantizationType.values().map { it.name },
                selectedQuant.name) { picked ->
                selectedQuant = QuantizationType.valueOf(picked)
                modelPathText = defaultModelPathFor(selectedModel, selectedQuant)
                rebuildUi()
            })
            spacer(body, 12)

            // MODEL PATH.
            body.addView(labelView(t, "MODEL PATH"))
            modelPathField = roundedEditText(t, modelPathText, mono = true,
                onTextChange = { modelPathText = it })
            body.addView(modelPathField)
            spacer(body, 12)

            // Load / Unload action row.
            val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            val loadLabel = if (loadStatus == "loaded") "↻  Reload model" else "↓  Load model"
            actions.addView(filledButton(t, loadLabel, fill = "horizontal") { onLoadClicked() })
            if (loadStatus == "loaded") {
                spacerH(actions, 8)
                actions.addView(tonalButton(t, "✕  Unload", danger = true) { onUnloadClicked() })
            }
            body.addView(actions)
        }
        return card
    }

    private fun buildRunTab(t: M3Tokens): View {
        val card = roundedCard(t, t.surfaceContainer)
        val header = sectionHeader(t, "⚡", t.secondaryContainer, t.onSurface,
            "One-shot run", "Raw prompt · streaming output")
        card.addView(header)
        spacer(card, 14)

        // PROMPT.
        card.addView(labelView(t, "PROMPT"))
        promptField = roundedEditText(t, promptText, multiline = true, mono = true, rows = 5,
            onTextChange = { promptText = it })
        card.addView(promptField)
        spacer(card, 14)

        // IMAGE INPUT.
        val imgLabelRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        imgLabelRow.addView(labelView(t, "IMAGE INPUT").also {
            it.layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT)
        })
        if (selectedModel != ModelId.GEMMA4) {
            spacerH(imgLabelRow, 6)
            val badge = TextView(this).apply {
                text = "GEMMA4 only"
                setTextColor(t.tertiary)
                textSize = 10f
                typeface = Typeface.DEFAULT_BOLD
                background = solid(t.tertiaryContainer, 4)
                setPadding(dp(6), dp(1), dp(6), dp(1))
            }
            imgLabelRow.addView(badge)
        }
        card.addView(imgLabelRow)
        spacer(card, 6)

        if (selectedImageBytes != null) {
            val attached = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                background = solid(t.surfaceContainerHigh, 12)
                setPadding(dp(12), dp(12), dp(12), dp(12))
            }
            val thumb = TextView(this).apply {
                text = "🖼"
                setTextColor(t.onSurface)
                textSize = 22f
                gravity = Gravity.CENTER
                background = gradient(
                    blendAlpha(t.primary, 0x55),
                    blendAlpha(t.tertiary, 0x55), 8
                )
                layoutParams = LinearLayout.LayoutParams(dp(48), dp(48))
            }
            attached.addView(thumb)
            spacerH(attached, 10)
            val info = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
            }
            info.addView(TextView(this).apply {
                text = "Selected image"
                setTextColor(t.onSurface)
                textSize = 13f
                typeface = Typeface.DEFAULT_BOLD
            })
            info.addView(TextView(this).apply {
                text = "${selectedImageBytes!!.size} bytes  ·  raw bytes ready"
                setTextColor(t.onSurfaceVar)
                textSize = 11f
                typeface = Typeface.MONOSPACE
            })
            attached.addView(info)
            val close = TextView(this).apply {
                text = "✕"
                setTextColor(t.onSurfaceVar)
                gravity = Gravity.CENTER
                textSize = 14f
                layoutParams = LinearLayout.LayoutParams(dp(32), dp(32))
                setOnClickListener { onClearImageClicked(); rebuildUi() }
            }
            attached.addView(close)
            card.addView(attached)
            // Keep the legacy imageStatusView reference happy — it is
            // touched by onClearImageClicked / readImageBytesAsync.
            imageStatusView = TextView(this).apply { visibility = View.GONE }
        } else {
            val dropzone = TextView(this).apply {
                text = "+  Pick image for multimodal run"
                setTextColor(t.onSurfaceVar)
                textSize = 13f
                typeface = Typeface.DEFAULT_BOLD
                gravity = Gravity.CENTER
                background = dashedBg(t)
                setPadding(dp(14), dp(14), dp(14), dp(14))
                setOnClickListener { onPickImageClicked() }
            }
            card.addView(dropzone)
            imageStatusView = TextView(this).apply { visibility = View.GONE }
        }
        spacer(card, 14)

        // RUN button.
        val runLabel = when {
            streaming -> "■  Stop streaming"
            selectedImageBytes != null -> "▶  Run multimodal (streaming)"
            else -> "▶  Run (streaming)"
        }
        val runBtn = filledButton(t, runLabel, fill = "vertical",
            danger = streaming) {
            if (streaming) onCancelClicked() else onRunClicked()
        }
        card.addView(runBtn)
        return card
    }

    private fun buildChatTab(t: M3Tokens): View {
        val container = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }

        // ── Session status card ──
        val sessionCard = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = solid(t.surfaceContainer, 20)
            setPadding(dp(14), dp(14), dp(14), dp(14))
        }
        val active = sessionIdText != null
        val sessionIcon = TextView(this).apply {
            text = "⌬"
            gravity = Gravity.CENTER
            textSize = 18f
            setTextColor(if (active) t.success else t.onSurfaceVar)
            background = solid(if (active) t.successContainer else t.surfaceContainerHigh, 10)
            layoutParams = LinearLayout.LayoutParams(dp(38), dp(38))
        }
        sessionCard.addView(sessionIcon)
        spacerH(sessionCard, 12)
        val sessionTextCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        }
        sessionTextCol.addView(TextView(this).apply {
            text = "SESSION"
            setTextColor(t.onSurfaceVar)
            textSize = 13f
            typeface = Typeface.DEFAULT_BOLD
        })
        chatSessionStatusView = TextView(this).apply {
            text = if (active) "${sessionIdText!!.take(8)}… active" else "none"
            setTextColor(t.onSurface)
            textSize = 14f
            typeface = Typeface.MONOSPACE
        }
        sessionTextCol.addView(chatSessionStatusView)
        sessionCard.addView(sessionTextCol)

        if (active) {
            sessionCard.addView(tonalButton(t, "↺ Rebuild") { onChatRebuildClicked() })
            spacerH(sessionCard, 6)
            sessionCard.addView(tonalButton(t, "✕ Close", danger = true) {
                onChatCloseClicked()
            })
        } else {
            sessionCard.addView(filledButton(t, "+ Open") { onChatOpenClicked() })
        }
        container.addView(sessionCard)
        spacer(container, 10)

        // ── Collapsible session config ──
        val configCard = collapsibleCard(
            t = t,
            iconGlyph = "⚙",
            iconBg = t.tertiaryContainer,
            iconFg = t.tertiary,
            title = "Session config",
            subtitle = "System prompt · sampling · thinking mode",
            rightAdornment = null,
            expanded = samplingExpanded,
            onToggle = {
                samplingExpanded = !samplingExpanded
                rebuildUi()
            }
        ) { body ->
            body.addView(labelView(t, "SYSTEM PROMPT"))
            chatSystemPromptField = roundedEditText(t, systemPromptText,
                multiline = true, rows = 2,
                placeholder = "You are a helpful assistant.",
                onTextChange = { systemPromptText = it })
            body.addView(chatSystemPromptField)
            spacer(body, 10)

            // 2x2 grid of numeric fields.
            val grid1 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            chatTemperatureField = roundedEditText(t, temperatureText, mono = true, numeric = true,
                placeholder = "0.7", onTextChange = { temperatureText = it })
            chatTopKField = roundedEditText(t, topKText, mono = true, numeric = true,
                placeholder = "40", onTextChange = { topKText = it })
            grid1.addView(labeledColumn(t, "TEMPERATURE", chatTemperatureField, weight = 1f))
            spacerH(grid1, 10)
            grid1.addView(labeledColumn(t, "TOP_K", chatTopKField, weight = 1f))
            body.addView(grid1)
            spacer(body, 10)

            val grid2 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            chatTopPField = roundedEditText(t, topPText, mono = true, numeric = true,
                placeholder = "0.95", onTextChange = { topPText = it })
            chatSeedField = roundedEditText(t, seedText, mono = true, numeric = true,
                placeholder = "random", onTextChange = { seedText = it })
            grid2.addView(labeledColumn(t, "TOP_P", chatTopPField, weight = 1f))
            spacerH(grid2, 10)
            grid2.addView(labeledColumn(t, "SEED", chatSeedField, weight = 1f))
            body.addView(grid2)
            spacer(body, 12)

            body.addView(labelView(t, "ENABLE_THINKING"))
            body.addView(chipRow(t, listOf("default", "true", "false"),
                thinkingChoice) { picked ->
                thinkingChoice = picked
                rebuildUi()
            })
        }
        container.addView(configCard)
        spacer(container, 10)

        // ── Composer ──
        val composer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = solid(t.surfaceContainer, 20)
            setPadding(dp(14), dp(14), dp(14), dp(14))
        }
        composer.addView(labelView(t, "CHAT MESSAGE"))
        spacer(composer, 6)
        val composerRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.BOTTOM
            background = solid(t.surfaceContainerHigh, 24)
            setPadding(dp(6), dp(6), dp(6), dp(6))
        }
        chatPromptField = EditText(this).apply {
            setText(chatPromptText)
            hint = "Type a chat message…"
            setHintTextColor(t.onSurfaceVar)
            setTextColor(t.onSurface)
            textSize = 14f
            background = null
            minLines = 2
            maxLines = 6
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            setPadding(dp(14), dp(10), dp(14), dp(10))
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
            addTextChangedListener(simpleWatcher { chatPromptText = it })
        }
        composerRow.addView(chatPromptField)

        val canSend = sessionIdText != null && !streaming
        val sendFab = TextView(this).apply {
            text = "▶"
            gravity = Gravity.CENTER
            textSize = 18f
            setTextColor(if (canSend) t.onPrimary else t.onSurfaceVar)
            background = solid(if (canSend) t.primary else t.outlineVariant, 22)
            layoutParams = LinearLayout.LayoutParams(dp(44), dp(44)).also {
                it.bottomMargin = dp(2); it.rightMargin = dp(2)
            }
            isEnabled = canSend
            setOnClickListener { onChatRunStreamingClicked() }
        }
        composerRow.addView(sendFab)
        composer.addView(composerRow)
        spacer(composer, 6)
        val blockingBtn = outlinedButton(t, "Send (blocking)") {
            onChatRunBlockingClicked()
        }.apply { isEnabled = sessionIdText != null }
        composer.addView(blockingBtn)
        container.addView(composer)
        return container
    }

    private fun buildOpenAiTab(t: M3Tokens): View {
        val card = roundedCard(t, t.surfaceContainer)
        card.addView(sectionHeader(t, "{ }", t.secondaryContainer, t.onSurface,
            "OpenAI messages",
            "Role-interleaved array · forwarded to chat template"))
        spacer(card, 12)

        // Parsed preview.
        var parseErr: String? = null
        val parsed: List<QuickAiChatMessage>? = try {
            parseOpenAIMessages(openAiJsonText)
        } catch (e: Throwable) { parseErr = e.message; null }

        if (parsed != null) {
            val previewWrap = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                background = solid(t.surfaceContainerHigh, 16)
                setPadding(dp(10), dp(10), dp(10), dp(10))
            }
            for (msg in parsed) {
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.TOP
                    setPadding(0, dp(3), 0, dp(3))
                }
                val (badgeBg, badgeFg, badgeLabel) = when (msg.role) {
                    QuickAiChatRole.SYSTEM    -> Triple(t.tertiaryContainer, t.tertiary, "SYSTEM")
                    QuickAiChatRole.USER      -> Triple(t.primaryContainer,  t.onPrimaryContainer, "USER")
                    QuickAiChatRole.ASSISTANT -> Triple(t.secondaryContainer, t.onSurface, "ASSISTANT")
                }
                val badge = TextView(this).apply {
                    text = badgeLabel
                    setTextColor(badgeFg)
                    textSize = 10f
                    typeface = Typeface.MONOSPACE
                    gravity = Gravity.CENTER
                    background = solid(badgeBg, 100)
                    setPadding(dp(8), dp(2), dp(8), dp(2))
                    layoutParams = LinearLayout.LayoutParams(dp(80), WRAP_CONTENT)
                }
                row.addView(badge)
                spacerH(row, 8)
                val content = TextView(this).apply {
                    val txtPart = msg.parts.firstOrNull() as? PromptPart.Text
                    text = txtPart?.text ?: ""
                    setTextColor(t.onSurface)
                    textSize = 13f
                    layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
                }
                row.addView(content)
                previewWrap.addView(row)
            }
            card.addView(previewWrap)
            spacer(card, 12)
        }
        if (parseErr != null) {
            val errBox = TextView(this).apply {
                text = "ⓘ  $parseErr"
                setTextColor(t.error)
                textSize = 12f
                typeface = Typeface.MONOSPACE
                background = solid(t.errorContainer, 10)
                setPadding(dp(12), dp(8), dp(12), dp(8))
            }
            card.addView(errBox)
            spacer(card, 12)
        }

        card.addView(labelView(t, "MESSAGES JSON"))
        openAIMessagesField = roundedEditText(t, openAiJsonText, multiline = true, mono = true, rows = 8,
            onTextChange = {
                val same = it == openAiJsonText
                openAiJsonText = it
                // Re-render the preview so role pills track edits live.
                if (!same) rebuildUi()
            })
        card.addView(openAIMessagesField)
        spacer(card, 12)

        val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val runBtn = filledButton(t, "▶  Run (streaming)", fill = "horizontal") {
            onOpenAIMessagesRunClicked()
        }.apply { isEnabled = !streaming && parseErr == null }
        actions.addView(runBtn)
        spacerH(actions, 8)
        val blockingBtn = tonalButton(t, "Blocking") {
            onOpenAIMessagesRunBlockingClicked()
        }.apply { isEnabled = !streaming && parseErr == null }
        actions.addView(blockingBtn)
        card.addView(actions)
        return card
    }

    private fun buildMetricsTab(t: M3Tokens): View {
        val column = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        val m = lastMetrics
        if (m == null) {
            val empty = roundedCard(t, t.surfaceContainer).apply {
                gravity = Gravity.CENTER
                setPadding(dp(20), dp(32), dp(20), dp(32))
            }
            val icon = TextView(this).apply {
                text = "▤"
                gravity = Gravity.CENTER
                textSize = 28f
                setTextColor(t.onSurfaceVar)
                background = solid(t.surfaceContainerHigh, 28)
                layoutParams = LinearLayout.LayoutParams(dp(56), dp(56)).also {
                    it.bottomMargin = dp(12); it.gravity = Gravity.CENTER_HORIZONTAL
                }
            }
            empty.addView(icon)
            empty.addView(TextView(this).apply {
                text = "No metrics yet"
                setTextColor(t.onSurface)
                textSize = 15f
                typeface = Typeface.DEFAULT_BOLD
                gravity = Gravity.CENTER
            })
            empty.addView(TextView(this).apply {
                text = "Run a prompt and tap Fetch metrics in the Run tab to populate counters."
                setTextColor(t.onSurfaceVar)
                textSize = 13f
                gravity = Gravity.CENTER
                setPadding(0, dp(4), 0, 0)
            })
            spacer(empty, 12)
            empty.addView(filledButton(t, "Fetch metrics", fill = "vertical") { onMetricsClicked() })
            column.addView(empty)
            return column
        }

        val tps = if (m.generationDurationMs > 0)
            String.format("%.1f", m.generationTokens / (m.generationDurationMs / 1000.0))
        else "—"
        val ttft = String.format("%.0f", m.prefillDurationMs)

        // Big stat tile: tokens/sec.
        column.addView(metricTile(t, "TOKENS PER SECOND", tps, "tok/s", big = true))
        spacer(column, 10)

        val grid = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        grid.addView(metricTile(t, "TTFT", ttft, "ms", weight = 1f))
        spacerH(grid, 10)
        grid.addView(metricTile(t, "TOTAL", String.format("%.2f", m.totalDurationMs / 1000.0), "s", weight = 1f))
        column.addView(grid)
        spacer(column, 10)

        val grid2 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        grid2.addView(metricTile(t, "PREFILL TOKENS", m.prefillTokens.toString(), "tok", weight = 1f))
        spacerH(grid2, 10)
        grid2.addView(metricTile(t, "GEN TOKENS", m.generationTokens.toString(), "tok", weight = 1f))
        column.addView(grid2)
        spacer(column, 10)

        val grid3 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        grid3.addView(metricTile(t, "INIT", String.format("%.0f", m.initializationDurationMs), "ms", weight = 1f))
        spacerH(grid3, 10)
        grid3.addView(metricTile(t, "PEAK MEMORY",
            String.format("%.1f", m.peakMemoryKb / 1024.0), "MB", weight = 1f))
        column.addView(grid3)
        spacer(column, 10)

        // Prefill ▸ Gen bar.
        val barCard = roundedCard(t, t.surfaceContainer)
        barCard.addView(TextView(this).apply {
            text = "PREFILL ▸ GENERATION"
            setTextColor(t.onSurfaceVar)
            textSize = 11f
            typeface = Typeface.DEFAULT_BOLD
            setPadding(0, 0, 0, dp(10))
        })
        val total = (m.totalDurationMs).coerceAtLeast(1.0)
        val prefillFrac = (m.prefillDurationMs / total).coerceIn(0.0, 1.0).toFloat()
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            background = solid(t.surfaceContainerHigh, 4)
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, dp(8))
        }
        bar.addView(View(this).apply {
            background = solid(t.tertiary, 0)
            layoutParams = LinearLayout.LayoutParams(0, MATCH_PARENT, prefillFrac)
        })
        bar.addView(View(this).apply {
            background = solid(t.primary, 0)
            layoutParams = LinearLayout.LayoutParams(0, MATCH_PARENT, 1f - prefillFrac)
        })
        barCard.addView(bar)
        spacer(barCard, 8)
        val barLegend = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        barLegend.addView(TextView(this).apply {
            text = "● prefill ${ttft}ms"
            setTextColor(t.tertiary)
            textSize = 11f
            typeface = Typeface.MONOSPACE
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        })
        barLegend.addView(TextView(this).apply {
            text = "● gen ${String.format("%.0f", m.generationDurationMs)}ms"
            setTextColor(t.primary)
            textSize = 11f
            typeface = Typeface.MONOSPACE
            gravity = Gravity.RIGHT
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        })
        barCard.addView(barLegend)
        column.addView(barCard)
        spacer(column, 10)
        column.addView(tonalButton(t, "↻  Refresh metrics") { onMetricsClicked() })
        return column
    }

    private fun buildOutputPanel(t: M3Tokens): View {
        val wrap = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = solid(t.codeBg, 20)
        }
        // Title bar with macOS-style traffic-light dots.
        val titleBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(14), dp(10), dp(14), dp(10))
        }
        val dotRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        for (color in intArrayOf(0xFFFF5F57.toInt(), 0xFFFEBC2E.toInt(), 0xFF28C840.toInt())) {
            val d = View(this).apply {
                background = circle(color)
                layoutParams = LinearLayout.LayoutParams(dp(10), dp(10)).also {
                    it.rightMargin = dp(4)
                }
            }
            dotRow.addView(d)
        }
        titleBar.addView(dotRow)
        titleBar.addView(TextView(this).apply {
            text = "output  ·  stream.kt"
            setTextColor(0x80FFFFFF.toInt())
            textSize = 11f
            typeface = Typeface.MONOSPACE
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        })
        val copy = TextView(this).apply {
            text = "📋"
            setTextColor(0x80FFFFFF.toInt())
            gravity = Gravity.CENTER
            textSize = 12f
            layoutParams = LinearLayout.LayoutParams(dp(24), dp(24))
            setOnClickListener {
                if (outputText.isNotEmpty()) {
                    val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                    val clip = ClipData.newPlainText("output", outputText)
                    clipboard.setPrimaryClip(clip)
                }
            }
        }
        titleBar.addView(copy)
        spacerH(titleBar, 8)
        val clear = TextView(this).apply {
            text = "🗑"
            setTextColor(0x80FFFFFF.toInt())
            gravity = Gravity.CENTER
            textSize = 12f
            layoutParams = LinearLayout.LayoutParams(dp(24), dp(24))
            setOnClickListener {
                outputText = ""
                outputView.text = ""
            }
        }
        titleBar.addView(clear)
        wrap.addView(titleBar)
        wrap.addView(View(this).apply {
            setBackgroundColor(0x10FFFFFF)
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, 1)
        })

        outputScrollView = NestedScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, dp(220))
        }
        outputView = TextView(this).apply {
            text = if (outputText.isEmpty())
                "// streaming output appears here…" else outputText
            setTextColor(if (outputText.isEmpty()) 0x4DFFFFFF else 0xFFEDE7F6.toInt())
            textSize = 13f
            typeface = Typeface.MONOSPACE
            setPadding(dp(16), dp(12), dp(16), dp(12))
            layoutParams = FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        outputScrollView.addView(outputView)
        wrap.addView(outputScrollView)
        return wrap
    }

    /* ════════════════════════════════════════════════════════════════
     * Reusable UI primitives (drawables, buttons, fields, chips, …)
     * ════════════════════════════════════════════════════════════════ */

    private fun roundedCard(t: M3Tokens, color: Int): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = solid(color, 20)
            setPadding(dp(16), dp(16), dp(16), dp(16))
        }
    }

    private fun collapsibleCard(
        t: M3Tokens,
        iconGlyph: String, iconBg: Int, iconFg: Int,
        title: String, subtitle: String,
        rightAdornment: View?,
        expanded: Boolean,
        onToggle: () -> Unit,
        body: (LinearLayout) -> Unit,
    ): View {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = strokedSolid(t.surface, 24, t.outlineVariant, 1)
            setPadding(dp(12), dp(12), dp(12), dp(12))
        }
        // Header row.
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(4), dp(6), dp(4), dp(6))
            isClickable = true
            setOnClickListener { onToggle() }
        }
        val iconBox = TextView(this).apply {
            text = iconGlyph
            gravity = Gravity.CENTER
            textSize = 16f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(iconFg)
            background = solid(iconBg, 10)
            layoutParams = LinearLayout.LayoutParams(dp(36), dp(36))
        }
        header.addView(iconBox)
        spacerH(header, 12)
        val titleCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        }
        titleCol.addView(TextView(this).apply {
            text = title
            setTextColor(t.onSurface)
            textSize = 15f
            typeface = Typeface.DEFAULT_BOLD
        })
        titleCol.addView(TextView(this).apply {
            text = subtitle
            setTextColor(t.onSurfaceVar)
            textSize = 12f
        })
        header.addView(titleCol)
        if (rightAdornment != null) header.addView(rightAdornment)
        header.addView(TextView(this).apply {
            text = if (expanded) "▲" else "▼"
            setTextColor(t.onSurfaceVar)
            textSize = 12f
            setPadding(dp(6), 0, 0, 0)
        })
        card.addView(header)
        if (expanded) {
            val bodyContainer = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(4), dp(12), dp(4), dp(4))
            }
            body(bodyContainer)
            card.addView(bodyContainer)
        }
        return card
    }

    private fun sectionHeader(
        t: M3Tokens, glyph: String, iconBg: Int, iconFg: Int,
        title: String, subtitle: String
    ): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val icon = TextView(this).apply {
            text = glyph
            gravity = Gravity.CENTER
            textSize = 16f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(iconFg)
            background = solid(iconBg, 10)
            layoutParams = LinearLayout.LayoutParams(dp(32), dp(32))
        }
        row.addView(icon)
        spacerH(row, 8)
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        col.addView(TextView(this).apply {
            text = title
            setTextColor(t.onSurface)
            textSize = 15f
            typeface = Typeface.DEFAULT_BOLD
        })
        col.addView(TextView(this).apply {
            text = subtitle
            setTextColor(t.onSurfaceVar)
            textSize = 12f
        })
        row.addView(col)
        return row
    }

    private fun labelView(t: M3Tokens, text: String): TextView = TextView(this).apply {
        this.text = text
        setTextColor(t.onSurfaceVar)
        textSize = 12f
        typeface = Typeface.DEFAULT_BOLD
        setPadding(dp(4), 0, 0, dp(6))
    }

    private fun roundedEditText(
        t: M3Tokens, value: String,
        multiline: Boolean = false,
        mono: Boolean = false,
        numeric: Boolean = false,
        rows: Int = 1,
        placeholder: String? = null,
        onTextChange: (String) -> Unit,
    ): EditText {
        val baseBg = strokedSolid(t.surfaceContainer, 12, Color.TRANSPARENT, 0)
        val focusBg = strokedSolid(t.surfaceContainer, 12, t.primary, 2)
        val field = EditText(this).apply {
            setText(value)
            if (placeholder != null) {
                hint = placeholder
                setHintTextColor(t.onSurfaceVar)
            }
            setTextColor(t.onSurface)
            textSize = 14f
            background = baseBg
            setPadding(dp(14), dp(12), dp(14), dp(12))
            if (mono) typeface = Typeface.MONOSPACE
            if (multiline) {
                inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
                minLines = rows
                maxLines = rows + 4
                setHorizontallyScrolling(false)
                gravity = Gravity.TOP
            }
            if (numeric) {
                inputType = InputType.TYPE_CLASS_NUMBER or
                    InputType.TYPE_NUMBER_FLAG_DECIMAL or
                    InputType.TYPE_NUMBER_FLAG_SIGNED
            }
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
            setOnFocusChangeListener { _, hasFocus ->
                background = if (hasFocus) focusBg else baseBg
                setPadding(dp(14), dp(12), dp(14), dp(12))
            }
            addTextChangedListener(simpleWatcher(onTextChange))
        }
        return field
    }

    private fun labeledColumn(t: M3Tokens, label: String, field: View, weight: Float): View {
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, weight)
        }
        col.addView(labelView(t, label))
        col.addView(field)
        return col
    }

    private fun chipRow(t: M3Tokens, options: List<String>, selected: String,
                        onPick: (String) -> Unit): View {
        val scroll = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            overScrollMode = View.OVER_SCROLL_NEVER
        }
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        for ((i, opt) in options.withIndex()) {
            val active = opt == selected
            val chip = TextView(this).apply {
                text = if (active) "✓ $opt" else opt
                setTextColor(if (active) t.onSurface else t.onSurfaceVar)
                textSize = 13f
                typeface = if (active) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
                background = if (active)
                    solid(t.secondaryContainer, 8)
                else
                    strokedSolid(Color.TRANSPARENT, 8, t.outline, 1)
                setPadding(dp(12), dp(6), dp(12), dp(6))
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).also {
                    if (i > 0) it.leftMargin = dp(6)
                }
                setOnClickListener { onPick(opt) }
            }
            row.addView(chip)
        }
        scroll.addView(row)
        return scroll
    }

    private fun filledButton(t: M3Tokens, label: String, fill: String? = null,
                             danger: Boolean = false, onClick: () -> Unit): Button {
        return Button(this).apply {
            text = label
            isAllCaps = false
            setTextColor(if (danger) Color.WHITE else t.onPrimary)
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            stateListAnimator = null
            background = solid(if (danger) t.error else t.primary, 100)
            setPadding(dp(20), dp(12), dp(20), dp(12))
            layoutParams = when (fill) {
                "vertical"   -> LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
                "horizontal" -> LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
                else         -> LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT)
            }
            setOnClickListener { onClick() }
        }
    }

    private fun tonalButton(t: M3Tokens, label: String, danger: Boolean = false,
                            onClick: () -> Unit): Button {
        return Button(this).apply {
            text = label
            isAllCaps = false
            setTextColor(if (danger) t.error else t.onSurface)
            textSize = 13f
            typeface = Typeface.DEFAULT_BOLD
            stateListAnimator = null
            background = solid(if (danger) t.errorContainer else t.secondaryContainer, 100)
            setPadding(dp(14), dp(8), dp(14), dp(8))
            layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT)
            setOnClickListener { onClick() }
        }
    }

    private fun outlinedButton(t: M3Tokens, label: String,
                               onClick: () -> Unit): Button {
        return Button(this).apply {
            text = label
            isAllCaps = false
            setTextColor(t.primary)
            textSize = 13f
            typeface = Typeface.DEFAULT_BOLD
            stateListAnimator = null
            background = strokedSolid(Color.TRANSPARENT, 100, t.outline, 1)
            setPadding(dp(14), dp(8), dp(14), dp(8))
            layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT)
            setOnClickListener { onClick() }
        }
    }

    private fun metricTile(t: M3Tokens, label: String, value: String, unit: String,
                           big: Boolean = false, weight: Float = 0f): View {
        val tile = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = solid(if (big) t.primaryContainer else t.surfaceContainer, 20)
            setPadding(dp(if (big) 20 else 14),
                       dp(if (big) 20 else 14),
                       dp(if (big) 20 else 14),
                       dp(if (big) 20 else 14))
            layoutParams = if (weight > 0)
                LinearLayout.LayoutParams(0, WRAP_CONTENT, weight)
            else
                LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        val fg = if (big) t.onPrimaryContainer else t.onSurface
        tile.addView(TextView(this).apply {
            text = label
            setTextColor(fg and 0x00FFFFFF or (0xB3 shl 24))
            textSize = 11f
            typeface = Typeface.DEFAULT_BOLD
        })
        val valueRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.BOTTOM
            setPadding(0, dp(4), 0, 0)
        }
        valueRow.addView(TextView(this).apply {
            text = value
            setTextColor(fg)
            textSize = if (big) 38f else 24f
            typeface = Typeface.MONOSPACE
        })
        valueRow.addView(TextView(this).apply {
            text = " $unit"
            setTextColor(fg and 0x00FFFFFF or (0x99 shl 24))
            textSize = if (big) 14f else 11f
            typeface = Typeface.DEFAULT_BOLD
            setPadding(dp(4), 0, 0, dp(if (big) 6 else 3))
        })
        tile.addView(valueRow)
        return tile
    }

    /* ───── Drawable / dimension helpers ───── */

    private fun solid(color: Int, radiusDp: Int): GradientDrawable = GradientDrawable().apply {
        setColor(color)
        cornerRadius = dpf(radiusDp)
    }

    private fun strokedSolid(fill: Int, radiusDp: Int, strokeColor: Int, strokeDp: Int): GradientDrawable =
        GradientDrawable().apply {
            setColor(fill)
            cornerRadius = dpf(radiusDp)
            if (strokeDp > 0) setStroke(dp(strokeDp), strokeColor)
        }

    private fun circle(color: Int): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.OVAL
        setColor(color)
    }

    private fun gradient(c1: Int, c2: Int, radiusDp: Int): GradientDrawable =
        GradientDrawable(GradientDrawable.Orientation.TL_BR, intArrayOf(c1, c2)).apply {
            cornerRadius = dpf(radiusDp)
        }

    private fun dashedBg(t: M3Tokens): GradientDrawable = GradientDrawable().apply {
        setColor(Color.TRANSPARENT)
        cornerRadius = dpf(12)
        setStroke(dp(1), t.outline, dpf(6), dpf(4))
    }

    private fun blendAlpha(color: Int, alpha: Int): Int =
        (color and 0x00FFFFFF) or (alpha shl 24)

    private fun dp(v: Int): Int = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics
    ).toInt()

    private fun dpf(v: Int): Float = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics
    )

    private fun spacer(parent: LinearLayout, h: Int) {
        parent.addView(View(this).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, dp(h))
        })
    }

    private fun spacerH(parent: LinearLayout, w: Int) {
        parent.addView(View(this).apply {
            layoutParams = LinearLayout.LayoutParams(dp(w), MATCH_PARENT)
        })
    }

    private fun simpleWatcher(onChange: (String) -> Unit): TextWatcher = object : TextWatcher {
        override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
        override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        override fun afterTextChanged(s: Editable?) { onChange(s?.toString() ?: "") }
    }

    private fun statusTone(): String {
        val s = statusText.lowercase()
        return when {
            "fail" in s || "error" in s || "empty" in s || "cancel" in s -> "error"
            "done" in s || "loaded" in s || "opened" in s || "ready" in s -> "success"
            "load" in s || "running" in s || "open" in s || "stream" in s -> "progress"
            else -> "neutral"
        }
    }

    /** Pretty label including the design's "MULTIMODAL/TEXT/QNN" tag. */
    private fun badgePlusLabel(m: ModelId): String = when (m) {
        ModelId.GEMMA4         -> "[MULTIMODAL]  ${m.name}"
        ModelId.QWEN3_0_6B     -> "[TEXT]        ${m.name}"
        ModelId.GAUSS3_8_QNN   -> "[QNN]         ${m.name}"
        ModelId.GAUSS3_6_QNN   -> "[QNN]         ${m.name}"
        ModelId.QWEN3_1_7B_Q40 -> "[TEXT]        ${m.name}"
    }

    /* ════════════════════════════════════════════════════════════════
     * Engine handlers (logic preserved from the original sample)
     * ════════════════════════════════════════════════════════════════ */

    override fun onDestroy() {
        // Fire-and-forget close on the engine thread so we don't leak the
        // native model handle / LiteRT-LM Engine on config changes.
        // engine.close() internally closes any active chat session.
        val e = engine
        engine = null
        loadedKey = null
        if (e != null) {
            engineExecutor.execute {
                try { e.close() } catch (_: Throwable) { /* best effort */ }
            }
        }
        super.onDestroy()
    }

    private fun onLoadClicked() {
        val req = buildLoadRequest()
        loadStatus = "loading"
        setStatus("Loading ${req.modelKey}…  (vision=${req.visionBackend?.name ?: "off"})")
        outputText = ""
        mainHandler.post { rebuildUi() }
        engineExecutor.execute { loadModelInternal(req) }
    }

    private fun onCancelClicked() {
        // Best-effort: chat sessions support cancel; one-shot run is
        // bounded by the streaming sink and will end naturally when the
        // engine returns.
        val e = engine
        if (e != null && e.chatSessionId != null) {
            e.chatCancel()
            setStatus("Cancel requested.")
        } else {
            streaming = false
            setStatus("Cancelled.")
            mainHandler.post { rebuildUi() }
        }
    }

    /**
     * @brief Build a [LoadModelRequest] from the current UI state. Must
     * be called on the main thread.
     */
    private fun buildLoadRequest(): LoadModelRequest {
        val model = selectedModel
        val backend = selectedBackend
        val quant = selectedQuant
        val modelPath = (if (::modelPathField.isInitialized) modelPathField.text.toString()
                          else modelPathText).trim().ifEmpty { null }
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
        if (loadedKey != null && loadedKey != req.modelKey) {
            try { engine?.close() } catch (_: Throwable) { /* best effort */ }
            engine = null
            loadedKey = null
        }
        if (engine != null && loadedKey == req.modelKey) {
            loadStatus = "loaded"
            loadedLabel = req.modelKey
            setStatus("Already loaded: ${req.modelKey}")
            mainHandler.post { rebuildUi() }
            return engine
        }

        val newEngine: QuickDotAI = when (req.model) {
            ModelId.GEMMA4 -> LiteRTLm(applicationContext)
            else -> NativeQuickDotAI(applicationContext)
        }
        return when (val r = newEngine.load(req)) {
            is BackendResult.Ok -> {
                engine = newEngine
                loadedKey = req.modelKey
                loadStatus = "loaded"
                loadedLabel = req.modelKey
                setStatus("Loaded ${req.modelKey} (${newEngine.kind}, arch=${newEngine.architecture ?: "?"})")
                mainHandler.post { rebuildUi() }
                newEngine
            }
            is BackendResult.Err -> {
                try { newEngine.close() } catch (_: Throwable) { /* best effort */ }
                loadStatus = "idle"
                setStatus("Load failed: [${r.error.name}] ${r.message ?: ""}")
                mainHandler.post { rebuildUi() }
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
        val imgBytes = selectedImageBytes
        outputText = ""
        mainHandler.post { outputView.text = "" }
        streaming = true
        setStatus(if (imgBytes != null) "Running multimodal (${imgBytes.size}B image)…"
                  else "Running…")
        mainHandler.post { rebuildUi() }

        engineExecutor.execute {
            val e = engine
            if (e == null) {
                streaming = false
                setStatus("No model loaded — tap Load first.")
                mainHandler.post { rebuildUi() }
                return@execute
            }
            val sink = object : StreamSink {
                override fun onDelta(text: String) {
                    outputText += text
                    mainHandler.post { outputView.append(text) }
                }
                override fun onDone() {
                    streaming = false
                    setStatus("Done.")
                    mainHandler.post { rebuildUi() }
                }
                override fun onError(error: QuickAiError, message: String?) {
                    streaming = false
                    setStatus("Run failed: [${error.name}] ${message ?: ""}")
                    mainHandler.post { rebuildUi() }
                }
            }
            try {
                if (imgBytes != null) {
                    val parts = listOf(
                        PromptPart.ImageBytes(imgBytes),
                        PromptPart.Text(prompt),
                    )
                    e.runMultimodalStreaming(parts, sink)
                } else {
                    e.runStreaming(prompt, sink)
                }
            } catch (t: Throwable) {
                streaming = false
                setStatus("Run threw: ${t.message}")
                mainHandler.post { rebuildUi() }
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
                    lastMetrics = r.value
                    setStatus("Metrics fetched.")
                    mainHandler.post { rebuildUi() }
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
                    loadedKey = null
                    loadStatus = "idle"
                    loadedLabel = ""
                    setStatus("Unloaded.")
                    mainHandler.post { rebuildUi() }
                }
                is BackendResult.Err ->
                    setStatus("Unload failed: [${r.error.name}] ${r.message ?: ""}")
            }
        }
    }

    /* ───── Chat session handlers ───── */

    private fun onChatOpenClicked() {
        val req = buildLoadRequest()
        val systemPrompt = systemPromptText.trim().ifEmpty { null }
        val temperature = temperatureText.trim().ifEmpty { null }?.toDoubleOrNull()
        val topK = topKText.trim().ifEmpty { null }?.toIntOrNull()
        val topP = topPText.trim().ifEmpty { null }?.toDoubleOrNull()
        val seed = seedText.trim().ifEmpty { null }?.toIntOrNull()

        setStatus("Opening chat session…")
        engineExecutor.execute {
            val e = loadModelInternal(req)
            if (e == null) {
                setStatus("Cannot open chat session — model load failed.")
                return@execute
            }
            if (e.chatSessionId != null) {
                try { e.closeChatSession() } catch (_: Throwable) {}
            }

            val sampling = if (temperature != null || topK != null || topP != null || seed != null) {
                QuickAiChatSamplingConfig(
                    temperature = temperature, topK = topK, topP = topP, seed = seed
                )
            } else null

            val templateKwargs = when (thinkingChoice) {
                "true"  -> QuickAiChatTemplateKwargs(enableThinking = true)
                "false" -> QuickAiChatTemplateKwargs(enableThinking = false)
                else    -> null
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
                    sessionIdText = r.value
                    setStatus("Chat session opened: ${r.value.take(8)}…")
                    mainHandler.post { rebuildUi() }
                }
                is BackendResult.Err -> {
                    setStatus("Chat open failed: [${r.error.name}] ${r.message ?: ""}")
                }
            }
        }
    }

    private fun onChatRunStreamingClicked() {
        val prompt = chatPromptField.text.toString()
        if (prompt.isBlank()) { setStatus("Chat message is empty."); return }
        val imgBytes = selectedImageBytes
        outputText = ""
        outputView.text = ""
        streaming = true
        setStatus("Chat streaming…")
        mainHandler.post { rebuildUi() }

        engineExecutor.execute {
            val e = engine
            if (e == null || e.chatSessionId == null) {
                streaming = false
                setStatus("No chat session — tap Open first.")
                mainHandler.post { rebuildUi() }
                return@execute
            }
            val sink = object : StreamSink {
                override fun onDelta(text: String) {
                    outputText += text
                    mainHandler.post { outputView.append(text) }
                }
                override fun onDone() {
                    streaming = false
                    setStatus("Chat done.")
                    mainHandler.post { rebuildUi() }
                }
                override fun onError(error: QuickAiError, message: String?) {
                    streaming = false
                    setStatus("Chat error: [${error.name}] ${message ?: ""}")
                    mainHandler.post { rebuildUi() }
                }
            }
            val parts = buildChatParts(prompt, imgBytes)
            val messages = listOf(QuickAiChatMessage(role = QuickAiChatRole.USER, parts = parts))
            try {
                when (val r = e.chatRunStreaming(messages, sink)) {
                    is BackendResult.Ok -> {
                        streaming = false
                        lastMetrics = r.value.metrics ?: lastMetrics
                        setStatus("Chat done. (${r.value.metrics?.totalDurationMs?.toLong() ?: "?"} ms)")
                        mainHandler.post { rebuildUi() }
                    }
                    is BackendResult.Err -> {
                        streaming = false
                        mainHandler.post { rebuildUi() }
                    }
                }
            } catch (t: Throwable) {
                streaming = false
                setStatus("Chat threw: ${t.message}")
                mainHandler.post { rebuildUi() }
            }
        }
    }

    private fun onChatRunBlockingClicked() {
        val prompt = chatPromptField.text.toString()
        if (prompt.isBlank()) { setStatus("Chat message is empty."); return }
        val imgBytes = selectedImageBytes
        outputText = ""
        outputView.text = ""
        setStatus("Chat running (blocking)…")

        engineExecutor.execute {
            val e = engine
            if (e == null || e.chatSessionId == null) {
                setStatus("No chat session — tap Open first.")
                return@execute
            }
            val parts = buildChatParts(prompt, imgBytes)
            val messages = listOf(QuickAiChatMessage(role = QuickAiChatRole.USER, parts = parts))
            try {
                when (val r = e.chatRun(messages)) {
                    is BackendResult.Ok -> {
                        outputText = r.value.content
                        lastMetrics = r.value.metrics ?: lastMetrics
                        mainHandler.post { outputView.text = r.value.content }
                        setStatus("Chat done. (${r.value.metrics?.totalDurationMs?.toLong() ?: "?"} ms)")
                    }
                    is BackendResult.Err ->
                        setStatus("Chat failed: [${r.error.name}] ${r.message ?: ""}")
                }
            } catch (t: Throwable) {
                setStatus("Chat threw: ${t.message}")
            }
        }
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
                is BackendResult.Ok ->
                    setStatus("Chat history cleared. Session still active.")
                is BackendResult.Err ->
                    setStatus("Chat rebuild failed: [${r.error.name}] ${r.message ?: ""}")
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
                    sessionIdText = null
                    setStatus("Chat session closed.")
                    mainHandler.post { rebuildUi() }
                }
                is BackendResult.Err ->
                    setStatus("Chat close failed: [${r.error.name}] ${r.message ?: ""}")
            }
        }
    }

    private fun buildChatParts(prompt: String, imgBytes: ByteArray?): List<PromptPart> {
        return if (imgBytes != null) {
            listOf(PromptPart.ImageBytes(imgBytes), PromptPart.Text(prompt))
        } else {
            listOf(PromptPart.Text(prompt))
        }
    }

    /* ───── OpenAI-style messages handlers ───── */

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
                    "system"    -> QuickAiChatRole.SYSTEM
                    "user"      -> QuickAiChatRole.USER
                    "assistant" -> QuickAiChatRole.ASSISTANT
                    else        -> continue
                }
                messages.add(QuickAiChatMessage(
                    role = quickRole,
                    parts = listOf(PromptPart.Text(content))
                ))
            }
            messages
        } catch (t: Throwable) { null }
    }

    private fun onOpenAIMessagesRunClicked() {
        val jsonText = openAIMessagesField.text.toString().trim()
        if (jsonText.isBlank()) { setStatus("Messages JSON is empty."); return }
        val messages = parseOpenAIMessages(jsonText)
        if (messages == null) { setStatus("Failed to parse messages JSON. Check format."); return }
        if (messages.isEmpty()) { setStatus("No messages found in JSON."); return }
        if (messages.last().role != QuickAiChatRole.USER) {
            setStatus("Last message must be role=\"user\" to trigger inference.")
            return
        }
        outputText = ""
        outputView.text = ""
        streaming = true
        setStatus("Opening session and running OpenAI messages…")
        mainHandler.post { rebuildUi() }

        val req = buildLoadRequest()
        engineExecutor.execute {
            val e = loadModelInternal(req)
            if (e == null) {
                streaming = false; setStatus("Model load failed.")
                mainHandler.post { rebuildUi() }; return@execute
            }
            val config: QuickAiChatSessionConfig? = null
            if (e.chatSessionId == null) {
                when (val openResult = e.openChatSession(config)) {
                    is BackendResult.Err -> {
                        streaming = false
                        setStatus("Failed to open session: ${openResult.message}")
                        mainHandler.post { rebuildUi() }; return@execute
                    }
                    is BackendResult.Ok -> {
                        sessionIdText = openResult.value
                        mainHandler.post { rebuildUi() }
                    }
                }
            }
            val sink = object : StreamSink {
                override fun onDelta(text: String) {
                    outputText += text
                    mainHandler.post { outputView.append(text) }
                }
                override fun onDone() {
                    streaming = false; setStatus("OpenAI messages chat done.")
                    mainHandler.post { rebuildUi() }
                }
                override fun onError(error: QuickAiError, message: String?) {
                    streaming = false; setStatus("Chat error: [${error.name}] ${message ?: ""}")
                    mainHandler.post { rebuildUi() }
                }
            }
            try {
                when (val r = e.chatRunStreaming(messages, sink)) {
                    is BackendResult.Ok -> {
                        streaming = false
                        lastMetrics = r.value.metrics ?: lastMetrics
                        setStatus("Done. (${r.value.metrics?.totalDurationMs?.toLong() ?: "?"} ms)")
                        mainHandler.post { rebuildUi() }
                    }
                    is BackendResult.Err -> {
                        streaming = false; mainHandler.post { rebuildUi() }
                    }
                }
            } catch (t: Throwable) {
                streaming = false; setStatus("Chat threw: ${t.message}")
                mainHandler.post { rebuildUi() }
            }
        }
    }

    private fun onOpenAIMessagesRunBlockingClicked() {
        val jsonText = openAIMessagesField.text.toString().trim()
        if (jsonText.isBlank()) { setStatus("Messages JSON is empty."); return }
        val messages = parseOpenAIMessages(jsonText)
        if (messages == null) { setStatus("Failed to parse messages JSON. Check format."); return }
        if (messages.isEmpty()) { setStatus("No messages found in JSON."); return }
        if (messages.last().role != QuickAiChatRole.USER) {
            setStatus("Last message must be role=\"user\" to trigger inference.")
            return
        }
        outputText = ""; outputView.text = ""
        setStatus("Opening session and running OpenAI messages (blocking)…")

        val req = buildLoadRequest()
        engineExecutor.execute {
            val e = loadModelInternal(req)
            if (e == null) { setStatus("Model load failed."); return@execute }
            val config: QuickAiChatSessionConfig? = null
            if (e.chatSessionId == null) {
                when (val openResult = e.openChatSession(config)) {
                    is BackendResult.Err -> {
                        setStatus("Failed to open session: ${openResult.message}"); return@execute
                    }
                    is BackendResult.Ok -> {
                        sessionIdText = openResult.value
                        mainHandler.post { rebuildUi() }
                    }
                }
            }
            try {
                when (val r = e.chatRun(messages)) {
                    is BackendResult.Ok -> {
                        outputText = r.value.content
                        lastMetrics = r.value.metrics ?: lastMetrics
                        mainHandler.post { outputView.text = r.value.content }
                        setStatus("Done. (${r.value.metrics?.totalDurationMs?.toLong() ?: "?"} ms)")
                    }
                    is BackendResult.Err ->
                        setStatus("Chat failed: [${r.error.name}] ${r.message ?: ""}")
                }
            } catch (t: Throwable) { setStatus("Chat threw: ${t.message}") }
        }
    }

    /* ───── Image picker handlers ───── */

    private fun onPickImageClicked() {
        imagePickerLauncher.launch(
            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
        )
        setStatus("Opening photo picker…")
    }

    private fun onClearImageClicked() {
        selectedImageBytes = null
        if (::imageStatusView.isInitialized) {
            mainHandler.post { imageStatusView.text = "Image: none" }
        }
        setStatus("Image cleared.")
    }

    private fun readImageBytesAsync(uri: Uri) {
        setStatus("Reading image…")
        Thread({
            try {
                val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
                if (bytes == null || bytes.isEmpty()) {
                    setStatus("Image read failed or empty."); return@Thread
                }
                selectedImageBytes = bytes
                setStatus("Image loaded (${bytes.size} bytes). Tap Run to send.")
                mainHandler.post { rebuildUi() }
            } catch (t: Throwable) {
                setStatus("Failed to read image: ${t.message}")
            }
        }, "SampleTestAPP-ImageRead").apply { isDaemon = true }.start()
    }

    /* ───── Misc helpers ───── */

    private fun setStatus(text: String) {
        statusText = text
        mainHandler.post {
            if (::statusView.isInitialized) statusView.text = text
        }
    }

    /**
     * @brief Builds the default on-device model path for the given
     * (model, quantization) pair rooted in this app's external files
     * dir, so the path lines up with the native C API's hardcoded
     * `./models/<name>-<quant>` prefix (resolve_model_path() in
     * quick_dot_ai_api.cpp).
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
            ModelId.GAUSS3_6_QNN ->
                "$base/models/gauss-3.6b-qnn"
            ModelId.QWEN3_1_7B_Q40 ->
                "$base/models/qwen3-1.7b-q40"
        }
    }

    private fun quantizationSuffix(quant: QuantizationType): String = when (quant) {
        QuantizationType.W4A32 -> "-w4a32"
        QuantizationType.W16A16 -> "-w16a16"
        QuantizationType.W8A16 -> "-w8a16"
        QuantizationType.W32A32 -> "-w32a32"
        QuantizationType.UNKNOWN -> "-w4a32"
    }
}

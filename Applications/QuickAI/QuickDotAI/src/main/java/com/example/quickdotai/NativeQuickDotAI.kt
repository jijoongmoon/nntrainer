// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    NativeQuickDotAI.kt
 * @brief   QuickDotAI implementation backed by the handle-based
 *          quick_dot_ai_api.h (routed through libquickai_jni.so → JNI →
 *          libcausallm_api.so).
 */
package com.example.quickdotai

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import java.io.File

/**
 * @brief Kotlin wrapper around a single `CausalLmHandle` in native code.
 *
 * Non-thread-safe by design — the host app must drive a single instance
 * from a single worker thread.
 *
 * @param appContext Application context required for multimodal image processing.
 *                   Must be non-null to enable runMultimodal/runMultimodalStreaming.
 */
class NativeQuickDotAI(
    private val appContext: Context
) : QuickDotAI {

    override val kind: String = "native"

    override var architecture: String? = null
        private set

    private var handle: Long = 0L
    private var loaded: Boolean = false

    // Image processor for multimodal inference
    private var imageProcessor: LlavaNextImageProcessor? = null

    // Vision backend type (null = text-only mode)
    private var visionBackend: BackendType? = null

    override fun load(req: LoadModelRequest): BackendResult<Unit> {
        Log.i(
            TAG,
            "load() entered: model=${req.model} backend=${req.backend} " +
                "quant=${req.quantization}"
        )
        if (loaded) {
            Log.i(TAG, "load(): already loaded, returning Ok")
            return BackendResult.Ok(Unit)
        }

        if (!NativeCausalLm.ensureLoaded()) {
            Log.e(TAG, "load(): native libs unavailable on this device")
            return BackendResult.Err(
                QuickAiError.MODEL_LOAD_FAILED,
                "libquickai_jni.so / libcausallm_api.so not available on this device"
            )
        }

        val nativeModelOrdinal = mapModelId(req.model)
            ?: run {
                Log.e(TAG, "load(): model ${req.model} has no native ordinal")
                return BackendResult.Err(
                    QuickAiError.UNSUPPORTED,
                    "Model ${req.model} is not supported by NativeQuickDotAI"
                )
            }

        // modelBasePath is passed directly from the caller (e.g. SampleTestAPP
        // sets it to ".../files/models"). The C API uses this as the base
        // directory for resolving model directories
        // (e.g. "<model_base_path>/gauss-3.6-qnn").
        val modelBasePath = req.modelBasePath
        if (modelBasePath != null) {
            Log.i(TAG, "load(): modelBasePath=$modelBasePath")
        }

        return try {
            Log.i(
                TAG,
                "load(): calling loadModelHandleNative(backend=${req.backend.ordinal}, " +
                    "model=$nativeModelOrdinal, quant=${req.quantization.ordinal}, " +
                    "nativeLibDir=${req.nativeLibDir}, modelBasePath=$modelBasePath)"
            )
            val result = NativeCausalLm.loadModelHandleNative(
                backendOrdinal = mapBackend(req.backend),
                modelOrdinal = nativeModelOrdinal,
                quantOrdinal = mapQuant(req.quantization),
                nativeLibDir = req.nativeLibDir,
                modelBasePath = modelBasePath
            )
            Log.i(
                TAG,
                "load(): loadModelHandleNative returned " +
                    "errorCode=${result.errorCode} handle=0x${result.handle.toString(16)}"
            )
            if (result.errorCode != 0 || result.handle == 0L) {
                Log.e(
                    TAG,
                    "load(): loadModelHandle FAILED — errorCode=${result.errorCode} " +
                        "(this is the NATIVE engine; for Gemma4 you want LiteRTLm " +
                        "— check that ModelId.GEMMA4 was NOT requested)"
                )
                BackendResult.Err(
                    QuickAiError.fromNativeCode(result.errorCode),
                    "loadModelHandle failed (errorCode=${result.errorCode})"
                )
            } else {
                handle = result.handle
                loaded = true
                // Architecture is resolved native-side; we report the
                // model key until we add a dedicated native getter.
                architecture = req.model.name

                // Initialize image processor if visionBackend is set
                visionBackend = req.visionBackend
                if (req.visionBackend != null) {
                    imageProcessor = LlavaNextImageProcessor(appContext)
                    Log.i(TAG, "load(): visionBackend=${req.visionBackend}, image processor initialized")
                }

                Log.i(TAG, "load(): SUCCESS, handle=0x${handle.toString(16)}")
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "load(): loadModelHandleNative threw", t)
            BackendResult.Err(QuickAiError.MODEL_LOAD_FAILED, t.message)
        }
    }

    override fun run(prompt: String): BackendResult<String> {
        if (!loaded || handle == 0L) {
            return BackendResult.Err(QuickAiError.NOT_INITIALIZED)
        }
        return try {
            val r = NativeCausalLm.runModelHandleNative(handle, prompt)
            if (r.errorCode != 0) {
                BackendResult.Err(QuickAiError.fromNativeCode(r.errorCode))
            } else {
                BackendResult.Ok(r.output.orEmpty())
            }
        } catch (t: Throwable) {
            Log.e(TAG, "runModelHandleNative threw", t)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    /**
     * @brief Streaming override that forwards deltas from the native
     * `runModelHandleStreaming` entry point into [sink].
     *
     * Threading: this method runs on the caller thread; the native
     * callback is invoked synchronously on the same thread for every
     * delta, so no JNI AttachCurrentThread is needed. Terminal events
     * (onDone / onError) are synthesized from the native return value
     * because the C API reports completion through its return code
     * rather than through the streamer vtable.
     */
    override fun runStreaming(
        prompt: String,
        sink: StreamSink
    ): BackendResult<Unit> {
        if (!loaded || handle == 0L) {
            Log.e(TAG, "runStreaming(): called before load()")
            val err = BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "NativeQuickDotAI has not been loaded yet"
            )
            sink.onError(err.error, err.message)
            return err
        }

        Log.i(TAG, "runStreaming(): prompt length=${prompt.length}")
        return try {
            val errorCode = NativeCausalLm.runModelHandleStreamingNative(
                handle,
                prompt
            ) { delta ->
                // Called on the caller thread (this one). Forward
                // straight to the sink — the contract is that sink
                // implementations are non-blocking.
                sink.onDelta(delta)
            }
            if (errorCode != 0) {
                val err = QuickAiError.fromNativeCode(errorCode)
                Log.e(
                    TAG,
                    "runStreaming(): runModelHandleStreaming failed " +
                        "errorCode=$errorCode (${err.name})"
                )
                sink.onError(err, "runModelHandleStreaming failed (errorCode=$errorCode)")
                BackendResult.Err(err, "runModelHandleStreaming failed (errorCode=$errorCode)")
            } else {
                Log.i(TAG, "runStreaming(): native runner returned NONE, signalling onDone")
                sink.onDone()
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "runStreaming(): runModelHandleStreamingNative threw", t)
            sink.onError(QuickAiError.INFERENCE_FAILED, t.message)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    override fun metrics(): BackendResult<PerformanceMetrics> {
        if (!loaded || handle == 0L) {
            return BackendResult.Err(QuickAiError.NOT_INITIALIZED)
        }
        return try {
            val m = NativeCausalLm.getPerformanceMetricsHandleNative(handle)
            if (m.errorCode != 0) {
                BackendResult.Err(QuickAiError.fromNativeCode(m.errorCode))
            } else {
                BackendResult.Ok(
                    PerformanceMetrics(
                        prefillTokens = m.prefillTokens,
                        prefillDurationMs = m.prefillDurationMs,
                        generationTokens = m.generationTokens,
                        generationDurationMs = m.generationDurationMs,
                        totalDurationMs = m.totalDurationMs,
                        initializationDurationMs = m.initializationDurationMs,
                        peakMemoryKb = m.peakMemoryKb
                    )
                )
            }
        } catch (t: Throwable) {
            Log.e(TAG, "getPerformanceMetricsHandleNative threw", t)
            BackendResult.Err(QuickAiError.UNKNOWN, t.message)
        }
    }

    override fun unload(): BackendResult<Unit> {
        // Cancel any in-flight inference before unloading
        cancel()

        if (!loaded || handle == 0L) {
            return BackendResult.Ok(Unit)
        }
        return try {
            val ec = NativeCausalLm.unloadModelHandleNative(handle)
            loaded = false
            if (ec != 0) {
                BackendResult.Err(QuickAiError.fromNativeCode(ec))
            } else {
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            Log.w(TAG, "unloadModelHandleNative threw", t)
            BackendResult.Err(QuickAiError.UNKNOWN, t.message)
        }
    }

    // --- chat session (dummy) --------------------------------------------

    private var activeSession: NativeChatSession? = null

    override val chatSessionId: String?
        get() = activeSession?.sessionId

    override fun openChatSession(
        config: QuickAiChatSessionConfig?
    ): BackendResult<String> {
        if (!loaded || handle == 0L) {
            return BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "NativeQuickDotAI has not been loaded yet"
            )
        }
        if (activeSession != null) {
            return BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "A chat session is already active (${activeSession!!.sessionId}). " +
                    "Close it before opening a new one."
            )
        }
        val session = NativeChatSession(
            handleProvider = { handle },
            architectureProvider = { architecture },
            config = config
        )
        activeSession = session
        Log.i(TAG, "openChatSession(): created session ${session.sessionId} with handle=0x${handle.toString(16)}")
        return BackendResult.Ok(session.sessionId)
    }

    override fun closeChatSession(): BackendResult<Unit> {
        val session = activeSession
        if (session == null) {
            return BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "No active chat session to close"
            )
        }
        session.close()
        activeSession = null
        Log.i(TAG, "closeChatSession(${session.sessionId}): closed")
        return BackendResult.Ok(Unit)
    }

    override fun chatRun(
        messages: List<QuickAiChatMessage>
    ): BackendResult<QuickAiChatResult> {
        val session = activeSession
            ?: return BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "No active chat session — call openChatSession() first"
            )
        return session.run(messages)
    }

    override fun chatRunStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult> {
        val session = activeSession
        if (session == null) {
            val err = BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "No active chat session — call openChatSession() first"
            )
            sink.onError(err.error, err.message)
            return err
        }
        return session.runStreaming(messages, sink)
    }

    override fun cancel() {
        Log.d(TAG, "cancel(): START, handle=0x${handle.toString(16)}")
        if (handle != 0L) {
            Log.d(TAG, "cancel(): calling NativeCausalLm.cancelModelHandleNative(handle=0x${handle.toString(16)})")
            val result = NativeCausalLm.cancelModelHandleNative(handle)
            Log.d(TAG, "cancel(): cancelModelHandleNative returned $result")
        } else {
            Log.w(TAG, "cancel(): no valid handle to cancel")
        }
    }

    override fun chatCancel() {
        activeSession?.cancel()
    }

    override fun chatRebuild(
        messages: List<QuickAiChatMessage>
    ): BackendResult<Unit> {
        val session = activeSession
            ?: return BackendResult.Err(
                QuickAiError.BAD_REQUEST,
                "No active chat session — call openChatSession() first"
            )
        return session.rebuild(messages)
    }

    override fun close() {
        activeSession?.close()
        activeSession = null
        if (handle != 0L) {
            try {
                NativeCausalLm.destroyModelHandleNative(handle)
            } catch (t: Throwable) {
                Log.w(TAG, "destroyModelHandleNative threw", t)
            }
            handle = 0L
        }
        loaded = false
    }

    // --- multimodal -------------------------------------------------------

    /**
     * @brief Blocking multimodal inference.
     *
     * Preprocesses images from [parts], combines with text prompt, and
     * runs inference through the native engine.
     *
     * @param parts List of PromptPart containing text and/or images
     * @return BackendResult with generated text on success
     */
    override fun runMultimodal(parts: List<PromptPart>): BackendResult<String> {
        if (!loaded || handle == 0L) {
            return BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "NativeQuickDotAI has not been loaded yet"
            )
        }

        val processor = imageProcessor
        if (processor == null) {
            return BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "Multimodal not enabled — reload with LoadModelRequest.visionBackend set"
            )
        }

        // Extract image and text from parts
        val multimodalInput = prepareMultimodalInput(parts, processor)
            ?: return BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "No valid image found in parts"
            )

        val textPrompt = extractTextPrompt(parts)

        Log.i(
            TAG,
            "runMultimodal(): numPatches=${multimodalInput.numPatches}, " +
                "originalSize=${multimodalInput.originalHeight}x${multimodalInput.originalWidth}, " +
                "prompt length=${textPrompt.length}"
        )

        return try {
            val result = NativeCausalLm.runMultimodalHandleNative(
                handle,
                textPrompt,
                multimodalInput.pixelValues,
                multimodalInput.numPatches,
                multimodalInput.originalHeight,
                multimodalInput.originalWidth
            )
            if (result.errorCode != 0) {
                val err = QuickAiError.fromNativeCode(result.errorCode)
                Log.e(TAG, "runMultimodal(): failed with errorCode=${result.errorCode}")
                BackendResult.Err(err, "runMultimodalHandle failed (errorCode=${result.errorCode})")
            } else {
                Log.i(TAG, "runMultimodal(): success, output length=${result.output?.length ?: 0}")
                BackendResult.Ok(result.output.orEmpty())
            }
        } catch (t: Throwable) {
            Log.e(TAG, "runMultimodal(): threw exception", t)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    /**
     * @brief Streaming multimodal inference.
     *
     * Preprocesses images from [parts], combines with text prompt, and
     * runs streaming inference through the native engine. Deltas are
     * forwarded to [sink] as they are generated.
     *
     * @param parts List of PromptPart containing text and/or images
     * @param sink StreamSink to receive streaming output
     * @return BackendResult<Unit> on completion
     */
    override fun runMultimodalStreaming(
        parts: List<PromptPart>,
        sink: StreamSink
    ): BackendResult<Unit> {
        if (!loaded || handle == 0L) {
            val err = BackendResult.Err(
                QuickAiError.NOT_INITIALIZED,
                "NativeQuickDotAI has not been loaded yet"
            )
            sink.onError(err.error, err.message)
            return err
        }

        val processor = imageProcessor
        if (processor == null) {
            val err = BackendResult.Err(
                QuickAiError.UNSUPPORTED,
                "MultimodalStreaming not enabled — reload with LoadModelRequest.visionBackend set"
            )
            sink.onError(err.error, err.message)
            return err
        }

        // Extract image and text from parts
        val multimodalInput = prepareMultimodalInput(parts, processor)
        if (multimodalInput == null) {
            val err = BackendResult.Err(
                QuickAiError.INVALID_PARAMETER,
                "No valid image found in parts"
            )
            sink.onError(err.error, err.message)
            return err
        }

        val textPrompt = extractTextPrompt(parts)

        Log.i(
            TAG,
            "runMultimodalStreaming(): numPatches=${multimodalInput.numPatches}, " +
                "originalSize=${multimodalInput.originalHeight}x${multimodalInput.originalWidth}, " +
                "prompt length=${textPrompt.length}"
        )

        return try {
            val errorCode = NativeCausalLm.runMultimodalHandleStreamingNative(
                handle,
                textPrompt,
                multimodalInput.pixelValues,
                multimodalInput.numPatches,
                multimodalInput.originalHeight,
                multimodalInput.originalWidth
            ) { delta ->
                sink.onDelta(delta)
            }

            if (errorCode != 0) {
                val err = QuickAiError.fromNativeCode(errorCode)
                Log.e(TAG, "runMultimodalStreaming(): failed with errorCode=$errorCode")
                sink.onError(err, "runMultimodalHandleStreaming failed (errorCode=$errorCode)")
                BackendResult.Err(err, "runMultimodalHandleStreaming failed (errorCode=$errorCode)")
            } else {
                Log.i(TAG, "runMultimodalStreaming(): success")
                sink.onDone()
                BackendResult.Ok(Unit)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "runMultimodalStreaming(): threw exception", t)
            sink.onError(QuickAiError.INFERENCE_FAILED, t.message)
            BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
        }
    }

    /**
     * @brief Prepare multimodal input from PromptPart list.
     *
     * Extracts the first image from parts and preprocesses it using
     * LlavaNextImageProcessor.
     *
     * @return MultimodalInput with preprocessed pixel values, or null if no image found
     */
    private fun prepareMultimodalInput(
        parts: List<PromptPart>,
        processor: LlavaNextImageProcessor
    ): NativeCausalLm.MultimodalInput? {
        for (part in parts) {
            when (part) {
                is PromptPart.ImageFile -> {
                    val file = File(part.absolutePath)
                    if (!file.exists() || !file.canRead()) {
                        Log.w(TAG, "Image file not readable: ${part.absolutePath}")
                        continue
                    }
                    val bitmap = BitmapFactory.decodeFile(part.absolutePath)
                    if (bitmap == null) {
                        Log.w(TAG, "Failed to decode image: ${part.absolutePath}")
                        continue
                    }
                    val modelInput = processor.preprocess(bitmap)
                    return NativeCausalLm.MultimodalInput(
                        pixelValues = modelInput.pixelValues,
                        numPatches = modelInput.pixelValues.size / (processor.getCropSize() * processor.getCropSize() * 3),
                        originalHeight = modelInput.originalSize.first,
                        originalWidth = modelInput.originalSize.second
                    )
                }
                is PromptPart.ImageBytes -> {
                    if (part.bytes.isEmpty()) {
                        Log.w(TAG, "Image bytes are empty")
                        continue
                    }
                    val bitmap = BitmapFactory.decodeByteArray(part.bytes, 0, part.bytes.size)
                    if (bitmap == null) {
                        Log.w(TAG, "Failed to decode image from bytes")
                        continue
                    }
                    val modelInput = processor.preprocess(bitmap)
                    return NativeCausalLm.MultimodalInput(
                        pixelValues = modelInput.pixelValues,
                        numPatches = modelInput.pixelValues.size / (processor.getCropSize() * processor.getCropSize() * 3),
                        originalHeight = modelInput.originalSize.first,
                        originalWidth = modelInput.originalSize.second
                    )
                }
                is PromptPart.Text -> { /* skip text parts */ }
            }
        }
        return null
    }

    /**
     * @brief Extract text prompt from PromptPart list.
     *
     * Concatenates all Text parts into a single prompt string.
     */
    private fun extractTextPrompt(parts: List<PromptPart>): String {
        return parts.filterIsInstance<PromptPart.Text>()
            .joinToString(" ") { it.text }
            .ifEmpty { "Describe this image." }
    }

    // --- enum → native-ordinal mapping ---------------------------------

    /**
     * @brief Maps a ModelId to the C enum ordinal in quick_dot_ai_api.h.
     * Returns null for values that are Kotlin-only (e.g. GEMMA4) —
     * those are routed to [LiteRTLm] instead and never reach this
     * engine.
     */
    private fun mapModelId(m: ModelId): Int? = when (m) {
        ModelId.QWEN3_0_6B -> 0 // CAUSAL_LM_MODEL_QWEN3_0_6B
        ModelId.GEMMA4 -> null
        ModelId.GAUSS3_6_QNN -> 2 // CAUSAL_LM_MODEL_GAUSS3_6_QNN
        ModelId.GAUSS3_8_QNN -> 3 // CAUSAL_LM_MODEL_GAUSS3_8_QNN
        ModelId.QWEN3_1_7B_Q40 -> 4 // CAUSAL_LM_MODEL_QWEN3_1_7B_Q40
        ModelId.GAUSS3_8_VISION_QNN -> 6 // CAUSAL_LM_MODEL_GAUSS3_8_VIT_QNN
        ModelId.GAUSS3_6 -> 7 // CAUSAL_LM_MODEL_GAUSS3_6
        ModelId.TINY_BERT ->8 // CAUSAL_LM_MODEL_TINY_BERT
    }

    private fun mapBackend(b: BackendType): Int = when (b) {
        BackendType.CPU -> 0
        BackendType.GPU -> 1
        BackendType.NPU -> 2
    }

    private fun mapQuant(q: QuantizationType): Int = when (q) {
        QuantizationType.UNKNOWN -> 0
        QuantizationType.W4A32 -> 1
        QuantizationType.W16A16 -> 2
        QuantizationType.W8A16 -> 3
        QuantizationType.W32A32 -> 4
    }

    companion object {
        private const val TAG = "NativeQuickDotAI"
    }
}

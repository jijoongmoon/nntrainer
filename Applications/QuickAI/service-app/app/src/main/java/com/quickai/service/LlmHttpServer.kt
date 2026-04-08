package com.quickai.service

import com.google.gson.Gson
import com.google.gson.JsonParser
import fi.iki.elonen.NanoHTTPD

/**
 * REST API server for LLM inference
 * Default port: 8080
 */
class LlmHttpServer(port: Int = 8080) : NanoHTTPD(port) {

    private val gson = Gson()

    override fun serve(session: IHTTPSession): Response {
        val uri = session.uri
        val method = session.method

        return try {
            when {
                // Health check
                uri == "/v1/health" && method == Method.GET ->
                    jsonResponse(mapOf("status" to "ok", "backend" to NativeEngine.nativeGetLoadedBackend()))

                // List models
                uri == "/v1/models" && method == Method.GET ->
                    handleGetModels()

                // Load model
                uri == "/v1/engine/load" && method == Method.POST ->
                    handleLoadModel(session)

                // Unload model
                uri == "/v1/engine/unload" && method == Method.POST ->
                    handleUnloadModel()

                // Generate text
                uri == "/v1/generate" && method == Method.POST ->
                    handleGenerate(session)

                // Performance metrics
                uri == "/v1/metrics" && method == Method.GET ->
                    handleGetMetrics()

                else ->
                    errorResponse(Response.Status.NOT_FOUND, "Not found: $uri")
            }
        } catch (e: Exception) {
            errorResponse(Response.Status.INTERNAL_ERROR, "Error: ${e.message}")
        }
    }

    private fun handleGetModels(): Response {
        val models = listOf(
            mapOf(
                "id" to "qwen3-0.6b",
                "name" to "Qwen3 0.6B",
                "backends" to listOf("cpu", "npu"),
                "model_type" to NativeEngine.MODEL_QWEN3_0_6B
            ),
            mapOf(
                "id" to "gemma4-e2b",
                "name" to "Gemma4 E2B",
                "backends" to listOf("gpu2"),
                "model_type" to NativeEngine.MODEL_GEMMA4_E2B
            )
        )
        return jsonResponse(mapOf("models" to models))
    }

    private fun handleLoadModel(session: IHTTPSession): Response {
        val body = readBody(session)
        val json = JsonParser.parseString(body).asJsonObject

        val backend = json.get("backend")?.asString ?: "cpu"
        val modelId = json.get("model_id")?.asString ?: "qwen3-0.6b"
        val quantType = json.get("quant_type")?.asInt ?: NativeEngine.QUANT_W4A32

        val backendInt = when (backend) {
            "cpu" -> NativeEngine.BACKEND_CPU
            "gpu" -> NativeEngine.BACKEND_GPU
            "npu" -> NativeEngine.BACKEND_NPU
            "gpu2" -> NativeEngine.BACKEND_GPU2
            else -> return errorResponse(Response.Status.BAD_REQUEST, "Unknown backend: $backend")
        }

        val modelType = when (modelId) {
            "qwen3-0.6b" -> NativeEngine.MODEL_QWEN3_0_6B
            "gemma4-e2b" -> NativeEngine.MODEL_GEMMA4_E2B
            else -> return errorResponse(Response.Status.BAD_REQUEST, "Unknown model: $modelId")
        }

        val err = NativeEngine.nativeLoadModel(backendInt, modelType, quantType)
        return if (err == NativeEngine.ERROR_NONE) {
            jsonResponse(mapOf("status" to "loaded", "backend" to backend, "model" to modelId))
        } else {
            errorResponse(Response.Status.INTERNAL_ERROR, "Load failed with error code: $err")
        }
    }

    private fun handleUnloadModel(): Response {
        val err = NativeEngine.nativeUnloadModel()
        return if (err == NativeEngine.ERROR_NONE) {
            jsonResponse(mapOf("status" to "unloaded"))
        } else {
            errorResponse(Response.Status.INTERNAL_ERROR, "Unload failed with error code: $err")
        }
    }

    private fun handleGenerate(session: IHTTPSession): Response {
        val body = readBody(session)
        val json = JsonParser.parseString(body).asJsonObject

        val prompt = json.get("prompt")?.asString
            ?: return errorResponse(Response.Status.BAD_REQUEST, "Missing 'prompt' field")

        val useChatTemplate = json.get("use_chat_template")?.asBoolean ?: true
        NativeEngine.nativeSetOptions(useChatTemplate, false, false)

        val output = NativeEngine.nativeRunModel(prompt)
            ?: return errorResponse(Response.Status.INTERNAL_ERROR, "Inference failed")

        val metrics = NativeEngine.getMetrics()

        val result = mutableMapOf<String, Any>(
            "text" to output,
        )
        if (metrics != null) {
            result["metrics"] = mapOf(
                "prefill_tokens" to metrics.prefillTokens,
                "prefill_duration_ms" to metrics.prefillDurationMs,
                "generation_tokens" to metrics.generationTokens,
                "generation_duration_ms" to metrics.generationDurationMs,
                "total_duration_ms" to metrics.totalDurationMs,
                "peak_memory_kb" to metrics.peakMemoryKb
            )
        }

        return jsonResponse(result)
    }

    private fun handleGetMetrics(): Response {
        val metrics = NativeEngine.getMetrics()
            ?: return errorResponse(Response.Status.INTERNAL_ERROR, "No metrics available")

        return jsonResponse(mapOf(
            "prefill_tokens" to metrics.prefillTokens,
            "prefill_duration_ms" to metrics.prefillDurationMs,
            "generation_tokens" to metrics.generationTokens,
            "generation_duration_ms" to metrics.generationDurationMs,
            "total_duration_ms" to metrics.totalDurationMs,
            "initialization_duration_ms" to metrics.initDurationMs,
            "peak_memory_kb" to metrics.peakMemoryKb
        ))
    }

    private fun readBody(session: IHTTPSession): String {
        val contentLength = session.headers["content-length"]?.toIntOrNull() ?: 0
        val buffer = ByteArray(contentLength)
        session.inputStream.read(buffer, 0, contentLength)
        return String(buffer)
    }

    private fun jsonResponse(data: Any): Response {
        val json = gson.toJson(data)
        return newFixedLengthResponse(Response.Status.OK, "application/json", json)
    }

    private fun errorResponse(status: Response.Status, message: String): Response {
        val json = gson.toJson(mapOf("error" to message))
        return newFixedLengthResponse(status, "application/json", json)
    }
}

package com.quickai.client.api

import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

/**
 * REST client for quick.ai.service
 */
class ServiceClient(
    private val host: String = "localhost",
    private val port: Int = 8080
) {
    private val baseUrl = "http://$host:$port"
    private val gson = Gson()
    private val jsonType = "application/json".toMediaType()

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(300, TimeUnit.SECONDS)  // LLM inference can be slow
        .writeTimeout(10, TimeUnit.SECONDS)
        .build()

    // --- Health ---

    data class HealthResponse(val status: String, val backend: Int)

    suspend fun health(): Result<HealthResponse> = get("/v1/health")

    // --- Models ---

    data class ModelInfo(
        val id: String,
        val name: String,
        val backends: List<String>,
        val model_type: Int
    )
    data class ModelsResponse(val models: List<ModelInfo>)

    suspend fun getModels(): Result<ModelsResponse> = get("/v1/models")

    // --- Engine Load/Unload ---

    data class LoadRequest(
        val backend: String,
        val model_id: String,
        val quant_type: Int = 1
    )
    data class StatusResponse(val status: String)

    suspend fun loadModel(backend: String, modelId: String, quantType: Int = 1): Result<StatusResponse> =
        post("/v1/engine/load", LoadRequest(backend, modelId, quantType))

    suspend fun unloadModel(): Result<StatusResponse> =
        post("/v1/engine/unload", "")

    // --- Model Download ---

    data class DownloadRequest(val files: Map<String, String> = emptyMap())

    suspend fun downloadModel(modelId: String, fileUrls: Map<String, String> = emptyMap()): Result<StatusResponse> =
        post("/v1/models/$modelId/download", DownloadRequest(fileUrls))

    data class ModelStatusResponse(
        val model_id: String = "",
        val status: String = "",
        val download: DownloadInfo? = null
    )
    data class DownloadInfo(
        val state: String = "",
        val progress: Float = 0f,
        val downloaded_bytes: Long = 0,
        val total_bytes: Long = 0,
        val error: String = ""
    )

    suspend fun getModelStatus(modelId: String): Result<ModelStatusResponse> =
        get("/v1/models/$modelId/status")

    suspend fun deleteModel(modelId: String): Result<StatusResponse> =
        delete("/v1/models/$modelId")

    data class StorageInfo(
        val models_dir: String = "",
        val used_mb: Long = 0,
        val free_mb: Long = 0
    )

    suspend fun getStorageInfo(): Result<StorageInfo> = get("/v1/storage")

    // --- Generate ---

    data class GenerateRequest(
        val prompt: String,
        val use_chat_template: Boolean = true
    )

    data class Metrics(
        val prefill_tokens: Int = 0,
        val prefill_duration_ms: Double = 0.0,
        val generation_tokens: Int = 0,
        val generation_duration_ms: Double = 0.0,
        val total_duration_ms: Double = 0.0,
        val peak_memory_kb: Long = 0
    )

    data class GenerateResponse(
        val text: String,
        val metrics: Metrics? = null
    )

    suspend fun generate(prompt: String, useChatTemplate: Boolean = true): Result<GenerateResponse> =
        post("/v1/generate", GenerateRequest(prompt, useChatTemplate))

    // --- Metrics ---

    data class MetricsResponse(
        val prefill_tokens: Int = 0,
        val prefill_duration_ms: Double = 0.0,
        val generation_tokens: Int = 0,
        val generation_duration_ms: Double = 0.0,
        val total_duration_ms: Double = 0.0,
        val initialization_duration_ms: Double = 0.0,
        val peak_memory_kb: Long = 0
    )

    suspend fun getMetrics(): Result<MetricsResponse> = get("/v1/metrics")

    // --- Internal HTTP helpers ---

    private suspend inline fun <reified T> get(path: String): Result<T> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder().url("$baseUrl$path").get().build()
            val response = client.newCall(request).execute()
            val body = response.body?.string() ?: ""
            if (response.isSuccessful) {
                Result.success(gson.fromJson(body, T::class.java))
            } else {
                Result.failure(ServiceException(response.code, body))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private suspend inline fun <reified T> post(path: String, body: Any): Result<T> = withContext(Dispatchers.IO) {
        try {
            val jsonBody = if (body is String && body.isEmpty()) "" else gson.toJson(body)
            val requestBody = jsonBody.toRequestBody(jsonType)
            val request = Request.Builder().url("$baseUrl$path").post(requestBody).build()
            val response = client.newCall(request).execute()
            val respBody = response.body?.string() ?: ""
            if (response.isSuccessful) {
                Result.success(gson.fromJson(respBody, T::class.java))
            } else {
                Result.failure(ServiceException(response.code, respBody))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private suspend inline fun <reified T> delete(path: String): Result<T> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder().url("$baseUrl$path").delete().build()
            val response = client.newCall(request).execute()
            val body = response.body?.string() ?: ""
            if (response.isSuccessful) {
                Result.success(gson.fromJson(body, T::class.java))
            } else {
                Result.failure(ServiceException(response.code, body))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    class ServiceException(val code: Int, val body: String) :
        Exception("HTTP $code: $body")
}

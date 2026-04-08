package com.quickai.service

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

/**
 * Manages model discovery, download, storage, and lifecycle.
 */
class ModelManager(private val appContext: Context) {

    companion object {
        private const val TAG = "ModelManager"
    }

    private val gson = Gson()
    private val modelsDir = File(appContext.filesDir, "models")
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(600, TimeUnit.SECONDS)
        .build()

    // Download progress tracking: model_id -> DownloadProgress
    private val downloadProgress = ConcurrentHashMap<String, DownloadProgress>()

    // Model manifest (built-in + from remote)
    private val manifest = mutableListOf<ModelEntry>()

    init {
        modelsDir.mkdirs()
        loadBuiltinManifest()
    }

    // --- Data classes ---

    data class ModelEntry(
        val id: String,
        val name: String,
        val backend: String,
        val model_type: Int,
        val files: List<ModelFile>,
        val total_size: Long = 0
    )

    data class ModelFile(
        val name: String,
        val url: String,
        val size: Long = 0
    )

    data class DownloadProgress(
        val modelId: String,
        var state: DownloadState = DownloadState.PENDING,
        var downloadedBytes: Long = 0,
        var totalBytes: Long = 0,
        var error: String? = null
    ) {
        val progress: Float
            get() = if (totalBytes > 0) downloadedBytes.toFloat() / totalBytes else 0f
    }

    enum class DownloadState {
        PENDING, DOWNLOADING, COMPLETED, FAILED
    }

    enum class ModelStatus {
        NOT_DOWNLOADED, DOWNLOADING, DOWNLOADED, LOADED
    }

    // --- Built-in manifest ---

    private fun loadBuiltinManifest() {
        manifest.clear()
        manifest.addAll(listOf(
            ModelEntry(
                id = "qwen3-0.6b",
                name = "Qwen3 0.6B (CPU)",
                backend = "cpu",
                model_type = NativeEngine.MODEL_QWEN3_0_6B,
                files = listOf(
                    ModelFile("qwen3-0.6b-q40-fp32-arm.bin", "", 415_000_000),
                    ModelFile("tokenizer.json", "", 2_500_000)
                ),
                total_size = 417_500_000
            ),
            ModelEntry(
                id = "gemma4-e2b",
                name = "Gemma4 E2B (GPU2/LiteRT-LM)",
                backend = "gpu2",
                model_type = NativeEngine.MODEL_GEMMA4_E2B,
                files = listOf(
                    ModelFile("gemma-4-E2B-it-litert-lm.task", "", 2_800_000_000)
                ),
                total_size = 2_800_000_000
            )
        ))
    }

    // --- Query ---

    fun getModels(): List<Map<String, Any>> {
        return manifest.map { entry ->
            mapOf(
                "id" to entry.id,
                "name" to entry.name,
                "backend" to entry.backend,
                "model_type" to entry.model_type,
                "total_size" to entry.total_size,
                "status" to getModelStatus(entry.id).name
            )
        }
    }

    fun getModelStatus(modelId: String): ModelStatus {
        // Check if currently downloading
        downloadProgress[modelId]?.let {
            if (it.state == DownloadState.DOWNLOADING) return ModelStatus.DOWNLOADING
        }

        // Check if loaded
        if (NativeEngine.nativeGetLoadedBackend() >= 0) {
            // Simple check - in production, track which model is loaded
            return ModelStatus.LOADED
        }

        // Check if files exist on disk
        val modelDir = File(modelsDir, modelId)
        val entry = manifest.find { it.id == modelId } ?: return ModelStatus.NOT_DOWNLOADED
        val allFilesExist = entry.files.all { File(modelDir, it.name).exists() }

        return if (allFilesExist) ModelStatus.DOWNLOADED else ModelStatus.NOT_DOWNLOADED
    }

    fun getDownloadProgress(modelId: String): DownloadProgress? {
        return downloadProgress[modelId]
    }

    // --- Download ---

    fun startDownload(modelId: String, fileUrls: Map<String, String>): Boolean {
        val entry = manifest.find { it.id == modelId } ?: return false

        if (downloadProgress[modelId]?.state == DownloadState.DOWNLOADING) {
            Log.w(TAG, "Download already in progress for $modelId")
            return false
        }

        val progress = DownloadProgress(
            modelId = modelId,
            state = DownloadState.PENDING,
            totalBytes = entry.total_size
        )
        downloadProgress[modelId] = progress

        // Run download in background thread
        Thread {
            try {
                progress.state = DownloadState.DOWNLOADING
                val modelDir = File(modelsDir, modelId)
                modelDir.mkdirs()

                for (file in entry.files) {
                    val url = fileUrls[file.name] ?: file.url
                    if (url.isEmpty()) {
                        Log.w(TAG, "No URL for ${file.name}, skipping")
                        continue
                    }

                    Log.i(TAG, "Downloading ${file.name} from $url")
                    downloadFile(url, File(modelDir, file.name), progress)
                }

                progress.state = DownloadState.COMPLETED
                Log.i(TAG, "Download completed for $modelId")

            } catch (e: Exception) {
                progress.state = DownloadState.FAILED
                progress.error = e.message
                Log.e(TAG, "Download failed for $modelId", e)
            }
        }.start()

        return true
    }

    private fun downloadFile(url: String, destFile: File, progress: DownloadProgress) {
        val request = Request.Builder().url(url).build()
        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw RuntimeException("Download failed: HTTP ${response.code}")
        }

        val body = response.body ?: throw RuntimeException("Empty response body")
        val contentLength = body.contentLength()

        body.byteStream().use { input ->
            FileOutputStream(destFile).use { output ->
                val buffer = ByteArray(8192)
                var bytesRead: Int
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    progress.downloadedBytes += bytesRead
                }
            }
        }

        Log.i(TAG, "Downloaded: ${destFile.name} (${destFile.length()} bytes)")
    }

    // --- Delete ---

    fun deleteModel(modelId: String): Boolean {
        val modelDir = File(modelsDir, modelId)
        if (!modelDir.exists()) return false
        modelDir.deleteRecursively()
        downloadProgress.remove(modelId)
        Log.i(TAG, "Deleted model: $modelId")
        return true
    }

    // --- Storage info ---

    fun getStorageInfo(): Map<String, Any> {
        val totalUsed = modelsDir.walkTopDown().filter { it.isFile }.sumOf { it.length() }
        val freeSpace = modelsDir.freeSpace
        return mapOf(
            "models_dir" to modelsDir.absolutePath,
            "used_bytes" to totalUsed,
            "used_mb" to totalUsed / (1024 * 1024),
            "free_bytes" to freeSpace,
            "free_mb" to freeSpace / (1024 * 1024)
        )
    }
}

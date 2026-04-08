package com.quickai.service

/**
 * JNI bridge to libcausallm_api.so
 */
object NativeEngine {

    init {
        System.loadLibrary("causallm_api")
    }

    // Backend types (matches causal_lm_api.h)
    const val BACKEND_CPU = 0
    const val BACKEND_GPU = 1
    const val BACKEND_NPU = 2
    const val BACKEND_GPU2 = 3

    // Model types
    const val MODEL_QWEN3_0_6B = 0
    const val MODEL_GEMMA4_E2B = 1

    // Quantization types
    const val QUANT_UNKNOWN = 0
    const val QUANT_W4A32 = 1
    const val QUANT_W16A16 = 2
    const val QUANT_W8A16 = 3
    const val QUANT_W32A32 = 4

    // Error codes
    const val ERROR_NONE = 0
    const val ERROR_INVALID_PARAMETER = 1
    const val ERROR_MODEL_LOAD_FAILED = 2
    const val ERROR_INFERENCE_FAILED = 3
    const val ERROR_NOT_INITIALIZED = 4

    external fun nativeSetOptions(useChatTemplate: Boolean, debugMode: Boolean, verbose: Boolean): Int
    external fun nativeLoadModel(backend: Int, modelType: Int, quantType: Int): Int
    external fun nativeRunModel(prompt: String): String?
    external fun nativeUnloadModel(): Int
    external fun nativeSetModelBasePath(basePath: String): Int
    external fun nativeGetLoadedBackend(): Int
    external fun nativeGetPerformanceMetrics(): FloatArray?

    data class PerformanceMetrics(
        val prefillTokens: Int,
        val prefillDurationMs: Double,
        val generationTokens: Int,
        val generationDurationMs: Double,
        val totalDurationMs: Double,
        val initDurationMs: Double,
        val peakMemoryKb: Long
    )

    fun getMetrics(): PerformanceMetrics? {
        val raw = nativeGetPerformanceMetrics() ?: return null
        if (raw.size < 7) return null
        return PerformanceMetrics(
            prefillTokens = raw[0].toInt(),
            prefillDurationMs = raw[1].toDouble(),
            generationTokens = raw[2].toInt(),
            generationDurationMs = raw[3].toDouble(),
            totalDurationMs = raw[4].toDouble(),
            initDurationMs = raw[5].toDouble(),
            peakMemoryKb = raw[6].toLong()
        )
    }
}

# QuickDotAI AAR — API

On-device LLM inference. `NativeQuickDotAI` routes non-Gemma models
through JNI to `libcausallm_api.so`; `LiteRTLm` routes Gemma-family
models through LiteRT-LM and also supports image input.

## Dependency

```kotlin
dependencies {
    implementation(project(":QuickDotAI"))
}
```

## API surface (`com.example.quickdotai`)

```kotlin
interface QuickDotAI {
    val kind: String                       // "native" or "litert-lm"
    val architecture: String?

    fun load(req: LoadModelRequest): BackendResult<Unit>

    fun run(prompt: String): BackendResult<String>
    fun runStreaming(prompt: String, sink: StreamSink): BackendResult<Unit>

    fun runMultimodal(parts: List<PromptPart>): BackendResult<String>
    fun runMultimodalStreaming(parts: List<PromptPart>, sink: StreamSink): BackendResult<Unit>

    fun unload(): BackendResult<Unit>
    fun metrics(): BackendResult<PerformanceMetrics>
    fun close()
}

data class LoadModelRequest(
    val backend: BackendType = BackendType.GPU,
    val model: ModelId,
    val quantization: QuantizationType = QuantizationType.W4A32,
    val modelPath: String? = null,         // required for GEMMA4
    val visionBackend: BackendType? = null, // non-null enables runMultimodal*
    val cacheDir: String? = null,
)

sealed class PromptPart {
    data class Text(val text: String) : PromptPart()
    data class ImageFile(val absolutePath: String) : PromptPart()
    data class ImageBytes(val bytes: ByteArray) : PromptPart()
}

sealed class BackendResult<out T> {
    data class Ok<T>(val value: T) : BackendResult<T>()
    data class Err(val error: QuickAiError, val message: String? = null) : BackendResult<Nothing>()
}

interface StreamSink {
    fun onDelta(text: String)
    fun onDone()
    fun onError(error: QuickAiError, message: String?)
}

enum class BackendType      { CPU, GPU, NPU }
enum class ModelId          { QWEN3_0_6B, GAUSS2_5, GEMMA4 }
enum class QuantizationType { UNKNOWN, W4A32, W16A16, W8A16, W32A32 }
enum class QuickAiError {
    NONE, INVALID_PARAMETER, MODEL_LOAD_FAILED, INFERENCE_FAILED,
    NOT_INITIALIZED, INFERENCE_NOT_RUN, UNKNOWN,
    QUEUE_FULL, MODEL_NOT_FOUND, UNSUPPORTED, BAD_REQUEST
}

data class PerformanceMetrics(
    val prefillTokens: Int, val prefillDurationMs: Double,
    val generationTokens: Int, val generationDurationMs: Double,
    val totalDurationMs: Double, val initializationDurationMs: Double,
    val peakMemoryKb: Long,
)
```

## Minimal example

```kotlin
val engine: QuickDotAI = when (req.model) {
    ModelId.GEMMA4 -> LiteRTLm(applicationContext)
    else           -> NativeQuickDotAI()
}

engine.load(LoadModelRequest(
    model = ModelId.GEMMA4,
    backend = BackendType.GPU,
    visionBackend = BackendType.GPU,     // enables images
    modelPath = "/sdcard/.../gemma-4-E2B-it.litertlm",
))

// Text
engine.runStreaming("Hi.", sink)

// Image + text
engine.runMultimodalStreaming(
    listOf(
        PromptPart.ImageFile("/sdcard/photo.jpg"),
        PromptPart.Text("Describe this picture."),
    ),
    sink,
)

engine.close()
```

## Rules

- Call `load()` exactly once before any `run*`.
- A single instance is **not thread-safe** — drive it from one worker thread.
- `runMultimodal*` on a text-only engine (or `NativeQuickDotAI`) returns `QuickAiError.UNSUPPORTED`.
- `arm64-v8a` only.

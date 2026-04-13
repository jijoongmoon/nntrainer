# QuickAI - Architecture & Implementation Plan

## 1. Overview

**QuickAI** is an on-device AI inference platform for Android that exposes
`libcausallm_api.so` (an nntrainer-backed CausalLM engine) to multiple client
apps through a shared background service.

```
 ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
 │  ClientApp #1  │   │  ClientApp #2  │   │  LauncherApp   │
 │ (arbitrary UI) │   │ (arbitrary UI) │   │  (starts svc)  │
 └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
         │   HTTP/JSON         │   HTTP/JSON        │
         └─────────────┬───────┴────────────────────┘
                       ▼
              ┌────────────────────┐
              │   QuickAIService   │   process = ":remote"
              │   (exported=true)  │   runs a loopback REST server
              │                    │
              │  ┌──────────────┐  │
              │  │ HttpServer   │  │   NanoHTTPD @ 127.0.0.1:3453
              │  └──────┬───────┘  │
              │         ▼          │
              │  ┌──────────────┐  │
              │  │ Dispatcher   │  │   routing + auth + model-id
              │  └──────┬───────┘  │
              │         ▼          │
              │  ┌──────────────┐  │
              │  │ModelRegistry │  │   model_key → ModelWorker
              │  └──┬───────┬───┘  │
              │     │       │      │
              │  ┌──▼──┐ ┌──▼──┐   │
              │  │ MW1 │ │ MW2 │   │   one Thread + FIFO queue
              │  └──┬──┘ └──┬──┘   │   per loaded model
              │     │       │      │
              │  ┌──▼──┐ ┌──▼──┐   │
              │  │Back │ │Back │   │   Backend = Native | LiteRT-LM
              │  │-end │ │-end │   │
              │  └──┬──┘ └──┬──┘   │
              └─────┼───────┼──────┘
                    ▼       ▼
           libcausallm_api.so   litert-lm (gemma4 only)
                    │
                    ▼
              nntrainer
```

## 2. Components

### 2.1 `LauncherApp` (Android Activity)
- **File:** `LauncherApp/src/main/java/com/example/QuickAI/LauncherApp.kt`
- **Role:** Minimal launcher UI whose only job is to
  1. Request the `POST_NOTIFICATIONS` runtime permission (Android 13+).
  2. Start `QuickAIService` as a **foreground service**.
  3. Show the service status (running / port / loaded models).

Android does not allow long-running background services without a hosting app
process, so the launcher exists purely to bootstrap the service. Once started,
the service remains alive independently of the launcher's UI lifecycle.

### 2.2 `QuickAIService` (Foreground Service, `:remote` process)
- **File:** `LauncherApp/src/main/java/com/example/QuickAI/QuickAIService.kt`
- **Manifest:**
  ```xml
  <service
      android:name=".QuickAIService"
      android:enabled="true"
      android:exported="true"
      android:process=":remote"
      android:foregroundServiceType="dataSync" />
  ```
- **Responsibilities:**
  - Promotes itself to foreground (ongoing notification) so it is not killed.
  - Starts the embedded HTTP server (NanoHTTPD) bound to `127.0.0.1:3453`.
  - Creates `ModelRegistry` and `RequestDispatcher`.
  - Publishes the actual bound port via a `ContentProvider` (`QuickAIPortProvider`)
    so clients do not have to hard-code the port if we ever change it.
- **Lifecycle:** `START_STICKY` — restarted by Android if killed.

### 2.3 REST HTTP Server (NanoHTTPD)
- **File:** `service/HttpServer.kt`
- **Binding:** `InetAddress.getByName("127.0.0.1")`, port **3453** (fallback
  to ephemeral if taken).
- **Library:** [NanoHTTPD](https://github.com/NanoHttpd/nanohttpd) — tiny
  (~1 jar, ~200 KB), no background executor of its own, easy to embed.
- **Why a fixed loopback port works on Android:**
  - Binding to `127.0.0.1` does **not** require the `INTERNET` permission for
    the server side, but clients in other apps DO need `INTERNET` to `connect()`
    to localhost, so `INTERNET` is declared on both apps.
  - Because the service runs in a single `:remote` process shared across all
    client apps, there is exactly one listener — no cross-app port conflicts.
  - From Android 9 (API 28) onward, plain-text HTTP is blocked by default for
    non-localhost hosts; `127.0.0.1` is whitelisted automatically, but we also
    ship a `network_security_config.xml` allowing cleartext to `127.0.0.1`.
- **Alternative considered:** `LocalServerSocket` (Unix domain socket). It
  avoids any port at all, but then the transport would no longer be "REST".
  We stay with loopback HTTP as the user requested.

### 2.4 REST API — 1:1 with `causal_lm_api.h`

Every native entry point is exposed verbatim. `model_id` is a service-side
identifier assigned when a model is loaded; it replaces the native global
state (see §3 on parallelism).

| Method | Path                              | Native call                 | Notes |
|--------|-----------------------------------|-----------------------------|-------|
| POST   | `/v1/options`                     | `setOptions`                | global, affects all future loads |
| POST   | `/v1/models`                      | `loadModelHandle`           | returns `{model_id}` |
| POST   | `/v1/models/{id}/run`             | `runModelHandle`            | returns `{output}` |
| POST   | `/v1/models/{id}/run_stream`      | `runModelHandle` (streaming)| NDJSON chunked stream — see §5.1 |
| GET    | `/v1/models/{id}/metrics`         | `getPerformanceMetricsHandle` | returns `PerformanceMetrics` JSON |
| DELETE | `/v1/models/{id}`                 | `destroyModelHandle`        | unloads / frees |
| GET    | `/v1/models`                      | (service-only)              | lists loaded models |
| POST   | `/v1/connect`                     | (service-only)              | explicit handshake (test) |
| GET    | `/v1/health`                      | (service-only)              | liveness probe |
| POST   | `/v1/models/{id}/chat/open`       | `openChatSession`           | returns `{session_id}` |
| POST   | `/v1/models/{id}/chat/run`        | `chatSession.run`           | structured chat inference |
| POST   | `/v1/models/{id}/chat/run_stream` | `chatSession.runStreaming`  | structured chat NDJSON stream |
| POST   | `/v1/models/{id}/chat/cancel`     | `chatSession.cancel`        | cancel in-flight generation |
| POST   | `/v1/models/{id}/chat/rebuild`    | `chatSession.rebuild`       | replace conversation history |
| POST   | `/v1/models/{id}/chat/close`      | `chatSession.close`         | close session, free resources |

Request / response bodies are JSON (kotlinx.serialization).

Example — load a model:
```http
POST /v1/models HTTP/1.1
Content-Type: application/json

{ "backend": "CPU", "model": "QWEN3_0_6B", "quantization": "W4A32" }
```
```http
200 OK
{ "model_id": "QWEN3_0_6B:W4A32", "architecture": "Qwen3ForCausalLM" }
```

Example — run:
```http
POST /v1/models/QWEN3_0_6B:W4A32/run HTTP/1.1
Content-Type: application/json

{ "prompt": "Hello" }
```
```http
200 OK
{ "output": "...", "error_code": 0 }
```

### 2.5 `RequestDispatcher`
- **File:** `service/RequestDispatcher.kt`
- Parses the path + JSON body, maps it to a `Request` sealed class, and
  forwards to `ModelRegistry`. For requests tied to a model (`run`, `metrics`,
  `destroy`) it looks up the owning `ModelWorker` and enqueues the request.

### 2.6 `ModelRegistry` + `ModelWorker` (Concurrency core)
- **Files:**
  - `service/ModelRegistry.kt`
  - `service/ModelWorker.kt`
- **Design:**
  - `ModelRegistry` maintains `ConcurrentHashMap<String, ModelWorker>` keyed by
    `model_id = "<MODEL>:<QUANT>"` (e.g. `"QWEN3_0_6B:W4A32"`).
  - On `loadModel`, if no worker exists for that key, create a new `ModelWorker`.
    Each worker owns:
    - its own `Thread` (or single-thread executor),
    - a `LinkedBlockingQueue<Job>` (FIFO),
    - a `Backend` instance (native or LiteRT-LM) that owns the model.
  - Requests targeting the same `model_id` land in the same queue and are
    processed strictly FIFO by a single thread.
  - Requests targeting different `model_id`s land in different workers and
    run in parallel — as many parallel workers as loaded models.
- **Backpressure & safety:**
  - Queue size is bounded (e.g. 32); if full, returns HTTP 503.
  - `destroyModelHandle` posts a sentinel job that drains the queue and
    terminates the thread.

### 2.7 Backends

```kotlin
interface Backend {
    fun load(req: LoadModelRequest): LoadModelResponse
    fun run(prompt: String): RunModelResponse
    /**
     * Streaming variant of [run]. The default implementation just calls
     * run() and delivers the whole result as a single delta, so non-
     * streaming backends automatically work through /v1/models/{id}/run_stream.
     * Streaming-capable backends (LiteRtLmBackend) override this to emit
     * progressive deltas via MessageCallback → sink.onDelta(...).
     */
    fun runStreaming(prompt: String, sink: StreamSink): BackendResult<Unit>
    fun metrics(): PerformanceMetrics
    fun close()
}

/** Where backends push streamed output. Single-threaded from the worker. */
interface StreamSink {
    fun onDelta(text: String)
    fun onDone()
    fun onError(error: QuickAiError, message: String?)
}
```

Two implementations:

- **`NativeCausalLmBackend`** (`service/backend/NativeCausalLmBackend.kt`)
  Thin Kotlin wrapper over JNI calls into `libcausallm_api.so`, using the
  new **handle-based** entry points (§3).

- **`LiteRtLmBackend`** (`service/backend/LiteRtLmBackend.kt`)
  Used only when `model == GEMMA4`. Routing happens at the Kotlin level in
  `ModelRegistry`, before a `ModelWorker` is created — so we never touch JNI
  for Gemma4. Delegates to the official LiteRT-LM Kotlin API via the Maven
  artifact `com.google.ai.edge.litertlm:litertlm-android`. On `load` it
  creates an `Engine` + `Conversation`, keeps them open for the ModelWorker's
  lifetime, and closes them in `close()`. Since LiteRT-LM requires an
  explicit `.litertlm` file path, Gemma4 `loadModel` requests **must**
  include a non-empty `model_path` field in the JSON body — otherwise the
  backend returns `INVALID_PARAMETER`. See
  [how-to-use-litert-lm-guide.md](../../how-to-use-litert-lm-guide.md) at
  the repo root for the Kotlin API surface we target.

### 2.8 `JNIBridge`
- **Files:**
  - `LauncherApp/src/main/cpp/quickai_jni.cpp`
  - `LauncherApp/src/main/cpp/CMakeLists.txt`
  - `LauncherApp/src/main/java/com/example/QuickAI/service/NativeCausalLm.kt`
- A thin C++ shim that forwards JNI calls directly to the new handle-based
  `causal_lm_api` functions. No business logic — purely data marshalling
  (`jstring ↔ const char*`, struct → Kotlin data class).
- Loads `libcausallm_api.so` at startup via
  `System.loadLibrary("causallm_api")` (plus its transitive dependencies
  `libnntrainer.so`, `libccapi-nntrainer.so`, `libcausallm_core.so`,
  `libc++_shared.so`). These prebuilt `.so` files are checked into git at
  `Applications/QuickAI/prebuilt_libs/` and a Gradle copy task in
  `LauncherApp/build.gradle.kts` stages them into `build/generated/jniLibs/
  arm64-v8a/` so Android Gradle's standard `jniLibs` pipeline picks them up
  at packaging time. Regenerate them with
  `Applications/CausalLM/build_api_lib.sh`.

### 2.9 `ClientApp`
- **Files:**
  - `clientapp/src/main/java/com/example/clientapp/MainActivity.kt`
  - `clientapp/src/main/java/com/example/clientapp/api/QuickAiClient.kt`
  - `clientapp/src/main/java/com/example/clientapp/api/Models.kt`
- **Transport:** OkHttp + kotlinx.serialization.
- **UX:** Minimal: model spinner, quantization spinner, prompt `EditText`,
  "Run" button, output `TextView`, metrics panel.
- **Routing to the service:** `BASE_URL = "http://127.0.0.1:3453"`.
- ClientApp does **not** link to `libcausallm_api.so`. Everything goes over
  REST. Multiple client apps can be installed side-by-side.

## 3. C API changes — from global singleton to handle-based

### 3.1 Problem
The current `causal_lm_api.cpp` uses a single `g_model` plus a process-wide
`g_mutex`. This prevents loading more than one model simultaneously and
serializes every call, so two clients asking for two **different** models
cannot run in parallel.

### 3.2 Proposed API additions (additive, fully backward-compatible)
New opaque handle type plus handle-scoped variants of every stateful call.
The old global-state functions stay for `test_api` compatibility.

```c
// causal_lm_api.h — additions

/** Opaque handle to a loaded model instance. */
typedef struct CausalLmModel *CausalLmHandle;

/** Load a model and return a handle you must later destroy. */
WIN_EXPORT ErrorCode loadModelHandle(BackendType compute,
                                     ModelType modeltype,
                                     ModelQuantizationType quant_type,
                                     CausalLmHandle *out_handle);

/** Run inference on a specific handle.
 *  outputText is owned by the handle; valid until the next runModelHandle
 *  call on the SAME handle, or until destroyModelHandle. */
WIN_EXPORT ErrorCode runModelHandle(CausalLmHandle handle,
                                    const char *inputTextPrompt,
                                    const char **outputText);

/** Retrieve metrics for a handle's last run. */
WIN_EXPORT ErrorCode getPerformanceMetricsHandle(CausalLmHandle handle,
                                                 PerformanceMetrics *metrics);

/** Release all resources of a handle. */
WIN_EXPORT ErrorCode destroyModelHandle(CausalLmHandle handle);
```

### 3.3 Implementation sketch (`causal_lm_api.cpp`)
- Introduce a `struct CausalLmModel { std::unique_ptr<causallm::Transformer>
  model; std::string architecture; std::string last_output; double init_ms;
  std::mutex mtx; }`.
- `loadModelHandle` becomes a refactored copy of today's `loadModel` body,
  writing into a freshly allocated `CausalLmModel*` instead of the globals.
- `runModelHandle` / `getPerformanceMetricsHandle` operate on the handle's
  own mutex, so different handles never block each other.
- The legacy `loadModel` / `runModel` / `getPerformanceMetrics` now delegate
  to a single static "default" handle for backward compatibility. `test_api`
  keeps working unchanged.

### 3.4 Why Kotlin also needs a per-model thread
Even though the new handle mutex would technically allow concurrent
`runModelHandle` calls on *different* handles from any caller thread,
nntrainer inference is compute-bound and not re-entrant-safe on a single
handle. We therefore keep the 1-thread-per-model discipline on the Kotlin
side — native concurrency is a nice-to-have, Kotlin serialization per-handle
is a must.

## 4. Gemma4 routing

Detection happens at Kotlin level inside `ModelRegistry.getOrCreate(...)`:

```kotlin
val backend = if (req.model == ModelId.GEMMA4) {
    LiteRtLmBackend()       // pure Kotlin path — no JNI
} else {
    NativeCausalLmBackend() // goes through JNI → libcausallm_api.so
}
```

Both implementations satisfy the same `Backend` interface, so `ModelWorker`
is unaware of which path is used. The REST surface is identical.

The `ModelType` enum on the wire is **not** the same as the C enum — the
Kotlin side carries `GEMMA4` as an additional value and only maps native
values to the C enum inside `NativeCausalLmBackend`.

## 5. Concurrency model summary

| Scenario                                            | Behavior |
|-----------------------------------------------------|----------|
| ClientA + ClientB, same model, same quant           | Same `ModelWorker`, FIFO, strictly sequential. |
| ClientA + ClientB, different models                 | Two `ModelWorker`s, two threads, run in parallel. |
| ClientA + ClientB, same model, different quant      | Two workers (id = `model:quant`), parallel. |
| ClientA issues overlapping requests on one model    | Serialized in that model's FIFO. |
| Service receives a request for an unloaded model    | 404 — client must `POST /v1/models` first. |

### 5.1 Streaming responses

The vanilla `POST /v1/models/{id}/run` endpoint buffers the entire generation
on the service side and returns it in one JSON response. That is simple but
wastes the interactivity that LiteRT-LM (and future nntrainer incremental
decoding) already provides — the first token is available in hundreds of
milliseconds, but the client only sees anything after the full generation
finishes.

To fix that we add a second, streaming endpoint:

```
POST /v1/models/{id}/run_stream
Content-Type: application/json

{ "prompt": "Tell me a joke" }
```

The response is **chunked HTTP** with an **NDJSON** body (one JSON object per
line, each terminated by `\n`). Frame shape:

```jsonl
{"type":"delta","text":"Why did"}
{"type":"delta","text":" the chicken"}
{"type":"delta","text":" cross the road?"}
{"type":"done","duration_ms":612}
```

or, on failure:

```jsonl
{"type":"delta","text":"Why did the"}
{"type":"error","error_code":3,"message":"kernel OOM at layer 12"}
```

**Why NDJSON over SSE?** SSE (`text/event-stream`) would work too, but it
wraps every payload in `data: ...\n\n` framing and complicates the server
implementation (NanoHTTPD has no native SSE helper). NDJSON is transport-
neutral, trivially parseable on the client with
`BufferedSource.readUtf8LineStrict()`, and keeps every chunk a self-contained
JSON object — perfect for future metrics / partial-metrics frames.

**Transport path through the existing pipeline**

```
 ChunkedStreamSink.inputStream  (producer/consumer queue)
              ▲
              │ onDelta(text) → encodeFrame → queue.put(bytes)
              │
 LiteRtLmBackend.runStreaming(prompt, sink)
              │
              │  Conversation.sendMessageAsync(prompt, MessageCallback)
              │     onMessage(m)  → defensive delta extraction → sink.onDelta(...)
              │     onDone()     → sink.onDone() + latch.countDown()
              │     onError(t)   → sink.onError(...) + latch.countDown()
              │  latch.await()  ← blocks the worker thread until callback done
              │
              ▼
 ModelWorker.runLoop() — single thread, FIFO discipline preserved
              │
              ▼
 RequestDispatcher.handleRunStream()
              │  Response.Chunked(200, "application/x-ndjson", sink.inputStream)
              │
              ▼
 HttpServer     NanoHTTPD.newChunkedResponse(OK, "application/x-ndjson",
                                             sink.inputStream)
              │
              ▼
 Client        OkHttp BufferedSource.readUtf8LineStrict() loop →
               decode each line as StreamFrame → dispatch to onChunk(chunk)
```

The `ChunkedStreamSink` lives on the backend side. It exposes:

- a `StreamSink` implementation that backends push deltas into, and
- an `InputStream` that NanoHTTPD drains to write the chunked body.

The two are connected by a `LinkedBlockingQueue<ByteArray>` so the backend
worker thread and the NanoHTTPD writer thread never block each other beyond
normal queue backpressure. An empty ByteArray serves as the EOF sentinel that
wakes the reader and closes the response.

**Why NanoHTTPD's `newChunkedResponse` is safe here.** That helper simply
copies from the `InputStream` to the socket in a background thread; it does
not assume `available()` or a finite length, so our queue-backed stream works
as long as `read()` blocks until data or EOF arrives. Once the backend calls
`sink.onDone()` we push the EOF sentinel and the NanoHTTPD writer naturally
terminates the chunked encoding.

**Concurrency discipline is preserved.** The ModelWorker thread is still the
only thread calling into the backend. Inside `runStreaming`, LiteRT-LM may
invoke `MessageCallback` on its own internal thread, but that callback only
touches the `ChunkedStreamSink` (which is itself thread-safe via the blocking
queue) — the worker thread holds a `CountDownLatch` until the callback signals
`onDone`/`onError`, so the worker does not return to the job loop while the
model is still emitting tokens. FIFO ordering across streaming and non-
streaming requests is therefore unchanged.

**Defensive delta extraction.** The LiteRT-LM Kotlin API documents
`sendMessageAsync(...).collect { print(it) }` as already emitting per-token
deltas, but the callback variant's `onMessage(Message)` contract is not
explicit about whether each `Message` is a delta or an accumulated snapshot.
The backend therefore keeps a running `StringBuilder` and emits
`full.substring(accumulated.length)` when the new message starts with the
accumulated buffer, falling back to the raw text otherwise. This copes with
both behaviours without double-printing tokens.

**Client consumption.** `QuickAiClient.runModelStreaming(modelId, req,
onChunk)` opens the response, reads its `BufferedSource` line-by-line with
`readUtf8LineStrict()`, decodes each line into a `StreamFrame`, and dispatches
a sealed `StreamChunk` (`Delta`, `Done`, `Error`) to the caller. In the
sample app the handler just calls `outputView.append(delta)` on the main
thread so tokens appear as they arrive.

**Non-streaming backends get streaming for free.** Because `Backend.runStreaming`
has a default implementation that calls `run(prompt)` and pushes the whole
string as a single `onDelta` followed by `onDone`, the native backend works
with `/run_stream` unchanged — it just emits one big chunk instead of many
small ones.

## 6. Permissions & Manifests

### LauncherApp `AndroidManifest.xml`
```xml
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.FOREGROUND_SERVICE"/>
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_DATA_SYNC"/>
<uses-permission android:name="android.permission.POST_NOTIFICATIONS"/>
```

### clientapp `AndroidManifest.xml`
```xml
<uses-permission android:name="android.permission.INTERNET"/>
```

Cleartext localhost is enabled via `res/xml/network_security_config.xml`
(`<domain includeSubdomains="false">127.0.0.1</domain>` with
`cleartextTrafficPermitted="true"`).

## 7. Build & Packaging

- **Native library:** `libcausallm_api.so` (plus transitive deps
  `libcausallm_core.so`, `libnntrainer.so`, `libccapi-nntrainer.so`,
  `libc++_shared.so`) is produced out-of-tree by
  `Applications/CausalLM/build_api_lib.sh` and the resulting artifacts are
  committed to `Applications/QuickAI/prebuilt_libs/` so consumers do not
  have to rebuild the engine themselves. A Gradle `Copy` task
  (`copyPrebuiltNativeLibs` in `LauncherApp/build.gradle.kts`) stages them
  into `build/generated/jniLibs/arm64-v8a/` which is wired into
  `android.sourceSets.main.jniLibs.srcDirs`, so Android Gradle packages
  them into the APK through the standard `jniLibs` pipeline.
- **JNI shim:** built via Android Gradle's CMake integration from
  `LauncherApp/src/main/cpp/CMakeLists.txt`. Produces `libquickai_jni.so`
  which links against `libcausallm_api.so` from the prebuilt_libs folder
  above.
- **Kotlin deps (LauncherApp):** NanoHTTPD, kotlinx-serialization-json,
  kotlinx-coroutines-android, `com.google.ai.edge.litertlm:litertlm-android`.
- **Kotlin deps (clientapp):** OkHttp, kotlinx-serialization-json,
  kotlinx-coroutines-android.

## 8. File layout (post-implementation)

```
Applications/QuickAI/
├── Architecture.md                        (this file)
├── prebuilt_libs/                         (checked-in .so artifacts)
│   ├── libcausallm_api.so
│   ├── libcausallm_core.so
│   ├── libnntrainer.so
│   ├── libccapi-nntrainer.so
│   └── libc++_shared.so
├── LauncherApp/
│   └── src/main/
│       ├── AndroidManifest.xml            (permissions + :remote service)
│       ├── cpp/
│       │   ├── CMakeLists.txt
│       │   └── quickai_jni.cpp
│       ├── res/xml/network_security_config.xml
│       └── java/com/example/QuickAI/
│           ├── LauncherApp.kt             (boots service)
│           ├── QuickAIService.kt          (foreground + HttpServer)
│           └── service/
│               ├── HttpServer.kt
│               ├── RequestDispatcher.kt
│               ├── ModelRegistry.kt
│               ├── ModelWorker.kt
│               ├── Protocol.kt            (DTOs, enums, errors)
│               ├── NativeCausalLm.kt      (JNI bindings)
│               └── backend/
│                   ├── Backend.kt
│                   ├── NativeCausalLmBackend.kt
│                   └── LiteRtLmBackend.kt
└── clientapp/
    └── src/main/java/com/example/clientapp/
        ├── MainActivity.kt
        └── api/
            ├── QuickAiClient.kt
            └── Models.kt

Applications/CausalLM/api/
├── causal_lm_api.h                        (+ handle-based declarations)
└── causal_lm_api.cpp                      (+ handle-based implementations)
```

## 9. Implementation phases

1. **C API — handle-based variants** (`causal_lm_api.h/.cpp`).
2. **Kotlin protocol + DTOs** (`Protocol.kt`).
3. **JNI shim** (`quickai_jni.cpp`, `CMakeLists.txt`, `NativeCausalLm.kt`).
4. **Backends** (`Backend.kt`, `NativeCausalLmBackend.kt`, `LiteRtLmBackend.kt`).
5. **Concurrency core** (`ModelWorker.kt`, `ModelRegistry.kt`).
6. **REST layer** (`HttpServer.kt`, `RequestDispatcher.kt`).
7. **Service lifecycle** (`QuickAIService.kt`, notification, foreground).
8. **Launcher UI** (`LauncherApp.kt` — status + start/stop button).
9. **Manifests + permissions + network_security_config**.
10. **Gradle wiring** (NanoHTTPD, serialization, externalNativeBuild).
11. **ClientApp REST client + test UI**.
12. **LiteRT-LM artifact integration** via
    `com.google.ai.edge.litertlm:litertlm-android` (done).

## 10. Open items

- **Model assets** (`./models/qwen3-0.6b-w4a32/...`,
  `/sdcard/Download/gemma4.litertlm`, …) must be present on the device. The
  native backend resolves paths relative to its working directory, and the
  LiteRT-LM backend takes the path verbatim from `LoadModelRequest.model_path`.
  A future `GET /v1/models/available` endpoint can advertise which assets
  are installed.
- **LiteRT-LM NPU backend** currently falls back to CPU because it needs a
  `Context` to locate `applicationInfo.nativeLibraryDir`. Plumbing a
  `Context` into `LiteRtLmBackend` via `ModelRegistry` is a small follow-up.
- **Streaming output** — done. `POST /v1/models/{id}/run_stream` now emits
  NDJSON chunks over HTTP chunked transfer encoding, driven by LiteRT-LM's
  `Conversation.sendMessageAsync(prompt, MessageCallback)` on the service
  side. See §5.1 for the full design. Non-streaming backends fall back to a
  single-chunk default implementation in `Backend.runStreaming`.

## 11. Structured Chat Session API

Added in response to aistudio-mobile requirements (request-mail1 + request-mail2).

### 11.1 Overview

The chat session API extends Quick.AI with structured multi-turn conversation
support. Instead of the flat `run(prompt)` API, clients open a **chat session**
on a loaded model, send structured messages with `system`/`user`/`assistant`
roles, and receive structured responses.

Key design decisions:
- **Multiple sessions per model**: `openChatSession()` returns independent
  sessions, each with its own conversation state and image cache.
- **Backend-managed history**: The session accumulates messages internally.
  Clients send only the new messages for each turn.
- **Rebuild support**: `rebuild(messages)` replaces the entire history —
  used after edits, sampling changes, or failed turns.
- **Stable image handling**: An `ImageStore` per session caches images by
  SHA-256 hash. The same image arriving via different temp file paths is
  recognized as identical.
- **Explicit cancellation**: `cancel()` is thread-safe and stops in-flight
  generation.

### 11.2 Session lifecycle

```
loadModel(GEMMA4) → openChatSession(config?) → run/runStreaming(messages)
                                              → cancel()
                                              → rebuild(messages)
                                              → close()
```

### 11.3 REST endpoints

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `/v1/models/{id}/chat/open` | `ChatOpenRequest` | `ChatOpenResponse{session_id}` |
| POST | `/v1/models/{id}/chat/run` | `ChatRunRequest{session_id, messages}` | `ChatRunResponse{content}` |
| POST | `/v1/models/{id}/chat/run_stream` | `ChatRunRequest{session_id, messages}` | NDJSON stream |
| POST | `/v1/models/{id}/chat/cancel` | `ChatSessionIdRequest{session_id}` | `ChatGenericResponse` |
| POST | `/v1/models/{id}/chat/rebuild` | `ChatRebuildRequest{session_id, messages}` | `ChatGenericResponse` |
| POST | `/v1/models/{id}/chat/close` | `ChatSessionIdRequest{session_id}` | `ChatGenericResponse` |

### 11.4 Message wire format

```json
{
  "session_id": "uuid",
  "messages": [
    {
      "role": "user",
      "parts": [
        { "type": "text", "text": "Describe this image" },
        { "type": "image_file", "absolute_path": "/sdcard/photo.jpg" },
        { "type": "image_bytes", "image_base64": "..." }
      ]
    }
  ]
}
```

### 11.5 Configuration

`ChatOpenRequest.config` accepts:
- `sampling`: `{temperature, top_k, top_p, min_p, max_tokens, seed}`
- `chat_template_kwargs`: `{enable_thinking}` — controls thinking-mode prompt

### 11.6 Backend support

| Feature | LiteRTLm (Gemma4) | NativeQuickDotAI (Qwen3) |
|---------|--------------------|--------------------------|
| Chat session | Full implementation | Dummy (UNSUPPORTED) |
| Multimodal turns | With visionBackend | UNSUPPORTED |
| Cancellation | AtomicBoolean flag | No-op |
| Rebuild | New Conversation | UNSUPPORTED |
| enable_thinking | Passed to template | Dummy |
| maxNumTokens | EngineConfig | Ignored |

### 11.7 Files added/modified

- `QuickDotAI/Types.kt` — new chat data classes + `maxNumTokens`
- `QuickDotAI/QuickDotAI.kt` — `QuickAiChatSession` interface + `openChatSession()`
- `QuickDotAI/ImageStore.kt` — SHA-256 image cache (NEW)
- `QuickDotAI/LiteRTLmChatSession.kt` — LiteRT-LM session impl (NEW)
- `QuickDotAI/LiteRTLm.kt` — session management + maxNumTokens
- `QuickDotAI/NativeChatSession.kt` — dummy session (NEW)
- `QuickDotAI/NativeQuickDotAI.kt` — dummy openChatSession
- `LauncherApp/service/Protocol.kt` — chat DTOs
- `LauncherApp/service/HttpServer.kt` — chat route parsing
- `LauncherApp/service/RequestDispatcher.kt` — chat handlers
- `LauncherApp/service/ModelWorker.kt` — chat job types + processing
- `clientapp/api/Models.kt` — client chat DTOs
- `clientapp/api/QuickAiClient.kt` — client chat methods

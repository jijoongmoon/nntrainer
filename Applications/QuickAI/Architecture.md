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
| GET    | `/v1/models/{id}/metrics`         | `getPerformanceMetricsHandle` | returns `PerformanceMetrics` JSON |
| DELETE | `/v1/models/{id}`                 | `destroyModelHandle`        | unloads / frees |
| GET    | `/v1/models`                      | (service-only)              | lists loaded models |
| GET    | `/v1/health`                      | (service-only)              | liveness probe |

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
    fun metrics(): PerformanceMetrics
    fun close()
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
- **Streaming output** (token-by-token SSE / LiteRT-LM
  `sendMessageAsync(...).collect { ... }`) is out of scope for iteration 1.
  The REST endpoints are blocking request/response.

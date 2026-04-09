# QuickAI — Async + Streaming for `libcausallm_api.so`

_Applies to: `Applications/CausalLM/api/causal_lm_api.{h,cpp}`,
`Applications/CausalLM/models/causal_lm.{h,cpp}`,
`Applications/QuickAI/LauncherApp/src/main/cpp/quickai_jni.cpp`,
`Applications/QuickAI/LauncherApp/.../service/NativeCausalLm.kt`,
`Applications/QuickAI/LauncherApp/.../service/backend/NativeCausalLmBackend.kt`._

This document specifies how token-by-token streaming is added to the
native `causal_lm_api` path so that **Qwen3 (and any future
`causallm::CausalLM` subclass) streams through `/v1/models/{id}/run_stream`
just like Gemma4 does through LiteRT-LM**.

It is the native-backend counterpart to Architecture.md §5.1 (which
covers the HTTP / NanoHTTPD / OkHttp end-to-end streaming pipeline that
is already in place for Gemma4 / LiteRT-LM).

## 1. Problem statement

As of commit `aa51039` (before this change):

1. `runModelHandle()` in `causal_lm_api.cpp` is blocking: it calls
   `causallm::Transformer::run(...)` which internally runs the entire
   prefill + generation loop before returning, then exposes the full
   answer via `causal_lm_model->getOutput(0)` and returns one
   `(errorCode, outputText)` tuple to the caller.
2. `causallm::CausalLM::registerOutputs()` is the point where each
   newly decoded token lands (after UTF-8 completion + punctuation
   hold). Currently it only does:
   ```cpp
   output_list[b].append(decoded_str);
   std::cout << decoded_str;   // dev-time log
   ```
   There is no hook for the API layer to intercept those per-token
   strings, so `runModelHandle()` cannot forward them anywhere while
   generation is still in flight.
3. `LauncherApp` Backend’s default `Backend.runStreaming()` in
   `backend/Backend.kt` therefore falls back to:
   > call blocking `run()`, then emit the whole output as one delta.

   Meaning native models appear on the wire as one big NDJSON `delta`
   frame followed by `done` — not streaming at all from the user’s
   point of view.
4. The `streaming.diff` patch file at the repo root is an earlier
   attempt at this, but (a) it modifies the **legacy global-singleton**
   API (`g_model`, `g_mutex`), not the handle-based API that QuickAI
   uses, and (b) its “async” flavor requires the caller to spawn
   producer/consumer threads themselves — the C API itself still
   blocks. That patch is kept only as reference; it is NOT merged.

So the user’s suspicion is correct: `causal_lm_api` does not currently
support async + streaming output.

## 2. Why synchronous C API + Kotlin-side async

QuickAI already has exactly one background thread per loaded model
(`ModelWorker`, Architecture.md §2.6). That thread:

- owns the backend instance,
- pulls `Job.RunStream` jobs from its `LinkedBlockingQueue`,
- calls `backend.runStreaming(prompt, sink)`.

Meanwhile the HTTP side (`HttpServer` + NanoHTTPD chunked response) runs
on a separate writer thread and drains `ChunkedStreamSink.inputStream`.
The two threads are already decoupled.

Given that, the cleanest contract for the C layer is:

> `runModelHandleStreaming()` is **synchronous** (blocks the caller) but
> **emits deltas progressively** through a callback while it runs.

- The caller (ModelWorker thread) blocks exactly as long as generation
  takes, which is fine — that’s its job.
- The sink sees deltas arrive token-by-token, which is forwarded to the
  HTTP chunked response on a different thread.
- No extra threads, no `pthread_create`, no new synchronization
  primitives inside `libcausallm_api.so`.

Adding internal thread ownership to the C API was considered and
rejected: it would duplicate the host-language worker model, force us to
reason about cross-thread JNI callback delivery, and buy nothing the
current structure doesn’t already give us.

## 3. Native-side design

### 3.1 Streamer vtable (`streamer.h`)

A small C-callable base type in `Applications/CausalLM/api/streamer.h`:

```c
typedef struct BaseStreamer BaseStreamer;

typedef struct {
    /* put: forward one UTF-8-decoded delta string to the streamer.
     *      returns 0 to continue generation, non-zero to request
     *      cancellation at the next token boundary. */
    int  (*put)(BaseStreamer *self, const char *decoded_utf8);
    /* end: called exactly once after the last put, regardless of
     *      whether generation finished normally, was cancelled, or
     *      threw. */
    void (*end)(BaseStreamer *self);
} BaseStreamerVTable;

struct BaseStreamer {
    const BaseStreamerVTable *vtable;
};
```

Tiny helpers `streamer_put(BaseStreamer*, const char*)` and
`streamer_end(BaseStreamer*)` keep the call sites in `CausalLM` free of
NULL checks.

This is deliberately a minimal subset of the `streamer.h` from
`streaming.diff`. We drop the `BufferedStreamer` queue type entirely —
QuickAI doesn’t need it because the Kotlin layer already has
`ChunkedStreamSink` playing that role on the host-language side.

### 3.2 Callback streamer (`callback_streamer.h/.cpp`)

```c
typedef int (*CausalLmTokenCallback)(const char *delta, void *user_data);

typedef struct {
    BaseStreamer base;                 /* MUST be first */
    CausalLmTokenCallback callback;
    void *user_data;
    int  cancelled;                    /* sticky flag, set if cb returns non-zero */
} CallbackStreamer;

void callback_streamer_init(CallbackStreamer *s,
                            CausalLmTokenCallback cb,
                            void *user_data);
```

`put()` dispatches to `cb(delta, user_data)` and stores the return value
in `cancelled`. `end()` is a no-op — termination is reported by
`runModelHandleStreaming()` returning, not via the streamer.

### 3.3 `CausalLM` streamer hook (`causal_lm.{h,cpp}`)

Minimum surgery on the existing class:

```cpp
// causal_lm.h  (public)
void setStreamer(BaseStreamer *s);  // nullptr to detach

protected:
    BaseStreamer *streamer_ = nullptr;
    bool stop_requested_ = false;
```

In `causal_lm.cpp`:

- `setStreamer(BaseStreamer *s)` just writes the member.
- `run()` resets `stop_requested_ = false;` at the top.
- `registerOutputs()`: after `output_list[b].append(decoded_str)`, if
  `streamer_ != nullptr` call `streamer_put(streamer_, decoded_str.c_str())`
  and `if (rc != 0) stop_requested_ = true;`.
- In `run()`’s generation loop (the `for (token_generation_idx …)`
  loop, causal_lm.cpp:484), after the existing EOS-detection block,
  add `if (stop_requested_) { free(input_sample); break; }`.
- Right before `run()` returns (inside the bottom of the function,
  after metrics are recorded), `if (streamer_ != nullptr) streamer_end(streamer_);`.

Note: holding `streamer_` as a raw pointer is fine because the only
code path that sets it — `runModelHandleStreaming()` — RAII-detaches it
before returning, and `CausalLM::run()` is serialized per-handle by
`CausalLmModel::mtx`. There is no observable state sharing with the
non-streaming `runModelHandle()`.

Existing callers (`runModelHandle` / `runModel`) are unaffected because
they never call `setStreamer`, so `streamer_ == nullptr` and all the new
branches are no-ops.

### 3.4 New public entry point (`causal_lm_api.h`)

```c
typedef int (*CausalLmTokenCallback)(const char *delta, void *user_data);

WIN_EXPORT ErrorCode runModelHandleStreaming(
    CausalLmHandle handle,
    const char *inputTextPrompt,
    CausalLmTokenCallback callback,
    void *user_data);
```

Semantics:

- Synchronous, blocks the caller for the full generation duration.
- Invokes `callback(delta, user_data)` one or more times with UTF-8
  text. `delta` is owned by the callee — valid only for the duration of
  the call. If the callback needs to retain the string it must copy.
- Returns `CAUSAL_LM_ERROR_NONE` on clean completion (EOS,
  `NUM_TO_GENERATE` reached, or cancelled via callback return value),
  or a specific `ErrorCode` on failure.
- Populates `h.last_output` with everything that was generated
  (including the partial output on cancel), so the subsequent
  `getPerformanceMetricsHandle()` call still returns valid metrics and
  the same handle can be reused for `runModelHandle()` or another
  `runModelHandleStreaming()`.

Implementation (`causal_lm_api.cpp`):

```cpp
ErrorCode runModelHandleStreaming(CausalLmHandle handle,
                                  const char *prompt,
                                  CausalLmTokenCallback cb,
                                  void *user_data) {
    if (handle == nullptr || prompt == nullptr || cb == nullptr)
        return CAUSAL_LM_ERROR_INVALID_PARAMETER;

    auto &h = *handle;
    std::lock_guard<std::mutex> lock(h.mtx);
    if (!h.initialized || !h.model) return CAUSAL_LM_ERROR_NOT_INITIALIZED;

    auto *causal = dynamic_cast<causallm::CausalLM *>(h.model.get());
    if (causal == nullptr) return CAUSAL_LM_ERROR_UNKNOWN;  // streaming only on CausalLM

    CallbackStreamer s;
    callback_streamer_init(&s, cb, user_data);
    causal->setStreamer(reinterpret_cast<BaseStreamer *>(&s));

    // RAII guard so we ALWAYS detach, even on exception.
    struct Detach {
        causallm::CausalLM *c;
        ~Detach() { c->setStreamer(nullptr); }
    } detach{causal};

    try {
        std::string input(prompt);
        if (g_use_chat_template) input = apply_chat_template(h.architecture, input);
        h.model->run(input, false, "", "", g_verbose);
        h.last_output = causal->getOutput(0);
    } catch (const std::exception &e) {
        std::cerr << "Exception in runModelHandleStreaming: " << e.what() << std::endl;
        return CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }
    return CAUSAL_LM_ERROR_NONE;
}
```

### 3.5 Threading model

- `runModelHandleStreaming()` holds `h.mtx` for the full duration. This
  mirrors `run_on_handle()` and guarantees per-handle serialization.
  Other handles are unaffected.
- The callback runs on **the same thread that called
  `runModelHandleStreaming()`**. No cross-thread delivery, no JNI
  `AttachCurrentThread`, no signaling.
- Cancellation is cooperative: a non-zero callback return value sets a
  sticky flag that `CausalLM::run()`’s inner loop checks once per
  generated token, so worst-case latency is one token.
- There is no timeout in the C layer; the host (ModelWorker / HTTP
  writer / OkHttp client) provides timeouts.

### 3.6 Build wiring

- `Applications/CausalLM/jni/Android.mk`: add `../api/streamer.cpp` and
  `../api/callback_streamer.cpp` to `causallm_api`’s `LOCAL_SRC_FILES`.
- `Applications/CausalLM/meson.build`: add
  `api/streamer.cpp` and `api/callback_streamer.cpp` to `causallm_src`.

## 4. JNI bridge

### 4.1 Kotlin listener interface (`NativeCausalLm.kt`)

```kotlin
fun interface NativeStreamListener {
    /** Called once per delta on the worker thread. MUST NOT block. */
    fun onDelta(text: String)
}

external fun runModelHandleStreamingNative(
    handle: Long,
    prompt: String,
    listener: NativeStreamListener
): Int   // ErrorCode; 0 on success
```

We intentionally keep this separate from the `StreamSink` interface in
`backend/Backend.kt`: `StreamSink` has three methods (delta/done/error),
but the JNI trampoline only needs one — `done` and `error` are
synthesized on the Kotlin side from the JNI return value.

### 4.2 C++ side (`quickai_jni.cpp`)

```cpp
struct StreamCtx {
    JNIEnv   *env;
    jobject   listener;
    jmethodID onDelta;
};

static int streamTrampoline(const char *delta, void *user_data) {
    auto *ctx = static_cast<StreamCtx *>(user_data);
    jstring js = ctx->env->NewStringUTF(delta);
    if (js == nullptr) return 1;               // OOM -> cancel
    ctx->env->CallVoidMethod(ctx->listener, ctx->onDelta, js);
    ctx->env->DeleteLocalRef(js);
    if (ctx->env->ExceptionCheck()) {          // Kotlin listener threw
        ctx->env->ExceptionClear();
        return 1;
    }
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_QuickAI_service_NativeCausalLm_runModelHandleStreamingNative(
    JNIEnv *env, jobject, jlong handleJlong, jstring promptJ, jobject listenerObj) {

    auto handle = reinterpret_cast<CausalLmHandle>(handleJlong);

    jclass cls = env->GetObjectClass(listenerObj);
    jmethodID onDelta = env->GetMethodID(cls, "onDelta", "(Ljava/lang/String;)V");
    if (onDelta == nullptr) return CAUSAL_LM_ERROR_INVALID_PARAMETER;

    const char *prompt = env->GetStringUTFChars(promptJ, nullptr);
    if (prompt == nullptr) return CAUSAL_LM_ERROR_INVALID_PARAMETER;

    StreamCtx ctx{env, listenerObj, onDelta};
    ErrorCode ec = runModelHandleStreaming(handle, prompt, &streamTrampoline, &ctx);
    env->ReleaseStringUTFChars(promptJ, prompt);
    return static_cast<jint>(ec);
}
```

Key points:

- Because the callback fires on the same thread that entered JNI, the
  captured `env` pointer is still valid — no `GetJavaVM` /
  `AttachCurrentThread` dance.
- `jmethodID` is resolved once per call (the listener is a `fun
  interface`, so the object class may differ call-to-call; caching
  globally is unsafe).
- Any Kotlin-side exception thrown from `onDelta` short-circuits the
  generation loop via the trampoline’s return code.

## 5. Kotlin-side integration

### 5.1 `NativeCausalLmBackend.runStreaming` override

Replaces the default “single big delta” implementation inherited from
`Backend`:

```kotlin
override fun runStreaming(prompt: String, sink: StreamSink): BackendResult<Unit> {
    if (!loaded || handle == 0L) {
        sink.onError(QuickAiError.NOT_INITIALIZED, null)
        return BackendResult.Err(QuickAiError.NOT_INITIALIZED)
    }
    return try {
        val ec = NativeCausalLm.runModelHandleStreamingNative(handle, prompt) { delta ->
            sink.onDelta(delta)
        }
        if (ec != 0) {
            val err = QuickAiError.fromNativeCode(ec)
            sink.onError(err, "runModelHandleStreaming failed (errorCode=$ec)")
            BackendResult.Err(err)
        } else {
            sink.onDone()
            BackendResult.Ok(Unit)
        }
    } catch (t: Throwable) {
        sink.onError(QuickAiError.INFERENCE_FAILED, t.message)
        BackendResult.Err(QuickAiError.INFERENCE_FAILED, t.message)
    }
}
```

The listener lambda is called on the same ModelWorker thread that owns
the sink. `ModelWorker.Job.RunStream` already wraps the sink in a guard
that enforces exactly-one terminal event — this override delivers
`onDone()` or `onError()` exactly once per happy/sad path.

### 5.2 What is NOT changed

- `StreamSink` interface — unchanged.
- `ChunkedStreamSink` — unchanged.
- `HttpServer` `/v1/models/{id}/run_stream` handler — unchanged.
- `QuickAiClient.runModelStreaming` — unchanged.
- `MainActivity.onRunClicked` — unchanged.

All of the above already work for Gemma4 via LiteRT-LM and will
automatically pick up native Qwen3 streaming now that
`NativeCausalLmBackend.runStreaming` forwards deltas from the C layer.

## 6. Failure and cancellation matrix

| Event                                       | Native behavior                                  | Kotlin sink                                 |
|---------------------------------------------|--------------------------------------------------|---------------------------------------------|
| generation finishes (EOS / NUM_TO_GENERATE) | `runModelHandleStreaming` returns NONE           | `onDone()`                                  |
| C++ exception inside `run()`                | caught → returns `INFERENCE_FAILED`              | `onError(INFERENCE_FAILED, e.what())`       |
| invalid args                                | returns `INVALID_PARAMETER`                      | `onError(INVALID_PARAMETER, …)`             |
| handle not loaded                           | returns `NOT_INITIALIZED`                        | `onError(NOT_INITIALIZED, …)`               |
| Kotlin listener throws                      | trampoline returns 1 → native loop stops at next token; `runModelHandleStreaming` still returns NONE, but the Kotlin override saw the exception via the sink path and the worker’s Job.RunStream guard will already have surfaced `onError` | `onError(INFERENCE_FAILED, …)` |
| worker shutdown mid-generation              | handle is destroyed only after worker.close(); mutex prevents torn state | guarded by `ModelWorker.drainAndFail()`     |

## 7. Cross-references

- Architecture.md §2.6 (worker threading model)
- Architecture.md §5.1 (end-to-end NDJSON streaming pipeline — the
  Gemma4 side of the same mechanism)
- `streaming.diff` at repo root — earlier prototype, superseded by this
  document for the handle-based API. Kept as reference only.

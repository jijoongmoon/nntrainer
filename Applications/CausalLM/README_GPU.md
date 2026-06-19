# GPU (OpenCL) Inference for CausalLM — Build, Run, Tune

This guide covers running the CausalLM stack (`nntr_causallm`) on the GPU via
OpenCL. It applies to **Gemma4-E2B**, **Gemma2-2B**, and **Qwen3-0.6B** (and any
future CausalLM model that opts into the GPU layers), on two backends:

| Backend | Device (verified) | FC / KV path | Selector env |
|---------|-------------------|--------------|--------------|
| **Adreno** | Galaxy S26 Ultra, Adreno 840 (`adb -s R3CY70LV96T`) | `image2d` weights + `image2d` KV (texture cache) | `NNTR_KV_IMG_ATTN=1`, no `NNTR_V8C_BUF` |
| **Intel**  | Arc `0x7d55` (Meteor Lake), x86 `build_cl` | `cl_mem` buffer + dp4a, flash attention | `NNTR_V8C_BUF=1`, no `NNTR_KV_IMG_ATTN` |

The activation/compute precision is **FP16** (`model_tensor_type: "QINT4-FP16"` —
4-bit weights, FP16 activations). FP16 activations are required for the full GPU
residency path (attention / RoPE / KV cache all on GPU).

> The engine is chosen by `causallm_engine()` (`llm_util.hpp`): it defaults to
> **gpu** and only drops to host CPU when `NNTR_ENGINE=cpu`. There is **no**
> `engine` key in `nntr_config.json`. The GPU path is then gated by the env vars
> below.

---

## 1. Quick start

### Intel Arc (x86, `build_cl`)

```bash
# Build (meson + ninja). The ninja target is the *path*, not the bare name.
ninja -C build_cl Applications/CausalLM/nntr_causallm
# (first-time configure, if build_cl is absent:)
#   meson setup build_cl . -Denable-opencl=true -Denable-fp16=true \
#       -Denable-clblast=true -Dwerror=false --buildtype=release

# Run (canonical Intel env). NNTR_V8C_BUF is MANDATORY (NEO can't read_imageui);
# NNTR_GPU_CLMEM_POOL is MANDATORY for coherence; do NOT set NNTR_KV_IMG_ATTN.
NNTR_GPU_SVM_POOL=1 NNTR_V8C_BUF=1 NNTR_MHA_GPU=1 NNTR_FC_GPU=1 \
NNTR_FC_INT8_GPU=1 NNTR_GPU_CLMEM_POOL=1 \
  ./build_cl/Applications/CausalLM/nntr_causallm <MODEL_DIR> ["prompt"]
```

Wrapper scripts: `.claude/scripts/run_gemma4_x86.sh`, `.claude/scripts/run_gemma2_x86.sh`.
x86 links the host's Intel NEO ICD (`/lib/x86_64-linux-gnu/libOpenCL.so.1`) — nothing to push.

### Adreno (Android, ndk-build)

```bash
export ANDROID_NDK=/path/to/android-ndk        # e.g. ~/Android/Sdk/ndk/27.2.12479018

# (a) Build libnntrainer + libccapi (meson). package_android.sh leaves OpenCL OFF
#     by default, so force it on the first time, then ninja install:
./tools/package_android.sh
meson configure builddir -Denable-opencl=true -Denable-clblast=false -Dwerror=false
ninja -C builddir install

# (b) Build the app (ndk-build):
cd Applications/CausalLM/jni
ndk-build NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj \
  APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk \
  causallm_core nntrainer_causallm -j$(nproc)
# (or simply: ANDROID_NDK=... ./Applications/CausalLM/build_android.sh)

# (c) Deploy ALL of these to /data/local/tmp/nntrainer/causallm (see §5):
#     nntrainer_causallm, libcausallm_core.so, libccapi-nntrainer.so,
#     libnntrainer.so, libOpenCL.so, libc++_shared.so

# (d) Run (canonical Adreno env). NNTR_KV_IMG_ATTN selects image2d KV;
#     NNTR_GPU_CLMEM_POOL is MANDATORY for coherence.
adb -s <SERIAL> shell 'cd /data/local/tmp/nntrainer/causallm && \
  LD_LIBRARY_PATH=$PWD NNTR_FC_INT8_GPU=1 NNTR_MHA_GPU=1 NNTR_GPU_SVM_POOL=1 \
  NNTR_KV_IMG_ATTN=1 NNTR_GPU_CLMEM_POOL=1 \
  ./nntrainer_causallm models/<MODEL_DIR> ["prompt"]'
```

---

## 2. Canonical environment sets

These are the minimal, verified-coherent env sets. Everything else in §4 is
optional tuning / diagnostics.

| | Adreno | Intel |
|---|---|---|
| `NNTR_FC_INT8_GPU` | `1` | `1` |
| `NNTR_MHA_GPU` | `1` | `1` |
| `NNTR_GPU_SVM_POOL` | `1` | `1` |
| `NNTR_GPU_CLMEM_POOL` | `1` (mandatory for coherence) | `1` (mandatory for coherence) |
| `NNTR_V8C_BUF` | — (image2d) | `1` (buffer/dp4a — mandatory on NEO) |
| `NNTR_KV_IMG_ATTN` | `1` (image2d KV) | — |
| `NNTR_MHA_GPU_DECODE` | — (image path; decode already on GPU) | `1` to enable flash-decoding (split-KV) |

> `NNTR_FC_GPU` appears in the wrapper scripts but is a **legacy no-op alias** —
> the real FC GPU gate is `NNTR_FC_INT8_GPU`. It is harmless and kept for script
> compatibility.

---

## 3. Measured performance

Prefill measured at **M=1024** (`prompt_1p2k.txt`, 1024-token prompt). Decode TPS
is reported at the corresponding ~1K context (decode throughput decreases as the
KV context grows). Measured 2026-06-19 at HEAD.

### Adreno 840 (S26 Ultra) — image2d, FP16

| Model | dir | prefill (TPS) | decode @~1K (TPS) |
|-------|-----|--------------:|------------------:|
| Gemma4-E2B (QINT4-FP16, untied lm_head int4) | `gemma4_lmint4` | **2401** | 16.1 (≈22.9 short-ctx) |
| Gemma2-2B (QINT4-FP16, Q6_K lm_head) | `gemma2_lg_q6k` | **839** | 13.8 |
| Qwen3-0.6B (QINT4-FP16, Q6_K lm_head) | `qwen3_lg_q6k` | **2116** | 21.9 |

### Intel Arc 0x7d55 (Meteor Lake) — `cl_mem` buffer, FP16

| Model | prefill (TPS) | decode @~1K (TPS) |
|-------|--------------:|------------------:|
| Gemma4-E2B | **1602** | 5.15 (→ ~7.9 with `NNTR_MHA_GPU_DECODE` flash-decode) |
| Gemma2-2B | **686** | 7.75 |
| Qwen3-0.6B | **1939** | 9.29 |

Notes:
- Gemma4 prefill is fast despite being ~2B because ~57% of its layers share KV
  and are skipped during prefill (`skip_prefill`, Gemma4-only — see §6).
- Adreno prefill > Intel because the image2d FC path benefits from the texture
  cache; Adreno decode > Intel because of the image-backed cooperative GEMV.
- All three models produce coherent output (e.g. *"The capital of South Korea is
  **Seoul**."*) on both backends.

---

## 4. Environment variable reference (selected)

Full list lives in the source (`grep -rn std::getenv`). The ones you actually
need are in §2; the table below adds the common tuning/diagnostic knobs.

| Var | Effect | Platform |
|-----|--------|----------|
| `NNTR_ENGINE` | `=cpu` forces host CPU layers; unset/other ⇒ GPU. | both |
| `NNTR_FC_INT8_GPU` | Master gate for the v8c int4/int8 quantized FC GEMM. | both |
| `NNTR_V8C_BUF` | Buffer-path v8c GEMM (cl_mem uint4) vs image2d. The Adreno⇄Intel switch. | Intel |
| `NNTR_KV_IMG_ATTN` | image2d KV mirrors + image KV attention (texture cache). | Adreno |
| `NNTR_MHA_GPU` | GPU multi-head attention instead of host. | both |
| `NNTR_MHA_GPU_DECODE` | Extends GPU attn+RoPE to the M=1 decode step; gates flash-decoding (split-KV). | both (flash-decode effective on Intel) |
| `NNTR_GPU_SVM_POOL` | In-order SVM-resident queue; skips per-layer `clFinish` drains. | both |
| `NNTR_GPU_CLMEM_POOL` | cl_mem activation pool; lets FC consume a producer's device output directly. Mandatory for coherence (with SVM_POOL). | both |
| `NNTR_KV_INT8` | int8 (quantized) KV cache instead of FP16 (Qwen3 entry; needs ENABLE_FP16). | both |
| `NNTR_ROPE_LUT_CAP` | Caps the GPU RoPE cos/sin LUT size (the GPU-rope prefill fix). | both |
| `NNTR_NO_GPU_ROPE` | Forces host RoPE. | both |
| `NNTR_VNORM_GPU` / `NNTR_VNORM_HOST` | Force the gamma-free v_norm / PLE-norm on GPU / host. | both |
| `NNTR_GEMV_COOP` | Cooperative split-K decode GEMV (Intel buffer-path decode lever). | Intel |
| `NNTR_OPENCL_PROFILING` | OpenCL event profiling (clprof) on the lm_head path. | both |
| `NNTR_LAYER_PROFILE` / `NNTR_PERLAYER_PROF` | Per-layer latency profiling. | both |

> q/k/v-norm GPU residency is **not** an env flag — it is wired structurally via
> the `engine` property on the `reshaped_rms_norm` layers and centralized in
> `CausalLM::registerCustomLayers`. Any model that builds its per-head norms with
> `engine=causallm_engine()` gets GPU residency automatically.

---

## 5. Build artifacts & deploy (Adreno)

A working device run needs **all six** of these co-located in
`/data/local/tmp/nntrainer/causallm` (with `LD_LIBRARY_PATH` pointing there):

1. `nntrainer_causallm` — the executable (`chmod 755` on device)
2. `libcausallm_core.so` — CausalLM model/layer core
3. **`libccapi-nntrainer.so`** — the nntrainer C++/CC-API: **holds the Tensor-API
   graph-compile logic**. ⚠️ **The #1 forgotten artifact.**
4. `libnntrainer.so` — nntrainer core + OpenCL GPU symbols (needs `enable-opencl=true`)
5. `libOpenCL.so` — the Adreno OpenCL ICD (from `builddir/opencl/lib/arm64-v8a`)
6. `libc++_shared.so` — NDK C++ runtime

ndk-build links into `obj/local/arm64-v8a/`; the `libs/arm64-v8a/` copy can lag —
prefer pushing from `obj/local/arm64-v8a/` and check timestamps.

---

## 6. `nntr_config.json` keys

| Key | Meaning |
|-----|---------|
| `model_type` | **Must be `"CausalLM"`** for these models, or the runtime throws a model_type mismatch and aborts. |
| `model_tensor_type` | `WEIGHT-ACTIVATION` pair, e.g. `"QINT4-FP16"` (4-bit weights, FP16 activations). The activation half sets compute precision; **FP16 is required for full GPU residency**. |
| `fc_layer_dtype` | Weight dtype for Q/K/V/O + FFN FC layers (e.g. `QINT4`, `Q4_0`). The v8c GPU GEMM expects QINT4/Q4_0. |
| `embedding_dtype` | Token-embedding weight dtype (e.g. `Q6_K`). Also the default for `lmhead_dtype`. |
| `lmhead_dtype` | LM-head weight dtype (e.g. `Q6_K`). Optional; falls back to `embedding_dtype`. Q6_K uses the GPU GEMV decode path. |
| `lmhead_untie` | Bool. Untied LM head weights vs sharing the embedding matrix. Default false. |
| `skip_prefill` | Bool. **Gemma4-only** KV-shared fast path (skips prefill for shared layers). NOT transferable to Qwen3/Gemma2 (no KV sharing ⇒ garbage if enabled). Default false. |
| `tokenizer_file` | Absolute path to `tokenizer.json`. ⚠️ A config pulled from a device has a `/data/local/tmp/...` path — edit it to the local path for x86 runs. |
| `init_seq_len` / `max_seq_len` | Prefill window M / KV-cache time dimension (prompt + generated). |
| `num_to_generate` | Decode tokens to generate after prefill. |
| `sample_input` | Default (chat-templated) prompt used when no CLI prompt is given. |
| `model_file_name` | The nntrainer weight `.bin` inside the model dir; must match the configured quantization. |

---

## 7. Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `allocateAndBindKVCache: cache placeholder dtype mismatch` (abort at layer 0) | **Stale `libccapi-nntrainer.so` on the device.** The KV-placeholder dtype is decided by the Tensor-API graph compile in libccapi (not libnntrainer); an old libccapi fails to preserve the FP16 placeholder ⇒ `kp=FP32 ≠ kc=FP16`. Push the fresh `libccapi-nntrainer.so`. (Verify with `md5sum` device vs `builddir/android_build_result/lib/arm64-v8a/`.) |
| Output collapses to a single repeated token (greedy-collapse) | Missing `NNTR_GPU_CLMEM_POOL=1` (mandatory for coherence on both backends). |
| `model_type mismatch` crash at load | `nntr_config.json` lacks `"model_type":"CausalLM"`. |
| `Failed to open file` (tokenizer) | `tokenizer_file` points at a device path; set the local absolute path for x86. |
| Silent garbage after editing a `.cl` kernel (Android) | ndk-build does **not** re-run meson's `.cl`→`.cpp` codegen. Regenerate kernels (`.claude/regen_cl.py` / `build_lib.sh`) before rebuilding the lib. |
| dlopen/undefined-symbol for `clSVM*` on Android | `libnntrainer.so` was built without OpenCL. Reconfigure `builddir` with `-Denable-opencl=true` and `ninja install`. |

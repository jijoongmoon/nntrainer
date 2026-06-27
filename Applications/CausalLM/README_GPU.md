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

Prefill at **M=1024** (`prompt_1p2k.txt`); decode at the corresponding ~1K
context. Measured **2026-06-25** on the unified branch (`gpu/v8c-unified`),
best-of-3, all coherent. Models are Gemma4-E2B / Gemma2-2B / Qwen3-0.6B, all
QINT4-FP16 (`gemma4_lmint4` / `gemma2_lg_q6k` / `qwen3_lg_q6k`).

### Adreno 840 (S26 Ultra) — image2d, FP16

```
LD_LIBRARY_PATH=$PWD NNTR_FC_INT8_GPU=1 NNTR_MHA_GPU=1 NNTR_GPU_SVM_POOL=1 \
NNTR_KV_IMG_ATTN=1 NNTR_GPU_CLMEM_POOL=1 ./nntrainer_causallm models/<DIR> "$PROMPT"
```

| Model | prefill (TPS) | decode @~1K (TPS) |
|-------|--------------:|------------------:|
| Gemma4-E2B | **2454** | 18.2 |
| Gemma2-2B | **827** | 14.5 |
| Qwen3-0.6B | **2151** | 30.0 |

### Intel Xe3 (Panther Lake iGPU) — `cl_mem` buffer + XMX, FP16

`NNTR_FC_XMX=1` (DPAS prefill GEMM, ~1.7–1.9× over dp4a) + `NNTR_XE3_SYNC=1`
(Xe3 coherence — **mandatory**) on the canonical Intel set:

```
NNTR_GPU_SVM_POOL=1 NNTR_V8C_BUF=1 NNTR_MHA_GPU=1 NNTR_FC_INT8_GPU=1 \
NNTR_GPU_CLMEM_POOL=1 NNTR_XE3_SYNC=1 NNTR_FC_XMX=1 \
  ./build_cl/Applications/CausalLM/nntr_causallm <MODEL_DIR> "$PROMPT"
```

| Model | prefill (TPS) | decode @~1K (TPS) |
|-------|--------------:|------------------:|
| Gemma4-E2B | **2964** | 18.2 |
| Gemma2-2B | **1756** | 13.8 |
| Qwen3-0.6B | **2301** | 37.6 |

### NVIDIA CUDA — RTX 5060 Laptop (Blackwell sm_120), discrete

block-Q attention (incl. the head_dim=128 kernel) → Qwen3 needs **no**
`NNTR_CUDA_GEMM_ATTN`. Integrated Orin (sm_87) instead appends
`NNTR_CUDA_GEMM_ATTN=1 NNTR_CUDA_GRAPH=1 NNTR_CUDA_M2B=1` (see `run_gemma4_fast.sh`).

```
NNTR_ENGINE=cuda NNTR_CUDA_DEV_ACT=1 NNTR_RMSNORM_CUDA_OFF=all \
NNTR_CUDA_ROPE=1 NNTR_CUDA_ATTN=1 NNTR_CUDA_QKNORM=1 NNTR_CUDA_GEGLU=1 \
NNTR_CUDA_ELTWISE=1 NNTR_CUDA_KV_UVM=1 NNTR_CUDA_VCOPY_PREFILL=1 \
NNTR_CUDA_FLASH_DECODE=64 NNTR_CUDA_BLOCKQ=1 NNTR_FC_CUDA_CUBLAS=1 NNTR_CUDA_PREWARM=1 \
  ./build_cuda/Applications/CausalLM/nntr_causallm <MODEL_DIR> "$PROMPT"
```

| Model | prefill (TPS) | decode @~1K (TPS) |
|-------|--------------:|------------------:|
| Gemma4-E2B | **5400** | 35.3 |
| Gemma2-2B | **3151** | 50.7 |
| Qwen3-0.6B | **4511** | 84.2 |

Notes:
- Gemma4 prefill is fast despite ~2B params because ~57% of its layers share KV
  and skip prefill (`skip_prefill`, Gemma4-only — see §6).
- Intel: XMX (`NNTR_FC_XMX`) lifts Xe3 prefill ~1.7–1.9× over dp4a.
- CUDA: block-Q attention beats cuBLAS for every head_dim on RTX; the new d128
  kernel takes Qwen3 prefill 916 (block-Q fall-through) → 4511 (vs 3969 with
  `NNTR_CUDA_GEMM_ATTN`). On integrated Orin (sm_87) cuBLAS still wins, so the
  Orin recipe keeps `NNTR_CUDA_GEMM_ATTN`.
- The cublas int8 K-chunk (sm_87 large-K workaround) is gated to integrated only,
  so discrete RTX runs the full-K FC (gemma4/gemma2 +3–6% vs chunking everywhere).
- All three models produce coherent output (continuing the 1K passage) on all
  backends.

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

---

## 8. How GPU support is implemented

This section is the engineering overview — *what we did to make nntrainer run a
transformer on the GPU*, and why it is structured the way it is. The guiding
principle is **additive**: nothing in the CPU / training path changed. Adding a
GPU backend is a new `Context` + allocator + op-table + a handful of GPU layers
+ a kernel library, all behind `#if ENABLE_OPENCL` (and `#if ENABLE_CUDA` for
NVIDIA), so an `enable-opencl=false`/`enable-cuda=false` build is byte-identical
to before.

### 8.1 An additive backend, in four pieces

`engine=gpu` selects `ClContext` — registered next to `"cpu"`/`"cuda"`/`"npu"`
in `Engine::add_default_object()` (`engine.cpp`), each under its own `#if`.
`parseComputeEngine()` resolves the name; the graph threads the chosen context's
`ContextData` into every layer's `RunContext` (`network_graph.cpp`). A backend is
exactly four things:

1. **Context** — `ClContext : Context, Singleton<ClContext>` (`cl_context.h`), the
   per-engine layer-factory map + kernel cache.
2. **MemAllocator** — `ClSVMAllocator` (`cl_svm_allocator.h`) routes `MemoryPool`
   alloc/free through `clSVMAlloc`, so a tensor on a CL context is **device-resident
   with no copy step**.
3. **ComputeOps op-table** — routes tensor ops to GPU kernels (§8.2).
4. **GPU layer factories** — the `cl_layers` (`FullyConnectedLayerCl`,
   `RMSNormLayerCl`, `SwiGLULayerCl`, `GeGLULayerCl`, `AdditionLayerCL`,
   `Concat/Reshape/TransposeLayerCl`), each registered only if its kernels compile.

### 8.2 The op-table (`ComputeOps`)

`ComputeOps` (`tensor/cpu_backend/compute_ops.h`) is an abstract virtual table:
base bodies throw "not implemented", and every accelerator-only op is paired with
a `supports_*()` predicate that defaults `false`. `ClComputeOps`
(`cl_operations/cl_compute_ops.cpp`) overrides **only** the int4/Q4_0 GEMM/GEMV
virtuals, flipping their `supports_*()` to `true` and forwarding to the CL
kernels; everything else falls through to the CPU base. `Tensor::getOps()` returns
the per-context table, and call sites check `supports_*()` before taking the GPU
path. That is how one `Tensor` implementation serves both backends — a tensor on a
CL context dispatches to GPU kernels, one on a CPU context computes on the host.

### 8.3 The quantized FC GEMM — the core compute (`v8c`, w4a8)

The FC layers dominate cost, so this is where the speedup lives
(`blas_kernels.cpp` + `cl_kernels/int8_int4_gemm_v8c.cl`). The scheme is **w4a8**:
4-bit weights × FP16 activations, with the activation quantized to **int8 on the
GPU** at the FC input.

- **Quantization.** Weights are offset-encoded int4 nibbles (`value+8`) with a
  per-channel scale and a precomputed row-sum; the activation is quantized per row
  to int8 with an **asymmetric** zero-point (asymmetric was necessary — symmetric
  `amax/127` let skewed post-SwiGLU outliers flip token logits). The epilogue
  removes the offset/zero-point terms and scales back to FP16 — **bit-identical
  across every kernel variant**.
- **GEMM vs GEMV split.** `M>4` (prefill) runs a tiled GEMM (`TM=4,TN=8`, ~87% of
  HW peak on Adreno). `M≤4` (decode) collapses to a GEMV, and preferentially a
  **64-wide K-split cooperative GEMV** that restores parallelism for the single
  decode row (which is otherwise fetch/latency-bound). The "4" is the GEMM tile
  height — it is the prefill(GEMM)↔decode(GEMV) boundary.
- **One source, two device paths.** Adreno reads weights/acts as `image2d`
  (`read_imageui`, texture-L1 cache); Intel NEO cannot *compile* integer-coordinate
  `read_imageui`, so the same bytes are loaded as plain buffers (`-DV8C_BUFFER_ONLY`,
  byte-identical math). This is the `NNTR_V8C_BUF` switch.
- **dp4a vs XMX/DPAS.** The portable path uses `dp4a` builtins. On Intel Xe2/Xe3,
  `gemm_xmx_i4` (`int8_int8_gemm_xmx.cl`) is a drop-in for the `M>4` GEMM using the
  systolic `i8_u8` DPAS (~30 TOP/s, `NNTR_FC_XMX`, **prefill-only**). It is
  prefill-only because decode (`M=1`) is a memory-bandwidth-bound GEMV — a
  compute-throughput engine can't speed up a fetch-bound kernel — so the `M>4` gate
  keeps it out of the decode path. Non-XMX devices fail kernel registration and
  fall through to dp4a.
- **lm_head + on-GPU argmax.** The decode lm_head is a Q6_K GEMV
  (`q6_k_sgemv.cl`); greedy sampling stays on-device via a 2-pass `argmax` that
  reads back **4 bytes**, not the full vocab — the precondition for a
  single-submission decode step.

### 8.4 Attention, RoPE, norms — full GPU residency

The per-token transformer path runs on-device through `MHACoreLayer`
(`Applications/CausalLM/layers/mha_core.cpp`) + `attention_kernels.cpp`:

- **GPU attention** (`NNTR_MHA_GPU`): Q·Kᵀ / softmax / ·V on-device. Adreno uses
  per-layer `image2d` KV mirrors (texture cache); Intel uses an SVM-buffer flash
  path.
- **GPU RoPE + LUT-cap fix.** Capping the cos/sin LUT to the actual max timestep
  (not `max_position_embeddings`=131072) shrinks the per-layer re-upload from tens
  of MB to hundreds of KB — this is what made `M≥32` prefill GPU-RoPE coherent
  (~+500 TPS @ M=1024).
- **q/k/v-norm residency** is structural, not a flag: the per-head
  `reshaped_rms_norm` layers carry `engine=gpu` (registered centrally in
  `CausalLM::registerCustomLayers`), so their outputs stay GPU-resident instead of
  bouncing to the host.
- **Flash-decoding (split-KV).** `M=1` decode splits the KV axis into chunks
  (`num_heads × n_chunks` parallel partials + a reduce), recovering parallelism for
  the lone query. Decode is the structurally hard case: `M=1` is bandwidth-bound
  with a per-op dispatch floor and host bounces.

### 8.5 Residency & coherence

- `NNTR_GPU_SVM_POOL` switches the CL queue to **in-order** with SVM-resident
  buffers, so consecutive layers hand off device→device with no host round-trip and
  no per-layer `clFinish`.
- `NNTR_GPU_CLMEM_POOL` stamps `GPU_CLMEM` residency on activation tensors, so an FC
  consumes its producer's device output directly. Both are mandatory for coherence.
- All pool memory flows through one `MemAllocator` (`MemoryPool` no longer embeds
  calloc/SVM macros); the base is host `aligned_alloc`, the GPU contexts install
  `ClSVMAllocator` (SVM = a single host+device pointer).

### 8.6 The CUDA backend — a peer, not a fork

`CudaContext` (`cuda_context.h`) is the direct mirror of `ClContext`, registered as
`"cuda"` alongside `"gpu"`, with NVRTC runtime kernel compilation + an on-disk PTX
cache. `CudaMemAllocator` is the SVM analogue: `cudaMallocManaged` (UVM — one
pointer host- and device-addressable) plus a `device_only` `cudaMalloc` variant for
the activation pool. Its `ComputeOps` is the **CPU table running on UVM**
(host-coherent), with CUDA layers/kernels layered on top; an unported op simply
falls back to a correct CPU computation on the same pointer. Techniques: cuBLAS int8
FC, block-Q attention (incl. a `head_dim=128` kernel so qwen3 needs no
`GEMM_ATTN` on RTX), flash-decode, and CUDA-graph capture/replay for decode.
Integrated (Orin sm_87) vs discrete (RTX) is one truth source —
`cuda::ContextManager::isIntegrated()` — which gates every discrete-VRAM assumption
(device-only act pool, K-chunk, sync, KV mirror).

### 8.7 One source, four devices

The same code drives **Adreno 840 / Intel Xe3 / RTX 5060 / Orin**. Device
differences are runtime knobs, not forks: Adreno image vs Intel buffer
(`NNTR_V8C_BUF`); Xe3's new-ISA in-order SVM coherence regression needs
`NNTR_XE3_SYNC` (a `clFinish` at the producer→consumer boundary); CUDA K-chunk
gated to Orin only. Each is gated so the other three are unaffected.

### 8.8 Where this is going — the multi-HW refactor

The knobs above and the residual `#if` leakage are being folded into a principled
**add-only** model (`nntrainer/docs/ARCHITECTURE_REFACTOR.md`): express backend
differences solely as op-table virtuals + `Context` capability/sync + `MemAllocator`
capability predicates, so a new device becomes "register a `Context`, report its
caps, provide an op-table subset" with **zero** edits to models or core. **Phase 0
has landed**: a read-only `DeviceCaps` probe (`Context::caps()`, log-only) and
`MemAllocator` capability predicates (`isHostAddressable`/`isDeviceVisible`/`isSVM`/
`needsRegister`) replacing the name-string hacks — both byte-identical and
TPS-neutral on all three backends. Next is opening the registry so vendors add a
backend without editing a closed enum, then collapsing the `cl_layers`/`cuda_layers`
forks into neutral layers over a completed op-table.

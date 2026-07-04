# nntrainer runtime environment-variable reference

This documents the `NNTR_*` runtime environment variables read by the inference
runtime (`nntrainer/` + `Applications/CausalLM/`). It is the catalog the code
points to (`nntrainer/cl_context.cpp`).

## How defaults work

Each GPU vendor gets its **validated max-performance flag set applied
automatically** at context init, so a bare run already matches the tuned
baselines — you do **not** need to export anything for a normal run:

```bash
# OpenCL GPU (Intel / Adreno) — auto-detected, profile applied for you:
./nntr_causallm <model> "<prompt>"

# CUDA — select the backend; the CUDA profile is applied for you:
NNTR_ENGINE=cuda ./nntr_causallm <model> "<prompt>"

# Hexagon NPU (ARM build with -Denable-htp):
NNTR_ENGINE=htp ./nntrainer_causallm <model> "<prompt>"
```

Mechanism and override semantics:

- The per-vendor profile is applied with `setenv(NAME, VALUE, /*overwrite=*/0)`.
  **An explicitly-exported env var always wins**, and `NAME=0` still disables a
  flag (for A/B testing). The profile is a *default layer*, not a mandate.
- Applied in `ClContext::initialize()` (OpenCL) and `CudaContext::initialize()`
  (CUDA), gated so it only fires for the **active** engine — an `engine=cuda` run
  on a dual-backend build does **not** get the OpenCL flags, and vice-versa.
- This is the seam a future `nntr_config.json` profile will plug into.

### What each HW gets for free (the auto-profile)

| Flag | Intel (0x8086) | Adreno (0x5143) | CUDA (discrete) | CUDA (integrated) |
|---|:---:|:---:|:---:|:---:|
| `NNTR_FC_XMX` | ✅ (if subgroups) | — | — | — |
| `NNTR_MHA_GPU` | ✅ | ✅ | — (uses `NNTR_CUDA_ATTN`) | — |
| `NNTR_KV_IMG_ATTN` | — | ✅ | — | — |
| `NNTR_GPU_CLMEM_POOL` | ✅ | ✅ | — | — |
| `NNTR_CUDA_{ROPE,ATTN,KV_UVM,GEGLU,ELTWISE,QKNORM}` | — | — | ✅ | ✅ |
| `NNTR_CUDA_FLASH_DECODE=64`, `NNTR_CUDA_BLOCKQ`, `NNTR_FC_CUDA_CUBLAS`, `NNTR_CUDA_PREWARM` | — | — | ✅ | ✅ |
| `NNTR_CUDA_{DEV_ACT,VCOPY_PREFILL,M2B,ASYNC}`, `NNTR_RMSNORM_CUDA_OFF=all` | — | — | ✅ | — |

Already HW-defaulted independently of the profile (do not need setting):
`NNTR_GPU_SVM_POOL` (default-on for GPU graphs), `NNTR_FC_INT8_GPU` (default-on on
the GPU FC path), `NNTR_XE3_SYNC` (caps-derived: Intel coarse-grain SVM),
`NNTR_V8C_BUF` (caps-derived: Intel NEO buffer path), `NNTR_CUDA_UVM_POOL`
(default-on when a CUDA node exists), `NNTR_CUDA_GEMM_ATTN` (caps-derived:
integrated/Orin).

### Legend

- **DEFAULT-ON** — on unless set to `0`.
- **DERIVED** — value comes from device caps/vendor; env overrides.
- **PROFILE** — opt-in in code, but auto-set for the matching HW by the profile above (`=0` disables).
- **OPT-IN** — off unless set.
- **VALUE** — takes a number / string / path argument.

---

## Engine selection

| Var | Meaning | Default |
|---|---|---|
| `NNTR_ENGINE` | Backend: unset/`gpu` = OpenCL, `cuda` = CUDA, `htp` = Hexagon NPU, `cpu` = host | unset (OpenCL if built, else CPU) |
| `NNTR_NUM_THREADS` | CPU compute worker threads | DERIVED (`hardware_concurrency()/2`) |

---

## GPU / OpenCL — common

| Var | Meaning | Default |
|---|---|---|
| `NNTR_FC_INT8_GPU` | v8c int4/int8 quantized FC GEMM on GPU | DEFAULT-ON (`=0` disables) |
| `NNTR_GPU_SVM_POOL` | In-order SVM-resident command queue (skips per-layer clFinish) | DEFAULT-ON for GPU graphs |
| `NNTR_GPU_CLMEM_POOL` | Device `cl_mem` activation residency pool | PROFILE (Intel+Adreno) |
| `NNTR_MHA_GPU` | GPU attention (Q·Kᵀ / softmax / ·V on device) | PROFILE (Intel+Adreno) |
| `NNTR_MHA_GPU_DECODE` | Extend GPU attention (+GPU-RoPE + flash-decode) to the M=1 decode step | OPT-IN |
| `NNTR_MIN_PREFILL` | Min prefill length to engage GPU MHA (⚠ do **not** set for qwen3) | VALUE |
| `NNTR_GEMV_COOP` | 64-wide K-split cooperative decode GEMV | DEFAULT-ON |
| `NNTR_LMHEAD_GPU` | Force lm_head GEMV onto GPU | VALUE (0/1) |
| `NNTR_GPU_ARGMAX` | On-GPU argmax over logits | OPT-IN |
| `NNTR_NO_FASTMATH` | Disable `-cl-fast-relaxed-math` in program build | OPT-IN |
| `NNTR_SVM_FINE` | Force fine-grain SVM buffers | OPT-IN |

### KV cache / flash-decode (OpenCL)

| Var | Meaning | Default |
|---|---|---|
| `NNTR_KV_INT8` | int8-quantized KV cache instead of FP16 | OPT-IN |
| `NNTR_KV_INT8_GPU` | GPU int8-KV attention path | OPT-IN |
| `NNTR_FLASH` / `NNTR_FLASH_*` | Flash-attention path + tiling knobs (`_BLOCKQ`, `_VEC`, `_COOP`, `_SG`, `_DEC_CHUNK`, …) | VALUE |
| `NNTR_QK_LWS` / `NNTR_SV_LWS` | Q·K / softmax·V local work-group sizes | VALUE |

### Residency / fusion (OpenCL, advanced tuning — mostly opt-in)

`NNTR_DEVRES`, `NNTR_RESIDENT_{ACT,FC,RMSNORM}`, `NNTR_FUSE_{ACT,ADDNORM,NORMQUANT,GEGLUQUANT}`,
`NNTR_FUSED_RMSQ`, `NNTR_FUSED_QKV_GPU`, `NNTR_RMSNORM_GPU`, `NNTR_RMSQ_GPU_*`,
`NNTR_RESIDUAL_PUBLISH`, `NNTR_*_DRAIN` — device-residency / kernel-fusion levers
used for GPU bring-up and A/B. Off by default; see the source for scope. The
`NNTR_CLMEM_*` family (`_RAISE`, `_LOWER`, `_CLASS_FILTER`, …) tunes cl_mem
residency tiers.

---

## Intel-specific (Xe / Meteor Lake / Panther Lake, NEO driver)

| Var | Meaning | Default |
|---|---|---|
| `NNTR_FC_XMX` | DPAS/XMX GEMM path (Xe2/Xe3 prefill +70~151%) | DERIVED (`cl_intel_subgroups`) + PROFILE |
| `NNTR_XMX_NT` / `NNTR_XMX_SGM` | XMX N-tile / subgroup-M tuning | VALUE |
| `NNTR_V8C_BUF` | Buffer-path v8c (cl_mem uint4/dp4a) vs image2d — mandatory on Intel NEO (can't `read_imageui`) | DERIVED (`caps_.image_v8c`) |
| `NNTR_XE3_SYNC` | Per-dispatch clFinish producer→consumer for Xe3 coarse-grain SVM coherence — mandatory on Panther Lake | DERIVED (Intel + no fine-grain SVM) |
| `NNTR_XE3_FC_SYNC` | Same coherence drain scoped to the FC dispatch | OPT-IN |
| `NNTR_V8C_{MFAST,LWS,PREFETCH,DIRECT_OUT}` | v8c GEMM tuning knobs | VALUE |

⚠️ On Panther Lake / Xe3, `NNTR_XE3_SYNC` is mandatory (auto-derived now); without
it the GPU output races → garbage. Do **not** set `NNTR_MIN_PREFILL` for qwen3.

---

## Adreno-specific (Qualcomm, image2d texture path)

| Var | Meaning | Default |
|---|---|---|
| `NNTR_KV_IMG_ATTN` | image2d KV mirrors + image KV attention (`read_imageui` texture cache) | PROFILE (Adreno) |
| `NNTR_QCOM_PERF_HINT` | Set `cl_qcom_perf_hint` (high performance) on the context | VALUE |
| `NNTR_KV_OHWI` / `NNTR_KV_OHWI_GPU_FORCE` / `NNTR_OHWI_*` | OHWI image layout for KV / RoPE | OPT-IN / VALUE |

Note: the image paths (`read_imageui`) build only on Adreno; they are absent on
Intel NEO.

---

## CUDA (`NNTR_CUDA_*`) — requires `NNTR_ENGINE=cuda`

The whole `NNTR_CUDA_*` op set below is auto-applied by the CUDA profile (see the
table at the top); listed here for override/reference.

| Var | Meaning | Default |
|---|---|---|
| `NNTR_CUDA_UVM_POOL` | Tensor pool via `cudaMallocManaged` (UVM) | DEFAULT-ON (cuda node present) |
| `NNTR_CUDA_ROPE` / `_ATTN` / `_QKNORM` / `_GEGLU` / `_ELTWISE` | RoPE / attention / q-k-norm / GeGLU / eltwise on CUDA | PROFILE |
| `NNTR_CUDA_KV_UVM` | KV cache residency in UVM | PROFILE |
| `NNTR_FC_CUDA_CUBLAS` | cuBLAS IMMA int8 prefill FC (M≥32) | PROFILE (VALUE) |
| `NNTR_CUDA_FLASH_DECODE` | Split-KV flash-decode for M=1 (value = split count, e.g. 64) | PROFILE (VALUE) |
| `NNTR_CUDA_BLOCKQ` | Block-Q warp-shuffle prefill attention | PROFILE |
| `NNTR_CUDA_PREWARM` | Load-time repack + scratch prewarm | PROFILE |
| `NNTR_CUDA_GEMM_ATTN` | GEMM-based attention (⚠ qwen3 d128 garbage — leave off) | DERIVED (auto-on integrated/Orin) |
| `NNTR_CUDA_DEV_ACT` | Device-only activation pool (discrete only) | PROFILE (discrete) |
| `NNTR_CUDA_VCOPY_PREFILL` | Copy V into the live KV slot during prefill | PROFILE (discrete) |
| `NNTR_CUDA_M2B` | memcpy-to-buffer decode CUDA-graph path | PROFILE (discrete) |
| `NNTR_CUDA_ASYNC` | Async (non-sync) stream execution | PROFILE (discrete) |
| `NNTR_RMSNORM_CUDA_OFF` | Disable CUDA RMSNorm (`all` or per-scope; host RMSNorm wins on discrete) | PROFILE (discrete) |
| `NNTR_CUDA_GRAPH` / `_PREFILL_GRAPH` | CUDA-graph decode / prefill capture-replay | OPT-IN |
| `NNTR_CUDA_DEVICE` | Select CUDA device ordinal | VALUE |
| `NNTR_CUDA_CACHE` | Override kernel/PTX cache dir (else `$HOME`) | VALUE (path) |
| `NNTR_FC_CUDA_QINT4` / `_DP4A`, `NNTR_CUBLAS_{WS_MB,KCHUNK,ALGO}` | CUDA FC kernel selection + cuBLAS tuning | VALUE |

⚠️ Do **not** set `NNTR_CUDA_GEMM_ATTN` for qwen3 (d128 garbage); the default
block-Q path is correct and faster.

---

## Hexagon NPU / HexKL (ARM build with `-Denable-htp`)

| Var | Meaning | Default |
|---|---|---|
| `NNTR_ENGINE=htp` | Select the HexKL/HTP NPU backend (stamps `engine=htp` on layers) | — |
| `NNTR_HTP_FC_MIN_M` | Min M to route the QS4CX FC to the HMX NPU; smaller M stays on CPU KleidiAI | VALUE (default 8) |

`ENABLE_HEXKL`/`-Denable-htp` is only the ARM/libsdkl **availability** gate; a run
**uses** the NPU only when `NNTR_ENGINE=htp` selects it at runtime.

---

## QNN (Qualcomm NPU)

`NNTR_QNN_DUMP` (dump graph/tensors), `QUICK_DOT_AI_BASE_DIR` /
`QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH` (asset/config paths).

---

## CPU / model / misc

| Var | Meaning | Default |
|---|---|---|
| `NNTR_NUM_THREADS` | Compute worker thread count | DERIVED |
| `NNTRAINER_PATH` | Extra search path for dynamically-loaded layer/optimizer plugins | VALUE (path) |
| `NNTR_QINT4_PLAIN` / `NNTR_QINT4_RANGE15` | Plain (non-packed) QINT4 layout / 15-level range | VALUE |

---

## Diagnostics / profiling / correctness-check (opt-in, debug only)

Not needed for normal runs. Families:

- **Profiling / timing** — `NNTR_OPENCL_PROFILING`, `NNTR_LAYER_PROF*`,
  `NNTR_*_TPROF` (FC/lmhead/norm/attn/kv-stage/rope), `NNTR_MHA_PROFILE`,
  `NNTR_V8C_PROFILE`, `NNTR_STAGE_PROFILE`, `NNTR_HOST_TIMING`, `NNTR_PEAK_BENCH`.
- **Numeric verification** — `NNTR_{MHA,ATTN,ROPE,RMSN,GEGLU,RESID}_VERIFY`,
  `NNTR_V8C_{GEMM,QUANT}_CHECK`, `NNTR_FUSED_RMSQ_CHECK`, `*_TRIP` one-shot logs.
- **cl_mem residency probes** — `NNTR_CLMEM_{PROBE,RESIDENCY_DUMP,CLASS_FILTER,…}`.
- **Record/replay** — `NNTR_RECQ`, `NNTR_RECQ_{TRACE,REPLAY,DESVM,KVSCALAR}`.
- **Dumps / debug logs** — `NNTR_DUMP_{LAYERS,FINAL,STATS,LOGITS}`, `NNTR_POOL_DUMP`,
  `NNTR_{FC,IGEMM,CUDA,UVM}_DBG`, `NNTR_CUDA_GRAPH_DBG`.

> Full enumeration (~210 vars) lives in the source; grep `getenv("NNTR_` under
> `nntrainer/` and `Applications/CausalLM/`. Load-bearing default classifications
> are verified in `nntrainer/cl_context.cpp` and `nntrainer/cuda_context.cpp`.


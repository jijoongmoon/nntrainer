# GPU Generalization Plan — gpu_native techniques onto the mainline layer-graph

Companion to [TENSOR_VIRTUALIZATION_PLAN.md](./TENSOR_VIRTUALIZATION_PLAN.md).
Paper reference: ML Drift (arXiv:2505.00232) §3.1–3.8.

## 0. Purpose & scope

A from-scratch OpenCL inference runtime under `Applications/CausalLM/gpu_native/`
(`qwen3_forward.cpp`, ~4.6k lines) is heavily perf-tuned for Qwen3 / Gemma2 on
Adreno (Android) and Intel Arc (x86). This plan generalizes its optimizations
onto nntrainer's **mainline layer-graph CL path** (`nntrainer/layers/cl_layers`,
`nntrainer/tensor/cl_operations`, `nntrainer/opencl`, `nntrainer/graph`,
`nntrainer/models`) so that **other models get the same fast GPU path
automatically** through the existing builder-override / Factory mechanism —
instead of one hand-written single-model runtime.

### Three hard constraints (all must hold at every step)

1. **GPU generalization** — a new dense-decoder model = subclass + builder
   overrides + factory line, with **zero per-model GPU/kernel work**.
2. **Performance non-regression** — every step must match or beat the current
   gpu_native numbers; no redundant `clFinish`, once-on-GPU-stays-on-GPU.
3. **CPU preservation** — `engine=cpu` execution and the OpenCL-disabled build
   (`enable-opencl=false`) must remain byte-identical and must keep building.

This plan was produced by five adversarially-verified multi-agent analyses
(workflow runs `wf_9c21b18f`, `wf_975f8001`, `wf_b397ef99`, `wf_2b87f893`,
`wf_a9742d5d`). All file:line anchors below were code-verified.

---

## 1. Key architectural facts (established)

- **Kernels are already single-source and shared.** v8c int8/int4 dp4a FC,
  image2d 3-kernel attention, flash Block-Q, rmsnorm, RoPE all live in
  `nntrainer/tensor/cl_operations/cl_kernels/` and compile into `libnntrainer`.
  gpu_native does **not** inline them — it `#include`s the library
  (`nntrainer::gemm_int8_v8c_cl`, `two_conv_attention_*`, …). The mainline
  `FullyConnectedLayerCl::forwarding` calls the same `dotCl_v8c`
  (`fc_layer_cl.cpp:174`).
- **gpu_native borrows the mainline OpenCL infra** — no own context/queue:
  `Engine::Global().getRegisteredContext("gpu")` (`qwen3_forward.cpp:823`).
- gpu_native's speed is **orchestration**, not better math: (a) no per-layer
  host round-trips (SVM residency), (b) kernel fusion, (c) attention living in
  image2d, (d) no per-op dispatch idle.
- **RunLayerContext is the single funnel.** `NeuralNetwork` →
  `forwarding(RunLayerContext&)` (`neuralnet.cpp:374`, incremental `:463`);
  RunLayerContext holds `weights/inputs/outputs/tensors` as `Weight*`/`Var_Grad*`
  (`layer_context.h:1010-1013`); accessors return references to pool-backed
  tensors (`layer_context.cpp:161-357`). Layers may not allocate
  (`layer_devel.h:189-192`).
- **Pool memory IS the RunContext tensor storage.** `MemoryPool::allocate`
  (single buffer, `memory_pool.cpp:118`) → `TensorPool::allocate`
  `setData(getMemory(token))` (`tensor_pool.cpp:233`) → `getMemory` tags SVM by
  allocator name `== "gpu-svm"` (`memory_pool.cpp:204`, `cl_svm_allocator.h:72`)
  → `Var_Grad::var` no-op wrapper (`var_grad.h:123`). **If the pool uses
  `ClSVMAllocator`, every RunContext activation/weight is GPU-resident with no
  copy.** The only missing switch is `engine_name` at `neuralnet.cpp:191`
  (graph-wide; per-tensor `t_engine` exists in specs but is *not consumed* —
  `manager.cpp:532`).
- **CPU/GPU layer objects are fully isolated.** `engine=cpu`→`AppContext`,
  `engine=gpu`→`ClContext` (`engine.cpp:50,52-56`, the latter inside
  `#if ENABLE_OPENCL`). Different classes/factories. CL-layer body edits live in
  OpenCL-gated TUs and cannot affect CPU.

---

## 2. Locked design decisions (2026-06-07)

| # | Decision | Choice |
|---|---|---|
| 1 | **Residency scope** | **graph-wide opt-in** via `engine_name` (not per-tensor). Pool→SVM; pure CPU graphs forced `"cpu"`. SVM is the base; ForwardScratch (gpu_native style) is **not** carried over. |
| 2 | **Adreno KV backing** | **cl_mem exception approved.** image attention K/V only = layer-owned `cl_mem` outside the planner (keeps 934 TPS). The single documented exception to the SVM-pool residency rule. Intel uses SVM-flash (zero-bridge). |
| 3 | **CPU transformer support** | **promote app-level CPU layers to core.** Register `Applications/CausalLM/layers/` CPU `RMSNormLayer`/`SwiGLULayer` in `app_context`; every new `ClContext` type also gets an `AppContext` CPU factory. One model definition runs on both backends by `engine` only. |
| 4 | **Decode performance** | **separate workstream (Step 8).** prefill generalization (Steps 1–7) first; decode already runs *correct* via `incremental_forwarding` (PicoGPT proves it). Step 1 executor pre-installs a capture-mode hook. |
| 5 | **Fusion breadth** | **#6 geglu+quant and #7 add+rmsnorm only**, as property-parametrized fused layers; generic fused infra deferred. GPU-gated realizer. |

**Derived:** pool owns *both* weights and activations; the host-ptr `TensorBacking`
bridge retires in favor of pool-token / `isSVM()` keying; the fast-path scratch
(`V8cScratch`, `v8c_weight_cache`) is **absorbed** into pool views, not run in
parallel.

---

## 3. The residency mechanism (how "once GPU, stay GPU" works)

Two halves:

1. **Allocator** — opt the graph-wide pool into `ClSVMAllocator` (§2.1). Then
   `getInput/getOutput/getWeight` hand the layer a tensor whose `MemoryData` is
   SVM. Producer node N writes; consumer node N+1 reads the *same* buffer
   (graph already view-shares: `network_graph.cpp:786-792` →
   `manager.cpp:640` → `tensor_pool.cpp:202`).
2. **Layer bodies stop calling `getData()`** — `getData()` triggers
   `validate()` = a host-bounce (`float_tensor.cpp:96`). Instead bind device
   memory: `getMemoryData()->getAddr()` + `clSetKernelArgSVMPointer` (SVM), or
   `getBacking()` for the image path. Branch only on
   `getMemoryData()->isSVM()`, never `getBacking()` (which is null after copies —
   `tensor.cpp` copy ctor does not propagate `gpu_backing_`).

The planner reuses offsets across layers (`optimized_v1_planner.cpp:171`); this
*preserves* residency (buffers are reused on-GPU) but requires a **linear SVM
layout** and execution-order dependency tracking (no `clFinish` between ops).

### image2d ↔ SVM bridge policy

`clCreateImage` needs a `cl_mem`; an SVM `void*` cannot back an image (no alias
API exists; gpu_native itself bridges via the `copy_svm_to_clmem_fp16` kernel,
`qwen3_forward.cpp:165-174`). Therefore:

- **Zero-bridge (SVM-direct):** element-wise, quant, FC (act-image sits on the
  FC-owned quant *output* cl_mem, not on the SVM activation), flash attention,
  buffer-load (`NNTR_V8C_BUF`, Intel).
- **One intra-GPU copy (unavoidable):** Adreno KV image only — SVM K/V cache →
  layer-owned cl_mem image mirror, fused into the existing per-token
  `k_scatter_ohwi` (this is the `bb893a6d` +2–4% path). **No host round-trip.**
- `clEnqueueSVMMemcpy` is not loaded → the bridge must be a kernel
  (`gpu_copy_svm_to_clmem`), never a runtime memcpy.

---

## 4. CPU-preservation invariants (apply to every step)

1. **`engine_name` is never unconditionally `"gpu"`.** Default `"cpu"`; resolve
   `"gpu"` only when `ENABLE_OPENCL` AND the `"gpu"` context is registered AND
   the model requests GPU. (Unconditional `"gpu"` makes
   `getRegisteredContext("gpu")` throw on OpenCL-off builds — `engine.h:133-135`
   — killing all CPU inference/training.)
2. **`MemoryData` carries no raw `cl_mem`.** It is a public header pulled into
   nearly every CPU TU (`memory_data.h`). Use the existing `void*` + `isSVM()`,
   or an incomplete `tv::TensorBacking*` forward-decl (`tensor.h:40-42,2146`).
   Never put `cl_mem` / `<CL/cl.h>` in shared headers.
3. **Fusion realizers are per-node GPU-gated.** The realizer loop runs
   unconditionally (`neuralnet.cpp:169-179`); a fusion realizer must no-op when a
   node's `compute_engine != GPU`, so CPU graphs are emitted identically (assert
   with a graph-equivalence unit test).
4. **Every `ClContext`-registered layer type has an `AppContext` CPU factory**
   (decision 3) so no model becomes GPU-only.
5. **No shared file (`layers/` non-cl, `models/`, `graph/`, `tensor/` non-cl)
   includes `cl_operations/`, `cl_layers/`, or `opencl/` without an
   `#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1` guard.**
6. **CI must add an `enable-opencl=false` build job** plus a CPU byte-identical
   golden test and the graph-equivalence test.

---

## 5. Roadmap

Per-step gate: **token-identical** + **perf non-regression A/B** +
**`enable-opencl=false` build passes** + **CPU byte-identical**.

| Step | Work | Primary files | Risk |
|---|---|---|---|
| **0 ✅** | Build-regression fix (done, commit `c26c8d7a`) + baseline freeze (§6) | `layers/addition_layer.cpp` | low |
| **1** | **SVM opt-in** + executor dispatch policy: `engine_name` gate (CPU-safe, inv. #1); executor enqueues to one in-order queue, no per-op sync, drain once per sequence, capture-mode hook; layers stop calling `getData()` | `models/neuralnet.cpp:191`, `graph/network_graph.cpp:398-428` | med |
| **2** | RMSNorm CL SVM-direct (+ subgroup-reduce); `isSVM()` gate | `layers/cl_layers/rmsnorm_layer_cl.cpp` | low |
| **3** | SwiGLU + Addition SVM-direct; **GPU residual add** (remove the CPU residual loop `addition_layer_cl.cpp:93-98`); remove unconditional `getData()` | `layers/cl_layers/{swiglu_cl,addition_layer_cl}.cpp` | low |
| **4** | FC onto SVM pool: branch "SVM pool vs cl_mem backing" (drop `.buffer()` assumption), SVM-direct quant, remove readback `blas_kernel_interface.cpp:1246` | `tensor/cl_operations/blas_kernel_interface.cpp` | med-high |
| **5** | Attention: generalize the SVM↔cl_mem bridge primitive; register MHA in `ClContext`; **resolve the numerical drift gate** `attention_kernels.cpp:500-522` (a *second* blocker beyond registration); device-cap KV residency (Intel SVM-flash / Adreno cl_mem-image, decision 2); RoPE/sliding-window as separate dispatch | `cl_context.cpp:145`, `Applications/CausalLM/layers/mha_core.cpp`, `attention_kernels.cpp` | high |
| **6** | lm_head GPU (wire the dead `sgemv_q6_k_cl`, `blas_kernels.cpp:811`; dtype branch Q6_K vs int4) + embedding residency; keep the one end-of-token logits readback on host | `blas_kernels.cpp`, embedding/lm_head layers | med |
| **7** | Fusion realizer (BnRealizer template `compiler/bn_realizer.cpp:27-75`, inserted after ActivationRealizer at `neuralnet.cpp:176`); `FusedAddRmsNormLayerCl` (#7) + `FusedGegluQuantFcLayerCl` (#6), property-parametrized, GPU-gated; **complete the dormant caller wiring** (`rms_norm.cpp:87`, `rms_norm_gpu.cpp:173` call `fused_rmsnorm_quant_resident_fp32` but discard output) — not deletion; gate on intermediate-size > L2 | `compiler/`, new fused CL layers | high |
| **8** | **Decode performance (separate)**: M=1 GEMV + M=1 attention kernels (absent in both stacks), GPU lm_head+argmax, graph capture/replay (recordable queues are *unimplemented* — verify-first) | new kernels, graph loop | highest |
| **9** | `.bin` self-describing header (tensor manifest) — **a prerequisite of the multi-model goal, not polish** (positional loader `qwen3_forward.cpp:2634` breaks "new model = zero GPU work") | loader, format | high |

**Ordering rationale:** residency (1–4) must precede fusion (7); pool↔GPU
wiring (1) precedes layer residency (2–4); the fast-path pool absorption (4)
must precede ever defaulting the pool on (else flag-flip routes to the slow
`gemm_int4_async_cl` SVM path, `blas_kernels.cpp:620-625`).

---

## 6. Frozen baseline (regression reference)

Step 0 freezes the **gpu_native** numbers as the oracle and perf target. The
Step 0 commit (`c26c8d7a`) touched only `addition_layer.cpp` (mainline, non-GPU),
so gpu_native perf is unchanged from the last verified measurement on this HEAD
lineage (`aa3d7530` / `bb893a6d`):

| Platform | Model | prefill (TPS) | first/decode token marker |
|---|---|---|---|
| Adreno 840 (S26, adb `R3CY70LV96T`) | Qwen3-0.6B QINT4 | ~838 (M=1024) | token 838 |
| Adreno 840 | Gemma2-2B QINT4 | ~933 (M=1024) | token 185 |
| Intel Arc [0x7d55] (Meteor Lake) | Qwen3-0.6B | ~2361 / 6712 | token 6712 |
| Intel Arc | Gemma2-2B | ~874 (M=1024) | token 476 |

> The first **real** A/B is at **Step 5** (when the layer-graph can run a model
> end-to-end on GPU). Steps 1–4 are mainline-layer changes with no end-to-end
> target yet. A fresh clean re-measurement (§7) should be run via the harness
> immediately before Step 5.

## 7. Measurement protocol (anti-contamination — mandatory)

Lessons are codified from prior contaminated sessions (see project memory):

- **Always target the device explicitly:** `adb -s R3CY70LV96T …`. Two devices
  are connected; `R3CN80CW3FY` (Note20/Adreno 650) is the wrong one and has
  caused contamination before.
- **OpenCL targets are Intel and Adreno only — never NVIDIA.** The RTX 4070 in
  this box must never be selected.
- **No background GPU work** on the same device during a run.
- **Best-of-3, foreground, ≥120 s cooldown between runs;** compare only
  **adjacent A/B/A/B**, never across thermal states.
- **Never report a number not present in tool output** (a past failure mode).
- gpu_native run: `nntrainer_qwen3_gpu` with `NNTR_MODEL_GEMMA2=1` /
  `NNTR_MODEL_4B=1` (default Qwen3-0.6B); TPS from
  `[prefill M=…] … => N TPS` (`main.cpp:749`) and the decode summary
  (`main.cpp:619`). Profiling adds a `clFinish` tax — keep `NNTR_STAGE_PROFILE`
  and `NNTR_OPENCL_PROFILING` **off** for wall-clock runs.

---

## 8. Top landmines (with avoidance)

- **L1 dispatch-idle resurrection** — a generic executor that inserts a
  `clFinish`/barrier per op revives the #10 win loss (Adreno 834→908). Keep one
  in-order queue, barriers only on OOO (Intel) devices, drain once per sequence.
- **L2 flag-flip → slow SVM kernel** — never default the pool on before Step 4.
- **L3 planner offset → image creation fail** — align every tensor offset
  (not just the pool base) when the image path is used.
- **L4 planner reuse race** — offset reuse needs exec-order dependency events
  (mandatory once `clFinish` is removed).
- **Drift gate (Step 5)** — `attention_kernels.cpp:500-522` deliberately forces
  GPU MHA to CPU for >N layers (28-layer numerical drift). Registration alone
  will not light up attention; the drift must be resolved.
- **`.bin` positional loader (Step 9)** — until self-describing, every new model
  needs a loader branch, breaking "zero GPU work".

---

## 9. Open / future

- per-tensor mixed residency (consume `t_engine`, split the pool) — deferred;
  graph-wide is the current design.
- recordable-queue decode replay — unimplemented; verify scalar-arg / SVM-ptr
  mutation on the Adreno 840 driver before committing.
- generic property-driven fused-layer infrastructure (beyond #6/#7).
- DeviceCaps consolidation (fold `tv::DeviceImageCaps` into
  `opencl_device_info` + trial-compile `read_imageui` probe to replace
  `NNTR_V8C_BUF`) and an ExecPlan resolver (`DeviceCaps × ModelFeatures`).

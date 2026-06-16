# Generic Tensor Virtualization Plan

Paper reference: ML Drift (arXiv:2505.00232) §3.1, §3.2, §3.6, §3.7, §3.8.

> "Tensor virtualization decouples the logical representation of a tensor
> from its physical storage on the GPU, allowing tensors to be realized
> using various types and numbers of GPU memory objects (textures, buffers,
> 1D image buffers, 2D textures, 3D textures, texture arrays)."

## Goal

Close the measured 17× prefill / 14× decode gap on Qwen3-0.6B (Adreno 830)
against paper-class numbers (~4900 / ~140 TPS). Profiling shows the gap
is not kernel-level (v8c GEMM hits 87% HW peak in isolation) but
**data-flow between layers**: per-FC activation upload + quantize +
read-back + CPU RMSNorm/SwiGLU/residual round-trip. Paper §3.6 closes
this with fused-on-GPU operators; §3.8 reshapes KV cache so attention
becomes a convolution; §3.7 splits prefill/decode into different quant
kernels.

## Paper-vs-state gap (confirmed 2026-05-28)

| Paper | Claim | Our state | Gap |
|---|---|---|---|
| §3.1 | 4-element SIMD C₄ layout for activations (~20% claim) | image2d view only for v8c FC weights | PHWC4 activations not applied |
| §3.2 | Decoupled logical/physical, shaders bound at codegen | `tv::TensorBacking` exists but disconnected from `nntrainer::Tensor` | Bridge missing (Step 1e) |
| §3.6 #1 | Single kernel: Q+K+V proj + RoPE + layout `(B,1,S,hq·dh)→(B·hkv,S·hq/hkv,dh)` | 3 separate GPU FCs + RoPE on CPU (`mha_core.cpp:~2248`) | 100% missing (Step 2) |
| §3.6 #2 | Single kernel: RMSNorm + residual + elementwise | RMSNorm 100% CPU NEON, residual separate layer | 100% missing (Step 4) |
| §3.6 #3 | Auto element-wise fusion w/ FC | No fusion logic | Low priority |
| §3.7 | Prefill = dedicated quant kernel + int8 GEMM; Decode = quant **inside** op kernel | Single v8c kernel for both; M-binning only in profiling | Stage split missing (Step 5) |
| §3.8 | K: OHWI `[cache_size, dh]` (Kᵀ form); V: reversed `[dh, cache_size]` | Row-major `[B,1,S,kv_width]` (`kv_cache_manager.cpp:39–48`); helpers in tree but unused | Layout reorder missing (Step 3) |
| Residency | Implicit GPU-resident activations between fused ops | Every v8c FC output `clEnqueueReadBuffer`'d to host (`blas_kernel_interface.cpp:1033–1035`) | Enabled by Steps 1e+2+4 |

## Re-revised: everything-on-GPU plan (2026-05-28)

The earlier segment-based plan (still useful as the chaining strategy,
preserved below) tried to convert *chains* of adjacent ops together to
avoid host materialize in between. That approach hit a hard wall in
Segment A.2 (`8a53efcf`): the GPU RMSNorm kernel's ~1e-6 reduction-
order drift from CPU NEON, amplified by v8c's int8 quantization
boundary across 28 layers, produced garbled output. The
CPU-norm+publish workaround is bit-exact but TPS-regressing because
producer side stays CPU.

**Conclusion:** mixed CPU/GPU numerics cannot coexist for this stack.
Either all-CPU or all-GPU within the residual chain. Pivot:

> **Move EVERY op onto GPU, then chain via TensorBacking. The chain
> uses one numerics system (GPU). Final model output may differ
> bit-for-bit from CPU baseline but stays coherent because every
> op shares the same drift.**

This is the architectural goal the user has set since Session 1.
Plan is reorganized by op-level migration priority (largest CPU
wedge first, then dependencies).

### Op-level migration priority (current time costs, 950 ms prefill)

| # | Op | CPU ms | Existing kernel | New work | Est. effort |
|---|---|---|---|---|---|
| **1** | **QK · softmax · V·S (mha core)** | **~150** | `two_conv_attention.cl` (3-kernel, VGPR-spilled, slower than CPU) | **single-kernel flash-style fusion** | **3-4 wk** |
| 2 | RoPE on Q, K | ~20 | `rotary_emb.cl` ✓ (not wired live) | wire + take cl_mem inputs | 1 wk |
| 3 | KV cache write + int8 quant | ~30 | none | new kernel | 1-2 wk |
| 4 | RMSNorm (3 sites: attn_norm, ffn_norm, output_norm) | ~10 | `rmsnorm.cl` exists (numeric drift) | fused with residual (paper §3.6 #2) — own numerics | 1-2 wk |
| 5 | residual add (×2 per layer) | ~15 | `addition.cl` ✓ (not wired) | wire + fused with norm (#4) | 0.5 wk |
| 6 | SwiGLU (gate × σ(gate) × up) | ~19 | `swiglu.cl` ✓ (not wired) | wire + take cl_mem | 0.5 wk |
| 7 | Async FC readback (drop `CL_TRUE`) | ~50 | n/a (queue restructure) | event-chain + barrier-with-wait-list | 1 wk |
| 8 | Embedding lookup | ~10 | none | new gather kernel | 1 wk |
| 9 | LM head (FC projection to vocab) | ~5 | v8c reusable | wire | 0.5 wk |
| 10 | q_norm / k_norm (small RMSNorm per head_dim) | ~5 | `rmsnorm.cl` variant | similar to #4 | 0.5 wk |

**Total ~3 months.** Expected end state: prefill ~600-800 TPS,
decode ~50-70 TPS. Beyond that, paper-class (~4900/140) requires
command-graph forward (separate execution-model rewrite).

### Why GPU attention (#1) first

- Largest single wedge (~150 ms, but reaches ~209 ms in some
  measurements when counting RoPE + KV write inside the same CPU
  island).
- Unblocks the entire attention chain to be GPU: once mha is GPU,
  RoPE/KV write/QK·softmax·VS are all in cl_mem; surrounding wq/wk/wv
  outputs (already GPU) can feed directly without host materialize.
- Existing `two_conv_attention.cl` is the wrong design (3 separate
  kernels + global intermediate scores). Flash-attention-style
  single kernel solves both the perf and the residency problem.

### Design constraints for GPU attention single-kernel

- Adreno 830: 32 KB local memory budget per workgroup (tight).
- VGPR per work-item: limited; full FP32 score row per WI is
  expensive. Use tile-based per-block accumulation with online
  softmax (Dao et al. 2022 flash attention pattern).
- Subgroup size: 64 (qcom "half" of natural 128).
- Per-attention-block sizing for Qwen3-0.6B:
  - 1 workgroup per (head, query-row) pair
  - head_dim = 128, num_heads_q = 16, num_heads_kv = 8 (GQA=2)
  - prefill: max M=282 queries × M=282 keys per head
  - decode: M=1 query × M_cache keys
- KV cache currently row-major `[B, 1, S, kv_width]`. New kernel
  reads via that layout for now; reordering to OHWI (paper §3.8)
  is a separate downstream win.

### Numerical-parity strategy

Once GPU attention lands, the chain CPU→GPU→CPU→GPU→... is broken
by replacing each remaining CPU op with its GPU variant in priority
order (table above). At each replacement step, the model output
will diverge from CPU baseline but should stay coherent (uniform
GPU drift, not mixed). Acceptance check at each step: run the
prompt, verify output looks like reasonable text (Korean/English
sentences, not random Unicode).

### Workspace flag conventions

Each migration uses an env gate so it can be toggled independently
during development:
- `NNTR_GPU_MHA=1` — GPU attention path (Task #16)
- `NNTR_GPU_ROPE=1` — wire `rotary_emb.cl`
- `NNTR_GPU_KV_WRITE=1` — new KV write/quant kernel
- `NNTR_GPU_RMSNORM=1` — fused RMSNorm+residual path
- (etc.)

When all flags are ON, the chain is fully GPU. Until then,
each flag enables one piece; CPU fallback handles the rest.

---

## Older: segment-based end-to-end GPU residency plan (2026-05-28)

(Kept for reference — chaining strategy still applies once each op
is GPU. Was the framing before the all-GPU pivot above.)


The previous 5-step plan focused on individual paper-section techniques
but assumed the FC stack was the bottleneck. After direct profiling
(`NNTR_V8C_PROFILE`, `NNTR_LAYER_PROFILE`, `NNTR_FC_PROFILE` on Qwen3-
0.6B), the actual time distribution is:

| Op | Time | % prefill (950ms) |
|---|---|---|
| FC GEMM kernel | 249 ms | 26% (already at 87% HW peak) |
| FC quant_kernel | 144 ms | 15% |
| FC write_act (host→dev) | 71 ms | 7% |
| FC misc framework | ~191 ms | 20% |
| **mha_core (CPU island)** | **209 ms** | **22%** |
| swiglu CPU | 19 ms | 2% |
| rms_norm CPU | 10 ms | 1% |

**Theoretical ceiling with current GEMM and current execution model:**
- If we eliminated 100% of write_act + quant + readback + non-FC CPU: 
  ~950 ms → ~249 ms = **3.8× = ~1100 prefill TPS**
- Paper-class on Qwen3-0.6B scaled (Adreno 830): ~4900 TPS
- **The gap to paper requires changing the execution model itself**,
  not just kernel-level optimizations.

The new plan therefore organizes work by **GPU residency segments**: a
segment is a chain of adjacent ops connected via `TensorBacking` with
ZERO host materialize internally. Boundaries between segments are the
only host transfers per forward pass.

### Qwen3 decoder block segments

```
input ──→ attention_norm ──┬─→ wq ─→ q_norm ─→ RoPE ──┐
   ┊      [Segment A]      ├─→ wk ─→ k_norm ─→ RoPE ──┼─→ mha ─→ wo
   ┊                       └─→ wv ───────────────────┘  [B,CPU]  ↓
   └────────────── residual ──────────────────────────────────→ decoder_add
                                                                  ↓
                                                            ffn_norm ──┬─→ wgate ──┐
                                                            [Segment C]│            ├─→ swiglu ─→ wdown
                                                                       └─→ wup ───┘   [Segment D]
                                                                                            ↓
                                                                                    decoder_output ── → next layer
                                                                                    [Segment E]
```

### Segments

#### Segment A — pre-attn RMSNorm → wq, wk, wv

- **Ops:** `attention_norm` (rms_norm, hidden=1024) → `wq`, `wk`, `wv`
- **Internal handoff:** attention_norm output kept in cl_mem, registered
  as `TensorBacking`; wq/wk/wv read from backing (skip upload + Step
  2b.0 quant cache continues to cover redundant quant)
- **Host boundary at exit:** wq/wk/wv outputs materialize to host because
  q_norm / k_norm / mha are CPU
- **Savings per layer:** ~3 uploads (~1.6 MB) + attention_norm CPU
  (~0.18 ms) + Step 2b.0 covers remaining quant skip

#### Segment B — mha CPU island (unchanged for now)

- **Ops:** q_norm, k_norm, RoPE, KV write+int8quant, QK matmul, softmax,
  V·S matmul
- **Status:** Stays CPU. GPU attention (`two_conv_attention.cl`) is
  currently 1.6× slower than CPU on Adreno 830 (memory
  [project_gpu_attention_status]). Replacing this is a separate algo-
  rithm-level problem; deferred.
- **Future:** When a competitive GPU mha lands, B merges with A and C
  → entire layer GPU-resident.

#### Segment C — wo → decoder_add → ffn_norm → wgate, wup

- **Ops:** `wo` (FC) → `decoder_add` (residual) → `ffn_norm` (rms_norm)
  → `wgate`, `wup` (FC)
- **Internal handoff:** wo output GPU-resident; decoder_add takes (wo,
  residual) — residual comes from layer input (still host today, GPU
  once Segment E links upstream); ffn_norm reads residual_add output;
  wgate/wup share ffn_norm output via backing
- **Host boundary at entry:** wo input from mha (CPU island, host)
- **Host boundary at exit:** wgate/wup outputs continue into Segment D
  (also GPU)
- **Savings:** residual_add CPU + ffn_norm CPU + 3 uploads (wo input
  re-route, ffn_norm output share) + Step 2b.0 quant skip

#### Segment D — wgate, wup → swiglu → wdown

- **Ops:** swiglu (element-wise gate × up) → wdown (FC)
- **Internal handoff:** wgate/wup outputs feed swiglu via backings;
  swiglu output → wdown
- **Savings:** swiglu CPU + 2 downloads + 1 upload

#### Segment E — wdown → decoder_output → next layer's attention_norm

- **Ops:** decoder_output (residual_add) → forward to next layer's
  Segment A
- **Internal handoff:** wdown output + residual GPU → next layer reads
  via backing
- **When all 28 layers' E lit:** the layer chain forms one continuous
  GPU residency from Segment A of layer 0 to Segment E of layer 27;
  only mha islands break it

### Order of attack

1. **Segment A** — smallest, lowest risk. Validates the producer-side
   (RMSNorm backing output) + consumer-side (FC backing input) wiring.
2. **Segment C** — biggest single chunk (4 ops, includes the bulky 
   residual+RMSNorm pair that paper §3.6 #2 targets). Once A works the
   pattern is templated.
3. **Segment D** — small (2 ops + swiglu). Quick after C.
4. **Segment E** — last hop, links layer boundaries.
5. **Old "Steps 2-5"** become OPTIMIZATIONS within segments:
   - Old Step 2 (fused QKV+RoPE) → Segment A optimization: merges 4
     kernels (RMSNorm + 3 FCs + RoPE) into 1
   - Old Step 4 (fused RMSNorm+residual+elementwise) → Segment C
     optimization
   - Old Step 3 (KV OHWI) → Segment B prerequisite
   - Old Step 5 (stage-aware decode quant) → cross-segment optimization
6. **Segment B GPU rewrite** — multi-week algorithmic problem;
   parallel track.

### Expected cumulative TPS (Qwen3-0.6B / SD8 Elite)

| After | Prefill TPS | Decode TPS |
|---|---|---|
| Today (`9ece904e`) | 293 | 10.65 |
| Segment A | ~330 | ~12 |
| Segment C | ~430 | ~16 |
| Segment D | ~470 | ~18 |
| Segment E | ~520 | ~22 |
| + fused-kernel optimizations (old steps 2/4) | ~750 | ~35 |
| + Segment B GPU mha | ~1100 (ceiling) | ~60 |
| Paper-scaled target | ~4900 | ~140 |

Beyond ~1100 the bottleneck shifts to execution-model overhead (per-
layer Tensor materialization, layer-node dispatch). Closing that
requires command-graph-style forward (a separate, larger refactor).

### Step 1 — Foundation (paper §3.2)

#### 1a–1d — DONE (commit `b3d395f8`)

- ViewKind enum (BUFFER / IMAGE_1D / IMAGE_2D / IMAGE_3D)
- ViewSpec with depth + slice_pitch_bytes
- Factory helpers: PHWC4 (fp16/int8), OHWI K-cache, OHWI_T V-cache
- `TensorBackingPool` singleton (name → backing)
- v8c FC weights migrated to ViewKind::IMAGE_2D

#### 1e — `nntrainer::Tensor` ↔ `TensorBacking` bridge (1 day)

- Forward-declare `nntrainer::tv::TensorBacking` in `tensor.h`.
- Add opt-in `tv::TensorBacking* backing_ = nullptr` member.
- Add `setBacking(tv::TensorBacking*)` / `getBacking()` accessors.
- CPU layers ignore (default null); GPU layers can set/read.
- **Validation:** route v8c FC's output tensor through setBacking →
  getBacking and assert pointer identity. Expect zero TPS change; the
  point is to prove the brigde compiles + survives the layer chain.

**Exit criterion:** prefill TPS within ±5% of pre-bridge baseline (282
TPS on Qwen3-0.6B / SD8 Elite).

### Step 2 — Fused RoPE + Q/K/V layout kernel (paper §3.6 #1) [2-3 weeks]

Single OpenCL kernel taking post-RMSNorm activation (PHWC4 image2d
view via `TensorBacking`) + Q/K/V weight backings, producing Q/K/V in
attention-input layout `[B·hkv, S·hq/hkv, dh]`.

#### Current path being replaced

Today (verified at `dab3e48a`):

| Pass | Where | What |
|---|---|---|
| Q FC | `qkv_layer.cpp:178` → `dotCl_v8c` | `[S,hidden] · Wq → [S, hq·dh]` FP16 |
| K FC | same | `[S,hidden] · Wk → [S, hkv·dh]` FP16 |
| V FC | same | `[S,hidden] · Wv → [S, hkv·dh]` FP16 |
| RoPE Q | `mha_core.cpp:978/1031` → `apply_rotary_emb_tensor_v2` (2208) → `compute_rotary_emb_value` | in-place on Q, CPU NEON, uses cached cos/sin table |
| RoPE K | `mha_core.cpp:980/1035` | same, on K (in-place or to write view) |
| V copy | `mha_core.cpp:1045` | `copyData` only — no rotation (paper convention) |

3 GPU dispatches + CPU pass for RoPE + CPU pass for V copy = 5
synchronization points per attention block. v8c GEMM hits 87% peak in
isolation but the e2e cost is dominated by inter-op overhead.

#### Qwen3-0.6B target shapes

- hidden = 1024, num_heads_q (hq) = 16, num_heads_kv (hkv) = 8,
  head_dim (dh) = 128, GQA_SIZE (hq/hkv) = 2
- Q FC: K=1024, N=hq·dh=2048
- K FC: K=1024, N=hkv·dh=1024
- V FC: K=1024, N=hkv·dh=1024
- RoPE: cos/sin table `[max_position, head_dim]` FP16, pair-rotation
  with `half_ = head_dim/2 = 64`. Cached in
  `MHACoreLayer::rope_freq_cache` (static, lifetime = process).

#### Kernel design (first cut)

Single OpenCL kernel `fused_qkv_rope_layout` that:

1. **Quantize activation once.** Activation `[1, S, 1024]` FP16 → int8
   per-row with FP32 scale + int32 zero-point. Same code path as v8c's
   quant_kernel, but only runs ONCE (today: 3×, once per FC).

2. **Three int4×int8 GEMM passes inline.** Each pass reads from the
   shared int8 activation buffer + its own QINT4 weight image2d view
   (existing v8c weight backings, already cached). Writes intermediate
   int32 output per output row.

3. **Dequantize each output inline.** Multiply by activation scale ·
   weight scale per row, subtract row-sum corrections, produce FP16
   per-element output.

4. **Apply RoPE on Q and K.** For each `(s, head, dim_pair_idx)` of Q
   and K, load cos[from+s, dim_pair_idx] and sin[from+s, dim_pair_idx]
   from `__constant`-pinned tables, do pair-wise rotation
   `(x0, x1) ← (x0·cos − x1·sin, x0·sin + x1·cos)`. V skipped.

5. **Layout transform on Q only.** Q output is the only one that
   needs a non-trivial reshape: from `[S, hq·dh]` row-major to
   `[hkv, S·(hq/hkv), dh]` = `[8, S·2, 128]` for Qwen3. K and V come
   out as `[S, hkv·dh]` which is already the OHWI weight-form `[S, dh]`
   per head — just a re-interpretation, no shuffle.

#### Output contract (consumed by Step 3 KV cache writer)

- Q tensor: `[B·hkv, S·hq/hkv, dh]`, packed PHWC4 image2d-ready.
- K tensor: `[B, S, hkv·dh]` row-major, byte-compatible with OHWI
  `[hkv, S, dh]` view (paper §3.8 K-cache form).
- V tensor: `[B, S, hkv·dh]` row-major. Step 3 will need to physically
  transpose to `[B, hkv, dh, S]` (OHWI_T form) at cache-write time, OR
  this kernel can produce it in OHWI_T directly. Decision deferred to
  Step 3.

#### Plumbing plan (this step's commits)

- **2a (skeleton):** new file `nntrainer/tensor/cl_operations/blas_kernels/fused_qkv_rope.cl`
  with kernel body stub; `blas_kernel_interface.{h,cpp}` exposes
  `fused_qkv_rope_layout_gpu(input, wq, wk, wv, cos_tbl, sin_tbl,
  from_pos, q_out, k_out, v_out)`. Env-gated `NNTR_FUSED_QKV_GPU=1`;
  default returns `false` to fall back to existing 3-FC path.
- **2b (kernel body):** flesh out the quant → 3-GEMM → dequant → RoPE
  → writeback. Unit-tested in isolation against synthetic input.
- **2c (validation):** synthetic-input harness producing reference Q/K/V
  via current 3-FC + CPU RoPE pipeline; compare element-wise with
  fused kernel output, gate on relL2 < 0.5%.
- **2d (live wire):** modify `qkv_layer.cpp:178` to call
  `fused_qkv_rope_layout_gpu` when env-gated, fall through to
  `input_step.dot(Weights, Outputs)` + current RoPE otherwise. Also
  needs to suppress mha_core's RoPE call (lines 978/980/1031/1035)
  when the fused path took over — likely a flag in
  `MHACoreLayer::incremental_forwarding`.
- **2e (TPS measurement):** Qwen3-0.6B prefill TPS on SD8 Elite.
  Expected ~700 prefill / ~12 decode (closes ~half the prefill gap).

**Exit criterion:** bit-equivalent (relL2 < 0.5%) vs current pipeline.

### Step 3 — KV cache OHWI / OHWI_T migration (paper §3.8) [1 week]

K cache stored as `[cache_size, dh]` per head (convolution-weight form);
V cache as `[dh, cache_size]` (reversed). Dynamic append (paper does not
specify) keeps a static slab + an append cursor; new tokens written at
the cursor, attention reads `[0..cursor]`.

**Replaces** `kv_cache_manager.cpp:39–48` row-major allocation. Writer
is the Step 2 kernel's K/V output. Reader is attention (still
two-1×1-conv shape, now consuming OHWI/OHWI_T directly via image2d).

**Exit criterion:** Qwen3-0.6B coherent output, attention TPS unchanged
or better vs current `NNTR_MHA_GPU=1` path (~80 prefill TPS).

### Step 4 — Fused RMSNorm + residual + elementwise (paper §3.6 #2) [2-3 weeks]

Single GPU kernel: input activation + residual stream + γ vector →
normalized PHWC4 image2d output. Output `TensorBacking` consumed by
Step 2's next-layer invocation. Eliminates current per-layer
write_act (~127 ms / prefill@282) + CPU RMSNorm (~13 ms).

**Output contract:** PHWC4 image2d activation backed by `TensorBacking`
in `TensorBackingPool`, registered under `layer_{i}_norm_out`.

**Exit criterion:** bit-equivalent vs CPU RMSNorm + residual.

### Step 5 — Stage-aware quantization (paper §3.7) [1-2 weeks]

Two distinct v8c FC code paths chosen by stage:

- **Prefill (M > 1):** keep current dedicated quant kernel → int8 GEMM
  with pre-quantized weights → dequant on output. Already what v8c
  does today; just confirm the path is well-isolated.
- **Decode (M = 1):** new kernel with activation quantization **fused
  into the FC kernel itself**. Eliminates the quant launch + scratch
  upload that today dominates decode at 10 TPS.

May also feed back to fixing the early-EOS bug in
`project_kv_int8_gpu_wip` since both touch the same decode quant path.

**Exit criterion:** Qwen3-0.6B decode TPS ≥ 30 (3× current 10 TPS).

## Cumulative TPS expectations (Qwen3-0.6B / SD8 Elite)

| After step | Prefill | Decode |
|---|---|---|
| Today (baseline) | 282 | 10 |
| Step 2 (fused QKV+RoPE) | ~700 | ~12 |
| Step 3 (OHWI KV) | ~900 | ~15 |
| Step 4 (fused RMSNorm+residual) | ~2500 | ~25 |
| Step 5 (stage-aware quant) | ~3000 | ~80 |
| Paper-scaled target | ~4900 | ~140 |

Estimates assume each step independently closes ~half of the remaining
gap in its dominant regime. We will recalibrate after each step.

## Risks (rolled up)

1. **`nntrainer::Tensor` bridge is invasive** — mitigated by opt-in
   default-null pointer; CPU layers untouched.
2. **TensorPool integration deferred** — Step 1 uses a parallel
   `TensorBackingPool` keyed by tensor name, independent of nntrainer's
   TensorPool. May need integration later when layer-graph router is
   touched.
3. **Step 2 kernel complexity** — single kernel doing 4 jobs (3 GEMMs +
   RoPE + layout). Mitigate by progressive validation: first verify
   3 GEMMs fused, then add RoPE, then add layout transform.
4. **Step 3 dynamic append semantics** — paper does not specify; we
   use static slab + cursor pattern. Watch for cache-line / image
   width-alignment hazards on Adreno.
5. **Step 5 decode kernel correctness** — fused quant inside FC is
   the same code shape that breaks today in `NNTR_KV_INT8_GPU` path.
   Land Steps 2–4 first; revisit with their TensorBacking machinery.

## Non-goals (deferred)

- Auto element-wise fusion (paper §3.6 #3) — low ROI vs Steps 2/4.
- Generic codegen / device specialization (paper §3.4).
- Texture arrays / 3D textures for KV (paper §3.2 enumeration).
- Migrating existing v8c FC weight path to PHWC4 — keep current
  IMAGE_2D + RGBA UINT32 packing; PHWC4 is for **activations**.

## Per-step verification gates

| Step | Gate |
|---|---|
| 1e | Bridge round-trip pointer-identity test passes; v8c FC e2e TPS ≥ 280 |
| 2 | Fused kernel output relL2 < 0.5% vs unfused pipeline; output backing in pool |
| 3 | Attention reads OHWI/OHWI_T KV via image2d; Qwen3-0.6B coherent |
| 4 | Fused RMSNorm+residual relL2 < 0.5%; backing registered in pool |
| 5 | Decode TPS ≥ 30; prefill regression < 5% |
| e2e | Prefill TPS ≥ 1.5× current after each of Steps 2, 4; decode ≥ 3× after Step 5 |

# Planner-Decided cl_mem Residency — Design

Status: DRAFT (2026-06-11). Author: Jijoong Moon + Claude.
Goal: move layer-graph GPU activations (and the weights the GPU reads) onto
cl_mem device residency, **decided by the memory planner at allocation time**, so
every CL layer binds cl_mem **uniformly** — no SVM/cl_mem hybrid boundary, no
runtime residency flipping. This is gpu_native's residency model expressed through
nntrainer's TensorPool / MemoryPool / MemoryPlanner / RunLayerContext, reusing the
planner's lifespan-based memory reuse.

---

## 1. Why the incremental approach failed (what we measured)

The current `ClBufferPool` keeps **two planes per tensor** (an SVM plane from
`MemoryPool::allocate` + a parallel cl_mem plane) and flips cl_mem-vs-SVM
**per edge at runtime** via `MemoryData.device_valid`. A converted producer writes
cl_mem and sets `device_valid`; a converted consumer reads cl_mem when it sees the
bit. Everything else stays SVM.

Device-measured consequences (Adreno 840, Gemma2-2B, isolated dir):

- Converting the FFN `pre_ffn_norm -> gate/up` cl_mem **input** edge across all 26
  layers => **garbage from token 1**. A SINGLE layer converted => **clean**.
- `attention_norm -> qkv` via the IDENTICAL mechanism => **clean**.
- 15+ targeted fixes ALL failed: per-token vs one shared sub-buffer handle
  (SHAREHANDLE), standalone `clCreateBuffer` per offset (DISCRETE = gpu_native's
  buffer model), kernel copy vs DMA (KERNELCOPY), fine-grain SVM (FINE, confirmed
  supported), rotating FC scratch (ROTSCRATCH), writing both planes (WRITEBOTH),
  every clFinish drain. None fixed it.
- The OFFSET REUSE is real and correct: all 26 layers' `pre_ffn_norm` +
  `ffn_down` + `decoder_output` share one planner offset (disjoint lifespans).
  But **reuse is not the bug** — gpu_native reuses by lifespan too. The bug is the
  **SVM/cl_mem mix + runtime flipping**, which creates fragile mid-pipeline
  boundaries (e.g. `rms` reads SVM `post_attn`, writes cl_mem) whose coherence
  the planner's SVM-based schedule never accounted for.

**Diagnostic caveat for future work:** reading the shared FC scratch `sc.y_fp16`
mid-pipeline (`clEnqueueReadBuffer` + `clFinish`) CORRUPTS the run on Adreno — OFF
+trace produced garbage while OFF alone is clean. Never probe by reading
mid-pipeline scratch; compare token output, or copy to a dedicated debug buffer.

**The principle that resolves it (J. Moon):** the planner already knows, from the
static model + input, exactly which tensor needs how much memory and for how long.
So "this tensor lives in cl_mem" must be a **planner/pool allocation decision (a
static tensor property)**, NOT a per-edge runtime flip — and it must be applied
**consistently to all of that tensor's producers and consumers**, with weights
included. gpu_native works precisely because it is uniformly cl_mem with the same
lifespan reuse.

---

## 2. Target architecture

> cl_mem residency is a **static property** set by the planner at allocation and
> applied **uniformly**. There is **no SVM plane for GPU activations/weights** and
> **no runtime `device_valid` flipping**. SVM/host exists only at the two model
> boundaries (input embedding, output logits).

### 2.1 Residency class (planner input) — AUTO-DERIVED
- Each tensor request carries a residency class: `GPU_CLMEM | SVM | HOST`.
- **Auto-derived in `Manager`/`TensorPool`** (not declared in the model builder):
  when the engine is gpu, all GPU activation tensors and all GPU-read weights
  (rms `gamma`, FC int4, attention) become `GPU_CLMEM`; only the model I/O boundary
  tensors that need host access stay `SVM`/`HOST`. The graph already knows the
  engine and each tensor's role, so the derivation lives there.
- The planner schedules `GPU_CLMEM` tensors into the cl_mem plane with its normal
  lifespan reuse (unchanged algorithm — reuse is fine).

### 2.2 Pool (`ClBufferPool`)
- For `GPU_CLMEM` tensors: allocate ONLY in the cl_mem plane (drop the SVM shadow
  for them). `MemoryData.device_mem` = the cl_mem sub-buffer; `isClMem()` true.
- **One sub-buffer handle per padded offset** (the SHAREHANDLE result): every token
  the planner placed at an offset binds the SAME handle, so distinct handles never
  alias one region. (Keep this even though it alone didn't fix the hybrid — under a
  uniform design it is the correct, coherent choice and matches gpu_native's one
  handle per logical buffer.)
- For `SVM`/`HOST` boundary tensors: SVM/host as today.

### 2.3 MemoryData / Tensor / RunLayerContext (DECIDED)
- **`getData()` returns the cl_mem handle (the device "address") for a `GPU_CLMEM`
  tensor** — it does NOT throw and there is NO separate accessor. Rationale (J.
  Moon): by the time `getData()` is called the tensor is ALREADY allocated in
  cl_mem per-tensor, so `getData()` simply hands back that cl_mem reference. The
  backing pointer stored in `MemoryData` IS the cl_mem handle; `getData<T>()`
  reinterprets it. `isClMem()` tells a layer how to USE the returned value (bind as
  `cl_mem` via `SetKernelArguments(&h, sizeof(cl_mem))` vs SVM via
  `SetKernelSVMArguments`). Host pointer arithmetic on a `GPU_CLMEM` tensor is a bug
  by construction (its memory is not host-addressable).
- **Multiple tensors can share ONE cl_mem** — two sources: (a) planner liveness
  reuse (disjoint-lifespan tensors at the same offset bind the same one-per-offset
  sub-buffer handle), and (b) READ_ONLY_VIEW / in-place edges that already share a
  `MemoryData`. The pool/`MemoryData` must return the SAME handle for all of them;
  layers must treat the handle as shared (no assumption of exclusive ownership).
- This is the "RunLayerContext considers cl_mem and applies it to MemoryData" piece
  from the original directive — realized by making `MemoryData`/`getData()`
  cl_mem-aware, not by a parallel API.

### 2.4 CL layers — uniform cl_mem binding
- Every CL layer (`rms_norm_gpu`, FC `dotCl_v8c`, `geglu_cl`, `addition_layer_cl`,
  attention, `reshape`/`transpose`), for `GPU_CLMEM` I/O, binds cl_mem handles for
  **inputs AND outputs**. No SVM path, no `device_valid` check, no env gate for
  these tensors.
- The existing cl_mem binding code (`device_clmem_in`, `clmem_out_edge`, geglu
  `in1_cl/in2_cl`, rms `out_clmem`) becomes the DEFAULT for `GPU_CLMEM`, driven by
  `isClMem()` (static), not `device_valid` (runtime).
- Ordering: the in-order SVM-pool queue already serialises kernels; **no per-op
  clFinish, no SVM map/unmap** (gpu_native proves an in-order queue suffices —
  `forward_one_layer_v2` has zero non-profiling clFinish).
- **Weights too:** load rms `gamma` (and any other GPU-read weight) into cl_mem, the
  way FC already does via `V8cWeightEntry` (`clCreateBuffer`). No SVM `gamma`.

### 2.5 Boundaries (explicit copies, never hybrid kernels) (DECIDED)
- **Input:** the token-embedding lookup writes the first activation into cl_mem.
- **Output:** keep cl_mem **all the way to the end** — `lm_head` reads cl_mem hidden
  and writes cl_mem logits; ONE explicit `clEnqueueReadBuffer` (cl_mem -> host)
  feeds sampling. (gpu_native does exactly this.)
- **Explicit "lower" op:** provide a user-callable `cl_mem -> SVM/host` copy (e.g.
  `Tensor::toHost()` / a context op) so a caller can bring a tensor down to SVM/host
  WHEN it genuinely needs host access. Boundaries are thus explicit and user-driven,
  never an implicit hybrid kernel that mixes SVM and cl_mem activation args.

### 2.6 Why this is coherent (and the hybrid was not)
- No mid-pipeline SVM/cl_mem boundary => no hybrid kernels => no coherence mix.
- Static residency (planner property) => consistent across all ops of a tensor; no
  runtime-state-dependent races.
- Planner lifespan reuse applies to the cl_mem plane uniformly; the in-order queue
  serialises cl_mem producer->consumer and the reuse boundary, exactly as it does
  for gpu_native.

---

## 3. Migration (the hard part)

**Lesson:** the conversion UNIT is a TENSOR (all its producers + consumers convert
together), NOT an edge. Converting an edge leaves the tensor SVM for some ops and
cl_mem for others => the corrupting hybrid. To avoid ANY interior boundary, convert
the connected activation graph as one unit, bounded only at model I/O.

The residual stream (embedding -> residual adds -> decoder_outputs -> final norm)
is the backbone; every norm/FC/geglu/attention hangs off it. So the coherent unit
is essentially the whole forward activation graph.

### Staged plan (each stage validated token-identical vs gpu_native / vs OFF)
- **S0 Infra:** residency class on tensor specs; `ClBufferPool` cl_mem-only alloc
  for `GPU_CLMEM` + one-handle-per-offset; `Tensor::getClMem()` / `isClMem()`;
  RunLayerContext plumbing; explicit cl_mem<->host copy helpers.
- **S1 Weights:** load `gamma` (+ any GPU-read weight) into cl_mem; rms binds cl_mem
  gamma. (FC weights already cl_mem.)
- **S2 Layers:** convert each CL layer to uniform cl_mem binding driven by
  `isClMem()`. Order: addition (residual backbone) -> rms -> FC -> geglu ->
  attention -> reshape/transpose. After this, with ALL activations marked
  `GPU_CLMEM`, the whole graph is cl_mem with no interior SVM.
- **S3 Boundaries:** embedding writes cl_mem; logits read back to host once.
- **S4 Validate:** Qwen3-0.6B first (smaller, faster cycles), then Gemma2-2B; expect
  token-identical to gpu_native and to the current SVM OFF path.

### Risk control
- Validate on 0.6B before 2B.
- Keep the current SVM path selectable (engine/env) until S4 passes, so OFF stays a
  reference. But do NOT run the half-converted hybrid as a "mode" — it is known
  unstable; conversion lands per-tensor-class, not per-edge.

---

## 4. Design decisions (RESOLVED 2026-06-11, J. Moon)
1. **Residency class is AUTO-DERIVED** in `Manager`/`TensorPool` from engine==gpu +
   tensor role (activation / GPU-read weight / boundary), NOT declared per-tensor in
   the model builder. The graph already knows engine and roles; derive it there.
2. **`getData()` returns the cl_mem handle** for a `GPU_CLMEM` tensor (no throw, no
   separate API) — see §2.3. Allocation is already cl_mem per-tensor, so `getData()`
   just hands back that reference; `isClMem()` says how to bind it.
3. **One cl_mem can back multiple tensors** (planner liveness reuse + view edges) —
   the pool returns the SAME handle; layers assume shared, non-exclusive ownership.
   KV cache = `GPU_CLMEM` (own offsets, no reuse); confirm the image attention path
   binds it as cl_mem uniformly.
4. **cl_mem all the way to the end** — lm_head reads cl_mem hidden, writes cl_mem
   logits; ONE host readback for sampling. Plus a user-callable explicit
   cl_mem->host "lower" op (§2.5) for any genuine host-access need.

## 5. Next: implement S0 (infra) — pending session resources.

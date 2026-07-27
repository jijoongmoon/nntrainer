# GPU / backend contribution conformance checklist

**What this is.** An operational checklist a reviewer or implementer runs down before landing any
GPU/backend contribution (a new layer's GPU path, a new op, a new backend). Each item states the
normative rule, the plain yes/no question, and a **mechanical verification** you can actually run.

**Normative source.** [`ARCHITECTURE_REFACTOR.md`](ARCHITECTURE_REFACTOR.md) is the design document
and the only normative source. This file is *derived* from it: Part A quotes it verbatim with its
section, and adds nothing. Part B is project-specific traps that the design doc does **not** cover
but that are load-bearing in this tree. Part C is the violation ledger as of the date below.

**Keep it in sync.** When `ARCHITECTURE_REFACTOR.md` changes, update the affected Part A items in the
same commit. If an item here and the design doc disagree, the design doc wins and this file is the bug.

**Enforceability.** Some rules cannot be satisfied today because the machinery they require has not
landed (notably: `rmsnorm`/`rope`/`attention` have no op-table virtuals). Every such item is marked
**⚠ NOT ENFORCEABLE TODAY** and states what *is* required instead. Do not block a PR on the
unenforceable half.

> Anchors below were verified against the tree on **2026-07-28** (branch `gpu-support/docs-arch-sync`).
> Anchors are given as `file: symbol` first and `(line N)` second, so a `grep` relocates them when
> they drift.

---

## Part A — Rules derived from the design doc

### 1. Backend add-only

> [§1] *"a new HW must be **add-only** (new files, no edits to model/core/other-backend)"*

**Check:** does the diff touch `Applications/CausalLM/models/*.cpp`, `nntrainer/graph/network_graph.cpp`,
`nntrainer/layers/layer_node.cpp`, or another backend's context / compute-ops file?

**Verify:**
```sh
git diff --stat <base>..<branch> -- Applications/CausalLM/models nntrainer/graph \
  nntrainer/layers/layer_node.cpp 'nntrainer/*_context.*' nntrainer/cuda nntrainer/tensor/cl_operations
```
Any hit needs a one-line justification in the commit message (e.g. "new `registerLayerFactory` call
only", "debug log line only"). A hit that changes control flow shared by other backends is a violation.

**Scope:** this is the *backend* add-only rule (adding a new `Context`/`ComputeOps`/`MemAllocator`
triad). The much more common case — adding a GPU path for an *existing* op on an *existing* backend —
is governed by item 4, not by this one.

### 2. Nobody names a backend except the registry

> [§1] *"backend differences are expressed only as (a) an `op_table` virtual, (b) a `Context`
> capability/policy method, or (c) a `MemAllocator` property — and **nobody names a backend except
> the registry**"*

**Check:** does new code branch on a backend name (`getName() == "gpu"`, `causallm_engine() == "cuda"`,
`#ifdef ENABLE_CUDA`) *inside a Layer's forwarding/finalize logic*? Inside a
`*_context.cpp` / `*_compute_ops.cpp` / `*_mem_allocator.*` such branching is the whole point and is fine.

**Verify:**
```sh
grep -n 'getName() *==\|causallm_engine() *==\|#if.*ENABLE_CUDA\|#if.*ENABLE_OPENCL' \
  <changed layer .cpp files>
```
A Layer that string-matches a backend name to change *behavior* (as opposed to guarding a registration
call) is a violation. Live violations of this rule already exist — see Part C items 3 and 4.

### 3. Thin Layer + whole-op op table

> [§9 Decision #2 / §3 Decision A] *"A: thin Layer + whole-op op_table. The Layer owns
> structure/shape/weight-binding/orchestration; ComputeOps owns the whole-op kernel (one
> `ops->rmsnorm/rope/attention/fc(...)` per op, never per-element)."*

**Check:** for every new/changed Layer, does `forwarding()`/`incremental_forwarding()` call
`getComputeOps()->someWholeOp(...)` (or the app-side `getOps()->fc/swiglu/...`) for the math, with the
Layer body holding only shape/index/weight-binding logic — no element loops, no raw kernel launches?

**Verify:**
```sh
grep -n 'clEnqueueNDRangeKernel\|cudaLaunchKernel\|kai_run_\|clblast::' <changed layer .cpp>
grep -n 'getComputeOps()->\|getOps()->\|ct_data->getComputeOps()' <changed layer .cpp>
```
A kernel-launch hit outside a `*_compute_ops.cpp` / `*_kernels.cpp` / `*_cl_op.cpp` file is dispatch
leaking into the Layer.

**⚠ NOT ENFORCEABLE TODAY for `attention` / `rmsnorm` / `rope`.** Those three virtuals do not exist:
`nntrainer/tensor/cpu_backend/compute_ops.h` declares `geglu`/`swiglu`/`sigmoid_glu`/`sigmoid_add`/`fc`/
`apply_activation` (lines 393/402/411/420/440/461) and **no** `rmsnorm`, `rope` or `attention` — the doc
itself lists them under §9 "STILL ABSENT". Verify with:
```sh
grep -n 'virtual void rmsnorm\|virtual void rope\|virtual void attention' \
  nntrainer/tensor/cpu_backend/compute_ops.h   # expect: no output
```
**What is required instead:** structure the layer so the later migration is a mechanical swap — the
whole-op body lives in *one* private helper taking whole `Tensor`s, called once from `forwarding()`,
with no per-element math scattered through the Layer. Then adding the virtual is a rename.

**Positive precedent:** `nntrainer/layers/llm/geglu_layer.cpp` — `in1.getOps()->geglu(...)`, one line,
no `#ifdef`. Also `Applications/CausalLM/layers/moe_expert_ffn_gpu.h` (branch
`gpu-support/moe-expert-ffn`) — router logic in the Layer, GEMM/GLU via `getOps()->fc`/`getOps()->swiglu`.

### 4. No new per-backend Layer forks (T7 collapse)

> [§4] *"**Rule:** every `*_gpu` / `cl_layers/` / `cuda_layers/` fork **COLLAPSES** into one neutral
> layer that dispatches through `ComputeOpsExt`."*
> [§9 T7] *"collapse `*_cl`/`*_cuda` into thin neutral Layers (#2=A)"*

**Check:** does the PR add a *new pair* of classes (`XLayerCl` + `CudaXLayer`, or `x_gpu.h` beside a CPU
`x.h`) registering the **same type string** on different contexts with different C++ classes and
duplicated forward logic — instead of one class whose math goes through `getOps()`?

**Verify:**
```sh
# same type string defined by more than one class in the diff
grep -rn 'inline static const std::string type = "<T>"' <diff>
grep -rn 'const std::string getType() const override' -A2 <diff>
# any newly added per-backend layer class
grep -rln 'class .*LayerCl\b\|class Cuda.*Layer\b' <diff>
```
Any new match is presumptively a fork and must justify why it is not the T7 pattern.

**⚠ PARTIALLY UNENFORCEABLE for the norms and activations.** A full T7 collapse dispatches through an
op-table virtual, and the `rmsnorm` virtual does not exist yet (item 3). Until it lands you cannot
demand a fully collapsed RMSNorm/LayerNorm layer.
**What is required instead:** do not ADD new per-backend Layer subclasses beyond the ones already in
tree. A new backend's implementation of an existing op goes into that backend's `ComputeOps` (adding a
new virtual if needed), reusing the existing Layer class. This is exactly the rule the LayerNorm/GELU
branches broke — Part C item 1.

### 5. Register through the public facade, never a downcast

> [§9 Decisions made, #1 Layer promotion scope] *"KEEP layers still register via the public
> `registerLayerFactory` API and dispatch through the same op_table."*
> [§11.3.B] *"A public registration facade — never downcast... Add `virtual registerFactory` to
> `Context` base."*

**Check:** does a new layer register via `nntrainer::Engine::Global().registerLayerFactory(engine,
createLayer<X>)` — never `static_cast<ClContext*>(...)->registerFactory(...)`?

**Verify:**
```sh
grep -rn 'static_cast<.*Context *\*>.*registerFactory' --include=*.cpp --include=*.h .
```
**Expect zero hits.** Confirmed zero tree-wide on 2026-07-28; every call site
(`causal_lm.cpp`, `transformer.cpp`, per-model `registerCustomLayers()`) already goes through the facade.
This is a baseline to preserve, not to re-derive: the facade landed in commit `d5dce6c4b`
(`nntrainer/context.h: virtual registerLayerFactory` line 461; `nntrainer/engine.h: Engine::registerLayerFactory`
line 191; overrides `app_context.cpp:757`, `cl_context.cpp:693`, `cuda_context.cpp:304`).

### 6. New residency-plane HW: exactly one closed-enum edit, and only after `residencyEngine()`

> [§9 Decision #7 / §8 step 1] *"open the closed enum to registered-names NOW"* — *"a self-registering
> vendor Context resolves with no enum edit"*; the remaining closed-enum consumer is the layer-level
> `getComputeEngine` (`common.h:49`, `base_properties.h:816`).

**Check:** a genuinely new *residency plane* may add one value to `enum LayerComputeEngine`
(`api/ccapi/include/common.h:49`) and one string to `ComputeEngineTypeInfo::EnumStr[]`
(`nntrainer/utils/base_properties.h:816`). That is expected and documented — but it must be the **only**
shared edit, and a backend that reuses an existing plane needs neither.

**Verify:**
```sh
grep -n 'toLayerComputeEngine' -A20 nntrainer/layers/layer_node.cpp
```
Confirm the function still tries `ctx->residencyEngine()` FIRST and only falls back to the `EnumList`/
`EnumStr` loop for names with no registered context (`layer_node.cpp:143-168`, verified 2026-07-28). A PR
that adds `if (name == "exynos")` inside `toLayerComputeEngine` instead of overriding
`Context::residencyEngine()` (`nntrainer/context.h:505`) reintroduces the closed pattern.

**Worked precedent:** `HtpContext` registers as `"htp"` (`nntrainer/engine.cpp:131`) with **no** enum
edit — `LayerComputeEngine` is still `{CPU,GPU,QNN,CUDA}`.

### 7. Residency lives in the MemAllocator, not in a `Context::residencyFor`

> [§9 Decision #4] *"B: MemAllocator capability predicates OWN residency (no separate
> `Context::residencyFor` method)."*

**Check:** does new code add a `Context::residencyFor(...)`? It should express residency via
`MemAllocator::isHostAddressable()/isDeviceVisible()/isSVM()/needsRegister()` overrides instead.

**Verify:**
```sh
grep -rn 'residencyFor' <diff>   # expect zero
```
Confirmed absent tree-wide today. Note `§3`'s class diagram still *draws* `ContextCapsExt::residencyFor`
— it is annotated as rejected; Decision #4 is the authority.

### 8. One `Context::runDecode` hook, not a new `#if` in `neuralnet.cpp`

> [§9 Decision #5] *"ONE `Context::runDecode` hook, per-backend and performance/caps-driven... The mode
> is `ExecPlan.mode` resolved by `caps × ExecMode`... not one forced uniform behavior."*

**Check:** does a new backend override `Context::runDecode(from,to,walk,emb)` rather than duplicating the
decode loop inside `neuralnet.cpp` behind a new `#if ENABLE_<BACKEND>`?

**Verify:**
```sh
grep -n 'runDecode' nntrainer/models/neuralnet.cpp
```
The base must remain the single plain walk — `Context::runDecode` returning
`nn.incremental_forwarding(...)` (`nntrainer/models/neuralnet.cpp:653`; the doc's `neuralnet.cpp:618`
anchor has drifted). Backend-specific decode branching added directly in `neuralnet.cpp` is a violation.

### 9. Phase-0 gates before deleting any old path

> [§10 Phase 0] *"CPU byte-identical, `enable-opencl=false`/`enable-cuda=false` build green, 4-HW
> token-identical + TPS within noise before deleting any old path."*

**Check:** does the PR body show (a) CPU-only build+run byte-identical before/after, (b) a build with the
new capability's meson flag OFF still compiling and linking, (c) token-identity (not "looks right") on
every HW the change touches — *before* any old code path is deleted?

**Verify:** not greppable. Require an explicit gate table in the PR body mirroring the doc's own Phase-0
gate column; every cell needs evidence (log excerpt or CI link), not a claim.

### 10. No semantic back-edges: core must not know app conventions

> [§11.2] two named back-edges must be cleaned: (a) the CL FC layer branches on
> `context.getName()=="output_of_causallm"`, (b) the `g_argmax_requested` global driven by CausalLM.

**Check:** does any new file under `nntrainer/` (outside `Applications/CausalLM/`) branch on an
app-defined layer-name string, or expose a mutable global only the app sets?

**Verify:**
```sh
grep -rn '"output_of_causallm"\|getName() *== *"' nntrainer/layers nntrainer/tensor
grep -rn 'g_argmax_requested\|extern bool g_' nntrainer/
```
Both known instances are still present and unfixed (Part C items 3 and 4). New code must not add a third.

### 11. NPU: Mode-1 whole-graph is the default; Mode-2 does not replace it

> [§9 Decision #3 / §8 step 3] *"Mode-1 whole-graph offload is the default NPU mode... Mode-2
> (`QualcommComputeOps`/HexKL op-by-op) is deferred to production/perf validation, not blocked on the
> code."*

**Check:** for HTP/QNN-adjacent work, is a Mode-2 per-op `ComputeOps` subclass being added as the only
path, with no Mode-1 `OffloadNode`/whole-graph option existing or planned?

**Verify:**
```sh
grep -rn 'HtpComputeOps\|QualcommComputeOps' nntrainer/
```
`HtpComputeOps : public CpuComputeOps` (`nntrainer/tensor/htp_backend/htp_compute_ops.cpp:32`, the Mode-2
skeleton) must coexist with, not delete, the `QNNGraph` whole-graph fat node (Mode-1). Both stay
reachable via the same claim knob.

### 12. Fusion stays inference-gated and profitability-gated

> [§9 Decision #6 / T10] *"FusionRealizer... CPU gets fusion by default (cache-locality win, not
> accelerator-only)"* — as built: env-gated `NNTR_FUSE_ACT` + **inference-gated** (fused backward drops
> `act'`).

**Check:** does new fusion work preserve the inference-only gate (checks `training == false`), and stay
caps/profitability-gated rather than blindly always-on for GPU?

**Verify:**
```sh
grep -n 'NNTR_FUSE\|ExecutionMode::INFERENCE' nntrainer/compiler/fusion_realizer.cpp \
  nntrainer/models/neuralnet.cpp
```
Gate site: `if (exec_mode == ExecutionMode::INFERENCE) realizers.emplace_back(new FusionRealizer());`
(`nntrainer/models/neuralnet.cpp:233-234`, commit `bfc0f2f0b`). A pass that fires during training, or that claims
"done" while ignoring the caps-gating the doc itself flags as future work, is not done.

### 13. Keep the registration facade — do not regress

> [§11.3.B] *"A public registration facade — never downcast."*

Same mechanics as item 5; listed separately because §11 tracks it as a repo-split blocker (S1). **Status:
landed** (`d5dce6c4b`) — §11.5/§11.6 are annotated accordingly. The review value here is purely
"do not regress to a downcast".

---

## Part B — Project traps (not in the design doc, but load-bearing here)

### 14. Scratch tensors must come from `context.requestTensor` — a heap `Tensor` silently runs on CPU

**Check:** does a new Layer construct a bare `nntrainer::Tensor tmp(...)` for compute scratch instead of
requesting it through `InitLayerContext::requestTensor(...)` and fetching it from the run context?

**Verify:**
```sh
grep -n 'Tensor [A-Za-z_]\+(' <new layer .cpp>     # local constructions used as compute scratch
```
**Why it matters:** dispatch follows the tensor's attached `ContextData`, not the layer's engine. A heap
`Tensor` has none, so `RunLayerContext::getComputeOps()` (`nntrainer/layers/layer_context.h:942-943`)
returns `nullptr` and the caller falls back to the global table, and `Tensor::getOps()`
(`nntrainer/tensor/tensor.h:2102`) resolves "attached ContextData > the global CPU table". The binary-op
compatibility check (`Tensor::checkContextCompatibility`, `tensor.h:2135`) is documented as *permissive*
when either side has no ContextData. Net effect: the op runs on the CPU, correctly, silently, inside your
GPU layer. This is the most dangerous silent-miscompile trap in this codebase — it produces a perf
regression or a coherence bug, never a build failure.

### 15. A CUDA op that merely inherits `CpuComputeOps` is UNACCELERATED — say so

**Check:** for any op newly exercised on the `cuda` engine, is there a real `CudaComputeOps::op(...)`
override, or does it fall through to `CpuComputeOps::op(...)` (a host loop over UVM memory)?

**Verify:**
```sh
grep -n '<opName>' nntrainer/cuda/cuda_compute_ops.cpp
```
`CudaComputeOps : public CpuComputeOps` (`nntrainer/cuda/cuda_compute_ops.cpp:51`) — inheritance is
exactly why the gap is invisible. If there is no override, the changelog must say "runs on the CPU path
over UVM memory", not "GPU-accelerated". For a PR that claims CUDA support for several ops, put a
claims-vs-reality table in the body.

### 16. An unregistered type string THROWS — the failure mode differs per backend

**Check:** for a type registered on `gpu`/`cuda` but not `cpu` (or vice versa), does a build with that
backend off fail cleanly rather than segfault or silently no-op?

**Verify:** run the "build with the other backend off" smoke test. Two distinct failure modes, both real:
- **Type not registered on the context** → `createObject` throws `exception::not_supported`
  ("Key is not found for the object", `nntrainer/cl_context.h:181`+, `nntrainer/cuda_context.h:147`+;
  the CPU context throws `std::invalid_argument("cannot create unknown object")`,
  `nntrainer/app_context.h:295`). There is **no** silent CPU fallback for an unregistered type string.
- **Op not implemented on the backend** → depends on which base was inherited:
  `ClComputeOps : public ComputeOps` (`nntrainer/tensor/cl_operations/cl_compute_ops.cpp:43`) inherits
  bases that `throwNotImplemented` (`compute_ops.h`, `NI(op)` in `compute_ops.cpp:67`), so CL *throws*;
  `CudaComputeOps : public CpuComputeOps` *silently runs the host loop*. CL is caught by an exception,
  CUDA only by benchmarking.

### 17. "Registered" ≠ "dispatched" — and a `cuda_context.cpp`-only grep will miss app-side registrations

**Check:** finding `ct_engine.registerLayerFactory("cuda", createLayer<X>)` does **not** prove X runs.
Confirm (a) the env/build gate around the call is reachable in a real `NNTR_ENGINE=cuda` run, (b) no other
registration of the same type string on the same context shadows it, (c) the model actually places the
node on that engine (`engine=` property).

**Verify:** static grep answers "registered"; only a runtime trace answers "dispatched" — e.g. the
`[CUDA-DBG] CudaRMSNormLayer USED` fprintf at `nntrainer/layers/cuda_layers/cuda_rmsnorm_layer.cpp:74`
(under `NNTR_CUDA_DBG`). **And scope the grep to the whole tree**: layers may be registered from core
(`nntrainer/cuda_context.cpp`) *or* app-side (`Applications/CausalLM/models/*.cpp`) through the
`Engine::registerLayerFactory` facade. A grep restricted to `registerFactory` in `cuda_context.cpp`
structurally cannot see the app-side half and will report live layers as dead code — this exact error was
made about `CudaRMSNormLayer` (see Part C item 2).
```sh
grep -rn 'registerLayerFactory\|registerFactory' --include=*.cpp . | grep -i '<ClassName>'
```

### 18. `causallm_engine()` is a per-TU static — one process cannot run two engines

`static std::string causallm_engine()` lives in a header (`Applications/CausalLM/llm_util.hpp:94`), so
every translation unit gets its own copy, each memoizing the environment at first call. Code that assumes
a single process can drive two backends simultaneously (or that changing the env mid-run re-resolves) is
wrong. Registration gates keyed on `causallm_engine() == "cuda"` are therefore per-TU decisions — see the
gate at `Applications/CausalLM/models/transformer.cpp:758`.

---

## Part C — Known open violations (as of 2026-07-28)

Marked **[pre-existing]** (predates the GPU campaign / on trunk) or **[campaign]** (introduced by the
2026-07 GPU-support branch series).

1. **[campaign] LayerNorm and GELU use the per-backend Layer-fork pattern.**
   `LayerNormLayerCl` (`gpu-support/layernorm-gpu`, `fc8fedf0d`) + `CudaLayerNormLayer`
   (`gpu-support/cuda-layernorm`, `54e93110d`); `ActivationLayerCl` (`gpu-support/gelu-gpu`, `a64a8e20c`)
   + `CudaActivationLayer` (`gpu-support/cuda-gelu`, `49f569b01`). Each pair registers the same type
   string (`"layer_normalization"` / `"activation"`) on two contexts as two different classes with
   duplicated forward logic. Violates item 4.
   *Fix direction:* one neutral Layer per op dispatching `getOps()->layer_norm(...)` / `getOps()->gelu(...)`,
   modelled on `nntrainer/layers/llm/geglu_layer.{h,cpp}`; requires adding those two whole-op virtuals
   first. Not yet on trunk — these are unmerged sibling branches, so this is fork debt owed, not shipped debt.

2. **[pre-existing] RMSNorm is a live three-way fork under one type string — and `CudaRMSNormLayer` is
   NOT dead code.**
   `"rms_norm"` resolves to `causallm::RMSNormLayer` on `cpu`, `causallm::RMSNormLayerGPU` on `gpu`, and
   `nntrainer::CudaRMSNormLayer` on `cuda` — the last registered app-side at
   `Applications/CausalLM/models/transformer.cpp:766-767` inside `Transformer::registerCustomLayers()`
   (called from `transformer.cpp:292`), gated by `causallm_engine() == "cuda" || NNTR_CUDA_EAGER_CTX`.
   Separately, core `RMSNormLayerCl` registers the *different* string `"rmsnorm"` on the gpu context
   (`nntrainer/cl_context.cpp:316-318`).
   **Correction of record:** a 2026-07-28 coverage audit called `CudaRMSNormLayer` dead code ("compiled
   but never registered"); that audit grepped only `registerFactory` in `cuda_context.cpp` and so could
   not see the app-side facade call. The class is live. See Part B item 17.
   *Fix direction:* a real T7 migration (neutral RMSNorm Layer + `ops->rmsnorm(...)`), not a deletion.
   This is the template the LayerNorm/GELU forks copied — collapsing only the copies leaves the pattern
   discoverable.

3. **[pre-existing] `nntrainer/layers/fc_layer_cl.cpp:73`** —
   `if (skip_prefill && context.getName() == "output_of_causallm")`. Core CL FC layer branching on an
   app-defined layer name. Violates items 2 and 10; named by §11.2(a).
   *Fix direction:* replace with a generic layer role/property.

4. **[pre-existing] `nntrainer/tensor/cl_operations/blas_kernels.cpp`** — `g_argmax_requested` static
   global (declared :3500, set :3518, read :3716, :3875, :4047), set from CausalLM, read by core CL
   kernels. Same class as item 3; named by §11.2(b).
   *Fix direction:* make it a public API parameter or move ownership app-side.

5. **[pre-existing, structural] `attention` / `rmsnorm` / `rope` have no op-table virtuals.**
   `nntrainer/tensor/cpu_backend/compute_ops.h` declares none of them (only a comment mentions
   "attention output gate"). Every current RMSNorm/MHA-core implementation therefore *cannot* satisfy
   item 3 today. Accurately flagged by the doc's own §9 "STILL ABSENT" list.
   *Fix direction:* T7 completion. Until then, apply the interim rule stated in items 3 and 4.

6. **[pre-existing] Concat / Reshape / Transpose exist only as `*_cl`.**
   `nntrainer/layers/cl_layers/{concat_cl,reshape_cl,transpose_cl}.{h,cpp}` with no CUDA equivalent
   (`nntrainer/layers/cuda_layers/` holds only `cuda_rmsnorm_layer.*`). Not a rule violation — add-only
   does not require every backend to implement every op — but a live asymmetry.
   *Fix direction:* a CUDA contribution touching these ops should close the gap through `ComputeOps`
   rather than fork a `CudaConcatLayer`.

7. **[resolved doc staleness, listed for traceability] §11.5 blocker 1 / §11.6 S1 described the
   registration facade as unresolved.** It landed in `d5dce6c4b`; a tree-wide
   `static_cast<*Context*>->registerFactory` grep returns zero. `ARCHITECTURE_REFACTOR.md` has been
   annotated accordingly. Recorded here because a reviewer working from an older copy of the doc may
   still block a PR over an already-retired pattern.

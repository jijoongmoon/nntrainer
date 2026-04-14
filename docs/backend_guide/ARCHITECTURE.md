# NNTrainer Compute Backend Architecture

## Overview

NNTrainer supports multiple compute backends (CPU, GPU, NPU) through a
layered dispatch architecture. Each backend provides its own implementation
of tensor operations, and the framework routes operations to the correct
backend at runtime based on the layer's `compute_engine` property.

```
┌─────────────────────────────────────────────────────────────┐
│                      User Model (.ini / C++ API)            │
│  [layer]                                                    │
│  type = fully_connected                                     │
│  compute_engine = gpu    ← selects backend                  │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                         Engine                              │
│  Routes layer creation to the correct Context               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ "cpu"    │ │ "gpu"    │ │ "qnn"    │ │ "cuda"   │       │
│  │AppContext│ │ClContext │ │QNNContext│ │(future)  │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
└───────┼─────────────┼────────────┼────────────┼─────────────┘
        │             │            │            │
┌───────▼─────────────▼────────────▼────────────▼─────────────┐
│                      Context (base class)                   │
│  - createLayerObject()     ← layer factory                  │
│  - getContextData()        ← shared backend data            │
│  - getName()               ← "cpu", "gpu", "qnn"           │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                    ContextData                              │
│  Extensible container shared from Context to layers.        │
│  Subclassable for vendor-specific data.                     │
│                                                             │
│  ┌─────────────────────────────────────────────────┐        │
│  │ Base members (all backends):                    │        │
│  │   ComputeOps*    ← per-op function pointers     │        │
│  │   MemAllocator*  ← memory management            │        │
│  ├─────────────────────────────────────────────────┤        │
│  │ Subclass data (vendor-specific):                │        │
│  │   QNNBackendVar: QNN graph handles, sessions    │        │
│  │   (future) CudaVar: CUDA streams, device mem    │        │
│  └─────────────────────────────────────────────────┘        │
│                                                             │
│  Type-safe access:                                          │
│    ctx->as<QNNBackendVar>()  → safe downcast                │
│    ctx->getType()            → "cpu" / "gpu" / "qnn"        │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                   ComputeOps Table                          │
│  Function pointer struct for ALL tensor operations.         │
│  Same struct for every backend — different implementations. │
│                                                             │
│  struct ComputeOps {                                        │
│    // BLAS                                                  │
│    void (*sgemm_fp32)(...);    // ARM: neon, x86: cblas     │
│    void (*sgemv_fp32)(...);    // GPU: sgemm_cl wrapper     │
│    float (*sdot_fp32)(...);    // NPU: vendor SDK           │
│    ...                                                      │
│    // Element-wise                                          │
│    void (*ele_add_fp32)(...);                               │
│    ...                                                      │
│    // Activation                                            │
│    void (*swiglu_fp32)(...);                                │
│    ...                                                      │
│    // Quantized GEMM — GGML block-quantized (Q4_0/K, Q6_K)  │
│    void (*gemm_q4_0_fp32)(...);                             │
│    void (*gemm_q4_K_fp32)(...);                             │
│    void (*gemm_q6_K_fp32)(...);                             │
│    size_t (*quantize_q4_0)(...);                            │
│    void (*dequantize_row_q4_0)(...);                        │
│    // Quantized GEMM — channel-wise int4 (qsi4cxp/QINT4)    │
│    //   ARM: KleidiAI NEON/SME, x86: AVX2, fallback: C ref  │
│    void (*gemm_qsi4cxp_fp32)(M, N, K, act, data, scales, out);│
│    ...                                                      │
│  };                                                         │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│              RunLayerContext                                 │
│  Runtime context passed to Layer::forwarding().             │
│                                                             │
│  Provides access to:                                        │
│    getInput(idx)       → Tensor (input data)                │
│    getOutput(idx)      → Tensor (output data)               │
│    getWeight(idx)      → Tensor (weights)                   │
│    getComputeOps()     → ComputeOps* (from ContextData)     │
│    getContextData()    → ContextData* (vendor data access)  │
│                                                             │
│  getComputeOps() delegates to ContextData:                  │
│    return ct_data ? ct_data->getComputeOps() : nullptr;     │
│                                                             │
│  ContextData is the SINGLE source of truth for backend data.│
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                  Tensor Operations (B-2 Pattern)            │
│                                                             │
│  ALL tensor ops accept optional ComputeOps* parameter:      │
│                                                             │
│  tensor.dot(input, output, trans, trans_in, beta, ops)      │
│  tensor.add(m, output, alpha, ops)                          │
│  tensor.multiply(m, output, beta, ops)                      │
│  tensor.l2norm(ops)                                         │
│  ... (23 files, all tensor types)                           │
│                                                             │
│  When ops != nullptr:                                       │
│    → Uses provided ops (GPU/NPU backend)                    │
│                                                             │
│  When ops == nullptr (default):                             │
│    → Falls back to getComputeOps() (global CPU ops)         │
│    → Backward compatible with existing code                 │
│                                                             │
│  Implementation pattern:                                    │
│    auto *o = ops ? ops : getComputeOps();                   │
│    o->sgemm_fp32(M, N, K, ...);                             │
│                                                             │
│  Thread-safe: ops flows as parameter, not global state.     │
│                                                             │
│  Example dispatch:                                          │
│    ┌─────────────┬─────────────┬─────────────┐              │
│    │ ARM backend │ GPU backend │ NPU backend │              │
│    │ neon::sgemm │ sgemm_cl   │ npu_sgemm   │              │
│    └─────────────┴─────────────┴─────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

---

## Initialization Flow

### 1. Engine Startup

```
Engine::initialize()
  └── Engine::add_default_object()
        │
        ├── AppContext::Global()                       ← CPU backend
        │     └── AppContext::initialize()
        │           ├── init_backend()                 ← sets g_compute_ops
        │           ├── cd->setComputeOps(g_compute_ops)
        │           ├── cd->setMemAllocator(...)
        │           ├── add_default_object()           ← 80+ layer factories
        │           └── add_extension_object()         ← plugins
        │
        ├── registerContext("cpu", &app_context)
        │
        ├── ClContext::Global()                        ← GPU (if enabled)
        │     └── ClContext::initialize()
        │           ├── clInit()                       ← OpenCL init
        │           ├── cd->setComputeOps(opencl_ops)  ← GPU ops table
        │           └── add_default_object()           ← 7 GPU layers
        │
        ├── registerContext("gpu", &cl_context)
        │
        └── (QNN/CUDA via plugin .so loading)
              └── dlopen("qnn_context.so")
                    → QNNContext::Global()
                    → QNNContext::initialize()
                          ├── init_backend()              ← CPU fallback
                          ├── cd->setComputeOps(g_compute_ops)
                          ├── QNN SDK init
                          └── register QNN layers
```

### 2. Layer Creation

```
Model config: compute_engine = gpu
  │
  ▼
Engine::createLayerObject("fully_connected", {"compute_engine=gpu"})
  │
  ├── parseComputeEngine(props) → "gpu"
  ├── getRegisteredContext("gpu") → ClContext*
  └── ClContext::createLayerObject("fully_connected")
        └── FullyConnectedLayerCl created
```

### 3. Graph Finalization (Context → RunLayerContext)

```
NetworkGraph::finalizeContext()
  │
  for each LayerNode:
  │
  ├── context = Engine::getRegisteredContext(lnode->getComputeEngineType())
  │     e.g., "cpu" → AppContext, "gpu" → ClContext, "qnn" → QNNContext
  │
  ├── ct_data = context->getContextData()
  │     Contains: ComputeOps* + MemAllocator* + vendor data (subclass)
  │
  └── lnode->configureRunContext(weights, inputs, outputs, tensors,
                                 loss_scale, ct_data)
        │
        └── RunLayerContext created
              ├── shared_ptr<ContextData> ct_data  ← from Context
              ├── getComputeOps() → ct_data->getComputeOps()
              └── getContextData() → ct_data (for vendor data)
```

### 4. Layer Execution

```
Layer::forwarding(RunLayerContext &context)
  │
  ├── CPU layer (FullyConnectedLayer):
  │     auto *ops = context.getComputeOps();          // cpu_ops
  │     input.dot(weight, output, false, false, 0, ops);
  │       → FloatTensor::dotFloat(... ops)
  │         → auto *o = ops ? ops : getComputeOps();
  │         → o->sgemm_fp32(...)
  │           → arm_backend::sgemm_fp32() → neon/cblas
  │
  ├── GPU layer (FullyConnectedLayerCl):
  │     auto *ops = context.getComputeOps();          // opencl_ops
  │     input.dot(weight, output, false, false, 0, ops);
  │       → FloatTensor::dotFloat(... ops)
  │         → auto *o = ops;                           // GPU ops used!
  │         → o->sgemm_fp32(...)
  │           → cl_sgemm_fp32() → sgemm_cl (OpenCL kernel)
  │
  └── NPU layer (QNNGraph):
        // Op-level: for preprocessing
        auto *ops = context.getComputeOps();          // cpu fallback
        input.multiply_i(scale, ops);
        
        // Graph-level: whole subgraph on NPU
        auto *qnn = context.getContextData()->as<QNNBackendVar>();
        qnn->getVar()->executeGraph(...)
          → QNN runtime (HTP hardware)
```

### 5. Non-layer Code (Model, Preprocessing)

Code outside layers (e.g., `neuralnet.cpp`) that needs backend ops
uses `getComputeOps()` directly — the global fallback (CPU ops):

```cpp
// neuralnet.cpp — output data conversion
getComputeOps()->scopy_fp16_to_fp32(buf_size, fp16_data, 1, fp32_data, 1);
```

This always uses CPU ops, which is correct because model-level data
handling (output copying, loss computation) happens on CPU regardless
of which backend individual layers use.

---

## Two Dispatch Models

### Op-level Dispatch (ComputeOps)

Individual tensor operations dispatched through function pointer table.

```
Layer → tensor.dot(input, output, ..., ops)
  → ops->sgemm_fp32(...)
    → ARM neon / x86 AVX / OpenCL kernel / NPU SDK
```

- **Who provides it**: Every Context (CPU, GPU, NPU must all set ComputeOps)
- **Who uses it**: All layers via tensor operations
- **Thread-safe**: ops passed as parameter, no global state dependency
- **Fallback**: NPU contexts should set CPU ops as fallback

### Graph-level Dispatch (ContextData subclass)

Entire subgraph delegated to accelerator runtime.

```
Layer → context.getContextData()->as<QNNBackendVar>()
  → qnn->getVar()->executeGraph(...)
    → QNN executes entire transformer block on HTP
```

- **Who provides it**: NPU/accelerator contexts only
- **Who uses it**: NPU-specific layers (QNNGraph, TFLiteDelegate)
- **Access**: Type-safe via `as<T>()` (returns nullptr if wrong type)

### Both models coexist in one model:

```
Embedding    → CPU  (op-level:  tensor.dot with cpu_ops)
Transformer  → QNN  (graph-level: QNN executes on HTP)
LM Head      → CPU  (op-level:  tensor.dot with cpu_ops)
```

---

## Ops Table Per-Architecture

Each CPU architecture provides its own ops table in a dedicated file.
The vendor namespace makes the binding explicit.

```
arm/arm_ops_table.cpp:
  #include <arm_compute_backend.h>              ← ARM header
  namespace arm_backend {
    static void sgemm_fp32(...) {
      nntrainer::sgemm(...);                    ← ARM implementation
    }
  }
  .sgemm_fp32 = arm_backend::sgemm_fp32,       ← clearly ARM

x86/x86_ops_table.cpp:
  #include <x86_compute_backend.h>              ← x86 header
  namespace x86_backend {
    static void sgemm_fp32(...) {
      nntrainer::sgemm(...);                    ← x86 implementation
    }
  }
  .sgemm_fp32 = x86_backend::sgemm_fp32,       ← clearly x86

cl_operations/cl_compute_ops.cpp:
  static void cl_sgemm_fp32(...) {
    sgemm_cl(TransA, TransB, ...);              ← OpenCL kernel
  }
  .sgemm_fp32 = cl_sgemm_fp32,                 ← clearly GPU
```

All vendors follow the same pattern:
1. Define wrapper functions in a vendor namespace
2. Map wrappers to the ComputeOps table
3. Set the table on ContextData during Context::initialize()

---

## Quantization Paths

nntrainer supports two distinct families of int4/int8 quantized weights.
They coexist — choose per-tensor based on target platform and deployment.

### Lane A: Channel-wise int4 (QINT4 / Int4QTensor)

**Format**: one fp32 scale per output channel, symmetric signed int4.

**On-disk layout** (matches `Int4QTensor::allocate()` + KleidiAI
`qai8dxp_qsi4cxp_unpacked` expectation):
```
[2B QScheme header] [K × ceil(N/2) packed nibbles, kxn] [N × fp32 scales]
```

**Nibble packing** (KleidiAI kxn convention):
- Even `n_idx` → LOW nibble (bits 0–3)
- Odd `n_idx` → HIGH nibble (bits 4–7)
- Stored as offset-binary: `stored = real_int4 + 8` (range 0..15)

**Runtime dispatch** (via `ComputeOps::gemm_qsi4cxp_fp32`):

| Platform | Implementation | Source file |
|----------|----------------|-------------|
| ARM | KleidiAI `qai8dxp_qsi4cxp` (NEON / SME / SME2) | `arm/kleidiai_interface_qai8dxp_qsi4cxp.cpp` |
| x86 | AVX2 direct kxn GEMM | `x86/x86_qsi4cxp.cpp` |
| fallback | pure C reference | `fallback/fallback_kleidiai.cpp` |
| NPU (QNN) | `AXIS_SCALE_OFFSET` encoding, axis=1, N scales | QNN runtime (HTP) |
| GPU (Adreno) | OpenCL custom kernel (planned) | `cl_operations/` |

**Key invariant**: the safetensors on-disk bytes can be consumed by all
4 backends without any layout translation. One weight file → every
target.

**Variant selection** (inside `nntr_kai_gemm_qai8dxp_qsi4cxp_rtp`):
```c
compile-time: ENABLE_SME2 / ENABLE_SME / neither
 ×
runtime:      getauxval(AT_HWCAP2) & HWCAP2_SME[2]
 ↓
select:  SME2 mopa/sdot  →  SME mopa/dot  →  NEON i8mm/dotprod
         (idx 10/11)        (idx 8/9)       (idx 0/5)
```
Runtime check is essential: a binary compiled with `ENABLE_SME2=1`
will SIGILL on SME instructions if the OS kernel hasn't enabled
userspace SME access (requires Linux 6.3+). The selection cascade
falls back gracefully.

**RHS pack caching** (critical for LLM decode perf):
The KleidiAI `_rtp` interface packs the weight into its native tile
layout on every call. Since FC weights are static in inference,
`nntr_kai_gemm_qai8dxp_qsi4cxp_rtp` caches the packed RHS by
`(rhs_ptr, variant_idx, transB)` key. First call packs; subsequent
calls reuse the cached buffer. This cuts per-call CPU time roughly
in half on LLM decode.

### Lane B: Block-quantized (Q4_0 / Q4_K / Q6_K / Q1_0)

**Format**: GGML-style super-blocks with embedded scales.

| Type | Block size | Scale storage | Consumer |
|------|-----------|---------------|----------|
| Q4_0 | 32 elements | fp16 per block | GGML gemm_q4_0 |
| Q4_K | 256 (8×32 sub) | fp16 super + uint8 sub | GGML gemm_q4_K |
| Q6_K | 256 | fp16 super | GGML gemm_q6_K |
| Q1_0 | 128 | nntr_ggml_impl custom | ggml_interface |

**Dispatch**: `ComputeOps::gemm_q4_0_fp32`, `gemm_q4_K_fp32`,
`gemm_q6_K_fp32`. The underlying kernels come from the vendored
`nntr_ggml_impl/` (AVX2 for x86, NEON/SVE for ARM).

**When to use which lane**:
- Deploying to Galaxy S26 / ARM with KleidiAI SME → Lane A (QINT4)
- NPU (QNN HTP) deployment → Lane A (AXIS_SCALE_OFFSET)
- CPU-only x86 server with existing GGML pipeline → Lane B (Q4_0)
- Mixed CPU+GPU with shared weight format → Lane A (wire-format
  compatible across all backends)

---

## Android Build Pipeline (ARM Feature Flags)

```
./tools/package_android.sh . --arm-arch=armv9.2-a
  │
  ├── Load tools/cross/android_armv9-2-a.json:
  │     arm_march  = "armv9.2-a+fp16+sve2+sme"
  │     enable_sme = true,  enable_sve2 = true
  │     enable_sme2 = false  (Oryon cores in S26 Ultra don't have SME2)
  │
  ├── meson configure with:
  │     -Darm-arch=armv9.2-a
  │     -Darm-march=-march=armv9.2-a+fp16+sve2+sme
  │     -Denable-fp16=true -Denable-sme=true -Denable-sve2=true
  │
  ├── meson.build (android branch): sets compile defines
  │     extra_defines += -DENABLE_FP16=1
  │                   += -DENABLE_SME=1        ← conditional
  │                   += -DENABLE_SVE2=1       ← conditional
  │
  ├── jni/meson.build: substitutes into Android.mk.in
  │     MESON_CFLAGS = flags + extra_defines
  │     MESON_APP_ABI = arm64-v8a  (from arm-arch)
  │
  └── ndk-build (NDK Clang 18):
        compiles kai/ SME/SME2 kernel sources (needs Clang 16+)
        kleidiai_interface_qai8dxp_qsi4cxp.cpp picks up
          #ifdef ENABLE_SME → registers SME kernel variants
```

The `subdir('kai')` and `kleidiai_interface_qai8dxp_qsi4cxp.cpp` are
compiled UNCONDITIONALLY on ARM (outside the `enable-fp16` guard)
because the KleidiAI int4 kernel is pure FP32→int8→int4→FP32 and has
no FP16 dependency. SME/SME2 kernel *files* are still gated on
`enable-sme`/`enable-sme2` because they use C99 `restrict` (needs
Clang 16+ or GCC 14+).

---

## ContextData Extension (Vendor Data)

ContextData is the extensible container for backend-specific data.
Vendors subclass it to add hardware handles, sessions, etc.

```cpp
// Base: ComputeOps + MemAllocator (all backends)
class ContextData {
  ComputeOps *compute_ops;
  MemAllocator *mem_allocator;

  // Type-safe downcast
  template<typename T> T* as() { return dynamic_cast<T*>(this); }
  virtual const char* getType() { return "cpu"; }
};

// QNN vendor extension
class QNNBackendVar : public ContextData {
  const char* getType() override { return "qnn"; }
  shared_ptr<QNNVar> data;  // graph handles, sessions, RPC memory
};

// Usage in QNN layer (safe)
auto *qnn = context.getContextData()->as<QNNBackendVar>();
if (!qnn) throw "not a QNN context";
qnn->getVar()->executeGraph(...);
```

---

## Tensor Ops Dispatch (B-2 Pattern)

All tensor operations accept an optional `ComputeOps *ops` parameter.
This is the mechanism by which GPU/NPU layers route operations to their backend.

### API Design

```cpp
// Default: ops=nullptr → global CPU fallback (backward compatible)
Tensor &dot(Tensor const &input, Tensor &output,
            bool trans = false, bool trans_in = false,
            float beta = 0.0f,
            ComputeOps *ops = nullptr) const;

Tensor &add(Tensor const &m, Tensor &output,
            float const alpha = 1,
            ComputeOps *ops = nullptr) const;

// ... same for multiply, divide, transpose, l2norm, etc.
```

### Implementation Pattern

```cpp
Tensor &FloatTensor::dotFloat(..., ComputeOps *ops) const {
  auto *o = ops ? ops : getComputeOps();  // use provided or fallback
  o->sgemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
```

### Usage in Layers

```cpp
// CPU layer — no change needed (ops=nullptr → CPU fallback)
void FullyConnectedLayer::forwarding(RunLayerContext &ctx) {
  input.dot(weight, output);  // uses global CPU ops
}

// GPU layer — pass context ops explicitly
void FullyConnectedLayerGpu::forwarding(RunLayerContext &ctx) {
  auto *ops = ctx.getComputeOps();
  input.dot(weight, output, false, false, 0.0f, ops);
}
```

### Thread Safety

```cpp
// Safe in parallel_for: ops captured by lambda
ComputeOps *ops = ctx.getComputeOps();
ThreadManager::Global().parallel_for(0, N, [=](size_t i) {
  partial.dot(input, output, false, false, 0.0f, ops);
});
```

### Why This Approach

| Alternative | Issue |
|-------------|-------|
| Tensor stores ops | Shared tensors between CPU/GPU — whose ops? |
| Thread-local | Worker threads in parallel_for don't inherit TLS |
| RAII scope | Not thread-safe for parallel layer execution |
| **Explicit parameter** | **Safe in all cases** |

---

## Plugin .so Loading (Dynamic Backend Registration)

Backends like QNN are built as separate `.so` shared libraries and
loaded at runtime. This decouples vendor SDK dependencies from the
main nntrainer library.

### Build Structure

```
libnntrainer.so          ← Core library (no QNN/CUDA dependency)
  ├── Engine, Context, ComputeOps, Tensor
  └── CPU backends (ARM/x86/fallback)

libqnn_context.so        ← QNN plugin (separate build, links QNN SDK)
  ├── qnn_context.cpp    ← built with -DPLUGGABLE
  ├── qnn_rpc_manager.cpp
  └── QNN SDK libraries
  Exports: ml_train_context_pluggable

libcuda_context.so       ← CUDA plugin (future, links cuBLAS)
  └── ...
```

### Loading Flow

```
Engine::add_default_object()
  │
  ├── registerContext("cpu", &AppContext::Global())   ← built-in
  ├── registerContext("gpu", &ClContext::Global())    ← built-in (if OpenCL)
  │
  └── registerContext("libqnn_context.so", "")        ← plugin .so
        │
        ├── dlopen("libqnn_context.so", RTLD_LAZY | RTLD_LOCAL)
        ├── dlsym("ml_train_context_pluggable")
        ├── pluggable->createfunc() → QNNContext::Global()
        │     └── QNNContext::initialize()
        │           ├── init_backend()              ← from libnntrainer.so
        │           ├── cd->setComputeOps(...)      ← CPU fallback or custom
        │           ├── QNN SDK init (vendor-specific)
        │           └── register QNN layers (QNNGraph, QNNLinear)
        └── registerContext("qnn", context)
```

### Plugin Entry Point

```cpp
// In qnn_context.cpp (built with -DPLUGGABLE):
#ifdef PLUGGABLE
extern "C" {
  nntrainer::ContextPluggable ml_train_context_pluggable{
    create_qnn_context, destroy_qnn_context
  };
}
#endif
```

### ComputeOps Override Strategy

**Strategy 1: Full vendor ops table**
```cpp
static ComputeOps qnn_ops = { .sgemm_fp32 = qnn_sgemm, ... };
cd->setComputeOps(&qnn_ops);
```

**Strategy 2: Copy CPU ops, override only accelerated ones** (recommended)
```cpp
static ComputeOps qnn_ops = *g_compute_ops;  // CPU base
qnn_ops.sgemm_fp32 = qnn_htp_sgemm;          // GEMM → QNN
// ele_add, softmax stay as CPU NEON/AVX fallback
cd->setComputeOps(&qnn_ops);
```

### Symbol Resolution Across .so

| Symbol | Location | Plugin Access |
|--------|----------|---------------|
| `g_compute_ops` | libnntrainer.so | ✅ dynamic linking |
| `init_backend()` | libnntrainer.so | ✅ dynamic linking |
| `getComputeOps()` | compute_ops.h (inline) | ✅ compiled into plugin |
| `as<T>()` (dynamic_cast) | RTTI | ✅ works across .so |

---

## Key Files

| File | Purpose |
|------|---------|
| `engine.h/cpp` | Backend router, context registration, plugin loading |
| `context.h` | Context base class, factory methods |
| `context_data.h` | ContextData (ComputeOps + MemAllocator + as<T>() + getType()) |
| `compute_ops.h` | ComputeOps struct, getComputeOps(), init_backend() |
| `compute_ops.cpp` | g_compute_ops global, ensureComputeOps() |
| `arm/arm_ops_table.cpp` | ARM ops table (arm_backend:: wrappers) |
| `x86/x86_ops_table.cpp` | x86 ops table (x86_backend:: wrappers) |
| `fallback/fallback_ops_table.cpp` | Fallback ops table |
| `cl_operations/cl_compute_ops.cpp` | OpenCL ops table |
| `arm/kleidiai_interface_qai8dxp_qsi4cxp.cpp` | KleidiAI channel-wise int4 driver (variant selection, RHS pack cache, NEON/SME/SME2 dispatch) |
| `arm/kai/` | KleidiAI upstream kernels (NEON dotprod/i8mm, SME, SME2) |
| `x86/x86_qsi4cxp.cpp` | x86 AVX2 channel-wise int4 GEMV/GEMM (reads kxn layout directly) |
| `fallback/fallback_kleidiai.cpp` | Pure C reference for qsi4cxp (always compiled) |
| `fallback/fallback_kleidiai_dispatch.cpp` | qsi4cxp wrapper when ENABLE_FP16 is off |
| `tensor/int4_tensor.h/cpp` | Int4QTensor — canonical QINT4 layout, scale_size=width(), PER_CHANNEL_AFFINE default |
| `jni/meson.build` | Android.mk.in template filler (MESON_CFLAGS, MESON_APP_ABI from arm-arch) |
| `jni/Android.mk.in` | NDK build template, `@MESON_*@` variables filled by configure_file |
| `tools/cross/android_*.json` | Per-arch cross-build config (arm_march, SME/SVE flags) |
| `tools/package_android.sh` | Android wrapper: loads JSON, invokes meson + ndk-build |
| `layer_context.h` | RunLayerContext (getComputeOps, getContextData) |
| `layer_node.cpp` | configureRunContext (ContextData → RunLayerContext) |
| `network_graph.cpp` | Graph finalization (Context → ContextData → LayerNode) |
| `app_context.h/cpp` | CPU backend context |
| `cl_context.h/cpp` | GPU backend context |
| `qnn_context.h/cpp` | QNN backend context (built as libqnn_context.so plugin) |
| `neuralnet.cpp` | Model execution (uses getComputeOps() for data conversion) |
| `tensor.h/cpp` | Tensor public API (ops parameter on all operations) |
| `tensor_base.h/cpp` | TensorBase virtual interface (ops parameter) |
| `float_tensor.h/cpp` | FP32 tensor implementation |
| `half_tensor.h/cpp` | FP16 tensor implementation |

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
│  compute_engine = cpu    ← selects backend                  │
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
│    void (*ele_mul_fp32)(...);                               │
│    ...                                                      │
│    // Activation                                            │
│    void (*swiglu_fp32)(...);                                │
│    void (*softmax_fp32)(...);                               │
│    ...                                                      │
│    // Quantized GEMM                                        │
│    void (*gemm_q4_0_fp32)(...);                             │
│    void (*quantize_q4_0)(...);                              │
│    ...                                                      │
│    // GPU-accelerated (nullable)                            │
│    void (*gemm_q4_0_batch_fp32)(...);  // nullptr on CPU    │
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
│    getComputeOps()     → ComputeOps* (ops dispatch)         │
│    getContextData()    → ContextData* (vendor data)         │
│                                                             │
│  ComputeOps comes from ContextData, which comes from        │
│  the Context that created this layer.                       │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                  Tensor Operations                          │
│                                                             │
│  tensor.dot(input, output)                                  │
│    → FloatTensor::dotFloat()                                │
│      → getComputeOps()->sgemm_fp32(...)                     │
│        ↓                                                    │
│    ┌─────────────┬─────────────┬─────────────┐              │
│    │ ARM backend │ GPU backend │ NPU backend │              │
│    │ neon::sgemm │ sgemm_cl   │ npu_sgemm   │              │
│    └─────────────┴─────────────┴─────────────┘              │
│                                                             │
│  Same tensor.dot() call dispatches to different HW          │
│  based on which ops table is set in the Context.            │
└─────────────────────────────────────────────────────────────┘
```

---

## Initialization Flow

### 1. Engine Startup

```
Engine::initialize()
  └── Engine::add_default_object()
        ├── AppContext::Global()                     ← CPU backend
        │     └── AppContext::initialize()
        │           ├── init_backend()               ← sets g_compute_ops
        │           ├── setComputeOps(g_compute_ops) ← ops table on ContextData
        │           ├── add_default_object()         ← 80+ layer factories
        │           └── add_extension_object()       ← plugins
        │
        ├── registerContext("cpu", &app_context)
        │
        ├── ClContext::Global()                      ← GPU backend (if enabled)
        │     └── ClContext::initialize()
        │           ├── clInit()                     ← OpenCL init
        │           ├── setComputeOps(opencl_ops)    ← GPU ops table
        │           └── add_default_object()         ← 7 GPU layers
        │
        ├── registerContext("gpu", &cl_context)
        │
        └── (plugin .so loading for QNN, CUDA, etc.)
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
  ├── for each LayerNode:
  │     ├── context = Engine::getRegisteredContext(lnode->getComputeEngineType())
  │     ├── ct_data = context->getContextData()
  │     │     └── Contains: ComputeOps* + MemAllocator* + vendor data
  │     │
  │     └── lnode->configureRunContext(weights, inputs, outputs, tensors,
  │                                    loss_scale, ct_data)
  │           └── RunLayerContext created with ct_data
  │                 ├── getComputeOps() → ct_data->getComputeOps()
  │                 └── getContextData() → ct_data (for vendor access)
```

### 4. Layer Execution

```
Layer::forwarding(RunLayerContext &context)
  │
  ├── CPU layer (FullyConnectedLayer):
  │     output.dot(input, weight)
  │       → FloatTensor::dotFloat()
  │         → getComputeOps()->sgemm_fp32(...)
  │           → arm_backend::sgemm_fp32() → neon::sgemm / cblas
  │
  ├── GPU layer (FullyConnectedLayerCl):
  │     output.dot(input, weight)
  │       → FloatTensor::dotFloat()
  │         → getComputeOps()->sgemm_fp32(...)
  │           → cl_sgemm_fp32() → sgemm_cl (OpenCL kernel)
  │
  └── NPU layer (QNNGraph):
        auto *qnn = context.getContextData()->as<QNNBackendVar>();
        qnn->getVar()->executeGraph(...)
          → QNN runtime (whole graph execution on NPU)
```

---

## Two Dispatch Models

NNTrainer supports two complementary dispatch models:

### Op-level Dispatch (ComputeOps)

Individual tensor operations dispatched through function pointer table.

- **Who provides it**: Every Context (CPU, GPU, NPU must all set ComputeOps)
- **Who uses it**: All layers that call tensor operations (dot, add, multiply)
- **Fallback**: NPU contexts should set CPU ops as fallback for non-accelerated ops
- **Example**: `getComputeOps()->sgemm_fp32(M, N, K, ...)`

### Graph-level Dispatch (ContextData subclass)

Entire subgraph delegated to accelerator runtime.

- **Who provides it**: Only NPU/accelerator contexts (via ContextData subclass)
- **Who uses it**: NPU-specific layers (QNNGraph, TFLiteDelegate)
- **Access**: `context.getContextData()->as<QNNBackendVar>()->getVar()`
- **Example**: QNN executes an entire transformer block on HTP

Both models coexist in the same model. A model can have:
- Embedding layer → CPU (op-level dispatch)
- Transformer block → QNN (graph-level dispatch)
- Output layer → CPU (op-level dispatch)

---

## Ops Table Per-Architecture

Each CPU architecture provides its own ops table in a dedicated file.
The table references functions from that architecture's backend.

```
arm/arm_ops_table.cpp:
  #include <arm_compute_backend.h>         ← ARM header
  namespace arm_backend {
    static void sgemm_fp32(...) {
      nntrainer::sgemm(...);               ← ARM implementation
    }
  }
  .sgemm_fp32 = arm_backend::sgemm_fp32,  ← clearly ARM

x86/x86_ops_table.cpp:
  #include <x86_compute_backend.h>         ← x86 header
  namespace x86_backend {
    static void sgemm_fp32(...) {
      nntrainer::sgemm(...);               ← x86 implementation
    }
  }
  .sgemm_fp32 = x86_backend::sgemm_fp32,  ← clearly x86

cl_operations/cl_compute_ops.cpp:
  static void cl_sgemm_fp32(...) {
    sgemm_cl(TransA, TransB, ...);         ← OpenCL kernel
  }
  .sgemm_fp32 = cl_sgemm_fp32,            ← clearly GPU
```

All vendors follow the same pattern:
1. Define wrapper functions in a vendor namespace
2. Map wrappers to the ComputeOps table
3. Set the table on ContextData during Context::initialize()

---

## ContextData Extension (QNN Example)

```cpp
// Vendor subclasses ContextData to add hardware-specific data
class QNNBackendVar : public ContextData {
public:
  const char *getType() const override { return "qnn"; }
  std::shared_ptr<QNNVar> &getVar() { return data; }
private:
  std::shared_ptr<QNNVar> data;  // QNN graph handles, sessions, etc.
};

// Context creates subclassed ContextData
QNNContext() : Context(std::make_shared<QNNBackendVar>()) {}

// QNN layer accesses vendor data safely
void QNNGraph::forwarding(RunLayerContext &context) {
  auto *qnn = context.getContextData()->as<QNNBackendVar>();
  NNTR_THROW_IF(!qnn, std::runtime_error) << "QNN context required";
  qnn->getVar()->executeGraph(...);
}
```

---

## Key Files

| File | Purpose |
|------|---------|
| `engine.h/cpp` | Backend router, context registration |
| `context.h` | Context base class, factory methods |
| `context_data.h` | ContextData (ComputeOps + MemAllocator + vendor extension) |
| `compute_ops.h` | ComputeOps struct, getComputeOps(), init_backend() |
| `compute_ops.cpp` | g_compute_ops global, ensureComputeOps() |
| `arm/arm_ops_table.cpp` | ARM ops table (arm_backend:: wrappers) |
| `x86/x86_ops_table.cpp` | x86 ops table (x86_backend:: wrappers) |
| `fallback/fallback_ops_table.cpp` | Fallback ops table |
| `cl_operations/cl_compute_ops.cpp` | OpenCL ops table (cl_ wrappers) |
| `layer_context.h` | RunLayerContext (getComputeOps, getContextData) |
| `layer_node.cpp` | configureRunContext (ContextData → RunLayerContext) |
| `network_graph.cpp` | Graph finalization (Context → ContextData → LayerNode) |
| `app_context.h/cpp` | CPU backend context (AppContext) |
| `cl_context.h/cpp` | GPU backend context (ClContext) |
| `qnn_context.h/cpp` | QNN backend context (QNNContext) |

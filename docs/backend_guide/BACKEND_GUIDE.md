# Adding a New Compute Backend to NNTrainer

This guide explains how to add a new hardware backend (CUDA, HMX, SLSI NPU, etc.) to nntrainer's compute dispatch system.

## Architecture Overview

```
Engine (router)
├── "cpu"  → AppContext   → cpu_ops (ARM NEON / x86 AVX / fallback)
├── "gpu"  → ClContext    → opencl_ops (OpenCL kernels)
├── "cuda" → CudaContext  → cuda_ops (cuBLAS)          ← plugin .so
├── "hmx"  → HmxContext   → hmx_ops (Qualcomm HMX)     ← plugin .so
└── "slsi" → SlsiContext  → slsi_ops (Samsung LSI NPU)  ← plugin .so
```

When a model specifies `engine=your_backend` on a layer, the Engine routes
layer creation and tensor operations to your Context. Your ComputeOps table
replaces the function pointers used by `Tensor::dot()`, `Tensor::add()`, etc.

**Key principle:** Layers don't know which backend they run on. The same
`tensor.dot()` call dispatches to ARM NEON, OpenCL, or your NPU based on
which ops table is set in the RunLayerContext.

## What You Need to Implement

| Component | Required? | Description |
|-----------|-----------|-------------|
| **ComputeOps table** | Yes | Function pointers for accelerated operations |
| **Context class** | Yes | Lifecycle management, registers with Engine |
| **Wrapper functions** | If needed | Adapt your hardware's API to BLAS-standard signatures |
| **Custom layers** | Optional | NPU-specific layer implementations |
| **Plugin entry point** | Optional | For `.so` dynamic loading |

## Step-by-Step Guide

### Step 1: Create Your ComputeOps Table

The `ComputeOps` struct (defined in `compute_ops.h`) contains function pointers
for every tensor operation. You need to fill in the ones your hardware accelerates
and use CPU fallbacks for the rest.

```cpp
#include <compute_ops.h>
#include <cpu_backend.h>  // for CPU fallback functions

// Your accelerated SGEMM
static void my_sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
                      const unsigned int M, const unsigned int N,
                      const unsigned int K, const float alpha, const float *A,
                      const unsigned int lda, const float *B,
                      const unsigned int ldb, const float beta, float *C,
                      const unsigned int ldc) {
  my_npu_driver_sgemm(handle, TransA, TransB, M, N, K, alpha, A, lda,
                      B, ldb, beta, C, ldc);
}

static nntrainer::ComputeOps my_ops = {
  // ── Accelerated ops ──
  .sgemm_fp32 = my_sgemm,      // NPU-accelerated
  .sgemv_fp32 = my_sgemv,      // NPU-accelerated

  // ── CPU fallback for the rest ──
  .sdot_fp32 = nntrainer::sdot,
  .saxpy_fp32 = nntrainer::saxpy,
  .ele_add_fp32 = nntrainer::ele_add,
  // ... fill all fields (see example_npu_context.cpp for complete list)
};
```

**Important:** All function pointers must follow the BLAS-standard signatures
defined in `compute_ops.h`. If your hardware has different calling conventions,
write thin wrapper functions.

**Nullable pointers:** GPU-specific batch ops (`gemm_q4_0_batch_fp32`, etc.)
can be set to `nullptr` if not supported. The tensor code checks for `nullptr`
before calling these.

### Step 2: Create Your Context Class

```cpp
#include <context.h>
#include <singleton.h>

class MyNpuContext : public nntrainer::Context,
                     public nntrainer::Singleton<MyNpuContext> {
  friend class nntrainer::Singleton<MyNpuContext>;

public:
  std::string getName() override { return "my_npu"; }

private:
  MyNpuContext() : Context(std::make_shared<nntrainer::ContextData>()) {}

  void initialize() noexcept override {
    // 1. Initialize your hardware
    my_npu_driver_init();

    // 2. Set memory allocator and compute ops table
    if (auto cd = getContextData()) {
      cd->setMemAllocator(std::make_shared<nntrainer::MemAllocator>());
      cd->setComputeOps(&my_ops);  // ← THE KEY STEP
    }

    // 3. Register layers (optional)
    add_default_object();
  }
};
```

### Step 3: Register with Engine

**Option A: Static registration** (compiled into nntrainer)

In `engine.cpp`, add:
```cpp
#include <my_npu_context.h>

void Engine::add_default_object() {
  // ... existing cpu/gpu registration ...

  auto &my_npu = MyNpuContext::Global();
  registerContext("my_npu", &my_npu);
}
```

**Option B: Plugin .so** (dynamically loaded at runtime)

Export the plugin entry point:
```cpp
extern "C" nntrainer::ContextPluggable ml_train_context_pluggable = {
  []() -> nntrainer::Context * {
    return &MyNpuContext::Global();
  },
  [](nntrainer::Context *) { /* Singleton, no delete */ }
};
```

Build as shared library and load via:
```cpp
engine.registerPluggableContext("path/to/libmy_npu.so");
```

### Step 4: Use Your Backend

In model configuration (.ini or C++ API):
```ini
[layer_fc]
type = fully_connected
unit = 512
compute_engine = my_npu    # ← routes to your ops table
```

Or in C++:
```cpp
auto layer = ml::train::createLayer("fully_connected",
  {"unit=512", "compute_engine=my_npu"});
```

## Data Flow

```
Model config: compute_engine=my_npu
  ↓
Engine::createLayerObject("fully_connected", {compute_engine=my_npu})
  ↓
Engine routes to MyNpuContext
  ↓
LayerNode gets MyNpuContext's ContextData (containing my_ops)
  ↓
RunLayerContext stores ContextData → getComputeOps() returns my_ops
  ↓
Layer::forwarding() calls tensor.dot()
  ↓
FloatTensor::dotFloat() calls ops()->sgemm_fp32()
  ↓
my_sgemm() → your NPU hardware
```

## ComputeOps Function Categories

| Category | Functions | Typical NPU Support |
|----------|-----------|-------------------|
| **BLAS** | sgemm, sgemv, sdot, saxpy, scopy, sscal | sgemm/sgemv usually accelerated |
| **Element-wise** | ele_add, ele_mul, ele_sub, ele_div | Sometimes accelerated |
| **Activation** | swiglu, tanh_gelu, softmax, max_val | Rarely accelerated (use CPU) |
| **Quantized GEMM** | gemm_q4_0, gemm_q4_K, gemm_q6_K | Often accelerated |
| **Data conversion** | copy_fp32_u16, scopy, etc. | Usually CPU fallback |
| **FP16** | All *_fp16 variants | If hardware supports FP16 |

## Fallback Strategy

For operations your hardware doesn't support, include `<cpu_backend.h>` and
use the CPU functions directly:

```cpp
.ele_add_fp32 = nntrainer::ele_add,  // CPU ARM NEON / x86 AVX
```

This is safe because CPU functions are always available. The tensor code
doesn't care whether the function pointer points to your NPU or the CPU —
it just calls through the pointer.

## Testing Your Backend

1. **Unit test:** Create a test that sets your ops table and runs tensor operations:
```cpp
g_compute_ops = get_my_npu_ops();
float a[] = {1,2,3}, b[] = {4,5,6};
float r = g_compute_ops->sdot_fp32(3, a, 1, b, 1);
assert(r == 32.0f);
```

2. **Integration test:** Run existing nntrainer tests with your backend:
```
meson test -C builddir
```

3. **Mock test:** Replace individual ops to verify dispatch:
```cpp
auto original = g_compute_ops->sgemm_fp32;
g_compute_ops->sgemm_fp32 = my_mock_sgemm;
// run tensor.dot() — verify mock was called
g_compute_ops->sgemm_fp32 = original;
```

## File Structure

```
nntrainer/
├── context.h              # Context base class
├── context_data.h         # ContextData (ComputeOps* + MemAllocator*)
├── tensor/cpu_backend/
│   ├── compute_ops.h      # ComputeOps struct definition
│   ├── compute_ops.cpp    # g_compute_ops global + ensureComputeOps()
│   ├��─ arm/               # ARM ops table
│   ├── x86/               # x86 ops table
│   └── fallback/          # Fallback ops table
├── tensor/cl_operations/
│   └── cl_compute_ops.cpp # OpenCL ops table (example of GPU backend)
├── cl_context.h/cpp       # OpenCL context (example of GPU context)
└── engine.cpp             # Engine registers contexts by name
```

## Complete Example

See `docs/backend_guide/example_npu_context.h` and
`docs/backend_guide/example_npu_context.cpp` for a complete, compilable
example of an NPU backend.

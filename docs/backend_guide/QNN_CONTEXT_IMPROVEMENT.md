# QNN Backend Integration

This document describes how the QNN (Qualcomm Neural Network) backend
integrates with the nntrainer compute dispatch architecture, including
channel-wise int4 weight support.

## Status (current)

The improvements originally proposed in this document have been
implemented. This file now documents the resulting architecture rather
than proposed changes. See `ARCHITECTURE.md` for the broader dispatch
model; this file zooms in on QNN specifics.

## QNN Context Structure

```cpp
class QNNContext : public Context, public Singleton<QNNContext> {
  QNNContext()
    : Context(std::make_shared<QNNBackendVar>()) {}

  void initialize() noexcept override {
    // 1. Ensure CPU fallback ComputeOps exist.
    init_backend();

    // 2. Set CPU ComputeOps on the QNN ContextData — this lets
    //    non-QNN tensor ops (pre/post-processing, tokenizer,
    //    sampling) continue to work when running a QNN model.
    if (auto cd = getContextData(); cd && g_compute_ops) {
      cd->setComputeOps(g_compute_ops);
    }

    // 3. Initialize QNN runtime (HTP backend, RPC memory, graph
    //    compile + load).
    auto *qnn_data = getContextData()->as<QNNBackendVar>();
    qnn_data->getVar()->initialize(...);

    // 4. Register QNN-specific layers (QNNGraph, etc.)
    add_default_object();
  }
};
```

## Type-safe Context Access

Layers access vendor data through the `as<T>()` downcast, which returns
nullptr when the context is of a different type:

```cpp
// In a QNN-specific layer:
void QNNGraph::forwarding(RunLayerContext &ctx) {
  auto *qnn_data = ctx.getContextData()->as<QNNBackendVar>();
  NNTR_THROW_IF(!qnn_data, std::runtime_error)
    << "QNNGraph requires QNN context, got: "
    << ctx.getContextData()->getType();

  qnn_data->getVar()->executeGraph(input_tensors, output_tensors);
}
```

`getType()` is a virtual on `ContextData`; `QNNBackendVar` overrides
it to return `"qnn"`, which is used in error messages and diagnostics.

## Dispatch Model Coexistence in a QNN Model

A single inference run typically mixes op-level and graph-level
dispatch:

```
Embedding layer    → op-level, CPU ComputeOps (token_id → vector)
Transformer block  → graph-level, QNN runtime on HTP
  (the whole block compiles to a single QNN graph)
LM head + softmax  → op-level, CPU ComputeOps (logits → sample)
Tokenizer, loss    → op-level, CPU ComputeOps
```

The QNN `ComputeOps` table is simply the CPU fallback — QNN doesn't
accelerate individual BLAS ops. The acceleration happens at the graph
level, by delegating an entire subgraph to the HTP.

## Channel-wise int4 (QINT4) Weight Integration

QNN's HTP supports int4 weights via the
`QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET` encoding. This matches
nntrainer's `Int4QTensor` canonical layout exactly, so the same
safetensors file can be loaded into either the ARM CPU (KleidiAI) or
the QNN HTP without any format conversion.

### Wire Format Agreement

| Property | Int4QTensor | QNN AXIS_SCALE_OFFSET |
|----------|-------------|----------------------|
| Nibble packing | even index = low nibble, offset-binary | same |
| Scale dtype | fp32 | fp32 |
| Scale axis | width (output columns, axis=1 in [K,N]) | axis = output dim, `numScaleOffsets == N` |
| Zero point | 0 (symmetric) | 0 (`offset = 0` for all scales) |
| group_size | 0 (pure per-channel) | 1 (per-channel) |

Same bytes, same semantic — the safetensors data section is binary
identical between the two consumers.

### Loader Flow

```
nn->load("qwen3-0.6b-int4.safetensors", SAFETENSORS)
  │
  ├── Parse JSON header:
  │     __metadata__.schema_version = "2"
  │     dense:weight.dtype = "I4"
  │     dense:weight.quant = {
  │         encoding: "axis_scale_offset",
  │         axis: 1, bitwidth: 4,
  │         group_size: 0, has_zero_point: false
  │     }
  │
  ├── For ARM CPU (AppContext):
  │     allocate Int4QTensor in memory pool
  │     memcpy raw bytes (qscheme header + nibbles + scales)
  │     → forward: KleidiAI qsi4cxp_unpacked + RHS pack cache
  │
  └── For QNN HTP (QNNContext):
        hand raw bytes to QNN's updateGraphTensors()
        QNN interprets with AXIS_SCALE_OFFSET encoding
        → forward: HTP int4 matmul (accelerated)
```

## ComputeOps Slots for QNN

QNN's ops table should set most entries to CPU fallback and leave
channel-wise int4 slots as nullptr (because QNN consumes int4 at the
graph level, not via op-level dispatch):

```cpp
// In QNNContext::initialize():
static ComputeOps qnn_ops = *g_compute_ops;   // CPU base
// No override — QNN doesn't replace op-level ops for a CausalLM.
// gemm_qsi4cxp_fp32 stays as CPU (KleidiAI on ARM, AVX2 on x86),
// but QNN-scheduled layers never reach this path — they go through
// QNNGraph which calls QNN runtime directly.
qnn_ops.gemm_qsi4cxp_fp32 = nullptr;  // optional: make op-level
                                        // channel-wise int4 unsupported
                                        // on QNN context (graph-level only)

cd->setComputeOps(&qnn_ops);
```

The `nullptr` choice signals intent clearly: QNN layers must use the
graph-level path. If a non-QNN layer (accidentally) tries int4 GEMM
through the QNN context, `FloatTensor::dotQInteger` throws with a
clear error rather than silently running CPU math.

## Build Integration

QNN is loaded as a plugin `.so` to keep QNN SDK dependencies out of
the core `libnntrainer.so`:

```
Engine::add_default_object()
  │
  ├── registerContext("cpu", &AppContext::Global())   ← always on
  ├── registerContext("gpu", &ClContext::Global())    ← if enable-opencl
  │
  └── registerPluggableContext("libqnn_context.so")   ← runtime dlopen
        └── exports: ml_train_context_pluggable = {
              create_qnn_context,
              destroy_qnn_context
            }
```

QNN layers register their factories inside `QNNContext::add_default_object`,
and the plugin pulls in QNN SDK symbols via its own link dependency.

## References

- `nntrainer/qnn_context.cpp` — QNNContext implementation
- `nntrainer/qnn/jni/qnn_context_var.h` — QNNBackendVar / QNNVar
- `nntrainer/tensor/int4_tensor.h` — canonical layout doc
- `docs/backend_guide/ARCHITECTURE.md` — broader dispatch model
- `docs/backend_guide/BACKEND_GUIDE.md` — guide for adding a new backend
- Qualcomm QNN SDK `QnnTensor.h` — `Qnn_QuantizeParams_t`,
  `QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET`

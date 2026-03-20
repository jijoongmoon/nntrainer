# Symbolic Tensor API

> **Status**: Experimental (`ml::train` namespace)
> **Minimum C++ Version**: C++17

NNTrainer's Symbolic Tensor API enables PyTorch/Keras-style functional model definition in C++. Instead of the traditional string-based `addLayer()` approach, you **call layers with tensors as arguments** to construct a computation graph, then extract it automatically via `Model::compile(input, output)`.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
  - [Tensor](#tensor)
  - [LayerHandle](#layerhandle)
  - [Model::compile Overloads](#modelcompile-overloads)
- [Lazy Chaining](#lazy-chaining)
- [Architecture Diagrams](#architecture-diagrams)
  - [Class Diagram](#class-diagram)
  - [Tensor Lifecycle (State Diagram)](#tensor-lifecycle)
  - [Graph Construction & Compilation (Sequence Diagram)](#graph-construction--compilation)
  - [Lazy Chaining (Sequence Diagram)](#lazy-chaining-sequence)
  - [API Level Comparison (Component Diagram)](#api-level-comparison)
  - [String-based vs Symbolic](#string-based-vs-symbolic)
- [Examples](#examples)
- [File Locations](#file-locations)

---

## Quick Start

```cpp
#include <tensor_api.h>
#include <model.h>

using namespace ml::train;

// 1. Create symbolic input tensor
Tensor input({1, 1, 1, 784}, "input");

// 2. Wrap layers in LayerHandle and call them
LayerHandle fc1 = createLayer("fully_connected", {"unit=128", "name=fc1"});
LayerHandle relu = createLayer("activation", {"activation=relu", "name=relu1"});
LayerHandle fc2 = createLayer("fully_connected", {"unit=10", "name=fc2"});

auto h = fc1(input);
h = relu(h);
auto output = fc2(h);

// 3. Pass input/output tensors to auto-extract graph + compile
auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
model->compile(input, output);  // compile + initialize + allocate in one call

// 4. Run inference
float data[784] = { /* ... */ };
input.copyFrom(data);
auto results = model->inference(1, {data});
```

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Symbolic Tensor** | A placeholder with only shape and name. No actual data. |
| **Eager Tensor** | Created via `zeros()`, `ones()`, `fromData()`. Holds data immediately. |
| **LayerHandle** | Wraps `createLayer()` result; `operator()` creates graph edges. |
| **SymbolicGraphNode** | Internal DAG node. Stores producing_layer + inputs. |
| **Materialization** | After `Model::compile()`, symbolic tensors are bound to real memory. |
| **Lazy Chaining** | Deferred execution via `chain().add_i().multiply_i().eval()`. |

---

## API Reference

### Tensor

#### Construction

```cpp
// Symbolic tensor (graph placeholder)
Tensor input(TensorDim({1, 1, 28, 28}), "input");

// Eager tensors (immediate data)
auto zeros = Tensor::zeros({1, 1, 3, 3});
auto ones  = Tensor::ones({1, 1, 3, 3});

// Wrap external memory (zero-copy)
float buf[12];
auto ext = Tensor::fromData({1, 1, 3, 4}, buf, "cache");
```

#### State Queries

| Method | Returns | Description |
|--------|---------|-------------|
| `isValid()` | `bool` | Whether the tensor has been properly constructed |
| `isMaterialized()` | `bool` | Whether actual data is accessible |
| `isExternal()` | `bool` | Whether it wraps user-managed memory via `fromData()` |
| `shape()` | `TensorDim` | Tensor dimensions |
| `name()` | `string` | Tensor name |
| `dtype()` | `DataType` | Data type (default FP32) |

#### Data Access (Requires Materialized State)

```cpp
const float *ptr = tensor.data<float>();      // Read-only access
float *mptr = tensor.mutable_data<float>();    // Mutable access
float val = tensor.getValue(b, c, h, w);       // Read single value
tensor.setValue(b, c, h, w, 42.0f);            // Write single value
tensor.copyFrom(src_buffer);                    // Copy from external buffer
```

#### Symbolic Operations (Create Implicit Layers)

```cpp
auto c = a.add(b);       // Creates an implicit Addition layer
auto c = a.multiply(b);  // Creates an implicit Multiply layer
auto y = x.reshape(dim); // Creates an implicit Reshape layer
```

#### Eager Operations (Return New Tensors)

```cpp
auto r = t.add(5.0f);           // Scalar addition
auto r = t.subtract(other);     // Tensor subtraction
auto r = t.multiply(3.0f);      // Scalar multiplication
auto r = t.divide(other);       // Tensor division
auto r = t.dot(other);          // Matrix multiplication
auto r = t.transpose("0:2:1");  // Transpose
auto r = t.pow(2.0f);           // Power
auto r = t.sum(axis);           // Sum along axis
auto r = t.average();           // Global average
float n = t.l2norm();           // L2 norm
auto ids = t.argmax();          // Argmax indices
```

#### Graph Traversal

```cpp
auto layer = output.getProducingLayer();   // Layer that produced this tensor
auto inputs = output.getInputTensors();    // Input tensors to producing layer
auto out0 = split_out.output(0);           // i-th output of a multi-output layer
```

### LayerHandle

```cpp
// Directly assign from createLayer (implicit conversion)
LayerHandle fc = createLayer("fully_connected", {"unit=256", "name=fc1"});

// Single input
auto output = fc(input);

// Multiple inputs (e.g., MHA)
auto attn = mha({q, k, v});

// Access layer properties
fc->getName();   // "fc1"
fc->getType();   // "fully_connected"
```

### Model::compile Overloads

```cpp
// Single input, single output
model->compile(input, output);

// Single input, multiple outputs
model->compile(input, {out1, out2});

// Multiple inputs, multiple outputs
model->compile({in1, in2}, {out1, out2});

// Specify execution mode
model->compile(input, output, ExecutionMode::INFERENCE);
```

---

## Lazy Chaining

Queue multiple in-place operations on a materialized tensor for **deferred batch execution**.

```cpp
auto t = Tensor::ones({1, 1, 2, 2});

// (1 + 2) * 3 - 1 = 8
t.chain()
  .add_i(2.0f)
  .multiply_i(3.0f)
  .subtract_i(1.0f)
  .eval();  // All operations execute here

// Tensor-tensor operations are also supported
auto other = Tensor::ones({1, 1, 2, 2});
t.chain().add_i(other, 0.5f).eval();
```

**Supported operations**: `add_i`, `subtract_i`, `multiply_i`, `divide_i`, `pow_i`, `inv_sqrt_i`

**Rules**:
- `chain()` clears any previously queued operations
- `eval()` executes queued operations in order, then clears the queue
- Calling `eval()` on a non-materialized tensor throws `std::runtime_error`

---

## Architecture Diagrams

### Class Diagram

```mermaid
classDiagram
    class Tensor {
        -impl_ : unique_ptr~Impl~
        +Tensor()
        +Tensor(dim, name)
        +fromData(dim, ptr, name)$ Tensor
        +zeros(dim, name)$ Tensor
        +ones(dim, name)$ Tensor
        +isValid() bool
        +isMaterialized() bool
        +isExternal() bool
        +shape() TensorDim
        +name() string
        +data~T~() const T*
        +mutable_data~T~() T*
        +getValue(b,c,h,w) float
        +setValue(b,c,h,w,val) void
        +add(other) Tensor
        +multiply(other) Tensor
        +reshape(shape) Tensor
        +getProducingLayer() shared_ptr~Layer~
        +output(index) Tensor
        +getInputTensors() vector~Tensor~
        +chain() Tensor&
        +add_i(value) Tensor&
        +multiply_i(value) Tensor&
        +eval() Tensor&
        +dot(other) Tensor
        +transpose(dir) Tensor
        +sum(axis) Tensor
        +argmax() vector~uint~
        +clone() Tensor
    }

    class LayerHandle {
        -ptr_ : shared_ptr~Layer~
        +LayerHandle(unique_ptr~Layer~)
        +LayerHandle(shared_ptr~Layer~)
        +operator()(input) Tensor
        +operator()(inputs) Tensor
        +get() Layer*
        +layer() shared_ptr~Layer~
    }

    class Layer {
        <<abstract>>
        +getType() string*
        +getName() string*
        +setProperty(props)*
    }

    class Impl["Tensor::Impl"] {
        +dim : TensorDim
        +name : string
        +valid : bool
        +external : bool
        +eager_data : shared_ptr~nntrainer_Tensor~
        +external_ptr : void*
        +graph_edge : shared_ptr~SymbolicGraphNode~
        +bound_tensor : nntrainer_Tensor*
        +call_chain : vector~function~
    }

    class SymbolicGraphNode {
        +producing_layer : shared_ptr~Layer~
        +inputs : vector~shared_ptr~
        +dim : TensorDim
        +name : string
        +output_index : int
    }

    class Model {
        +compile(input, output, mode) int
        +compile(inputs, outputs, mode) int
        +addLayer(layer) int
        +inference(batch, inputs) vector~float*~
    }

    Tensor *-- Impl : pimpl
    Impl --> SymbolicGraphNode : graph_edge
    SymbolicGraphNode --> SymbolicGraphNode : inputs (DAG)
    SymbolicGraphNode --> Layer : producing_layer
    LayerHandle --> Layer : wraps
    LayerHandle ..> Tensor : creates symbolic output
    Model ..> Tensor : compile(in, out)
```

### Tensor Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Invalid

    Invalid --> Symbolic : Tensor(dim, name)
    Invalid --> Eager : zeros() / ones()
    Invalid --> External : fromData(dim, ptr)

    Symbolic --> SymbolicWithEdge : LayerHandle call
    SymbolicWithEdge --> SymbolicWithEdge : Additional LayerHandle calls

    SymbolicWithEdge --> Materialized : Model compile
    Eager --> Materialized : Already has data
    External --> Materialized : External memory bound

    Materialized --> LazyChaining : chain()
    LazyChaining --> LazyChaining : add_i / multiply_i / etc
    LazyChaining --> Materialized : eval()

    note right of Invalid
        Default constructor
        isValid() == false
    end note

    note right of Symbolic
        Only dim and name
        isMaterialized() == false
    end note

    note right of SymbolicWithEdge
        Has graph_edge
        Layer connected in DAG
    end note

    note right of Materialized
        data() accessible
        getValue / setValue available
    end note

    note right of LazyChaining
        Operations queued in call_chain
        Batch executed on eval()
    end note
```

### Graph Construction & Compilation

```mermaid
sequenceDiagram
    participant C as Client
    participant T as Tensor
    participant LH as LayerHandle
    participant SG as SymbolicGraphNode
    participant M as Model

    Note over C,M: 1. Create symbolic tensor
    C->>T: Tensor input({1,1,1,8}, "input0")

    Note over C,M: 2. Create LayerHandle
    C->>LH: LayerHandle fc = createLayer("fc", props)

    Note over C,M: 3. Call layer - creates graph edge
    C->>LH: auto out = fc(input)
    LH->>SG: new SymbolicGraphNode
    Note right of SG: producing_layer = fc
    LH->>T: return Tensor(out_dim)

    Note over C,M: 4. Chained calls
    C->>LH: auto y = relu(fc2(out))
    LH->>SG: node(fc2, inputs=[node(fc)])
    LH->>SG: node(relu, inputs=[node(fc2)])

    Note over C,M: 5. Implicit layers (Tensor methods)
    C->>T: auto z = x.add(y)
    T->>LH: create Addition LayerHandle
    LH->>SG: node(Addition, inputs=[x, y])
    T-->>C: return Tensor z

    Note over C,M: 6. Model compile (graph extraction)
    C->>M: model.compile(input, output)

    rect rgb(240, 248, 255)
        M->>SG: DFS traversal (output to input)
        SG-->>M: layers in topological order
        M->>M: addLayer(input_layer)
        M->>M: addLayer(each_layer)
        M->>M: compile + initialize + allocate
    end

    M->>T: bind tensors to allocated buffers
    M-->>C: return ML_ERROR_NONE

    Note over C,M: 7. Run inference
    C->>T: input.copyFrom(data)
    C->>M: model.inference(...)
    C->>T: output.data()
```

### Lazy Chaining Sequence

```mermaid
sequenceDiagram
    participant C as Client
    participant T as Tensor
    participant I as Impl
    participant N as nntrainer Tensor

    Note over C,N: Tensor is already materialized

    C->>T: chain()
    T->>I: call_chain.clear()
    T-->>C: return this

    C->>T: add_i(2.0f)
    T->>I: push_back(fn: add_i 2.0)
    T-->>C: return this

    C->>T: multiply_i(3.0f)
    T->>I: push_back(fn: multiply_i 3.0)
    T-->>C: return this

    C->>T: eval()
    rect rgb(255, 245, 238)
        T->>I: for each fn in call_chain
        I->>N: add_i(2.0f)
        N-->>I: done
        I->>N: multiply_i(3.0f)
        N-->>I: done
    end
    T->>I: call_chain.clear()
    T-->>C: return this
```

### API Level Comparison

```mermaid
flowchart TB
    subgraph PUBLIC["Public API - ml::train"]
        direction TB

        subgraph SYMBOLIC["Symbolic Tensor API - New"]
            T[Tensor - symbolic placeholder]
            LH[LayerHandle - operator call]
            SGN[SymbolicGraphNode - DAG]
            T -->|"fc(input)"| LH
            LH -->|creates| SGN
            SGN -->|"inputs[]"| SGN
        end

        subgraph COMPILE["Model::compile - Tensor, Tensor"]
            DFS[DFS traversal - output to input]
            EXTRACT[Layer extraction - topological order]
            DFS --> EXTRACT
        end

        subgraph LEGACY["Legacy API"]
            CL["createLayer(type, props)"]
            AL["model.addLayer(layer)"]
            CP["model.compile(mode)"]
            IN["model.initialize(mode)"]
            CL --> AL --> CP --> IN
        end

        SYMBOLIC --> COMPILE
        COMPILE --> LEGACY
    end

    subgraph INTERNAL["Internal - nntrainer namespace"]
        direction LR
        NT[nntrainer::Tensor - actual data]
        MP[MemoryPool - memory allocation]
        GN[GraphNode - execution graph]
        LT[LazyTensor - legacy lazy eval]
        LT --> NT
        MP --> NT
    end

    PUBLIC --> INTERNAL

    style SYMBOLIC fill:#e1f5fe,stroke:#0288d1
    style COMPILE fill:#fff3e0,stroke:#f57c00
    style LEGACY fill:#f3e5f5,stroke:#7b1fa2
    style INTERNAL fill:#e8f5e9,stroke:#388e3c
```

### String-based vs Symbolic

```mermaid
flowchart LR
    subgraph OLD["Legacy: String-based"]
        direction TB
        O1["createLayer('fc',
        name=fc1,
        input_layers=input0,
        unit=256)"]
        O2["layers.push_back(layer)"]
        O3["... repeat for all layers ..."]
        O4["for each layer:
        model.addLayer(l)"]
        O5["model.compile(mode)"]
        O6["model.initialize(mode)"]
        O1 --> O2 --> O3 --> O4 --> O5 --> O6
    end

    subgraph NEW["New: Symbolic Tensor"]
        direction TB
        N1["Tensor input({1,1,1,8})"]
        N2["LayerHandle fc = createLayer(...)"]
        N3["auto out = fc(input)
        // auto-connected!"]
        N4["... continue chaining ..."]
        N5["model.compile(input, output)
        // auto graph extraction"]
        N1 --> N2 --> N3 --> N4 --> N5
    end

    OLD -.->|migration| NEW

    style OLD fill:#ffebee,stroke:#c62828
    style NEW fill:#e8f5e9,stroke:#2e7d32
```

---

## Examples

### Residual Connection (Skip Connection)

```cpp
using namespace ml::train;

auto x = Tensor({1, 1, 1, 256}, "input");

LayerHandle fc = createLayer("fully_connected", {"unit=256", "name=fc_res"});
auto h = fc(x);
auto out = x.add(h);  // implicit Addition layer

auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
model->compile(x, out);
```

### Transformer Decoder Block (CausalLM Pattern)

```cpp
using namespace ml::train;

const unsigned int DIM = 256, FF_DIM = 512, NUM_HEADS = 4;

Tensor input({1, 1, 1, 4}, "input0");

// Embedding
LayerHandle embedding = createLayer("fully_connected",
    {"name=embedding0", "unit=" + std::to_string(DIM), "disable_bias=true"});
Tensor x = embedding(input);

// Decoder blocks (can be repeated)
for (int i = 0; i < NUM_LAYERS; ++i) {
    std::string p = "layer" + std::to_string(i);

    // Attention
    LayerHandle att_norm = createLayer("layer_normalization",
        {"name=" + p + "_att_norm", "axis=3", "epsilon=1e-5"});
    Tensor normed = att_norm(x);

    LayerHandle q = createLayer("fully_connected",
        {"name=" + p + "_wq", "unit=" + std::to_string(DIM)});
    LayerHandle k = createLayer("fully_connected",
        {"name=" + p + "_wk", "unit=" + std::to_string(DIM)});
    LayerHandle v = createLayer("fully_connected",
        {"name=" + p + "_wv", "unit=" + std::to_string(DIM)});

    // Self-attention (Q, K, V all from same normed input)
    LayerHandle mha = createLayer("multi_head_attention",
        {"name=" + p + "_mha", "num_heads=" + std::to_string(NUM_HEADS)});
    auto attn_out = mha({q(normed), k(normed), v(normed)});

    // Residual
    Tensor residual = x.add(attn_out);

    // FFN
    LayerHandle ffn_norm = createLayer("layer_normalization",
        {"name=" + p + "_ffn_norm", "axis=3"});
    LayerHandle fc1 = createLayer("fully_connected",
        {"name=" + p + "_fc1", "unit=" + std::to_string(FF_DIM), "activation=gelu"});
    LayerHandle fc2 = createLayer("fully_connected",
        {"name=" + p + "_fc2", "unit=" + std::to_string(DIM)});

    auto ffn_out = fc2(fc1(ffn_norm(residual)));

    // Residual
    x = residual.add(ffn_out);
}

// Final norm + LM head
LayerHandle final_norm = createLayer("layer_normalization", {"name=final_norm"});
LayerHandle lmhead = createLayer("fully_connected",
    {"name=lmhead", "unit=" + std::to_string(VOCAB_SIZE)});
Tensor output = lmhead(final_norm(x));

auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
model->compile(input, output, ExecutionMode::INFERENCE);
```

### External KV Cache (MHA with fromData)

```cpp
using namespace ml::train;

auto input = Tensor({1, 1, 4, 64}, "input");

// External memory for KV cache (zero-copy)
float key_buf[1 * 1 * 32 * 64] = {};
float val_buf[1 * 1 * 32 * 64] = {};
auto key_cache = Tensor::fromData({1, 1, 32, 64}, key_buf, "key_cache");
auto val_cache = Tensor::fromData({1, 1, 32, 64}, val_buf, "val_cache");

LayerHandle q_proj = createLayer("fully_connected", {"unit=64", "name=q_proj"});
LayerHandle k_proj = createLayer("fully_connected", {"unit=64", "name=k_proj"});
LayerHandle v_proj = createLayer("fully_connected", {"unit=64", "name=v_proj"});

LayerHandle mha = createLayer("multi_head_attention",
    {"name=mha", "num_heads=4"});

// Pass cache tensors as additional inputs
auto attn = mha({q_proj(input), k_proj(input), v_proj(input),
                  key_cache, val_cache});

auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
model->compile(input, attn);

// Cache tensors remain external — update directly
// key_cache.isExternal() == true
```

### Lazy Chaining (Post-processing)

```cpp
// Post-process inference results
auto logits = /* model output tensor */;

// Deferred chain: (logits / temperature) + bias
logits.chain()
    .divide_i(temperature)
    .add_i(bias_tensor)
    .eval();

// Further processing (e.g., softmax)
auto probs = logits.apply([](float x) { return std::exp(x); });
```

---

## File Locations

| Component | Header | Implementation |
|-----------|--------|----------------|
| Tensor, LayerHandle | `api/ccapi/include/tensor_api.h` | `api/ccapi/src/tensor_api.cpp` |
| SymbolicGraphNode | (internal) | `api/ccapi/src/tensor_api.cpp` |
| Model::compile (Tensor) | `api/ccapi/include/model.h` | `api/ccapi/src/tensor_api.cpp` |
| LazyTensor (internal) | `nntrainer/tensor/lazy_tensor.h` | `nntrainer/tensor/lazy_tensor.cpp` |
| Unit Tests | | `test/ccapi/unittest_ccapi_tensor.cpp` |
| CausalLM Example | `Applications/CausalLM/causal_lm.h` | |

# NNTrainer Weight Binary Format Specification

This document describes the weight serialization formats used by NNTrainer,
including the legacy BIN format and the new safetensors format, along with
the TorchFXConverter integration for HuggingFace model conversion.

---

## Format Overview

NNTrainer supports two weight file formats:

```
                    +-------------------+
                    |   Weight Formats  |
                    +-------------------+
                           |
              +------------+------------+
              |                         |
     +--------+--------+     +---------+---------+
     |   BIN v0        |     |   Safetensors     |
     |   (Legacy)      |     |   (Name-based)    |
     +-----------------+     +-------------------+
     | Sequential      |     | JSON header with  |
     | offset-based    |     | name -> offset    |
     | Order-dependent |     | Order-independent |
     +-----------------+     +-------------------+
```

| Aspect              | BIN v0 (Legacy)          | Safetensors             |
|---------------------|--------------------------|-------------------------|
| Header              | None                     | 8B size + JSON header   |
| Offset lookup       | Sequential accumulation  | Name-based (JSON)       |
| Order dependency    | Strict (topological)     | None (name-based)       |
| Metadata            | None                     | dtype, shape, offsets   |
| Parallel loading    | Yes (per-layer threads)  | Yes (per-layer threads) |
| Cross-graph compat  | No (order must match)    | Yes (name lookup)       |

---

## 1. BIN v0 Format (Legacy)

The original binary format stores weights sequentially in the order they
appear during model graph iteration (topological sort order).

### File Layout

```
+================================================================+
|                     BIN v0 File Layout                         |
+================================================================+

Byte offset
    0  +-----------------------------------------------------+
       | Weight data (Layer 0, Weight 0)                      |
       |   Raw bytes: getMemoryBytes() of tensor              |
       +-----------------------------------------------------+
       | Weight data (Layer 0, Weight 1)  [if exists]         |
       +-----------------------------------------------------+
       | Weight data (Layer 1, Weight 0)                      |
       +-----------------------------------------------------+
       |                    ...                               |
       +-----------------------------------------------------+
       | Weight data (Layer N, Weight M)                      |
       +=====================================================+
       | [Optional] "adam" magic (4 bytes ASCII)              |
       +-----------------------------------------------------+
       | [Optional] Adam optimizer state per weight           |
       |   (same layer/weight iteration order)                |
       +=====================================================+
       | [Optional, TRAIN mode] epoch_idx (size_t)            |
       +-----------------------------------------------------+
       | [Optional, TRAIN mode] iteration (size_t)            |
       +-----------------------------------------------------+
```

### Offset Calculation

```
Weight offsets are calculated sequentially during load():

  start_from = 0
  for each layer in model_graph (topological order):
      for each weight in layer:
          weight.setFileOffset(start_from)
          start_from += weight.getMemoryBytes()
          if quantized type (not FP32/FP16/Q6_K/Q4_0):
              start_from += 2   // uint16_t qparam
```

### Limitation

```
  TorchFXConverter                    NeuralNetwork
  (FX execution order)                (Topological sort order)
  +-----------------------+           +-----------------------+
  | 1. embedding          |           | 1. embedding          |
  | 2. attn_q_proj        |           | 2. attn_k_proj   <--- | Different!
  | 3. attn_k_proj        |           | 3. attn_q_proj   <--- |
  | 4. attn_v_proj        |           | 4. attn_v_proj        |
  | 5. ffn_gate           |           | 5. ffn_down      <--- |
  | 6. ffn_up             |           | 6. ffn_gate      <--- |
  | 7. ffn_down           |           | 7. ffn_up        <--- |
  +-----------------------+           +-----------------------+
      |                                     |
      v                                     v
  Writes weights in                 Reads weights in
  FX order to .bin                  topological order
                                          |
                                          v
                                    WRONG WEIGHTS!
                                    (offset mismatch)
```

**This ordering mismatch is why safetensors was introduced.**

---

## 2. Safetensors Format

The safetensors format uses a JSON header with per-tensor name, dtype,
shape, and byte offset information. This makes weight loading
**order-independent** -- weights are found by name, not position.

### File Layout

```
+================================================================+
|                  Safetensors File Layout                        |
+================================================================+

Byte 0                    8              8 + header_size
  |                       |                    |
  v                       v                    v
  +----------+---------------------------+---------------------------+
  | header   |       JSON Header         |       Data Section        |
  | size     |    (8-byte aligned,       |    (raw tensor bytes)     |
  | (8B LE)  |     padded with ' ')      |                           |
  +----------+---------------------------+---------------------------+

  |<- 8B  -->|<----- header_size B ----->|<---- total data size ---->|
```

### Header Detail

```
+----------+
| uint64   |  header_size (little-endian, 8 bytes)
+----------+
|          |
|  JSON    |  {
|  Header  |    "__metadata__": {
|          |      "format": "nntrainer"
|  (UTF-8) |    },
|          |    "layer0_wq:weight": {
|          |      "dtype": "F32",
|          |      "shape": [64, 64],
|          |      "data_offsets": [0, 16384]
|          |    },
|          |    "layer0_wk:weight": {
|          |      "dtype": "F32",
|          |      "shape": [32, 64],
|          |      "data_offsets": [16384, 24576]
|          |    },
|          |    ...
|          |  }
|          |
+----------+  (padded to 8-byte boundary with spaces)
```

### Data Section Detail

```
data_offsets are relative to the start of the data section:

  Data section start = 8 + header_size

  +--------+--------+--------+--------+--------+-----+
  | Tensor | Tensor | Tensor | Tensor | Tensor | ... |
  |   0    |   1    |   2    |   3    |   4    |     |
  +--------+--------+--------+--------+--------+-----+
  ^        ^        ^
  |        |        |
  off=0    off=sz0  off=sz0+sz1
  |                 |
  |  data_offsets   |
  |  [0, sz0]       [sz0, sz0+sz1]
```

### Supported Data Types

| TensorDim::DataType | Safetensors dtype | Byte size per element |
|---------------------|-------------------|-----------------------|
| FP32                | "F32"             | 4                     |
| FP16                | "F16"             | 2                     |
| QINT4               | "I4"              | 0.5                   |
| QINT8               | "I8"              | 1                     |
| QINT16              | "I16"             | 2                     |
| UINT4               | "U4"              | 0.5                   |
| UINT8               | "U8"              | 1                     |
| UINT16              | "U16"             | 2                     |
| UINT32              | "U32"             | 4                     |

### Shape Encoding

Tensor dimensions are encoded compactly by stripping leading 1s:

```
  TensorDim [1, 1, 512, 768]  -->  "shape": [512, 768]
  TensorDim [1, 1, 1, 64]     -->  "shape": [64]
  TensorDim [1, 1, 1, 1]      -->  "shape": [1]
```

---

## 3. Offset Assignment: BIN vs Safetensors

```
                        load() function
                             |
                +------------+------------+
                |                         |
      +---------v----------+    +---------v-----------+
      |    BIN v0           |    |    Safetensors      |
      +--------------------+    +---------------------+
      |                    |    |                     |
      |  Sequential scan:  |    |  Parse JSON header: |
      |  start_from = 0    |    |  header_size -> JSON|
      |  for each weight:  |    |  -> name_offset_map |
      |    offset = start  |    |                     |
      |    start += size   |    |  Name-based lookup: |
      |                    |    |  for each weight:   |
      |  ORDER MATTERS!    |    |    name -> (off,sz) |
      |                    |    |    offset = data_   |
      |                    |    |      section_start  |
      |                    |    |      + off          |
      |                    |    |                     |
      |                    |    |  ORDER FREE!        |
      +--------------------+    +---------------------+
                |                         |
                +------------+------------+
                             |
                             v
                   Per-layer parallel
                   loading (threads)
```

---

## 4. Parallel Loading Architecture

Both formats support per-layer parallel loading during inference.
Each layer reads its weights independently using either per-thread
file streams or memory-mapped I/O.

```
                    load() -- INFERENCE mode
                              |
             +----------------+----------------+
             |                                 |
    +--------v--------+              +---------v--------+
    |  Non-MMAP mode  |              |    MMAP mode     |
    +-----------------+              +------------------+
    |                 |              |                  |
    | Thread per layer|              | Thread per layer |
    | Each opens own  |              | Each creates own |
    | ifstream to     |              | mmap mapping     |
    | same file       |              | to same file     |
    +-----------------+              +------------------+

    Thread 0           Thread 1           Thread 2
    +------------+     +------------+     +------------+
    | Layer 0    |     | Layer 1    |     | Layer 2    |
    | read at    |     | read at    |     | read at    |
    | offset 0   |     | offset 4K  |     | offset 12K |
    +-----+------+     +-----+------+     +-----+------+
          |                  |                  |
          v                  v                  v
    +-----+------------------+------------------+------+
    |  File: model.safetensors                         |
    |  [header] [tensor0] [tensor1] [tensor2] [...]    |
    +--------------------------------------------------+
```

### POSIX mmap Strategy

```
  Per thread:
    1. fd = open(file, O_RDONLY)
    2. mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0)
    3. close(fd)                          // fd not needed after mmap
    4. posix_madvise(ptr, size, POSIX_MADV_RANDOM)  // scattered access
    5. node->read(view, ...)              // read weights at offset
    6. posix_madvise(ptr, size, POSIX_MADV_DONTNEED) // drop pages
    7. munmap(ptr, size)                  // release mapping
```

### Windows mmap Strategy

```
  Per thread:
    1. hFile = CreateFileA(path, GENERIC_READ, ...)
    2. hMap = CreateFileMapping(hFile, PAGE_READONLY, ...)
    3. view = MapViewOfFile(hMap, FILE_MAP_READ, ...)
    4. node->read(view, ...)
    5. UnmapViewOfFile(view)              // early page release
    6. CloseHandle(hMap)
    7. CloseHandle(hFile)
```

---

## 5. TorchFXConverter Integration

The TorchFXConverter pipeline generates weights in two formats,
with safetensors as the primary format for order-independent loading.

### Conversion Pipeline

```
  HuggingFace Model
  (state_dict)
        |
        v
  +---------------------+
  | WeightConverter      |
  |                      |
  | 1. build_weight_map()|     Weight ordering:
  |    layers -> entries |     FX symbolic graph
  |    (FX exec order)   |     execution order
  |                      |
  | 2. For each entry:   |
  |    - Load HF tensor  |
  |    - Apply transform |     Linear: transpose
  |      (transpose for  |     [out,in] -> [in,out]
  |       Linear layers) |
  |    - Convert dtype   |     float32 / float16
  +----------+-----------+
             |
    +--------+--------+
    |                  |
    v                  v
  .bin               .safetensors
  (raw sequential)   (JSON header + data)
```

### Tensor Naming Convention

Every writer that produces a safetensors file for nntrainer **must** use
the canonical weight name that the loader expects. nntrainer assembles
that name in `nntrainer/layers/layer_context.h::InitLayerContext::requestWeight`:

```cpp
weights_spec.emplace_back(
    dim, dim_g, init, reg, ..., prefix + ":" + name, ...);
```

where

- `prefix` = the layer's user-given name (the `name=<...>` property passed
  to `createLayer()` — e.g. `fc1`, `embedding0`, `layer0_attn_q_proj`);
- `name` = the short role string hard-coded by the layer implementation
  (e.g. `"weight"`, `"bias"`, `"gamma"`, `"beta"`, `"lora_A_weight"`).

The resulting canonical form is therefore:

```
  <layer_name>:<param_role>
```

Common roles used across layers:

```
  Layer                Role strings (produced by layer code)
  -------------------------------------------------------------
  fully_connected      "weight", "bias", "lora_A_weight", "lora_B_weight"
  embedding            "weight"
  conv2d / conv1d      "weight", "bias"
  batch_normalization  "mu", "var", "gamma", "beta"
  layer_normalization  "gamma", "beta"
  rms_norm             "gamma"
  multi_head_attention "q_proj:weight", "k_proj:weight", ... (nested)
  lstm / gru / rnn     "weight_ih", "weight_hh", "bias_h"
```

Writers of safetensors files (TorchFXConverter, per-model
`res/*/weight_converter.py`, and the planned `bin_to_safetensors` CLI)
must map every tensor they emit to this `<layer>:<role>` key so that
`NeuralNetwork::load()` finds it by name. A name that is not present
in the header falls back to the legacy sequential offset path with a
warning in the log — which defeats the whole point of the format.

Example mapping from HuggingFace state_dict keys to nntrainer canonical
names, assuming the CausalLM pipeline names its layers
`embedding0`, `layer<i>_attn_q_proj`, `output_norm`:

```
  HuggingFace key                           nntrainer canonical name
  ----------------------------------------------------------------------
  model.embed_tokens.weight                 embedding0:weight
  model.layers.0.self_attn.q_proj.weight    layer0_attn_q_proj:weight
  model.layers.0.self_attn.q_proj.bias      layer0_attn_q_proj:bias
  model.layers.0.input_layernorm.weight     layer0_attn_norm:gamma
  model.norm.weight                         output_norm:gamma
```

The actual layer names in any one pipeline are determined by the
`name=` property passed to `createLayer()` for each layer — so a
writer must either (a) match the names its companion converter hard-codes,
or (b) consume a canonical name list produced by nntrainer itself
(e.g. a `nntr_dump_weight_names` utility) as the ground truth.

### Weight Transformation Rules

```
  Layer Type        Transform        Shape Change
  ─────────────────────────────────────────────────
  Linear (FC)       transpose        [out, in] → [in, out]
  Embedding         none             [vocab, dim] → [vocab, dim]
  RMSNorm           none             [dim] → [dim]
  LayerNorm         none             [dim] → [dim]
  Tied weights      skip (shared)    stored once, referenced by name
```

---

## 6. Format Detection in load()

```
  load(file_path, format)
        |
        +-- format == MODEL_FORMAT_SAFETENSORS ?
        |         |
        |    YES  |  NO (MODEL_FORMAT_BIN)
        |         |
        v         v
  +-----------+  +------------------+
  | Read 8B   |  | Sequential       |
  | header    |  | offset calc      |
  | size      |  | (topological     |
  |           |  |  order)           |
  | Read JSON |  |                  |
  | header    |  | start = 0       |
  |           |  | for each weight: |
  | Parse     |  |   off = start   |
  | name ->   |  |   start += size |
  | (off,sz)  |  +------------------+
  | map       |
  +-----------+
        |
        v
  Assign file offsets to weights
  (name-based or sequential)
        |
        v
  Per-layer parallel read (threads + mmap)
```

---

## 7. Example: Qwen3-0.6B Weight File

For a tiny Qwen3 model (2 layers, hidden=64):

```
  Safetensors file:  78,090,344 bytes
  Raw BIN file:      78,087,680 bytes
  Header overhead:        2,664 bytes  (8B size + 2,656B JSON)

  Weight breakdown:
  ┌──────────────────────────────────────────────────┐
  │  Type              Count    Params      Bytes    │
  ├──────────────────────────────────────────────────┤
  │  Embedding           1    9,723,904   38,895,616 │
  │  Attention (Q/K/V/O) 8      varies    ~varies   │
  │  FFN (gate/up/down)  6      varies    ~varies   │
  │  Norm (RMS/QK)       9       ~576       ~2,304  │
  │  LM Head             1    (tied to embedding)    │
  ├──────────────────────────────────────────────────┤
  │  Total              25   19,521,920   78,087,680 │
  └──────────────────────────────────────────────────┘
```

### JSON Header Sample (abridged)

```json
{
  "__metadata__": {"format": "nntrainer"},
  "model_embed_tokens:weight": {
    "dtype": "F32",
    "shape": [151936, 64],
    "data_offsets": [0, 38895616]
  },
  "model_layers_0_input_layernorm:weight": {
    "dtype": "F32",
    "shape": [64],
    "data_offsets": [38895616, 38895872]
  },
  "model_layers_0_self_attn_q_proj:weight": {
    "dtype": "F32",
    "shape": [64, 64],
    "data_offsets": [38895872, 38912256]
  }
}
```

---

## 8. API Usage

### C++ API

```cpp
#include <model.h>

// Save as safetensors
model.save("model.safetensors", ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);

// Load from safetensors (auto name-based offset, parallel if INFERENCE)
model.load("model.safetensors", ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);

// Convert BIN to safetensors
model.convertBinToSafetensors("model.bin", "model.safetensors");
```

### Python API (TorchFXConverter)

```python
from weight_converter import WeightConverter

converter = WeightConverter(layers)

# Auto-detect from extension
converter.convert(state_dict, "model.safetensors")  # -> safetensors
converter.convert(state_dict, "model.bin")           # -> raw binary

# Explicit format
converter.convert(state_dict, "out.dat", output_format="safetensors")

# From pretrained
converter.convert_from_pretrained("Qwen/Qwen3-0.6B", "model.safetensors")
```

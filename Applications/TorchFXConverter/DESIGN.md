# Torch.FX to NNTrainer Converter - Design Document

## 1. Overview

This converter takes a PyTorch model (primarily HuggingFace models), traces it using
a runtime-based Torch.FX tracer (not symbolic tracing), and converts the resulting
FX graph into an NNTrainer model definition.

The converter works in 3 phases:
1. **Trace**: Run the model with real input using `TorchFunctionMode`-based tracer
2. **Map**: Convert FX graph nodes to NNTrainer layer/op definitions
3. **Emit**: Generate NNTrainer model construction code (or JSON config)

## 2. Conversion Priority Strategy

### Priority 1: Map to CausalLM Application-Level Layers (Coarse-Grained)

When a subgraph pattern matches a known CausalLM building block, convert it as a
single NNTrainer custom layer. This preserves the high-level semantics and leverages
optimized implementations (KV-cache, incremental inference, etc.).

| HuggingFace Pattern | NNTrainer CausalLM Layer | Type String |
|---|---|---|
| `model.embed_tokens` (nn.Embedding) | `EmbeddingLayer` / `TieWordEmbedding` | `"embedding_layer"` / `"tie_word_embeddings"` |
| Q/K/V Linear projections + Attention + O projection | `MHACoreLayer` (+ FC layers) | `"mha_core"` + `"fully_connected"` |
| SwiGLU(gate_proj, up_proj) + down_proj pattern | `SwiGLULayer` + FC layers | `"swiglu"` + `"fully_connected"` |
| RMSNorm (weight * x / rms(x)) | `RMSNormLayer` | `"rms_norm"` |
| Reshaped RMSNorm | `ReshapedRMSNormLayer` | `"reshaped_rms_norm"` |
| LM Head (final linear projection) | `LmHeadLayer` | `"lm_head"` |
| QKV fused projection | `QKVLayer` | `"qkv_layer"` |
| Embedding + L2 Normalize | `EmbeddingNormalizeLayer` | `"embedding_normalize"` |
| Embedding + Pooling (sentence-transformer) | `EmbeddingPoolingLayer` | `"embedding_pooling"` |

### Priority 2: Map to NNTrainer Built-in Layers (Medium-Grained)

For modules/patterns that don't match CausalLM blocks but correspond to NNTrainer's
built-in layer types:

| PyTorch nn.Module | NNTrainer Layer | Type String |
|---|---|---|
| `nn.Linear` | FullyConnectedLayer | `"fully_connected"` |
| `nn.Conv1d` | Conv1DLayer | `"conv1d"` |
| `nn.Conv2d` | Conv2DLayer | `"conv2d"` |
| `nn.Conv2dTranspose` | Conv2DTransposeLayer (NYI: Conv3d, ConvTranspose1d/3d) | `"conv2dtranspose"` |
| `nn.BatchNorm1d/2d/3d` | BatchNormalizationLayer | `"batch_normalization"` |
| `nn.LayerNorm` | LayerNormalizationLayer | `"layer_normalization"` |
| `nn.GroupNorm` | GroupNormLayer (NYI) | - |
| `nn.Embedding` | EmbeddingLayer (built-in) | `"embedding"` |
| `nn.ReLU` | ActivationLayer (act=relu) | `"activation"` with `activation=relu` |
| `nn.GELU` | ActivationLayer (act=gelu) | `"activation"` with `activation=gelu` |
| `nn.Sigmoid` | ActivationLayer (act=sigmoid) | `"activation"` with `activation=sigmoid` |
| `nn.Tanh` | ActivationLayer (act=tanh) | `"activation"` with `activation=tanh` |
| `nn.SiLU` | ActivationLayer (act=swish) | `"activation"` with `activation=swish` |
| `nn.Softmax` | ActivationLayer (act=softmax) | `"activation"` with `activation=softmax` |
| `nn.Dropout` | DropoutLayer | `"dropout"` |
| `nn.LSTM` | LSTMLayer | `"lstm"` |
| `nn.GRU` | GRULayer | `"gru"` |
| `nn.RNN` | RNNLayer | `"rnn"` |
| `nn.MultiheadAttention` | MultiHeadAttentionLayer | `"multi_head_attention"` |
| `nn.AdaptiveAvgPool2d` | Pooling2DLayer | `"pooling2d"` |
| `nn.AvgPool2d` | Pooling2DLayer | `"pooling2d"` |
| `nn.MaxPool2d` | Pooling2DLayer | `"pooling2d"` |
| Residual Add pattern | AdditionLayer | `"addition"` |
| `torch.cat` along axis | ConcatLayer | `"concat"` |
| `torch.split` | SplitLayer | `"split"` |
| `Tensor.reshape` / `Tensor.view` | ReshapeLayer | `"reshape"` |
| `Tensor.flatten` | FlattenLayer | `"flatten"` |
| `Tensor.permute` | PermuteLayer | `"permute"` |

### Priority 3: Map to NNTrainer Tensor Operations (Fine-Grained)

For operations that don't have a direct layer mapping, convert to NNTrainer's
tensor-level operations:

| PyTorch Operation | NNTrainer Layer/Op | Type String |
|---|---|---|
| `torch.add` / `Tensor.__add__` | AddLayer | `"add"` |
| `torch.sub` / `Tensor.__sub__` | SubtractLayer | `"subtract"` |
| `torch.mul` / `Tensor.__mul__` | MultiplyLayer | `"multiply"` |
| `torch.div` / `Tensor.__div__` | DivideLayer | `"divide"` |
| `torch.matmul` / `Tensor.matmul` | MatMulLayer | `"matmul"` |
| `torch.pow` | PowLayer | `"pow"` |
| `torch.sqrt` | SqrtLayer | `"sqrt"` |
| `torch.sin` | SinLayer | `"sin"` |
| `torch.cos` | CosLayer | `"cos"` |
| `torch.tan` | TanLayer | `"tan"` |
| `torch.neg` / `-x` | NegativeLayer | `"negative"` |
| `torch.mean` | ReduceMeanLayer | `"reduce_mean"` |
| `torch.sum` | ReduceSumLayer | `"reduce_sum"` |
| `Tensor[..., idx]` / gather | GatherLayer | `"gather"` |
| `Tensor[..., a:b]` / slice | SliceLayer | `"slice"` |
| `Tensor.to(dtype)` | CastLayer | `"cast"` |

### Priority 4: Unsupported Operations (Fallback)

Operations that cannot be mapped to any NNTrainer construct:
- Log a warning with the operation name and location in the graph
- Mark the node as `UNSUPPORTED` with full details
- Optionally generate a stub/placeholder layer for manual implementation

## 3. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TorchFXConverter                      │
│                                                          │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │  Tracer   │──▶│ PatternMatcher│──▶│  NNTrainer       │ │
│  │ (runtime) │   │ (subgraph    │   │  ModelEmitter    │ │
│  │           │   │  detection)  │   │  (code/config    │ │
│  │           │   │              │   │   generation)    │ │
│  └──────────┘   └──────────────┘   └──────────────────┘ │
│       │               │                    │             │
│       ▼               ▼                    ▼             │
│  FX Graph        Mapped Nodes      NNTrainer Model      │
│  (nodes with     (with nntrainer   (C++ code or         │
│   metadata)       layer mapping)    JSON config)        │
└─────────────────────────────────────────────────────────┘
```

### 3.1 Component Details

#### A. Tracer (Existing Code - Enhanced)

The provided `Tracer` class using `TorchFunctionMode` + module hooks.

**Enhancements needed:**
- Update `LEAF_MODULES` to include NNTrainer-mappable modules (see Section 4)
- Add metadata collection for weight shapes, dtypes, and parameter values
- Track module hierarchy for pattern matching

#### B. PatternMatcher

Detects subgraph patterns in the FX graph that correspond to CausalLM building blocks.

**Key patterns to detect:**

1. **Transformer Decoder Block Pattern:**
   ```
   RMSNorm -> Q_proj, K_proj, V_proj -> Attention -> O_proj -> Add (residual)
   -> RMSNorm -> gate_proj, up_proj -> SwiGLU -> down_proj -> Add (residual)
   ```

2. **RMSNorm Pattern:**
   ```
   variance = mean(x^2, dim=-1, keepdim=True)
   x_norm = x * rsqrt(variance + eps)
   output = weight * x_norm
   ```

3. **SwiGLU Pattern:**
   ```
   gate = silu(gate_proj(x))
   up = up_proj(x)
   output = gate * up
   ```

4. **Rotary Embedding Pattern:**
   ```
   cos/sin computation -> complex multiply -> apply to Q/K
   ```

#### C. NNTrainer ModelEmitter

Generates NNTrainer model construction code:

- **JSON Config Mode**: Produces `nntr_config.json` compatible with CausalLM app
- **C++ Code Mode**: Generates `createLayer()` calls matching `transformer.cpp` style
- **Weight Converter**: Generates `weight_converter.py` script for PyTorch→NNTrainer
  weight format conversion

## 4. Updated LEAF_MODULES

The `LEAF_MODULES` in the Tracer should be redefined to match NNTrainer's conversion
granularity:

```python
# Level 1: CausalLM Application-level leaf modules
# These are treated as atomic units and mapped to CausalLM custom layers
CAUSALLM_LEAF_MODULES = ()
# Note: These are detected by PatternMatcher as subgraph patterns,
# not as single nn.Modules. HuggingFace models decompose these
# differently per architecture.

# Level 2: NNTrainer built-in layer equivalents
NNTRAINER_LAYER_MODULES = (
    nn.Linear,           # -> fully_connected
    nn.Conv1d,           # -> conv1d
    nn.Conv2d,           # -> conv2d
    nn.ConvTranspose2d,  # -> conv2dtranspose
    nn.BatchNorm1d,      # -> batch_normalization
    nn.BatchNorm2d,      # -> batch_normalization
    nn.LayerNorm,        # -> layer_normalization
    nn.Embedding,        # -> embedding
    nn.LSTM,             # -> lstm
    nn.GRU,              # -> gru
    nn.RNN,              # -> rnn
    nn.MultiheadAttention,  # -> multi_head_attention
    nn.Dropout,          # -> dropout
)

# Level 3: Activation modules (mapped to nntrainer activation layer)
NNTRAINER_ACTIVATION_MODULES = (
    nn.ReLU,       # -> activation(relu)
    nn.ReLU6,      # -> activation(relu) [clamp needed]
    nn.GELU,       # -> activation(gelu)
    nn.SiLU,       # -> activation(swish)
    nn.Sigmoid,    # -> activation(sigmoid)
    nn.Tanh,       # -> activation(tanh)
    nn.Softmax,    # -> activation(softmax)
)

# HuggingFace-specific modules to treat as leaf (model-specific)
HUGGINGFACE_LEAF_MODULES = ()
# Populated dynamically based on model architecture detection:
#   - LlamaRMSNorm -> rms_norm
#   - LlamaMLP -> swiglu pattern
#   - LlamaAttention -> mha_core pattern
#   - Qwen2RMSNorm -> rms_norm
#   - GemmaRMSNorm -> rms_norm
#   etc.

# Combined LEAF_MODULES for the Tracer
LEAF_MODULES = (
    NNTRAINER_LAYER_MODULES +
    NNTRAINER_ACTIVATION_MODULES +
    tuple(HUGGINGFACE_LEAF_MODULES)
)
```

## 5. HuggingFace Model Architecture Detection

The converter should auto-detect the HuggingFace model architecture and configure
pattern matching accordingly:

```python
# Supported HuggingFace architectures and their module mappings
HF_ARCHITECTURE_MAP = {
    "LlamaForCausalLM": {
        "rms_norm": ["LlamaRMSNorm"],
        "attention": ["LlamaAttention", "LlamaSdpaAttention"],
        "mlp": ["LlamaMLP"],       # SwiGLU pattern
        "decoder_layer": ["LlamaDecoderLayer"],
        "embedding": ["embed_tokens"],
        "lm_head": ["lm_head"],
    },
    "Qwen2ForCausalLM": {
        "rms_norm": ["Qwen2RMSNorm"],
        "attention": ["Qwen2Attention", "Qwen2SdpaAttention"],
        "mlp": ["Qwen2MLP"],       # SwiGLU pattern
        "decoder_layer": ["Qwen2DecoderLayer"],
        "embedding": ["embed_tokens"],
        "lm_head": ["lm_head"],
    },
    "Qwen3ForCausalLM": {
        "rms_norm": ["Qwen3RMSNorm"],
        "attention": ["Qwen3Attention"],
        "mlp": ["Qwen3MLP"],
        "decoder_layer": ["Qwen3DecoderLayer"],
        "embedding": ["embed_tokens"],
        "lm_head": ["lm_head"],
    },
    "Qwen3MoeForCausalLM": {
        "rms_norm": ["Qwen3MoeRMSNorm"],
        "attention": ["Qwen3MoeAttention"],
        "mlp": ["Qwen3MoeSparseMoeBlock"],  # MoE pattern
        "decoder_layer": ["Qwen3MoeDecoderLayer"],
        "embedding": ["embed_tokens"],
        "lm_head": ["lm_head"],
    },
    "GemmaForCausalLM": {
        "rms_norm": ["GemmaRMSNorm"],
        "attention": ["GemmaAttention", "GemmaSdpaAttention"],
        "mlp": ["GemmaMLP"],       # GELU pattern
        "decoder_layer": ["GemmaDecoderLayer"],
        "embedding": ["embed_tokens"],
        "lm_head": ["lm_head"],
    },
    "Gemma3ForCausalLM": {
        "rms_norm": ["Gemma3RMSNorm"],
        "attention": ["Gemma3Attention"],
        "mlp": ["Gemma3MLP"],
        "decoder_layer": ["Gemma3DecoderLayer"],
        "embedding": ["embed_tokens"],
        "lm_head": ["lm_head"],
    },
}
```

## 6. Conversion Pipeline

```python
class TorchFXToNNTrainerConverter:
    def __init__(self, model, sample_input):
        self.model = model
        self.sample_input = sample_input
        self.arch = self._detect_architecture()

    def convert(self):
        # Phase 1: Trace
        leaf_modules = self._build_leaf_modules()
        tracer = Tracer(self.model, leaf_modules=leaf_modules)
        with tracer:
            output = self.model(**self.sample_input)
        graph = tracer.graph

        # Phase 2: Pattern Match & Map
        mapped_graph = self._map_nodes(graph)

        # Phase 3: Emit
        nntrainer_config = self._emit_config(mapped_graph)
        weight_converter = self._emit_weight_converter(mapped_graph)

        return nntrainer_config, weight_converter

    def _detect_architecture(self):
        """Detect HuggingFace model architecture from model class name."""
        ...

    def _build_leaf_modules(self):
        """Build LEAF_MODULES tuple based on detected architecture."""
        ...

    def _map_nodes(self, graph):
        """Map FX graph nodes to NNTrainer layer definitions."""
        for node in graph.nodes:
            if node.op == 'call_module':
                self._map_module_node(node)
            elif node.op == 'call_function':
                self._map_function_node(node)
            elif node.op == 'call_method':
                self._map_method_node(node)
        ...

    def _map_module_node(self, node):
        """Map a call_module node using Priority 1 -> 2 -> 3 -> 4."""
        ...

    def _map_function_node(self, node):
        """Map torch.* function calls to NNTrainer ops."""
        ...

    def _map_method_node(self, node):
        """Map tensor method calls (reshape, view, etc.) to NNTrainer ops."""
        ...
```

## 7. Weight Conversion Strategy

NNTrainer uses a flat binary weight format. The converter generates a Python script
that:

1. Loads PyTorch model state_dict
2. Reorders weights to match NNTrainer's layer ordering
3. Handles dtype conversion (FP32 -> FP16/INT8 as specified)
4. Handles transposition differences (NNTrainer FC weights may differ from PyTorch)
5. Saves in NNTrainer binary format

This follows the existing pattern in `Applications/CausalLM/res/*/weight_converter.py`.

## 8. Output Artifacts

The converter produces:
1. **`nntr_config.json`** - NNTrainer model configuration
2. **`weight_converter.py`** - Weight conversion script
3. **`conversion_report.txt`** - Mapping report showing:
   - Successfully mapped layers
   - Unsupported operations (with warnings)
   - Shape validation results

## 9. File Structure

```
Applications/TorchFXConverter/
├── DESIGN.md                    # This document
├── converter.py                 # Main converter class
├── tracer.py                    # Enhanced Tracer (from provided code)
├── pattern_matcher.py           # Subgraph pattern detection
├── node_mapper.py               # FX node -> NNTrainer layer mapping
├── model_emitter.py             # NNTrainer config/code generation
├── weight_converter_gen.py      # Weight converter script generator
├── hf_architectures.py          # HuggingFace architecture definitions
└── tests/
    ├── test_converter_linear.py
    ├── test_converter_llama.py
    └── test_converter_qwen.py
```

## 10. Key Design Decisions

### 10.1 Why Runtime Tracing (not Symbolic)?

HuggingFace models contain extensive conditionals (cache handling, attention mask
computation, sliding window logic, etc.) that make symbolic tracing impractical.
Runtime tracing with real inputs captures the exact execution path for a given
configuration.

### 10.2 Why Coarse-Grained Mapping First?

CausalLM layers like `MHACoreLayer` include critical optimizations:
- KV-cache management for incremental inference
- RoPE (Rotary Position Embedding) integration
- Sliding window attention
- GQA (Grouped Query Attention) support

Mapping at the fine-grained (tensor op) level would lose these optimizations.

### 10.3 Architecture-Specific vs Generic?

We support both:
- **Architecture-specific path**: For known HuggingFace architectures (Llama, Qwen,
  Gemma), use predefined patterns for reliable conversion
- **Generic path**: For unknown architectures, fall back to layer-by-layer mapping
  using Priority 2-3 rules

### 10.4 Handling of Unsupported Operations

Some PyTorch operations have no NNTrainer equivalent:
- `nn.ReLU6`: Use `relu` + clamp (or implement as custom layer)
- `nn.Hardswish`: Requires custom implementation
- `nn.PReLU`, `nn.LeakyReLU`: Not in NNTrainer's activation set
- `nn.AdaptiveAvgPool1d`, `nn.AvgPool1d`, `nn.MaxPool1d`: 1D pooling not available
  (use reshape + 2D pooling workaround)
- `nn.Dropout2d`, `nn.Dropout3d`: Use standard dropout
- `nn.GroupNorm`: Not available in NNTrainer

These will be flagged in the conversion report with suggested workarounds.

# Torch.FX General-Purpose HuggingFace-to-NNTrainer Converter

## Design Philosophy

This converter is **architecture-agnostic**. Rather than hardcoding patterns for
specific models (Qwen3, BERT, T5, etc.), it:

1. **Traces** any HuggingFace model using callback-based FX tracing
2. **Maps** traced leaf modules to NNTrainer layer types via a registry
3. **Detects patterns** (attention, FFN, residual connections) from graph topology
4. **Emits** NNTrainer-compatible output (C++ code, JSON config, weight converter)

The converter should work for:
- **Decoder-only**: Qwen3, LLaMA, GPT-2, Mistral
- **Encoder-only**: BERT, RoBERTa, sentence-transformers
- **Encoder-Decoder**: T5, mT5, BART

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Tracer     │───►│  Node Mapper │───►│   Pattern    │───►│   Emitter    │
│  (FX Graph)  │    │  (Leaf→NNTR) │    │  Detector    │    │  (C++/JSON)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                    │                   │                    │
   Callback-based     Module type →        Graph topology      Code generation
   hook tracing       NNTrainer layer      analysis             + weight map
```

### Module Layout

```
Applications/TorchFXConverter/
├── tracer.py              # FX graph tracer (callback-based)
├── node_mapper.py         # Maps FX nodes → NNTrainer layer definitions
├── decomposer.py          # Adaptive multi-pass conversion pipeline
├── pattern_detector.py    # Detects attention/FFN/residual patterns
├── emitter_cpp.py         # Generates C++ model construction code
├── emitter_ini.py         # Generates NNTrainer .ini configuration
├── emitter_json.py        # Generates JSON model config + weight map
├── weight_converter.py    # HF → NNTrainer binary weight conversion
├── converter.py           # Main CLI entry point (Phase 6)
├── nntrainer_layers.py    # NNTrainer layer type definitions & registry
├── DESIGN.md              # This file
└── tests/
    ├── test_tracer_simple.py
    ├── test_tracer_qwen3.py
    ├── test_node_mapper.py
    ├── test_decomposer.py
    ├── test_pattern_detector.py
    ├── test_coverage.py
    ├── test_multi_arch.py
    ├── test_unmapped_ops.py
    └── test_emitters.py   # C++, INI, JSON, weight converter tests
```

## NNTrainer Layer Type Mapping

| HuggingFace Module          | NNTrainer Layer Type     | Key Properties                          |
|-----------------------------|--------------------------|----------------------------------------|
| nn.Linear                   | fully_connected          | unit, disable_bias, weight_dtype       |
| nn.Embedding                | embedding_layer          | in_dim, out_dim, weight_dtype          |
| nn.Embedding (tied)         | tie_word_embeddings      | in_dim, out_dim, shared_from           |
| *RMSNorm                    | rms_norm                 | epsilon, packed                        |
| *RMSNorm (reshaped, Q/K)    | reshaped_rms_norm        | epsilon, feature_size                  |
| nn.LayerNorm                | layer_norm               | (for BERT/T5)                          |
| Attention pattern           | mha_core                 | num_heads, num_heads_kv, rope_theta... |
| SwiGLU pattern              | swiglu                   | input_layers                           |
| Residual add                | addition                 | input_layers                           |
| Input placeholder           | input                    | input_shape                            |

## Pattern Detection Strategy

Instead of matching fixed subgraph patterns (which breaks on new architectures),
we detect patterns from **module hierarchy** and **data flow**:

### 1. Attention Detection
- Find groups of Linear layers whose outputs flow into a common "attention" operation
- Identify Q/K/V/O projections by:
  - Module name matching (q_proj, k_proj, v_proj, o_proj / query, key, value)
  - Shape analysis (Q has num_heads * head_dim, K/V have num_kv_heads * head_dim)
- Detect post-projection norms (Qwen3's reshaped_rms_norm on Q/K)

### 2. FFN Detection
- Find groups of Linear layers within MLP/FFN modules
- Detect gate mechanism:
  - SwiGLU: up_proj + gate_proj → SiLU(gate) * up → down_proj
  - GELU FFN: fc1 → GELU → fc2 (BERT-style)
  - Standard FFN: fc1 → ReLU → fc2

### 3. Residual Connection Detection
- Track tensor flow: if output = input + sublayer_output, emit `addition` layer
- Detected from `torch.add` / `operator.add` in the FX graph

### 4. Normalization Detection
- RMSNorm modules → rms_norm
- LayerNorm modules → layer_norm (BERT/T5)
- Position: pre-norm (LLaMA/Qwen3) vs post-norm (BERT original)

## Phase Plan

### Phase 1: Tracer Foundation
- Task 1.1: Setup tracer.py with NNTrainer-adapted LEAF_MODULES
- Task 1.2: Trace a simple model and verify graph
- Task 1.3: Trace Qwen3-0.6B (or tiny config) and dump graph

### Phase 2: NNTrainer Layer Definitions & Node Mapper
- Task 2.1: Create nntrainer_layers.py with NNTrainerLayerDef dataclass
- Task 2.2: Create node_mapper.py mapping leaf modules to NNTrainer types
- Task 2.3: Map function/method nodes (add, matmul, reshape, etc.)
- Task 2.4: Map activation functions (SiLU, GELU, ReLU, softmax)

### Phase 3: Pattern Detection (Architecture-Agnostic)
- Task 3.1: Architecture detector from model.config
- Task 3.2: Attention pattern detection (works for GQA, MHA, MQA)
- Task 3.3: FFN pattern detection (SwiGLU, GELU-FFN, standard FFN)
- Task 3.4: Residual connection & normalization detection
- Task 3.5: Full decoder block assembly
- Task 3.6: Full encoder block assembly (for BERT/T5-encoder)
- Task 3.7: Encoder-decoder assembly (for T5/BART)

### Phase 4: Emitter ✓
- Task 4.1: C++ code emitter (emitter_cpp.py) ✓
- Task 4.2: INI config emitter (emitter_ini.py) ✓
- Task 4.3: JSON config emitter (emitter_json.py) ✓
- Task 4.4: Weight converter (weight_converter.py) ✓

### Phase 5: End-to-End Validation ✓
- Task 5.1: Qwen3 full pipeline + reference comparison ✓
- Task 5.2: BERT encoder-only test ✓
- Task 5.3: mT5 encoder-decoder test ✓
- Task 5.4: Layer-by-layer comparison with CausalLM reference ✓
- Task 5.5: INI section validation + connectivity check ✓
- Task 5.6: JSON schema validation + roundtrip ✓
- Task 5.7: Weight binary roundtrip verification ✓

### Phase 6: CLI ✓
- Task 6.1: converter.py main entry point ✓
- Task 6.2: Multi-format output (--format cpp ini json) ✓
- Task 6.3: Weight conversion (--weights --dtype float16) ✓

## Weight Conversion Rules

NNTrainer expects weights in a specific order:
1. **Linear (fully_connected)**: Transpose weight [out,in] → [in,out]
2. **Embedding**: Keep as-is [vocab, dim]
3. **RMSNorm**: Single weight vector
4. **LayerNorm**: gamma, beta (two vectors)
5. **Tied embeddings**: Shared weight reference

Weight save order follows the layer creation order in constructModel().

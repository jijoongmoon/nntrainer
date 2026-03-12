# Implementation Plan: Torch.FX Converter for Qwen3-0.6B

## Target Model
- **Model**: Qwen/Qwen3-0.6B (HuggingFace)
- **Architecture**: Qwen3ForCausalLM
- **Parameters**: hidden_size=1024, num_layers=28, num_heads=16, num_kv_heads=8, head_dim=128, intermediate_size=3072, vocab_size=151936
- **Key features**: tie_word_embeddings=true, Q/K norm (reshaped_rms_norm), SwiGLU FFN
- **Validation**: Compare against existing CausalLM/models/qwen3/ implementation

## Task Breakdown

### Phase 1: Tracer Foundation
Basic tracer that can trace a HuggingFace model and produce a valid FX graph.

#### Task 1.1: Setup tracer.py with the provided Tracer code
- Copy the provided Tracer code into `Applications/TorchFXConverter/tracer.py`
- Add minimal LEAF_MODULES for Qwen3
- **Test**: Trace a simple `nn.Linear` model, verify graph has placeholder + call_module + output nodes

#### Task 1.2: Trace Qwen3-0.6B and dump the graph
- Load Qwen3-0.6B from HuggingFace (or use a tiny config for testing)
- Configure LEAF_MODULES to capture Qwen3 modules as leaf nodes
- Run tracer with a sample input
- **Test**: Print `graph.print_tabular()`, verify all layers are captured
- Save graph node list to a file for reference

### Phase 2: Node Mapper
Map each FX graph node to an NNTrainer layer definition.

#### Task 2.1: Create node_mapper.py with basic module mapping
- Map `call_module` nodes for leaf modules:
  - `nn.Linear` -> `fully_connected`
  - `nn.Embedding` -> `embedding` (or `tie_word_embeddings`)
  - `nn.Dropout` -> `dropout`
- Store mapping result as a list of `NNTrainerLayerDef` dataclass objects
- **Test**: Trace a simple model with Linear+ReLU, verify mapping output

#### Task 2.2: Add function/method node mapping
- Map `call_function` nodes:
  - `torch.add` -> `add`, `torch.mul` -> `multiply`, `torch.matmul` -> `matmul`
  - `torch.cat` -> `concat`, `torch.split` -> `split`
  - `operator.getitem` -> pass through (tuple unpacking)
- Map `call_method` nodes:
  - `reshape`/`view` -> `reshape`, `permute` -> `permute`
  - `to` -> `cast`, `contiguous` -> no-op
- **Test**: Trace a model with tensor ops, verify correct mapping

#### Task 2.3: Add activation function mapping
- Map activation `call_function`/`call_method` nodes:
  - `F.silu` -> `activation(swish)`
  - `F.gelu` -> `activation(gelu)`
  - `F.relu` -> `activation(relu)`
  - `F.softmax` -> `activation(softmax)`
- Map activation `call_module` nodes (nn.SiLU, nn.GELU, etc.)
- **Test**: Trace a model with activations, verify mapping

### Phase 3: Qwen3-Specific Pattern Matching
Detect Qwen3 subgraph patterns and map to CausalLM layers.

#### Task 3.1: Create hf_architectures.py and detect Qwen3
- Detect model architecture from `model.config.architectures`
- Define Qwen3-specific module class names to watch for
- **Test**: Load Qwen3 config, verify architecture detection returns "Qwen3ForCausalLM"

#### Task 3.2: RMSNorm pattern detection
- Detect HuggingFace `Qwen3RMSNorm` modules in the traced graph
- Map to NNTrainer `rms_norm` layer with epsilon parameter
- **Test**: Verify RMSNorm nodes are correctly identified and mapped

#### Task 3.3: Attention pattern detection
- Detect the Qwen3 attention pattern:
  - FC(V) + FC(K) + ReshapedRMSNorm(K) + FC(Q) + ReshapedRMSNorm(Q) + MHACore + FC(O)
- Extract: num_heads, num_kv_heads, head_dim, rope_theta, sliding_window
- Map to NNTrainer layer sequence matching `qwen3_causallm.cpp:createAttention()`
- **Test**: Verify attention layers match the reference implementation

#### Task 3.4: SwiGLU FFN pattern detection
- Detect the Qwen3 MLP pattern:
  - FC(up_proj) + FC(gate_proj) + SwiGLU + FC(down_proj)
- Map to NNTrainer layer sequence matching `transformer.cpp:createMlp()`
- **Test**: Verify FFN layers match the reference implementation

#### Task 3.5: Transformer decoder block assembly
- Detect the full decoder block pattern:
  - RMSNorm -> Attention -> Addition(residual) -> RMSNorm -> FFN -> Addition(residual)
- Verify correct input_layers connections between blocks
- **Test**: Verify layer connectivity matches `transformer.cpp:createTransformerDecoderBlock()`

#### Task 3.6: Full model assembly
- Assemble: Input -> Embedding -> [DecoderBlock x 28] -> RMSNorm -> LMHead
- Handle tie_word_embeddings (Qwen3 uses shared weights)
- **Test**: Verify full layer list matches what `constructModel()` + `CausalLM::constructModel()` would produce

### Phase 4: Model Emitter
Generate NNTrainer-compatible output.

#### Task 4.1: Create model_emitter.py for nntr_config.json generation
- Generate the JSON config matching `res/qwen3/qwen3-4b/nntr_config.json` format
- Fill in model parameters from HuggingFace config
- **Test**: Compare generated JSON against reference config (adjusted for 0.6B params)

#### Task 4.2: Create C++ code emitter
- Generate C++ createLayer() calls matching `transformer.cpp` / `qwen3_causallm.cpp` style
- Output should be compilable with existing CausalLM build system
- **Test**: Diff generated code against existing Qwen3 implementation

#### Task 4.3: Create weight_converter.py generator
- Generate weight conversion script matching `res/qwen3/qwen3-4b/weight_converter.py` pattern
- Map HuggingFace state_dict keys to NNTrainer weight order
- Handle weight transposition (PyTorch Linear stores [out, in], nntrainer expects [in, out])
- Handle Q/K norm weights
- **Test**: Compare generated weight_converter.py against reference script

### Phase 5: End-to-End Validation

#### Task 5.1: Full pipeline test with Qwen3-0.6B
- Run complete converter: Trace -> Map -> Emit
- Compare generated nntr_config.json with what the existing code would produce
- Compare generated weight_converter.py with reference

#### Task 5.2: Layer-by-layer comparison
- For each layer in generated output, compare against reference:
  - Layer type string matches
  - Layer properties match (unit, num_heads, epsilon, etc.)
  - Input connections match
  - Weight ordering matches
- **Test**: Automated comparison script

#### Task 5.3: Weight conversion validation
- Convert actual Qwen3-0.6B weights using generated script
- Compare binary output against weights converted by reference script
- **Test**: Byte-level comparison of weight files

#### Task 5.4: Inference validation (if nntrainer build available)
- Load converted model in nntrainer CausalLM app
- Run inference with sample prompt
- Compare output against HuggingFace model output
- **Test**: Token-level comparison

### Phase 6: Converter CLI

#### Task 6.1: Create converter.py main entry point
- CLI interface: `python converter.py --model Qwen/Qwen3-0.6B --output ./output/`
- Options: --format (json/cpp), --dtype, --max_seq_len, --batch_size
- **Test**: Run CLI end-to-end

#### Task 6.2: Create conversion_report.txt output
- List all mapped layers with type strings
- Flag any unsupported operations
- Include model parameter summary
- **Test**: Verify report is generated and accurate

## Implementation Order

```
Phase 1 (Tracer)      ──► Phase 2 (Mapper)     ──► Phase 3 (Qwen3 Patterns)
  Task 1.1 → 1.2          Task 2.1 → 2.2 → 2.3    Task 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6
                                                          │
                                                          ▼
Phase 6 (CLI)         ◄── Phase 5 (Validation)  ◄── Phase 4 (Emitter)
  Task 6.1 → 6.2          Task 5.1 → 5.2 → 5.3     Task 4.1 → 4.2 → 4.3
```

## Validation Strategy

At each task, we verify correctness by comparing against the **existing reference implementation**:
- `Applications/CausalLM/models/transformer.cpp` - base transformer structure
- `Applications/CausalLM/models/qwen3/qwen3_causallm.cpp` - Qwen3 attention with Q/K norm
- `Applications/CausalLM/res/qwen3/qwen3-4b/nntr_config.json` - config format
- `Applications/CausalLM/res/qwen3/qwen3-4b/weight_converter.py` - weight mapping

The existing implementation serves as ground truth. The converter output must produce
functionally identical NNTrainer models.

# TorchFXConverter Architecture

## 1. Software Architecture

```mermaid
graph TB
    subgraph "User Interface"
        CLI["converter.py<br/>(CLI Entry Point)"]
        TestHarness["test_build_and_run.py<br/>(Integration Tests)"]
    end

    subgraph "Core Pipeline"
        Tracer["Tracer<br/>(torch.fx Graph Capture)"]
        NodeMapper["NodeMapper<br/>(FX Node → NNTrainer Layer)"]
        Decomposer["AdaptiveConverter<br/>(Multi-Pass Resolution)"]
        PatternDetector["PatternDetector<br/>(Block/Attention/FFN Detection)"]
    end

    subgraph "Mapper Layer"
        ModuleMapper["module_mapper<br/>(nn.Module → Layer)"]
        FunctionMapper["function_mapper<br/>(torch.* → Layer)"]
        MethodMapper["method_mapper<br/>(Tensor.* → Layer)"]
        OpRegistry["op_registry<br/>(Lookup Tables)"]
    end

    subgraph "Pattern Detection"
        ConfigExtractor["config.py<br/>(HF Config Metadata)"]
        ScopeAnalyzer["scope.py<br/>(Module Hierarchy)"]
        AttnDetector["attention.py<br/>(MHA Pattern)"]
        FFNDetector["ffn.py<br/>(MLP Pattern)"]
        BlockDetector["block.py<br/>(Transformer Block)"]
    end

    subgraph "Output Generation"
        CppEmitter["CppEmitter<br/>(.h + .cpp)"]
        IniEmitter["IniEmitter<br/>(.ini config)"]
        JsonEmitter["JsonEmitter<br/>(.json metadata)"]
        WeightConverter["WeightConverter<br/>(.bin weights)"]
    end

    subgraph "Model Support"
        HFAutoModel["HuggingFace<br/>AutoModel"]
        CustomLoaders["custom_models.py<br/>(GLiNER2, etc.)"]
    end

    subgraph "NNTrainer Runtime (C++)"
        NNTrainerLib["libnntrainer.so"]
        JNI["jni/main.cpp<br/>(Test Driver)"]
        MesonBuild["jni/meson.build"]
    end

    CLI --> Tracer
    CLI --> HFAutoModel
    CLI --> CustomLoaders
    HFAutoModel --> Tracer
    CustomLoaders --> Tracer

    Tracer -->|"torch.fx.Graph"| Decomposer
    Decomposer -->|"creates"| NodeMapper
    NodeMapper --> ModuleMapper
    NodeMapper --> FunctionMapper
    NodeMapper --> MethodMapper
    ModuleMapper --> OpRegistry
    FunctionMapper --> OpRegistry
    MethodMapper --> OpRegistry
    NodeMapper -->|"List&lt;NNTrainerLayerDef&gt;"| Decomposer
    Decomposer -->|"cleaned layers"| PatternDetector

    PatternDetector --> ConfigExtractor
    PatternDetector --> ScopeAnalyzer
    PatternDetector --> AttnDetector
    PatternDetector --> FFNDetector
    PatternDetector --> BlockDetector

    Decomposer -->|"ConversionResult"| CLI
    PatternDetector -->|"ModelStructure"| Decomposer

    CLI --> CppEmitter
    CLI --> IniEmitter
    CLI --> JsonEmitter
    CLI --> WeightConverter

    CppEmitter -->|".cpp/.h"| JNI
    JNI --> MesonBuild
    MesonBuild --> NNTrainerLib

    TestHarness --> CLI
    TestHarness --> MesonBuild
    TestHarness --> JNI
```

## 2. Class Diagram

```mermaid
classDiagram
    direction TB

    class Tracer {
        -root: nn.Module
        -graph: torch.fx.Graph
        -leaf_modules: tuple
        -_tensor_to_node: dict
        -_modules: dict
        -exclude_leaf_types: set
        +__init__(root, leaf_modules, exclude_leaf_types)
        +__torch_function__(func, types, args, kwargs)
        +__enter__()
        +__exit__(exc_type, exc_val, exc_tb)
        +get_leaf_modules() dict
        +print_graph_summary()
    }
    Tracer --|> TorchFunctionMode : inherits

    class AdaptiveConverter {
        -model: nn.Module
        -model_config: Any
        -training: bool
        +__init__(model, model_config, training)
        +convert(input_kwargs, max_passes) ConversionResult
        -_remove_passthrough_layers(layers, type, label)
        -_remove_position_id_chains(layers)
    }

    class ConversionResult {
        +layers: List~NNTrainerLayerDef~
        +decomposed_module_types: Set~str~
        +unsupported_ops: List
        +unknown_layers: List
        +tensor_ops: List
        +lazy_chains: List~LazyTensorChain~
        +graph: torch.fx.Graph
        +model_structure: ModelStructure
        +training: bool
        +is_fully_mapped: bool
        +summary()
    }

    class LazyTensorChain {
        +ops: List
        +to_cpp_chain(input_var) str
    }

    class NodeMapper {
        -model: nn.Module
        -graph: torch.fx.Graph
        -model_config: Any
        -node_to_layer: dict
        +__init__(model, graph, model_config)
        +map_all() List~NNTrainerLayerDef~
        +get_unknown_layers() List
        +get_unknown_module_types() Set~str~
        -_map_node(node) NNTrainerLayerDef
    }

    class NNTrainerLayerDef {
        +layer_type: str
        +name: str
        +properties: dict
        +input_layers: list
        +fx_node_name: str
        +hf_module_name: str
        +hf_module_type: str
        +has_weight: bool
        +has_bias: bool
        +weight_hf_key: str
        +bias_hf_key: str
        +transpose_weight: bool
        +shared_from: str
        +to_properties_list() list
        +to_cpp_call() str
    }

    class PatternDetector {
        -layers: List~NNTrainerLayerDef~
        -model_config: Any
        +__init__(layers, model_config)
        +detect() ModelStructure
    }

    class ModelStructure {
        +arch_type: str
        +model_type: str
        +embedding: dict
        +blocks: List~TransformerBlockPattern~
        +lm_head: dict
        +final_norm: dict
        +tie_word_embeddings: bool
        +vocab_size: int
        +hidden_size: int
        +num_layers: int
        +num_heads: int
        +num_kv_heads: int
        +head_dim: int
        +intermediate_size: int
        +rope_theta: float
        +norm_eps: float
        +num_encoder_layers: int
        +num_decoder_layers: int
    }

    class TransformerBlockPattern {
        +block_idx: int
        +block_role: str
        +pre_attn_norm: dict
        +attention: AttentionPattern
        +post_attn_norm: dict
        +attn_residual: dict
        +pre_ffn_norm: dict
        +ffn: FFNPattern
        +post_ffn_norm: dict
        +ffn_residual: dict
        +norm_type: str
        +cross_attention: AttentionPattern
        +operator_type: str
    }

    class AttentionPattern {
        +block_idx: int
        +q_proj: str
        +k_proj: str
        +v_proj: str
        +o_proj: str
        +q_norm: str
        +k_norm: str
        +sdpa: str
        +attention_type: str
        +num_heads: int
        +num_kv_heads: int
        +head_dim: int
        +has_rope: bool
        +has_qk_norm: bool
        +layer_names: list
    }

    class FFNPattern {
        +block_idx: int
        +ffn_type: str
        +gate_proj: str
        +up_proj: str
        +down_proj: str
        +activation: str
        +gate_multiply: str
        +intermediate_size: int
        +layer_names: list
    }

    class BaseEmitter {
        <<abstract>>
        #layers: List~NNTrainerLayerDef~
        #structure: ModelStructure
        #model_name: str
        +__init__(layers, structure, model_name)
        +emit()* str
        +format_property(key, value)$ str
    }

    class CppEmitter {
        +emit() str
        +emit_header() str
        +emit_source() str
    }

    class IniEmitter {
        -batch_size: int
        +emit(mode) str
    }

    class JsonEmitter {
        +emit() dict
        +emit_string(indent) str
    }

    class WeightConverter {
        -layers: List~NNTrainerLayerDef~
        -weight_map: WeightMap
        +__init__(layers)
        +convert(state_dict, output_path, dtype)
        +convert_from_pretrained(model_path, output_path, dtype)
        +generate_script() str
    }

    class WeightMap {
        -entries: list
        +add(hf_key, nntr_layer, transform, is_bias)
        +__iter__()
        +__len__()
    }

    %% Relationships
    AdaptiveConverter --> Tracer : creates & uses
    AdaptiveConverter --> NodeMapper : creates & uses
    AdaptiveConverter --> PatternDetector : creates & uses
    AdaptiveConverter --> ConversionResult : produces
    AdaptiveConverter --> LazyTensorChain : detects
    NodeMapper --> NNTrainerLayerDef : produces
    PatternDetector --> ModelStructure : produces
    ModelStructure --> TransformerBlockPattern : contains 0..*
    TransformerBlockPattern --> AttentionPattern : contains
    TransformerBlockPattern --> FFNPattern : contains
    ConversionResult --> ModelStructure : references
    ConversionResult --> NNTrainerLayerDef : contains 0..*
    ConversionResult --> LazyTensorChain : contains 0..*

    CppEmitter --|> BaseEmitter : inherits
    IniEmitter --|> BaseEmitter : inherits
    JsonEmitter --|> BaseEmitter : inherits
    BaseEmitter --> NNTrainerLayerDef : uses
    BaseEmitter --> ModelStructure : uses
    WeightConverter --> WeightMap : creates
    WeightConverter --> NNTrainerLayerDef : reads
```

### Mapper Dispatch Detail

```mermaid
classDiagram
    direction LR

    class NodeMapper {
        +map_all() List~NNTrainerLayerDef~
        -_map_node(node) NNTrainerLayerDef
    }

    class module_mapper {
        <<module>>
        +map_module_node(node, modules, node_to_layer) NNTrainerLayerDef
        +MULTI_OUTPUT_LAYER_TYPES: frozenset
    }

    class function_mapper {
        <<module>>
        +map_function_node(node, node_to_layer) NNTrainerLayerDef
        -_map_cat(node, scope, inputs)
        -_map_getitem(node, scope, node_to_layer)
    }

    class method_mapper {
        <<module>>
        +map_method_node(node) NNTrainerLayerDef
    }

    class op_registry {
        <<module>>
        +MODULE_TYPE_MAP: dict
        +FUNC_SIMPLE_OPS: dict
        +FUNC_ACTIVATION_OPS: dict
        +METHOD_SIMPLE_OPS: dict
        +METHOD_SHAPE_OPS: dict
        +METHOD_RESHAPE_NAMES: set
    }

    class mapper_helpers {
        <<module>>
        +get_input_node_names(node) list
        +sanitize_name(name) str
        +make_scoped_name(scope, node) str
        +extract_clamp_params(node, name, props)
    }

    NodeMapper --> module_mapper : call_module nodes
    NodeMapper --> function_mapper : call_function nodes
    NodeMapper --> method_mapper : call_method nodes
    module_mapper --> op_registry : type lookup
    function_mapper --> op_registry : func lookup
    method_mapper --> op_registry : method lookup
    module_mapper --> mapper_helpers : name utilities
    function_mapper --> mapper_helpers : name utilities
    method_mapper --> mapper_helpers : name utilities
```

## 3. Sequence Diagram — Full Conversion Pipeline

```mermaid
sequenceDiagram
    participant User
    participant CLI as converter.py
    participant HF as HuggingFace/<br/>CustomLoaders
    participant AC as AdaptiveConverter
    participant TR as Tracer
    participant NM as NodeMapper
    participant MM as module_mapper
    participant FM as function_mapper
    participant MeM as method_mapper
    participant PD as PatternDetector
    participant CE as CppEmitter
    participant WC as WeightConverter

    User->>CLI: python converter.py --model X --output Y
    CLI->>HF: Load model & config
    HF-->>CLI: model, config, trace_inputs

    CLI->>AC: AdaptiveConverter(model, config)
    CLI->>AC: convert(trace_inputs)

    Note over AC: === Pass 1: Trace ===
    AC->>TR: Tracer(model, LEAF_MODULES)
    AC->>TR: with tracer: model(**inputs)
    TR->>TR: __torch_function__() intercepts ops
    TR->>TR: Records nodes in fx.Graph
    TR-->>AC: tracer.graph (fx.Graph)

    Note over AC: === Pass 1: Map ===
    AC->>NM: NodeMapper(model, graph, config)
    AC->>NM: map_all()

    loop For each FX node
        NM->>NM: _map_node(node)
        alt call_module
            NM->>MM: map_module_node(node, modules)
            MM-->>NM: NNTrainerLayerDef
        else call_function
            NM->>FM: map_function_node(node)
            FM-->>NM: NNTrainerLayerDef
        else call_method
            NM->>MeM: map_method_node(node)
            MeM-->>NM: NNTrainerLayerDef
        end
    end
    NM-->>AC: List[NNTrainerLayerDef]

    Note over AC: === Pass 2 (if unknowns) ===
    AC->>AC: Check for unknown layers
    opt Has unknown module types
        AC->>TR: Re-trace with unknowns excluded
        TR-->>AC: Decomposed graph
        AC->>NM: map_all() on new graph
        NM-->>AC: Updated layers
    end

    Note over AC: === Pass 3: Cleanup ===
    AC->>AC: Remove dropout layers
    AC->>AC: Remove noop layers (expand, contiguous, etc.)
    AC->>AC: Remove position_id chains
    AC->>AC: Convert OP_RESHAPE → LAYER_RESHAPE
    AC->>AC: Add input layers & shape info
    AC->>AC: Fix slice axis/params (PyTorch→NCHW)
    AC->>AC: Fix gather axis (PyTorch→NCHW)
    AC->>AC: Detect lazy chains

    Note over AC: === Pattern Detection ===
    AC->>PD: PatternDetector(layers, config)
    AC->>PD: detect()
    PD->>PD: extract_config_metadata()
    PD->>PD: detect_embedding_and_head()
    PD->>PD: find_block_scopes()
    PD->>PD: detect_block() for each scope
    PD-->>AC: ModelStructure

    AC-->>CLI: ConversionResult

    Note over CLI: === Output Generation ===
    CLI->>CE: CppEmitter(layers, structure)
    CE-->>CLI: header.h + source.cpp

    opt --weights flag
        CLI->>WC: WeightConverter(layers)
        CLI->>WC: convert(state_dict, path, dtype)
        WC-->>CLI: weights.bin
    end

    CLI-->>User: Generated files
```

### Sequence Diagram — Build & Run Integration Test

```mermaid
sequenceDiagram
    participant Test as test_build_and_run.py
    participant Conv as converter.py<br/>(subprocess)
    participant FS as File System
    participant Meson as meson setup
    participant Ninja as ninja build
    participant Exec as Test Executable
    participant NNT as libnntrainer.so

    Test->>Test: Create synthetic model<br/>(tiny config, random weights)
    Test->>FS: Write config.json + weights

    Test->>Conv: subprocess: converter.py --model dir --output out
    Conv->>Conv: Load model, trace, map, detect patterns
    Conv->>FS: Write model.cpp + model.h
    Conv-->>Test: exit code 0

    Test->>FS: Copy .cpp/.h to jni/

    Test->>Meson: meson setup --reconfigure builddir
    Meson->>FS: Scan jni/ for new source files
    Meson-->>Test: Build files regenerated

    Test->>Ninja: ninja -C builddir target_name
    Ninja->>Ninja: Compile model.cpp + main.cpp
    Ninja->>NNT: Link against libnntrainer.so
    Ninja-->>Test: Executable built

    Test->>Exec: Run converter_model_test
    Exec->>Exec: model.initialize()
    Exec->>NNT: registerCustomLayers()
    Exec->>NNT: constructModel() — addLayer() for each layer
    Exec->>NNT: model->compile(INFERENCE)
    NNT->>NNT: Finalize each layer<br/>(validate shapes, allocate tensors)
    Exec->>NNT: model->initialize(INFERENCE)
    Exec->>NNT: model->summarize()
    Exec-->>Test: exit code 0 — SUCCESS

    Test->>FS: Cleanup generated files from jni/
```

## 4. Data Flow Diagram

```mermaid
flowchart LR
    subgraph Input
        HFModel["HuggingFace Model<br/>(nn.Module)"]
        HFConfig["Model Config<br/>(AutoConfig)"]
        TraceInputs["Trace Inputs<br/>(dict of Tensors)"]
    end

    subgraph "Tracing Phase"
        FXGraph["torch.fx.Graph<br/>• placeholder nodes<br/>• call_module nodes<br/>• call_function nodes<br/>• call_method nodes<br/>• output node<br/><br/>Each node has:<br/>  .meta[scope]<br/>  .meta[output_shape]<br/>  .meta[module_type]"]
    end

    subgraph "Mapping Phase"
        LayerList["List&lt;NNTrainerLayerDef&gt;<br/><br/>Each layer:<br/>  .layer_type (e.g. fully_connected)<br/>  .name (unique ID)<br/>  .properties (unit, axis, etc.)<br/>  .input_layers (connections)<br/>  .fx_node_name<br/>  .hf_module_name<br/>  .weight/bias keys"]
    end

    subgraph "Detection Phase"
        Structure["ModelStructure<br/><br/>  .arch_type (decoder_only, ...)<br/>  .model_type (qwen3, llama, ...)<br/>  .blocks[]: TransformerBlockPattern<br/>    .attention: AttentionPattern<br/>    .ffn: FFNPattern<br/>    .norms, residuals<br/>  .embedding, .lm_head<br/>  .hidden_size, .num_heads, ..."]
    end

    subgraph Output
        CppH[".h header<br/>(class declaration)"]
        CppS[".cpp source<br/>(constructModel,<br/> createBlock,<br/> initialize)"]
        INI[".ini config<br/>(layer definitions)"]
        JSON[".json metadata<br/>(model info,<br/> layer list,<br/> weight map)"]
        BIN[".bin weights<br/>(binary tensors)"]
    end

    HFModel --> FXGraph
    HFConfig --> FXGraph
    TraceInputs --> FXGraph
    FXGraph --> LayerList
    LayerList --> Structure
    HFConfig --> Structure
    LayerList --> CppH
    LayerList --> CppS
    LayerList --> INI
    LayerList --> JSON
    LayerList --> BIN
    Structure --> CppH
    Structure --> CppS
    Structure --> INI
    Structure --> JSON
```

## 5. NNTrainer Layer Type Mapping

Key PyTorch → NNTrainer layer type mappings used by the converter:

| PyTorch Operation | NNTrainer Layer | Properties |
|---|---|---|
| `nn.Linear` | `fully_connected` | unit, disable_bias |
| `nn.Embedding` | `embedding_layer` | in_dim, out_dim |
| `nn.LayerNorm` | `layer_normalization` | axis=3, epsilon |
| `RMSNorm` | `rms_norm` | epsilon, packed |
| `nn.Conv1d` | `conv1d` | filters, kernel_size, stride, padding |
| `nn.Dropout` | `dropout` | dropout_rate |
| `nn.ReLU/GELU/SiLU` | `activation` | activation type |
| `torch.cat` | `concat` | axis |
| `torch.gather` | `gather` | axis (1-3, NCHW) |
| `Tensor.view/reshape` | `reshape` | target_shape (C:H:W) |
| `Tensor.__getitem__` | `slice` | axis, start_index, end_index |
| `Tensor.mul` | `multiply` | (broadcasting supported) |
| `Tensor.add` | `addition` | — |
| `Tensor.softmax` | `activation` | activation=softmax |
| `Tensor.permute` | `permute` | — |
| `Tensor.transpose` | `transpose` | — |

### NCHW Dimension Convention

NNTrainer uses 4D `[Batch, Channel, Height, Width]` tensors. PyTorch tensors are mapped as:

| PyTorch Rank | NCHW Mapping | Axis Shift (dim > 0) |
|---|---|---|
| 4D `[B,C,H,W]` | Direct | +0 |
| 3D `[B,H,W]` | `[B, 1, H, W]` | +1 |
| 2D `[B,W]` | `[B, 1, 1, W]` | +2 |

Formula: `nchw_axis = pytorch_dim + (4 - tensor_rank)` for dims > 0.

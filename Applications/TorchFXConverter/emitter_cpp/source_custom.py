"""C++ registerCustomLayers() and initialize() method generation."""

from .helpers import _class_name


# Known NNTrainer layer type -> C++ class name mapping
# These must match the actual class names in CausalLM/layers/
CUSTOM_LAYER_CLASS = {
    "embedding_layer": "EmbeddingLayer",
    "tie_word_embeddings": "TieWordEmbedding",
    "rms_norm": "RMSNormLayer",
    "reshaped_rms_norm": "ReshapedRMSNormLayer",
    "mha_core": "MHACoreLayer",
    "swiglu": "SwiGLULayer",
    "short_conv": "ShortConvLayer",
}

# C++ class name -> header file mapping
CUSTOM_LAYER_HEADER = {
    "EmbeddingLayer": "embedding_layer.h",
    "TieWordEmbedding": "tie_word_embedding.h",
    "RMSNormLayer": "rms_norm.h",
    "ReshapedRMSNormLayer": "reshaped_rms_norm.h",
    "MHACoreLayer": "mha_core.h",
    "SwiGLULayer": "swiglu.h",
    "ShortConvLayer": "short_conv.h",
}


def collect_custom_layer_classes(structure, norm_type, attn_block):
    """Collect all custom C++ layer classes needed by this model.

    Args:
        structure: ModelStructure
        norm_type: "rms_norm" or "layer_normalization"
        attn_block: first block with attention, or None

    Returns:
        Sorted list of C++ class name strings.
    """
    s = structure
    classes = set()

    if s.embedding:
        classes.add("EmbeddingLayer")
    if s.tie_word_embeddings:
        classes.add("TieWordEmbedding")
    if norm_type == "rms_norm":
        classes.add("RMSNormLayer")

    if attn_block:
        classes.add("MHACoreLayer")
        if attn_block.attention.has_qk_norm:
            classes.add("ReshapedRMSNormLayer")

    for b in s.blocks:
        if b.ffn and b.ffn.ffn_type == "swiglu":
            classes.add("SwiGLULayer")
            break

    for b in s.blocks:
        for layer in (b.operator_layers or []):
            cls = CUSTOM_LAYER_CLASS.get(layer.layer_type)
            if cls:
                classes.add(cls)

    return sorted(classes)


def emit_custom_layer_includes(custom_classes):
    """Generate #include directives for custom layer headers.

    Args:
        custom_classes: sorted list of custom layer class names

    Returns:
        String of #include lines.
    """
    L = []
    for cls in custom_classes:
        header = CUSTOM_LAYER_HEADER.get(cls)
        if header:
            L.append(f"#include <{header}>")
    return "\n".join(L)


def emit_register_custom_layers(cname, custom_classes):
    """Generate registerCustomLayers() method body.

    Args:
        cname: C++ class name
        custom_classes: sorted list of custom layer class names
    """
    L = []

    L.append(f"void {cname}::registerCustomLayers() {{")
    L.append(f"  auto &ct_engine = nntrainer::Engine::Global();")
    L.append(f'  auto app_context =')
    L.append(f'    static_cast<nntrainer::AppContext *>('
             f'ct_engine.getRegisteredContext("cpu"));')
    L.append(f"")
    L.append(f"  try {{")

    for cls in custom_classes:
        L.append(f"    app_context->registerFactory("
                 f"nntrainer::createLayer<causallm::{cls}>);")

    L.append(f"  }} catch (std::invalid_argument &e) {{")
    L.append(f'    std::cerr << "failed to register factory, reason: " '
             f'<< e.what() << std::endl;')
    L.append(f"  }}")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)


def emit_initialize(cname):
    """Generate initialize() method that wires up the full model pipeline.

    Calls registerCustomLayers() -> constructModel() -> compile -> initialize.

    Args:
        cname: C++ class name
    """
    L = []
    L.append(f"void {cname}::initialize() {{")
    L.append(f"  // Set default sequence length if not configured")
    L.append(f"  if (INIT_SEQ_LEN == 0) {{")
    L.append(f"    INIT_SEQ_LEN = 8;")
    L.append(f"  }}")
    L.append(f"")
    L.append(f"  registerCustomLayers();")
    L.append(f"  constructModel();")
    L.append(f"")
    L.append(f"  model->setProperty({{")
    L.append(f'    withKey("batch_size", 1),')
    L.append(f'    withKey("epochs", "1"),')
    L.append(f'    withKey("model_tensor_type", "FP32-FP32")')
    L.append(f"  }});")
    L.append(f"")
    L.append(f"  if (model->compile(ml::train::ExecutionMode::INFERENCE)) {{")
    L.append(f'    throw std::invalid_argument("Model compilation failed.");')
    L.append(f"  }}")
    L.append(f"")
    L.append(f"  if (model->initialize(ml::train::ExecutionMode::INFERENCE)) {{")
    L.append(f'    throw std::invalid_argument("Model initialization failed.");')
    L.append(f"  }}")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)

"""C++ registerCustomLayers() method generation."""

from .helpers import _class_name


# Known NNTrainer layer type -> C++ class name mapping
CUSTOM_LAYER_CLASS = {
    "embedding_layer": "EmbeddingLayer",
    "tie_word_embeddings": "TieWordEmbeddingLayer",
    "rms_norm": "RMSNormLayer",
    "reshaped_rms_norm": "ReshapedRMSNormLayer",
    "mha_core": "MHACore",
    "swiglu": "SwiGLULayer",
    "short_conv": "ShortConvLayer",
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
        classes.add("TieWordEmbeddingLayer")
    if norm_type == "rms_norm":
        classes.add("RMSNormLayer")

    if attn_block:
        classes.add("MHACore")
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
                 f"nntrainer::createLayer<{cls}>);")

    L.append(f"  }} catch (std::invalid_argument &e) {{")
    L.append(f'    std::cerr << "failed to register factory, reason: " '
             f'<< e.what() << std::endl;')
    L.append(f"  }}")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)

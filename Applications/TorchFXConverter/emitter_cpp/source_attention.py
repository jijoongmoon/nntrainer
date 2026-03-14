"""C++ createAttention() method generation."""

from .helpers import _cpp_layer, _class_name


def emit_attention_method(cname, block):
    """Generate createAttention() method body."""
    attn = block.attention
    has_qk_norm = attn.has_qk_norm
    has_rope = attn.has_rope
    L = []

    L.append(f"std::vector<LayerHandle>")
    L.append(f"{cname}::createAttention(const int layer_id, int seq_len, "
             f"int n_heads,")
    L.append(f"                         int head_dim, "
             f"std::string query_name,")
    L.append(f"                         std::string key_name, "
             f"std::string value_name) {{")
    L.append(f"")
    L.append(f"  std::vector<LayerHandle> layers;")
    L.append(f"")
    L.append(f'  auto Q = "layer" + std::to_string(layer_id) + "_wq";')
    L.append(f'  auto K = "layer" + std::to_string(layer_id) + "_wk";')
    L.append(f'  auto V = "layer" + std::to_string(layer_id) + "_wv";')
    if has_qk_norm:
        L.append(f'  auto Q_norm = "layer" + std::to_string(layer_id) '
                 f'+ "_q_norm";')
        L.append(f'  auto K_norm = "layer" + std::to_string(layer_id) '
                 f'+ "_k_norm";')
    L.append(f'  auto A = "layer" + std::to_string(layer_id) '
             f'+ "_attention";')
    L.append(f'  auto O = "layer" + std::to_string(layer_id) '
             f'+ "_attention_out";')
    L.append(f"")

    # V layer
    L.append(f"  // V layer")
    L.extend(_cpp_layer("fully_connected", [
        'withKey("name", V)',
        'withKey("unit", head_dim * n_heads / GQA_SIZE)',
        'withKey("disable_bias", "true")',
        'withKey("input_layers", value_name)',
    ]))

    # K layer
    L.append(f"")
    L.append(f"  // K layer")
    L.extend(_cpp_layer("fully_connected", [
        'withKey("name", K)',
        'withKey("unit", head_dim * n_heads / GQA_SIZE)',
        'withKey("disable_bias", "true")',
        'withKey("input_layers", key_name)',
    ]))

    # Q layer
    L.append(f"")
    L.append(f"  // Q layer")
    L.extend(_cpp_layer("fully_connected", [
        'withKey("name", Q)',
        'withKey("unit", head_dim * n_heads)',
        'withKey("disable_bias", "true")',
        'withKey("input_layers", query_name)',
    ]))

    # Q/K norms (Qwen3-style)
    if has_qk_norm:
        L.append(f"")
        L.append(f"  // K norm (reshaped RMS norm)")
        L.extend(_cpp_layer("reshaped_rms_norm", [
            'withKey("name", K_norm)',
            'withKey("input_layers", K)',
            'withKey("packed", "false")',
            'withKey("epsilon", NORM_EPS)',
            'withKey("feature_size", head_dim)',
        ]))

        L.append(f"")
        L.append(f"  // Q norm (reshaped RMS norm)")
        L.extend(_cpp_layer("reshaped_rms_norm", [
            'withKey("name", Q_norm)',
            'withKey("input_layers", Q)',
            'withKey("packed", "false")',
            'withKey("epsilon", NORM_EPS)',
            'withKey("feature_size", head_dim)',
        ]))

    # MHA core
    L.append(f"")
    L.append(f"  // Attention core layer")
    q_in = "Q_norm" if has_qk_norm else "Q"
    k_in = "K_norm" if has_qk_norm else "K"

    mha_props = [
        'withKey("name", A)',
        'withKey("num_heads", n_heads)',
        'withKey("num_heads_kv", n_heads / GQA_SIZE)',
        'withKey("max_timestep", std::to_string(INIT_SEQ_LEN + '
        'NUM_TO_GENERATE))',
    ]
    mha_props.append('withKey("sliding_window", SLIDING_WINDOW)')
    if has_rope:
        mha_props.append('withKey("rope_theta", ROPE_THETA)')
        mha_props.append(
            'withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS)')
    mha_props.append('withKey("max_new_tokens", NUM_TO_GENERATE)')
    mha_props.append(
        f'withKey("input_layers", {{{q_in}, {k_in}, V}})')
    L.extend(_cpp_layer("mha_core", mha_props))

    # O layer
    L.append(f"")
    L.append(f"  // O layer")
    L.extend(_cpp_layer("fully_connected", [
        'withKey("name", O)',
        'withKey("unit", DIM)',
        'withKey("disable_bias", "true")',
        'withKey("input_layers", A)',
    ]))

    L.append(f"")
    L.append(f"  return layers;")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)

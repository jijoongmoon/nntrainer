"""C++ createAttention() method generation using symbolic Tensor API."""

from .helpers import _cpp_tensor_layer, _class_name


def emit_attention_method(cname, block, arch_type="decoder_only",
                          external_kv_cache=False):
    """Generate createAttention() method body using symbolic Tensor API."""
    attn = block.attention
    is_decoder = arch_type in ("decoder_only", "encoder_decoder")
    has_qk_norm = attn.has_qk_norm
    has_rope = attn.has_rope
    L = []

    L.append(f"Tensor")
    L.append(f"{cname}::createAttention(const int layer_id, int seq_len, "
             f"int n_heads,")
    L.append(f"                         int head_dim, "
             f"Tensor query, Tensor key, Tensor value) {{")
    L.append(f"")
    L.append(f'  auto prefix = "layer" + std::to_string(layer_id);')
    L.append(f"")

    # V layer
    L.append(f"  // V layer")
    L.extend(_cpp_tensor_layer("fully_connected", "v_out", [
        'withKey("name", prefix + "_wv")',
        'withKey("unit", head_dim * n_heads / GQA_SIZE)',
        'withKey("disable_bias", "true")',
    ], "value"))

    # K layer
    L.append(f"")
    L.append(f"  // K layer")
    L.extend(_cpp_tensor_layer("fully_connected", "k_out", [
        'withKey("name", prefix + "_wk")',
        'withKey("unit", head_dim * n_heads / GQA_SIZE)',
        'withKey("disable_bias", "true")',
    ], "key"))

    # Q layer
    L.append(f"")
    L.append(f"  // Q layer")
    L.extend(_cpp_tensor_layer("fully_connected", "q_out", [
        'withKey("name", prefix + "_wq")',
        'withKey("unit", head_dim * n_heads)',
        'withKey("disable_bias", "true")',
    ], "query"))

    # Q/K norms (Qwen3-style)
    if has_qk_norm:
        L.append(f"")
        L.append(f"  // K norm (reshaped RMS norm)")
        L.extend(_cpp_tensor_layer("reshaped_rms_norm", "k_normed", [
            'withKey("name", prefix + "_k_norm")',
            'withKey("packed", "false")',
            'withKey("epsilon", NORM_EPS)',
            'withKey("feature_size", head_dim)',
        ], "k_out"))

        L.append(f"")
        L.append(f"  // Q norm (reshaped RMS norm)")
        L.extend(_cpp_tensor_layer("reshaped_rms_norm", "q_normed", [
            'withKey("name", prefix + "_q_norm")',
            'withKey("packed", "false")',
            'withKey("epsilon", NORM_EPS)',
            'withKey("feature_size", head_dim)',
        ], "q_out"))

    # MHA core
    L.append(f"")
    L.append(f"  // Attention core layer")
    q_in = "q_normed" if has_qk_norm else "q_out"
    k_in = "k_normed" if has_qk_norm else "k_out"

    mha_props = [
        'withKey("name", prefix + "_attention")',
        'withKey("num_heads", n_heads)',
        'withKey("num_heads_kv", n_heads / GQA_SIZE)',
    ]
    if is_decoder:
        mha_props.append(
            'withKey("max_timestep", std::to_string(INIT_SEQ_LEN + '
            'NUM_TO_GENERATE))')
        mha_props.append('withKey("sliding_window", SLIDING_WINDOW)')
    else:
        mha_props.append(
            'withKey("max_timestep", std::to_string(INIT_SEQ_LEN))')
    if has_rope:
        mha_props.append('withKey("rope_theta", ROPE_THETA)')
        mha_props.append(
            'withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS)')
    if is_decoder:
        mha_props.append('withKey("max_new_tokens", NUM_TO_GENERATE)')

    # MHA core takes multiple inputs: Q, K, V (and optionally KV cache tensors)
    if external_kv_cache:
        mha_inputs = [q_in, k_in, "v_out",
                      "key_cache_tensor_names[layer_id]",
                      "val_cache_tensor_names[layer_id]"]
    else:
        mha_inputs = [q_in, k_in, "v_out"]

    L.extend(_cpp_tensor_layer("mha_core", "attn_out",
                               mha_props, mha_inputs))

    # O layer
    L.append(f"")
    L.append(f"  // O layer")
    L.extend(_cpp_tensor_layer("fully_connected", "o_out", [
        'withKey("name", prefix + "_attention_out")',
        'withKey("unit", DIM)',
        'withKey("disable_bias", "true")',
    ], "attn_out"))

    L.append(f"")
    L.append(f"  return o_out;")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)

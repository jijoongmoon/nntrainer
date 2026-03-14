"""C++ createTransformerBlock() and operator layer generation."""

from .helpers import _cpp_layer, get_norm_type


def emit_block_method(cname, block_type, block, structure, is_encoder=None):
    """Emit a block method for any operator type.

    The method structure is always:
      norm -> operator -> residual -> norm -> FFN -> residual

    For attention blocks, the operator is emitted via createAttention.
    For other blocks (conv, ssm, etc.), the operator layers are emitted
    directly from block.operator_layers.
    """
    norm_type = get_norm_type(structure.model_type)
    op_type = block.operator_type
    L = []

    # Determine prefix for layer names
    if is_encoder is True:
        prefix_expr = '"enc_layer" + std::to_string(layer_id)'
    elif is_encoder is False:
        prefix_expr = '"dec_layer" + std::to_string(layer_id)'
    else:
        prefix_expr = '"layer" + std::to_string(layer_id)'

    # Method signature
    L.append(f"std::vector<LayerHandle>")
    if block_type in ("EncoderBlock",):
        L.append(f"{cname}::create{block_type}("
                 f"const int layer_id,")
        L.append(f"  std::string input_name) {{")
    elif block_type == "DecoderBlock" and is_encoder is False:
        L.append(f"{cname}::create{block_type}("
                 f"const int layer_id,")
        L.append(f"  std::string input_name, "
                 f"std::string encoder_output) {{")
    else:
        op_label = (op_type.capitalize() if op_type != "attention"
                    else "Transformer")
        method_name = (f"createTransformer{block_type}"
                       if op_type == "attention"
                       else f"create{op_label}{block_type}")
        L.append(f"{cname}::{method_name}("
                 f"const int layer_id,")
        L.append(f"  std::string input_name) {{")

    L.append(f"")
    L.append(f"  std::vector<LayerHandle> layers;")
    L.append(f"  auto prefix = {prefix_expr};")
    L.append(f"")

    norm_suffix = ("_attention_norm" if op_type == "attention"
                   else f"_{op_type}_norm")

    # Pre-operator norm
    if block.pre_attn_norm:
        _emit_pre_norm(L, norm_type, norm_suffix, op_type)

    # Operator
    op_input = (f'prefix + "{norm_suffix}"'
                if block.pre_attn_norm else "input_name")
    if block.attention:
        _emit_attention_call(L, op_input)
        op_output_suffix = "_attention_out"
    elif block.operator_layers:
        op_output_suffix = _emit_operator_layers(L, block, op_input, prefix_expr)
    else:
        op_output_suffix = norm_suffix

    # Operator residual
    if block.attn_residual:
        residual_suffix = (f"_{op_type}_add" if op_type != "attention"
                           else "_self_attn_add")
        _emit_residual(L, op_type, residual_suffix, op_output_suffix)
        last_residual = f'prefix + "{residual_suffix}"'
    else:
        last_residual = f'prefix + "{op_output_suffix}"'

    # Cross-attention (decoder blocks in encoder-decoder)
    if block.cross_attention and is_encoder is False:
        last_residual = _emit_cross_attention(L, block, norm_type, last_residual)

    # Block output naming
    if is_encoder is None:
        block_out_name = '"_decoder_output"'
    else:
        block_out_name = '"_block_output"'

    # Pre-FFN norm
    if block.pre_ffn_norm:
        _emit_pre_ffn_norm(L, norm_type, last_residual)
        ffn_in = 'prefix + "_ffn_norm"'
    else:
        ffn_in = last_residual

    # FFN
    if block.ffn:
        L.append(f"  auto ffn_layer = createMlp(layer_id, DIM, "
                 f"INTERMEDIATE_SIZE,")
        L.append(f"                             {ffn_in});")
        L.append(f"  layers.insert(layers.end(), ffn_layer.begin(), "
                 f"ffn_layer.end());")
        L.append(f"")

    # FFN residual
    if block.ffn_residual:
        L.append(f"  // FFN residual connection")
        L.extend(_cpp_layer("addition", [
            f'withKey("name", prefix + {block_out_name})',
            f'withKey("input_layers", {last_residual} + "," + '
            f'prefix + "_ffn_down")',
        ]))
        L.append(f"")

    L.append(f"  return layers;")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)


def _emit_pre_norm(L, norm_type, norm_suffix, op_type):
    """Emit pre-operator normalization."""
    L.append(f"  // Pre-{op_type} normalization")
    norm_props = [
        f'withKey("name", prefix + "{norm_suffix}")',
        'withKey("input_layers", input_name)',
        'withKey("epsilon", NORM_EPS)',
    ]
    if norm_type == "rms_norm":
        norm_props.append('withKey("packed", "false")')
    L.extend(_cpp_layer(norm_type, norm_props))
    L.append(f"")


def _emit_attention_call(L, op_input):
    """Emit createAttention() call."""
    L.append(f"  auto att_layer =")
    L.append(f"    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, "
             f"HEAD_DIM,")
    L.append(f"                    {op_input},")
    L.append(f"                    {op_input},")
    L.append(f"                    {op_input});")
    L.append(f"")
    L.append(f"  layers.insert(layers.end(), att_layer.begin(), "
             f"att_layer.end());")
    L.append(f"")


def _emit_residual(L, op_type, residual_suffix, op_output_suffix):
    """Emit operator residual connection."""
    L.append(f"  // {op_type.capitalize()} residual connection")
    L.extend(_cpp_layer("addition", [
        f'withKey("name", prefix + "{residual_suffix}")',
        f'withKey("input_layers", input_name + "," + '
        f'prefix + "{op_output_suffix}")',
    ]))
    L.append(f"")


def _emit_cross_attention(L, block, norm_type, last_residual):
    """Emit cross-attention section. Returns updated last_residual."""
    if block.cross_attn_norm:
        L.append(f"  // Cross-attention normalization")
        cross_norm_props = [
            'withKey("name", prefix + "_cross_attn_norm")',
            f'withKey("input_layers", {last_residual})',
            'withKey("epsilon", NORM_EPS)',
        ]
        if norm_type == "rms_norm":
            cross_norm_props.append('withKey("packed", "false")')
        L.extend(_cpp_layer(norm_type, cross_norm_props))
        L.append(f"")
        cross_q = 'prefix + "_cross_attn_norm"'
    else:
        cross_q = last_residual

    L.append(f"  auto cross_att =")
    L.append(f"    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, "
             f"HEAD_DIM,")
    L.append(f"                    {cross_q},")
    L.append(f"                    encoder_output,")
    L.append(f"                    encoder_output);")
    L.append(f"")
    L.append(f"  layers.insert(layers.end(), cross_att.begin(), "
             f"cross_att.end());")
    L.append(f"")

    if block.cross_attn_residual:
        L.append(f"  // Cross-attention residual")
        L.extend(_cpp_layer("addition", [
            'withKey("name", prefix + "_cross_attn_add")',
            f'withKey("input_layers", {last_residual} + "," + '
            f'prefix + "_attention_out")',
        ]))
        L.append(f"")
        return 'prefix + "_cross_attn_add"'
    return last_residual


def _emit_pre_ffn_norm(L, norm_type, last_residual):
    """Emit pre-FFN normalization."""
    L.append(f"  // Pre-FFN normalization")
    norm_props = [
        'withKey("name", prefix + "_ffn_norm")',
        f'withKey("input_layers", {last_residual})',
        'withKey("epsilon", NORM_EPS)',
    ]
    if norm_type == "rms_norm":
        norm_props.append('withKey("packed", "false")')
    L.extend(_cpp_layer(norm_type, norm_props))
    L.append(f"")


def _emit_operator_layers(L, block, op_input, prefix_expr):
    """Emit non-attention operator layers (auto-generated from model).

    Returns the suffix of the last emitted layer's name (for residual).
    """
    op_type = block.operator_type
    scope = block.operator_scope
    scope_san = scope.replace(".", "_")
    block_scope = scope.rsplit(".", 1)[0] if "." in scope else scope
    block_scope_san = block_scope.replace(".", "_")

    L.append(f"  // {op_type.capitalize()} operator (auto-generated)")
    last_suffix = ""
    prev_suffix = None

    for i, layer in enumerate(block.operator_layers):
        if layer.name.startswith(block_scope_san + "_"):
            suffix = layer.name[len(block_scope_san):]
        else:
            suffix = "_" + layer.name

        if i == 0:
            input_expr = op_input
        else:
            input_expr = f'prefix + "{prev_suffix}"'

        props = [f'withKey("name", prefix + "{suffix}")']
        for k, v in layer.properties.items():
            if isinstance(v, bool):
                props.append(f'withKey("{k}", '
                             f'"{str(v).lower()}")')
            elif isinstance(v, str):
                props.append(f'withKey("{k}", "{v}")')
            else:
                props.append(f'withKey("{k}", {v})')
        props.append(f'withKey("input_layers", {input_expr})')
        L.extend(_cpp_layer(layer.layer_type, props))
        L.append(f"")

        prev_suffix = suffix
        last_suffix = suffix

    return last_suffix

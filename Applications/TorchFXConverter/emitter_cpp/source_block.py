"""C++ createTransformerBlock() and operator layer generation using Tensor API."""

from .helpers import _cpp_tensor_layer, get_norm_type


def emit_block_method(cname, block_type, block, structure, is_encoder=None):
    """Emit a block method for any operator type using symbolic Tensor API.

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

    # Method signature - returns Tensor, accepts Tensor input
    L.append(f"Tensor")
    if block_type in ("EncoderBlock",):
        L.append(f"{cname}::create{block_type}("
                 f"const int layer_id,")
        L.append(f"  Tensor input) {{")
    elif block_type == "DecoderBlock" and is_encoder is False:
        L.append(f"{cname}::create{block_type}("
                 f"const int layer_id,")
        L.append(f"  Tensor input, Tensor encoder_output) {{")
    else:
        op_label = (op_type.capitalize() if op_type != "attention"
                    else "Transformer")
        method_name = (f"createTransformer{block_type}"
                       if op_type == "attention"
                       else f"create{op_label}{block_type}")
        L.append(f"{cname}::{method_name}("
                 f"const int layer_id,")
        L.append(f"  Tensor input) {{")

    L.append(f"")
    L.append(f"  auto prefix = {prefix_expr};")
    L.append(f"")

    norm_suffix = ("_attention_norm" if op_type == "attention"
                   else f"_{op_type}_norm")

    # Pre-operator norm
    if block.pre_attn_norm:
        _emit_pre_norm(L, norm_type, norm_suffix, op_type)
        cur_var = "normed"
    else:
        cur_var = "input"

    # Operator
    if block.attention:
        _emit_attention_call(L, cur_var)
        op_out_var = "att_out"
    elif block.operator_layers:
        op_out_var = _emit_operator_layers(L, block, cur_var, prefix_expr)
    else:
        op_out_var = cur_var

    # Operator residual
    if block.attn_residual:
        residual_suffix = (f"_{op_type}_add" if op_type != "attention"
                           else "_self_attn_add")
        _emit_residual(L, op_type, residual_suffix, op_out_var)
        cur_residual_var = "residual"
    else:
        cur_residual_var = op_out_var

    # Cross-attention (decoder blocks in encoder-decoder)
    if block.cross_attention and is_encoder is False:
        cur_residual_var = _emit_cross_attention(
            L, block, norm_type, cur_residual_var)

    # Block output naming
    if is_encoder is None:
        block_out_name = '"_decoder_output"'
    else:
        block_out_name = '"_block_output"'

    # Pre-FFN norm
    if block.pre_ffn_norm:
        _emit_pre_ffn_norm(L, norm_type, cur_residual_var)
        ffn_in_var = "ffn_normed"
    else:
        ffn_in_var = cur_residual_var

    # FFN
    if block.ffn:
        L.append(f"  auto ffn_out = createMlp(layer_id, DIM, "
                 f"INTERMEDIATE_SIZE, {ffn_in_var});")
        L.append(f"")

    # FFN residual
    if block.ffn_residual:
        L.append(f"  // FFN residual connection")
        ffn_res_props = [
            f'withKey("name", prefix + {block_out_name})',
        ]
        L.extend(_cpp_tensor_layer("addition", "block_out",
                                   ffn_res_props,
                                   [cur_residual_var, "ffn_out"]))
        L.append(f"")
        L.append(f"  return block_out;")
    else:
        L.append(f"  return ffn_out;")

    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)


def _emit_pre_norm(L, norm_type, norm_suffix, op_type):
    """Emit pre-operator normalization using symbolic tensor API."""
    L.append(f"  // Pre-{op_type} normalization")
    norm_props = [
        f'withKey("name", prefix + "{norm_suffix}")',
        'withKey("epsilon", NORM_EPS)',
    ]
    if norm_type == "rms_norm":
        norm_props.append('withKey("packed", "false")')
    elif norm_type == "layer_normalization":
        norm_props.append('withKey("axis", 3)')
    L.extend(_cpp_tensor_layer(norm_type, "normed", norm_props, "input"))
    L.append(f"")


def _emit_attention_call(L, input_var):
    """Emit createAttention() call with symbolic tensors."""
    L.append(f"  auto att_out =")
    L.append(f"    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, "
             f"HEAD_DIM,")
    L.append(f"                    {input_var}, {input_var}, {input_var});")
    L.append(f"")


def _emit_residual(L, op_type, residual_suffix, op_out_var):
    """Emit operator residual connection using symbolic tensor API."""
    L.append(f"  // {op_type.capitalize()} residual connection")
    res_props = [
        f'withKey("name", prefix + "{residual_suffix}")',
    ]
    L.extend(_cpp_tensor_layer("addition", "residual",
                               res_props, ["input", op_out_var]))
    L.append(f"")


def _emit_cross_attention(L, block, norm_type, input_var):
    """Emit cross-attention section. Returns updated variable name."""
    if block.cross_attn_norm:
        L.append(f"  // Cross-attention normalization")
        cross_norm_props = [
            'withKey("name", prefix + "_cross_attn_norm")',
            'withKey("epsilon", NORM_EPS)',
        ]
        if norm_type == "rms_norm":
            cross_norm_props.append('withKey("packed", "false")')
        elif norm_type == "layer_normalization":
            cross_norm_props.append('withKey("axis", 3)')
        L.extend(_cpp_tensor_layer(norm_type, "cross_normed",
                                   cross_norm_props, input_var))
        L.append(f"")
        cross_q_var = "cross_normed"
    else:
        cross_q_var = input_var

    L.append(f"  auto cross_att_out =")
    L.append(f"    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, "
             f"HEAD_DIM,")
    L.append(f"                    {cross_q_var}, encoder_output, "
             f"encoder_output);")
    L.append(f"")

    if block.cross_attn_residual:
        L.append(f"  // Cross-attention residual")
        cross_res_props = [
            'withKey("name", prefix + "_cross_attn_add")',
        ]
        L.extend(_cpp_tensor_layer("addition", "cross_residual",
                                   cross_res_props,
                                   [input_var, "cross_att_out"]))
        L.append(f"")
        return "cross_residual"
    return input_var


def _emit_pre_ffn_norm(L, norm_type, input_var):
    """Emit pre-FFN normalization using symbolic tensor API."""
    L.append(f"  // Pre-FFN normalization")
    norm_props = [
        'withKey("name", prefix + "_ffn_norm")',
        'withKey("epsilon", NORM_EPS)',
    ]
    if norm_type == "rms_norm":
        norm_props.append('withKey("packed", "false")')
    elif norm_type == "layer_normalization":
        norm_props.append('withKey("axis", 3)')
    L.extend(_cpp_tensor_layer(norm_type, "ffn_normed",
                               norm_props, input_var))
    L.append(f"")


def _emit_operator_layers(L, block, input_var, prefix_expr):
    """Emit non-attention operator layers (auto-generated from model).

    Returns the variable name of the last emitted layer's output.
    """
    op_type = block.operator_type
    scope = block.operator_scope
    scope_san = scope.replace(".", "_")
    block_scope = scope.rsplit(".", 1)[0] if "." in scope else scope
    block_scope_san = block_scope.replace(".", "_")

    L.append(f"  // {op_type.capitalize()} operator (auto-generated)")
    prev_var = None
    last_var = input_var

    for i, layer in enumerate(block.operator_layers):
        if layer.name.startswith(block_scope_san + "_"):
            suffix = layer.name[len(block_scope_san):]
        else:
            suffix = "_" + layer.name

        cur_input = input_var if i == 0 else prev_var
        var_name = f"op_{i}"

        props = [f'withKey("name", prefix + "{suffix}")']
        for k, v in layer.properties.items():
            if isinstance(v, bool):
                props.append(f'withKey("{k}", '
                             f'"{str(v).lower()}")')
            elif isinstance(v, str):
                props.append(f'withKey("{k}", "{v}")')
            else:
                props.append(f'withKey("{k}", {v})')
        L.extend(_cpp_tensor_layer(layer.layer_type, var_name,
                                   props, cur_input))
        L.append(f"")

        prev_var = var_name
        last_var = var_name

    return last_var

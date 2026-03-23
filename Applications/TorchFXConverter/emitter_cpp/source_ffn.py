"""C++ createMlp() method generation using symbolic Tensor graph."""

from .helpers import _cpp_tensor_layer, _class_name


def emit_ffn_method(cname, block):
    """Generate createMlp() method body using Tensor flow."""
    ffn = block.ffn
    L = []

    L.append(f"Tensor {cname}::createMlp(")
    L.append(f"  const int layer_id, int dim, int hidden_dim,")
    L.append(f"  Tensor input) {{")
    L.append(f"")
    L.append(f"  using ml::train::createLayer;")
    L.append(f"")

    is_gated = (ffn.ffn_type in ("swiglu", "geglu")
                or ffn.ffn_type.startswith("gated_"))

    if ffn.ffn_type == "swiglu":
        last_var = _emit_swiglu_ffn(L)
    elif is_gated:
        last_var = _emit_gated_ffn(L, ffn)
    else:
        last_var = _emit_standard_ffn(L, ffn)

    L.append(f"")
    L.append(f"  return {last_var};")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)


def _emit_swiglu_ffn(L):
    """Emit SwiGLU FFN layers using Tensor flow. Returns last output var."""
    # Up projection
    lines, up_out = _cpp_tensor_layer("ffn_up", "fully_connected", [
        'withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up")',
        'withKey("unit", hidden_dim)',
        'withKey("disable_bias", "true")',
    ], "input")
    L.extend(lines)
    L.append(f"")

    # Gate projection
    lines, gate_out = _cpp_tensor_layer("ffn_gate", "fully_connected", [
        'withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate")',
        'withKey("unit", hidden_dim)',
        'withKey("disable_bias", "true")',
    ], "input")
    L.extend(lines)
    L.append(f"")

    # SwiGLU activation
    lines, swiglu_out = _cpp_tensor_layer("swiglu", "swiglu", [
        'withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu")',
    ], f'{{{up_out}, {gate_out}}}')
    L.extend(lines)
    L.append(f"")

    # Down projection
    lines, down_out = _cpp_tensor_layer("ffn_down", "fully_connected", [
        'withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down")',
        'withKey("unit", dim)',
        'withKey("disable_bias", "true")',
    ], swiglu_out)
    L.extend(lines)

    return down_out


def _emit_gated_ffn(L, ffn):
    """Emit gated FFN (GeGLU, gated_relu, etc.) with separate activation
    and multiply layers. Returns last output var."""
    # Determine activation type from ffn_type
    act_map = {"geglu": "gelu", "gated_relu": "relu",
               "gated_tanh": "tanh", "gated_sigmoid": "sigmoid"}
    act = act_map.get(ffn.ffn_type, "gelu")

    # Gate projection
    lines, gate_out = _cpp_tensor_layer("ffn_gate", "fully_connected", [
        'withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate")',
        'withKey("unit", hidden_dim)',
        'withKey("disable_bias", "true")',
    ], "input")
    L.extend(lines)
    L.append(f"")

    # Gate activation
    lines, act_out = _cpp_tensor_layer("ffn_act", "activation", [
        'withKey("name", "layer" + std::to_string(layer_id) + "_ffn_act")',
        f'withKey("activation", "{act}")',
    ], gate_out)
    L.extend(lines)
    L.append(f"")

    # Up projection
    lines, up_out = _cpp_tensor_layer("ffn_up", "fully_connected", [
        'withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up")',
        'withKey("unit", hidden_dim)',
        'withKey("disable_bias", "true")',
    ], "input")
    L.extend(lines)
    L.append(f"")

    # Multiply gate_activation * up
    lines, mul_out = _cpp_tensor_layer("ffn_mul", "multiply", [
        'withKey("name", "layer" + std::to_string(layer_id) + "_ffn_mul")',
    ], f'{{{act_out}, {up_out}}}')
    L.extend(lines)
    L.append(f"")

    # Down projection
    lines, down_out = _cpp_tensor_layer("ffn_down", "fully_connected", [
        'withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down")',
        'withKey("unit", dim)',
        'withKey("disable_bias", "true")',
    ], mul_out)
    L.extend(lines)

    return down_out


def _emit_standard_ffn(L, ffn):
    """Emit GELU or ReLU FFN layers using Tensor flow. Returns last output var."""
    act = "gelu" if ffn.ffn_type == "gelu_ffn" else "relu"

    # FC1
    lines, fc1_out = _cpp_tensor_layer("ffn_fc1", "fully_connected", [
        'withKey("name", "layer" + std::to_string(layer_id) + "_ffn_fc1")',
        'withKey("unit", hidden_dim)',
    ], "input")
    L.extend(lines)
    L.append(f"")

    # Activation
    lines, act_out = _cpp_tensor_layer("ffn_act", "activation", [
        'withKey("name", "layer" + std::to_string(layer_id) + "_ffn_act")',
        f'withKey("activation", "{act}")',
    ], fc1_out)
    L.extend(lines)
    L.append(f"")

    # Down
    lines, down_out = _cpp_tensor_layer("ffn_down", "fully_connected", [
        'withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down")',
        'withKey("unit", dim)',
    ], act_out)
    L.extend(lines)

    return down_out

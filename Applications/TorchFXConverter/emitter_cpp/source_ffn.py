"""C++ createMlp() method generation using symbolic Tensor API."""

from .helpers import _cpp_tensor_layer, _class_name


def emit_ffn_method(cname, block):
    """Generate createMlp() method body using symbolic Tensor API."""
    ffn = block.ffn
    L = []

    L.append(f"Tensor {cname}::createMlp(")
    L.append(f"  const int layer_id, int dim, int hidden_dim,")
    L.append(f"  Tensor input) {{")
    L.append(f"")
    L.append(f'  auto prefix = "layer" + std::to_string(layer_id);')
    L.append(f"")

    if ffn.ffn_type == "swiglu":
        _emit_swiglu_ffn(L)
    else:
        _emit_standard_ffn(L, ffn)

    L.append(f"")
    L.append(f"  return ffn_down;")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)


def _emit_swiglu_ffn(L):
    """Emit SwiGLU FFN layers using symbolic tensor API."""
    L.extend(_cpp_tensor_layer("fully_connected", "ffn_up", [
        'withKey("name", prefix + "_ffn_up")',
        'withKey("unit", hidden_dim)',
        'withKey("disable_bias", "true")',
    ], "input"))

    L.append(f"")
    L.extend(_cpp_tensor_layer("fully_connected", "ffn_gate", [
        'withKey("name", prefix + "_ffn_gate")',
        'withKey("unit", hidden_dim)',
        'withKey("disable_bias", "true")',
    ], "input"))

    L.append(f"")
    L.extend(_cpp_tensor_layer("swiglu", "ffn_swiglu", [
        'withKey("name", prefix + "_ffn_swiglu")',
    ], ["ffn_up", "ffn_gate"]))

    L.append(f"")
    L.extend(_cpp_tensor_layer("fully_connected", "ffn_down", [
        'withKey("name", prefix + "_ffn_down")',
        'withKey("unit", dim)',
        'withKey("disable_bias", "true")',
    ], "ffn_swiglu"))


def _emit_standard_ffn(L, ffn):
    """Emit GELU or ReLU FFN layers using symbolic tensor API."""
    act = "gelu" if ffn.ffn_type == "gelu_ffn" else "relu"

    L.extend(_cpp_tensor_layer("fully_connected", "ffn_fc1", [
        'withKey("name", prefix + "_ffn_fc1")',
        'withKey("unit", hidden_dim)',
    ], "input"))

    L.append(f"")
    L.extend(_cpp_tensor_layer("activation", "ffn_act", [
        'withKey("name", prefix + "_ffn_act")',
        f'withKey("activation", "{act}")',
    ], "ffn_fc1"))

    L.append(f"")
    L.extend(_cpp_tensor_layer("fully_connected", "ffn_down", [
        'withKey("name", prefix + "_ffn_down")',
        'withKey("unit", dim)',
    ], "ffn_act"))

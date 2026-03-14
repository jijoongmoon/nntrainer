"""C++ createMlp() method generation."""

from .helpers import _cpp_layer, _class_name


def emit_ffn_method(cname, block):
    """Generate createMlp() method body."""
    ffn = block.ffn
    L = []

    L.append(f"std::vector<LayerHandle> {cname}::createMlp(")
    L.append(f"  const int layer_id, int dim, int hidden_dim,")
    L.append(f"  std::string input_name) {{")
    L.append(f"")
    L.append(f"  std::vector<LayerHandle> layers;")
    L.append(f"")

    if ffn.ffn_type == "swiglu":
        _emit_swiglu_ffn(L)
    else:
        _emit_standard_ffn(L, ffn)

    L.append(f"")
    L.append(f"  return layers;")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)


def _emit_swiglu_ffn(L):
    """Emit SwiGLU FFN layers."""
    L.extend(_cpp_layer("fully_connected", [
        'withKey("name", "layer" + std::to_string(layer_id) '
        '+ "_ffn_up")',
        'withKey("unit", hidden_dim)',
        'withKey("disable_bias", "true")',
        'withKey("input_layers", input_name)',
    ]))

    L.append(f"")
    L.extend(_cpp_layer("fully_connected", [
        'withKey("name", "layer" + std::to_string(layer_id) '
        '+ "_ffn_gate")',
        'withKey("unit", hidden_dim)',
        'withKey("disable_bias", "true")',
        'withKey("input_layers", input_name)',
    ]))

    L.append(f"")
    L.extend(_cpp_layer("swiglu", [
        'withKey("name", "layer" + std::to_string(layer_id) '
        '+ "_ffn_swiglu")',
        'withKey("input_layers", "layer" + std::to_string(layer_id) '
        '+ "_ffn_up," + "layer" + std::to_string(layer_id) '
        '+ "_ffn_gate")',
    ]))

    L.append(f"")
    L.extend(_cpp_layer("fully_connected", [
        'withKey("name", "layer" + std::to_string(layer_id) '
        '+ "_ffn_down")',
        'withKey("unit", dim)',
        'withKey("disable_bias", "true")',
        'withKey("input_layers", "layer" + std::to_string(layer_id) '
        '+ "_ffn_swiglu")',
    ]))


def _emit_standard_ffn(L, ffn):
    """Emit GELU or ReLU FFN layers."""
    act = "gelu" if ffn.ffn_type == "gelu_ffn" else "relu"

    L.extend(_cpp_layer("fully_connected", [
        'withKey("name", "layer" + std::to_string(layer_id) '
        '+ "_ffn_fc1")',
        'withKey("unit", hidden_dim)',
        'withKey("input_layers", input_name)',
    ]))

    L.append(f"")
    L.extend(_cpp_layer("activation", [
        'withKey("name", "layer" + std::to_string(layer_id) '
        '+ "_ffn_act")',
        f'withKey("activation", "{act}")',
        'withKey("input_layers", "layer" + std::to_string(layer_id) '
        '+ "_ffn_fc1")',
    ]))

    L.append(f"")
    L.extend(_cpp_layer("fully_connected", [
        'withKey("name", "layer" + std::to_string(layer_id) '
        '+ "_ffn_down")',
        'withKey("unit", dim)',
        'withKey("input_layers", "layer" + std::to_string(layer_id) '
        '+ "_ffn_act")',
    ]))

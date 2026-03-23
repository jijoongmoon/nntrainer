"""C++ createMlp() method generation using symbolic Tensor graph."""

from .helpers import _cpp_tensor_layer, _class_name
from .source_generic import emit_generic_tensor_ops


def emit_ffn_method(cname, block, layers_by_name=None):
    """Generate createMlp() method body using Tensor flow.

    Uses pattern-specific emitters for known FFN types (SwiGLU, GeGLU,
    standard). Falls back to generic tensor-op emission when the FFN type
    is unrecognized, preserving the original layer graph connectivity.
    """
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
    elif ffn.ffn_type in ("gelu_ffn", "standard"):
        last_var = _emit_standard_ffn(L, ffn)
    elif layers_by_name and ffn.layer_names:
        last_var = _emit_generic_ffn(L, ffn, layers_by_name)
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


def _emit_generic_ffn(L, ffn, layers_by_name):
    """Emit unrecognized FFN as generic tensor ops using layer graph.

    Falls back to emitting individual NNTrainer layers with their actual
    properties and input_layers connectivity.  Also pulls in any
    intermediate layers (e.g. multiply) that are referenced in
    input_layers but missing from ffn.layer_names.
    """
    prefix_expr = '"layer" + std::to_string(layer_id)'

    # Start with explicitly listed layers, then chase missing references
    layer_set = set(ffn.layer_names)
    queue = list(ffn.layer_names)
    while queue:
        name = queue.pop(0)
        layer = layers_by_name.get(name)
        if not layer:
            continue
        for inp in layer.input_layers:
            if inp in layers_by_name and inp not in layer_set:
                # Check that this intermediate layer belongs to the FFN
                # (its inputs reference other FFN layers)
                inp_layer = layers_by_name[inp]
                if any(i in layer_set for i in inp_layer.input_layers):
                    layer_set.add(inp)
                    queue.append(inp)

    # Topological sort: emit layers whose inputs are already emitted
    remaining = {n: layers_by_name[n] for n in layer_set
                 if n in layers_by_name}
    ffn_layers = []
    emitted = set()
    changed = True
    while remaining and changed:
        changed = False
        for name in list(remaining):
            layer = remaining[name]
            deps = [i for i in layer.input_layers if i in layer_set]
            if all(d in emitted for d in deps):
                ffn_layers.append(layer)
                emitted.add(name)
                del remaining[name]
                changed = True
    # Append any remaining (circular deps or unresolved - shouldn't happen)
    ffn_layers.extend(remaining.values())

    if not ffn_layers:
        return "input"

    # Determine block scope for clean variable naming
    block_scope = ""
    name = ffn_layers[0].name
    parts = name.split("_")
    for end in range(len(parts) - 1, 0, -1):
        candidate = "_".join(parts[:end]) + "_"
        if all(l.name.startswith(candidate) for l in ffn_layers):
            block_scope = candidate
            break

    L.append(f"  // FFN (generic tensor-op fallback: {ffn.ffn_type})")
    lines, last_var = emit_generic_tensor_ops(
        ffn_layers, "input", prefix_expr, block_scope)
    L.extend(lines)
    return last_var

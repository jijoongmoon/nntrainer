"""Generic tensor-op emitter for NNTrainer layers without specialized patterns.

When a model component (FFN, attention, etc.) does not match any known
NNTrainer pattern, this module emits the raw NN layers as individual
LayerHandle + Tensor operations, preserving the original graph connectivity
through input_layers references.

This serves as a universal fallback: any valid NNTrainerLayerDef graph can
be emitted as C++ code regardless of whether a specialized pattern exists.
"""

from .helpers import _cpp_tensor_layer


def emit_generic_tensor_ops(layers, input_var, prefix_expr,
                            block_scope="", indent=1):
    """Emit a list of NNTrainerLayerDef as generic Tensor flow operations.

    Unlike pattern-specific emitters that hardcode layer structure, this
    resolves actual input_layers connections from the NN layer graph.

    Args:
        layers: list of NNTrainerLayerDef objects (should be in topological
                order or at least dependency-safe order)
        input_var: C++ variable name for the method's input tensor
                   (used when a layer references an input not in this scope)
        prefix_expr: C++ expression for runtime name prefix, e.g.
                     '"layer" + std::to_string(layer_id)'
        block_scope: sanitized block scope prefix to strip from layer names
                     for cleaner variable naming (e.g. "layers_0_mlp_")
        indent: indentation level (1 = 2 spaces)

    Returns:
        tuple (lines, last_output_var) where lines is a list of C++ strings
        and last_output_var is the Tensor variable name of the last layer.
    """
    if not layers:
        return [], input_var

    L = []
    # Map layer name -> C++ tensor variable name
    tensor_vars = {}
    last_var = input_var

    for i, layer in enumerate(layers):
        # Derive a short, readable C++ variable name from the layer name
        var_name = _derive_var_name(layer, i, block_scope)

        # Resolve input expression from layer's input_layers
        input_expr = _resolve_inputs(
            layer.input_layers, tensor_vars, input_var)

        # Build property list as withKey() calls
        props = _build_props(layer, prefix_expr, block_scope)

        # Emit LayerHandle + Tensor
        lines, out_var = _cpp_tensor_layer(
            var_name, layer.layer_type, props, input_expr, indent)
        L.extend(lines)
        L.append("")

        tensor_vars[layer.name] = out_var
        last_var = out_var

    return L, last_var


def _derive_var_name(layer, index, block_scope):
    """Derive a short C++ variable name from the NN layer name.

    Strips the block scope prefix and uses the remaining suffix.
    Falls back to op_{index} if the name is too long or empty.
    """
    name = layer.name
    # Strip common block scope prefix for readability
    if block_scope and name.startswith(block_scope):
        suffix = name[len(block_scope):].lstrip("_")
    else:
        # Try stripping everything up to the last meaningful parts
        parts = name.split("_")
        # Use last 2 parts for readability (e.g. "gate_proj")
        suffix = "_".join(parts[-2:]) if len(parts) > 2 else name

    # Sanitize for C++ variable name
    suffix = suffix.replace("-", "_").replace(".", "_")

    if not suffix or len(suffix) > 30:
        return f"op_{index}"

    # Avoid C++ reserved words and collisions
    if suffix[0].isdigit():
        suffix = "l_" + suffix

    return suffix


def _resolve_inputs(input_layers, tensor_vars, fallback_var):
    """Resolve input_layers references to C++ tensor variable expressions.

    Args:
        input_layers: list of NN layer names this layer depends on
        tensor_vars: dict mapping layer_name -> C++ variable name
        fallback_var: variable name to use when input is not found
                     (typically the method's input parameter)

    Returns:
        C++ expression string for the layer's input
    """
    if not input_layers:
        return fallback_var

    resolved = []
    for name in input_layers:
        if name in tensor_vars:
            resolved.append(tensor_vars[name])
        else:
            # Input is from outside this scope (method parameter or
            # earlier block output) - use fallback
            resolved.append(fallback_var)

    if len(resolved) == 1:
        return resolved[0]
    return "{" + ", ".join(resolved) + "}"


def _build_props(layer, prefix_expr, block_scope):
    """Build list of withKey() property expressions for a layer.

    Generates a runtime name using prefix_expr and includes all layer
    properties except input_layers (handled by Tensor connectivity).
    """
    # Derive the name suffix for runtime naming
    name = layer.name
    if block_scope and name.startswith(block_scope):
        name_suffix = "_" + name[len(block_scope):].lstrip("_")
    else:
        parts = name.split("_")
        name_suffix = "_" + "_".join(parts[-2:]) if len(parts) > 2 else \
            "_" + name

    props = [f'withKey("name", {prefix_expr} + "{name_suffix}")']

    for k, v in layer.properties.items():
        if k in ("input_layers", "name"):
            continue
        if isinstance(v, bool):
            props.append(f'withKey("{k}", "{str(v).lower()}")')
        elif isinstance(v, (int, float)):
            props.append(f'withKey("{k}", {v})')
        elif isinstance(v, str):
            props.append(f'withKey("{k}", "{v}")')
        else:
            props.append(f'withKey("{k}", "{v}")')

    return props

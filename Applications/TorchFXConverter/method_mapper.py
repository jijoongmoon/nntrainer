"""Method mapper: Maps call_method FX nodes to NNTrainer layer definitions.

Handles Tensor.* method calls such as add, mul, view, reshape, permute,
transpose, softmax, etc.
"""

import math

from nntrainer_layers import (
    NNTrainerLayerDef,
    LAYER_ACTIVATION, LAYER_CLAMP,
    OP_NOOP, OP_RESHAPE, OP_UNSUPPORTED,
)
from op_registry import (
    METHOD_SIMPLE_OPS, METHOD_ACTIVATION_OPS, METHOD_SHAPE_OPS,
    METHOD_RESHAPE_NAMES, METHOD_DECOMPOSE_OPS, METHOD_CLAMP_NAMES,
    METHOD_NOOP_NAMES, METHOD_SPLIT_NAMES,
)


def _get_input_node_names(node):
    names = []
    for arg in node.args:
        if hasattr(arg, 'name'):
            names.append(arg.name)
    return names


def _sanitize_name(name: str) -> str:
    return name.replace(".", "_")


def _make_scoped_name(scope, node, suffix=""):
    if scope:
        name = f"{_sanitize_name(scope)}_{node.name}"
    else:
        name = node.name
    if suffix:
        name = f"{name}_{suffix}"
    return name


def _extract_clamp_params(node, op_name, props):
    """Extract min/max parameters from clamp/clip FX node."""
    args = node.args
    kwargs = node.kwargs or {}

    if op_name == "clamp_min":
        props["min"] = str(args[1] if len(args) > 1 else kwargs.get("min", 0))
        props["max"] = str(math.inf)
    elif op_name == "clamp_max":
        props["min"] = str(-math.inf)
        props["max"] = str(args[1] if len(args) > 1 else kwargs.get("max", 0))
    else:
        if len(args) > 1:
            props["min"] = str(args[1])
        elif "min" in kwargs and kwargs["min"] is not None:
            props["min"] = str(kwargs["min"])
        else:
            props["min"] = str(-math.inf)

        if len(args) > 2:
            props["max"] = str(args[2])
        elif "max" in kwargs and kwargs["max"] is not None:
            props["max"] = str(kwargs["max"])
        else:
            props["max"] = str(math.inf)


def map_method_node(node):
    """Map a call_method node to NNTrainerLayerDef.

    Args:
        node: FX graph node with op == "call_method"

    Returns:
        NNTrainerLayerDef or None
    """
    method_name = node.target  # string
    scope = node.meta.get("scope", "")
    input_names = _get_input_node_names(node)

    # === Simple ops (table lookup) ===
    layer_type = METHOD_SIMPLE_OPS.get(method_name)
    if layer_type is not None:
        return NNTrainerLayerDef(
            layer_type=layer_type,
            name=_make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope if layer_type.endswith("_op") else "",
        )

    # === Activation methods (table lookup) ===
    act_type = METHOD_ACTIVATION_OPS.get(method_name)
    if act_type is not None:
        return NNTrainerLayerDef(
            layer_type=LAYER_ACTIVATION,
            name=_sanitize_name(scope) if scope else node.name,
            properties={"activation": act_type},
            input_layers=input_names,
            hf_module_name=scope,
            hf_module_type="Tensor",
        )

    # === Shape ops (view, reshape, permute, transpose) ===
    shape_type = METHOD_SHAPE_OPS.get(method_name)
    if shape_type is not None:
        return NNTrainerLayerDef(
            layer_type=shape_type,
            name=node.name,
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Reshape names (unsqueeze, squeeze, repeat, expand_as) ===
    if method_name in METHOD_RESHAPE_NAMES:
        return NNTrainerLayerDef(
            layer_type=OP_RESHAPE,
            name=_make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Decompose ops ===
    decompose_props = METHOD_DECOMPOSE_OPS.get(method_name)
    if decompose_props is not None:
        return NNTrainerLayerDef(
            layer_type=OP_UNSUPPORTED,
            name=_make_scoped_name(scope, node),
            properties=dict(decompose_props),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Clamp ===
    if method_name in METHOD_CLAMP_NAMES:
        props = {"original_op": method_name}
        _extract_clamp_params(node, method_name, props)
        return NNTrainerLayerDef(
            layer_type=LAYER_CLAMP,
            name=_make_scoped_name(scope, node),
            properties=props,
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Split/chunk ===
    if method_name in METHOD_SPLIT_NAMES:
        return NNTrainerLayerDef(
            layer_type="split",
            name=_make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === No-ops ===
    if method_name in METHOD_NOOP_NAMES:
        return NNTrainerLayerDef(
            layer_type=OP_NOOP,
            name=_make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Unknown method ===
    print(f"  [WARNING] Unmapped method: {method_name} in scope={scope}")
    return NNTrainerLayerDef(
        layer_type=f"unknown_method({method_name})",
        name=_make_scoped_name(scope, node),
        input_layers=input_names,
        hf_module_name=scope,
    )

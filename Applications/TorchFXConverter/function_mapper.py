"""Function mapper: Maps call_function FX nodes to NNTrainer layer definitions.

Handles torch.* functions (torch.add, torch.matmul, etc.), operator.*
(operator.add, operator.mul, etc.), and F.* functional calls.
"""

import operator
import math

import torch
import torch.nn.functional as F

from nntrainer_layers import (
    NNTrainerLayerDef,
    LAYER_ACTIVATION, LAYER_DROPOUT, LAYER_CONCAT, LAYER_MATMUL,
    LAYER_CLAMP, LAYER_IDENTITY, LAYER_GATHER,
    ACT_SWISH,
    OP_SDPA, OP_NOOP, OP_RESHAPE, OP_UNSUPPORTED,
)
from op_registry import (
    FUNCTION_SIMPLE_OPS, FUNCTION_NAME_SIMPLE_OPS,
    FUNCTION_NOOP_NAMES, FUNCTION_RESHAPE_NAMES, FUNCTION_DECOMPOSE_OPS,
    FUNCTION_ACTIVATION_OPS, FUNCTION_ACTIVATION_NAMES,
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


# Layer types that produce tuple outputs (for operator.getitem handling)
_MULTI_OUTPUT_LAYER_TYPES = frozenset({"gru", "lstm", "rnn"})


def map_function_node(node, node_to_layer):
    """Map a call_function node to NNTrainerLayerDef.

    Args:
        node: FX graph node with op == "call_function"
        node_to_layer: dict mapping node.name -> NNTrainerLayerDef

    Returns:
        NNTrainerLayerDef or None
    """
    func = node.target
    scope = node.meta.get("scope", "")
    input_names = _get_input_node_names(node)
    func_name = getattr(func, "__name__", str(func))

    # === No-ops ===
    if func_name in FUNCTION_NOOP_NAMES:
        return NNTrainerLayerDef(
            layer_type=OP_NOOP,
            name=_make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Simple ops (table lookup by callable identity) ===
    layer_type = FUNCTION_SIMPLE_OPS.get(func)
    if layer_type is not None:
        return NNTrainerLayerDef(
            layer_type=layer_type,
            name=_make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope if layer_type.endswith("_op") else "",
        )

    # === Simple ops (table lookup by name) ===
    layer_type = FUNCTION_NAME_SIMPLE_OPS.get(func_name)
    if layer_type is not None:
        return NNTrainerLayerDef(
            layer_type=layer_type,
            name=_make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Reshape by name ===
    if func_name in FUNCTION_RESHAPE_NAMES:
        return NNTrainerLayerDef(
            layer_type=OP_RESHAPE,
            name=_make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Decompose ops ===
    decompose_props = FUNCTION_DECOMPOSE_OPS.get(func_name)
    if decompose_props is not None:
        # Use torch.abs identity check for abs specifically
        if func_name == "abs" and func is not torch.abs:
            pass  # fall through if name matches but not the expected func
        else:
            return NNTrainerLayerDef(
                layer_type=OP_UNSUPPORTED,
                name=_make_scoped_name(scope, node),
                properties=dict(decompose_props),
                input_layers=input_names,
                hf_module_name=scope,
            )

    # torch.abs by identity (in case func_name didn't match above)
    if func is torch.abs:
        return NNTrainerLayerDef(
            layer_type=OP_UNSUPPORTED,
            name=_make_scoped_name(scope, node),
            properties={"original_op": "abs", "decompose_to": "sqrt(pow(x, 2))"},
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Clamp ===
    if func_name in ("clamp", "clip", "clamp_min", "clamp_max"):
        props = {"original_op": func_name}
        _extract_clamp_params(node, func_name, props)
        return NNTrainerLayerDef(
            layer_type=LAYER_CLAMP,
            name=_make_scoped_name(scope, node),
            properties=props,
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Activation functions (table lookup) ===
    act_type = FUNCTION_ACTIVATION_OPS.get(func)
    if act_type is not None:
        return NNTrainerLayerDef(
            layer_type=LAYER_ACTIVATION,
            name=_make_scoped_name(scope, node, act_type),
            properties={"activation": act_type},
            input_layers=input_names,
        )

    # Activation by name (torch.tanh / F.tanh)
    act_type = FUNCTION_ACTIVATION_NAMES.get(func_name)
    if act_type is not None:
        return NNTrainerLayerDef(
            layer_type=LAYER_ACTIVATION,
            name=_make_scoped_name(scope, node, func_name),
            properties={"activation": act_type},
            input_layers=input_names,
        )

    # === F.dropout ===
    if func is F.dropout:
        return NNTrainerLayerDef(
            layer_type=LAYER_DROPOUT,
            name=_make_scoped_name(scope, node),
            input_layers=input_names,
        )

    # === F.pad -> noop ===
    if func is F.pad:
        return NNTrainerLayerDef(
            layer_type=OP_NOOP,
            name=_make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === torch.cat ===
    if func is torch.cat:
        return _map_cat(node, scope, input_names)

    # === torch.stack ===
    if func is torch.stack:
        return _map_stack(node, scope, input_names)

    # === torch.gather ===
    if func is torch.gather:
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)
        return NNTrainerLayerDef(
            layer_type=LAYER_GATHER,
            name=_make_scoped_name(scope, node),
            properties={"axis": dim},
            input_layers=input_names,
        )

    # === torch.einsum ===
    if func is torch.einsum:
        equation = node.args[0] if len(node.args) > 0 else ""
        return NNTrainerLayerDef(
            layer_type=LAYER_MATMUL,
            name=_make_scoped_name(scope, node),
            properties={"equation": equation},
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === F.scaled_dot_product_attention ===
    if func_name == "scaled_dot_product_attention":
        return NNTrainerLayerDef(
            layer_type=OP_SDPA,
            name=_make_scoped_name(scope, node, "sdpa"),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === operator.getitem (tuple unpacking for multi-output modules) ===
    if func is operator.getitem:
        return _map_getitem(node, scope, node_to_layer)

    # === Unknown function ===
    print(f"  [WARNING] Unmapped function: {func_name} in scope={scope}")
    return NNTrainerLayerDef(
        layer_type=f"unknown_func({func_name})",
        name=_make_scoped_name(scope, node),
        input_layers=input_names,
        hf_module_name=scope,
    )


def _map_cat(node, scope, input_names):
    """Map torch.cat to concat layer."""
    cat_inputs = []
    if node.args:
        tensor_list = node.args[0]
        if isinstance(tensor_list, (list, tuple)):
            cat_inputs = [a.name for a in tensor_list if hasattr(a, 'name')]
    if not cat_inputs:
        cat_inputs = input_names
    dim = node.kwargs.get('dim', 0)
    if len(node.args) > 1 and isinstance(node.args[1], int):
        dim = node.args[1]
    props = {}
    if dim != 0:
        props["concat_dimension"] = dim
    return NNTrainerLayerDef(
        layer_type=LAYER_CONCAT,
        name=_make_scoped_name(scope, node),
        properties=props,
        input_layers=cat_inputs,
    )


def _map_stack(node, scope, input_names):
    """Map torch.stack to concat layer."""
    stack_inputs = []
    if node.args:
        tensor_list = node.args[0]
        if isinstance(tensor_list, (list, tuple)):
            stack_inputs = [a.name for a in tensor_list if hasattr(a, 'name')]
    if not stack_inputs:
        stack_inputs = input_names
    dim = node.kwargs.get('dim', 0)
    if len(node.args) > 1 and isinstance(node.args[1], int):
        dim = node.args[1]
    return NNTrainerLayerDef(
        layer_type=LAYER_CONCAT,
        name=_make_scoped_name(scope, node),
        properties={"concat_dimension": dim, "stack": True},
        input_layers=stack_inputs,
    )


def _map_getitem(node, scope, node_to_layer):
    """Map operator.getitem for multi-output module tuple unpacking."""
    if len(node.args) >= 2 and hasattr(node.args[0], 'name'):
        parent_name = node.args[0].name
        parent_layer = node_to_layer.get(parent_name)
        idx = node.args[1]
        if (parent_layer and
                parent_layer.layer_type in _MULTI_OUTPUT_LAYER_TYPES
                and idx == 0):
            return NNTrainerLayerDef(
                layer_type=LAYER_IDENTITY,
                name=node.name,
                input_layers=[parent_name],
                hf_module_name=scope,
            )
    return None

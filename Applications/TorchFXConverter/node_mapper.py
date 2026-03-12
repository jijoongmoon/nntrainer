#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Node Mapper: Converts FX graph nodes to NNTrainer layer definitions.

Maps three types of FX nodes:
- call_module: Module forward calls (Linear, Embedding, etc.)
- call_function: Torch function calls (torch.add, F.silu, etc.)
- call_method: Tensor method calls (.view, .mul, .add, etc.)
"""

import operator
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NNTrainerLayerDef:
    """Definition of a single NNTrainer layer."""

    layer_type: str  # e.g. "fully_connected", "activation", "add"
    name: str  # unique layer name
    params: dict = field(default_factory=dict)  # layer-specific parameters
    input_layers: list[str] = field(default_factory=list)
    # Original FX node info for debugging
    fx_op: str = ""
    fx_target: str = ""
    fx_scope: str = ""

    def __repr__(self):
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        inputs = ", ".join(self.input_layers) if self.input_layers else ""
        return (
            f"NNTrainerLayerDef({self.layer_type}, name={self.name}, "
            f"params={{{p}}}, inputs=[{inputs}])"
        )


# ---- Module mapping: call_module target type -> NNTrainer layer type ----

def _map_linear(node, module):
    """Map nn.Linear to fully_connected."""
    params = {
        "unit": module.out_features,
    }
    if module.bias is None:
        params["disable_bias"] = "true"
    return NNTrainerLayerDef(
        layer_type="fully_connected",
        name=node.name,
        params=params,
    )


def _map_embedding(node, module):
    """Map nn.Embedding to embedding."""
    return NNTrainerLayerDef(
        layer_type="embedding",
        name=node.name,
        params={
            "in_dim": module.num_embeddings,
            "out_dim": module.embedding_dim,
        },
    )


def _map_layernorm(node, module):
    """Map nn.LayerNorm to layer_normalization."""
    return NNTrainerLayerDef(
        layer_type="layer_normalization",
        name=node.name,
        params={
            "epsilon": str(module.eps),
        },
    )


def _map_batchnorm(node, module):
    """Map nn.BatchNorm to batch_normalization."""
    return NNTrainerLayerDef(
        layer_type="batch_normalization",
        name=node.name,
        params={
            "epsilon": str(module.eps),
            "momentum": str(module.momentum),
        },
    )


def _map_conv2d(node, module):
    """Map nn.Conv2d to conv2d."""
    return NNTrainerLayerDef(
        layer_type="conv2d",
        name=node.name,
        params={
            "filters": module.out_channels,
            "kernel_size": f"{module.kernel_size[0]},{module.kernel_size[1]}",
            "stride": f"{module.stride[0]},{module.stride[1]}",
            "padding": f"{module.padding[0]},{module.padding[1]}",
        },
    )


def _map_dropout(node, module):
    """Map nn.Dropout to dropout."""
    return NNTrainerLayerDef(
        layer_type="dropout",
        name=node.name,
        params={"dropout_rate": str(module.p)},
    )


def _map_identity(node, module):
    """Map nn.Identity to identity."""
    return NNTrainerLayerDef(
        layer_type="identity",
        name=node.name,
    )


# Activation module mapping
_ACTIVATION_MODULE_MAP = {
    nn.ReLU: "relu",
    nn.GELU: "gelu",
    nn.SiLU: "swish",
    nn.Sigmoid: "sigmoid",
    nn.Tanh: "tanh",
    nn.LeakyReLU: "leaky_relu",
    nn.ELU: "elu",
    nn.SELU: "selu",
    nn.Softplus: "softplus",
    nn.Mish: "mish",
    nn.Softmax: "softmax",
    nn.ReLU6: "relu",  # closest match
}


def _map_activation_module(node, module):
    """Map activation modules to activation layer."""
    for mod_type, acti_type in _ACTIVATION_MODULE_MAP.items():
        if isinstance(module, mod_type):
            return NNTrainerLayerDef(
                layer_type="activation",
                name=node.name,
                params={"activation": acti_type},
            )
    return None


# Master module type -> mapper function
MODULE_MAPPERS = {
    nn.Linear: _map_linear,
    nn.Embedding: _map_embedding,
    nn.LayerNorm: _map_layernorm,
    nn.BatchNorm1d: _map_batchnorm,
    nn.BatchNorm2d: _map_batchnorm,
    nn.BatchNorm3d: _map_batchnorm,
    nn.Conv2d: _map_conv2d,
    nn.Dropout: _map_dropout,
    nn.Identity: _map_identity,
}
# Add activation modules
for _acti_cls in _ACTIVATION_MODULE_MAP:
    MODULE_MAPPERS[_acti_cls] = _map_activation_module


# ---- Function mapping: call_function target -> NNTrainer op ----

def _get_func_name(func):
    """Get a normalized name for a torch function."""
    if hasattr(func, "__name__"):
        return func.__name__
    if hasattr(func, "__qualname__"):
        return func.__qualname__
    return str(func)


# Activation function targets
_ACTIVATION_FUNC_MAP = {
    "silu": "swish",
    "relu": "relu",
    "gelu": "gelu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "leaky_relu": "leaky_relu",
    "elu": "elu",
    "selu": "selu",
    "softmax": "softmax",
    "softplus": "softplus",
    "mish": "mish",
}

# Direct function -> nntrainer layer type mapping
_FUNCTION_LAYER_MAP = {
    torch.add: "add",
    torch.sub: "subtract",
    torch.mul: "multiply",
    torch.div: "divide",
    torch.matmul: "matmul",
    torch.cat: "concat",
    torch.neg: "negative",
    torch.sqrt: "sqrt",
    torch.pow: "pow",
    torch.sin: "sin",
    torch.cos: "cos",
    torch.tan: "tan",
    torch.mean: "reduce_mean",
    torch.sum: "reduce_sum",
    torch.rsqrt: "inv_sqrt",  # nntrainer inv_sqrt = 1/sqrt
    operator.add: "add",
    operator.sub: "subtract",
    operator.mul: "multiply",
    operator.truediv: "divide",
    operator.neg: "negative",
    operator.getitem: None,  # tuple unpacking, no-op
}

# Tensor method -> nntrainer layer type
_METHOD_LAYER_MAP = {
    "add": "add",
    "sub": "subtract",
    "mul": "multiply",
    "div": "divide",
    "neg": "negative",
    "pow": "pow",
    "sqrt": "sqrt",
    "mean": "reduce_mean",
    "sum": "reduce_sum",
    "view": "reshape",
    "reshape": "reshape",
    "permute": "permute",
    "transpose": "permute",  # transpose is a specific permute
    "contiguous": None,  # no-op
    "detach": None,  # no-op
    "clone": None,  # no-op
    "to": None,  # dtype cast - often can be ignored
    "float": None,  # dtype cast
    "half": None,  # dtype cast
    "int": None,  # dtype cast
    "unsqueeze": "reshape",
    "squeeze": "reshape",
    "expand": None,  # handled by broadcasting
    "__getitem__": None,  # indexing/slicing
    "cos": "cos",
    "sin": "sin",
    "matmul": "matmul",
}


def map_call_function(node) -> Optional[NNTrainerLayerDef]:
    """Map a call_function FX node to an NNTrainer layer def."""
    func = node.target
    func_name = _get_func_name(func)

    # Check activation functions first
    for acti_name, nntr_acti in _ACTIVATION_FUNC_MAP.items():
        if func_name == acti_name or func_name.endswith(f".{acti_name}"):
            return NNTrainerLayerDef(
                layer_type="activation",
                name=node.name,
                params={"activation": nntr_acti},
                fx_op="call_function",
                fx_target=func_name,
            )

    # Check direct function mapping
    if func in _FUNCTION_LAYER_MAP:
        layer_type = _FUNCTION_LAYER_MAP[func]
        if layer_type is None:
            return None  # no-op
        return NNTrainerLayerDef(
            layer_type=layer_type,
            name=node.name,
            fx_op="call_function",
            fx_target=func_name,
        )

    # Check by name for built-in methods (torch.Tensor.rsqrt, etc.)
    if "rsqrt" in func_name:
        return NNTrainerLayerDef(
            layer_type="inv_sqrt",
            name=node.name,
            fx_op="call_function",
            fx_target=func_name,
        )
    if "cat" in func_name and "scat" not in func_name:
        return NNTrainerLayerDef(
            layer_type="concat",
            name=node.name,
            fx_op="call_function",
            fx_target=func_name,
        )
    if "scaled_dot_product_attention" in func_name:
        return NNTrainerLayerDef(
            layer_type="scaled_dot_product_attention",
            name=node.name,
            fx_op="call_function",
            fx_target=func_name,
        )
    if "tensor" == func_name or func_name.endswith(".tensor"):
        return None  # constant creation, no-op
    if "_set_grad_enabled" in func_name:
        return None  # no-op
    if "getitem" in func_name:
        return None  # tuple unpacking

    # Unknown function - return as unmapped
    return NNTrainerLayerDef(
        layer_type="UNMAPPED_FUNCTION",
        name=node.name,
        params={"original_target": func_name},
        fx_op="call_function",
        fx_target=func_name,
    )


def map_call_method(node) -> Optional[NNTrainerLayerDef]:
    """Map a call_method FX node to an NNTrainer layer def."""
    method_name = node.target  # string like "view", "mul", "add"

    if method_name in _METHOD_LAYER_MAP:
        layer_type = _METHOD_LAYER_MAP[method_name]
        if layer_type is None:
            return None  # no-op
        return NNTrainerLayerDef(
            layer_type=layer_type,
            name=node.name,
            fx_op="call_method",
            fx_target=method_name,
        )

    # Check activation methods
    if method_name in _ACTIVATION_FUNC_MAP:
        return NNTrainerLayerDef(
            layer_type="activation",
            name=node.name,
            params={"activation": _ACTIVATION_FUNC_MAP[method_name]},
            fx_op="call_method",
            fx_target=method_name,
        )

    # Unknown method
    return NNTrainerLayerDef(
        layer_type="UNMAPPED_METHOD",
        name=node.name,
        params={"original_target": method_name},
        fx_op="call_method",
        fx_target=method_name,
    )


def map_call_module(node, modules_dict: dict) -> Optional[NNTrainerLayerDef]:
    """Map a call_module FX node to an NNTrainer layer def.

    Args:
        node: FX graph node with op="call_module"
        modules_dict: dict from model.named_modules()
    """
    target = node.target  # e.g. "model.layers.0.self_attn.q_proj"
    module = modules_dict.get(target)
    if module is None:
        return NNTrainerLayerDef(
            layer_type="UNMAPPED_MODULE",
            name=node.name,
            params={"original_target": target},
            fx_op="call_module",
            fx_target=target,
        )

    # Try module type mappers
    for mod_type, mapper in MODULE_MAPPERS.items():
        if isinstance(module, mod_type):
            result = mapper(node, module)
            if result is not None:
                result.fx_op = "call_module"
                result.fx_target = target
                result.fx_scope = node.meta.get("scope", "")
                return result

    # Unknown module type
    return NNTrainerLayerDef(
        layer_type="UNMAPPED_MODULE",
        name=node.name,
        params={
            "original_target": target,
            "module_type": type(module).__name__,
        },
        fx_op="call_module",
        fx_target=target,
    )


def map_graph(graph, model: nn.Module) -> list[Optional[NNTrainerLayerDef]]:
    """Map all nodes in an FX graph to NNTrainer layer definitions.

    Returns a list parallel to graph.nodes. None entries indicate
    nodes that don't map to an NNTrainer layer (placeholders, outputs,
    get_attr, no-ops).
    """
    modules_dict = dict(model.named_modules())
    result = []

    for node in graph.nodes:
        if node.op == "call_module":
            result.append(map_call_module(node, modules_dict))
        elif node.op == "call_function":
            result.append(map_call_function(node))
        elif node.op == "call_method":
            result.append(map_call_method(node))
        elif node.op == "placeholder":
            result.append(NNTrainerLayerDef(
                layer_type="placeholder",
                name=node.name,
                fx_op="placeholder",
                fx_target=str(node.target),
            ))
        elif node.op == "output":
            result.append(NNTrainerLayerDef(
                layer_type="output",
                name=node.name,
                fx_op="output",
            ))
        else:
            # get_attr, etc.
            result.append(None)

    return result

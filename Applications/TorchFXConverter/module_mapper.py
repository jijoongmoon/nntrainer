"""Module mapper: Maps call_module FX nodes to NNTrainer layer definitions.

Handles nn.Module leaf nodes such as Linear, Embedding, LayerNorm, RMSNorm,
Conv1d/Conv2d, BatchNorm, GRU/LSTM/RNN, activation modules, and Dropout.
"""

import torch.nn as nn

from nntrainer_layers import (
    NNTrainerLayerDef,
    LAYER_FC, LAYER_EMBEDDING, LAYER_RMS_NORM, LAYER_LAYER_NORM,
    LAYER_ACTIVATION, LAYER_DROPOUT,
    LAYER_CONV1D, LAYER_CONV2D, LAYER_BATCH_NORM,
    LAYER_GRU, LAYER_LSTM, LAYER_RNN,
    ACT_RELU, ACT_GELU, ACT_SWISH, ACT_SIGMOID, ACT_TANH, ACT_SOFTMAX,
)
from tracer import _is_rmsnorm, _is_gelu_variant
from mapper_helpers import get_input_node_names, sanitize_name

# Backward-compatible aliases
_sanitize_name = sanitize_name
_get_input_node_names = get_input_node_names


# Set of layer types that return tuple outputs (output, hidden_state)
MULTI_OUTPUT_LAYER_TYPES = frozenset({LAYER_GRU, LAYER_LSTM, LAYER_RNN})

# Activation module type -> activation constant
_MODULE_ACTIVATIONS = [
    (nn.ReLU,    ACT_RELU),
    (nn.GELU,    ACT_GELU),
    (nn.SiLU,    ACT_SWISH),
    (nn.Sigmoid, ACT_SIGMOID),
    (nn.Tanh,    ACT_TANH),
    (nn.Softmax, ACT_SOFTMAX),
]


def map_module_node(node, modules, node_to_layer):
    """Map a call_module node to NNTrainerLayerDef.

    Args:
        node: FX graph node with op == "call_module"
        modules: dict from model.named_modules()
        node_to_layer: dict mapping node.name -> NNTrainerLayerDef (for lookups)

    Returns:
        NNTrainerLayerDef or None
    """
    module_name = node.target
    module = modules.get(module_name)
    if module is None:
        return None

    module_type = type(module).__name__
    input_names = _get_input_node_names(node)

    # nn.Linear -> fully_connected
    if isinstance(module, nn.Linear):
        return NNTrainerLayerDef(
            layer_type=LAYER_FC,
            name=_sanitize_name(module_name),
            properties={
                "unit": module.out_features,
                "disable_bias": module.bias is None,
            },
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
            has_weight=True,
            has_bias=module.bias is not None,
            weight_hf_key=f"{module_name}.weight",
            bias_hf_key=f"{module_name}.bias" if module.bias is not None else "",
            transpose_weight=True,
        )

    # nn.Embedding -> embedding_layer
    if isinstance(module, nn.Embedding):
        props = {
            "in_dim": module.num_embeddings,
            "out_dim": module.embedding_dim,
        }
        if hasattr(module, "scalar_embed_scale"):
            val = module.scalar_embed_scale
            if isinstance(val, (int, float)):
                props["embed_scale"] = float(val)
            elif hasattr(val, 'item'):
                props["embed_scale"] = float(val.item())
            else:
                props["embed_scale"] = str(val)
        return NNTrainerLayerDef(
            layer_type=LAYER_EMBEDDING,
            name=_sanitize_name(module_name),
            properties=props,
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
            has_weight=True,
            weight_hf_key=f"{module_name}.weight",
            transpose_weight=False,
        )

    # nn.LayerNorm -> layer_normalization
    if isinstance(module, nn.LayerNorm):
        return NNTrainerLayerDef(
            layer_type=LAYER_LAYER_NORM,
            name=_sanitize_name(module_name),
            properties={"epsilon": module.eps},
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
            has_weight=module.elementwise_affine,
            has_bias=module.elementwise_affine,
            weight_hf_key=f"{module_name}.weight" if module.elementwise_affine else "",
            bias_hf_key=f"{module_name}.bias" if module.elementwise_affine else "",
        )

    # *RMSNorm (any HF variant) -> rms_norm
    if _is_rmsnorm(module):
        eps = getattr(module, "eps", None) or getattr(module, "variance_epsilon", 1e-6)
        return NNTrainerLayerDef(
            layer_type=LAYER_RMS_NORM,
            name=_sanitize_name(module_name),
            properties={"epsilon": eps, "packed": False},
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
            has_weight=hasattr(module, "weight"),
            weight_hf_key=f"{module_name}.weight" if hasattr(module, "weight") else "",
        )

    # Standard activation modules
    for module_cls, act_type in _MODULE_ACTIVATIONS:
        if isinstance(module, module_cls):
            return _make_activation(module_name, module_type, act_type, input_names)

    # HuggingFace custom GELU variants
    if _is_gelu_variant(module):
        return _make_activation(module_name, module_type, ACT_GELU, input_names)

    # Dropout
    if isinstance(module, nn.Dropout):
        return NNTrainerLayerDef(
            layer_type=LAYER_DROPOUT,
            name=_sanitize_name(module_name),
            properties={"dropout_rate": module.p},
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
        )

    # Conv layers
    if isinstance(module, nn.Conv1d):
        return _map_conv1d(module, module_name, module_type, input_names)

    if isinstance(module, nn.Conv2d):
        return _map_conv2d(module, module_name, module_type, input_names)

    # BatchNorm
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return NNTrainerLayerDef(
            layer_type=LAYER_BATCH_NORM,
            name=_sanitize_name(module_name),
            properties={"epsilon": module.eps, "momentum": module.momentum},
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
            has_weight=module.affine,
            has_bias=module.affine,
            weight_hf_key=f"{module_name}.weight" if module.affine else "",
            bias_hf_key=f"{module_name}.bias" if module.affine else "",
        )

    # RNN family
    if isinstance(module, nn.GRU):
        return _map_rnn_module(module, module_name, module_type, input_names, LAYER_GRU)
    if isinstance(module, nn.LSTM):
        return _map_rnn_module(module, module_name, module_type, input_names, LAYER_LSTM)
    if isinstance(module, nn.RNN):
        return _map_rnn_module(module, module_name, module_type, input_names, LAYER_RNN)

    # Unknown module type
    print(f"  [WARNING] Unknown module type: {module_type} at {module_name}")
    return NNTrainerLayerDef(
        layer_type=f"unknown({module_type})",
        name=_sanitize_name(module_name),
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
    )


def _make_activation(module_name, module_type, act_type, input_names):
    return NNTrainerLayerDef(
        layer_type=LAYER_ACTIVATION,
        name=_sanitize_name(module_name),
        properties={"activation": act_type},
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
    )


def _map_conv1d(module, module_name, module_type, input_names):
    return NNTrainerLayerDef(
        layer_type=LAYER_CONV1D,
        name=_sanitize_name(module_name),
        properties={
            "filters": module.out_channels,
            "kernel_size": module.kernel_size[0],
            "stride": module.stride[0],
            "padding": module.padding[0],
        },
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
        has_weight=True,
        has_bias=module.bias is not None,
        weight_hf_key=f"{module_name}.weight",
        bias_hf_key=f"{module_name}.bias" if module.bias is not None else "",
    )


def _map_conv2d(module, module_name, module_type, input_names):
    return NNTrainerLayerDef(
        layer_type=LAYER_CONV2D,
        name=_sanitize_name(module_name),
        properties={
            "filters": module.out_channels,
            "kernel_size": f"{module.kernel_size[0]},{module.kernel_size[1]}",
            "stride": f"{module.stride[0]},{module.stride[1]}",
            "padding": f"{module.padding[0]},{module.padding[1]}",
        },
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
        has_weight=True,
        has_bias=module.bias is not None,
        weight_hf_key=f"{module_name}.weight",
        bias_hf_key=f"{module_name}.bias" if module.bias is not None else "",
    )


def _map_rnn_module(module, module_name, module_type, input_names, layer_type):
    """Map nn.GRU/LSTM/RNN to NNTrainer layer def."""
    if module.num_layers != 1:
        print(f"  [WARNING] Multi-layer {module_type} (num_layers="
              f"{module.num_layers}) not yet supported, mapping layer 0 only")

    props = {
        "unit": module.hidden_size,
        "return_sequences": True,
    }

    if hasattr(module, 'dropout') and module.dropout > 0:
        props["dropout_rate"] = module.dropout

    if module.bidirectional:
        if layer_type == LAYER_LSTM:
            props["bidirectional"] = True
        else:
            print(f"  [WARNING] Bidirectional {module_type} not supported "
                  f"in NNTrainer, using forward-only")

    weight_keys = {
        "weight_ih": f"{module_name}.weight_ih_l0",
        "weight_hh": f"{module_name}.weight_hh_l0",
    }
    bias_keys = {}
    if module.bias:
        bias_keys["bias_ih"] = f"{module_name}.bias_ih_l0"
        bias_keys["bias_hh"] = f"{module_name}.bias_hh_l0"

    layer_def = NNTrainerLayerDef(
        layer_type=layer_type,
        name=_sanitize_name(module_name),
        properties=props,
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
        has_weight=True,
        has_bias=module.bias,
        weight_hf_key=weight_keys["weight_ih"],
        bias_hf_key=bias_keys.get("bias_ih", ""),
        transpose_weight=True,
    )
    layer_def.properties["_weight_hh_key"] = weight_keys["weight_hh"]
    if module.bias:
        layer_def.properties["_bias_hh_key"] = bias_keys["bias_hh"]

    return layer_def

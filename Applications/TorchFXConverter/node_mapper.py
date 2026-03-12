"""
Node mapper: Maps FX graph nodes to NNTrainer layer definitions.

This module takes the traced FX graph and converts each node into
an NNTrainerLayerDef. It handles:
  1. Leaf module nodes (call_module) -> direct mapping
  2. Function nodes (call_function) -> operation mapping
  3. Method nodes (call_method) -> operation mapping

The mapper is architecture-agnostic. It maps individual nodes based
on their type, not based on what model they came from. Pattern detection
(attention blocks, FFN blocks, etc.) is handled separately in pattern_detector.py.
"""

import operator
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from nntrainer_layers import (
    NNTrainerLayerDef,
    LAYER_FC, LAYER_EMBEDDING, LAYER_TIE_WORD_EMBEDDINGS,
    LAYER_RMS_NORM, LAYER_LAYER_NORM, LAYER_ADDITION,
    LAYER_ACTIVATION, LAYER_DROPOUT, LAYER_CONCAT,
    LAYER_RESHAPE, LAYER_PERMUTE, LAYER_MATMUL,
    LAYER_CONV1D, LAYER_CONV2D, LAYER_BATCH_NORM, LAYER_POOLING2D,
    LAYER_ADD, LAYER_MULTIPLY, LAYER_SUBTRACT, LAYER_DIVIDE,
    LAYER_POW, LAYER_SQRT, LAYER_NEGATIVE,
    LAYER_SIN, LAYER_COS, LAYER_GATHER, LAYER_SLICE,
    ACT_RELU, ACT_GELU, ACT_SWISH, ACT_SIGMOID, ACT_TANH, ACT_SOFTMAX,
    OP_RESHAPE, OP_TRANSPOSE, OP_PERMUTE, OP_SDPA, OP_NOOP,
)
from tracer import _is_rmsnorm


def _get_input_node_names(node):
    """Get names of input nodes for a given FX node."""
    names = []
    for arg in node.args:
        if hasattr(arg, 'name'):
            names.append(arg.name)
    return names


def _sanitize_name(name: str) -> str:
    """Convert HF module path to NNTrainer-compatible layer name.

    e.g. 'model.layers.0.self_attn.q_proj' -> 'layer0_wq'
    """
    return name.replace(".", "_")


class NodeMapper:
    """Maps FX graph nodes to NNTrainer layer definitions.

    Usage:
        mapper = NodeMapper(model, graph, config)
        layers = mapper.map_all()
    """

    def __init__(self, model, graph, model_config=None):
        """
        Args:
            model: The traced nn.Module
            graph: The FX graph from tracing
            model_config: HuggingFace model config (optional, for extracting params)
        """
        self.model = model
        self.graph = graph
        self.config = model_config
        self._modules = dict(model.named_modules())
        self._node_to_layer = {}  # node.name -> NNTrainerLayerDef

    def map_all(self):
        """Map all graph nodes to NNTrainer layer definitions.

        Returns a list of NNTrainerLayerDef objects in graph order,
        plus a dict mapping node names to layer defs.
        """
        layers = []
        for node in self.graph.nodes:
            layer_def = self._map_node(node)
            if layer_def is not None:
                layers.append(layer_def)
                self._node_to_layer[node.name] = layer_def
        return layers

    def _map_node(self, node) -> Optional[NNTrainerLayerDef]:
        """Map a single FX node to an NNTrainer layer definition."""
        if node.op == "call_module":
            return self._map_module_node(node)
        elif node.op == "call_function":
            return self._map_function_node(node)
        elif node.op == "call_method":
            return self._map_method_node(node)
        elif node.op == "placeholder":
            # Input placeholders are handled by pattern_detector
            return None
        elif node.op == "output":
            return None
        elif node.op == "get_attr":
            # Parameter/buffer access - not a layer
            return None
        return None

    def _map_module_node(self, node) -> Optional[NNTrainerLayerDef]:
        """Map a call_module node to NNTrainer layer def."""
        module_name = node.target
        module = self._modules.get(module_name)
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
                transpose_weight=True,  # PyTorch [out, in] -> NNTrainer [in, out]
            )

        # nn.Embedding -> embedding_layer
        if isinstance(module, nn.Embedding):
            return NNTrainerLayerDef(
                layer_type=LAYER_EMBEDDING,
                name=_sanitize_name(module_name),
                properties={
                    "in_dim": module.num_embeddings,
                    "out_dim": module.embedding_dim,
                },
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
                properties={
                    "epsilon": module.eps,
                },
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
                properties={
                    "epsilon": eps,
                    "packed": False,
                },
                input_layers=input_names,
                hf_module_name=module_name,
                hf_module_type=module_type,
                has_weight=hasattr(module, "weight"),
                weight_hf_key=f"{module_name}.weight" if hasattr(module, "weight") else "",
            )

        # Activation modules
        if isinstance(module, nn.ReLU):
            return self._make_activation(module_name, module_type, ACT_RELU, input_names)
        if isinstance(module, nn.GELU):
            return self._make_activation(module_name, module_type, ACT_GELU, input_names)
        if isinstance(module, nn.SiLU):
            return self._make_activation(module_name, module_type, ACT_SWISH, input_names)
        if isinstance(module, nn.Sigmoid):
            return self._make_activation(module_name, module_type, ACT_SIGMOID, input_names)
        if isinstance(module, nn.Tanh):
            return self._make_activation(module_name, module_type, ACT_TANH, input_names)
        if isinstance(module, nn.Softmax):
            return self._make_activation(module_name, module_type, ACT_SOFTMAX, input_names)

        # Dropout -> skip in inference, but record for completeness
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

        if isinstance(module, nn.Conv2d):
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

        # BatchNorm
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            return NNTrainerLayerDef(
                layer_type=LAYER_BATCH_NORM,
                name=_sanitize_name(module_name),
                properties={
                    "epsilon": module.eps,
                    "momentum": module.momentum,
                },
                input_layers=input_names,
                hf_module_name=module_name,
                hf_module_type=module_type,
                has_weight=module.affine,
                has_bias=module.affine,
                weight_hf_key=f"{module_name}.weight" if module.affine else "",
                bias_hf_key=f"{module_name}.bias" if module.affine else "",
            )

        # Unknown module type - record as-is for debugging
        print(f"  [WARNING] Unknown module type: {module_type} at {module_name}")
        return NNTrainerLayerDef(
            layer_type=f"unknown({module_type})",
            name=_sanitize_name(module_name),
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
        )

    def _make_scoped_name(self, scope, node, suffix=""):
        """Generate a scoped layer name for function/method nodes."""
        if scope:
            name = f"{_sanitize_name(scope)}_{node.name}"
        else:
            name = node.name
        if suffix:
            name = f"{name}_{suffix}"
        return name

    def _map_function_node(self, node) -> Optional[NNTrainerLayerDef]:
        """Map a call_function node to NNTrainer layer def."""
        func = node.target
        scope = node.meta.get("scope", "")
        input_names = _get_input_node_names(node)
        func_name = getattr(func, "__name__", str(func))

        # === No-ops: internal torch/runtime functions ===
        if func_name in ("_set_grad_enabled", "tensor", "arange",
                          "zeros", "zeros_like", "ones", "ones_like",
                          "full_like", "empty_like"):
            return NNTrainerLayerDef(
                layer_type=OP_NOOP,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
                hf_module_name=scope,
            )

        # === Arithmetic: map to NNTrainer element-wise tensor ops ===

        # torch.add / operator.add -> addition
        if func in (torch.add, operator.add):
            return NNTrainerLayerDef(
                layer_type=LAYER_ADDITION,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # torch.sub / operator.sub -> subtract
        if func in (torch.sub, operator.sub):
            return NNTrainerLayerDef(
                layer_type=LAYER_SUBTRACT,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # torch.mul / operator.mul -> multiply
        if func in (torch.mul, operator.mul):
            return NNTrainerLayerDef(
                layer_type=LAYER_MULTIPLY,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # torch.div / operator.truediv -> divide
        if func in (torch.div, operator.truediv):
            return NNTrainerLayerDef(
                layer_type=LAYER_DIVIDE,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # torch.matmul -> matmul
        if func is torch.matmul:
            return NNTrainerLayerDef(
                layer_type=LAYER_MATMUL,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # torch.pow -> pow
        if func is torch.pow:
            return NNTrainerLayerDef(
                layer_type=LAYER_POW,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # torch.sqrt -> sqrt
        if func is torch.sqrt:
            return NNTrainerLayerDef(
                layer_type=LAYER_SQRT,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # torch.abs -> pow (abs can be represented as sqrt(x^2) or kept as-is)
        if func is torch.abs:
            return NNTrainerLayerDef(
                layer_type="abs",  # Will need custom handling or decomposition
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # torch.neg / operator.neg -> negative
        if func in (torch.neg, operator.neg):
            return NNTrainerLayerDef(
                layer_type=LAYER_NEGATIVE,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # === Trigonometric / math functions ===

        # torch.sin -> sin
        if func is torch.sin:
            return NNTrainerLayerDef(
                layer_type=LAYER_SIN,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # torch.cos -> cos
        if func is torch.cos:
            return NNTrainerLayerDef(
                layer_type=LAYER_COS,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # torch.log -> needs decomposition or custom op
        if func_name == "log":
            return NNTrainerLayerDef(
                layer_type="log",  # Not directly in NNTrainer; pattern_detector may handle
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # === Shape / structure ops ===

        # torch.cat -> concat
        if func is torch.cat:
            return NNTrainerLayerDef(
                layer_type=LAYER_CONCAT,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # torch.gather -> gather
        if func is torch.gather:
            return NNTrainerLayerDef(
                layer_type=LAYER_GATHER,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # === Activation functions ===

        if func is F.silu:
            return NNTrainerLayerDef(
                layer_type=LAYER_ACTIVATION,
                name=self._make_scoped_name(scope, node, "silu"),
                properties={"activation": ACT_SWISH},
                input_layers=input_names,
            )

        if func is F.gelu:
            return NNTrainerLayerDef(
                layer_type=LAYER_ACTIVATION,
                name=self._make_scoped_name(scope, node, "gelu"),
                properties={"activation": ACT_GELU},
                input_layers=input_names,
            )

        if func is F.relu:
            return NNTrainerLayerDef(
                layer_type=LAYER_ACTIVATION,
                name=self._make_scoped_name(scope, node, "relu"),
                properties={"activation": ACT_RELU},
                input_layers=input_names,
            )

        if func is F.softmax:
            return NNTrainerLayerDef(
                layer_type=LAYER_ACTIVATION,
                name=self._make_scoped_name(scope, node, "softmax"),
                properties={"activation": ACT_SOFTMAX},
                input_layers=input_names,
            )

        # torch.tanh / F.tanh -> activation(tanh)
        if func is torch.tanh or func_name == "tanh":
            return NNTrainerLayerDef(
                layer_type=LAYER_ACTIVATION,
                name=self._make_scoped_name(scope, node, "tanh"),
                properties={"activation": ACT_TANH},
                input_layers=input_names,
            )

        # torch.sigmoid -> activation(sigmoid)
        if func is torch.sigmoid:
            return NNTrainerLayerDef(
                layer_type=LAYER_ACTIVATION,
                name=self._make_scoped_name(scope, node, "sigmoid"),
                properties={"activation": ACT_SIGMOID},
                input_layers=input_names,
            )

        # F.dropout -> dropout (skipped in inference)
        if func is F.dropout:
            return NNTrainerLayerDef(
                layer_type=LAYER_DROPOUT,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # === Attention ===

        # F.scaled_dot_product_attention -> intermediate sdpa (pattern_detector -> mha_core)
        if func_name == "scaled_dot_product_attention":
            return NNTrainerLayerDef(
                layer_type=OP_SDPA,
                name=self._make_scoped_name(scope, node, "sdpa"),
                input_layers=input_names,
                hf_module_name=scope,
            )

        # === Pass-through / no-op ===

        # operator.getitem -> pass-through (tuple unpacking)
        if func is operator.getitem:
            return None

        # Comparison / conditional ops -> noop for inference graph structure
        # (where, min, max are used in relative position bias computation in T5)
        if func_name in ("where", "min", "max"):
            return NNTrainerLayerDef(
                layer_type=OP_NOOP,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
                hf_module_name=scope,
            )

        # Anything else: record as unknown with a warning
        print(f"  [WARNING] Unmapped function: {func_name} in scope={scope}")
        return NNTrainerLayerDef(
            layer_type=f"unknown_func({func_name})",
            name=self._make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    def _map_method_node(self, node) -> Optional[NNTrainerLayerDef]:
        """Map a call_method node to NNTrainer layer def."""
        method_name = node.target  # string
        scope = node.meta.get("scope", "")
        input_names = _get_input_node_names(node)

        # === Arithmetic tensor ops -> NNTrainer element-wise layers ===

        # Tensor.add / add_ -> addition
        if method_name in ("add", "add_"):
            return NNTrainerLayerDef(
                layer_type=LAYER_ADDITION,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # Tensor.sub / sub_ -> subtract
        if method_name in ("sub", "sub_"):
            return NNTrainerLayerDef(
                layer_type=LAYER_SUBTRACT,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # Tensor.mul / mul_ -> multiply
        if method_name in ("mul", "mul_"):
            return NNTrainerLayerDef(
                layer_type=LAYER_MULTIPLY,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # Tensor.div / div_ -> divide
        if method_name in ("div", "div_"):
            return NNTrainerLayerDef(
                layer_type=LAYER_DIVIDE,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # Tensor.matmul -> matmul
        if method_name == "matmul":
            return NNTrainerLayerDef(
                layer_type=LAYER_MATMUL,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # Tensor.neg -> negative
        if method_name == "neg":
            return NNTrainerLayerDef(
                layer_type=LAYER_NEGATIVE,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # Tensor.pow -> pow
        if method_name == "pow":
            return NNTrainerLayerDef(
                layer_type=LAYER_POW,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # Tensor.sqrt -> sqrt
        if method_name == "sqrt":
            return NNTrainerLayerDef(
                layer_type=LAYER_SQRT,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # === Trigonometric -> NNTrainer sin/cos layers ===

        if method_name == "cos":
            return NNTrainerLayerDef(
                layer_type=LAYER_COS,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        if method_name == "sin":
            return NNTrainerLayerDef(
                layer_type=LAYER_SIN,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # === Shape operations -> intermediate (collapsed by pattern_detector) ===

        if method_name in ("view", "reshape"):
            return NNTrainerLayerDef(
                layer_type=OP_RESHAPE,
                name=node.name,
                input_layers=input_names,
                hf_module_name=scope,
            )

        if method_name == "permute":
            return NNTrainerLayerDef(
                layer_type=OP_PERMUTE,
                name=node.name,
                input_layers=input_names,
                hf_module_name=scope,
            )

        if method_name == "transpose":
            return NNTrainerLayerDef(
                layer_type=OP_TRANSPOSE,
                name=node.name,
                input_layers=input_names,
                hf_module_name=scope,
            )

        # Tensor.__getitem__ (indexing/slicing) -> slice
        if method_name == "__getitem__":
            return NNTrainerLayerDef(
                layer_type=LAYER_SLICE,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
            )

        # Tensor.unsqueeze / squeeze -> reshape (shape manipulation)
        if method_name in ("unsqueeze", "squeeze"):
            return NNTrainerLayerDef(
                layer_type=OP_RESHAPE,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
                hf_module_name=scope,
            )

        # === No-ops: dtype/memory operations (safe to skip in inference) ===
        if method_name in ("contiguous", "detach", "clone", "to", "float",
                           "half", "bfloat16", "type_as", "expand",
                           "size", "dim", "numel",
                           "__bool__", "all", "any", "item"):
            return NNTrainerLayerDef(
                layer_type=OP_NOOP,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
                hf_module_name=scope,
            )

        # === Comparison ops (used in T5 relative position bias) ===
        if method_name in ("gt", "lt", "le", "ge", "eq", "ne"):
            return NNTrainerLayerDef(
                layer_type=OP_NOOP,
                name=self._make_scoped_name(scope, node),
                input_layers=input_names,
                hf_module_name=scope,
            )

        # Anything else: record as unknown with a warning
        print(f"  [WARNING] Unmapped method: {method_name} in scope={scope}")
        return NNTrainerLayerDef(
            layer_type=f"unknown_method({method_name})",
            name=self._make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    def _make_activation(self, module_name, module_type, act_type, input_names):
        """Helper to create an activation layer def."""
        return NNTrainerLayerDef(
            layer_type=LAYER_ACTIVATION,
            name=_sanitize_name(module_name),
            properties={"activation": act_type},
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
        )

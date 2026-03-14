"""Unified operation registry for mapping PyTorch ops to NNTrainer layer types.

This module centralizes the mapping tables that were previously duplicated
across _map_function_node() and _map_method_node() in NodeMapper.

Three categories of mappings:
  1. FUNCTION_OPS  - torch.* functions and operator.* to layer types
  2. METHOD_OPS    - Tensor.* method names to layer types
  3. Shared tables - ops that appear in both contexts (arithmetic, trig, etc.)
"""

import operator
import torch
import torch.nn.functional as F

from nntrainer_layers import (
    LAYER_ADDITION, LAYER_SUBTRACT, LAYER_MULTIPLY, LAYER_DIVIDE,
    LAYER_MATMUL, LAYER_POW, LAYER_SQRT, LAYER_NEGATIVE,
    LAYER_SIN, LAYER_COS, LAYER_TAN,
    LAYER_EXP, LAYER_LOG, LAYER_CLAMP,
    LAYER_REDUCE_MEAN, LAYER_REDUCE_SUM, LAYER_FLATTEN,
    LAYER_ACTIVATION, LAYER_DROPOUT, LAYER_CONCAT,
    LAYER_GATHER, LAYER_SLICE,
    ACT_RELU, ACT_GELU, ACT_SWISH, ACT_SIGMOID, ACT_TANH, ACT_SOFTMAX,
    OP_RESHAPE, OP_TRANSPOSE, OP_PERMUTE, OP_SDPA, OP_NOOP, OP_UNSUPPORTED,
)


# ---------------------------------------------------------------------------
# Function-based ops: maps (callable) -> layer_type
# Simple ops that only need layer_type + standard input_layers.
# ---------------------------------------------------------------------------
FUNCTION_SIMPLE_OPS = {
    # Arithmetic
    torch.add:         LAYER_ADDITION,
    operator.add:      LAYER_ADDITION,
    torch.sub:         LAYER_SUBTRACT,
    operator.sub:      LAYER_SUBTRACT,
    torch.mul:         LAYER_MULTIPLY,
    operator.mul:      LAYER_MULTIPLY,
    torch.div:         LAYER_DIVIDE,
    operator.truediv:  LAYER_DIVIDE,
    torch.matmul:      LAYER_MATMUL,
    torch.pow:         LAYER_POW,
    torch.sqrt:        LAYER_SQRT,
    torch.neg:         LAYER_NEGATIVE,
    operator.neg:      LAYER_NEGATIVE,
    # Trigonometric
    torch.sin:         LAYER_SIN,
    torch.cos:         LAYER_COS,
    torch.tan:         LAYER_TAN,
    # Reduction
    torch.mean:        LAYER_REDUCE_MEAN,
    torch.sum:         LAYER_REDUCE_SUM,
    # Shape
    torch.flatten:     LAYER_FLATTEN,
    torch.reshape:     OP_RESHAPE,
    torch.transpose:   OP_TRANSPOSE,
}

# Function ops looked up by name (func.__name__) rather than identity.
# These are for cases where the callable can't be compared by `is`.
FUNCTION_NAME_SIMPLE_OPS = {
    "exp":  LAYER_EXP,
    "log":  LAYER_LOG,
}

# Functions that map to OP_NOOP (internal torch/runtime functions)
FUNCTION_NOOP_NAMES = frozenset({
    "_set_grad_enabled", "tensor", "arange",
    "zeros", "zeros_like", "ones", "ones_like",
    "full_like", "empty_like",
    "custom_function_call",
    # Comparison / conditional ops (T5 relative position bias)
    "where", "min", "max",
})

# Functions that map to OP_RESHAPE
FUNCTION_RESHAPE_NAMES = frozenset({"unsqueeze", "squeeze"})

# Functions that map to OP_UNSUPPORTED with decomposition info
FUNCTION_DECOMPOSE_OPS = {
    "rsqrt":      {"original_op": "rsqrt",      "decompose_to": "pow(x, -0.5)"},
    "abs":        {"original_op": "abs",         "decompose_to": "sqrt(pow(x, 2))"},
    "reciprocal": {"original_op": "reciprocal",  "decompose_to": "divide(1, x)"},
}

# Function-based activation mappings: callable -> activation type
FUNCTION_ACTIVATION_OPS = {
    F.silu:     ACT_SWISH,
    F.gelu:     ACT_GELU,
    F.relu:     ACT_RELU,
    F.softmax:  ACT_SOFTMAX,
    torch.sigmoid: ACT_SIGMOID,
}

# Function activation by name (for torch.tanh / F.tanh ambiguity)
FUNCTION_ACTIVATION_NAMES = {
    "tanh": ACT_TANH,
}

# Function-based identity ops: callable -> layer_type
# These are checked by `func is <target>` and need special handling
# (e.g. multi-input extraction or parameter extraction).
FUNCTION_IDENTITY_OPS = {
    F.dropout:  LAYER_DROPOUT,
    F.pad:      OP_NOOP,
}

# Function-based clamp names (matching METHOD_CLAMP_NAMES pattern)
FUNCTION_CLAMP_NAMES = frozenset({
    "clamp", "clip", "clamp_min", "clamp_max",
})

# Layer types that produce tuple outputs (for operator.getitem handling)
MULTI_OUTPUT_LAYER_TYPES = frozenset({"gru", "lstm", "rnn"})

# ---------------------------------------------------------------------------
# Method-based ops: maps method_name (str) -> layer_type
# Simple ops that only need layer_type + standard input_layers.
# ---------------------------------------------------------------------------
METHOD_SIMPLE_OPS = {
    # Arithmetic
    "add":    LAYER_ADDITION,
    "add_":   LAYER_ADDITION,
    "sub":    LAYER_SUBTRACT,
    "sub_":   LAYER_SUBTRACT,
    "mul":    LAYER_MULTIPLY,
    "mul_":   LAYER_MULTIPLY,
    "div":    LAYER_DIVIDE,
    "div_":   LAYER_DIVIDE,
    "matmul": LAYER_MATMUL,
    "neg":    LAYER_NEGATIVE,
    "pow":    LAYER_POW,
    "sqrt":   LAYER_SQRT,
    # Trigonometric
    "cos":    LAYER_COS,
    "sin":    LAYER_SIN,
    # Reduction
    "mean":   LAYER_REDUCE_MEAN,
    "sum":    LAYER_REDUCE_SUM,
    # Shape
    "flatten": LAYER_FLATTEN,
    # Math
    "exp":    LAYER_EXP,
    "log":    LAYER_LOG,
    # Indexing
    "__getitem__": LAYER_SLICE,
}

# Method-based activation mappings: method_name -> activation type
METHOD_ACTIVATION_OPS = {
    "relu":      ACT_RELU,
    "relu_":     ACT_RELU,
    "sigmoid":   ACT_SIGMOID,
    "sigmoid_":  ACT_SIGMOID,
    "tanh":      ACT_TANH,
    "tanh_":     ACT_TANH,
    "softmax":   ACT_SOFTMAX,
}

# Method-based shape ops: method_name -> layer_type
METHOD_SHAPE_OPS = {
    "view":      OP_RESHAPE,
    "reshape":   OP_RESHAPE,
    "permute":   OP_PERMUTE,
    "transpose": OP_TRANSPOSE,
}

# Method ops that map to OP_RESHAPE
METHOD_RESHAPE_NAMES = frozenset({"unsqueeze", "squeeze", "repeat", "expand_as"})

# Method ops that map to OP_UNSUPPORTED with decomposition info
METHOD_DECOMPOSE_OPS = {
    "rsqrt": {"original_op": "rsqrt", "decompose_to": "pow(x, -0.5)"},
    "abs":   {"original_op": "abs",   "decompose_to": "sqrt(pow(x, 2))"},
}

# Method ops that are clamp variants
METHOD_CLAMP_NAMES = frozenset({
    "clamp", "clip", "clamp_", "clamp_min", "clamp_max",
})

# Method ops that are no-ops in inference
METHOD_NOOP_NAMES = frozenset({
    "contiguous", "detach", "clone", "to", "float",
    "half", "bfloat16", "type_as", "expand",
    "size", "dim", "numel",
    "new_ones", "new_zeros", "new_full", "new_empty",
    "fill_", "zero_", "masked_fill", "masked_fill_",
    "__bool__", "__or__", "__and__",
    "__xor__", "__invert__",
    "all", "any", "item",
    # Comparison ops (T5 relative position bias)
    "gt", "lt", "le", "ge", "eq", "ne",
})

# Method ops that map to "split"
METHOD_SPLIT_NAMES = frozenset({"chunk", "split"})

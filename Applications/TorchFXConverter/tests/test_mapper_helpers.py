"""Tests for mapper_helpers.py and op_registry.py extensions.

Tests:
  - mapper_helpers: sanitize_name, make_scoped_name, extract_clamp_params,
                    get_input_node_names
  - op_registry: FUNCTION_IDENTITY_OPS, FUNCTION_CLAMP_NAMES,
                 MULTI_OUTPUT_LAYER_TYPES
  - Backward compatibility: old aliases in mapper files still work
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Mock FX node for testing
# ============================================================================

class _MockArg:
    def __init__(self, name):
        self.name = name


class _MockNode:
    def __init__(self, name, args=None, kwargs=None, meta=None):
        self.name = name
        self.args = args or []
        self.kwargs = kwargs or {}
        self.meta = meta or {}


# ============================================================================
# Test mapper_helpers.py
# ============================================================================

def test_sanitize_name():
    from mapper_helpers import sanitize_name
    assert sanitize_name("model.layers.0.self_attn") == "model_layers_0_self_attn"
    assert sanitize_name("simple") == "simple"
    assert sanitize_name("a.b.c") == "a_b_c"


def test_make_scoped_name_with_scope():
    from mapper_helpers import make_scoped_name
    node = _MockNode("add_1")
    result = make_scoped_name("model.layers.0", node)
    assert result == "model_layers_0_add_1"


def test_make_scoped_name_without_scope():
    from mapper_helpers import make_scoped_name
    node = _MockNode("add_1")
    result = make_scoped_name("", node)
    assert result == "add_1"


def test_make_scoped_name_with_suffix():
    from mapper_helpers import make_scoped_name
    node = _MockNode("act")
    result = make_scoped_name("model.layers.0", node, suffix="silu")
    assert result == "model_layers_0_act_silu"


def test_get_input_node_names():
    from mapper_helpers import get_input_node_names
    node = _MockNode("add_1", args=[_MockArg("x"), _MockArg("y"), 42])
    result = get_input_node_names(node)
    assert result == ["x", "y"]


def test_get_input_node_names_empty():
    from mapper_helpers import get_input_node_names
    node = _MockNode("const", args=[42, "hello"])
    result = get_input_node_names(node)
    assert result == []


def test_extract_clamp_params_basic():
    from mapper_helpers import extract_clamp_params
    node = _MockNode("clamp", args=[_MockArg("x"), 0.0, 1.0])
    props = {}
    extract_clamp_params(node, "clamp", props)
    assert props["min"] == "0.0"
    assert props["max"] == "1.0"


def test_extract_clamp_params_min_only():
    from mapper_helpers import extract_clamp_params
    node = _MockNode("clamp_min", args=[_MockArg("x"), 0.0])
    props = {}
    extract_clamp_params(node, "clamp_min", props)
    assert props["min"] == "0.0"
    assert props["max"] == str(math.inf)


def test_extract_clamp_params_max_only():
    from mapper_helpers import extract_clamp_params
    node = _MockNode("clamp_max", args=[_MockArg("x"), 1.0])
    props = {}
    extract_clamp_params(node, "clamp_max", props)
    assert props["min"] == str(-math.inf)
    assert props["max"] == "1.0"


def test_extract_clamp_params_kwargs():
    from mapper_helpers import extract_clamp_params
    node = _MockNode("clamp", args=[_MockArg("x")],
                     kwargs={"min": -5, "max": 5})
    props = {}
    extract_clamp_params(node, "clamp", props)
    assert props["min"] == "-5"
    assert props["max"] == "5"


def test_extract_clamp_params_no_bounds():
    from mapper_helpers import extract_clamp_params
    node = _MockNode("clamp", args=[_MockArg("x")])
    props = {}
    extract_clamp_params(node, "clamp", props)
    assert props["min"] == str(-math.inf)
    assert props["max"] == str(math.inf)


# ============================================================================
# Test op_registry.py extensions
# ============================================================================

def test_function_identity_ops():
    import torch.nn.functional as F
    from op_registry import FUNCTION_IDENTITY_OPS
    from nntrainer_layers import LAYER_DROPOUT, OP_NOOP

    assert F.dropout in FUNCTION_IDENTITY_OPS
    assert FUNCTION_IDENTITY_OPS[F.dropout] == LAYER_DROPOUT
    assert F.pad in FUNCTION_IDENTITY_OPS
    assert FUNCTION_IDENTITY_OPS[F.pad] == OP_NOOP


def test_function_clamp_names():
    from op_registry import FUNCTION_CLAMP_NAMES
    assert "clamp" in FUNCTION_CLAMP_NAMES
    assert "clip" in FUNCTION_CLAMP_NAMES
    assert "clamp_min" in FUNCTION_CLAMP_NAMES
    assert "clamp_max" in FUNCTION_CLAMP_NAMES


def test_multi_output_layer_types():
    from op_registry import MULTI_OUTPUT_LAYER_TYPES
    assert "gru" in MULTI_OUTPUT_LAYER_TYPES
    assert "lstm" in MULTI_OUTPUT_LAYER_TYPES
    assert "rnn" in MULTI_OUTPUT_LAYER_TYPES


# ============================================================================
# Test backward compatibility (old aliases)
# ============================================================================

def test_function_mapper_backward_compat_aliases():
    """function_mapper still exports old private names."""
    from function_mapper import (
        _get_input_node_names, _sanitize_name,
        _make_scoped_name, _extract_clamp_params,
        _MULTI_OUTPUT_LAYER_TYPES,
    )
    from mapper_helpers import (
        get_input_node_names, sanitize_name,
        make_scoped_name, extract_clamp_params,
    )
    from op_registry import MULTI_OUTPUT_LAYER_TYPES

    assert _get_input_node_names is get_input_node_names
    assert _sanitize_name is sanitize_name
    assert _make_scoped_name is make_scoped_name
    assert _extract_clamp_params is extract_clamp_params
    assert _MULTI_OUTPUT_LAYER_TYPES is MULTI_OUTPUT_LAYER_TYPES


def test_method_mapper_backward_compat_aliases():
    """method_mapper still exports old private names."""
    from method_mapper import (
        _get_input_node_names, _sanitize_name,
        _make_scoped_name, _extract_clamp_params,
    )
    from mapper_helpers import (
        get_input_node_names, sanitize_name,
        make_scoped_name, extract_clamp_params,
    )
    assert _get_input_node_names is get_input_node_names
    assert _sanitize_name is sanitize_name
    assert _make_scoped_name is make_scoped_name
    assert _extract_clamp_params is extract_clamp_params


def test_module_mapper_backward_compat_aliases():
    """module_mapper still exports old private names."""
    from module_mapper import _sanitize_name, _get_input_node_names
    from mapper_helpers import sanitize_name, get_input_node_names
    assert _sanitize_name is sanitize_name
    assert _get_input_node_names is get_input_node_names


def test_node_mapper_backward_compat_aliases():
    """node_mapper still exports old private names."""
    from node_mapper import _get_input_node_names, _sanitize_name
    from mapper_helpers import get_input_node_names, sanitize_name
    assert _get_input_node_names is get_input_node_names
    assert _sanitize_name is sanitize_name

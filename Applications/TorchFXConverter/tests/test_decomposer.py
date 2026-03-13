"""Test adaptive decomposition pipeline with LazyTensor support.

Tests:
  1. Known models (Qwen3, BERT, mT5) pass through without decomposition
  2. Custom unknown modules are automatically decomposed into tensor ops
  3. Unsupported ops resolved via Tensor methods or layer decomposition
  4. LazyTensor chain detection for consecutive arithmetic ops
  5. Exclude leaf types in tracer
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from decomposer import (
    AdaptiveConverter, resolve_unsupported_ops, detect_lazy_chains,
    LazyTensorChain,
)
from nntrainer_layers import (
    NNTrainerLayerDef, OP_UNSUPPORTED, LAYER_POW, LAYER_SQRT,
    LAYER_MULTIPLY, LAYER_DIVIDE, LAYER_ADD, LAYER_ADDITION, LAYER_SUBTRACT,
    TENSOR_DIRECT_METHODS,
)

# backward compat alias
decompose_unsupported_ops = resolve_unsupported_ops


# ============================================================================
# Test 1: Known models pass through cleanly (single pass, no decomposition)
# ============================================================================

def test_qwen3_no_decomposition():
    """Qwen3 should be fully mapped in a single pass."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(
        vocab_size=151936, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=2048, rms_norm_eps=1e-6,
        tie_word_embeddings=True, rope_theta=1000000.0, sliding_window=None,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()

    converter = AdaptiveConverter(model, config)
    result = converter.convert({"input_ids": torch.randint(0, config.vocab_size, (1, 8))})

    result.summary()
    assert len(result.unknown_layers) == 0, f"Qwen3 has unknowns: {result.unknown_layers}"
    assert len(result.decomposed_module_types) == 0, "Qwen3 should not need module decomposition"
    print("  PASS: Qwen3 fully mapped without decomposition")


def test_bert_no_decomposition():
    """BERT should be fully mapped in a single pass."""
    from transformers import BertConfig, BertModel
    config = BertConfig(
        vocab_size=30522, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=128, max_position_embeddings=512,
    )
    model = BertModel(config)
    model.eval()

    converter = AdaptiveConverter(model, config)
    result = converter.convert({
        "input_ids": torch.randint(0, config.vocab_size, (1, 16)),
        "attention_mask": torch.ones(1, 16, dtype=torch.long),
    })

    result.summary()
    assert len(result.unknown_layers) == 0, f"BERT has unknowns: {result.unknown_layers}"
    print("  PASS: BERT fully mapped without decomposition")


def test_mt5_no_decomposition():
    """mT5 should be fully mapped in a single pass."""
    from transformers import MT5Config, MT5ForConditionalGeneration
    config = MT5Config(
        vocab_size=250112, d_model=64, d_kv=16, d_ff=128,
        num_heads=4, num_layers=2, num_decoder_layers=2,
        relative_attention_num_buckets=32, relative_attention_max_distance=128,
    )
    model = MT5ForConditionalGeneration(config)
    model.eval()

    converter = AdaptiveConverter(model, config)
    result = converter.convert({
        "input_ids": torch.randint(0, config.vocab_size, (1, 8)),
        "decoder_input_ids": torch.randint(0, config.vocab_size, (1, 4)),
    })

    result.summary()
    assert len(result.unknown_layers) == 0, f"mT5 has unknowns: {result.unknown_layers}"
    print("  PASS: mT5 fully mapped without decomposition")


# ============================================================================
# Test 2: Custom unknown module gets decomposed into tensor ops
# ============================================================================

class CustomNorm(nn.Module):
    """A hypothetical normalization not in LEAF_MODULES or RMSNorm detection.

    forward() uses: mean, subtract, pow, mean, add, rsqrt, multiply
    All of these should be captured as tensor ops when decomposed.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        x = x - mean
        var = (x * x).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class ModelWithCustomNorm(nn.Module):
    """Simple model using a custom normalization module."""
    def __init__(self, dim=64):
        super().__init__()
        self.embed = nn.Embedding(1000, dim)
        self.norm = CustomNorm(dim)
        self.proj = nn.Linear(dim, 1000)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.norm(x)
        x = self.proj(x)
        return x


def test_custom_module_auto_decomposition():
    """Custom module NOT in LEAF_MODULES is auto-decomposed by the tracer.

    Since CustomNorm is not in LEAF_MODULES, the tracer automatically traces
    through its forward(), decomposing it into tensor ops. No 2nd pass needed.
    This is the default behavior for unknown modules.
    """
    model = ModelWithCustomNorm(dim=64)
    model.eval()

    converter = AdaptiveConverter(model)
    result = converter.convert({"input_ids": torch.randint(0, 1000, (1, 8))})

    result.summary()

    # No unknown module layers should remain (CustomNorm was never a leaf)
    assert len(result.unknown_layers) == 0, \
        f"Should have no unknown layers, got: {result.unknown_layers}"

    # Check that tensor ops from CustomNorm's forward() are present
    layer_types = {l.layer_type for l in result.layers}
    assert "reduce_mean" in layer_types or "subtract" in layer_types or "multiply" in layer_types, \
        f"Expected tensor ops from decomposed CustomNorm, got types: {sorted(layer_types)}"

    print("  PASS: CustomNorm auto-decomposed into tensor ops (not in LEAF_MODULES)")


def test_forced_leaf_decomposition():
    """Test the 2-pass mechanism: force a known module type to be leaf, then decompose.

    This simulates a scenario where a module IS detected as a leaf (e.g. through
    LEAF_MODULES or RMSNorm detection) but the mapper can't handle it. The
    adaptive converter should detect this and re-trace with that type excluded.
    """
    from tracer import Tracer, LEAF_MODULES
    from node_mapper import NodeMapper

    model = ModelWithCustomNorm(dim=32)
    model.eval()
    input_kwargs = {"input_ids": torch.randint(0, 1000, (1, 4))}

    # Manually trace with CustomNorm as a LEAF (simulating a new HF module type
    # that was added to LEAF_MODULES but not to NodeMapper)
    extended_leaves = LEAF_MODULES + (CustomNorm,)
    tracer = Tracer(model, leaf_modules=extended_leaves)
    with tracer:
        with torch.no_grad():
            model(**input_kwargs)

    mapper = NodeMapper(model, tracer.graph)
    layers = mapper.map_all()

    # CustomNorm should appear as unknown(CustomNorm)
    unknown_types = mapper.get_unknown_module_types(layers)
    assert "CustomNorm" in unknown_types, \
        f"CustomNorm should be unknown when forced as leaf, got: {unknown_types}"

    # Now re-trace with CustomNorm excluded -> decomposition
    tracer2 = Tracer(model, leaf_modules=extended_leaves,
                     exclude_leaf_types={"CustomNorm"})
    with tracer2:
        with torch.no_grad():
            model(**input_kwargs)

    mapper2 = NodeMapper(model, tracer2.graph)
    layers2 = mapper2.map_all()
    unknown_types2 = mapper2.get_unknown_module_types(layers2)

    assert "CustomNorm" not in unknown_types2, \
        f"CustomNorm should be decomposed after exclusion, still unknown: {unknown_types2}"

    print("  PASS: 2-pass decomposition works (leaf -> exclude -> tensor ops)")


# ============================================================================
# Test 3: Op-level decomposition (rsqrt, abs)
# ============================================================================

def test_rsqrt_tensor_method():
    """rsqrt should resolve to Tensor::inv_sqrt() (direct method)."""
    layer = NNTrainerLayerDef(
        layer_type=OP_UNSUPPORTED,
        name="test_rsqrt",
        properties={"original_op": "rsqrt"},
        input_layers=["input"],
    )
    result = resolve_unsupported_ops([layer])
    assert len(result) == 1
    assert result[0].layer_type == "tensor_op:rsqrt"
    assert result[0].properties["tensor_method"] == "inv_sqrt"
    print("  PASS: rsqrt -> Tensor::inv_sqrt() (direct method)")


def test_abs_tensor_method():
    """abs should resolve to Tensor::abs() (direct method)."""
    layer = NNTrainerLayerDef(
        layer_type=OP_UNSUPPORTED,
        name="test_abs",
        properties={"original_op": "abs"},
        input_layers=["input"],
    )
    result = resolve_unsupported_ops([layer])
    assert len(result) == 1
    assert result[0].layer_type == "tensor_op:abs"
    assert result[0].properties["tensor_method"] == "abs"
    print("  PASS: abs -> Tensor::abs() (direct method)")


def test_all_tensor_direct_methods():
    """All ops in TENSOR_DIRECT_METHODS should resolve to tensor_op: types."""
    for op_name, (method_name, _) in TENSOR_DIRECT_METHODS.items():
        layer = NNTrainerLayerDef(
            layer_type=OP_UNSUPPORTED,
            name=f"test_{op_name}",
            properties={"original_op": op_name},
            input_layers=["input"],
        )
        result = resolve_unsupported_ops([layer])
        assert len(result) == 1, f"{op_name}: expected 1 result, got {len(result)}"
        assert result[0].layer_type == f"tensor_op:{op_name}", \
            f"{op_name}: expected tensor_op:{op_name}, got {result[0].layer_type}"
        assert result[0].properties["tensor_method"] == method_name, \
            f"{op_name}: expected method {method_name}"
    print(f"  PASS: All {len(TENSOR_DIRECT_METHODS)} Tensor direct methods resolve correctly")


def test_reciprocal_layer_decomposition():
    """reciprocal should decompose to divide(1, x) via layer decomposition."""
    layer = NNTrainerLayerDef(
        layer_type=OP_UNSUPPORTED,
        name="test_reciprocal",
        properties={"original_op": "reciprocal"},
        input_layers=["input"],
    )
    result = resolve_unsupported_ops([layer])
    assert len(result) == 1
    assert result[0].layer_type == LAYER_DIVIDE
    assert result[0].properties.get("numerator") == 1.0
    print("  PASS: reciprocal -> divide(1, x) (layer decomposition)")


def test_unsupported_op_preserved():
    """Ops without any resolution should be preserved with warning."""
    layer = NNTrainerLayerDef(
        layer_type=OP_UNSUPPORTED,
        name="test_exp",
        properties={"original_op": "exp"},
        input_layers=["input"],
    )
    result = resolve_unsupported_ops([layer])
    assert len(result) == 1
    assert result[0].layer_type == OP_UNSUPPORTED
    print("  PASS: exp preserved as unsupported (no resolution available)")


# ============================================================================
# Test 3b: LazyTensor chain detection
# ============================================================================

def test_lazy_chain_detection():
    """Consecutive arithmetic ops should be detected as LazyTensor chains."""
    layers = [
        NNTrainerLayerDef(layer_type="embedding_layer", name="embed", input_layers=[]),
        NNTrainerLayerDef(layer_type=LAYER_ADD, name="add1", input_layers=["embed"]),
        NNTrainerLayerDef(layer_type=LAYER_MULTIPLY, name="mul1", input_layers=["add1"]),
        NNTrainerLayerDef(layer_type=LAYER_SUBTRACT, name="sub1", input_layers=["mul1"]),
        NNTrainerLayerDef(layer_type="fully_connected", name="fc", input_layers=["sub1"]),
        NNTrainerLayerDef(layer_type=LAYER_ADDITION, name="residual", input_layers=["fc"]),
        NNTrainerLayerDef(layer_type=LAYER_DIVIDE, name="div1", input_layers=["residual"]),
    ]

    chains = detect_lazy_chains(layers)
    assert len(chains) == 2, f"Expected 2 chains, got {len(chains)}"

    # First chain: add1 -> mul1 -> sub1 (3 ops)
    assert chains[0].chain_length == 3
    assert chains[0].start_idx == 1

    # Second chain: residual -> div1 (2 ops)
    assert chains[1].chain_length == 2
    assert chains[1].start_idx == 5

    print(f"  PASS: LazyTensor chains detected: {chains}")


def test_lazy_chain_cpp_generation():
    """LazyTensor chain should generate valid C++ code."""
    layers = [
        NNTrainerLayerDef(layer_type=LAYER_ADD, name="add1",
                          input_layers=["x", "bias"]),
        NNTrainerLayerDef(layer_type=LAYER_MULTIPLY, name="mul1",
                          input_layers=["add1", "scale"]),
    ]

    chain = LazyTensorChain(layers, 0)
    cpp = chain.to_cpp_chain("hidden")
    assert "hidden.chain()" in cpp
    assert "add_i(bias)" in cpp
    assert "multiply_i(scale)" in cpp
    assert "run()" in cpp
    print(f"  PASS: LazyTensor C++ code: {cpp}")


# ============================================================================
# Test 4: Exclude leaf types in tracer
# ============================================================================

def test_exclude_leaf_types():
    """Verify that exclude_leaf_types forces module decomposition."""
    from tracer import Tracer

    model = ModelWithCustomNorm(dim=32)
    model.eval()
    input_ids = torch.randint(0, 1000, (1, 4))

    # Pass 1: Normal tracing - Linear and Embedding appear as leaf modules
    tracer1 = Tracer(model)
    with tracer1:
        with torch.no_grad():
            model(input_ids)

    linear_modules_1 = [n for n in tracer1.graph.nodes
                        if n.op == "call_module" and n.meta.get("module_type") == "Linear"]
    assert len(linear_modules_1) > 0, "Linear should be a leaf module"

    # Pass 2: Exclude Linear -> no Linear call_module nodes
    tracer2 = Tracer(model, exclude_leaf_types={"Linear"})
    with tracer2:
        with torch.no_grad():
            model(input_ids)

    linear_modules_2 = [n for n in tracer2.graph.nodes
                        if n.op == "call_module" and n.meta.get("module_type") == "Linear"]
    assert len(linear_modules_2) == 0, \
        "Linear should NOT appear as leaf when excluded"

    print(f"  PASS: exclude_leaf_types works (Linear leaf nodes: {len(linear_modules_1)} -> {len(linear_modules_2)})")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEST: Tensor method resolution (rsqrt, abs, etc.)")
    print("=" * 70)
    test_rsqrt_tensor_method()
    test_abs_tensor_method()
    test_all_tensor_direct_methods()
    test_reciprocal_layer_decomposition()
    test_unsupported_op_preserved()

    print("\n" + "=" * 70)
    print("TEST: LazyTensor chain detection")
    print("=" * 70)
    test_lazy_chain_detection()
    test_lazy_chain_cpp_generation()

    print("\n" + "=" * 70)
    print("TEST: Tracer exclude_leaf_types")
    print("=" * 70)
    test_exclude_leaf_types()

    print("\n" + "=" * 70)
    print("TEST: Custom module decomposition")
    print("=" * 70)
    test_custom_module_auto_decomposition()
    test_forced_leaf_decomposition()

    print("\n" + "=" * 70)
    print("TEST: Known models (no decomposition needed)")
    print("=" * 70)
    test_qwen3_no_decomposition()
    test_bert_no_decomposition()
    test_mt5_no_decomposition()

    print("\n" + "=" * 70)
    print("ALL DECOMPOSER TESTS PASSED!")
    print("=" * 70)

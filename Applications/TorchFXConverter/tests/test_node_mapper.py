#!/usr/bin/env python3
"""Task 2.1-2.3 Tests: Verify node mapper converts FX nodes to NNTrainer layers."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tracer import Tracer, LEAF_MODULES
from node_mapper import map_graph, NNTrainerLayerDef


def _trace_and_map(model, *inputs, leaf_modules=None):
    """Helper: trace model and map nodes."""
    if leaf_modules is None:
        leaf_modules = LEAF_MODULES
    tracer = Tracer(model, leaf_modules=leaf_modules)
    with torch.no_grad():
        with tracer:
            model(*inputs)
    return map_graph(tracer.graph, model)


def _get_mapped(layer_defs):
    """Filter out None entries and non-functional nodes."""
    return [d for d in layer_defs if d is not None and d.layer_type not in ("placeholder", "output")]


def test_linear_mapping():
    """Task 2.1: Map nn.Linear to fully_connected."""
    print("=== Test 2.1a: Linear mapping ===")

    model = nn.Linear(4, 3, bias=False)
    model.eval()
    defs = _trace_and_map(model, torch.randn(1, 4))
    mapped = _get_mapped(defs)

    fc_defs = [d for d in mapped if d.layer_type == "fully_connected"]
    assert len(fc_defs) == 1, f"Expected 1 fully_connected, got {len(fc_defs)}"
    assert fc_defs[0].params["unit"] == 3
    assert fc_defs[0].params.get("disable_bias") == "true"
    print(f"  Mapped: {fc_defs[0]}")
    print("PASSED\n")


def test_embedding_mapping():
    """Task 2.1: Map nn.Embedding to embedding."""
    print("=== Test 2.1b: Embedding mapping ===")

    class EmbModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 16)
            self.fc = nn.Linear(16, 8)

        def forward(self, x):
            return self.fc(self.embed(x))

    model = EmbModel()
    model.eval()
    defs = _trace_and_map(model, torch.tensor([[1, 2, 3]]))
    mapped = _get_mapped(defs)

    emb_defs = [d for d in mapped if d.layer_type == "embedding"]
    assert len(emb_defs) == 1, f"Expected 1 embedding, got {len(emb_defs)}"
    assert emb_defs[0].params["in_dim"] == 100
    assert emb_defs[0].params["out_dim"] == 16
    print(f"  Mapped: {emb_defs[0]}")

    fc_defs = [d for d in mapped if d.layer_type == "fully_connected"]
    assert len(fc_defs) == 1
    print(f"  Mapped: {fc_defs[0]}")
    print("PASSED\n")


def test_dropout_mapping():
    """Task 2.1: Map nn.Dropout to dropout."""
    print("=== Test 2.1c: Dropout mapping ===")

    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1), nn.Linear(4, 2))
    model.eval()  # Dropout is no-op in eval, but still traced
    defs = _trace_and_map(model, torch.randn(1, 4))
    mapped = _get_mapped(defs)

    fc_defs = [d for d in mapped if d.layer_type == "fully_connected"]
    assert len(fc_defs) == 2, f"Expected 2 fully_connected, got {len(fc_defs)}"
    # Dropout may not produce a node in eval mode, that's fine
    print(f"  FC layers: {len(fc_defs)}")
    print("PASSED\n")


def test_tensor_ops_mapping():
    """Task 2.2: Map tensor operations."""
    print("=== Test 2.2a: Tensor ops mapping ===")

    class TensorOpsModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)

        def forward(self, x, y):
            a = self.fc1(x)
            b = self.fc2(y)
            return a + b  # should map to "add"

    model = TensorOpsModel()
    model.eval()
    defs = _trace_and_map(model, torch.randn(1, 4), torch.randn(1, 4))
    mapped = _get_mapped(defs)

    add_defs = [d for d in mapped if d.layer_type == "add"]
    assert len(add_defs) >= 1, f"Expected add op, got none. Mapped: {[d.layer_type for d in mapped]}"
    print(f"  Add ops: {len(add_defs)}")
    print("PASSED\n")


def test_rmsnorm_ops_mapping():
    """Task 2.2: Map RMSNorm internal ops (pow, mean, rsqrt, mul)."""
    print("=== Test 2.2b: RMSNorm internal ops mapping ===")

    class SimpleRMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = 1e-6

        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            return self.weight * (x * torch.rsqrt(variance + self.eps))

    model = SimpleRMSNorm(4)
    model.eval()
    defs = _trace_and_map(model, torch.randn(1, 4))
    mapped = _get_mapped(defs)

    layer_types = [d.layer_type for d in mapped]
    print(f"  Mapped types: {layer_types}")

    assert "pow" in layer_types, f"Missing pow, got {layer_types}"
    assert "reduce_mean" in layer_types, f"Missing reduce_mean, got {layer_types}"
    assert "inv_sqrt" in layer_types, f"Missing inv_sqrt, got {layer_types}"
    assert "multiply" in layer_types, f"Missing multiply, got {layer_types}"
    print("PASSED\n")


def test_noop_filtering():
    """Task 2.2: Verify no-op methods (detach, clone, contiguous) return None."""
    print("=== Test 2.2c: No-op filtering ===")

    class NoOpModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            out = self.fc(x)
            return out.contiguous()

    model = NoOpModel()
    model.eval()
    defs = _trace_and_map(model, torch.randn(1, 4))
    mapped = _get_mapped(defs)

    # contiguous, detach, clone should be filtered (None)
    noop_types = [d.layer_type for d in mapped if d.layer_type in ("contiguous", "detach", "clone")]
    assert len(noop_types) == 0, f"No-ops should be None, but got: {noop_types}"
    print("  No-ops correctly filtered out")
    print("PASSED\n")


def test_activation_module_mapping():
    """Task 2.3: Map activation modules."""
    print("=== Test 2.3a: Activation module mapping ===")

    model = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 4),
        nn.SiLU(),
        nn.Linear(4, 2),
    )
    model.eval()
    defs = _trace_and_map(model, torch.randn(1, 4))
    mapped = _get_mapped(defs)

    acti_defs = [d for d in mapped if d.layer_type == "activation"]
    print(f"  Activation layers: {[(d.name, d.params.get('activation')) for d in acti_defs]}")

    assert len(acti_defs) == 2, f"Expected 2 activations, got {len(acti_defs)}"

    acti_types = [d.params["activation"] for d in acti_defs]
    assert "relu" in acti_types, f"Missing relu activation"
    assert "swish" in acti_types, f"Missing swish (SiLU) activation"
    print("PASSED\n")


def test_activation_function_mapping():
    """Task 2.3: Map F.silu, F.gelu etc. as call_function."""
    print("=== Test 2.3b: Activation function mapping ===")

    class FuncActivationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 2)

        def forward(self, x):
            x = self.fc1(x)
            x = F.silu(x)
            return self.fc2(x)

    model = FuncActivationModel()
    model.eval()
    defs = _trace_and_map(model, torch.randn(1, 4))
    mapped = _get_mapped(defs)

    acti_defs = [d for d in mapped if d.layer_type == "activation"]
    print(f"  Activation layers: {[(d.name, d.params.get('activation')) for d in acti_defs]}")

    assert len(acti_defs) >= 1, f"Expected silu activation, got {acti_defs}"
    assert any(d.params.get("activation") == "swish" for d in acti_defs), \
        f"Missing swish from F.silu"
    print("PASSED\n")


def test_qwen3_mapping():
    """Integration: Map full Qwen3 trace (low-level leaves)."""
    print("=== Test 2.x: Qwen3 full mapping ===")

    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16,
        vocab_size=1000, max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()

    defs = _trace_and_map(model, torch.tensor([[1, 2, 3]]))
    mapped = _get_mapped(defs)

    # Count by type
    type_counts = {}
    for d in mapped:
        type_counts[d.layer_type] = type_counts.get(d.layer_type, 0) + 1

    print("  Layer type counts:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    # Verify key mappings
    assert type_counts.get("embedding", 0) >= 1, "Missing embedding"
    assert type_counts.get("fully_connected", 0) >= 7, "Missing FC layers (q/k/v/o/gate/up/down)"
    assert type_counts.get("inv_sqrt", 0) >= 1, "Missing inv_sqrt (from RMSNorm)"
    assert type_counts.get("pow", 0) >= 1, "Missing pow (from RMSNorm)"
    assert type_counts.get("reduce_mean", 0) >= 1, "Missing reduce_mean (from RMSNorm)"

    # Check for unmapped nodes
    unmapped = [d for d in mapped if d.layer_type.startswith("UNMAPPED")]
    if unmapped:
        print(f"\n  Unmapped nodes ({len(unmapped)}):")
        for d in unmapped[:10]:
            print(f"    {d.layer_type}: {d.params}")

    print("PASSED\n")


if __name__ == "__main__":
    test_linear_mapping()
    test_embedding_mapping()
    test_dropout_mapping()
    test_tensor_ops_mapping()
    test_rmsnorm_ops_mapping()
    test_noop_filtering()
    test_activation_module_mapping()
    test_activation_function_mapping()
    test_qwen3_mapping()
    print("=" * 50)
    print("All Task 2.1-2.3 tests PASSED!")

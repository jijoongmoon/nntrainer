#!/usr/bin/env python3
"""Task 1.1 Test: Verify the runtime tracer works on basic models."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from tracer import Tracer, LEAF_MODULES


def test_single_linear():
    """Trace a single nn.Linear and verify graph structure."""
    print("=== Test 1: Single nn.Linear ===")

    model = nn.Linear(4, 3)
    model.eval()

    tracer = Tracer(model, leaf_modules=LEAF_MODULES)
    with torch.no_grad():
        with tracer:
            x = torch.randn(1, 4)
            out = model(x)

    graph = tracer.graph
    print("\nGraph nodes:")
    for node in graph.nodes:
        print(f"  op={node.op:15s} name={node.name:20s} target={node.target}")

    # Verify node types
    nodes = list(graph.nodes)
    ops = [n.op for n in nodes]

    assert "placeholder" in ops, "Missing placeholder node for input"
    assert "output" in ops, "Missing output node"
    # nn.Linear is a leaf module, so it should appear as call_module
    call_modules = [n for n in nodes if n.op == "call_module"]
    assert len(call_modules) >= 1, (
        f"Expected at least 1 call_module node, got {len(call_modules)}"
    )
    print("PASSED\n")


def test_linear_relu_sequential():
    """Trace Linear -> ReLU sequential model."""
    print("=== Test 2: Linear -> ReLU Sequential ===")

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )
    model.eval()

    tracer = Tracer(model, leaf_modules=LEAF_MODULES)
    with torch.no_grad():
        with tracer:
            x = torch.randn(1, 4)
            out = model(x)

    graph = tracer.graph
    print("\nGraph nodes:")
    for node in graph.nodes:
        print(f"  op={node.op:15s} name={node.name:20s} target={node.target}")

    nodes = list(graph.nodes)
    call_modules = [n for n in nodes if n.op == "call_module"]

    # Should have: Linear(0), ReLU(1), Linear(2) = 3 call_module nodes
    assert len(call_modules) == 3, (
        f"Expected 3 call_module nodes, got {len(call_modules)}: "
        f"{[n.target for n in call_modules]}"
    )
    print("PASSED\n")


def test_multi_input_model():
    """Trace a model with multiple inputs and tensor operations."""
    print("=== Test 3: Multi-input with tensor ops ===")

    class TwoInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)

        def forward(self, x, y):
            a = self.fc1(x)
            b = self.fc2(y)
            return a + b  # element-wise add (traced as torch function)

    model = TwoInputModel()
    model.eval()

    tracer = Tracer(model, leaf_modules=LEAF_MODULES)
    with torch.no_grad():
        with tracer:
            x = torch.randn(1, 4)
            y = torch.randn(1, 4)
            out = model(x, y)

    graph = tracer.graph
    print("\nGraph nodes:")
    for node in graph.nodes:
        print(f"  op={node.op:15s} name={node.name:20s} target={node.target}")

    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    call_modules = [n for n in nodes if n.op == "call_module"]

    assert len(placeholders) == 2, (
        f"Expected 2 placeholders, got {len(placeholders)}"
    )
    assert len(call_modules) == 2, (
        f"Expected 2 call_module (fc1, fc2), got {len(call_modules)}"
    )

    # The add operation should appear as call_function or call_method
    add_nodes = [
        n
        for n in nodes
        if n.op in ("call_function", "call_method")
        and "add" in str(n.target)
    ]
    assert len(add_nodes) >= 1, "Missing add operation node"
    print("PASSED\n")


def test_non_leaf_module():
    """Trace a model where an inner module is NOT a leaf - its ops should be traced."""
    print("=== Test 4: Non-leaf module (inner ops traced) ===")

    class CustomNorm(nn.Module):
        """A custom normalization that is NOT in LEAF_MODULES."""

        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = 1e-6

        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            x_norm = x * torch.rsqrt(variance + self.eps)
            return self.weight * x_norm

    class ModelWithCustomNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.norm = CustomNorm(4)

        def forward(self, x):
            return self.norm(self.fc(x))

    model = ModelWithCustomNorm()
    model.eval()

    tracer = Tracer(model, leaf_modules=LEAF_MODULES)
    with torch.no_grad():
        with tracer:
            x = torch.randn(1, 4)
            out = model(x)

    graph = tracer.graph
    print("\nGraph nodes:")
    for node in graph.nodes:
        meta_scope = node.meta.get("scope", "")
        print(
            f"  op={node.op:15s} name={node.name:30s} "
            f"target={str(node.target):30s} scope={meta_scope}"
        )

    nodes = list(graph.nodes)
    # fc should be call_module (it's a leaf)
    call_modules = [n for n in nodes if n.op == "call_module"]
    assert any("fc" in str(n.target) for n in call_modules), (
        "fc should appear as call_module"
    )

    # CustomNorm is NOT a leaf, so its internal ops (pow, mean, rsqrt, mul)
    # should appear as call_function or call_method nodes
    func_method_nodes = [
        n for n in nodes if n.op in ("call_function", "call_method")
    ]
    assert len(func_method_nodes) >= 3, (
        f"Expected internal ops from CustomNorm, got {len(func_method_nodes)}"
    )
    print("PASSED\n")


def test_embedding_model():
    """Trace a model with nn.Embedding (important for LLM conversion)."""
    print("=== Test 5: Embedding model ===")

    class EmbeddingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 16)
            self.fc = nn.Linear(16, 8)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            return self.fc(x)

    model = EmbeddingModel()
    model.eval()

    tracer = Tracer(model, leaf_modules=LEAF_MODULES)
    with torch.no_grad():
        with tracer:
            ids = torch.tensor([[1, 2, 3, 4]])
            out = model(ids)

    graph = tracer.graph
    print("\nGraph nodes:")
    for node in graph.nodes:
        print(f"  op={node.op:15s} name={node.name:20s} target={node.target}")

    nodes = list(graph.nodes)
    call_modules = [n for n in nodes if n.op == "call_module"]
    targets = [str(n.target) for n in call_modules]

    assert "embed" in targets, f"Missing embed call_module, got {targets}"
    assert "fc" in targets, f"Missing fc call_module, got {targets}"
    print("PASSED\n")


if __name__ == "__main__":
    test_single_linear()
    test_linear_relu_sequential()
    test_multi_input_model()
    test_non_leaf_module()
    test_embedding_model()
    print("=" * 50)
    print("All Task 1.1 tests PASSED!")

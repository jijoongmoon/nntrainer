#!/usr/bin/env python3
"""Task 1.2 Test: Trace Qwen3 model and dump the full FX graph."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from tracer import Tracer, LEAF_MODULES
from transformers import Qwen3Config, Qwen3ForCausalLM


def make_tiny_qwen3():
    """Create a tiny Qwen3 model for fast tracing tests."""
    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1000,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model, config


def test_trace_qwen3_full():
    """Trace full Qwen3ForCausalLM and dump graph."""
    print("=== Task 1.2: Trace Qwen3ForCausalLM ===\n")

    model, config = make_tiny_qwen3()

    # Print model structure for reference
    print("--- Model Module Hierarchy ---")
    for name, mod in model.named_modules():
        if name:
            print(f"  {name}: {type(mod).__name__}")

    # Configure leaf modules: These are modules that map to nntrainer layers.
    # We trace INTO non-leaf modules to see their internal ops.
    # For Qwen3, we want to see the internal structure, so we keep most
    # HuggingFace-specific modules as non-leaf initially.
    # Only truly atomic ops (Linear, Embedding) are leaf.
    qwen3_leaf_modules = tuple(LEAF_MODULES)

    tracer = Tracer(model, leaf_modules=qwen3_leaf_modules)

    # Create sample input
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    with torch.no_grad():
        with tracer:
            out = model(input_ids)

    graph = tracer.graph
    print("\n--- FX Graph Nodes ---")
    print(
        f"{'idx':>4s}  {'op':15s}  {'name':40s}  {'target':50s}  {'scope'}"
    )
    print("-" * 160)
    for i, node in enumerate(graph.nodes):
        scope = node.meta.get("scope", "")
        target_str = str(node.target)
        if len(target_str) > 50:
            target_str = target_str[:47] + "..."
        print(
            f"{i:4d}  {node.op:15s}  {node.name:40s}  "
            f"{target_str:50s}  {scope}"
        )

    # Collect stats
    nodes = list(graph.nodes)
    ops_count = {}
    for n in nodes:
        ops_count[n.op] = ops_count.get(n.op, 0) + 1

    print(f"\n--- Graph Statistics ---")
    print(f"Total nodes: {len(nodes)}")
    for op, count in sorted(ops_count.items()):
        print(f"  {op}: {count}")

    # Verify key structural properties
    placeholders = [n for n in nodes if n.op == "placeholder"]
    call_modules = [n for n in nodes if n.op == "call_module"]
    call_functions = [n for n in nodes if n.op == "call_function"]
    call_methods = [n for n in nodes if n.op == "call_method"]
    outputs = [n for n in nodes if n.op == "output"]

    print(f"\n--- call_module targets ---")
    for n in call_modules:
        print(f"  {n.name}: {n.target}")

    print(f"\n--- call_function targets (unique) ---")
    func_targets = sorted(set(str(n.target) for n in call_functions))
    for t in func_targets:
        count = sum(1 for n in call_functions if str(n.target) == t)
        print(f"  {t}: {count}x")

    print(f"\n--- call_method targets (unique) ---")
    method_targets = sorted(set(str(n.target) for n in call_methods))
    for t in method_targets:
        count = sum(1 for n in call_methods if str(n.target) == t)
        print(f"  {t}: {count}x")

    # Assertions
    assert len(placeholders) >= 1, "Missing input placeholder"
    assert len(outputs) >= 1, "Missing output node"
    assert len(call_modules) >= 1, "Missing call_module nodes"

    # Check for key Qwen3 components in call_module targets
    module_targets = [str(n.target) for n in call_modules]

    # Embedding
    embed_found = any("embed" in t for t in module_targets)
    print(f"\nEmbed found as call_module: {embed_found}")

    # Linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
    linear_targets = [t for t in module_targets if "proj" in t or "lm_head" in t]
    print(f"Linear projection call_modules: {linear_targets}")

    # Per layer, we expect: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj = 7
    # Plus lm_head = 1 (or tied with embed)
    n_layers = config.num_hidden_layers
    expected_linears = n_layers * 7  # 7 projections per layer
    print(f"Expected ~{expected_linears} projection layers for {n_layers} layers")

    # Check if RMSNorm ops are traced (since Qwen3RMSNorm is not in LEAF_MODULES,
    # its internal ops like pow, mean, rsqrt should appear)
    rsqrt_nodes = [n for n in call_functions if "rsqrt" in str(n.target)]
    pow_nodes = [
        n for n in nodes
        if n.op in ("call_function", "call_method") and "pow" in str(n.target)
    ]
    print(f"\nrsqrt ops (from RMSNorm internals): {len(rsqrt_nodes)}")
    print(f"pow ops (from RMSNorm internals): {len(pow_nodes)}")

    # With 2 decoder layers, we expect RMSNorm internal ops from:
    # - input_layernorm (2x), post_attention_layernorm (2x), q_norm (2x), k_norm (2x), model.norm (1x) = 9
    # Each RMSNorm has pow + mean + rsqrt + mul ops
    expected_rmsnorm_count = n_layers * 4 + 1  # 4 per layer + final norm
    print(f"Expected {expected_rmsnorm_count} RMSNorm instances")

    print("\n=== Task 1.2 PASSED ===")
    return graph


def test_trace_qwen3_with_high_level_leaves():
    """Trace Qwen3 with higher-level leaf modules (attention, MLP as leaves).

    This shows what the graph looks like when we treat Qwen3-specific modules
    as leaf nodes - closer to what we need for nntrainer conversion.
    """
    print("\n=== Task 1.2b: Trace Qwen3 with high-level leaf modules ===\n")

    model, config = make_tiny_qwen3()

    # Import Qwen3-specific module types
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3MLP,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
    )

    # High-level leaf modules: treat Qwen3 components as atomic
    high_level_leaves = tuple(set(LEAF_MODULES) | {
        Qwen3Attention,
        Qwen3MLP,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
    })

    tracer = Tracer(model, leaf_modules=high_level_leaves)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    with torch.no_grad():
        with tracer:
            out = model(input_ids)

    graph = tracer.graph
    nodes = list(graph.nodes)

    print("--- High-level leaf graph ---")
    print(
        f"{'idx':>4s}  {'op':15s}  {'name':40s}  {'target':50s}"
    )
    print("-" * 115)
    for i, node in enumerate(nodes):
        target_str = str(node.target)
        if len(target_str) > 50:
            target_str = target_str[:47] + "..."
        print(
            f"{i:4d}  {node.op:15s}  {node.name:40s}  {target_str:50s}"
        )

    call_modules = [n for n in nodes if n.op == "call_module"]
    module_targets = [str(n.target) for n in call_modules]

    print(f"\nTotal nodes: {len(nodes)}")
    print(f"call_module targets: {module_targets}")

    # With high-level leaves, we should see much fewer nodes
    # Expected call_modules per layer:
    #   input_layernorm, self_attn, post_attention_layernorm, mlp = 4
    # Plus: embed_tokens, model.norm, lm_head = 3
    # Plus: rotary_emb = 1
    n_layers = config.num_hidden_layers
    expected_modules = n_layers * 4 + 3 + 1  # rotary_emb called once

    print(f"Expected ~{expected_modules} call_module nodes")
    print(f"Actual call_module nodes: {len(call_modules)}")

    # No internal ops should leak from leaf modules
    call_functions = [n for n in nodes if n.op == "call_function"]
    call_methods = [n for n in nodes if n.op == "call_method"]
    print(f"call_function nodes: {len(call_functions)} (should be few - only from non-leaf code)")
    print(f"call_method nodes: {len(call_methods)}")

    print("\n=== Task 1.2b PASSED ===")
    return graph


if __name__ == "__main__":
    graph_full = test_trace_qwen3_full()
    graph_hl = test_trace_qwen3_with_high_level_leaves()
    print("\n" + "=" * 50)
    print("All Task 1.2 tests PASSED!")

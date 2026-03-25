"""
Fused op pattern definitions and matching.

Each fused NNTrainer op (mha_core, swiglu, etc.) defines the expected
op sequence it replaces. When the actual FX trace matches the pattern,
the fused op can be used safely. When it doesn't match, the converter
keeps decomposed ops and reports what mismatched.

This is the core mechanism for ensuring correctness: only use fused ops
when the mathematical computation is identical.
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple


@dataclass
class FusedOpPattern:
    """Definition of what a fused NNTrainer op expects."""
    name: str                    # e.g. "mha_core"
    description: str             # human-readable description
    # Op types that must be present in the subgraph
    required_ops: Set[str] = field(default_factory=set)
    # Op types that may be present (handled internally by the fused op)
    optional_ops: Set[str] = field(default_factory=set)
    # Op types that must NOT be present (would be silently dropped)
    forbidden_ops: Set[str] = field(default_factory=set)


@dataclass
class PatternMatchResult:
    """Result of matching an actual op sequence against a fused op pattern."""
    matched: bool
    fused_op: str                # e.g. "mha_core"
    missing_ops: List[str] = field(default_factory=list)     # required but not found
    extra_ops: List[str] = field(default_factory=list)       # found but not in pattern
    forbidden_found: List[str] = field(default_factory=list) # forbidden ops found
    reason: str = ""             # human-readable mismatch reason


# =============================================================================
# Pattern definitions
# =============================================================================

# mha_core handles:
#   Q/K/V projections (FC) → optional Q/K norm → optional RoPE →
#   scaled dot-product attention → output projection
# Internally: RoPE via rope_theta, causal mask, GQA expansion
MHA_CORE_ATTENTION = FusedOpPattern(
    name="mha_core",
    description="Multi-head attention with RoPE, GQA, causal mask",
    required_ops={
        "fully_connected",   # Q, K, V, O projections
        "reshape",           # head splitting
        "transpose",         # batch/head/seq rearrangement
        "sdpa",              # scaled dot-product attention
    },
    optional_ops={
        "rms_norm",          # Q/K normalization (Qwen3-style)
        # RoPE computation chain ops (handled internally via rope_theta)
        "multiply",          # RoPE cos/sin multiplication
        "slice",             # RoPE half-rotation slicing
        "negative",          # RoPE rotation negation
        "concat",            # RoPE recombination
        "addition",          # RoPE cos + sin addition
    },
    forbidden_ops={
        # These indicate the model's attention differs from mha_core
        # If any are found in the attention subgraph, mha_core cannot be used
    },
)

# SwiGLU custom layer handles:
#   gate_proj(x) → silu(gate) → gate * up_proj(x)
# Replaces: two FC layers + silu activation + element-wise multiply
SWIGLU_FFN = FusedOpPattern(
    name="swiglu",
    description="SwiGLU gated FFN (gate + up → silu → multiply → down)",
    required_ops={
        "fully_connected",   # gate_proj, up_proj, down_proj
    },
    optional_ops={
        "activation",        # silu/swish (handled by swiglu layer)
        "multiply",          # gate * up (handled by swiglu layer)
        "swiglu",            # already-decomposed fused swiglu layer
    },
    forbidden_ops=set(),
)


# =============================================================================
# Pattern matching
# =============================================================================

def _extract_attention_ops(layers, block_idx=0):
    """Extract the op type sequence for attention in a given block.

    Returns set of unique op types found in the attention subgraph.
    """
    ops = set()
    scope_prefix = f"layers.{block_idx}.self_attn"
    # Collect ops by hf_module_name scope
    attn_layer_names = set()
    for l in layers:
        if scope_prefix in l.hf_module_name:
            ops.add(l.layer_type)
            attn_layer_names.add(l.name)

    # Also collect ops that reference attention layers but have empty hf_module_name
    # (e.g., RoPE intermediate ops like multiply, slice, negative)
    for l in layers:
        if not l.hf_module_name and l.input_layers:
            if any(inp in attn_layer_names for inp in l.input_layers):
                ops.add(l.layer_type)
                attn_layer_names.add(l.name)

    return ops


def _extract_residual_ops(layers, block_idx=0):
    """Extract residual connection ops for a block.

    Returns list of op types in the residual path (between attention
    output and FFN input, and between FFN output and next block).
    """
    ops = []
    scope_prefix = f"layers.{block_idx}"
    for l in layers:
        # Residual ops typically have empty hf_module_name or are at block level
        in_block = (l.name and f"layers_{block_idx}_" in l.name and
                    "self_attn" not in l.name and
                    "mlp" not in l.name and "feed" not in l.name and
                    "norm" not in l.name and "shared_mlp" not in l.name)
        if in_block and l.layer_type in ("addition", "multiply"):
            ops.append(l.layer_type)
    return ops


def _extract_ffn_ops(layers, block_idx=0):
    """Extract FFN op types for a given block."""
    ops = set()
    for l in layers:
        hf = l.hf_module_name
        if (f"layers.{block_idx}.mlp" in hf or
            f"layers.{block_idx}.feed_forward" in hf or
            f"layers.{block_idx}.shared_mlp" in hf):
            ops.add(l.layer_type)
    return ops


def match_attention_pattern(layers, block_idx=0):
    """Check if the attention subgraph matches mha_core's expected pattern.

    Returns PatternMatchResult with detailed mismatch info.
    """
    pattern = MHA_CORE_ATTENTION
    actual_ops = _extract_attention_ops(layers, block_idx)
    residual_ops = _extract_residual_ops(layers, block_idx)

    result = PatternMatchResult(matched=True, fused_op=pattern.name)

    # Check required ops
    for req in pattern.required_ops:
        if req not in actual_ops:
            result.missing_ops.append(req)

    # Check forbidden ops
    for forbid in pattern.forbidden_ops:
        if forbid in actual_ops:
            result.forbidden_found.append(forbid)

    # Check for extra ops not in required or optional
    all_known = pattern.required_ops | pattern.optional_ops
    for op in actual_ops:
        if op not in all_known:
            result.extra_ops.append(op)

    # Check residual ops — mha_core expects simple residual (addition only)
    # Extra multiply in residual path indicates scaling (Granite-style)
    residual_multiplies = residual_ops.count("multiply")
    if residual_multiplies > 1:
        result.extra_ops.append(f"residual_scaling({residual_multiplies}x multiply)")

    # Determine match result
    if result.missing_ops:
        result.matched = False
        result.reason = (f"Missing required ops: {result.missing_ops}")
    elif result.forbidden_found:
        result.matched = False
        result.reason = (f"Forbidden ops found: {result.forbidden_found}")
    elif result.extra_ops:
        result.matched = False
        result.reason = (
            f"Extra ops not handled by {pattern.name}: {result.extra_ops}. "
            f"These would be silently dropped, producing wrong results.")

    return result


def match_ffn_pattern(layers, block_idx=0):
    """Check if the FFN subgraph matches swiglu's expected pattern.

    Returns PatternMatchResult.
    """
    pattern = SWIGLU_FFN
    actual_ops = _extract_ffn_ops(layers, block_idx)

    result = PatternMatchResult(matched=True, fused_op=pattern.name)

    for req in pattern.required_ops:
        if req not in actual_ops:
            result.missing_ops.append(req)

    all_known = pattern.required_ops | pattern.optional_ops
    for op in actual_ops:
        if op not in all_known:
            result.extra_ops.append(op)

    if result.missing_ops:
        result.matched = False
        result.reason = f"Missing required ops: {result.missing_ops}"
    elif result.extra_ops:
        result.matched = False
        result.reason = (
            f"Extra ops not handled by {pattern.name}: {result.extra_ops}")

    return result


def check_fused_op_compatibility(layers):
    """Check all fused op patterns against the actual layer graph.

    Returns dict of {op_name: PatternMatchResult}.
    """
    results = {}

    # Check attention pattern (use block 0 as representative)
    attn_result = match_attention_pattern(layers, block_idx=0)
    results["attention"] = attn_result

    # Check FFN pattern
    ffn_result = match_ffn_pattern(layers, block_idx=0)
    results["swiglu"] = ffn_result

    return results


def print_compatibility_report(results):
    """Print human-readable compatibility report."""
    print(f"\n  [COMPAT] Fused op compatibility check:")
    for op_name, result in results.items():
        if result.matched:
            print(f"    ✓ {result.fused_op}: COMPATIBLE — can safely use fused op")
        else:
            print(f"    ✗ {result.fused_op}: INCOMPATIBLE — using decomposed ops")
            print(f"      Reason: {result.reason}")

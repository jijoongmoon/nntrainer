"""
Adaptive decomposition pipeline for HuggingFace -> NNTrainer conversion.

Strategy:
  1. FUSED OPS FIRST: Trace with LEAF_MODULES, map to NNTrainer fused layers
     (fully_connected, rms_norm, mha_core, swiglu, etc.)
  2. DECOMPOSE UNKNOWNS: If any modules can't be mapped, re-trace with those
     modules excluded from leaves. Their forward() is automatically decomposed
     into tensor ops by the tracer.
  3. OP DECOMPOSITION: For individual ops without direct NNTrainer layer support,
     resolve using:
     a) Direct Tensor methods (inv_sqrt, abs, neg, erf, pow, sqrt, sin, cos, tan)
     b) LazyTensor chains for consecutive arithmetic ops (add, sub, mul, div)
     c) Layer-level decomposition into supported primitives

NNTrainer Tensor & LazyTensor integration:
  - Tensor class provides: pow, sqrt, abs, neg, inv_sqrt, erf, sin, cos, tan
  - LazyTensor (Tensor::chain()) chains: add_i, subtract_i, multiply_i, divide_i,
    dot, transpose, sum, average
  - Ops like rsqrt map directly to Tensor::inv_sqrt() - no decomposition needed
  - Consecutive arithmetic ops can be fused into a single LazyTensor chain
"""

import torch
from typing import Optional

from nntrainer_layers import (
    NNTrainerLayerDef,
    LAYER_INPUT, LAYER_IDENTITY,
    LAYER_POW, LAYER_SQRT, LAYER_MULTIPLY, LAYER_DIVIDE, LAYER_NEGATIVE,
    LAYER_ADDITION, LAYER_SUBTRACT,
    LAYER_DROPOUT, LAYER_EMBEDDING,
    LAYER_RESHAPE, LAYER_PERMUTE, LAYER_TRANSPOSE,
    LAZY_TENSOR_OPS, TENSOR_DIRECT_METHODS,
    OP_UNSUPPORTED, OP_NOOP, OP_RESHAPE, OP_TRANSPOSE, OP_PERMUTE,
    OP_SDPA,
)
from tracer import Tracer, LEAF_MODULES
from node_mapper import NodeMapper
from pattern_detector import detect_patterns


# =============================================================================
# Tensor Method Resolution
# =============================================================================
# Ops that have direct NNTrainer Tensor methods don't need layer-level
# decomposition. Instead, they emit a single Tensor method call in C++.
# This is more efficient than creating intermediate NNTrainer layers.

def _resolve_to_tensor_method(layer):
    """Try to resolve an unsupported op to a direct NNTrainer Tensor method.

    Returns:
        NNTrainerLayerDef with tensor_method metadata if resolvable, else None.
    """
    original_op = layer.properties.get("original_op", "")
    method_info = TENSOR_DIRECT_METHODS.get(original_op)
    if not method_info:
        return None

    method_name, is_inplace = method_info
    # Create a layer that carries the tensor method info for the emitter
    resolved = NNTrainerLayerDef(
        layer_type=f"tensor_op:{original_op}",
        name=layer.name,
        properties={
            "tensor_method": method_name,
            "is_inplace": is_inplace,
            "original_op": original_op,
        },
        input_layers=layer.input_layers,
        hf_module_name=layer.hf_module_name,
    )
    return resolved


# =============================================================================
# Layer-level Decomposition Registry (fallback)
# =============================================================================
# For ops that are neither NNTrainer layers nor direct Tensor methods,
# decompose into sequences of supported layers.

def _decompose_reciprocal(layer):
    """reciprocal(x) -> divide(1, x)"""
    return [NNTrainerLayerDef(
        layer_type=LAYER_DIVIDE,
        name=layer.name,
        properties={"numerator": 1.0},
        input_layers=layer.input_layers,
        hf_module_name=layer.hf_module_name,
    )]


# Registry: original_op -> decomposition function (layer-level fallback)
DECOMPOSITION_REGISTRY = {
    "reciprocal": _decompose_reciprocal,
}

# Ops with no decomposition and no Tensor method - truly unsupported
_NO_DECOMPOSITION_OPS = {"exp", "log", "clamp", "clip", "clamp_", "clamp_min", "clamp_max"}


# =============================================================================
# LazyTensor Chain Detection
# =============================================================================

class LazyTensorChain:
    """Represents a sequence of ops that can be fused into a LazyTensor chain.

    In NNTrainer C++, this becomes:
        Tensor result = input.chain()
            .add_i(tensor_a)
            .multiply_i(scalar)
            .subtract_i(tensor_b)
            .run();

    Instead of creating separate NNTrainer layers for each op.
    """
    def __init__(self, layers, start_idx):
        self.layers = layers    # List of NNTrainerLayerDef in the chain
        self.start_idx = start_idx
        self.end_idx = start_idx + len(layers)

    @property
    def chain_length(self):
        return len(self.layers)

    def to_cpp_chain(self, input_var="input"):
        """Generate C++ LazyTensor chain code.

        Returns a string like:
            input.chain().add_i(a).multiply_i(2.0f).sqrt_i().run()
        """
        # Map layer_type -> LazyTensor method for binary ops (with tensor/scalar)
        _BINARY_OPS = {
            "add": "add_i", "addition": "add_i",
            "subtract": "subtract_i",
            "multiply": "multiply_i",
            "divide": "divide_i",
        }
        # Map layer_type -> LazyTensor method for unary ops (no args)
        _UNARY_OPS = {
            "sqrt": "sqrt_i",
            "negative": "neg",
            "exp": "exp_i",
            "log": "log_i",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
        }

        parts = [f"{input_var}.chain()"]
        for layer in self.layers:
            lt = layer.layer_type
            if lt in _BINARY_OPS:
                method = _BINARY_OPS[lt]
                if layer.input_layers:
                    parts.append(f"{method}({layer.input_layers[-1]})")
                else:
                    val = layer.properties.get("value", "0")
                    parts.append(f"{method}({val}f)")
            elif lt == "pow":
                exp = layer.properties.get("exponent", "2.0")
                parts.append(f"pow_i({exp}f)")
            elif lt == "clamp":
                mn = layer.properties.get("min", "-inf")
                mx = layer.properties.get("max", "inf")
                parts.append(f"clamp_i({mn}f, {mx}f)")
            elif lt in _UNARY_OPS:
                parts.append(f"{_UNARY_OPS[lt]}()")
        parts.append("run()")
        return ".".join(parts)

    def __repr__(self):
        ops = [l.layer_type for l in self.layers]
        return f"LazyTensorChain({' -> '.join(ops)})"


def detect_lazy_chains(layers, min_chain_length=2):
    """Find sequences of consecutive LazyTensor-compatible ops.

    Args:
        layers: List of NNTrainerLayerDef.
        min_chain_length: Minimum ops to form a chain (default 2).

    Returns:
        List of LazyTensorChain objects found in the layer sequence.
    """
    chains = []
    current_chain = []
    chain_start = 0

    for i, layer in enumerate(layers):
        if layer.layer_type in LAZY_TENSOR_OPS:
            if not current_chain:
                chain_start = i
            current_chain.append(layer)
        else:
            if len(current_chain) >= min_chain_length:
                chains.append(LazyTensorChain(current_chain, chain_start))
            current_chain = []

    # Handle chain at end of list
    if len(current_chain) >= min_chain_length:
        chains.append(LazyTensorChain(current_chain, chain_start))

    return chains


# =============================================================================
# Op Resolution Pipeline
# =============================================================================

def resolve_unsupported_ops(layers):
    """Resolve unsupported ops using the best available strategy.

    Resolution order (most efficient first):
      1. Direct Tensor method (inv_sqrt, abs, neg, erf, pow, sqrt, sin, cos, tan)
      2. Layer-level decomposition (reciprocal -> divide)
      3. Keep as unsupported with warning

    Args:
        layers: List of NNTrainerLayerDef from NodeMapper.

    Returns:
        New list with unsupported ops resolved.
    """
    result = []
    tensor_method_count = 0
    decomposed_count = 0
    unresolvable = []

    for layer in layers:
        if layer.layer_type != OP_UNSUPPORTED:
            result.append(layer)
            continue

        original_op = layer.properties.get("original_op", "")

        # Strategy 1: Direct Tensor method
        resolved = _resolve_to_tensor_method(layer)
        if resolved:
            result.append(resolved)
            tensor_method_count += 1
            continue

        # Strategy 2: Layer-level decomposition
        decompose_fn = DECOMPOSITION_REGISTRY.get(original_op)
        if decompose_fn:
            decomposed = decompose_fn(layer)
            result.extend(decomposed)
            decomposed_count += len(decomposed)
            continue

        # Strategy 3: No resolution available
        unresolvable.append(layer)
        result.append(layer)

    if tensor_method_count > 0:
        print(f"  [RESOLVE] {tensor_method_count} ops -> direct Tensor methods "
              f"(inv_sqrt, abs, etc.)")
    if decomposed_count > 0:
        print(f"  [RESOLVE] {decomposed_count} ops -> layer decomposition")
    if unresolvable:
        print(f"  [WARNING] {len(unresolvable)} ops have no resolution:")
        for layer in unresolvable:
            op = layer.properties.get("original_op", "?")
            print(f"    - {op} at {layer.hf_module_name} ({layer.name})")

    return result


# Keep backward compatibility
decompose_unsupported_ops = resolve_unsupported_ops


def _propagate_noop_forward(layers):
    """Propagate NOOP status through layers whose ALL inputs are NOOP.

    In models like T5/mT5, position computation chains (arange → __getitem__
    → add → sub → ...) produce NOOP-derived values.  The initial mapping only
    marks the root ops (arange, zeros_like, etc.) as NOOP.  Downstream ops
    like slice, addition, subtract remain as their original types, which
    causes them to survive NOOP removal and become disconnected nodes.

    This pass iteratively converts such layers to NOOP so they are cleanly
    removed in the subsequent NOOP removal pass.
    """
    # Layer types safe to convert to NOOP when all inputs are NOOP.
    # These are pure arithmetic/indexing ops that cannot produce meaningful
    # outputs from position-only inputs.  Module layers with learned weights
    # (embedding, fully_connected, norm) are excluded.
    _NOOP_PROPAGATABLE = frozenset({
        "slice", LAYER_ADDITION, LAYER_SUBTRACT, LAYER_MULTIPLY,
        LAYER_DIVIDE, LAYER_NEGATIVE, LAYER_POW, LAYER_SQRT,
        "log", "concat",
        OP_RESHAPE, OP_TRANSPOSE, OP_PERMUTE,
        LAYER_RESHAPE, LAYER_PERMUTE, LAYER_TRANSPOSE,
    })

    def _is_propagatable(layer_type):
        if layer_type in _NOOP_PROPAGATABLE:
            return True
        # tensor_op:abs, tensor_op:neg, etc. — resolved direct Tensor methods
        if layer_type.startswith("tensor_op:"):
            return True
        return False

    noop_names = {l.name for l in layers if l.layer_type == OP_NOOP}
    total_converted = 0
    changed = True
    while changed:
        changed = False
        for l in layers:
            if l.name in noop_names:
                continue
            if not _is_propagatable(l.layer_type):
                continue
            if l.input_layers and all(
                inp in noop_names for inp in l.input_layers
            ):
                l.layer_type = OP_NOOP
                noop_names.add(l.name)
                total_converted += 1
                changed = True

    if total_converted:
        print(f"  [CLEANUP] Propagated NOOP to {total_converted} "
              f"position-derived layers")
    return layers


def _remove_passthrough_layers(layers, layer_type, label):
    """Remove layers of a given type and rewire downstream inputs.

    For each removed layer, downstream layers that referenced it are
    rewired to point at the removed layer's own input instead.
    Handles chains (e.g. noop -> noop -> real layer).

    Returns the filtered layer list.
    """
    target_names = {l.name for l in layers if l.layer_type == layer_type}
    if not target_names:
        return layers

    # Build bypass map: removed_name -> its input (or None)
    bypass = {}
    for l in layers:
        if l.layer_type == layer_type:
            bypass[l.name] = l.input_layers[0] if l.input_layers else None

    # Resolve chains
    def _resolve(name):
        visited = set()
        while name in bypass and name not in visited:
            visited.add(name)
            name = bypass[name]
        return name

    # Rewire inputs of surviving layers
    for l in layers:
        if l.layer_type != layer_type and l.input_layers:
            l.input_layers = [
                _resolve(inp) if inp in target_names else inp
                for inp in l.input_layers
            ]
            l.input_layers = [x for x in l.input_layers if x]

    filtered = [l for l in layers if l.layer_type != layer_type]
    print(f"  [CLEANUP] Removed {len(target_names)} {label} layers")
    return filtered


# =============================================================================
# Dead Layer Removal
# =============================================================================

# Layer types safe to remove when they have no consumers.  These are
# "infrastructure" ops generated by tuple unpacking, slicing, etc.
# Computational layers (FC, activation, norm, ...) are never removed
# because they may be model outputs in multi-head architectures.
_DEAD_REMOVABLE_TYPES = frozenset({
    LAYER_IDENTITY,
    "slice",        # Unused tuple-unpacking slices
    OP_NOOP,
    OP_RESHAPE,
    LAYER_RESHAPE,
    OP_TRANSPOSE,
    LAYER_TRANSPOSE,
    OP_PERMUTE,
    LAYER_PERMUTE,
})


def _remove_dead_layers(layers):
    """Remove infrastructure layers that have no consumers.

    Only removes "safe" layer types (identity, slice, reshape, etc.)
    that are clearly dead code.  Computational layers (FC, norm, activation)
    are preserved since they may be legitimate multi-head outputs.

    Iterates to handle chains of dead infrastructure layers.

    Returns the filtered layer list.
    """
    if not layers:
        return layers

    total_removed = 0
    changed = True
    while changed:
        changed = False
        consumed = set()
        for l in layers:
            for inp in (l.input_layers or []):
                consumed.add(inp)

        dead = []
        for l in layers:
            if (l.name not in consumed
                    and l.layer_type in _DEAD_REMOVABLE_TYPES):
                dead.append(l.name)

        if dead:
            dead_set = set(dead)
            layers = [l for l in layers if l.name not in dead_set]
            total_removed += len(dead)
            changed = True

    if total_removed:
        print(f"  [CLEANUP] Removed {total_removed} dead layers")
    return layers


# =============================================================================
# Position ID Chain Removal
# =============================================================================

# Layer types that are part of position ID computation chains.
# These are arithmetic/reshape ops used to compute position indices
# from attention masks before feeding into position embedding.
_POSITION_CHAIN_OPS = frozenset({
    LAYER_ADDITION, LAYER_SUBTRACT, LAYER_MULTIPLY, LAYER_DIVIDE,
    OP_RESHAPE,
})


def _remove_position_id_chains(layers):
    """Remove position ID computation chains (XLM-RoBERTa, etc.).

    In models like XLM-RoBERTa, position IDs are computed from input masks:
        input_ids → ne → int → cumsum → type_as → add → mul → add → Embedding

    After noop removal (ne, int, cumsum, type_as are all noops), the remaining
    arithmetic layers (add, mul, add) only serve to compute position indices.
    NNTrainer handles position IDs internally, so these are redundant.

    This function detects arithmetic layers that exclusively feed into
    embedding layers (directly or through other arithmetic layers) and
    removes them. It uses iterative fixed-point analysis to correctly
    handle chains of arbitrary length.

    Returns the filtered layer list.
    """
    by_name = {l.name: l for l in layers}

    # Build consumer graph: layer_name -> set of consumer layer names
    consumers = {}
    for l in layers:
        for inp in (l.input_layers or []):
            consumers.setdefault(inp, set()).add(l.name)

    embedding_names = {l.name for l in layers if l.layer_type == LAYER_EMBEDDING}

    # Iterative fixed-point: mark arithmetic layers whose ALL consumers
    # are either embedding layers or already-marked removable layers.
    removable = set()
    changed = True
    while changed:
        changed = False
        for l in layers:
            if l.name in removable or l.layer_type not in _POSITION_CHAIN_OPS:
                continue
            layer_consumers = consumers.get(l.name, set())
            if not layer_consumers:
                continue
            if all(c in embedding_names or c in removable
                   for c in layer_consumers):
                removable.add(l.name)
                changed = True

    if not removable:
        return layers

    # Rewire embedding inputs past the removed chain
    for l in layers:
        if l.name not in removable and l.input_layers:
            l.input_layers = [
                inp for inp in l.input_layers if inp not in removable
            ]

    filtered = [l for l in layers if l.name not in removable]
    print(f"  [CLEANUP] Removed {len(removable)} position ID "
          f"computation layers")
    return filtered


# =============================================================================
# RoPE Chain Collapse (llama.cpp-style single op)
# =============================================================================
# RoPE is handled internally by mha_core (NEON/AVX2 optimized), so the
# decomposed sin/cos/rotate_half ops from torch.fx are redundant.
#
# Computation chain (shared): inv_freq → matmul → cat → cos/sin
# Application chain (per layer): Q*cos + rotate_half(Q)*sin
#
# This pass removes both chains and relies on mha_core's rope_theta parameter.

# Layer types that appear in RoPE application chains (rotate_half + apply).
_ROPE_APP_TYPES = frozenset({
    LAYER_MULTIPLY, LAYER_ADDITION, LAYER_NEGATIVE, LAYER_SUBTRACT,
    "slice", "concat",
    LAYER_RESHAPE, LAYER_PERMUTE, LAYER_TRANSPOSE,
    OP_RESHAPE, OP_TRANSPOSE, OP_PERMUTE,
})


def _is_rope_computation(layer):
    """Check if a layer is part of the RoPE frequency computation chain.

    Matches layers whose name or HF scope contains 'rotary_emb' or 'rope'.
    These layers compute cos/sin embeddings from inv_freq and positions.
    """
    name = layer.name.lower()
    scope = layer.hf_module_name.lower()
    return ("rotary_emb" in name or "rotary_emb" in scope
            or "rotary_embedding" in name or "rotary_embedding" in scope
            or ("rope" in name and "rope" not in "properties"))


def _is_rope_app_type(layer_type):
    """Check if a layer type can appear in a RoPE application chain."""
    return layer_type in _ROPE_APP_TYPES or layer_type.startswith("tensor_op:")


def _find_pre_rope_input(layer_name, by_name, removable):
    """Trace backward through RoPE chain to find the pre-rotation input.

    For Q*cos + rotate_half(Q)*sin, returns Q (the tensor before rotation).
    Walks through removable layers, prioritizing paths through multiply ops
    that have a non-removable input (the original Q/K tensor).
    """
    visited = set()

    def walk(name):
        if name in visited:
            return None
        visited.add(name)
        layer = by_name.get(name)
        if not layer:
            return None

        rope_inputs = []
        non_rope_inputs = []
        for inp in (layer.input_layers or []):
            if inp in removable:
                rope_inputs.append(inp)
            else:
                non_rope_inputs.append(inp)

        # multiply(Q, cos): Q is non-rope, cos is rope → return Q
        if layer.layer_type in (LAYER_MULTIPLY, "multiply") and non_rope_inputs:
            return non_rope_inputs[0]

        # For other types (addition, concat): trace through rope inputs first
        for inp in rope_inputs:
            result = walk(inp)
            if result:
                return result

        # Fallback: return first non-rope input
        return non_rope_inputs[0] if non_rope_inputs else None

    return walk(layer_name)


def _decompose_fused_swiglu(layers):
    """Decompose fused gate+up SwiGLU into standard gate/up/swiglu/down.

    Detects the pattern: FC(2N) → split → silu → multiply → FC(out)
    and replaces it with: gate_proj(N) → up_proj(N) → swiglu → down_proj

    This converts Granite-style fused shared_mlp into NNTrainer's standard
    SwiGLU which uses separate gate/up FC layers + SwiGLU custom layer.
    """
    from nntrainer_layers import LAYER_FC, LAYER_ACTIVATION, NNTrainerLayerDef

    by_name = {l.name: l for l in layers}
    consumers = {}
    for l in layers:
        for inp in l.input_layers:
            consumers.setdefault(inp, []).append(l.name)

    replacements = []  # list of (fused_fc_name, split_name, act_name, mul_name)

    for layer in layers:
        if layer.layer_type != "split":
            continue
        # split's input should be a FC layer
        if not layer.input_layers:
            continue
        fc_name = layer.input_layers[0]
        fc = by_name.get(fc_name)
        if not fc or fc.layer_type != LAYER_FC:
            continue
        fc_unit = int(fc.properties.get("unit", 0))
        if fc_unit <= 0 or fc_unit % 2 != 0:
            continue

        # split's consumers should include activation and multiply
        split_consumers = consumers.get(layer.name, [])
        act_name = None
        mul_name = None
        for c_name in split_consumers:
            c = by_name.get(c_name)
            if not c:
                continue
            if c.layer_type == LAYER_ACTIVATION:
                act_name = c_name
            elif c.layer_type == LAYER_MULTIPLY:
                mul_name = c_name

        if not act_name or not mul_name:
            # Also check: activation → multiply chain
            if act_name and not mul_name:
                act_consumers = consumers.get(act_name, [])
                for c_name in act_consumers:
                    c = by_name.get(c_name)
                    if c and c.layer_type == LAYER_MULTIPLY:
                        mul_name = c_name
                        break

        if not act_name or not mul_name:
            continue

        replacements.append((fc_name, layer.name, act_name, mul_name))

    if not replacements:
        return layers

    remove_names = set()
    new_layers = {}

    for fc_name, split_name, act_name, mul_name in replacements:
        fc = by_name[fc_name]
        mul = by_name[mul_name]
        half_unit = int(fc.properties.get("unit", 0)) // 2

        # Determine the FFN scope for naming
        scope = fc.hf_module_name.rsplit(".", 1)[0] if fc.hf_module_name else ""
        prefix = fc.name.rsplit("_", 1)[0] if "_" in fc.name else fc.name

        # Create gate_proj (first half of fused weight)
        gate = NNTrainerLayerDef(
            layer_type=LAYER_FC, name=prefix + "_gate_proj")
        gate.properties["unit"] = str(half_unit)
        gate.properties["disable_bias"] = "true"
        gate.input_layers = list(fc.input_layers)
        gate.has_weight = True
        gate.transpose_weight = True
        gate.weight_hf_key = fc.weight_hf_key
        gate.weight_split = "first_half"  # custom marker
        # Set hf_module_name so _match_fc_roles recognizes gate/up suffixes
        gate.hf_module_name = scope + ".gate_proj"

        # Create up_proj (second half of fused weight)
        up = NNTrainerLayerDef(
            layer_type=LAYER_FC, name=prefix + "_up_proj")
        up.properties["unit"] = str(half_unit)
        up.properties["disable_bias"] = "true"
        up.input_layers = list(fc.input_layers)
        up.has_weight = True
        up.transpose_weight = True
        up.weight_hf_key = fc.weight_hf_key
        up.weight_split = "second_half"  # custom marker
        up.hf_module_name = scope + ".up_proj"

        # Create swiglu layer
        swiglu = NNTrainerLayerDef(
            layer_type="swiglu", name=prefix + "_swiglu")
        swiglu.input_layers = [gate.name, up.name]
        swiglu.hf_module_name = scope

        # Rewire multiply's consumers to point to swiglu
        for l in layers:
            l.input_layers = [
                swiglu.name if inp == mul_name else inp
                for inp in l.input_layers
            ]

        new_layers[fc_name] = [gate, up, swiglu]
        remove_names.update({fc_name, split_name, act_name, mul_name})

    # Rebuild layer list
    result = []
    for layer in layers:
        if layer.name in remove_names:
            if layer.name in new_layers:
                result.extend(new_layers[layer.name])
            continue
        result.append(layer)

    if replacements:
        print(f"  [FUSED-SWIGLU] Decomposed {len(replacements)} fused "
              f"gate+up SwiGLU patterns into standard SwiGLU")

    return result


def _collapse_rope_chains(layers):
    """Remove RoPE computation and application chains from the layer graph.

    RoPE is handled internally by mha_core using the rope_theta parameter
    with NEON/AVX2 optimized kernels, making the decomposed sin/cos/
    rotate_half ops redundant.

    Uses iterative fixed-point analysis (like _remove_position_id_chains):
    1. Identify computation chain layers by scope (rotary_emb/rope)
    2. Forward-propagate to find application chain layers
    3. Build bypass map to rewire downstream consumers
    4. Remove all RoPE layers

    Returns:
        Tuple of (filtered layers, set of collapsed layer names).
    """
    by_name = {l.name: l for l in layers}

    # Build consumer graph: layer_name -> set of consumer layer names
    consumers = {}
    for l in layers:
        for inp in (l.input_layers or []):
            consumers.setdefault(inp, set()).add(l.name)

    # Phase 1: Identify RoPE computation layers by scope
    rope_comp = set()
    for l in layers:
        if _is_rope_computation(l):
            rope_comp.add(l.name)

    if not rope_comp:
        return layers, set()

    # Phase 2a: Forward-propagate from rope_comp to find all
    # "rope-connected" layers (transitively consume rope outputs).
    rope_connected = set(rope_comp)
    changed = True
    while changed:
        changed = False
        for l in layers:
            if l.name in rope_connected:
                continue
            if not _is_rope_app_type(l.layer_type):
                continue
            if any(inp in rope_connected for inp in (l.input_layers or [])):
                rope_connected.add(l.name)
                changed = True

    # Phase 2b: Backward pressure — mark rope-connected layers as removable
    # if ALL their consumers are also rope-connected or SDPA.
    # Uses rope_connected for reachability, removable for convergence.
    removable = set(rope_comp)
    changed = True
    while changed:
        changed = False
        for name in rope_connected:
            if name in removable:
                continue
            lc = consumers.get(name, set())
            if not lc:
                # No consumers - dead node, safe to remove
                removable.add(name)
                changed = True
                continue
            if all(c in removable
                   or (c in by_name and by_name[c].layer_type == OP_SDPA)
                   for c in lc):
                removable.add(name)
                changed = True

    # Phase 2c: Backward propagation — catch layers that exclusively
    # feed into removable layers (e.g. rotate_half: slice → neg → concat
    # that feed into the mul(rotated, sin) which is already removable).
    changed = True
    while changed:
        changed = False
        for l in layers:
            if l.name in removable:
                continue
            if not _is_rope_app_type(l.layer_type):
                continue
            lc = consumers.get(l.name, set())
            if not lc:
                continue
            if all(c in removable for c in lc):
                removable.add(l.name)
                changed = True

    # Phase 3: Build bypass map for final RoPE outputs
    # For each removable layer consumed by a non-removable layer,
    # find the pre-RoPE input to route to instead.
    bypass = {}
    for name in removable:
        lc = consumers.get(name, set())
        if any(c not in removable for c in lc):
            source = _find_pre_rope_input(name, by_name, removable)
            if source:
                bypass[name] = source

    # Phase 4: Rewire downstream layers
    for l in layers:
        if l.name not in removable and l.input_layers:
            new_inputs = []
            for inp in l.input_layers:
                if inp in bypass:
                    new_inputs.append(bypass[inp])
                elif inp not in removable:
                    new_inputs.append(inp)
                # else: input is removable with no bypass → skip
            l.input_layers = new_inputs

    # Phase 5: Remove and collect FX node names for VS Code bridge
    collapsed_fx_names = set()
    for name in removable:
        layer = by_name.get(name)
        if layer:
            # Prefer fx_node_name (matches FX graph); fall back to layer name
            fx_name = layer.fx_node_name or layer.name
            collapsed_fx_names.add(fx_name)

    filtered = [l for l in layers if l.name not in removable]
    comp_count = len(rope_comp)
    app_count = len(removable) - comp_count
    print(f"  [ROPE] Collapsed {len(removable)} RoPE layers "
          f"({comp_count} computation + {app_count} application)")

    return filtered, collapsed_fx_names


# =============================================================================
# Adaptive Converter Pipeline
# =============================================================================

class AdaptiveConverter:
    """Adaptive converter: fused ops first, tensor op fallback for unknowns.

    This is the main entry point for converting any HuggingFace model to
    NNTrainer layer definitions.

    Args:
        model: HuggingFace model.
        config: HuggingFace model config.
        plugin_registry: Optional custom layer plugin registry.
        fused_ops: Set of op types to use fused (optimized) versions.
                   Default: {"attention", "swiglu"} for verified models,
                   empty set for unverified models (decomposed ops).
                   "attention" -> mha_core + RoPE collapse
                   "swiglu" -> SwiGLU custom layer

    Pipeline:
        Pass 1: Trace with full LEAF_MODULES
                -> Map to NNTrainer layers
                -> Identify unknown module types

        Pass 2 (if needed): Re-trace with unknown modules excluded from leaves
                -> Their forward() automatically decomposes into tensor ops
                -> Map tensor ops to NNTrainer layers

        Pass 3: Resolve unsupported ops:
                -> Tensor methods (rsqrt -> inv_sqrt, abs -> abs)
                -> Layer decomposition (reciprocal -> divide)

        Pass 4: Detect LazyTensor chain opportunities
                -> Consecutive add/sub/mul/div -> single chain() call

    Usage:
        converter = AdaptiveConverter(model, config)
        result = converter.convert({"input_ids": input_ids})
        result.summary()
    """

    def __init__(self, model, model_config=None, training=False,
                 plugin_registry=None):
        """
        Args:
            model: The HuggingFace model to convert.
            model_config: HuggingFace model config (optional).
            training: If False (default), dropout layers are removed from
                the output since they are no-ops during inference. If True,
                dropout layers are preserved for training use.
            plugin_registry: Optional PluginRegistry for custom layer mappings.
                If provided, registered custom module types are treated as
                leaf modules (not decomposed) and mapped via the registry.
        """
        self.model = model
        self.config = model_config
        self.training = training
        self.fused_ops = set()  # default: no fused ops (accurate mode)
        if plugin_registry is not None:
            from plugin_registry import get_global_registry, _global_registry
            # Merge into global registry so module_mapper can find them
            for matcher, spec in plugin_registry._entries:
                _global_registry.register(matcher, spec)

    def convert(self, input_kwargs, max_passes=3):
        """Run the adaptive conversion pipeline.

        Args:
            input_kwargs: Dict of model inputs (e.g. {"input_ids": tensor}).
            max_passes: Maximum re-trace passes for unknown modules.

        Returns:
            ConversionResult with layers, lazy chains, and diagnostics.
        """
        exclude_types = set()
        all_decomposed_types = set()

        # Iterative passes: trace -> map -> find unknowns -> exclude -> re-trace
        for pass_num in range(1, max_passes + 1):
            print(f"\n  [PASS {pass_num}] Tracing with "
                  f"{len(exclude_types)} excluded module types...")

            tracer = Tracer(self.model, exclude_leaf_types=exclude_types)
            with tracer:
                with torch.no_grad():
                    self.model(**input_kwargs)

            mapper = NodeMapper(self.model, tracer.graph, self.config)
            layers = mapper.map_all()

            # Check for unknown module types
            unknown_types = mapper.get_unknown_module_types(layers)

            if not unknown_types:
                print(f"  [PASS {pass_num}] All modules mapped successfully.")
                break

            # Found unknowns - exclude them and re-trace
            new_types = unknown_types - exclude_types
            if not new_types:
                print(f"  [PASS {pass_num}] No new unknown types to decompose.")
                break

            print(f"  [PASS {pass_num}] Decomposing {len(new_types)} unknown "
                  f"module types: {', '.join(sorted(new_types))}")
            exclude_types |= new_types
            all_decomposed_types |= new_types

        # Pass 3: Resolve unsupported ops (Tensor methods > layer decomposition)
        layers = resolve_unsupported_ops(layers)

        # Pass 3.5: Remove dropout layers for inference mode
        if not self.training:
            layers = _remove_passthrough_layers(
                layers, LAYER_DROPOUT, "dropout")

        # Pass 3.55: Remove identity layers (getitem[0] on multi-output
        # modules, etc.) — rewires downstream to point at the parent layer
        layers = _remove_passthrough_layers(
            layers, LAYER_IDENTITY, "identity")

        # Pass 3.58: Propagate NOOP forward through position-derived chains
        # (T5/mT5 relative position bias computation, seq-length arithmetic)
        layers = _propagate_noop_forward(layers)

        # Pass 3.6: Remove noop layers (expand, size, _set_grad_enabled, etc.)
        layers = _remove_passthrough_layers(layers, OP_NOOP, "noop")

        # Pass 3.65: Convert single-input arithmetic ops to identity.
        # After NOOP removal, some additions/subtractions lose their
        # mask/zero operand and become single-input passthrough ops
        # (e.g. attn_scores + causal_mask → just attn_scores).
        _ARITH_TYPES = frozenset({
            LAYER_ADDITION, LAYER_SUBTRACT,
        })
        collapsed = 0
        for layer in layers:
            if (layer.layer_type in _ARITH_TYPES
                    and len(layer.input_layers) <= 1):
                layer.layer_type = LAYER_IDENTITY
                collapsed += 1
        if collapsed:
            layers = _remove_passthrough_layers(
                layers, LAYER_IDENTITY, "identity (post-noop)")

        # Pass 3.7: Remove position ID computation chains
        # (arithmetic ops that exclusively feed position embeddings)
        layers = _remove_position_id_chains(layers)

        # Pass 3.8: Convert intermediate op types to final NNTrainer types
        _OP_TO_LAYER = {
            OP_RESHAPE: LAYER_RESHAPE,
            OP_TRANSPOSE: LAYER_TRANSPOSE,
            OP_PERMUTE: LAYER_PERMUTE,
        }
        for layer in layers:
            if layer.layer_type in _OP_TO_LAYER:
                layer.layer_type = _OP_TO_LAYER[layer.layer_type]

        # Pass 3.9: Add input layers for external inputs and extract
        # reshape/slice parameters from FX graph metadata
        layers = _add_input_layers_and_shape_info(
            layers, tracer.graph, input_kwargs)

        # Pass 3.95: Remove dead layers — layers with no consumers and
        # no side effects (e.g. unused getitem[1] from multi-output
        # modules).  Iterates to handle chains of dead layers.
        layers = _remove_dead_layers(layers)

        # Pass 3.96: Collapse RoPE chains — only when fused attention is enabled,
        # because mha_core handles RoPE internally via rope_theta.
        # In accurate mode (no fused attention), RoPE ops are preserved as
        # individual tensor ops for exact computation.
        collapsed_rope = set()
        if "attention" in self.fused_ops:
            layers, collapsed_rope = _collapse_rope_chains(layers)
            if collapsed_rope:
                layers = _remove_dead_layers(layers)
        else:
            print("  [MODE] Accurate mode: RoPE preserved as individual ops "
                  "(mha_core not used)")

        # Pass 3.97: Decompose fused gate+up SwiGLU patterns
        # (FC → chunk/split → silu → multiply → FC) into standard SwiGLU
        # (gate_proj + up_proj → SwiGLU → down_proj) by splitting the
        # fused FC layer into two separate layers with halved weights.
        layers = _decompose_fused_swiglu(layers)

        # Pass 4: Detect LazyTensor chain opportunities
        lazy_chains = detect_lazy_chains(layers)
        if lazy_chains:
            total_ops = sum(c.chain_length for c in lazy_chains)
            print(f"  [LAZY] Found {len(lazy_chains)} LazyTensor chain "
                  f"opportunities (total {total_ops} ops)")
            print(f"  [LAZY] Note: These chains are decomposed forms of "
                  f"higher-level ops (e.g. GELU activation, relative "
                  f"position bias) that NNTrainer handles internally via "
                  f"built-in layers (activation, mha_core). No C++ "
                  f"LazyTensor code generation needed for weight conversion.")

        # Pass 5: Detect structural patterns (attention, FFN, blocks)
        model_structure = detect_patterns(layers, self.config)

        # Store fused_ops in structure for emitter to use
        model_structure.fused_ops = self.fused_ops

        # Optimization notifications
        if model_structure.blocks:
            available_opts = []
            if "attention" not in self.fused_ops:
                has_attn = any(b.attention for b in model_structure.blocks)
                if has_attn:
                    available_opts.append(
                        "attention -> mha_core (requires verification)")
            if "swiglu" not in self.fused_ops:
                has_swiglu = any(
                    b.ffn and b.ffn.ffn_type == "swiglu"
                    for b in model_structure.blocks)
                if has_swiglu:
                    available_opts.append(
                        "SwiGLU FFN -> swiglu custom layer (requires verification)")
            if available_opts:
                print(f"  [OPT] Optimization opportunities (use --fused-ops "
                      f"to enable after verification):")
                for opt in available_opts:
                    print(f"    - {opt}")

        # Collect diagnostics
        remaining_unknowns = [l for l in layers
                              if l.layer_type.startswith("unknown")
                              or l.layer_type == OP_UNSUPPORTED]
        tensor_ops = [l for l in layers
                      if l.layer_type.startswith("tensor_op:")]

        return ConversionResult(
            layers=layers,
            decomposed_module_types=all_decomposed_types,
            unsupported_ops=[l for l in remaining_unknowns
                             if l.layer_type == OP_UNSUPPORTED],
            unknown_layers=[l for l in remaining_unknowns
                            if l.layer_type.startswith("unknown")],
            tensor_ops=tensor_ops,
            lazy_chains=lazy_chains,
            graph=tracer.graph,
            model_structure=model_structure,
            training=self.training,
            collapsed_rope_layers=collapsed_rope,
        )


def _add_input_layers_and_shape_info(layers, graph, input_kwargs):
    """Add input layers for external inputs and extract shape metadata.

    1. Detects external inputs (referenced but not defined) and creates
       NNTrainer input layers with proper input_shape.
    2. Extracts target_shape for reshape/view operations from FX graph args.
    3. Extracts slice parameters (start_index, end_index, axis) for
       __getitem__ operations from FX graph args.

    Args:
        layers: List of NNTrainerLayerDef
        graph: FX graph from tracer
        input_kwargs: Dict of model inputs with tensor shapes

    Returns:
        Updated layers list with input layers prepended.
    """
    import torch

    # Build name->node lookup from FX graph
    fx_nodes = {node.name: node for node in graph.nodes}

    # Detect external inputs
    defined = set(l.name for l in layers)
    external_inputs = []
    seen = set()
    for l in layers:
        for inp in l.input_layers:
            if inp not in defined and inp not in seen:
                seen.add(inp)
                external_inputs.append(inp)

    # Create input layers for external inputs
    input_layers = []
    for inp_name in external_inputs:
        tensor = input_kwargs.get(inp_name)
        if tensor is not None and isinstance(tensor, torch.Tensor):
            # Convert PyTorch shape to NNTrainer input_shape (C:H:W).
            # Strip batch dimension (dim 0); pad to 3D if needed.
            dims = list(tensor.shape[1:])  # skip batch
            while len(dims) < 3:
                dims.insert(0, 1)
            input_shape = ":".join(str(d) for d in dims[-3:])
        else:
            input_shape = "1:1:1"  # fallback

        input_layers.append(NNTrainerLayerDef(
            layer_type=LAYER_INPUT,
            name=inp_name,
            properties={"input_shape": input_shape},
        ))

    if input_layers:
        count = len(input_layers)
        names = ", ".join(l.name for l in input_layers)
        print(f"  [CLEANUP] Added {count} input layers: {names}")

    # Extract shape info for reshape/view/unsqueeze/squeeze layers
    for layer in layers:
        if layer.layer_type not in (LAYER_RESHAPE, OP_RESHAPE):
            continue
        # Look up FX node by fx_node_name (preferred) or layer name
        node = fx_nodes.get(layer.fx_node_name) or fx_nodes.get(layer.name)
        if node is None:
            continue

        target = getattr(node, 'target', '')

        if target in ('view', 'reshape'):
            # node.args = (input_node, dim1, dim2, ...) or
            # node.args = (input_node, (dim1, dim2, ...))
            shape_args = node.args[1:]
            if len(shape_args) == 1 and isinstance(shape_args[0], (list, tuple)):
                shape_args = shape_args[0]
            # Strip batch dim (first dim), convert to C:H:W
            dims = [a for a in shape_args if isinstance(a, int)]
            if dims:
                dims = dims[1:]  # skip batch
                while len(dims) < 3:
                    dims.insert(0, 1)
                layer.properties["target_shape"] = \
                    ":".join(str(d) for d in dims[-3:])

        else:
            # For unsqueeze, squeeze, or any other reshape variant:
            # use output_shape captured during tracing
            out_shape = node.meta.get('output_shape')
            if out_shape is not None:
                dims = list(out_shape[1:])  # skip batch
                while len(dims) < 3:
                    dims.insert(0, 1)
                layer.properties["target_shape"] = \
                    ":".join(str(d) for d in dims[-3:])

    # Extract slice parameters for __getitem__ layers
    for layer in layers:
        if layer.layer_type != "slice":
            continue
        node = fx_nodes.get(layer.fx_node_name) or fx_nodes.get(layer.name)
        if node is None or len(node.args) < 2:
            continue

        # Determine input tensor rank for PyTorch→NCHW axis conversion
        input_node = node.args[0] if hasattr(node.args[0], 'name') else None
        input_rank = 4  # default
        if input_node:
            in_shape = input_node.meta.get('output_shape')
            if in_shape:
                input_rank = len(in_shape)

        index_arg = node.args[1]
        # Handle multi-dimensional indexing: tensor[..., idx]
        # e.g. span_idx[:, :, 0] → args[1] = (slice(None), slice(None), 0)
        if isinstance(index_arg, (list, tuple)):
            # Find the axis being indexed (first non-slice(None) element)
            for ax, idx in enumerate(index_arg):
                if isinstance(idx, int):
                    # Convert PyTorch dim to NCHW axis (1-3).
                    # For a tensor of rank R mapped to 4D NCHW:
                    #   nchw_dim = pytorch_dim + (4 - R) for dims > 0
                    nn_axis = ax + (4 - input_rank)
                    nn_axis = max(1, min(3, nn_axis))  # clamp to 1-3
                    if idx < 0:
                        # Need input shape to resolve negative index
                        in_shape = input_node.meta.get('output_shape') \
                            if input_node else None
                        if in_shape and ax < len(in_shape):
                            idx = in_shape[ax] + idx  # resolve negative
                        else:
                            break
                    # NNTrainer slice: 1-based, end is exclusive
                    layer.properties["axis"] = nn_axis
                    layer.properties["start_index"] = idx + 1
                    layer.properties["end_index"] = idx + 2
                    break
                elif isinstance(idx, slice) and idx != slice(None):
                    # Range slicing: tensor[:, :seq_len] →
                    # slice(None, seq_len) or slice(start, stop)
                    nn_axis = ax + (4 - input_rank)
                    nn_axis = max(1, min(3, nn_axis))
                    start = idx.start if idx.start is not None else 0
                    stop = idx.stop
                    if stop is None:
                        # open-ended slice like tensor[start:] — need
                        # input shape to resolve
                        in_shape = (input_node.meta.get('output_shape')
                                    if input_node else None)
                        if in_shape and ax < len(in_shape):
                            stop = in_shape[ax]
                        else:
                            break
                    if start < 0 or stop < 0:
                        in_shape = (input_node.meta.get('output_shape')
                                    if input_node else None)
                        if in_shape and ax < len(in_shape):
                            if start < 0:
                                start = in_shape[ax] + start
                            if stop < 0:
                                stop = in_shape[ax] + stop
                        else:
                            break
                    # NNTrainer slice: 1-based, end is exclusive
                    layer.properties["axis"] = nn_axis
                    layer.properties["start_index"] = start + 1
                    layer.properties["end_index"] = stop + 1
                    break
        elif isinstance(index_arg, int):
            # Simple integer indexing on first non-batch dim
            nn_axis = 1 + (4 - input_rank)
            nn_axis = max(1, min(3, nn_axis))
            layer.properties["axis"] = nn_axis
            layer.properties["start_index"] = index_arg + 1
            layer.properties["end_index"] = index_arg + 2
        elif isinstance(index_arg, slice) and index_arg != slice(None):
            # Simple 1D range slice: tensor[start:stop]
            nn_axis = 1 + (4 - input_rank)
            nn_axis = max(1, min(3, nn_axis))
            start = index_arg.start if index_arg.start is not None else 0
            stop = index_arg.stop
            if stop is None:
                in_shape = (input_node.meta.get('output_shape')
                            if input_node else None)
                if in_shape and len(in_shape) > 0:
                    stop = in_shape[0]
                else:
                    continue
            layer.properties["axis"] = nn_axis
            layer.properties["start_index"] = start + 1
            layer.properties["end_index"] = stop + 1

    # Fix gather axis: convert PyTorch dim to NCHW axis (1-3)
    for layer in layers:
        if layer.layer_type != "gather":
            continue
        if "axis" not in layer.properties:
            continue
        node = fx_nodes.get(layer.fx_node_name) or fx_nodes.get(layer.name)
        if node is None:
            continue
        # Gather's first input is the data tensor; get its rank
        data_node = node.args[0] if hasattr(node.args[0], 'name') else None
        if data_node:
            data_shape = data_node.meta.get('output_shape')
            if data_shape:
                data_rank = len(data_shape)
                pytorch_axis = layer.properties["axis"]
                # Convert: nchw_dim = pytorch_dim + (4 - rank) for dim > 0
                nn_axis = pytorch_axis + (4 - data_rank)
                nn_axis = max(1, min(3, nn_axis))
                layer.properties["axis"] = nn_axis

    return input_layers + layers


class ConversionResult:
    """Result of adaptive conversion pipeline."""

    def __init__(self, layers, decomposed_module_types, unsupported_ops,
                 unknown_layers, tensor_ops, lazy_chains, graph,
                 model_structure=None, training=False,
                 collapsed_rope_layers=None):
        self.layers = layers
        self.decomposed_module_types = decomposed_module_types
        self.unsupported_ops = unsupported_ops
        self.unknown_layers = unknown_layers
        self.tensor_ops = tensor_ops
        self.lazy_chains = lazy_chains
        self.graph = graph
        self.model_structure = model_structure
        self.training = training
        self.collapsed_rope_layers = collapsed_rope_layers or set()

    @property
    def is_fully_mapped(self):
        """True if all ops are mapped (no unknowns or unsupported)."""
        return len(self.unsupported_ops) == 0 and len(self.unknown_layers) == 0

    def summary(self):
        """Print a summary of the conversion result."""
        type_counts = {}
        for layer in self.layers:
            type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1

        mode = "training" if self.training else "inference"
        print(f"\n{'='*70}")
        print(f"CONVERSION SUMMARY ({mode} mode)")
        print(f"{'='*70}")
        print(f"Total layers: {len(self.layers)}")
        print(f"Fully mapped: {self.is_fully_mapped}")
        if not self.training:
            print(f"Dropout layers: removed (inference mode)")

        if self.collapsed_rope_layers:
            print(f"RoPE: collapsed {len(self.collapsed_rope_layers)} layers "
                  f"-> mha_core (NEON/AVX2 optimized)")
        if self.decomposed_module_types:
            print(f"Decomposed module types: "
                  f"{', '.join(sorted(self.decomposed_module_types))}")
        if self.tensor_ops:
            tensor_op_names = set(l.properties.get("original_op", "?")
                                  for l in self.tensor_ops)
            print(f"Tensor method ops: {', '.join(sorted(tensor_op_names))} "
                  f"({len(self.tensor_ops)} total)")
        if self.lazy_chains:
            print(f"LazyTensor chains: {len(self.lazy_chains)} chains "
                  f"({sum(c.chain_length for c in self.lazy_chains)} ops fusible)")
            for i, chain in enumerate(self.lazy_chains):
                print(f"  chain[{i}]: {chain}")
            print(f"  (Info: These are decomposed higher-level ops like GELU "
                  f"and position bias. NNTrainer built-in layers handle "
                  f"them; no LazyTensor C++ code needed.)")
        if self.unsupported_ops:
            print(f"Unsupported ops (no NNTrainer equivalent): "
                  f"{len(self.unsupported_ops)}")
            for op in self.unsupported_ops:
                print(f"  - {op.properties.get('original_op', '?')} "
                      f"at {op.hf_module_name}")
        if self.unknown_layers:
            print(f"Unknown layers: {len(self.unknown_layers)}")
            for l in self.unknown_layers:
                print(f"  - {l.layer_type} at {l.hf_module_name}")

        print(f"\nLayer type breakdown:")
        for lt, count in sorted(type_counts.items()):
            marker = ""
            if lt.startswith("unknown"):
                marker = " <<<< UNKNOWN"
            elif lt == OP_UNSUPPORTED:
                marker = " <<<< UNSUPPORTED"
            elif lt.startswith("tensor_op:"):
                marker = " (Tensor method)"
            print(f"  {lt:30s}: {count}{marker}")
        print(f"{'='*70}")

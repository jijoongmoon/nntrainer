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
    LAYER_POW, LAYER_SQRT, LAYER_MULTIPLY, LAYER_DIVIDE, LAYER_NEGATIVE,
    LAZY_TENSOR_OPS, TENSOR_DIRECT_METHODS,
    OP_UNSUPPORTED, OP_NOOP,
)
from tracer import Tracer, LEAF_MODULES
from node_mapper import NodeMapper


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


# =============================================================================
# Adaptive Converter Pipeline
# =============================================================================

class AdaptiveConverter:
    """Adaptive converter: fused ops first, tensor op fallback for unknowns.

    This is the main entry point for converting any HuggingFace model to
    NNTrainer layer definitions.

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

    def __init__(self, model, model_config=None):
        self.model = model
        self.config = model_config

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

        # Pass 4: Detect LazyTensor chain opportunities
        lazy_chains = detect_lazy_chains(layers)
        if lazy_chains:
            print(f"  [LAZY] Found {len(lazy_chains)} LazyTensor chain "
                  f"opportunities (total {sum(c.chain_length for c in lazy_chains)} ops)")

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
        )


class ConversionResult:
    """Result of adaptive conversion pipeline."""

    def __init__(self, layers, decomposed_module_types, unsupported_ops,
                 unknown_layers, tensor_ops, lazy_chains, graph):
        self.layers = layers
        self.decomposed_module_types = decomposed_module_types
        self.unsupported_ops = unsupported_ops
        self.unknown_layers = unknown_layers
        self.tensor_ops = tensor_ops
        self.lazy_chains = lazy_chains
        self.graph = graph

    @property
    def is_fully_mapped(self):
        """True if all ops are mapped (no unknowns or unsupported)."""
        return len(self.unsupported_ops) == 0 and len(self.unknown_layers) == 0

    def summary(self):
        """Print a summary of the conversion result."""
        type_counts = {}
        for layer in self.layers:
            type_counts[layer.layer_type] = type_counts.get(layer.layer_type, 0) + 1

        print(f"\n{'='*70}")
        print(f"CONVERSION SUMMARY")
        print(f"{'='*70}")
        print(f"Total layers: {len(self.layers)}")
        print(f"Fully mapped: {self.is_fully_mapped}")

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

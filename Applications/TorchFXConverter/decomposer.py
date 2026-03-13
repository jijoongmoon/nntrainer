"""
Adaptive decomposition pipeline for HuggingFace -> NNTrainer conversion.

Strategy:
  1. FUSED OPS FIRST: Trace with LEAF_MODULES, map to NNTrainer fused layers
     (fully_connected, rms_norm, mha_core, swiglu, etc.)
  2. DECOMPOSE UNKNOWNS: If any modules can't be mapped, re-trace with those
     modules excluded from leaves. Their forward() is automatically decomposed
     into tensor ops by the tracer.
  3. OP DECOMPOSITION: For individual ops without direct NNTrainer support
     (rsqrt, abs, exp, log, clamp), decompose into supported primitives.

This ensures the converter works for ANY HuggingFace model:
  - Known modules (Linear, Embedding, RMSNorm, etc.) -> fused NNTrainer layers
  - Unknown modules -> decomposed into tensor ops automatically
  - Unsupported tensor ops -> decomposed into supported primitives
"""

import torch
from typing import Optional

from nntrainer_layers import (
    NNTrainerLayerDef,
    LAYER_POW, LAYER_SQRT, LAYER_MULTIPLY, LAYER_DIVIDE,
    OP_UNSUPPORTED, OP_NOOP,
)
from tracer import Tracer, LEAF_MODULES
from node_mapper import NodeMapper


# =============================================================================
# Op Decomposition Registry
# =============================================================================
# Maps unsupported op names to decomposition functions.
# Each function takes the original NNTrainerLayerDef and returns a list of
# NNTrainerLayerDef that implement the same operation using supported ops.

def _decompose_rsqrt(layer):
    """rsqrt(x) -> pow(x, -0.5)

    NNTrainer supports pow with arbitrary exponents.
    """
    return [NNTrainerLayerDef(
        layer_type=LAYER_POW,
        name=layer.name,
        properties={"exponent": -0.5},
        input_layers=layer.input_layers,
        hf_module_name=layer.hf_module_name,
    )]


def _decompose_abs(layer):
    """abs(x) -> sqrt(multiply(x, x))

    Decompose into: x_sq = x * x, then sqrt(x_sq).
    """
    sq_name = f"{layer.name}_sq"
    return [
        NNTrainerLayerDef(
            layer_type=LAYER_MULTIPLY,
            name=sq_name,
            input_layers=layer.input_layers + layer.input_layers,  # x * x
            hf_module_name=layer.hf_module_name,
        ),
        NNTrainerLayerDef(
            layer_type=LAYER_SQRT,
            name=layer.name,
            input_layers=[sq_name],
            hf_module_name=layer.hf_module_name,
        ),
    ]


def _decompose_reciprocal(layer):
    """reciprocal(x) -> divide(1, x)

    NNTrainer divide supports scalar numerator.
    """
    return [NNTrainerLayerDef(
        layer_type=LAYER_DIVIDE,
        name=layer.name,
        properties={"numerator": 1.0},
        input_layers=layer.input_layers,
        hf_module_name=layer.hf_module_name,
    )]


# Ops that have no decomposition into NNTrainer primitives.
# These are kept as OP_UNSUPPORTED with a warning.
# Future NNTrainer versions may add native support.
_NO_DECOMPOSITION_OPS = {"exp", "log", "clamp", "clip", "clamp_", "clamp_min", "clamp_max"}


# Registry: original_op -> decomposition function
DECOMPOSITION_REGISTRY = {
    "rsqrt": _decompose_rsqrt,
    "abs": _decompose_abs,
    "reciprocal": _decompose_reciprocal,
}


def decompose_unsupported_ops(layers):
    """Replace unsupported ops with decomposed equivalents where possible.

    Args:
        layers: List of NNTrainerLayerDef from NodeMapper.

    Returns:
        New list with unsupported ops decomposed into supported primitives.
        Ops without known decompositions are kept as-is with warnings.
    """
    result = []
    decomposed_count = 0
    undecomposable = []

    for layer in layers:
        if layer.layer_type != OP_UNSUPPORTED:
            result.append(layer)
            continue

        original_op = layer.properties.get("original_op", "")
        decompose_fn = DECOMPOSITION_REGISTRY.get(original_op)

        if decompose_fn:
            decomposed = decompose_fn(layer)
            result.extend(decomposed)
            decomposed_count += len(decomposed)
        else:
            # No decomposition available - keep as unsupported
            undecomposable.append(layer)
            result.append(layer)

    if decomposed_count > 0:
        print(f"  [DECOMPOSE] Decomposed {decomposed_count} ops into supported primitives")
    if undecomposable:
        print(f"  [WARNING] {len(undecomposable)} ops have no decomposition:")
        for layer in undecomposable:
            op = layer.properties.get("original_op", "?")
            print(f"    - {op} at {layer.hf_module_name} ({layer.name})")

    return result


# =============================================================================
# Adaptive Converter Pipeline
# =============================================================================

class AdaptiveConverter:
    """Two-pass converter: fused ops first, tensor op fallback for unknowns.

    This is the main entry point for converting any HuggingFace model to
    NNTrainer layer definitions.

    Pipeline:
        Pass 1: Trace with full LEAF_MODULES
                -> Map to NNTrainer layers
                -> Identify unknown module types

        Pass 2 (if needed): Re-trace with unknown modules excluded from leaves
                -> Their forward() automatically decomposes into tensor ops
                -> Map tensor ops to NNTrainer layers

        Pass 3: Decompose unsupported individual ops (rsqrt, abs, etc.)
                -> Replace with sequences of supported NNTrainer tensor ops

    Usage:
        converter = AdaptiveConverter(model, config)
        result = converter.convert({"input_ids": input_ids})
        # result.layers: List[NNTrainerLayerDef]
        # result.unknown_module_types: set of types that were decomposed
        # result.unsupported_ops: list of ops with no decomposition
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
            ConversionResult with layers, metadata, and diagnostics.
        """
        exclude_types = set()
        all_decomposed_types = set()

        # Iterative passes: trace -> map -> find unknowns -> exclude -> re-trace
        for pass_num in range(1, max_passes + 1):
            print(f"\n  [PASS {pass_num}] Tracing with {len(exclude_types)} excluded module types...")

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

            print(f"  [PASS {pass_num}] Decomposing {len(new_types)} unknown module types: "
                  f"{', '.join(sorted(new_types))}")
            exclude_types |= new_types
            all_decomposed_types |= new_types

        # Pass 3: Decompose unsupported individual ops
        layers = decompose_unsupported_ops(layers)

        # Collect diagnostics
        remaining_unknowns = [l for l in layers
                              if l.layer_type.startswith("unknown")
                              or l.layer_type == OP_UNSUPPORTED]

        return ConversionResult(
            layers=layers,
            decomposed_module_types=all_decomposed_types,
            unsupported_ops=[l for l in remaining_unknowns
                             if l.layer_type == OP_UNSUPPORTED],
            unknown_layers=[l for l in remaining_unknowns
                            if l.layer_type.startswith("unknown")],
            graph=tracer.graph,
        )


class ConversionResult:
    """Result of adaptive conversion pipeline."""

    def __init__(self, layers, decomposed_module_types, unsupported_ops,
                 unknown_layers, graph):
        self.layers = layers
        self.decomposed_module_types = decomposed_module_types
        self.unsupported_ops = unsupported_ops
        self.unknown_layers = unknown_layers
        self.graph = graph

    @property
    def is_fully_mapped(self):
        """True if all ops are mapped to NNTrainer layer types."""
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
            print(f"Decomposed module types: {', '.join(sorted(self.decomposed_module_types))}")
        if self.unsupported_ops:
            print(f"Unsupported ops (no NNTrainer equivalent): {len(self.unsupported_ops)}")
            for op in self.unsupported_ops:
                print(f"  - {op.properties.get('original_op', '?')} at {op.hf_module_name}")
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
            print(f"  {lt:30s}: {count}{marker}")
        print(f"{'='*70}")

"""Custom model loaders for non-HuggingFace model types.

When a model uses a custom ``model_type`` not registered in HuggingFace
transformers (e.g. GLiNER2's ``"extractor"``), the converter cannot use
``AutoModel.from_pretrained``.  This module provides a registry of
lightweight loaders that reconstruct a *traceable* ``nn.Module`` from a
model directory (config files + weight files) **without requiring the
original training package**.

Each loader returns ``(model, config, trace_inputs)`` so the converter
can treat every model uniformly.

Adding a new custom model type
------------------------------
1. Write a loader function with signature::

       def load_mymodel(model_dir, config, seq_len, verbose):
           ...
           return model, config, trace_inputs

2. Register it::

       CUSTOM_LOADERS["my_model_type"] = load_mymodel
"""

import argparse
import json
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps model_type string -> loader function.
CUSTOM_LOADERS: Dict[str, Callable] = {}


def load_custom_model(
    model_type: str,
    model_dir: str,
    config: Any,
    seq_len: int = 8,
    verbose: bool = True,
) -> Tuple[nn.Module, Any, Dict[str, torch.Tensor]]:
    """Load a custom model using the registered loader.

    Returns:
        (model, config, trace_inputs) — ready for AdaptiveConverter.
    """
    loader = CUSTOM_LOADERS.get(model_type)
    if loader is None:
        raise ValueError(
            f"No custom loader registered for model_type '{model_type}'.  "
            f"Available: {sorted(CUSTOM_LOADERS.keys())}")
    return loader(model_dir, config, seq_len, verbose)


# ---------------------------------------------------------------------------
# Utility: load weights from a model directory
# ---------------------------------------------------------------------------

def _load_state_dict(model_dir: str) -> Dict[str, torch.Tensor]:
    """Load state_dict from model.safetensors or pytorch_model.bin."""
    safetensors = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.isfile(safetensors):
        try:
            from safetensors.torch import load_file
            return load_file(safetensors)
        except ImportError:
            pass
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"No model weights in {model_dir} "
        f"(expected model.safetensors or pytorch_model.bin)")


def _load_extra_config(model_dir: str, filename: str) -> Optional[dict]:
    """Load an optional JSON config file, returning None if absent."""
    path = os.path.join(model_dir, filename)
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Utility: infer nn.Sequential projection from weight shapes
# ---------------------------------------------------------------------------

def _build_sequential_from_weights(
    state: Dict[str, torch.Tensor],
    prefix: str,
) -> nn.Sequential:
    """Build an nn.Sequential MLP by inspecting weight shapes.

    Scans ``state`` for keys matching ``<prefix>0.weight``,
    ``<prefix>0.bias``, ``<prefix>3.weight``, etc. and infers the layer
    sizes from the tensor shapes.  Assumes the pattern
    ``Linear – ReLU – Dropout – Linear – ...`` (indices 0, 1, 2, 3, ...).
    """
    # Collect linear layer indices and their weight shapes
    linears = {}  # idx -> (in_features, out_features)
    for key, tensor in state.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        # e.g. "0.weight", "3.bias"
        parts = suffix.split(".")
        if len(parts) == 2 and parts[1] == "weight" and tensor.dim() == 2:
            idx = int(parts[0])
            out_f, in_f = tensor.shape
            linears[idx] = (in_f, out_f)

    if not linears:
        raise RuntimeError(
            f"No linear layers found with prefix '{prefix}' in state_dict")

    # Build layers in index order
    modules = []
    for idx in sorted(linears.keys()):
        # Insert ReLU + Dropout between linear layers
        if modules:
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(0.0))  # placeholder, removed at eval
        in_f, out_f = linears[idx]
        modules.append(nn.Linear(in_f, out_f))

    return nn.Sequential(*modules)


# ---------------------------------------------------------------------------
# Utility: infer span marker module from weight shapes
# ---------------------------------------------------------------------------

class _SpanMarker(nn.Module):
    """Generic span marker built from weight shapes (gather + MLP + concat)."""

    def __init__(self, project_start, project_end, out_project, max_width):
        super().__init__()
        self.max_width = max_width
        self.project_start = project_start
        self.project_end = project_end
        self.out_project = out_project

    def forward(self, h, span_idx):
        B, L, D = h.size()
        start_rep = self.project_start(h)
        end_rep = self.project_end(h)
        idx_s = span_idx[:, :, 0].unsqueeze(2).expand(-1, -1, D)
        idx_e = span_idx[:, :, 1].unsqueeze(2).expand(-1, -1, D)
        start_span = torch.gather(start_rep, 1, idx_s)
        end_span = torch.gather(end_rep, 1, idx_e)
        cat = torch.cat([start_span, end_span], dim=-1).relu()
        return self.out_project(cat).view(B, L, self.max_width, D)


def _build_span_marker_from_weights(
    state: Dict[str, torch.Tensor],
    prefix: str,
    max_width: int,
) -> _SpanMarker:
    """Build a SpanMarker by inspecting weight shapes under ``prefix``."""
    project_start = _build_sequential_from_weights(
        state, f"{prefix}project_start.")
    project_end = _build_sequential_from_weights(
        state, f"{prefix}project_end.")
    out_project = _build_sequential_from_weights(
        state, f"{prefix}out_project.")
    return _SpanMarker(project_start, project_end, out_project, max_width)


# ---------------------------------------------------------------------------
# GLiNER / GLiNER2 loader  (model_type = "extractor")
# ---------------------------------------------------------------------------

class _ExtractorPromptCore(nn.Module):
    """Traceable core: SpanMarker + prompt projection + einsum scoring.

    Variant used by original GLiNER (prompt-based scoring).
    """

    def __init__(self, rnn, span_marker, prompt_proj, has_rnn=True):
        super().__init__()
        self.has_rnn = has_rnn
        if has_rnn and rnn is not None:
            self.rnn = rnn
        self.span_marker = span_marker
        self.prompt_proj = prompt_proj

    def forward(self, words_embedding, span_idx, prompts_embedding):
        if self.has_rnn:
            words_embedding, _ = self.rnn(words_embedding)
        span_rep = self.span_marker(words_embedding, span_idx)
        prompts_embedding = self.prompt_proj(prompts_embedding)
        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_embedding)
        return scores


class _ExtractorClassifierCore(nn.Module):
    """Traceable core: SpanMarker + classifier scoring.

    Variant used by GLiNER2 (learned classifier instead of prompt projection).
    Scoring: element-wise product of span and class representations,
    then classifier MLP to produce per-(span, class) scalar scores.
    """

    def __init__(self, span_marker, classifier):
        super().__init__()
        self.span_marker = span_marker
        self.classifier = classifier

    def forward(self, words_embedding, span_idx, prompts_embedding):
        span_rep = self.span_marker(words_embedding, span_idx)
        # span_rep: [B, L, K, D], prompts_embedding: [B, C, D]
        B, L, K, D = span_rep.shape
        C = prompts_embedding.shape[1]
        span_flat = span_rep.view(B, L * K, D)
        # Element-wise product for each (span, class) pair
        span_exp = span_flat.unsqueeze(2).expand(B, L * K, C, D)
        prompt_exp = prompts_embedding.unsqueeze(1).expand(B, L * K, C, D)
        combined = span_exp * prompt_exp  # [B, L*K, C, D]
        scores = self.classifier(combined).squeeze(-1)  # [B, L*K, C]
        return scores.view(B, L, K, C)


def _find_prefix(state, leaf_pattern):
    """Find the key prefix before ``leaf_pattern`` in state_dict."""
    for key in state:
        idx = key.find(leaf_pattern)
        if idx >= 0:
            return key[:idx]
    return None


def _load_extractor(model_dir, config, seq_len, verbose):
    """Load GLiNER/GLiNER2 'extractor' model from config + weights."""
    # Merge gliner_config.json if present (has span_mode, max_width, etc.)
    extra = _load_extra_config(model_dir, "gliner_config.json")
    if extra:
        for k, v in extra.items():
            if not hasattr(config, k):
                setattr(config, k, v)
        if verbose:
            print(f"  Loaded gliner_config.json "
                  f"(span_mode={extra.get('span_mode', '?')})")

    hidden_size = getattr(config, "hidden_size", 768)
    max_width = getattr(config, "max_width", 12)
    num_rnn_layers = getattr(config, "num_rnn_layers", 1)

    # Load weights and discover key prefixes
    state = _load_state_dict(model_dir)

    # Find module prefixes by searching for known leaf key patterns
    lstm_prefix = _find_prefix(state, "lstm.weight_ih_l0")
    span_prefix = _find_prefix(state, "project_start.0.weight")

    # Detect scoring variant: prompt-based or classifier-based
    prompt_prefix = None
    for key in state:
        if "prompt" in key and key.endswith(".0.weight"):
            prompt_prefix = key[:-len("0.weight")]
            break

    classifier_prefix = _find_prefix(state, "classifier.0.weight")
    if classifier_prefix is not None:
        classifier_prefix = classifier_prefix + "classifier."

    variant = "prompt" if prompt_prefix is not None else \
              "classifier" if classifier_prefix is not None else None

    if verbose:
        top = sorted({k.split(".")[0] for k in state})
        print(f"  Weight prefixes: lstm={lstm_prefix!r}, "
              f"span={span_prefix!r}, prompt={prompt_prefix!r}, "
              f"classifier={classifier_prefix!r}")
        print(f"  Top-level keys: {top}")
        print(f"  Scoring variant: {variant}")

    if variant is None:
        raise RuntimeError(
            "Could not detect scoring variant (no prompt_rep or classifier "
            f"weights found) in {model_dir}")

    # --- Build RNN (optional) ---
    has_rnn = num_rnn_layers > 0 and lstm_prefix is not None
    rnn = None
    if has_rnn:
        w = state[f"{lstm_prefix}lstm.weight_ih_l0"]
        input_size = w.shape[1]
        lstm_hidden = w.shape[0] // 4  # 4 gates
        rnn = nn.LSTM(
            input_size=input_size, hidden_size=lstm_hidden,
            num_layers=num_rnn_layers, bidirectional=True, batch_first=True,
        )
        rnn_state = {}
        full_prefix = f"{lstm_prefix}lstm."
        for key, val in state.items():
            if key.startswith(full_prefix):
                rnn_state[key[len(full_prefix):]] = val
        rnn.load_state_dict(rnn_state)

    # --- Build span marker ---
    if span_prefix is None:
        raise RuntimeError(
            "Could not find span marker weights (project_start.0.weight) "
            f"in {model_dir}")
    span_marker = _build_span_marker_from_weights(state, span_prefix,
                                                  max_width)
    span_state = {}
    for key, val in state.items():
        if key.startswith(span_prefix) and any(
                s in key for s in ("project_start", "project_end",
                                   "out_project")):
            span_state[key[len(span_prefix):]] = val
    span_marker.load_state_dict(span_state)

    # --- Build scoring module and assemble core ---
    if variant == "prompt":
        prompt_proj = _build_sequential_from_weights(state, prompt_prefix)
        prompt_state = {}
        for key, val in state.items():
            if key.startswith(prompt_prefix):
                suffix = key[len(prompt_prefix):]
                if suffix and suffix[0].isdigit():
                    prompt_state[suffix] = val
        prompt_proj.load_state_dict(prompt_state)
        model = _ExtractorPromptCore(rnn, span_marker, prompt_proj,
                                     has_rnn=has_rnn)
    else:
        # classifier variant
        classifier = _build_sequential_from_weights(state, classifier_prefix)
        cls_state = {}
        for key, val in state.items():
            if key.startswith(classifier_prefix):
                suffix = key[len(classifier_prefix):]
                if suffix and suffix[0].isdigit():
                    cls_state[suffix] = val
        classifier.load_state_dict(cls_state)
        model = _ExtractorClassifierCore(span_marker, classifier)

    model.eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Extractor core built ({variant}): "
              f"{n_params / 1e6:.1f}M params, "
              f"hidden={hidden_size}, max_width={max_width}, "
              f"rnn={'yes' if has_rnn else 'no'}")

    # Trace inputs
    W = seq_len
    num_classes = 5  # dummy entity type count
    trace_inputs = {
        "words_embedding": torch.randn(1, W, hidden_size),
        "span_idx": torch.randint(0, W, (1, W * max_width, 2)),
        "prompts_embedding": torch.randn(1, num_classes, hidden_size),
    }

    return model, config, trace_inputs


# Register the loader
CUSTOM_LOADERS["extractor"] = _load_extractor

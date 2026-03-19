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

    # Build layers matching the original indices exactly.
    # The gap between consecutive linear indices tells us what non-linear
    # layers sit between them:
    #   gap=2 (e.g. 0,2): Linear – ReLU – Linear
    #   gap=3 (e.g. 0,3): Linear – ReLU – Dropout – Linear
    sorted_indices = sorted(linears.keys())
    modules = {}
    for i, idx in enumerate(sorted_indices):
        in_f, out_f = linears[idx]
        modules[idx] = nn.Linear(in_f, out_f)
        if i < len(sorted_indices) - 1:
            next_idx = sorted_indices[i + 1]
            gap = next_idx - idx
            modules[idx + 1] = nn.ReLU()
            if gap >= 3:
                modules[idx + 2] = nn.Dropout(0.0)

    max_idx = max(modules.keys())
    ordered = [modules[i] for i in range(max_idx + 1) if i in modules]
    return nn.Sequential(*ordered)


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


# ---------------------------------------------------------------------------
# Mamba loader (model_type = "mamba")
# ---------------------------------------------------------------------------
# HuggingFace Mamba models normally load via AutoModelForCausalLM, but the
# mamba_ssm CUDA package may not be available. This loader patches the model
# to force the pure-PyTorch slow_forward path which is fully traceable by
# our FX tracer.

def _patch_mamba_slow_forward(model):
    """Patch MambaMixer modules to use slow_forward (pure PyTorch).

    The HuggingFace MambaMixer.forward() tries to use mamba_ssm CUDA
    kernels first, falling back to slow_forward(). For tracing we need
    the slow path which is decomposable into standard tensor ops.
    """
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "MambaMixer" in cls_name or "Mamba2Mixer" in cls_name:
            if hasattr(module, "slow_forward"):
                # Replace forward with slow_forward so the tracer
                # decomposes the SSM into tensor operations
                module.forward = module.slow_forward
    return model


def _load_mamba(model_dir, config, seq_len, verbose):
    """Load Mamba model with pure-PyTorch forward for tracing."""
    from transformers import AutoModelForCausalLM, AutoConfig

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32)
    model.eval()

    # Patch to use slow_forward (pure PyTorch, no CUDA kernels)
    _patch_mamba_slow_forward(model)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        state_size = getattr(config, "state_size", "?")
        d_conv = getattr(config, "conv_kernel",
                 getattr(config, "d_conv", "?"))
        expand = getattr(config, "expand", "?")
        print(f"  Mamba model loaded: {n_params / 1e6:.1f}M params, "
              f"state_size={state_size}, d_conv={d_conv}, expand={expand}")
        print(f"  Patched MambaMixer to use slow_forward (pure PyTorch)")

    vocab_size = getattr(config, "vocab_size", 30000)
    input_kwargs = {"input_ids": torch.randint(0, vocab_size, (1, seq_len))}

    return model, config, input_kwargs


CUSTOM_LOADERS["mamba"] = _load_mamba
CUSTOM_LOADERS["mamba2"] = _load_mamba


# ---------------------------------------------------------------------------
# FLUX loader  (model_type = "flux")
# ---------------------------------------------------------------------------
# FLUX is a Diffusion Transformer (DiT) model from Black Forest Labs.
# It uses diffusers FluxTransformer2DModel, not standard HuggingFace
# AutoModel. This loader creates a tiny or full FLUX model for tracing.

def _load_flux(model_dir, config, seq_len, verbose):
    """Load FLUX diffusion transformer model for tracing.

    Supports both local paths (with diffusers model_index.json) and
    tiny model creation for testing.
    """
    from diffusers.models.transformers.transformer_flux import (
        FluxTransformer2DModel,
    )

    # Extract config params
    in_channels = getattr(config, "in_channels", 64)
    num_layers = getattr(config, "num_layers", 19)
    num_single_layers = getattr(config, "num_single_layers", 38)
    attention_head_dim = getattr(config, "attention_head_dim", 128)
    num_attention_heads = getattr(config, "num_attention_heads", 24)
    joint_attention_dim = getattr(config, "joint_attention_dim", 4096)
    pooled_projection_dim = getattr(config, "pooled_projection_dim", 768)
    guidance_embeds = getattr(config, "guidance_embeds", False)
    axes_dims_rope = tuple(getattr(config, "axes_dims_rope", (16, 56, 56)))

    # Check if this is a local diffusers model directory
    transformer_dir = os.path.join(model_dir, "transformer")
    if os.path.isdir(transformer_dir):
        model = FluxTransformer2DModel.from_pretrained(
            transformer_dir, torch_dtype=torch.float32)
    elif os.path.isfile(os.path.join(model_dir, "config.json")):
        # Direct transformer directory
        model = FluxTransformer2DModel.from_pretrained(
            model_dir, torch_dtype=torch.float32)
    else:
        # Create model from config (for testing with tiny configs)
        model = FluxTransformer2DModel(
            patch_size=getattr(config, "patch_size", 1),
            in_channels=in_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )

    model.eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        hidden = num_attention_heads * attention_head_dim
        print(f"  FLUX model loaded: {n_params / 1e6:.1f}M params")
        print(f"    hidden_size={hidden}, layers={num_layers}+"
              f"{num_single_layers}, heads={num_attention_heads}, "
              f"head_dim={attention_head_dim}")
        print(f"    joint_attention_dim={joint_attention_dim}, "
              f"in_channels={in_channels}")

    # Prepare trace inputs matching FluxTransformer2DModel.forward()
    hidden_size = num_attention_heads * attention_head_dim
    # Spatial dims: small patch count for tracing
    h_patches = 2
    w_patches = 2
    num_patches = h_patches * w_patches
    txt_len = min(seq_len, 4)

    input_kwargs = {
        "hidden_states": torch.randn(1, num_patches, in_channels),
        "encoder_hidden_states": torch.randn(1, txt_len,
                                             joint_attention_dim),
        "pooled_projections": torch.randn(1, pooled_projection_dim),
        "timestep": torch.tensor([1.0]),
        "img_ids": torch.zeros(num_patches, 3),
        "txt_ids": torch.zeros(txt_len, 3),
    }

    return model, config, input_kwargs


CUSTOM_LOADERS["flux"] = _load_flux


# ---------------------------------------------------------------------------
# Conformer loader  (model_type = "conformer")
# ---------------------------------------------------------------------------
# Conformer is a speech/audio model combining convolution and transformer.
# It uses torchaudio's Conformer implementation, not HuggingFace AutoModel.

def _load_conformer(model_dir, config, seq_len, verbose):
    """Load Conformer model for tracing."""
    from torchaudio.models import Conformer

    input_dim = getattr(config, "input_dim", getattr(config, "hidden_size", 80))
    num_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", 4))
    ffn_dim = getattr(config, "intermediate_size", getattr(config, "ffn_dim", 256))
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "num_layers", 12))
    depthwise_conv_kernel_size = getattr(config, "depthwise_conv_kernel_size", 31)

    model = Conformer(
        input_dim=input_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        depthwise_conv_kernel_size=depthwise_conv_kernel_size,
    )
    model.eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Conformer model loaded: {n_params / 1e6:.1f}M params")
        print(f"    input_dim={input_dim}, layers={num_layers}, "
              f"heads={num_heads}, ffn_dim={ffn_dim}, "
              f"conv_kernel={depthwise_conv_kernel_size}")

    # Conformer.forward(input, lengths) -> (output, lengths)
    # input shape: (batch, time, input_dim)
    time_steps = max(seq_len, 16)
    input_kwargs = {
        "input": torch.randn(1, time_steps, input_dim),
        "lengths": torch.tensor([time_steps]),
    }

    return model, config, input_kwargs


CUSTOM_LOADERS["conformer"] = _load_conformer


# ---------------------------------------------------------------------------
# Zipformer loader  (model_type = "zipformer")
# ---------------------------------------------------------------------------
# Zipformer is a speech recognition model from k2-fsa/icefall.
# It requires the icefall source (zipformer.py + scaling.py) to be
# available in sys.path.

def _load_zipformer(model_dir, config, seq_len, verbose):
    """Load Zipformer2 model for tracing.

    Expects either:
    - A directory containing zipformer.py + scaling.py (icefall-style), or
    - Config with Zipformer parameters to create a model from scratch.
    """
    import sys

    # Try to import from model_dir first, then from well-known locations
    zipformer_paths = [
        model_dir,
        os.path.join(model_dir, "zipformer"),
        "/tmp/zipformer_standalone",
    ]
    for p in zipformer_paths:
        if os.path.isfile(os.path.join(p, "zipformer.py")):
            if p not in sys.path:
                sys.path.insert(0, p)
            break

    from zipformer import Zipformer2

    # Extract config — Zipformer uses tuple params for multi-stack
    num_stacks = getattr(config, "num_stacks", 2)
    encoder_dim = getattr(config, "encoder_dim", None)
    if encoder_dim is None:
        hidden = getattr(config, "hidden_size", 384)
        encoder_dim = tuple([hidden] * num_stacks)
    elif isinstance(encoder_dim, int):
        encoder_dim = tuple([encoder_dim] * num_stacks)

    num_encoder_layers = getattr(config, "num_encoder_layers", None)
    if num_encoder_layers is None:
        n_layers = getattr(config, "num_hidden_layers", 4)
        num_encoder_layers = tuple([n_layers] * num_stacks)
    elif isinstance(num_encoder_layers, int):
        num_encoder_layers = tuple([num_encoder_layers] * num_stacks)

    ffn_dim = getattr(config, "feedforward_dim", None)
    if ffn_dim is None:
        inter = getattr(config, "intermediate_size", 1536)
        ffn_dim = tuple([inter] * num_stacks)
    elif isinstance(ffn_dim, int):
        ffn_dim = tuple([ffn_dim] * num_stacks)

    downsampling = getattr(config, "downsampling_factor", (1, 2))
    if isinstance(downsampling, int):
        downsampling = tuple([downsampling] * num_stacks)

    cnn_kernel = getattr(config, "cnn_module_kernel", (31, 31))
    if isinstance(cnn_kernel, int):
        cnn_kernel = tuple([cnn_kernel] * num_stacks)

    num_heads = getattr(config, "num_attention_heads",
                        getattr(config, "num_heads", 8))
    query_head_dim = getattr(config, "query_head_dim", 24)
    pos_head_dim = getattr(config, "pos_head_dim", 4)
    value_head_dim = getattr(config, "value_head_dim", 12)
    pos_dim = getattr(config, "pos_dim", 192)

    model = Zipformer2(
        output_downsampling_factor=getattr(
            config, "output_downsampling_factor", 2),
        downsampling_factor=downsampling,
        encoder_dim=encoder_dim,
        num_encoder_layers=num_encoder_layers,
        encoder_unmasked_dim=getattr(
            config, "encoder_unmasked_dim",
            tuple([min(256, d) for d in encoder_dim])),
        query_head_dim=query_head_dim,
        pos_head_dim=pos_head_dim,
        value_head_dim=value_head_dim,
        num_heads=num_heads,
        feedforward_dim=ffn_dim,
        cnn_module_kernel=cnn_kernel,
        pos_dim=pos_dim,
        dropout=0.0,
        causal=False,
    )
    model.eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Zipformer model loaded: {n_params / 1e6:.1f}M params")
        print(f"    encoder_dim={encoder_dim}, "
              f"layers={num_encoder_layers}, heads={num_heads}")
        print(f"    ffn_dim={ffn_dim}, "
              f"downsampling={downsampling}, "
              f"cnn_kernel={cnn_kernel}")

    # Zipformer2.forward(x, x_lens) -> (output, output_lens)
    # x shape: (batch, time, encoder_dim[0])
    input_dim = encoder_dim[0] if isinstance(encoder_dim, tuple) else encoder_dim
    time_steps = max(seq_len, 32)
    input_kwargs = {
        "x": torch.randn(1, time_steps, input_dim),
        "x_lens": torch.tensor([time_steps]),
    }

    return model, config, input_kwargs


CUSTOM_LOADERS["zipformer"] = _load_zipformer


# ─── YOLOv2 ───────────────────────────────────────────────────────────

def _load_yolov2(model_dir, config, seq_len, verbose):
    """Load YOLOv2 model from nntrainer Applications/YOLOv2."""
    import sys as _sys
    import os as _os
    import importlib as _il

    yolov2_path = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "YOLOv2", "PyTorch"))
    # Ensure correct yolo module is loaded (avoid name clash with YOLOv3)
    _sys.modules.pop("yolo", None)
    if yolov2_path not in _sys.path:
        _sys.path.insert(0, yolov2_path)
    else:
        # Move to front so YOLOv2's yolo.py is found first
        _sys.path.remove(yolov2_path)
        _sys.path.insert(0, yolov2_path)

    import yolo as _yolo_mod
    _il.reload(_yolo_mod)
    YoloV2 = _yolo_mod.YoloV2

    num_classes = getattr(config, "num_classes", 5)
    num_anchors = getattr(config, "num_anchors", 5)
    image_size = getattr(config, "image_size", 416)

    model = YoloV2(num_classes=num_classes, num_anchors=num_anchors)
    model.eval()

    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"  [yolov2] {n/1e6:.2f}M params, "
              f"classes={num_classes}, anchors={num_anchors}, "
              f"img={image_size}")

    input_kwargs = {"x": torch.randn(1, 3, image_size, image_size)}
    return model, config, input_kwargs


CUSTOM_LOADERS["yolov2"] = _load_yolov2


# ─── YOLOv3 ───────────────────────────────────────────────────────────

def _load_yolov3(model_dir, config, seq_len, verbose):
    """Load YOLOv3 model from nntrainer Applications/YOLOv3."""
    import sys as _sys
    import os as _os
    import types as _types
    import importlib as _il

    # YOLOv3's yolo.py imports torchconverter (a local util); mock it out
    if "torchconverter" not in _sys.modules:
        _mock = _types.ModuleType("torchconverter")
        _mock.save_bin = lambda *a, **kw: None
        _sys.modules["torchconverter"] = _mock

    yolov3_path = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "YOLOv3", "PyTorch"))
    # Ensure correct yolo module is loaded (avoid name clash with YOLOv2)
    _sys.modules.pop("yolo", None)
    if yolov3_path not in _sys.path:
        _sys.path.insert(0, yolov3_path)
    else:
        _sys.path.remove(yolov3_path)
        _sys.path.insert(0, yolov3_path)

    import yolo as _yolo_mod
    _il.reload(_yolo_mod)
    YoloV3 = _yolo_mod.YoloV3

    num_classes = getattr(config, "num_classes", 5)
    image_size = getattr(config, "image_size", 416)

    model = YoloV3(num_classes=num_classes)
    model.eval()

    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"  [yolov3] {n/1e6:.2f}M params, "
              f"classes={num_classes}, img={image_size}")

    input_kwargs = {"x": torch.randn(1, 3, image_size, image_size)}
    return model, config, input_kwargs


CUSTOM_LOADERS["yolov3"] = _load_yolov3

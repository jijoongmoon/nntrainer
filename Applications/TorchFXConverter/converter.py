#!/usr/bin/env python3
"""
TorchFX-to-NNTrainer Converter CLI.

Converts HuggingFace models to NNTrainer format:
  - C++ model construction code
  - INI configuration file
  - JSON model config + weight map
  - Binary weight file (.bin)

Usage:
  python converter.py --model Qwen/Qwen3-0.6B --output ./output/
  python converter.py --model bert-base-uncased --output ./output/ --format ini
  python converter.py --model google/mt5-small --output ./output/ --format all

Phase 6 of the TorchFX converter pipeline (DESIGN.md).
"""

import argparse
import os
import sys
import json

import torch
import torch.nn as nn


def _gliner_projection(hidden_size, dropout, out_dim=None):
    """Two-layer projection (matches GLiNER create_projection_layer)."""
    if out_dim is None:
        out_dim = hidden_size
    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4), nn.ReLU(),
        nn.Dropout(dropout), nn.Linear(out_dim * 4, out_dim),
    )


class _SpanMarkerV0(nn.Module):
    """Span endpoint marker (matches GLiNER SpanMarkerV0)."""

    def __init__(self, hidden_size, max_width, dropout=0.4):
        super().__init__()
        self.max_width = max_width
        self.project_start = _gliner_projection(hidden_size, dropout)
        self.project_end = _gliner_projection(hidden_size, dropout)
        self.out_project = _gliner_projection(hidden_size * 2, dropout,
                                              hidden_size)

    def forward(self, h, span_idx):
        B, L, D = h.size()
        start_rep = self.project_start(h)
        end_rep = self.project_end(h)
        # gather start/end representations
        idx_s = span_idx[:, :, 0].unsqueeze(2).expand(-1, -1, D)
        idx_e = span_idx[:, :, 1].unsqueeze(2).expand(-1, -1, D)
        start_span = torch.gather(start_rep, 1, idx_s)
        end_span = torch.gather(end_rep, 1, idx_e)
        cat = torch.cat([start_span, end_span], dim=-1).relu()
        return self.out_project(cat).view(B, L, self.max_width, D)


class _GLiNERCore(nn.Module):
    """Traceable core of a GLiNER/GLiNER2 model (post-preprocessing).

    Wraps the BiLSTM, SpanMarkerV0, prompt projection, and einsum scoring.
    The DeBERTa encoder and data-dependent preprocessing are excluded
    (handled outside the NNTrainer graph).

    This class can be built either by extracting sub-modules from a loaded
    GLiNER model, or by constructing from scratch using config + state_dict
    (no ``gliner`` package dependency).
    """

    def __init__(self, rnn_lstm, span_rep_layer, prompt_rep_layer,
                 has_rnn=True):
        super().__init__()
        self.has_rnn = has_rnn
        if has_rnn and rnn_lstm is not None:
            self.rnn = rnn_lstm  # raw nn.LSTM (not LstmSeq2SeqEncoder)
        self.span_rep_layer = span_rep_layer
        self.prompt_rep_layer = prompt_rep_layer

    def forward(self, words_embedding, span_idx, prompts_embedding):
        if self.has_rnn:
            words_embedding, _ = self.rnn(words_embedding)
        span_rep = self.span_rep_layer(words_embedding, span_idx)
        prompts_embedding = self.prompt_rep_layer(prompts_embedding)
        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_embedding)
        return scores


def _build_gliner_core(gliner_model, config):
    """Extract the core sub-modules from a loaded GLiNER model."""
    model = gliner_model
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        model = model.model

    has_rnn = hasattr(model, "rnn")
    rnn_lstm = None
    if has_rnn:
        rnn_module = model.rnn
        if hasattr(rnn_module, "lstm"):
            rnn_lstm = rnn_module.lstm
        elif isinstance(rnn_module, nn.LSTM):
            rnn_lstm = rnn_module
        else:
            raise RuntimeError(
                f"Unexpected RNN module type: {type(rnn_module).__name__}")

    span_rep_layer = getattr(model, "span_rep_layer", None)
    prompt_rep_layer = getattr(model, "prompt_rep_layer", None)

    if span_rep_layer is None or prompt_rep_layer is None:
        raise RuntimeError(
            "Could not locate span_rep_layer / prompt_rep_layer in the "
            f"GLiNER model ({type(model).__name__}).  Available sub-modules: "
            f"{[n for n, _ in model.named_children()]}")

    return _GLiNERCore(rnn_lstm, span_rep_layer, prompt_rep_layer,
                       has_rnn=has_rnn)


def _build_gliner_core_from_config(model_dir, config):
    """Build GLiNER core directly from config + weights (no gliner package).

    Reconstructs only the traceable core modules (BiLSTM, SpanMarkerV0,
    prompt projection) from the saved weights, bypassing the need for
    the ``gliner`` or ``gliner2`` Python packages.
    """
    hidden_size = getattr(config, "hidden_size", 768)
    max_width = getattr(config, "max_width", 12)
    dropout = getattr(config, "dropout", 0.4)
    num_rnn_layers = getattr(config, "num_rnn_layers", 1)
    span_mode = getattr(config, "span_mode", "markerV0")

    if span_mode != "markerV0":
        raise NotImplementedError(
            f"GLiNER span_mode '{span_mode}' not yet supported in "
            f"standalone loader.  Only 'markerV0' is supported.")

    # Build core modules matching GLiNER's parameter layout
    has_rnn = num_rnn_layers > 0
    rnn_lstm = None
    if has_rnn:
        rnn_lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size // 2,
            num_layers=num_rnn_layers, bidirectional=True, batch_first=True,
        )

    # SpanRepLayer -> SpanMarkerV0 (nested as span_rep_layer.span_rep_layer)
    inner_span = _SpanMarkerV0(hidden_size, max_width, dropout)
    span_rep_layer = nn.Module()
    span_rep_layer.add_module("span_rep_layer", inner_span)
    # Make it callable by forwarding
    span_rep_layer.forward = lambda x, idx: inner_span(x, idx)

    prompt_rep_layer = _gliner_projection(hidden_size, dropout)

    core = _GLiNERCore(rnn_lstm, span_rep_layer, prompt_rep_layer,
                       has_rnn=has_rnn)

    # Load weights from model file
    model_dir_path = os.path.join(model_dir) if isinstance(model_dir, str) \
        else str(model_dir)
    safetensors_path = os.path.join(model_dir_path, "model.safetensors")
    bin_path = os.path.join(model_dir_path, "pytorch_model.bin")

    if os.path.isfile(safetensors_path):
        try:
            from safetensors.torch import load_file
            full_state = load_file(safetensors_path)
        except ImportError:
            full_state = torch.load(bin_path, map_location="cpu",
                                    weights_only=True)
    elif os.path.isfile(bin_path):
        full_state = torch.load(bin_path, map_location="cpu",
                                weights_only=True)
    else:
        raise FileNotFoundError(
            f"No model weights found in {model_dir_path} "
            f"(looked for model.safetensors and pytorch_model.bin)")

    # Extract only the keys that belong to our core modules.
    # GLiNER state_dict uses prefixes like:
    #   rnn.lstm.*  -> our rnn.*
    #   span_rep_layer.span_rep_layer.*  -> same
    #   prompt_rep_layer.*  -> same
    core_state = {}
    prefix_map = {
        "rnn.lstm.": "rnn.",           # LstmSeq2SeqEncoder.lstm -> raw LSTM
        "span_rep_layer.": "span_rep_layer.",
        "prompt_rep_layer.": "prompt_rep_layer.",
    }
    for key, value in full_state.items():
        for src_prefix, dst_prefix in prefix_map.items():
            if key.startswith(src_prefix):
                new_key = dst_prefix + key[len(src_prefix):]
                core_state[new_key] = value
                break

    missing, unexpected = core.load_state_dict(core_state, strict=False)
    # Filter out expected missing keys (e.g. if no RNN)
    real_missing = [k for k in missing if not (
        not has_rnn and k.startswith("rnn.")
    )]
    if real_missing:
        raise RuntimeError(
            f"Missing weights for GLiNER core: {real_missing}")

    return core


def convert_model(model_name_or_path, output_dir, formats=None,
                  batch_size=1, seq_len=8, dtype="float32",
                  convert_weights=False, verbose=True,
                  model_name=None):
    """Run the full conversion pipeline.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        output_dir: Output directory for generated files.
        formats: List of output formats ("cpp", "ini", "json", "weights").
                 None means all except weights.
        batch_size: Batch size for INI config.
        seq_len: Sequence length for tracing.
        dtype: Weight dtype ("float32" or "float16").
        convert_weights: Whether to convert and save weights.
        verbose: Print progress messages.
    """
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    from decomposer import AdaptiveConverter
    from emitter_cpp import emit_cpp, emit_cpp_header, emit_cpp_source, get_output_filenames
    from emitter_ini import emit_ini
    from emitter_json import emit_json_string
    from weight_converter import WeightConverter

    if formats is None:
        formats = ["cpp", "ini", "json"]
    if convert_weights and "weights" not in formats:
        formats.append("weights")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load model and config
    if verbose:
        print(f"Loading model: {model_name_or_path}")

    # Try standard AutoConfig first; fall back for custom model types
    # (e.g. GLiNER2's model_type="extractor") that aren't registered in
    # HuggingFace transformers.
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
    except ValueError:
        # Load config.json manually for unregistered model types
        config_path = os.path.join(model_name_or_path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config_dict = json.load(f)
            # Create a simple namespace so attribute access works
            config = argparse.Namespace(**config_dict)
            if verbose:
                print(f"  Loaded custom config (model_type="
                      f"{getattr(config, 'model_type', 'unknown')})")
        else:
            raise

    model_type = getattr(config, "model_type", "")

    # Choose model class based on architecture
    causal_types = {
        "qwen3", "qwen2", "llama", "mistral", "gpt2", "gpt_neo", "gpt_neox",
        "phi", "gemma", "gemma2", "gemma3_text", "starcoder2", "codegen",
    }
    encoder_decoder_types = {
        "t5", "mt5", "bart", "mbart", "pegasus", "marian",
    }
    # Custom model types that require their own loading logic
    custom_model_types = {
        "extractor",  # GLiNER / GLiNER2
    }

    # Detect if this is an embedding/feature-extraction model by checking
    # the architectures field in the config (e.g. ["GemmaModel"] vs
    # ["GemmaForCausalLM"])
    architectures = getattr(config, "architectures", []) or []
    is_embedding_model = any(
        not arch.endswith(("ForCausalLM", "ForConditionalGeneration",
                           "ForSeq2SeqLM", "ForMaskedLM",
                           "ForSequenceClassification",
                           "ForTokenClassification"))
        for arch in architectures
    ) if architectures else False

    is_causal = model_type in causal_types and not is_embedding_model
    is_encoder_decoder = model_type in encoder_decoder_types
    is_custom = model_type in custom_model_types

    if is_custom and model_type == "extractor":
        # GLiNER/GLiNER2: we only need the core computation graph
        # (BiLSTM + SpanRep + Projection + Einsum).  Try loading via
        # the gliner packages first; fall back to standalone loading
        # from config + weights (no package dependency).
        gliner_config = config  # already loaded from config.json
        # Also try to load gliner_config.json which has span_mode etc.
        gliner_cfg_path = os.path.join(model_name_or_path,
                                       "gliner_config.json")
        if os.path.isfile(gliner_cfg_path):
            with open(gliner_cfg_path) as f:
                gcfg = json.load(f)
            # Merge gliner-specific fields into config
            for k, v in gcfg.items():
                if not hasattr(gliner_config, k):
                    setattr(gliner_config, k, v)
            if verbose:
                print(f"  Loaded gliner_config.json "
                      f"(span_mode={gcfg.get('span_mode', 'unknown')})")

        loaded_via_package = False
        # Try gliner2 package
        try:
            from gliner2 import Extractor, ExtractorConfig
            ec = ExtractorConfig.from_pretrained(model_name_or_path)
            full_model = Extractor.from_pretrained(model_name_or_path)
            config = ec
            model = _build_gliner_core(full_model, config)
            loaded_via_package = True
        except ImportError:
            pass

        # Try gliner package
        if not loaded_via_package:
            try:
                from gliner.model import GLiNER
                full_model = GLiNER.from_pretrained(model_name_or_path)
                config = full_model.config
                model = _build_gliner_core(full_model, config)
                loaded_via_package = True
            except ImportError:
                pass

        # Standalone: build core from config + weights (no package needed)
        if not loaded_via_package:
            if verbose:
                print("  No gliner package found; loading core from "
                      "config + weights directly")
            model = _build_gliner_core_from_config(
                model_name_or_path, gliner_config)
            config = gliner_config

        model.eval()
    elif is_causal:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float32)
        model.eval()
    else:
        model = AutoModel.from_pretrained(
            model_name_or_path, torch_dtype=torch.float32)
        model.eval()

    if verbose:
        print(f"Model type: {model_type}, Parameters: "
              f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Step 2: Prepare trace inputs
    if is_custom and model_type == "extractor":
        # GLiNER/GLiNER2: trace the core computation graph.
        hidden_size = getattr(config, "hidden_size", 768)
        max_width = getattr(config, "max_width", 12)
        num_classes = 5  # dummy entity type count
        W = seq_len  # word-level sequence length
        input_kwargs = {
            "words_embedding": torch.randn(1, W, hidden_size),
            "span_idx": torch.randint(0, W, (1, W * max_width, 2)),
            "prompts_embedding": torch.randn(1, num_classes, hidden_size),
        }
        if verbose:
            print(f"  GLiNER core: hidden={hidden_size}, "
                  f"max_width={max_width}")
    else:
        vocab_size = getattr(config, "vocab_size", 30000)
        input_kwargs = {"input_ids": torch.randint(0, vocab_size, (1, seq_len))}
        if not is_causal and not is_encoder_decoder:
            input_kwargs["attention_mask"] = torch.ones(1, seq_len, dtype=torch.long)
        if is_encoder_decoder:
            input_kwargs["decoder_input_ids"] = torch.randint(
                0, min(vocab_size, 1000), (1, max(1, seq_len // 2)))

    # Step 3: Run conversion pipeline
    if verbose:
        print("Running conversion pipeline...")

    converter = AdaptiveConverter(model, config)
    result = converter.convert(input_kwargs)

    layers = result.layers
    structure = result.model_structure

    if verbose:
        print(f"Converted: {len(layers)} layers, "
              f"{len(result.lazy_chains)} LazyTensor chains")
        if structure:
            print(f"Architecture: {structure.arch_type} ({structure.model_type})")
            print(f"Blocks: {structure.num_layers}")

    # Step 4: Emit outputs
    outputs = {}

    # Derive filenames: use explicit model_name if provided, otherwise fall
    # back to the model ID (last path component) so that e.g.
    # "KaLM-embedding-v2.5" produces "kalm_embedding_v2_5.cpp".
    effective_name = model_name or model_name_or_path
    filenames = get_output_filenames(
        structure.model_type if structure else model_type,
        structure.arch_type if structure else "decoder_only",
        model_name=effective_name,
    )

    if "cpp" in formats:
        header_code = emit_cpp_header(layers, structure, model_name=effective_name)
        source_code = emit_cpp_source(layers, structure, model_name=effective_name)
        header_path = os.path.join(output_dir, filenames["header"])
        source_path = os.path.join(output_dir, filenames["source"])
        with open(header_path, "w") as f:
            f.write(header_code)
        with open(source_path, "w") as f:
            f.write(source_code)
        outputs["cpp_header"] = header_path
        outputs["cpp_source"] = source_path
        if verbose:
            print(f"  C++ header: {header_path} ({len(header_code)} bytes)")
            print(f"  C++ source: {source_path} ({len(source_code)} bytes)")

    if "ini" in formats:
        ini_text = emit_ini(layers, structure, batch_size=batch_size,
                            mode="structured")
        ini_path = os.path.join(output_dir, filenames["ini"])
        with open(ini_path, "w") as f:
            f.write(ini_text)
        outputs["ini"] = ini_path
        if verbose:
            print(f"  INI config: {ini_path} ({len(ini_text)} bytes)")

    if "json" in formats:
        json_str = emit_json_string(layers, structure, indent=2)
        json_path = os.path.join(output_dir, filenames["json"])
        with open(json_path, "w") as f:
            f.write(json_str)
        outputs["json"] = json_path
        if verbose:
            print(f"  JSON config: {json_path} ({len(json_str)} bytes)")

    if "weights" in formats:
        wc = WeightConverter(layers)
        state_dict = model.state_dict()
        bin_path = os.path.join(output_dir, "model.bin")
        wc.convert(state_dict, bin_path, dtype=dtype)
        outputs["weights"] = bin_path
        if verbose:
            size_mb = os.path.getsize(bin_path) / 1024 / 1024
            print(f"  Weights: {bin_path} ({size_mb:.1f} MB)")

        # Also save standalone weight conversion script
        script = wc.generate_script()
        script_path = os.path.join(output_dir, "convert_weights.py")
        with open(script_path, "w") as f:
            f.write(script)
        outputs["weight_script"] = script_path
        if verbose:
            print(f"  Weight script: {script_path}")

    if verbose:
        print(f"\nConversion complete! Output: {output_dir}")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to NNTrainer format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model Qwen/Qwen3-0.6B --output ./qwen3/
  %(prog)s --model bert-base-uncased --output ./bert/ --format ini json
  %(prog)s --model google/mt5-small --output ./mt5/ --format all
  %(prog)s --model ./local_model/ --output ./out/ --weights --dtype float16
""")

    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True,
                        help="Output directory")
    parser.add_argument("--format", nargs="+",
                        choices=["cpp", "ini", "json", "all"],
                        default=["all"],
                        help="Output formats (default: all)")
    parser.add_argument("--weights", action="store_true",
                        help="Convert and save model weights")
    parser.add_argument("--dtype", choices=["float32", "float16"],
                        default="float32",
                        help="Weight dtype (default: float32)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for INI config (default: 1)")
    parser.add_argument("--seq-len", type=int, default=8,
                        help="Sequence length for tracing (default: 8)")
    parser.add_argument("--model-name", default=None,
                        help="Override output file naming (default: derived "
                             "from --model). e.g. --model-name KaLM-embedding")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress messages")

    args = parser.parse_args()

    formats = args.format
    if "all" in formats:
        formats = ["cpp", "ini", "json"]

    convert_model(
        model_name_or_path=args.model,
        output_dir=args.output,
        formats=formats,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dtype=args.dtype,
        convert_weights=args.weights,
        verbose=not args.quiet,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()

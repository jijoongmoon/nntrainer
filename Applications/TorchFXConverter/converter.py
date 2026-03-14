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


class _GLiNERCore(nn.Module):
    """Traceable core of a GLiNER/GLiNER2 model (post-preprocessing).

    Wraps the BiLSTM, SpanRepLayer, prompt projection, and einsum scoring
    sub-modules extracted from a full GLiNER model.  The DeBERTa encoder
    and the data-dependent ``_extract_prompt_features_and_word_embeddings``
    step are intentionally excluded because they require dynamic indexing
    that cannot be expressed in a static NNTrainer graph.

    Note: The original ``LstmSeq2SeqEncoder`` uses ``pack_padded_sequence``
    which is not traceable by torch.fx.  We extract the raw ``nn.LSTM``
    instead and call it directly (without packing).
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
    """Extract the core computation sub-modules from a GLiNER model."""
    # The actual nn.Module lives at different levels depending on the
    # GLiNER wrapper.  Try common locations.
    model = gliner_model
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        model = model.model  # GLiNER wrapper -> inner BaseModel

    # Extract the raw nn.LSTM from LstmSeq2SeqEncoder (which uses
    # pack_padded_sequence that can't be traced by torch.fx).
    has_rnn = hasattr(model, "rnn")
    rnn_lstm = None
    if has_rnn:
        rnn_module = model.rnn
        # LstmSeq2SeqEncoder wraps nn.LSTM as self.lstm
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
        # GLiNER2 / GLiNER models use a custom Extractor class
        try:
            from gliner2 import Extractor, ExtractorConfig
            gliner_config = ExtractorConfig.from_pretrained(model_name_or_path)
            model = Extractor.from_pretrained(model_name_or_path)
            config = gliner_config
        except ImportError:
            try:
                from gliner.model import GLiNER
                model = GLiNER.from_pretrained(model_name_or_path)
                config = model.config
            except ImportError:
                raise ImportError(
                    "GLiNER2 model detected (model_type='extractor') but "
                    "neither 'gliner2' nor 'gliner' package is installed. "
                    "Install with: pip install gliner2")
    elif is_causal:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float32)
    else:
        model = AutoModel.from_pretrained(
            model_name_or_path, torch_dtype=torch.float32)
    model.eval()

    if verbose:
        print(f"Model type: {model_type}, Parameters: "
              f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Step 2: Prepare trace inputs
    if is_custom and model_type == "extractor":
        # GLiNER/GLiNER2: trace only the core computation graph
        # (post-preprocessing).  The preprocessing step uses
        # data-dependent indexing and is handled outside NNTrainer.
        hidden_size = getattr(config, "hidden_size", 768)
        max_width = getattr(config, "max_width", 12)
        num_classes = 5  # dummy entity type count
        W = seq_len  # word-level sequence length
        input_kwargs = {
            "words_embedding": torch.randn(1, W, hidden_size),
            "span_idx": torch.randint(0, W, (1, W * max_width, 2)),
            "prompts_embedding": torch.randn(1, num_classes, hidden_size),
        }
        # Extract the core sub-modules (rnn + span_rep + prompt_rep)
        # into a standalone nn.Module for tracing.
        core_model = _build_gliner_core(model, config)
        core_model.eval()
        model = core_model
        if verbose:
            print(f"  GLiNER core extracted: hidden={hidden_size}, "
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

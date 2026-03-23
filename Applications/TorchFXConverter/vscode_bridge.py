#!/usr/bin/env python3
"""VS Code bridge for TorchFXConverter.
Runs conversion and outputs full result as JSON for the visualizer.
Supports both HuggingFace models and local .py files with nn.Module classes."""

import sys
import os
import json
import argparse
import importlib.util
import inspect

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decomposer import AdaptiveConverter
from emitter_ini import emit_ini
from emitter_cpp import emit_cpp_source, emit_cpp_header
from nntrainer_layers import NNTrainerLayerDef


def serialize_fx_graph(graph):
    """Serialize torch.fx Graph to list of node dicts."""
    nodes = []
    for node in graph.nodes:
        entry = {
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "args": [str(a) for a in node.args if hasattr(a, 'name')],
            "output_shape": node.meta.get("output_shape"),
            "module_type": node.meta.get("module_type"),
            "scope": node.meta.get("scope", ""),
            "meta": {}
        }
        safe_keys = [
            "leaf_module", "has_weight", "has_bias",
            "in_features", "out_features", "num_embeddings", "embedding_dim",
            "normalized_shape", "eps", "hidden_size",
            "is_rmsnorm", "is_rnn_module"
        ]
        for key in safe_keys:
            if key in node.meta:
                val = node.meta[key]
                try:
                    json.dumps(val)
                    entry["meta"][key] = val
                except (TypeError, ValueError):
                    entry["meta"][key] = str(val)
        nodes.append(entry)
    return nodes


def serialize_layer(layer):
    """Serialize NNTrainerLayerDef to dict."""
    props = {}
    for k, v in layer.properties.items():
        try:
            json.dumps(v)
            props[k] = v
        except (TypeError, ValueError):
            props[k] = str(v)

    return {
        "name": layer.name,
        "layer_type": layer.layer_type,
        "properties": props,
        "input_layers": list(layer.input_layers),
        "fx_node_name": layer.fx_node_name,
        "hf_module_name": layer.hf_module_name,
        "hf_module_type": layer.hf_module_type,
        "has_weight": layer.has_weight,
        "has_bias": layer.has_bias,
        "weight_hf_key": layer.weight_hf_key,
        "bias_hf_key": layer.bias_hf_key,
        "transpose_weight": layer.transpose_weight,
        "shared_from": layer.shared_from,
    }


def build_node_mapping(layers, fx_graph, collapsed_rope_layers=None):
    """Build mapping between fx nodes and nntrainer layers."""
    mappings = []
    fx_node_names = {node.name for node in fx_graph.nodes}
    mapped_fx = set()
    rope_names = collapsed_rope_layers or set()

    for layer in layers:
        if layer.fx_node_name and layer.fx_node_name in fx_node_names:
            mappings.append({
                "fxNodeName": layer.fx_node_name,
                "nntrainerLayerName": layer.name,
                "hfModuleName": layer.hf_module_name,
                "mappingType": "direct"
            })
            mapped_fx.add(layer.fx_node_name)

    for node in fx_graph.nodes:
        if node.name not in mapped_fx and node.op not in ("placeholder", "output"):
            # Mark RoPE nodes as collapsed (handled by mha_core).
            # Check both the explicit set AND scope-based detection
            # (noop-removed rotary nodes like float/expand/to won't be
            # in rope_names since they were removed before RoPE collapse).
            is_rope = node.name in rope_names
            if not is_rope:
                is_rope = ("rotary_emb" in node.name.lower()
                           or "rotary_embedding" in node.name.lower())
            mappings.append({
                "fxNodeName": node.name,
                "nntrainerLayerName": "",
                "hfModuleName": "",
                "mappingType": "collapsed_rope" if is_rope else "skipped"
            })

    return mappings


def serialize_structure(structure):
    """Serialize ModelStructure to dict."""
    if structure is None:
        return None

    blocks = []
    for b in (structure.blocks or []):
        block = {
            "block_idx": b.block_idx,
            "pre_attn_norm": b.pre_attn_norm,
            "attn_residual": b.attn_residual,
            "pre_ffn_norm": b.pre_ffn_norm,
            "ffn_residual": b.ffn_residual,
            "norm_type": b.norm_type,
            "attention": None,
            "ffn": None,
        }
        if b.attention:
            a = b.attention
            block["attention"] = {
                "block_idx": a.block_idx,
                "q_proj": a.q_proj,
                "k_proj": a.k_proj,
                "v_proj": a.v_proj,
                "o_proj": a.o_proj,
                "attention_type": a.attention_type,
                "has_rope": a.has_rope,
                "num_heads": getattr(a, "num_heads", 0),
                "num_kv_heads": getattr(a, "num_kv_heads", 0),
                "head_dim": getattr(a, "head_dim", 0),
            }
        if b.ffn:
            f = b.ffn
            block["ffn"] = {
                "block_idx": f.block_idx,
                "ffn_type": f.ffn_type,
                "gate_proj": getattr(f, "gate_proj", ""),
                "up_proj": getattr(f, "up_proj", ""),
                "down_proj": getattr(f, "down_proj", ""),
                "intermediate_size": getattr(f, "intermediate_size", 0),
            }
        blocks.append(block)

    return {
        "model_type": structure.model_type,
        "arch_type": structure.arch_type,
        "vocab_size": structure.vocab_size,
        "hidden_size": structure.hidden_size,
        "num_layers": structure.num_layers,
        "num_heads": structure.num_heads,
        "num_kv_heads": structure.num_kv_heads,
        "head_dim": structure.head_dim,
        "intermediate_size": structure.intermediate_size,
        "norm_eps": structure.norm_eps,
        "tie_word_embeddings": structure.tie_word_embeddings,
        "rope_theta": structure.rope_theta,
        "embedding": structure.embedding,
        "final_norm": structure.final_norm,
        "lm_head": structure.lm_head,
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# Local .py model loading
# ---------------------------------------------------------------------------

def load_local_model(py_path, class_name):
    """Dynamically load an nn.Module class from a local .py file."""
    module_dir = os.path.dirname(os.path.abspath(py_path))
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    module_name = os.path.splitext(os.path.basename(py_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, class_name):
        # Try to find nn.Module subclasses if exact name not found
        candidates = [
            name for name, obj in inspect.getmembers(mod)
            if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module
        ]
        raise ValueError(
            f"Class '{class_name}' not found in {py_path}. "
            f"Available nn.Module classes: {candidates}"
        )

    cls = getattr(mod, class_name)
    if not (inspect.isclass(cls) and issubclass(cls, nn.Module)):
        raise ValueError(f"'{class_name}' is not an nn.Module subclass")

    return cls, mod


def infer_constructor_args(cls):
    """Try to infer reasonable default constructor arguments."""
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is not inspect.Parameter.empty:
            kwargs[name] = param.default
        else:
            # Guess common parameter names
            guesses = {
                "vocab_size": 1000, "num_embeddings": 1000,
                "hidden_size": 256, "d_model": 256, "embed_dim": 256,
                "num_heads": 4, "nhead": 4, "num_attention_heads": 4,
                "num_layers": 2, "num_hidden_layers": 2,
                "intermediate_size": 512, "dim_feedforward": 512,
                "num_classes": 10, "output_size": 10,
                "input_size": 256, "in_features": 256, "in_channels": 3,
                "out_features": 256, "out_channels": 64,
                "kernel_size": 3, "dropout": 0.0,
                "max_seq_len": 512, "max_position_embeddings": 512,
                "bias": True, "batch_first": True,
            }
            if name in guesses:
                kwargs[name] = guesses[name]
            else:
                raise ValueError(
                    f"Cannot infer constructor arg '{name}' for {cls.__name__}. "
                    f"Provide --input-desc with constructor args."
                )
    return kwargs


def build_trace_inputs(model, input_desc, seq_len):
    """Build trace input tensors from user description or by inference."""
    if input_desc:
        desc = json.loads(input_desc)

        # Check if desc has constructor args vs input shapes
        if "trace_inputs" in desc:
            inputs = {}
            for key, shape in desc["trace_inputs"].items():
                if "int" in key or "ids" in key or "mask" in key:
                    inputs[key] = torch.randint(0, 100, shape)
                else:
                    inputs[key] = torch.randn(*shape)
            return inputs

        # Assume desc is {name: shape} for trace inputs
        inputs = {}
        for key, shape in desc.items():
            if "int" in key or "ids" in key or "mask" in key:
                inputs[key] = torch.randint(0, 100, shape)
            else:
                inputs[key] = torch.randn(*shape)
        return inputs

    # Auto-detect from forward() signature
    sig = inspect.signature(model.forward)
    inputs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in ("input_ids", "src", "tgt"):
            inputs[name] = torch.randint(0, 100, (1, seq_len))
        elif name in ("attention_mask", "src_mask", "mask"):
            inputs[name] = torch.ones(1, seq_len, dtype=torch.long)
        elif name in ("x", "input", "inputs"):
            # Try to guess from first layer
            first_param = next(model.parameters(), None)
            if first_param is not None:
                in_dim = first_param.shape[-1] if first_param.dim() >= 2 else first_param.shape[0]
                inputs[name] = torch.randn(1, seq_len, in_dim)
            else:
                inputs[name] = torch.randn(1, seq_len, 256)
        elif param.default is inspect.Parameter.empty:
            inputs[name] = torch.randn(1, seq_len, 256)
        # Skip optional params with defaults

    if not inputs:
        inputs = {"x": torch.randn(1, seq_len, 256)}

    return inputs


def build_dummy_config(model, class_name):
    """Build a minimal config namespace for the converter."""
    config = argparse.Namespace()
    config.model_type = class_name.lower()
    config.architectures = [class_name]

    # Try to extract common attributes from the model
    for attr in ["vocab_size", "hidden_size", "num_heads", "num_layers",
                 "num_attention_heads", "num_hidden_layers", "intermediate_size",
                 "max_position_embeddings"]:
        if hasattr(model, attr):
            setattr(config, attr, getattr(model, attr))
        elif hasattr(model, "config") and hasattr(model.config, attr):
            setattr(config, attr, getattr(model.config, attr))

    if not hasattr(config, "vocab_size"):
        # Estimate from embedding layer
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                config.vocab_size = m.num_embeddings
                break
        else:
            config.vocab_size = 1000

    return config


def main():
    parser = argparse.ArgumentParser(description="VS Code bridge for TorchFXConverter")
    # HuggingFace model mode
    parser.add_argument("--model", default=None, help="HuggingFace model ID or local path")
    # Local .py file mode
    parser.add_argument("--local-model", default=None, help="Path to local .py file")
    parser.add_argument("--class-name", default=None, help="nn.Module class name in the .py file")
    parser.add_argument("--input-desc", default=None, help="JSON describing input shapes or constructor args")
    # Common options
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length")
    parser.add_argument("--format", nargs="+", default=["all"], help="Output formats")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--model-name", default=None, help="Override model name")
    parser.add_argument("--plugin-config", default=None, help="Plugin config path")
    args = parser.parse_args()

    torch_source_code = ""
    torch_source_path = ""

    if args.local_model:
        # ---- Local .py model mode ----
        print(f"PROGRESS: Loading local model from {args.local_model}...", flush=True)
        torch_source_path = os.path.abspath(args.local_model)
        with open(torch_source_path) as f:
            torch_source_code = f.read()

        cls, mod = load_local_model(args.local_model, args.class_name)

        # Parse input-desc for possible constructor args
        constructor_kwargs = {}
        input_desc_for_trace = args.input_desc
        if args.input_desc:
            try:
                desc = json.loads(args.input_desc)
                if "constructor_args" in desc:
                    constructor_kwargs = desc["constructor_args"]
                    input_desc_for_trace = json.dumps(desc.get("trace_inputs", {})) if "trace_inputs" in desc else None
            except json.JSONDecodeError:
                pass

        if constructor_kwargs:
            model = cls(**constructor_kwargs)
        else:
            try:
                model = cls()
            except TypeError:
                print("PROGRESS: Inferring constructor arguments...", flush=True)
                constructor_kwargs = infer_constructor_args(cls)
                model = cls(**constructor_kwargs)

        model.eval()
        print(f"PROGRESS: Model loaded: {args.class_name} "
              f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)", flush=True)

        config = build_dummy_config(model, args.class_name)
        input_kwargs = build_trace_inputs(model, input_desc_for_trace, args.seq_len)
        model_name = args.model_name or args.class_name

    elif args.model:
        # ---- HuggingFace model mode ----
        # Reuse converter.py's full model loading logic (handles config-only,
        # custom models, causal vs encoder-decoder, etc.)
        from converter import convert_model
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        from custom_models import CUSTOM_LOADERS, load_custom_model

        print("PROGRESS: Loading HuggingFace model...", flush=True)

        try:
            config = AutoConfig.from_pretrained(args.model)
        except ValueError:
            config_path = os.path.join(args.model, "config.json")
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    config_dict = json.load(f)
                config = argparse.Namespace(**config_dict)
            else:
                raise

        model_type = getattr(config, "model_type", "")
        causal_types = {
            "qwen3", "qwen2", "llama", "mistral", "gpt2", "gpt_neo", "gpt_neox",
            "phi", "gemma", "gemma2", "gemma3_text", "gemma3n_text", "starcoder2",
            "codegen", "lfm2", "granitemoehybrid", "mamba", "mamba2",
        }
        encoder_decoder_types = {
            "t5", "mt5", "bart", "mbart", "pegasus", "marian",
        }
        is_custom = model_type in CUSTOM_LOADERS
        is_causal = model_type in causal_types
        is_encoder_decoder = model_type in encoder_decoder_types

        if is_custom:
            model, config, input_kwargs = load_custom_model(
                model_type, args.model, config, args.seq_len, verbose=True)
        else:
            # Try from_pretrained first; fall back to from_config for weight-less dirs
            loaded = False
            if is_causal:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model, dtype=torch.float32)
                    loaded = True
                except (OSError, ValueError):
                    pass
                if not loaded:
                    model = AutoModelForCausalLM.from_config(config)
            else:
                try:
                    model = AutoModel.from_pretrained(
                        args.model, dtype=torch.float32)
                    loaded = True
                except (OSError, ValueError):
                    pass
                if not loaded:
                    model = AutoModel.from_config(config)
            model.eval()

            vocab_size = getattr(config, "vocab_size", 30000)
            input_kwargs = {"input_ids": torch.randint(0, vocab_size, (1, args.seq_len))}
            if not is_causal and not is_encoder_decoder:
                input_kwargs["attention_mask"] = torch.ones(1, args.seq_len, dtype=torch.long)
            if is_encoder_decoder:
                input_kwargs["decoder_input_ids"] = torch.randint(
                    0, min(vocab_size, 1000), (1, max(1, args.seq_len // 2)))

        model_name = args.model_name or args.model.replace("/", "_").replace("-", "_")

    else:
        parser.error("Either --model or --local-model is required")

    print("PROGRESS: Running adaptive conversion...", flush=True)
    converter = AdaptiveConverter(model, config)
    result = converter.convert(input_kwargs)

    layers = result.layers
    structure = result.model_structure

    print("PROGRESS: Generating outputs...", flush=True)

    cpp_source = ""
    ini_config = ""
    try:
        cpp_source = emit_cpp_source(layers, structure, model_name=model_name)
    except Exception as e:
        print(f"Warning: C++ emission failed: {e}", file=sys.stderr)
    try:
        ini_config = emit_ini(layers, structure, batch_size=args.batch_size)
    except Exception as e:
        print(f"Warning: INI emission failed: {e}", file=sys.stderr)

    weight_map = []
    for layer in layers:
        if layer.has_weight and layer.weight_hf_key:
            weight_map.append({
                "layer_name": layer.name,
                "layer_type": layer.layer_type,
                "weight_key": layer.weight_hf_key,
                "transpose_weight": layer.transpose_weight,
            })

    print("PROGRESS: Serializing conversion result...", flush=True)

    output = {
        "nntrainerLayers": [serialize_layer(l) for l in layers],
        "fxGraph": serialize_fx_graph(result.graph),
        "modelStructure": serialize_structure(structure),
        "weightMap": weight_map,
        "nodeMapping": build_node_mapping(layers, result.graph,
                                          result.collapsed_rope_layers),
        "unsupportedOps": [l.name for l in result.unsupported_ops],
        "unknownLayers": [l.name for l in result.unknown_layers],
        "decomposedModules": list(result.decomposed_module_types),
        "cppSource": cpp_source,
        "iniConfig": ini_config,
        "torchSourceCode": torch_source_code,
        "torchSourcePath": torch_source_path,
    }

    os.makedirs(args.output, exist_ok=True)
    result_path = os.path.join(args.output, "conversion_result.json")
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"PROGRESS: Done. Result saved to {result_path}", flush=True)


if __name__ == "__main__":
    main()

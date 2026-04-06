## @file weight_converter.py
## @brief weight conversion script for qwen3-0.6b model
## @note  Supports both .bin and .safetensors output formats.
##        For .bin: weights are saved in C++ CausalLM layer creation order.
##        For .safetensors: weights are saved with CausalLM layer names
##        for name-based loading (order-independent).
##
## C++ Qwen3 layer creation order (per decoder block):
##   attention_norm -> V -> K -> K_norm -> Q -> Q_norm -> O
##   -> ffn_norm -> ffn_up -> ffn_gate -> ffn_down

import argparse
import json
import struct
import torch
import numpy as np
from collections import OrderedDict
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# Weight order mapping: HF key -> (CausalLM name, transform)
# =============================================================================

def build_weight_entries(params, n_layers):
    """Build ordered list of (hf_key, causallm_name, transform) tuples.

    The order matches the C++ Qwen3CausalLM layer creation order:
      embedding0
      For each layer i:
        layer{i}_attention_norm   (input_layernorm)
        layer{i}_wv               (self_attn.v_proj, transposed)
        layer{i}_wk               (self_attn.k_proj, transposed)
        layer{i}_k_norm           (self_attn.k_norm)
        layer{i}_wq               (self_attn.q_proj, transposed)
        layer{i}_q_norm           (self_attn.q_norm)
        layer{i}_attention_out    (self_attn.o_proj, transposed)
        layer{i}_ffn_norm         (post_attention_layernorm)
        layer{i}_ffn_up           (mlp.up_proj, transposed)
        layer{i}_ffn_gate         (mlp.gate_proj, transposed)
        layer{i}_ffn_down         (mlp.down_proj, transposed)
      output_norm                 (model.norm)
      output_of_causallm          (lm_head, transposed, only if not tied)
    """
    entries = []

    # Embedding
    entries.append(("model.embed_tokens.weight", "embedding0", "none"))

    for i in range(n_layers):
        lp = f"model.layers.{i}."

        # Attention norm
        entries.append((f"{lp}input_layernorm.weight",
                        f"layer{i}_attention_norm", "none"))

        # V projection
        entries.append((f"{lp}self_attn.v_proj.weight",
                        f"layer{i}_wv", "transpose"))

        # K projection
        entries.append((f"{lp}self_attn.k_proj.weight",
                        f"layer{i}_wk", "transpose"))

        # K norm
        k_norm_key = f"{lp}self_attn.k_norm.weight"
        if k_norm_key in params:
            entries.append((k_norm_key, f"layer{i}_k_norm", "none"))

        # Q projection
        entries.append((f"{lp}self_attn.q_proj.weight",
                        f"layer{i}_wq", "transpose"))

        # Q norm
        q_norm_key = f"{lp}self_attn.q_norm.weight"
        if q_norm_key in params:
            entries.append((q_norm_key, f"layer{i}_q_norm", "none"))

        # O projection
        entries.append((f"{lp}self_attn.o_proj.weight",
                        f"layer{i}_attention_out", "transpose"))

        # FFN norm
        entries.append((f"{lp}post_attention_layernorm.weight",
                        f"layer{i}_ffn_norm", "none"))

        # FFN up
        entries.append((f"{lp}mlp.up_proj.weight",
                        f"layer{i}_ffn_up", "transpose"))

        # FFN gate
        entries.append((f"{lp}mlp.gate_proj.weight",
                        f"layer{i}_ffn_gate", "transpose"))

        # FFN down
        entries.append((f"{lp}mlp.down_proj.weight",
                        f"layer{i}_ffn_down", "transpose"))

    # Output norm
    entries.append(("model.norm.weight", "output_norm", "none"))

    # LM head (only if not tied with embedding)
    if "lm_head.weight" in params:
        entries.append(("lm_head.weight", "output_of_causallm", "transpose"))

    return entries


def get_tensor(params, hf_key, transform, dtype):
    """Get a tensor from state_dict with optional transformation."""
    tensor = params[hf_key]
    if dtype is not None:
        tensor = tensor.to(dtype) if isinstance(dtype, torch.dtype) else tensor
    if transform == "transpose" and tensor.dim() == 2:
        tensor = tensor.t().contiguous()
    return tensor


# =============================================================================
# Binary format converter
# =============================================================================

def save_bin(params, entries, dtype, output_path):
    """Save weights in binary format matching C++ layer creation order."""
    np_dtype = dtype if isinstance(dtype, str) else "float32"
    with open(output_path, "wb") as f:
        for hf_key, cl_name, transform in entries:
            tensor = get_tensor(params, hf_key, transform, None)
            np.array(tensor.cpu().numpy(), dtype=np_dtype).tofile(f)
    print(f"Saved {len(entries)} weight tensors to {output_path} (binary)")


# =============================================================================
# Safetensors format converter
# =============================================================================

def save_safetensors(params, entries, torch_dtype, dtype_str, output_path):
    """Save weights in safetensors format with CausalLM layer names."""
    sf_dtype_map = {
        "float32": "F32", "float16": "F16", "bfloat16": "BF16",
    }
    sf_dtype = sf_dtype_map.get(dtype_str, "F32")

    tensor_data = []
    header_entries = OrderedDict()
    data_offset = 0

    for hf_key, cl_name, transform in entries:
        tensor = get_tensor(params, hf_key, transform, torch_dtype)
        data = tensor.cpu().numpy().tobytes()
        data_size = len(data)

        # Use CausalLM layer name as safetensors tensor name
        tensor_name = cl_name + ":weight"
        shape = list(tensor.shape)

        header_entries[tensor_name] = {
            "dtype": sf_dtype,
            "shape": shape,
            "data_offsets": [data_offset, data_offset + data_size],
        }

        tensor_data.append(data)
        data_offset += data_size

    # Build JSON header
    header_dict = {"__metadata__": {"format": "nntrainer"}}
    header_dict.update(header_entries)
    header_json = json.dumps(header_dict, separators=(",", ":"),
                             ensure_ascii=True)

    # Pad to 8-byte alignment
    header_bytes = header_json.encode("utf-8")
    pad_len = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad_len

    # Write file
    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for data in tensor_data:
            f.write(data)

    print(f"Saved {len(entries)} weight tensors to {output_path} (safetensors)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-0.6B weights for NNTrainer CausalLM")
    parser.add_argument("--model_path", type=str, default="./Qwen3-0.6B",
                        help="HuggingFace model path or ID")
    parser.add_argument("--output_name", type=str,
                        default="./nntr_qwen3_0.6b_fp32.safetensors",
                        help="Output file path (.bin or .safetensors)")
    parser.add_argument("--data_type", type=str, default="float32",
                        choices=["float32", "float16"],
                        help="Target data type")
    args = parser.parse_args()

    model_path = args.model_path
    output_name = args.output_name
    data_dtype = args.data_type

    print(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()

    params = model.state_dict()
    entries = build_weight_entries(params, config.num_hidden_layers)

    print(f"\nModel: {model_path}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {config.num_hidden_layers}")
    print(f"  num_heads: {config.num_attention_heads}")
    print(f"  num_kv_heads: {config.num_key_value_heads}")
    print(f"  tie_word_embeddings: {config.tie_word_embeddings}")
    print(f"  Total weight entries: {len(entries)}")
    print(f"  Output: {output_name} ({data_dtype})")
    print()

    # Print weight mapping for verification
    print("Weight mapping (C++ order):")
    print(f"  {'HF Key':<55} {'CausalLM Name':<25} {'Transform'}")
    print("  " + "-" * 90)
    for hf_key, cl_name, transform in entries[:15]:
        print(f"  {hf_key:<55} {cl_name:<25} {transform}")
    if len(entries) > 15:
        print(f"  ... ({len(entries) - 15} more entries)")
    print()

    torch_dtype = torch.float32 if data_dtype == "float32" else torch.float16

    if output_name.endswith(".safetensors"):
        save_safetensors(params, entries, torch_dtype, data_dtype, output_name)
    else:
        save_bin(params, entries, data_dtype, output_name)

    print("Done!")

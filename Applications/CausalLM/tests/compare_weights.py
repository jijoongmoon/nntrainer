#!/usr/bin/env python3
"""Compare weight values between HF model and generated safetensors file.

Usage:
    python compare_weights.py --model_path /path/to/Qwen3-0.6B \
                              --safetensors /path/to/converted.safetensors
"""
import argparse
import json
import struct
import numpy as np


def read_safetensors(path):
    """Read all tensors from safetensors file."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data_start = 8 + header_size
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            off = info["data_offsets"]
            f.seek(data_start + off[0])
            raw = f.read(off[1] - off[0])
            shape = info["shape"]
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            tensors[name] = arr
    return tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--safetensors", required=True)
    args = parser.parse_args()

    # Load safetensors
    sf_tensors = read_safetensors(args.safetensors)
    print(f"Safetensors: {len(sf_tensors)} tensors from {args.safetensors}")

    # Load HF model
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    sd = model.state_dict()

    # Build expected mapping: safetensors_name -> (hf_key, needs_transpose)
    mapping = {}
    mapping["embedding0:weight"] = ("model.embed_tokens.weight", False)
    mapping["output_norm:weight"] = ("model.norm.weight", False)

    for i in range(config.num_hidden_layers):
        lp = f"model.layers.{i}."
        p = f"layer{i}_"
        mapping[f"{p}attention_norm:weight"] = (f"{lp}input_layernorm.weight", False)
        mapping[f"{p}wq:weight"] = (f"{lp}self_attn.q_proj.weight", True)
        mapping[f"{p}wk:weight"] = (f"{lp}self_attn.k_proj.weight", True)
        mapping[f"{p}wv:weight"] = (f"{lp}self_attn.v_proj.weight", True)
        mapping[f"{p}q_norm:weight"] = (f"{lp}self_attn.q_norm.weight", False)
        mapping[f"{p}k_norm:weight"] = (f"{lp}self_attn.k_norm.weight", False)
        mapping[f"{p}attention_out:weight"] = (f"{lp}self_attn.o_proj.weight", True)
        mapping[f"{p}ffn_norm:weight"] = (f"{lp}post_attention_layernorm.weight", False)
        mapping[f"{p}ffn_up:weight"] = (f"{lp}mlp.up_proj.weight", True)
        mapping[f"{p}ffn_gate:weight"] = (f"{lp}mlp.gate_proj.weight", True)
        mapping[f"{p}ffn_down:weight"] = (f"{lp}mlp.down_proj.weight", True)

    # Compare
    print(f"\n{'Safetensors Name':<35} {'HF Key':<45} {'Shape Match':>11} {'Value Match':>11} {'MaxDiff':>10}")
    print("-" * 120)

    ok_count = 0
    fail_count = 0
    for sf_name, (hf_key, transpose) in mapping.items():
        if sf_name not in sf_tensors:
            print(f"{sf_name:<35} MISSING in safetensors!")
            fail_count += 1
            continue
        if hf_key not in sd:
            print(f"{sf_name:<35} {hf_key:<45} HF key not found!")
            fail_count += 1
            continue

        sf_arr = sf_tensors[sf_name]
        hf_arr = sd[hf_key].cpu().numpy()
        if transpose:
            hf_arr = hf_arr.T

        shape_ok = sf_arr.shape == hf_arr.shape
        if shape_ok:
            max_diff = np.max(np.abs(sf_arr - hf_arr))
            val_ok = max_diff < 1e-6
        else:
            max_diff = float('inf')
            val_ok = False

        status_shape = "OK" if shape_ok else f"FAIL {sf_arr.shape} vs {hf_arr.shape}"
        status_val = "OK" if val_ok else "FAIL"

        if val_ok:
            ok_count += 1
        else:
            fail_count += 1

        flag = "" if val_ok else " <-- MISMATCH"
        print(f"{sf_name:<35} {hf_key:<45} {status_shape:>11} {status_val:>11} {max_diff:>10.2e}{flag}")

        if not val_ok and shape_ok:
            print(f"  sf first4:  {sf_arr.flatten()[:4]}")
            print(f"  hf first4:  {hf_arr.flatten()[:4]}")

    print(f"\nResult: {ok_count} OK, {fail_count} FAIL (total {ok_count + fail_count})")


if __name__ == "__main__":
    main()

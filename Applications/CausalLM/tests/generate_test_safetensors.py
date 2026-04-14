#!/usr/bin/env python3
"""Generate a tiny test safetensors file + configs for verifying weight loading.

Creates a 1-layer, tiny Qwen3-like model with random weights.
Tensor names use ":weight" suffix (converter convention).
C++ model expects ":Embedding", ":gamma", ":weight" — prefix fallback handles this.

Usage:
    python generate_test_safetensors.py [output_dir]

Output files:
    output_dir/config.json
    output_dir/generation_config.json
    output_dir/nntr_config.json
    output_dir/test_model.safetensors
    output_dir/expected_values.txt   (first 4 values per tensor for verification)
"""

import json
import struct
import os
import sys
import numpy as np
from collections import OrderedDict

# Tiny Qwen3-like config
HIDDEN_SIZE = 32
INTERMEDIATE_SIZE = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 8
NUM_LAYERS = 2
VOCAB_SIZE = 100
GQA_SIZE = NUM_HEADS // NUM_KV_HEADS  # 2

np.random.seed(42)


def make_weight_entries():
    """Build (tensor_name, shape, needs_transpose) list in C++ creation order."""
    entries = []

    # embedding0 (C++ suffix: "Embedding", safetensors: ":weight")
    entries.append(("embedding0:weight", [VOCAB_SIZE, HIDDEN_SIZE]))

    for i in range(NUM_LAYERS):
        # attention_norm (C++ suffix: "gamma")
        entries.append((f"layer{i}_attention_norm:weight", [HIDDEN_SIZE]))

        # V projection [in, out] (transposed from HF [out, in])
        kv_dim = HEAD_DIM * NUM_HEADS // GQA_SIZE
        entries.append((f"layer{i}_wv:weight", [HIDDEN_SIZE, kv_dim]))
        # K projection
        entries.append((f"layer{i}_wk:weight", [HIDDEN_SIZE, kv_dim]))
        # K norm (C++ suffix: "gamma")
        entries.append((f"layer{i}_k_norm:weight", [HEAD_DIM]))
        # Q projection
        entries.append((f"layer{i}_wq:weight", [HIDDEN_SIZE, HIDDEN_SIZE]))
        # Q norm (C++ suffix: "gamma")
        entries.append((f"layer{i}_q_norm:weight", [HEAD_DIM]))
        # O projection
        entries.append((f"layer{i}_attention_out:weight", [HIDDEN_SIZE, HIDDEN_SIZE]))

        # FFN norm (C++ suffix: "gamma")
        entries.append((f"layer{i}_ffn_norm:weight", [HIDDEN_SIZE]))
        # FFN up
        entries.append((f"layer{i}_ffn_up:weight", [HIDDEN_SIZE, INTERMEDIATE_SIZE]))
        # FFN gate
        entries.append((f"layer{i}_ffn_gate:weight", [HIDDEN_SIZE, INTERMEDIATE_SIZE]))
        # FFN down
        entries.append((f"layer{i}_ffn_down:weight", [INTERMEDIATE_SIZE, HIDDEN_SIZE]))

    # output_norm (C++ suffix: "gamma")
    entries.append(("output_norm:weight", [HIDDEN_SIZE]))

    # No lm_head (tie_word_embeddings=true)
    return entries


def write_safetensors(entries, tensor_data_map, output_path):
    """Write safetensors file."""
    header_entries = OrderedDict()
    data_offset = 0
    all_data = []

    for name, shape in entries:
        data = tensor_data_map[name]
        data_size = len(data)
        header_entries[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [data_offset, data_offset + data_size],
        }
        all_data.append(data)
        data_offset += data_size

    header_dict = {"__metadata__": {"format": "nntrainer"}}
    header_dict.update(header_entries)
    header_json = json.dumps(header_dict, separators=(",", ":"),
                             ensure_ascii=True)
    header_bytes = header_json.encode("utf-8")
    pad_len = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad_len

    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for data in all_data:
            f.write(data)

    return header_entries


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nntr_test_model"
    os.makedirs(output_dir, exist_ok=True)

    entries = make_weight_entries()

    # Generate random data and save expected values
    tensor_data_map = {}
    expected_lines = []
    for name, shape in entries:
        arr = np.random.randn(*shape).astype(np.float32) * 0.1
        tensor_data_map[name] = arr.tobytes()
        first_vals = arr.flatten()[:4]
        expected_lines.append(
            f"{name:<40} shape={shape!s:<20} first4={list(first_vals)}")

    # Write safetensors
    sf_path = os.path.join(output_dir, "test_model.safetensors")
    write_safetensors(entries, tensor_data_map, sf_path)

    # Write expected values
    with open(os.path.join(output_dir, "expected_values.txt"), "w") as f:
        f.write(f"# Seed=42, {len(entries)} tensors\n")
        for line in expected_lines:
            f.write(line + "\n")

    # Write config.json (HF-style)
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_attention_heads": NUM_HEADS,
        "num_hidden_layers": NUM_LAYERS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB_SIZE,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 500000,
        "tie_word_embeddings": True,
        "sliding_window": None,
        "bos_token_id": 1,
        "eos_token_id": [2],
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Write generation_config.json
    gen_config = {
        "bos_token_id": 1,
        "eos_token_id": [2],
        "do_sample": False,
    }
    with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
        json.dump(gen_config, f, indent=2)

    # Write nntr_config.json
    nntr_config = {
        "model_type": "CausalLM",
        "model_tensor_type": "FP32-FP32",
        "model_file_name": "test_model.safetensors",
        "fc_layer_dtype": "FP32",
        "embedding_dtype": "FP32",
        "bad_word_ids": [],
        "fsu": False,
        "num_to_generate": 4,
        "init_seq_len": 8,
        "max_seq_len": 16,
        "batch_size": 1,
        "tokenizer_file": "__PLACEHOLDER__",
        "sample_input": "hello",
    }
    with open(os.path.join(output_dir, "nntr_config.json"), "w") as f:
        json.dump(nntr_config, f, indent=2)

    # Summary
    print(f"Generated test model in: {output_dir}")
    print(f"  safetensors: {sf_path} ({os.path.getsize(sf_path)} bytes)")
    print(f"  tensors: {len(entries)}")
    print(f"  config: hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, "
          f"heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS}")
    print()

    # Verify: read back safetensors header
    with open(sf_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))

    print("Safetensors header verification:")
    for name, info in header.items():
        if name == "__metadata__":
            continue
        print(f"  {name:<40} shape={info['shape']}")

    sf_names = set(k for k in header if k != "__metadata__")

    print()
    print("Simulated C++ loading (prefix fallback test):")
    # Build all C++ expected weight names
    all_cpp_names = []
    all_cpp_names.append("embedding0:Embedding")
    for i in range(NUM_LAYERS):
        all_cpp_names.append(f"layer{i}_attention_norm:gamma")
        all_cpp_names.append(f"layer{i}_wv:weight")
        all_cpp_names.append(f"layer{i}_wk:weight")
        all_cpp_names.append(f"layer{i}_k_norm:gamma")
        all_cpp_names.append(f"layer{i}_wq:weight")
        all_cpp_names.append(f"layer{i}_q_norm:gamma")
        all_cpp_names.append(f"layer{i}_attention_out:weight")
        all_cpp_names.append(f"layer{i}_ffn_norm:gamma")
        all_cpp_names.append(f"layer{i}_ffn_up:weight")
        all_cpp_names.append(f"layer{i}_ffn_gate:weight")
        all_cpp_names.append(f"layer{i}_ffn_down:weight")
    all_cpp_names.append("output_norm:gamma")

    sf_prefixes = {k.split(":")[0]: k for k in sf_names}

    match = prefix_cnt = miss = 0
    for cpp_name in all_cpp_names:
        if cpp_name in sf_names:
            match += 1
            print(f"  [MATCH]  '{cpp_name}'")
        else:
            prefix_key = cpp_name.split(":")[0]
            if prefix_key in sf_prefixes:
                prefix_cnt += 1
                print(f"  [PREFIX] '{cpp_name}' -> '{sf_prefixes[prefix_key]}'")
            else:
                miss += 1
                print(f"  [MISS]   '{cpp_name}'")

    print(f"\nResult: {match} exact, {prefix_cnt} prefix, {miss} miss "
          f"(total {match + prefix_cnt + miss})")

    if miss == 0:
        print("All weights should load correctly with prefix fallback.")
    else:
        print(f"WARNING: {miss} weights will MISS!")


if __name__ == "__main__":
    main()

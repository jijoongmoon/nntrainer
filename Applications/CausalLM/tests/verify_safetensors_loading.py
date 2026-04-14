#!/usr/bin/env python3
"""Verify safetensors loading by simulating C++ neuralnet.cpp logic exactly.

Simulates:
1. Parsing safetensors JSON header -> name_offset_map
2. Building prefix_offset_map (fallback)
3. For each C++ weight name: exact match or prefix fallback
4. Reading data from file at the resolved offset
5. Comparing with expected values

This is the exact same logic as neuralnet.cpp lines 1007-1095.

Usage:
    python verify_safetensors_loading.py [safetensors_path] [expected_values_path]
"""

import json
import struct
import sys
import numpy as np


def parse_safetensors_header(path):
    """Parse safetensors header. Returns (header_dict, data_section_start)."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8").rstrip()
        data_section_start = 8 + header_size
    header = json.loads(header_json)
    return header, data_section_start


def read_tensor_at_offset(path, offset, num_bytes):
    """Read raw float32 bytes from file at given offset."""
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(num_bytes)
    return np.frombuffer(data, dtype=np.float32)


def build_cpp_weight_names(num_layers):
    """Build the list of weight names as C++ NNTrainer would see them.

    C++ layer types register weights with these suffixes:
      EmbeddingLayer: "Embedding"
      RMSNormLayer / ReshapedRMSNormLayer: "gamma"
      FullyConnectedLayer: "weight"
      TieWordEmbedding (shared): same as embedding
    """
    names = []

    # embedding0 -> requestWeight(..., "Embedding")
    names.append(("embedding0:Embedding", "embedding0"))

    for i in range(num_layers):
        # attention_norm -> RMSNorm -> "gamma"
        names.append((f"layer{i}_attention_norm:gamma", f"layer{i}_attention_norm"))
        # V proj -> FC -> "weight"
        names.append((f"layer{i}_wv:weight", f"layer{i}_wv"))
        # K proj -> FC -> "weight"
        names.append((f"layer{i}_wk:weight", f"layer{i}_wk"))
        # K norm -> ReshapedRMSNorm -> "gamma"
        names.append((f"layer{i}_k_norm:gamma", f"layer{i}_k_norm"))
        # Q proj -> FC -> "weight"
        names.append((f"layer{i}_wq:weight", f"layer{i}_wq"))
        # Q norm -> ReshapedRMSNorm -> "gamma"
        names.append((f"layer{i}_q_norm:gamma", f"layer{i}_q_norm"))
        # O proj -> FC -> "weight"
        names.append((f"layer{i}_attention_out:weight", f"layer{i}_attention_out"))
        # FFN norm -> RMSNorm -> "gamma"
        names.append((f"layer{i}_ffn_norm:gamma", f"layer{i}_ffn_norm"))
        # FFN up -> FC -> "weight"
        names.append((f"layer{i}_ffn_up:weight", f"layer{i}_ffn_up"))
        # FFN gate -> FC -> "weight"
        names.append((f"layer{i}_ffn_gate:weight", f"layer{i}_ffn_gate"))
        # FFN down -> FC -> "weight"
        names.append((f"layer{i}_ffn_down:weight", f"layer{i}_ffn_down"))

    # output_norm -> RMSNorm -> "gamma"
    names.append(("output_norm:gamma", "output_norm"))

    return names


def main():
    sf_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nntr_test_model/test_model.safetensors"
    expected_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/nntr_test_model/expected_values.txt"

    # --- Step 1: Parse safetensors header (same as neuralnet.cpp) ---
    header, data_section_start = parse_safetensors_header(sf_path)

    name_offset_map = {}  # name -> (offset, size)
    for name, info in header.items():
        if name == "__metadata__":
            continue
        offsets = info["data_offsets"]
        name_offset_map[name] = (offsets[0], offsets[1] - offsets[0])

    print(f"[safetensors] Loaded {len(name_offset_map)} tensor entries")
    print(f"[safetensors] data_section_start = {data_section_start}")
    print()

    # --- Step 2: Build prefix_offset_map (same as neuralnet.cpp) ---
    prefix_offset_map = {}
    for name, (offset, size) in name_offset_map.items():
        colon_pos = name.find(":")
        if colon_pos != -1:
            prefix = name[:colon_pos]
            if prefix not in prefix_offset_map:
                prefix_offset_map[prefix] = (offset, size)

    # --- Step 3: Read config to get num_layers ---
    import os
    config_dir = os.path.dirname(sf_path)
    config_path = os.path.join(config_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    num_layers = config["num_hidden_layers"]

    # --- Step 4: Simulate C++ weight loading ---
    cpp_names = build_cpp_weight_names(num_layers)

    match_count = 0
    prefix_count = 0
    miss_count = 0
    loaded_values = {}  # prefix -> first4 values

    print("=== Simulated C++ loading ===")
    for cpp_name, layer_prefix in cpp_names:
        # Exact match (neuralnet.cpp line 1053)
        if cpp_name in name_offset_map:
            offset, size = name_offset_map[cpp_name]
            abs_offset = data_section_start + offset
            arr = read_tensor_at_offset(sf_path, abs_offset, size)
            first4 = arr.flatten()[:4].tolist()
            loaded_values[layer_prefix] = first4
            print(f"  [MATCH]  '{cpp_name}' -> offset={abs_offset} first4={first4}")
            match_count += 1
        else:
            # Prefix fallback (neuralnet.cpp line 1064-1074)
            colon_pos = cpp_name.find(":")
            prefix = cpp_name[:colon_pos] if colon_pos != -1 else cpp_name
            if prefix in prefix_offset_map:
                offset, size = prefix_offset_map[prefix]
                abs_offset = data_section_start + offset
                arr = read_tensor_at_offset(sf_path, abs_offset, size)
                first4 = arr.flatten()[:4].tolist()
                loaded_values[layer_prefix] = first4
                print(f"  [PREFIX] '{cpp_name}' matched by '{prefix}' "
                      f"-> offset={abs_offset} first4={first4}")
                prefix_count += 1
            else:
                print(f"  [MISS]   '{cpp_name}' NOT FOUND")
                miss_count += 1

    print(f"\n=== Summary: {match_count} match, {prefix_count} prefix, "
          f"{miss_count} miss ===\n")

    # --- Step 5: Compare with expected values ---
    if not os.path.exists(expected_path):
        print(f"No expected_values.txt found at {expected_path}, skipping comparison")
        return miss_count == 0

    print("=== Value Comparison ===")
    expected_map = {}
    with open(expected_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Parse: "embedding0:weight    shape=[100, 32]    first4=[...]"
            parts = line.split("first4=")
            if len(parts) != 2:
                continue
            tensor_name = parts[0].split()[0]  # e.g. "embedding0:weight"
            colon_pos = tensor_name.find(":")
            prefix = tensor_name[:colon_pos] if colon_pos != -1 else tensor_name
            expected_vals = eval(parts[1])  # list of 4 floats
            expected_map[prefix] = expected_vals

    all_ok = True
    for prefix, loaded_vals in loaded_values.items():
        if prefix not in expected_map:
            print(f"  [SKIP] '{prefix}' not in expected_values.txt")
            continue
        expected_vals = expected_map[prefix]
        match = all(abs(a - b) < 1e-6 for a, b in zip(loaded_vals, expected_vals))
        status = "OK" if match else "FAIL"
        if not match:
            all_ok = False
        print(f"  [{status}] {prefix:<30} loaded={loaded_vals}")
        if not match:
            print(f"       {'':30} expect={expected_vals}")

    print()
    if all_ok and miss_count == 0:
        print("ALL PASSED: All weights loaded correctly with matching values.")
    elif miss_count > 0:
        print(f"FAILED: {miss_count} weights could not be found.")
    else:
        print("FAILED: Some values don't match.")

    return all_ok and miss_count == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

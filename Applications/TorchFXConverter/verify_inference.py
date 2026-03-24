#!/usr/bin/env python3
"""
End-to-end inference verification: Python (PyTorch) vs C++ (NNTrainer).

Creates a tiny Qwen3 model with random weights, runs Python forward pass,
converts to NNTrainer C++ code + binary weights, then builds and runs the
C++ inference driver to compare logits.

Usage:
  python verify_inference.py                     # full pipeline
  python verify_inference.py --skip-build        # skip C++ build/run (Python-only)
  python verify_inference.py --output ./my_out   # custom output directory
"""

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch

# Ensure converter package is importable
CONVERTER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CONVERTER_DIR)

NNTRAINER_ROOT = os.path.abspath(os.path.join(CONVERTER_DIR, "..", ".."))
BUILD_DIR = os.path.join(NNTRAINER_ROOT, "builddir")
JNI_DIR = os.path.join(CONVERTER_DIR, "jni")

# ---------------------------------------------------------------------------
# Tiny Qwen3 config (matches test_build_and_run.py QWEN3_CONFIG)
# ---------------------------------------------------------------------------
TINY_QWEN3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
}

# Fixed input for reproducibility
SEQ_LEN = 8
BATCH_SIZE = 1
FIXED_INPUT_IDS = [1, 42, 100, 7, 256, 500, 3, 99]


def create_tiny_model(model_dir):
    """Create a tiny Qwen3 model with fixed random weights."""
    from transformers import AutoConfig, AutoModelForCausalLM

    os.makedirs(model_dir, exist_ok=True)

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(TINY_QWEN3_CONFIG, f, indent=2)

    config = AutoConfig.from_pretrained(model_dir)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch.float32)
    model.eval()
    model.save_pretrained(model_dir)

    print(f"[verify] Created tiny Qwen3 model in {model_dir}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")
    return model, config


def run_python_inference(model, input_ids_list):
    """Run Python forward pass and return logits as numpy array."""
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    logits_np = logits.cpu().numpy()
    print(f"[verify] Python logits shape: {logits_np.shape}")
    print(f"  min={logits_np.min():.6f}, max={logits_np.max():.6f}, "
          f"mean={logits_np.mean():.6f}")
    return logits_np


def save_reference_data(output_dir, input_ids_list, logits_np):
    """Save input and reference output for C++ comparison."""
    os.makedirs(output_dir, exist_ok=True)

    # Save input as text (one token per line)
    input_path = os.path.join(output_dir, "reference_input.txt")
    with open(input_path, "w") as f:
        for tok in input_ids_list:
            f.write(f"{tok}\n")

    # Save input as binary float32 (for C++ to read directly)
    input_bin_path = os.path.join(output_dir, "reference_input.bin")
    with open(input_bin_path, "wb") as f:
        for tok in input_ids_list:
            f.write(struct.pack("f", float(tok)))

    # Save logits as binary float32
    logits_path = os.path.join(output_dir, "reference_logits.bin")
    logits_np.astype(np.float32).tofile(logits_path)

    # Save logits shape info
    meta = {
        "input_ids": input_ids_list,
        "logits_shape": list(logits_np.shape),
        "seq_len": len(input_ids_list),
        "batch_size": 1,
        "vocab_size": logits_np.shape[-1],
    }
    meta_path = os.path.join(output_dir, "reference_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[verify] Saved reference data to {output_dir}")
    return input_bin_path, logits_path


def run_converter(model_dir, output_dir):
    """Run TorchFX converter to generate C++ code + weights."""
    cmd = [
        sys.executable, os.path.join(CONVERTER_DIR, "converter.py"),
        "--model", model_dir,
        "--output", output_dir,
        "--format", "cpp", "json",
        "--weights",
        "--seq-len", str(SEQ_LEN),
        "--model-name", "qwen3_verify",
    ]

    print(f"[verify] Running converter...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"[verify] Converter FAILED:\n{result.stderr}")
        return False
    print(result.stdout)
    return True


def copy_to_jni(output_dir):
    """Copy generated C++ files to jni/ directory for meson build."""
    copied = []
    for fname in os.listdir(output_dir):
        if fname.endswith((".h", ".cpp")):
            src = os.path.join(output_dir, fname)
            dst = os.path.join(JNI_DIR, fname)
            shutil.copy2(src, dst)
            copied.append(dst)
            print(f"[verify] Copied {fname} -> jni/")
    return copied


def meson_reconfigure():
    """Reconfigure meson to pick up new source files."""
    cmd = ["meson", "setup", "--reconfigure", BUILD_DIR]
    print(f"[verify] Reconfiguring meson...")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=NNTRAINER_ROOT, timeout=120)
    if result.returncode != 0:
        print(f"[verify] Meson reconfigure FAILED:\n{result.stderr[-500:]}")
        return False
    return True


def build_inference_target():
    """Build the inference test executable."""
    target = "Applications/TorchFXConverter/jni/converter_qwen3_verify_inference"
    cmd = ["ninja", "-C", BUILD_DIR, target]
    print(f"[verify] Building {target}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"[verify] Build FAILED:\n{result.stderr[-1000:]}")
        return False, None
    exe_path = os.path.join(BUILD_DIR, target)
    return True, exe_path


def run_cpp_inference(exe_path, weight_path, input_bin_path, output_logits_path,
                      vocab_size=1000):
    """Run C++ inference executable."""
    env = os.environ.copy()
    lib_paths = [
        os.path.join(BUILD_DIR, "nntrainer"),
        os.path.join(BUILD_DIR, "Applications", "CausalLM", "layers"),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths) + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        exe_path,
        "--weights", weight_path,
        "--input", input_bin_path,
        "--output", output_logits_path,
        "--seq-len", str(SEQ_LEN),
        "--vocab-size", str(vocab_size),
    ]
    print(f"[verify] Running C++ inference...")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                            timeout=120)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[verify] C++ inference FAILED:\n{result.stderr}")
        return False
    return True


def compare_logits(python_logits_path, cpp_logits_path, meta_path):
    """Compare Python and C++ logits (last position only).

    C++ incremental_inference returns only the last position's logits,
    so we compare against the last position from Python's full output.
    """
    with open(meta_path) as f:
        meta = json.load(f)
    shape = tuple(meta["logits_shape"])
    vocab_size = meta["vocab_size"]

    py_logits = np.fromfile(python_logits_path, dtype=np.float32).reshape(shape)
    # C++ output is (vocab_size,) - last position only
    cpp_last = np.fromfile(cpp_logits_path, dtype=np.float32)[:vocab_size]

    # Python last position
    py_last = py_logits[0, -1, :]

    abs_diff = np.abs(py_last - cpp_last)
    max_abs_error = abs_diff.max()
    mean_abs_error = abs_diff.mean()

    # Relative error (avoid division by zero)
    denom = np.maximum(np.abs(py_last), 1e-8)
    rel_diff = abs_diff / denom
    max_rel_error = rel_diff.max()
    mean_rel_error = rel_diff.mean()

    print("\n" + "=" * 60)
    print("INFERENCE COMPARISON RESULTS (last position logits)")
    print("=" * 60)
    print(f"  Vocab size:         {vocab_size}")
    print(f"  Max absolute error: {max_abs_error:.8f}")
    print(f"  Mean absolute error:{mean_abs_error:.8f}")
    print(f"  Max relative error: {max_rel_error:.8f}")
    print(f"  Mean relative error:{mean_rel_error:.8f}")

    # Check top-k prediction match
    py_top5 = np.argsort(py_last)[-5:][::-1]
    cpp_top5 = np.argsort(cpp_last)[-5:][::-1]

    print(f"\n  Python  top-5 tokens: {py_top5.tolist()}")
    print(f"  C++     top-5 tokens: {cpp_top5.tolist()}")
    top1_match = py_top5[0] == cpp_top5[0]
    print(f"  Top-1 match: {'YES' if top1_match else 'NO'}")

    # Sample values
    print(f"\n  Python first 5: {py_last[:5]}")
    print(f"  C++    first 5: {cpp_last[:5]}")

    # Tolerance check
    TOLERANCE = 1e-4
    passed = max_abs_error < TOLERANCE
    print(f"\n  Tolerance: {TOLERANCE}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify NNTrainer inference matches PyTorch")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: temp)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip C++ build/run (Python reference only)")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep generated files in jni/ after test")
    args = parser.parse_args()

    use_temp = args.output is None
    if use_temp:
        tmpdir = tempfile.mkdtemp(prefix="verify_inference_")
        output_base = tmpdir
    else:
        output_base = os.path.abspath(args.output)
        os.makedirs(output_base, exist_ok=True)

    model_dir = os.path.join(output_base, "model")
    converter_output = os.path.join(output_base, "converter_output")
    reference_dir = os.path.join(output_base, "reference")

    generated_jni_files = []

    try:
        # Step 1: Create tiny model
        print("\n--- Step 1: Create tiny Qwen3 model ---")
        model, config = create_tiny_model(model_dir)

        # Step 2: Run Python inference
        print("\n--- Step 2: Run Python forward pass ---")
        logits_np = run_python_inference(model, FIXED_INPUT_IDS)
        input_bin, logits_bin = save_reference_data(
            reference_dir, FIXED_INPUT_IDS, logits_np)

        # Step 3: Run converter
        print("\n--- Step 3: Convert to NNTrainer format ---")
        if not run_converter(model_dir, converter_output):
            print("[verify] ABORT: Conversion failed")
            sys.exit(1)

        if args.skip_build:
            print("\n[verify] --skip-build: Skipping C++ build/run.")
            print(f"[verify] Reference data saved in: {reference_dir}")
            print(f"[verify] Converter output in: {converter_output}")
            return

        # Step 4: Copy to jni/ and build
        print("\n--- Step 4: Build C++ inference driver ---")
        generated_jni_files = copy_to_jni(converter_output)
        if not meson_reconfigure():
            print("[verify] ABORT: Meson reconfigure failed")
            sys.exit(1)

        ok, exe_path = build_inference_target()
        if not ok:
            print("[verify] ABORT: Build failed")
            sys.exit(1)

        # Step 5: Run C++ inference
        print("\n--- Step 5: Run C++ inference ---")
        weight_path = os.path.join(converter_output, "model.bin")
        cpp_logits_path = os.path.join(reference_dir, "cpp_logits.bin")

        if not run_cpp_inference(exe_path, weight_path, input_bin,
                                 cpp_logits_path,
                                 vocab_size=TINY_QWEN3_CONFIG["vocab_size"]):
            print("[verify] ABORT: C++ inference failed")
            sys.exit(1)

        # Step 6: Compare
        print("\n--- Step 6: Compare logits ---")
        meta_path = os.path.join(reference_dir, "reference_meta.json")
        passed = compare_logits(logits_bin, cpp_logits_path, meta_path)

        sys.exit(0 if passed else 1)

    finally:
        # Cleanup generated jni files
        if not args.keep_files:
            for f in generated_jni_files:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"[verify] Cleaned up {os.path.basename(f)}")
        if use_temp and not args.keep_files:
            shutil.rmtree(output_base, ignore_errors=True)


if __name__ == "__main__":
    main()

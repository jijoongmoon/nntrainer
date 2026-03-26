#!/usr/bin/env python3
"""
Test driver for CausalLM model verification.
Creates a minimal tokenizer and runs the C++ test binary to verify
all model architectures (Qwen3, Qwen2, Gemma3, Llama) build,
initialize, and run correctly with random weights.
"""

import json
import os
import subprocess
import sys
import tempfile


def create_minimal_tokenizer(output_path):
    """Create a minimal tokenizer.json using the tokenizers library."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace

    # Build a minimal BPE tokenizer
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    # Add single character tokens for ASCII range
    for i in range(32, 128):
        vocab[chr(i)] = len(vocab)
    # Add common pairs (all constituents must be in vocab already)
    vocab["he"] = len(vocab)
    vocab["ll"] = len(vocab)
    vocab["lo"] = len(vocab)
    vocab["hel"] = len(vocab)
    vocab["hell"] = len(vocab)
    vocab["hello"] = len(vocab)

    merges = [("h", "e"), ("l", "l"), ("l", "o"), ("he", "l"), ("hel", "l"), ("hell", "o")]

    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(output_path)
    print(f"Created tokenizer at {output_path}")


def find_test_binary(build_dir):
    """Find the test_causallm binary."""
    candidates = [
        os.path.join(build_dir, "Applications/CausalLM/test_causallm"),
        os.path.join(build_dir, "test_causallm"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    # Find project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    build_dir = os.path.join(project_root, "builddir")

    # Build first
    print("=== Building test binary ===")
    ret = subprocess.run(
        ["ninja", "-C", build_dir, "Applications/CausalLM/test_causallm"],
        capture_output=False
    )
    if ret.returncode != 0:
        print("BUILD FAILED")
        sys.exit(1)
    print("Build OK\n")

    # Find binary
    binary = find_test_binary(build_dir)
    if not binary:
        print(f"ERROR: test_causallm binary not found in {build_dir}")
        sys.exit(1)
    print(f"Binary: {binary}")

    # Create temp directory with tokenizer
    with tempfile.TemporaryDirectory(prefix="test_causallm_") as tmp_dir:
        tokenizer_path = os.path.join(tmp_dir, "tokenizer.json")
        create_minimal_tokenizer(tokenizer_path)

        # Determine which archs to test
        archs = sys.argv[1:] if len(sys.argv) > 1 else []
        cmd = [binary, tokenizer_path, tmp_dir] + archs

        print(f"\n=== Running: {' '.join(cmd)} ===\n")

        # Set up LD_LIBRARY_PATH
        env = os.environ.copy()
        lib_paths = [
            os.path.join(build_dir, "nntrainer"),
            os.path.join(build_dir, "api/ccapi"),
            os.path.join(build_dir, "Applications/CausalLM"),
            os.path.join(build_dir, "Applications/CausalLM/layers"),
            os.path.join(build_dir, "Applications/CausalLM/models/gpt_oss"),
            os.path.join(build_dir, "Applications/CausalLM/models/gpt_oss_cached_slim"),
            os.path.join(build_dir, "Applications/CausalLM/models/qwen3_moe"),
            os.path.join(build_dir, "Applications/CausalLM/models/qwen3_slim_moe"),
            os.path.join(build_dir, "Applications/CausalLM/models/qwen3_cached_slim_moe"),
        ]
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_paths) + (":" + existing if existing else "")

        ret = subprocess.run(cmd, env=env)
        sys.exit(ret.returncode)


if __name__ == "__main__":
    main()

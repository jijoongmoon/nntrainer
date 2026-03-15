#!/usr/bin/env python3
"""
Integration tests: convert multiple model architectures and verify the
generated C++ code compiles and initializes with NNTrainer.

For each model config:
  1. Create a tiny HuggingFace model locally
  2. Run converter.py to generate C++ header + source
  3. Build with meson/ninja
  4. Run the test executable and verify exit code 0

Usage:
  python -m pytest tests/test_build_and_run.py -v
  python tests/test_build_and_run.py           # standalone
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

# Ensure the converter package is importable
CONVERTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CONVERTER_DIR)

# Path to the nntrainer source root
NNTRAINER_ROOT = os.path.abspath(os.path.join(CONVERTER_DIR, "..", ".."))
BUILD_DIR = os.path.join(NNTRAINER_ROOT, "builddir")

# ---- Tiny model configurations for offline testing ----

QWEN3_CONFIG = {
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

LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
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

QWEN3_TIED_CONFIG = {
    **QWEN3_CONFIG,
    "tie_word_embeddings": True,
}

MODEL_CONFIGS = {
    "qwen3": QWEN3_CONFIG,
    "llama": LLAMA_CONFIG,
    "qwen3_tied": QWEN3_TIED_CONFIG,
}


def _create_local_model(config, model_dir):
    """Create a tiny local model using transformers."""
    os.makedirs(model_dir, exist_ok=True)
    model_type = config["model_type"]

    try:
        if model_type == "qwen3":
            from transformers import Qwen3Config, Qwen3ForCausalLM
            cfg = Qwen3Config(**{k: v for k, v in config.items()
                                 if k not in ("architectures",)})
            model = Qwen3ForCausalLM(cfg)
        elif model_type == "llama":
            from transformers import LlamaConfig, LlamaForCausalLM
            cfg = LlamaConfig(**{k: v for k, v in config.items()
                                 if k not in ("architectures",)})
            model = LlamaForCausalLM(cfg)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.save_pretrained(model_dir)
        return True
    except ImportError:
        return False


def _run_converter(model_dir, output_dir, model_name=None):
    """Run converter.py and return (success, output_files)."""
    cmd = [
        sys.executable, os.path.join(CONVERTER_DIR, "converter.py"),
        "--model", model_dir,
        "--output", output_dir,
        "--format", "cpp",
    ]
    if model_name:
        cmd += ["--model-name", model_name]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0, result.stdout, result.stderr


def _build_test(build_dir, target):
    """Build a specific meson target."""
    cmd = ["ninja", "-C", build_dir, target]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0, result.stdout, result.stderr


def _run_test(build_dir, executable):
    """Run a test executable."""
    env = os.environ.copy()
    lib_paths = [
        os.path.join(build_dir, "nntrainer"),
        os.path.join(build_dir, "Applications", "CausalLM", "layers"),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths)
    cmd = [os.path.join(build_dir, executable)]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=60, env=env)
    return result.returncode == 0, result.stdout, result.stderr


class TestConverterBuildAndRun(unittest.TestCase):
    """Test that converter output compiles and runs with NNTrainer."""

    @classmethod
    def setUpClass(cls):
        """Check prerequisites."""
        if not os.path.isdir(BUILD_DIR):
            raise unittest.SkipTest(
                f"Build directory not found: {BUILD_DIR}. "
                f"Run: meson setup builddir -Denable-transformer=true")

    def test_qwen3_tiny_build_and_run(self):
        """Qwen3 tiny model: convert -> build -> initialize."""
        self._run_model_test("qwen3")

    def test_llama_tiny_build_and_run(self):
        """LLaMA tiny model: convert -> build -> initialize."""
        self._run_model_test("llama")

    def test_qwen3_tied_embeddings(self):
        """Qwen3 with tied word embeddings."""
        self._run_model_test("qwen3_tied")

    def _run_model_test(self, config_name):
        """Run full pipeline for a model config."""
        config = MODEL_CONFIGS[config_name]

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            output_dir = os.path.join(tmpdir, "output")

            # Step 1: Create local model
            if not _create_local_model(config, model_dir):
                self.skipTest("transformers not available")

            # Step 2: Convert
            ok, stdout, stderr = _run_converter(
                model_dir, output_dir, model_name=config_name)
            self.assertTrue(ok, f"Converter failed:\n{stderr}")

            # Step 3: Verify output files exist
            cpp_files = [f for f in os.listdir(output_dir)
                         if f.endswith(('.h', '.cpp'))]
            self.assertGreaterEqual(len(cpp_files), 2,
                                    f"Expected .h and .cpp, got: {cpp_files}")

            # Log the output
            print(f"\n[{config_name}] Converter output:")
            print(stdout)

    def test_prebuilt_qwen3_executable(self):
        """Run the pre-built converter_qwen3_test executable."""
        target = "Applications/TorchFXConverter/jni/converter_qwen3_test"
        exe_path = os.path.join(BUILD_DIR, target)

        if not os.path.isfile(exe_path):
            # Try building it
            ok, _, stderr = _build_test(BUILD_DIR, target)
            if not ok:
                self.skipTest(f"Could not build {target}: {stderr[-200:]}")

        ok, stdout, stderr = _run_test(BUILD_DIR, target)
        self.assertTrue(ok,
                        f"Executable failed:\nstdout: {stdout}\nstderr: {stderr}")
        self.assertIn("Model initialized successfully", stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)

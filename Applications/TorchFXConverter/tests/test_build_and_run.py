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

# Path to the nntrainer source root and build output
NNTRAINER_ROOT = os.path.abspath(os.path.join(CONVERTER_DIR, "..", ".."))
BUILD_DIR = os.path.join(NNTRAINER_ROOT, "builddir")
JNI_DIR = os.path.join(CONVERTER_DIR, "jni")

# ---- Tiny model configurations for offline testing ----

# --- Decoder-only (CausalLM) models ---

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

# Qwen3-0.6B: standard Qwen3 CausalLM (same arch, slightly different config)
QWEN3_06B_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 192,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 1500,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
}

# Qwen3-1.7B: larger Qwen3 CausalLM (more heads, more intermediate)
QWEN3_17B_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 128,
    "intermediate_size": 384,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "vocab_size": 2000,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
}

# Granite 4.0: GraniteMoeHybrid in dense mode (num_local_experts=0)
GRANITE_40_CONFIG = {
    "architectures": ["GraniteMoeHybridForCausalLM"],
    "model_type": "granitemoehybrid",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
    # Granite-specific: dense mode (no MoE), no Mamba
    "num_local_experts": 0,
    "num_experts_per_tok": 0,
    "layer_types": ["attention", "attention"],
    # Granite scaling multipliers
    "embedding_multiplier": 1.0,
    "logits_scaling": 1.0,
    "residual_multiplier": 1.0,
    "attention_multiplier": 1.0,
}

# LFM-700M: Liquid Foundation Model (lfm2), full-attention only
LFM_700M_CONFIG = {
    "architectures": ["Lfm2ForCausalLM"],
    "model_type": "lfm2",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "norm_eps": 1e-5,
    "tie_word_embeddings": True,
    # lfm2-specific: only full-attention layers (no conv/SSM)
    "layer_types": ["full_attention", "full_attention"],
    "conv_L_cache": 3,
}

# --- Embedding models (base models without LM head) ---

# Qwen3-Embedding-0.6B: Qwen3 base model for sentence embeddings
QWEN3_EMBEDDING_CONFIG = {
    "architectures": ["Qwen3Model"],
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

# EmbeddingGemma-300M: Gemma2 base model for embeddings
EMBEDDING_GEMMA_CONFIG = {
    "architectures": ["Gemma2Model"],
    "model_type": "gemma2",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "tie_word_embeddings": True,
}

# KaLM-Embedding-v2.5: Qwen2 base model for embeddings
KALM_EMBEDDING_CONFIG = {
    "architectures": ["Qwen2Model"],
    "model_type": "qwen2",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
}

# --- Encoder-only models ---

# Multilingual-E5-tiny-instruct: XLM-RoBERTa encoder
MULTILINGUAL_E5_CONFIG = {
    "architectures": ["XLMRobertaModel"],
    "model_type": "xlm-roberta",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "hidden_act": "gelu",
    "type_vocab_size": 1,
}

# --- Custom models ---

# GLiNER2-multi-v1: custom "extractor" model_type (NER model)
# Uses custom_models.py loader; requires synthetic weights
GLINER2_CONFIG = {
    "architectures": ["ExtractorModel"],
    "model_type": "extractor",
    "hidden_size": 64,
    "max_width": 4,
    "num_rnn_layers": 1,
    "span_mode": "markerV0",
}

# --- Encoder-Decoder models ---

# T5Gemma2-270M: Gemma2-based encoder-decoder (multimodal)
# NOTE: T5Gemma2 has a complex nested config (encoder + decoder + vision tower)
# that requires special handling. This config is for conversion testing only.
T5GEMMA2_CONFIG = {
    "architectures": ["T5Gemma2ForConditionalGeneration"],
    "model_type": "t5gemma2",
    "is_encoder_decoder": True,
    "tie_word_embeddings": True,
    "vocab_size": 1000,
    # Sub-configs are set programmatically in _create_local_model
}


# ---- Registry of all test models ----

MODEL_CONFIGS = {
    # Decoder-only CausalLM
    "qwen3": QWEN3_CONFIG,
    "llama": LLAMA_CONFIG,
    "qwen3_tied": QWEN3_TIED_CONFIG,
    "qwen3_06b": QWEN3_06B_CONFIG,
    "qwen3_17b": QWEN3_17B_CONFIG,
    "granite_40": GRANITE_40_CONFIG,
    "lfm_700m": LFM_700M_CONFIG,
    # Embedding models (base without LM head)
    "qwen3_embedding": QWEN3_EMBEDDING_CONFIG,
    "embedding_gemma": EMBEDDING_GEMMA_CONFIG,
    "kalm_embedding": KALM_EMBEDDING_CONFIG,
    # Encoder-only
    "multilingual_e5": MULTILINGUAL_E5_CONFIG,
    # Custom models
    "gliner2": GLINER2_CONFIG,
    # Encoder-decoder
    "t5gemma2": T5GEMMA2_CONFIG,
}

# Mapping: config_name -> meson executable target name.
# Must match the entries in TorchFXConverter/jni/meson.build.
MODEL_BUILD_TARGETS = {
    "qwen3": "converter_qwen3_gen_test",
    "llama": "converter_llama_test",
    "qwen3_tied": "converter_qwen3_tied_test",
    "qwen3_06b": "converter_qwen3_06b_test",
    "qwen3_17b": "converter_qwen3_17b_test",
    "granite_40": "converter_granite_40_test",
    "lfm_700m": "converter_lfm_700m_test",
    "qwen3_embedding": "converter_qwen3_embedding_test",
    "embedding_gemma": "converter_embedding_gemma_test",
    "kalm_embedding": "converter_kalm_embedding_test",
    "multilingual_e5": "converter_multilingual_e5_test",
    "gliner2": "converter_gliner2_test",
}


# ---- Model creation helpers ----

def _create_local_model(config, model_dir):
    """Create a tiny local model using transformers."""
    import torch
    os.makedirs(model_dir, exist_ok=True)
    model_type = config["model_type"]
    architectures = config.get("architectures", [])
    arch = architectures[0] if architectures else ""

    try:
        # --- Decoder-only CausalLM ---
        if model_type == "qwen3" and arch == "Qwen3ForCausalLM":
            from transformers import Qwen3Config, Qwen3ForCausalLM
            cfg = Qwen3Config(**{k: v for k, v in config.items()
                                 if k not in ("architectures",)})
            model = Qwen3ForCausalLM(cfg)

        elif model_type == "llama" and arch == "LlamaForCausalLM":
            from transformers import LlamaConfig, LlamaForCausalLM
            cfg = LlamaConfig(**{k: v for k, v in config.items()
                                 if k not in ("architectures",)})
            model = LlamaForCausalLM(cfg)

        elif model_type == "granitemoehybrid":
            from transformers import (GraniteMoeHybridConfig,
                                      GraniteMoeHybridForCausalLM)
            cfg = GraniteMoeHybridConfig(**{
                k: v for k, v in config.items()
                if k not in ("architectures",)})
            cfg.architectures = config["architectures"]
            model = GraniteMoeHybridForCausalLM(cfg)

        elif model_type == "lfm2":
            from transformers import Lfm2Config, Lfm2ForCausalLM
            cfg = Lfm2Config(**{k: v for k, v in config.items()
                                if k not in ("architectures",)})
            cfg.architectures = config["architectures"]
            model = Lfm2ForCausalLM(cfg)

        # --- Embedding / base models ---
        elif model_type == "qwen3" and arch == "Qwen3Model":
            from transformers import Qwen3Config, Qwen3Model
            cfg = Qwen3Config(**{k: v for k, v in config.items()
                                 if k not in ("architectures",)})
            cfg.architectures = ["Qwen3Model"]
            model = Qwen3Model(cfg)

        elif model_type == "gemma2" and arch == "Gemma2Model":
            from transformers import Gemma2Config, Gemma2Model
            cfg = Gemma2Config(**{k: v for k, v in config.items()
                                  if k not in ("architectures",)})
            cfg.architectures = ["Gemma2Model"]
            model = Gemma2Model(cfg)

        elif model_type == "qwen2" and arch == "Qwen2Model":
            from transformers import Qwen2Config, Qwen2Model
            cfg = Qwen2Config(**{k: v for k, v in config.items()
                                 if k not in ("architectures",)})
            cfg.architectures = ["Qwen2Model"]
            model = Qwen2Model(cfg)

        # --- Encoder-only ---
        elif model_type == "xlm-roberta":
            from transformers import XLMRobertaConfig, XLMRobertaModel
            cfg = XLMRobertaConfig(**{k: v for k, v in config.items()
                                      if k not in ("architectures",)})
            cfg.architectures = config["architectures"]
            model = XLMRobertaModel(cfg)

        # --- Custom: GLiNER2 (extractor) ---
        elif model_type == "extractor":
            return _create_gliner2_model(config, model_dir)

        # --- Encoder-Decoder: T5Gemma2 ---
        elif model_type == "t5gemma2":
            return _create_t5gemma2_model(config, model_dir)

        else:
            raise ValueError(f"Unknown model type/arch: {model_type}/{arch}")

        model.save_pretrained(model_dir)
        return True

    except ImportError as e:
        print(f"  Import error: {e}")
        return False
    except Exception as e:
        print(f"  Error creating {model_type}: {e}")
        return False


def _create_gliner2_model(config, model_dir):
    """Create a synthetic GLiNER2 extractor model with fake weights."""
    import torch
    os.makedirs(model_dir, exist_ok=True)

    hidden_size = config["hidden_size"]
    max_width = config.get("max_width", 4)

    # Save config.json
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save gliner_config.json
    gliner_cfg = {
        "span_mode": config.get("span_mode", "markerV0"),
        "max_width": max_width,
        "num_rnn_layers": config.get("num_rnn_layers", 1),
    }
    with open(os.path.join(model_dir, "gliner_config.json"), "w") as f:
        json.dump(gliner_cfg, f, indent=2)

    # Create synthetic weights matching extractor structure
    # LSTM: bidirectional, 1 layer
    state = {}
    lstm_h = hidden_size
    # LSTM weight shapes: (4*hidden_size, input_size) for ih, (4*hidden_size, hidden_size) for hh
    state["rnn.lstm.weight_ih_l0"] = torch.randn(4 * lstm_h, hidden_size)
    state["rnn.lstm.bias_ih_l0"] = torch.randn(4 * lstm_h)
    state["rnn.lstm.weight_hh_l0"] = torch.randn(4 * lstm_h, lstm_h)
    state["rnn.lstm.bias_hh_l0"] = torch.randn(4 * lstm_h)
    # Reverse direction
    state["rnn.lstm.weight_ih_l0_reverse"] = torch.randn(4 * lstm_h, hidden_size)
    state["rnn.lstm.bias_ih_l0_reverse"] = torch.randn(4 * lstm_h)
    state["rnn.lstm.weight_hh_l0_reverse"] = torch.randn(4 * lstm_h, lstm_h)
    state["rnn.lstm.bias_hh_l0_reverse"] = torch.randn(4 * lstm_h)

    # Span marker: project_start, project_end, out_project
    # Each is a 2-layer MLP: Linear(D, D) -> ReLU -> Linear(D, D)
    D = hidden_size
    for proj in ["project_start", "project_end"]:
        state[f"span_rep.{proj}.0.weight"] = torch.randn(D, D)
        state[f"span_rep.{proj}.0.bias"] = torch.randn(D)
        state[f"span_rep.{proj}.3.weight"] = torch.randn(D, D)
        state[f"span_rep.{proj}.3.bias"] = torch.randn(D)
    # out_project: Linear(2*D, D) — output per-span features that get reshaped
    state["span_rep.out_project.0.weight"] = torch.randn(D, 2 * D)
    state["span_rep.out_project.0.bias"] = torch.randn(D)

    # Classifier variant (GLiNER2 uses classifier, not prompt)
    state["classifier.0.weight"] = torch.randn(32, D)
    state["classifier.0.bias"] = torch.randn(32)
    state["classifier.3.weight"] = torch.randn(1, 32)
    state["classifier.3.bias"] = torch.randn(1)

    torch.save(state, os.path.join(model_dir, "pytorch_model.bin"))
    return True


def _create_t5gemma2_model(config, model_dir):
    """Create a tiny T5Gemma2 encoder-decoder model."""
    try:
        from transformers import (T5Gemma2Config,
                                  T5Gemma2ForConditionalGeneration)
    except ImportError:
        return False

    vocab_size = config.get("vocab_size", 1000)

    cfg = T5Gemma2Config(
        vocab_size=vocab_size,
        tie_word_embeddings=True,
    )
    # Override encoder to be tiny (text-only, no vision)
    cfg.encoder.hidden_size = 64
    cfg.encoder.intermediate_size = 128
    cfg.encoder.num_hidden_layers = 2
    cfg.encoder.num_attention_heads = 4
    cfg.encoder.num_key_value_heads = 2
    cfg.encoder.head_dim = 16
    cfg.encoder.vocab_size = vocab_size
    # Decoder
    cfg.decoder.hidden_size = 64
    cfg.decoder.intermediate_size = 128
    cfg.decoder.num_hidden_layers = 2
    cfg.decoder.num_attention_heads = 4
    cfg.decoder.num_key_value_heads = 2
    cfg.decoder.head_dim = 16
    cfg.decoder.vocab_size = vocab_size

    try:
        model = T5Gemma2ForConditionalGeneration(cfg)
        model.save_pretrained(model_dir)
        return True
    except Exception as e:
        print(f"  T5Gemma2 creation failed: {e}")
        return False


# ---- Test infrastructure ----

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


def _meson_reconfigure(build_dir):
    """Reconfigure meson to pick up new/removed source files."""
    cmd = ["meson", "setup", "--reconfigure", build_dir]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=120, cwd=NNTRAINER_ROOT)
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

    # ---- Decoder-only CausalLM ----

    def test_qwen3_tiny_build_and_run(self):
        """Qwen3 tiny model: convert -> build -> initialize."""
        self._run_model_test("qwen3")

    def test_llama_tiny_build_and_run(self):
        """LLaMA tiny model: convert -> build -> initialize."""
        self._run_model_test("llama")

    def test_qwen3_tied_embeddings(self):
        """Qwen3 with tied word embeddings."""
        self._run_model_test("qwen3_tied")

    def test_qwen3_06b(self):
        """Qwen3-0.6B style: tied embeddings, high rope_theta."""
        self._run_model_test("qwen3_06b")

    def test_qwen3_17b(self):
        """Qwen3-1.7B style: more heads, larger intermediate."""
        self._run_model_test("qwen3_17b")

    def test_granite_40(self):
        """Granite 4.0: GraniteMoeHybrid in dense mode (no MoE/Mamba)."""
        self._run_model_test("granite_40")

    def test_lfm_700m(self):
        """LFM-700M: Liquid Foundation Model (lfm2) with full-attention."""
        self._run_model_test("lfm_700m")

    # ---- Embedding models ----

    def test_qwen3_embedding(self):
        """Qwen3-Embedding-0.6B: Qwen3 base model for embeddings."""
        self._run_model_test("qwen3_embedding")

    def test_embedding_gemma(self):
        """EmbeddingGemma-300M: Gemma2 base model for embeddings."""
        self._run_model_test("embedding_gemma")

    def test_kalm_embedding(self):
        """KaLM-Embedding-v2.5: Qwen2 base model for embeddings."""
        self._run_model_test("kalm_embedding")

    # ---- Encoder-only ----

    def test_multilingual_e5(self):
        """Multilingual-E5-tiny-instruct: XLM-RoBERTa encoder."""
        self._run_model_test("multilingual_e5")

    # ---- Custom models ----

    def test_gliner2(self):
        """GLiNER2-multi-v1: custom extractor model (NER)."""
        self._run_model_test("gliner2")

    # ---- Encoder-Decoder ----

    @unittest.skip("T5Gemma2 is multimodal (text+vision) with deeply nested "
                   "configs; cannot create a memory-efficient tiny model")
    def test_t5gemma2(self):
        """T5Gemma2-270M: Gemma2-based encoder-decoder."""
        self._run_model_test("t5gemma2")

    # ---- Common pipeline ----

    def _run_model_test(self, config_name):
        """Run full pipeline: create model -> convert -> build -> run.

        Steps:
          1. Create a tiny HuggingFace model in a temp directory
          2. Run converter.py to generate C++ header + source
          3. Copy generated files to jni/ directory
          4. Reconfigure meson to detect the new source files
          5. Build the model-specific test executable with ninja
          6. Run the executable and verify NNTrainer initializes it
        """
        config = MODEL_CONFIGS[config_name]
        generated_files = []

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                model_dir = os.path.join(tmpdir, "model")
                output_dir = os.path.join(tmpdir, "output")

                # Step 1: Create local model
                if not _create_local_model(config, model_dir):
                    self.skipTest("transformers not available or model "
                                  "creation failed")

                # Step 2: Convert
                ok, stdout, stderr = _run_converter(
                    model_dir, output_dir, model_name=config_name)
                self.assertTrue(ok, f"Converter failed:\n{stderr}")

                # Step 3: Verify output files exist
                cpp_files = [f for f in os.listdir(output_dir)
                             if f.endswith(('.h', '.cpp'))]
                self.assertGreaterEqual(len(cpp_files), 2,
                                        f"Expected .h and .cpp, got: "
                                        f"{cpp_files}")

                print(f"\n[{config_name}] Converter output:")
                print(stdout)

                # Step 4: Copy generated C++ files to jni/ for meson build
                if config_name not in MODEL_BUILD_TARGETS:
                    return  # no build target defined, conversion-only test

                for f in cpp_files:
                    src = os.path.join(output_dir, f)
                    dst = os.path.join(JNI_DIR, f)
                    shutil.copy2(src, dst)
                    generated_files.append(dst)

            # Step 5: Reconfigure meson to pick up the new files
            ok, stdout, stderr = _meson_reconfigure(BUILD_DIR)
            self.assertTrue(ok,
                            f"meson reconfigure failed:\n{stderr[-500:]}")

            # Step 6: Build the test executable
            target_name = MODEL_BUILD_TARGETS[config_name]
            target = (f"Applications/TorchFXConverter/jni/"
                      f"{target_name}")
            ok, stdout, stderr = _build_test(BUILD_DIR, target)
            self.assertTrue(ok, f"Build failed for {config_name}:\n"
                            f"stdout: {stdout[-1000:]}\n"
                            f"stderr: {stderr[-1000:]}")

            # Step 7: Run the executable
            ok, stdout, stderr = _run_test(BUILD_DIR, target)
            self.assertTrue(ok,
                            f"Run failed for {config_name}:\n"
                            f"stdout: {stdout}\nstderr: {stderr}")
            self.assertIn("Model initialized successfully", stdout)

        finally:
            # Cleanup: remove generated files from jni/
            for f in generated_files:
                if os.path.exists(f):
                    os.remove(f)

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

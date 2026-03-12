#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for generic model_inspector.py.

Validates that the generic inspector correctly handles both Qwen3 and Gemma3
model architectures without any model-specific hardcoding.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from model_inspector import (
    NNTrainerLayer,
    DecoderLayerInfo,
    ModelStructure,
    inspect_model,
    generate_layers,
    generate_weight_order,
    generate_nntr_config,
    generate_weight_converter_script,
    _classify_module,
)


# ============================================================
# Mock HuggingFace modules (must have correct class names for detection)
# ============================================================

class Qwen3RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x


class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, has_qk_norm=True):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        if has_qk_norm:
            self.q_norm = Qwen3RMSNorm(head_dim)
            self.k_norm = Qwen3RMSNorm(head_dim)

    def forward(self, x):
        return x


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return x


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.self_attn = Qwen3Attention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)
        self.input_layernorm = Qwen3RMSNorm(hidden_size)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size)

    def forward(self, x):
        return x


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                config["hidden_size"], config["intermediate_size"],
                config["num_attention_heads"], config["num_key_value_heads"],
                config["head_dim"],
            )
            for _ in range(config["num_hidden_layers"])
        ])
        self.norm = Qwen3RMSNorm(config["hidden_size"])

    def forward(self, x):
        return x


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.model = Qwen3Model(config_dict)
        self.config = type("Config", (), config_dict)()
        if not config_dict.get("tie_word_embeddings", False):
            self.lm_head = nn.Linear(config_dict["hidden_size"],
                                      config_dict["vocab_size"], bias=False)

    def forward(self, x):
        return x


# ---------- Gemma3 mock ----------

class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x


class Gemma3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Gemma3RMSNorm(head_dim)
        self.k_norm = Gemma3RMSNorm(head_dim)

    def forward(self, x):
        return x


class GELUTanh(nn.Module):
    """Mock tanh GELU activation (matches HF's NewGELUActivation or similar)."""
    def forward(self, x):
        return x


class Gemma3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = GELUTanh()

    def forward(self, x):
        return x


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.self_attn = Gemma3Attention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = Gemma3MLP(hidden_size, intermediate_size)
        self.input_layernorm = Gemma3RMSNorm(hidden_size)
        self.post_attention_layernorm = Gemma3RMSNorm(hidden_size)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(hidden_size)
        self.post_feedforward_layernorm = Gemma3RMSNorm(hidden_size)

    def forward(self, x):
        return x


class Gemma3TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            Gemma3DecoderLayer(
                config["hidden_size"], config["intermediate_size"],
                config["num_attention_heads"], config["num_key_value_heads"],
                config["head_dim"],
            )
            for _ in range(config["num_hidden_layers"])
        ])
        self.norm = Gemma3RMSNorm(config["hidden_size"])

    def forward(self, x):
        return x


class Gemma3ForCausalLM(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.model = Gemma3TextModel(config_dict)
        cfg = type("Config", (), {
            **config_dict,
            "text_config": None,  # no nested text_config for simplicity
        })()
        self.config = cfg

    def forward(self, x):
        return x


# ============================================================
# Helper: create mock models
# ============================================================

def make_qwen3_model(num_layers=2):
    config = {
        "vocab_size": 1000,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 256,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "sliding_window": 32768,
    }
    return Qwen3ForCausalLM(config)


def make_gemma3_model(num_layers=2):
    config = {
        "vocab_size": 2000,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "max_position_embeddings": 512,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "sliding_window": 512,
        "layer_types": ["sliding_attention", "full_attention"] * (num_layers // 2),
        "attn_logit_softcapping": 50.0,
    }
    return Gemma3ForCausalLM(config)


# ============================================================
# Tests
# ============================================================

def test_classify_module():
    """Test module classification detects types correctly."""
    print("=== Test: classify_module ===")

    assert _classify_module("x", nn.Linear(4, 4)) == "linear"
    assert _classify_module("x", nn.Embedding(10, 4)) == "embedding"
    assert _classify_module("x", Qwen3RMSNorm(4)) == "rmsnorm"
    assert _classify_module("x", Gemma3RMSNorm(4)) == "rmsnorm"
    assert _classify_module("x", nn.SiLU()) == "activation:swish"
    assert _classify_module("x", GELUTanh()) == "activation:tanh_gelu"
    assert _classify_module("x", Qwen3DecoderLayer(64, 128, 4, 2, 16)) == "decoder_layer"
    assert _classify_module("x", Gemma3DecoderLayer(64, 128, 4, 2, 16)) == "decoder_layer"
    assert _classify_module("x", Qwen3Attention(64, 4, 2, 16)) == "attention_block"
    assert _classify_module("x", Qwen3MLP(64, 128)) == "mlp_block"

    print("PASSED\n")


def test_inspect_qwen3():
    """Test inspection of Qwen3-style model."""
    print("=== Test: inspect_model (Qwen3) ===")

    model = make_qwen3_model(num_layers=2)
    structure = inspect_model(model)

    assert structure.hidden_size == 64
    assert structure.intermediate_size == 128
    assert structure.num_attention_heads == 4
    assert structure.num_key_value_heads == 2
    assert structure.head_dim == 16
    assert structure.vocab_size == 1000
    assert structure.tie_word_embeddings is True
    assert structure.num_layers == 2
    assert structure.embed_module == "model.embed_tokens"
    assert structure.final_norm == "model.norm"
    assert structure.lm_head == ""  # tied embeddings, no separate lm_head

    # Check decoder layer info
    layer0 = structure.layers[0]
    assert layer0.has_qk_norm is True
    assert layer0.has_post_attn_norm is False
    assert layer0.has_pre_ffn_norm is False
    assert layer0.has_post_ffn_norm is False
    assert layer0.mlp_activation == "swish"
    assert layer0.q_proj.endswith("q_proj")
    assert layer0.k_proj.endswith("k_proj")
    assert layer0.v_proj.endswith("v_proj")
    assert layer0.o_proj.endswith("o_proj")
    assert layer0.q_norm.endswith("q_norm")
    assert layer0.k_norm.endswith("k_norm")
    assert layer0.has_bias is False

    print("  Structure detected correctly")
    print("PASSED\n")


def test_inspect_gemma3():
    """Test inspection of Gemma3-style model."""
    print("=== Test: inspect_model (Gemma3) ===")

    model = make_gemma3_model(num_layers=2)
    structure = inspect_model(model)

    assert structure.hidden_size == 128
    assert structure.intermediate_size == 256
    assert structure.num_attention_heads == 8
    assert structure.num_key_value_heads == 4
    assert structure.head_dim == 16
    assert structure.vocab_size == 2000
    assert structure.tie_word_embeddings is True
    assert structure.num_layers == 2
    assert structure.attn_logit_softcapping == 50.0
    assert structure.layer_types == ["sliding_attention", "full_attention"]

    # Check decoder layer info
    layer0 = structure.layers[0]
    assert layer0.has_qk_norm is True
    assert layer0.has_post_attn_norm is True  # Gemma3 has post-attention norm
    assert layer0.has_pre_ffn_norm is True   # Gemma3 has pre-feedforward norm
    assert layer0.has_post_ffn_norm is True  # Gemma3 has post-feedforward norm
    assert layer0.mlp_activation == "tanh_gelu"

    print("  Gemma3 structure detected correctly")
    print("PASSED\n")


def test_generate_layers_qwen3():
    """Test NNTrainer layer generation for Qwen3."""
    print("=== Test: generate_layers (Qwen3) ===")

    model = make_qwen3_model(num_layers=2)
    structure = inspect_model(model)
    layers = generate_layers(structure)

    # Print all layers
    for i, layer in enumerate(layers):
        print(f"  [{i:3d}] {layer.layer_type:25s} {layer.name}")

    # Qwen3 decoder block has 15 layers (same as qwen3_converter)
    # input(1) + embedding(1) + 2*15 + output_norm(1) + lm_head(1) = 34
    expected_total = 1 + 1 + 2 * 15 + 1 + 1
    assert len(layers) == expected_total, f"Expected {expected_total}, got {len(layers)}"

    # Check first/last layers
    assert layers[0].layer_type == "input"
    assert layers[1].layer_type == "tie_word_embeddings"
    assert layers[-2].layer_type == "rms_norm"
    assert layers[-2].name == "output_norm"
    assert layers[-1].layer_type == "tie_word_embeddings"
    assert layers[-1].name == "output_of_causallm"
    assert layers[-1].params["shared_from"] == "embedding0"

    # Check decoder block 0 structure
    block_start = 2
    expected_block = [
        ("rms_norm", "layer0_attention_norm"),
        ("fully_connected", "layer0_wv"),
        ("fully_connected", "layer0_wk"),
        ("reshaped_rms_norm", "layer0_k_norm"),
        ("fully_connected", "layer0_wq"),
        ("reshaped_rms_norm", "layer0_q_norm"),
        ("mha_core", "layer0_attention"),
        ("fully_connected", "layer0_attention_out"),
        ("addition", "layer0_decoder_add"),
        ("rms_norm", "layer0_ffn_norm"),
        ("fully_connected", "layer0_ffn_up"),
        ("fully_connected", "layer0_ffn_gate"),
        ("swiglu", "layer0_ffn_swiglu"),
        ("fully_connected", "layer0_ffn_down"),
        ("addition", "layer0_decoder_output"),
    ]

    for j, (exp_type, exp_name) in enumerate(expected_block):
        actual = layers[block_start + j]
        assert actual.layer_type == exp_type, (
            f"Block[{j}]: expected {exp_type}, got {actual.layer_type}"
        )
        assert actual.name == exp_name, (
            f"Block[{j}]: expected {exp_name}, got {actual.name}"
        )

    print("PASSED\n")


def test_generate_layers_gemma3():
    """Test NNTrainer layer generation for Gemma3."""
    print("=== Test: generate_layers (Gemma3) ===")

    model = make_gemma3_model(num_layers=2)
    structure = inspect_model(model)
    layers = generate_layers(structure)

    # Print all layers
    for i, layer in enumerate(layers):
        print(f"  [{i:3d}] {layer.layer_type:25s} {layer.name}")

    # Gemma3 decoder block structure (from gemma3_causallm.cpp):
    # attention_norm, V, K, K_norm, Q, Q_norm, MHA, O,
    # post_attention_norm, addition(residual),
    # pre_ffn_norm,
    # gate, activation, up, multiply, down,
    # post_ffn_norm, addition(residual)
    # = 19 layers per block
    expected_block = [
        ("rms_norm", "layer0_attention_norm"),
        ("fully_connected", "layer0_wv"),
        ("fully_connected", "layer0_wk"),
        ("reshaped_rms_norm", "layer0_k_norm"),
        ("fully_connected", "layer0_wq"),
        ("reshaped_rms_norm", "layer0_q_norm"),
        ("mha_core", "layer0_attention"),
        ("fully_connected", "layer0_attention_out"),
        ("rms_norm", "layer0_post_attention_norm"),
        ("addition", "layer0_post_attention"),
        ("rms_norm", "layer0pre_ffn_norm"),
        ("fully_connected", "layer0_ffn_gate"),
        ("activation", "layer0_ffn_gate_gelu"),
        ("fully_connected", "layer0_ffn_up"),
        ("multiply", "layer0_ffn_geglu"),
        ("fully_connected", "layer0_ffn_down"),
        ("rms_norm", "layer0post_ffn_norm"),
        ("addition", "layer0_decoder_output"),
    ]

    block_start = 2  # after input + embedding
    for j, (exp_type, exp_name) in enumerate(expected_block):
        actual = layers[block_start + j]
        assert actual.layer_type == exp_type, (
            f"Block[{j}]: expected {exp_type}, got {actual.layer_type}"
        )
        assert actual.name == exp_name, (
            f"Block[{j}]: expected {exp_name}, got {actual.name}"
        )

    # Total: input(1) + emb(1) + 2*18 + norm(1) + lm_head(1) = 40
    expected_total = 1 + 1 + 2 * len(expected_block) + 1 + 1
    assert len(layers) == expected_total, f"Expected {expected_total}, got {len(layers)}"

    print("PASSED\n")


def test_layer_connectivity_qwen3():
    """Verify input_layers connections for Qwen3."""
    print("=== Test: layer connectivity (Qwen3) ===")

    model = make_qwen3_model(num_layers=2)
    structure = inspect_model(model)
    layers = generate_layers(structure)
    name_map = {l.name: l for l in layers}

    # Block 0
    assert name_map["layer0_attention_norm"].params["input_layers"] == "embedding0"
    assert name_map["layer0_wv"].params["input_layers"] == "layer0_attention_norm"
    assert name_map["layer0_wk"].params["input_layers"] == "layer0_attention_norm"
    assert name_map["layer0_k_norm"].params["input_layers"] == "layer0_wk"
    assert name_map["layer0_wq"].params["input_layers"] == "layer0_attention_norm"
    assert name_map["layer0_q_norm"].params["input_layers"] == "layer0_wq"
    assert name_map["layer0_attention_out"].params["input_layers"] == "layer0_attention"
    assert name_map["layer0_decoder_add"].params["input_layers"] == "embedding0,layer0_attention_out"
    assert name_map["layer0_ffn_norm"].params["input_layers"] == "layer0_decoder_add"
    assert name_map["layer0_ffn_up"].params["input_layers"] == "layer0_ffn_norm"
    assert name_map["layer0_ffn_gate"].params["input_layers"] == "layer0_ffn_norm"
    assert name_map["layer0_ffn_swiglu"].params["input_layers"] == "layer0_ffn_up,layer0_ffn_gate"
    assert name_map["layer0_ffn_down"].params["input_layers"] == "layer0_ffn_swiglu"
    assert name_map["layer0_decoder_output"].params["input_layers"] == "layer0_decoder_add,layer0_ffn_down"

    # Block 1 connects to block 0 output
    assert name_map["layer1_attention_norm"].params["input_layers"] == "layer0_decoder_output"

    # Output
    assert name_map["output_norm"].params["input_layers"] == "layer1_decoder_output"
    assert name_map["output_of_causallm"].params["input_layers"] == "output_norm"

    print("PASSED\n")


def test_layer_connectivity_gemma3():
    """Verify input_layers connections for Gemma3."""
    print("=== Test: layer connectivity (Gemma3) ===")

    model = make_gemma3_model(num_layers=2)
    structure = inspect_model(model)
    layers = generate_layers(structure)
    name_map = {l.name: l for l in layers}

    # Block 0 - Gemma3 pattern
    assert name_map["layer0_attention_norm"].params["input_layers"] == "embedding0"
    assert name_map["layer0_attention_out"].params["input_layers"] == "layer0_attention"
    assert name_map["layer0_post_attention_norm"].params["input_layers"] == "layer0_attention_out"
    assert name_map["layer0_post_attention"].params["input_layers"] == "embedding0,layer0_post_attention_norm"
    assert name_map["layer0pre_ffn_norm"].params["input_layers"] == "layer0_post_attention"
    assert name_map["layer0_ffn_gate"].params["input_layers"] == "layer0pre_ffn_norm"
    assert name_map["layer0_ffn_gate_gelu"].params["input_layers"] == "layer0_ffn_gate"
    assert name_map["layer0_ffn_up"].params["input_layers"] == "layer0pre_ffn_norm"
    assert name_map["layer0_ffn_geglu"].params["input_layers"] == "layer0_ffn_gate_gelu,layer0_ffn_up"
    assert name_map["layer0_ffn_down"].params["input_layers"] == "layer0_ffn_geglu"
    assert name_map["layer0_decoder_output"].params["input_layers"] == "layer0_post_attention,layer0post_ffn_norm"

    print("PASSED\n")


def test_gemma3_sliding_window():
    """Verify Gemma3 per-layer sliding window and rope_theta."""
    print("=== Test: Gemma3 sliding window ===")

    model = make_gemma3_model(num_layers=2)
    structure = inspect_model(model)
    layers = generate_layers(structure)
    name_map = {l.name: l for l in layers}

    # Layer 0: sliding_attention → use sliding_window, rope_theta=10000
    mha0 = name_map["layer0_attention"]
    assert mha0.params["sliding_window"] == 512
    assert mha0.params["rope_theta"] == 10000  # sliding uses local rope

    # Layer 1: full_attention → UINT_MAX, global rope_theta
    mha1 = name_map["layer1_attention"]
    assert mha1.params["sliding_window"] == 4294967295  # UINT_MAX
    assert mha1.params["rope_theta"] == 500000  # global rope

    # attn_logit_softcapping
    assert mha0.params["attn_logit_softcapping"] == "50.0"
    assert mha1.params["attn_logit_softcapping"] == "50.0"

    print("PASSED\n")


def test_weight_order_qwen3():
    """Verify weight order for Qwen3 matches layer creation order."""
    print("=== Test: weight_order (Qwen3) ===")

    model = make_qwen3_model(num_layers=2)
    structure = inspect_model(model)
    weights = generate_weight_order(structure)

    for w in weights:
        t = " (T)" if w["transpose"] else ""
        print(f"  {w['hf_key']:55s} -> {w['nntr_name']}{t}")

    # First: embedding
    assert weights[0]["hf_key"] == "model.embed_tokens.weight"
    assert weights[0]["transpose"] is False

    # Layer 0: norm, V, K, K_norm, Q, Q_norm, O, ffn_norm, up, gate, down
    layer0 = [w for w in weights if "layers.0." in w["hf_key"]]
    expected_suffixes = [
        "input_layernorm.weight",
        "self_attn.v_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.k_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.up_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
    ]
    assert len(layer0) == len(expected_suffixes), (
        f"Expected {len(expected_suffixes)} weights, got {len(layer0)}"
    )
    for i, w in enumerate(layer0):
        assert w["hf_key"].endswith(expected_suffixes[i]), (
            f"Weight {i}: expected ...{expected_suffixes[i]}, got {w['hf_key']}"
        )

    # Last: final norm (no lm_head because tied)
    assert weights[-1]["hf_key"] == "model.norm.weight"

    # Transpose checks
    for w in weights:
        if "proj.weight" in w["hf_key"]:
            assert w["transpose"] is True, f"{w['hf_key']} should be transposed"
        elif "norm" in w["hf_key"] or "embed" in w["hf_key"]:
            assert w["transpose"] is False, f"{w['hf_key']} should NOT be transposed"

    print("PASSED\n")


def test_weight_order_gemma3():
    """Verify weight order for Gemma3 (extra norms)."""
    print("=== Test: weight_order (Gemma3) ===")

    model = make_gemma3_model(num_layers=2)
    structure = inspect_model(model)
    weights = generate_weight_order(structure)

    for w in weights:
        t = " (T)" if w["transpose"] else ""
        print(f"  {w['hf_key']:60s} -> {w['nntr_name']}{t}")

    # Gemma3 layer 0 weight order:
    # input_layernorm, V, K, K_norm, Q, Q_norm, O,
    # post_attention_layernorm, pre_feedforward_layernorm,
    # gate, up, down, post_feedforward_layernorm
    layer0 = [w for w in weights if "layers.0." in w["hf_key"]]
    expected_suffixes = [
        "input_layernorm.weight",
        "self_attn.v_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.k_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "mlp.gate_proj.weight",  # GELU: gate first
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "post_feedforward_layernorm.weight",
    ]
    assert len(layer0) == len(expected_suffixes), (
        f"Expected {len(expected_suffixes)} weights, got {len(layer0)}"
    )
    for i, w in enumerate(layer0):
        assert w["hf_key"].endswith(expected_suffixes[i]), (
            f"Weight {i}: expected ...{expected_suffixes[i]}, got {w['hf_key']}"
        )

    print("PASSED\n")


def test_nntr_config_generation():
    """Verify nntr_config.json generation."""
    print("=== Test: nntr_config generation ===")

    model = make_qwen3_model()
    structure = inspect_model(model)
    config = generate_nntr_config(
        structure,
        model_file="nntr_model.bin",
        tokenizer_file="/path/tokenizer.json",
        sample_input="Hello",
    )

    assert config["model_type"] == "CausalLM"
    assert config["model_file_name"] == "nntr_model.bin"
    assert config["init_seq_len"] == 1024
    assert config["num_to_generate"] == 512
    assert config["tokenizer_file"] == "/path/tokenizer.json"
    assert config["sample_input"] == "Hello"

    print("PASSED\n")


def test_weight_converter_script():
    """Verify weight converter script generation."""
    print("=== Test: weight_converter_script ===")

    model = make_qwen3_model()
    structure = inspect_model(model)
    script = generate_weight_converter_script(structure, "Qwen3-tiny")

    assert "import torch" in script
    assert "import numpy as np" in script
    assert "permute(1, 0)" in script
    assert "model.embed_tokens.weight" in script
    assert "self_attn.q_proj.weight" in script

    print(f"  Script length: {len(script)} chars")
    print("PASSED\n")


def test_cpp_code_generation():
    """Verify C++ code generation from NNTrainerLayer."""
    print("=== Test: C++ code generation ===")

    layer = NNTrainerLayer(
        layer_type="fully_connected",
        name="layer0_wq",
        params={"unit": 128, "disable_bias": "true", "input_layers": "layer0_norm"},
    )
    cpp = layer.to_cpp_create_layer()
    assert '"fully_connected"' in cpp
    assert '"unit"' in cpp
    assert '"disable_bias"' in cpp

    layer2 = NNTrainerLayer(
        layer_type="mha_core",
        name="layer0_attention",
        params={"input_layers": ["q", "k", "v"], "num_heads": 4},
    )
    cpp2 = layer2.to_cpp_create_layer()
    assert '"mha_core"' in cpp2
    assert "{q,k,v}" in cpp2

    print("PASSED\n")


def test_to_dict():
    """Verify to_dict serialization."""
    print("=== Test: to_dict ===")

    layer = NNTrainerLayer(
        layer_type="rms_norm", name="norm0",
        params={"epsilon": "1e-6", "packed": "false"},
    )
    d = layer.to_dict()
    assert d["type"] == "rms_norm"
    assert d["name"] == "norm0"
    assert d["epsilon"] == "1e-6"

    print("PASSED\n")


def test_non_tied_embeddings():
    """Verify correct handling when tie_word_embeddings=False."""
    print("=== Test: non-tied embeddings ===")

    config = {
        "vocab_size": 1000,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
    }
    model = Qwen3ForCausalLM(config)
    structure = inspect_model(model)

    assert structure.tie_word_embeddings is False
    assert structure.lm_head == "lm_head"

    layers = generate_layers(structure)
    assert layers[1].layer_type == "embedding_layer"  # not tie_word_embeddings
    assert layers[-1].layer_type == "lm_head"  # not tie_word_embeddings
    assert "shared_from" not in layers[-1].params

    # Weight order should include lm_head
    weights = generate_weight_order(structure)
    assert weights[-1]["hf_key"] == "lm_head.weight"
    assert weights[-1]["transpose"] is True

    print("PASSED\n")


if __name__ == "__main__":
    test_classify_module()
    test_inspect_qwen3()
    test_inspect_gemma3()
    test_generate_layers_qwen3()
    test_generate_layers_gemma3()
    test_layer_connectivity_qwen3()
    test_layer_connectivity_gemma3()
    test_gemma3_sliding_window()
    test_weight_order_qwen3()
    test_weight_order_gemma3()
    test_nntr_config_generation()
    test_weight_converter_script()
    test_cpp_code_generation()
    test_to_dict()
    test_non_tied_embeddings()
    print("=" * 50)
    print("All model_inspector tests PASSED!")

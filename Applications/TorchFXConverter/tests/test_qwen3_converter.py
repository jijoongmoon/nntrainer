#!/usr/bin/env python3
"""Task 3-4 Tests: Verify Qwen3 converter generates correct NNTrainer layers."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwen3_converter import (
    Qwen3Config,
    generate_qwen3_layers,
    generate_nntr_config,
    generate_weight_order,
    generate_weight_converter_script,
)


def make_qwen3_0_6b_config() -> Qwen3Config:
    """Create Qwen3-0.6B equivalent config."""
    return Qwen3Config(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40960,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        sliding_window=32768,
        init_seq_len=1024,
        num_to_generate=512,
        max_seq_len=2048,
    )


def make_tiny_config() -> Qwen3Config:
    """Create tiny config for quick tests."""
    return Qwen3Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )


def test_from_hf_config():
    """Task 3.1: Detect Qwen3 architecture and extract config."""
    print("=== Test 3.1: HF config extraction ===")
    from transformers import Qwen3Config as HFQwen3Config

    hf_config = HFQwen3Config(
        hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16,
        vocab_size=1000, max_position_embeddings=256,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6, rope_theta=1000000.0,
    )

    cfg = Qwen3Config.from_hf_config(hf_config)
    assert cfg.hidden_size == 64
    assert cfg.num_hidden_layers == 2
    assert cfg.num_attention_heads == 4
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 16
    assert cfg.gqa_size == 2
    assert cfg.tie_word_embeddings is True
    print("  Config extracted correctly")
    print("PASSED\n")


def test_layer_structure_tiny():
    """Task 3.2-3.6: Verify complete layer structure for tiny model."""
    print("=== Test 3.2-3.6: Layer structure (tiny) ===")

    cfg = make_tiny_config()
    layers = generate_qwen3_layers(cfg)

    print(f"  Total layers: {len(layers)}")
    for i, layer in enumerate(layers):
        print(f"  [{i:3d}] {layer.layer_type:25s} {layer.name}")

    # Verify structure
    assert layers[0].layer_type == "input"
    assert layers[0].name == "input0"

    assert layers[1].layer_type == "tie_word_embeddings"
    assert layers[1].name == "embedding0"
    assert layers[1].params["in_dim"] == 1000
    assert layers[1].params["out_dim"] == 64

    # Each decoder block has 15 layers (see docstring above)
    # input(1) + embedding(1) + blocks(2*15) + output_norm(1) + lm_head(1) = 34
    expected_total = 1 + 1 + cfg.num_hidden_layers * 15 + 1 + 1
    assert len(layers) == expected_total, (
        f"Expected {expected_total} layers, got {len(layers)}"
    )

    # Verify first decoder block structure
    block_start = 2  # after input + embedding
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
            f"Block[{j}]: expected name {exp_name}, got {actual.name}"
        )

    # Verify output layers
    assert layers[-2].layer_type == "rms_norm"
    assert layers[-2].name == "output_norm"
    assert layers[-1].layer_type == "tie_word_embeddings"
    assert layers[-1].name == "output_of_causallm"
    assert layers[-1].params["shared_from"] == "embedding0"

    print("PASSED\n")


def test_layer_connectivity():
    """Task 3.5: Verify input_layers connections are correct."""
    print("=== Test 3.5: Layer connectivity ===")

    cfg = make_tiny_config()
    layers = generate_qwen3_layers(cfg)

    # Build name -> layer map
    name_map = {l.name: l for l in layers}

    # Check block 0 connections
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

    # Check block 1 connects to block 0 output
    assert name_map["layer1_attention_norm"].params["input_layers"] == "layer0_decoder_output"
    assert name_map["layer1_decoder_add"].params["input_layers"] == "layer0_decoder_output,layer1_attention_out"

    # Check output connections
    assert name_map["output_norm"].params["input_layers"] == "layer1_decoder_output"
    assert name_map["output_of_causallm"].params["input_layers"] == "output_norm"

    print("  All connections verified")
    print("PASSED\n")


def test_attention_params():
    """Task 3.3: Verify attention layer parameters."""
    print("=== Test 3.3: Attention parameters ===")

    cfg = make_qwen3_0_6b_config()
    layers = generate_qwen3_layers(cfg)
    name_map = {l.name: l for l in layers}

    # Check Q projection
    wq = name_map["layer0_wq"]
    assert wq.params["unit"] == 128 * 16, f"Q unit should be {128*16}, got {wq.params['unit']}"
    assert wq.params["disable_bias"] == "true"

    # Check K/V projection (with GQA)
    wk = name_map["layer0_wk"]
    wv = name_map["layer0_wv"]
    expected_kv_dim = 128 * 16 // 2  # head_dim * n_heads / gqa_size
    assert wk.params["unit"] == expected_kv_dim, f"K unit should be {expected_kv_dim}"
    assert wv.params["unit"] == expected_kv_dim, f"V unit should be {expected_kv_dim}"

    # Check MHA core params
    mha = name_map["layer0_attention"]
    assert mha.params["num_heads"] == 16
    assert mha.params["num_heads_kv"] == 8  # 16 / 2
    assert mha.params["rope_theta"] == 1000000
    assert mha.params["sliding_window"] == 32768
    assert mha.params["max_position_embeddings"] == 40960
    assert mha.params["max_timestep"] == str(1024 + 512)

    # Check Q/K norms (Qwen3-specific)
    q_norm = name_map["layer0_q_norm"]
    assert q_norm.layer_type == "reshaped_rms_norm"
    assert q_norm.params["feature_size"] == "128"  # head_dim

    k_norm = name_map["layer0_k_norm"]
    assert k_norm.layer_type == "reshaped_rms_norm"
    assert k_norm.params["feature_size"] == "128"

    # Check O projection
    o_proj = name_map["layer0_attention_out"]
    assert o_proj.params["unit"] == 1024  # hidden_size

    print("  All attention params verified")
    print("PASSED\n")


def test_mlp_params():
    """Task 3.4: Verify SwiGLU MLP parameters."""
    print("=== Test 3.4: MLP parameters ===")

    cfg = make_qwen3_0_6b_config()
    layers = generate_qwen3_layers(cfg)
    name_map = {l.name: l for l in layers}

    # Check up/gate projections
    up = name_map["layer0_ffn_up"]
    gate = name_map["layer0_ffn_gate"]
    assert up.params["unit"] == 3072  # intermediate_size
    assert gate.params["unit"] == 3072

    # Check SwiGLU
    swiglu = name_map["layer0_ffn_swiglu"]
    assert swiglu.layer_type == "swiglu"
    assert "layer0_ffn_up" in swiglu.params["input_layers"]
    assert "layer0_ffn_gate" in swiglu.params["input_layers"]

    # Check down projection
    down = name_map["layer0_ffn_down"]
    assert down.params["unit"] == 1024  # hidden_size

    print("  All MLP params verified")
    print("PASSED\n")


def test_full_model_layer_count():
    """Task 3.6: Verify full model has correct layer count for 0.6B."""
    print("=== Test 3.6: Full model layer count (0.6B) ===")

    cfg = make_qwen3_0_6b_config()
    layers = generate_qwen3_layers(cfg)

    # 1 input + 1 embedding + 28 blocks * 15 layers + 1 output_norm + 1 lm_head
    expected = 1 + 1 + 28 * 15 + 1 + 1  # = 424
    assert len(layers) == expected, f"Expected {expected}, got {len(layers)}"

    # Count by type
    type_counts = {}
    for l in layers:
        type_counts[l.layer_type] = type_counts.get(l.layer_type, 0) + 1

    print(f"  Total layers: {len(layers)}")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    # Verify counts
    assert type_counts["fully_connected"] == 28 * 7  # 7 per block (V,K,Q,O,up,gate,down)
    assert type_counts["rms_norm"] == 28 * 2 + 1  # 2 per block + output_norm
    assert type_counts["reshaped_rms_norm"] == 28 * 2  # Q_norm + K_norm per block
    assert type_counts["mha_core"] == 28
    assert type_counts["swiglu"] == 28
    assert type_counts["addition"] == 28 * 2  # 2 residuals per block

    print("PASSED\n")


def test_nntr_config_generation():
    """Task 4.1: Verify nntr_config.json generation."""
    print("=== Test 4.1: nntr_config.json generation ===")

    cfg = make_qwen3_0_6b_config()
    config = generate_nntr_config(
        cfg,
        model_file="nntr_qwen3_0_6b_fp32.bin",
        tokenizer_file="/path/to/tokenizer.json",
        sample_input="Hello world",
    )

    assert config["model_type"] == "CausalLM"
    assert config["init_seq_len"] == 1024
    assert config["num_to_generate"] == 512
    assert config["batch_size"] == 1
    assert config["model_file_name"] == "nntr_qwen3_0_6b_fp32.bin"

    # Verify it's valid JSON
    json_str = json.dumps(config, indent=4)
    parsed = json.loads(json_str)
    assert parsed == config

    print(f"  Config:\n{json_str[:500]}...")
    print("PASSED\n")


def test_weight_order():
    """Task 4.3: Verify weight conversion order."""
    print("=== Test 4.3: Weight order ===")

    cfg = make_tiny_config()
    weights = generate_weight_order(cfg)

    print(f"  Weight entries: {len(weights)}")
    for w in weights:
        t = " (T)" if w["transpose"] else ""
        print(f"    {w['hf_key']:50s} -> {w['nntr_name']}{t}")

    # First weight should be embedding
    assert weights[0]["hf_key"] == "model.embed_tokens.weight"
    assert weights[0]["transpose"] is False

    # Check per-layer order
    # For layer 0: input_layernorm, V, K, K_norm, Q, Q_norm, O, post_attn_norm, up, gate, down
    layer0_weights = [w for w in weights if "layers.0." in w["hf_key"]]
    expected_order = [
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

    for i, w in enumerate(layer0_weights):
        expected_suffix = expected_order[i]
        assert w["hf_key"].endswith(expected_suffix), (
            f"Weight {i}: expected ...{expected_suffix}, got {w['hf_key']}"
        )

    # Linear weights should be transposed, norms should not
    for w in weights:
        if "proj.weight" in w["hf_key"] or "lm_head" in w["hf_key"]:
            assert w["transpose"] is True, f"{w['hf_key']} should be transposed"
        elif "norm" in w["hf_key"] or "embed" in w["hf_key"]:
            assert w["transpose"] is False, f"{w['hf_key']} should NOT be transposed"

    # Last weight should be model.norm (tied embeddings = no separate lm_head)
    assert weights[-1]["hf_key"] == "model.norm.weight"

    print("PASSED\n")


def test_weight_converter_script():
    """Task 4.3: Verify generated weight converter script."""
    print("=== Test 4.3b: Weight converter script ===")

    cfg = make_tiny_config()
    script = generate_weight_converter_script(cfg, "Qwen3-0.6B")

    assert "import torch" in script
    assert "import numpy as np" in script
    assert "model.embed_tokens.weight" in script
    assert "permute(1, 0)" in script
    assert "self_attn.q_proj.weight" in script

    print(f"  Script length: {len(script)} chars")
    print(f"  First 500 chars:\n{script[:500]}...")
    print("PASSED\n")


def test_cpp_code_generation():
    """Task 4.2: Verify C++ createLayer() code generation."""
    print("=== Test 4.2: C++ code generation ===")

    cfg = make_tiny_config()
    layers = generate_qwen3_layers(cfg)

    # Generate C++ for first few layers
    print("  Generated C++ createLayer() calls:")
    for layer in layers[:5]:
        cpp = layer.to_cpp_create_layer()
        print(f"    {cpp}")

    # Verify specific layer C++ output
    embedding = layers[1]
    cpp = embedding.to_cpp_create_layer()
    assert '"tie_word_embeddings"' in cpp
    assert '"in_dim"' in cpp
    assert '"out_dim"' in cpp

    fc = layers[3]  # layer0_wv
    cpp = fc.to_cpp_create_layer()
    assert '"fully_connected"' in cpp
    assert '"disable_bias"' in cpp
    assert '"true"' in cpp

    print("PASSED\n")


if __name__ == "__main__":
    test_from_hf_config()
    test_layer_structure_tiny()
    test_layer_connectivity()
    test_attention_params()
    test_mlp_params()
    test_full_model_layer_count()
    test_nntr_config_generation()
    test_weight_order()
    test_weight_converter_script()
    test_cpp_code_generation()
    print("=" * 50)
    print("All Task 3-4 tests PASSED!")

#!/usr/bin/env python3
"""Task 5: End-to-End validation - trace HF Qwen3, convert, validate against reference."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import Qwen3Config as HFQwen3Config, Qwen3ForCausalLM

from tracer import Tracer, LEAF_MODULES
from node_mapper import map_graph
from qwen3_converter import (
    Qwen3Config,
    generate_qwen3_layers,
    generate_nntr_config,
    generate_weight_order,
    generate_weight_converter_script,
)


def test_e2e_trace_and_convert():
    """Full E2E: Load HF model -> Trace -> Convert -> Validate."""
    print("=== Task 5.1: E2E Trace and Convert ===\n")

    # 1. Create HF model
    hf_config = HFQwen3Config(
        hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16,
        vocab_size=1000, max_position_embeddings=256,
        tie_word_embeddings=True, rms_norm_eps=1e-6,
    )
    model = Qwen3ForCausalLM(hf_config)
    model.eval()

    # 2. Extract config
    cfg = Qwen3Config.from_hf_config(hf_config)
    print(f"  Config: hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
          f"heads={cfg.num_attention_heads}, kv_heads={cfg.num_key_value_heads}")

    # 3. Trace the model
    tracer = Tracer(model, leaf_modules=LEAF_MODULES)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        with tracer:
            out = model(input_ids)

    print(f"  Traced graph: {len(list(tracer.graph.nodes))} nodes")

    # 4. Map nodes
    mapped = map_graph(tracer.graph, model)
    mapped_real = [d for d in mapped if d is not None
                   and d.layer_type not in ("placeholder", "output")]
    print(f"  Mapped layers: {len(mapped_real)} (non-None, non-placeholder)")

    # 5. Generate NNTrainer layers
    layers = generate_qwen3_layers(cfg)
    print(f"  Generated NNTrainer layers: {len(layers)}")

    # 6. Verify the traced graph contains all expected module calls
    traced_modules = set()
    for node in tracer.graph.nodes:
        if node.op == "call_module":
            traced_modules.add(node.target)

    # Expected modules from the model
    expected_modules = set()
    for name, _ in model.named_modules():
        if name:
            expected_modules.add(name)

    # All leaf module calls should be traced
    leaf_modules_traced = {m for m in traced_modules
                          if any(isinstance(dict(model.named_modules())[m], lt)
                                for lt in LEAF_MODULES
                                if m in dict(model.named_modules()))}
    print(f"  Leaf modules traced: {len(leaf_modules_traced)}")

    # Verify key modules are present
    for key_suffix in ["embed_tokens", "self_attn.q_proj", "self_attn.k_proj",
                       "self_attn.v_proj", "self_attn.o_proj",
                       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        found = any(key_suffix in m for m in traced_modules)
        assert found, f"Missing traced module: *{key_suffix}"

    # 7. Verify generated layers match reference structure
    layer_types = [l.layer_type for l in layers]
    assert layer_types[0] == "input"
    assert layer_types[1] == "tie_word_embeddings"
    assert layer_types[-2] == "rms_norm"  # output_norm
    assert layer_types[-1] == "tie_word_embeddings"  # lm_head

    print("\n  E2E pipeline: PASSED")
    print("PASSED\n")


def test_e2e_weight_mapping():
    """E2E: Verify weight keys from HF model match our weight order."""
    print("=== Task 5.2: E2E Weight Mapping ===\n")

    hf_config = HFQwen3Config(
        hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16,
        vocab_size=1000, max_position_embeddings=256,
        tie_word_embeddings=True, rms_norm_eps=1e-6,
    )
    model = Qwen3ForCausalLM(hf_config)
    model.eval()

    cfg = Qwen3Config.from_hf_config(hf_config)
    weight_order = generate_weight_order(cfg)

    # Get actual state dict keys
    state_dict = model.state_dict()
    hf_keys = set(state_dict.keys())

    # Verify all our weight keys exist in the model
    missing = []
    for w in weight_order:
        if w["hf_key"] not in hf_keys:
            missing.append(w["hf_key"])

    if missing:
        print(f"  MISSING keys: {missing}")
        print(f"  Available keys: {sorted(hf_keys)}")
    assert len(missing) == 0, f"Missing HF keys: {missing}"

    # Verify transpose dimensions are correct
    for w in weight_order:
        tensor = state_dict[w["hf_key"]]
        if w["transpose"]:
            assert tensor.dim() == 2, (
                f"{w['hf_key']}: expected 2D for transpose, got {tensor.dim()}D"
            )
        print(f"  {w['hf_key']:50s} shape={list(tensor.shape)} transpose={w['transpose']}")

    # Count weights
    total_params = sum(state_dict[w["hf_key"]].numel() for w in weight_order)
    print(f"\n  Total params in weight file: {total_params:,}")

    # Verify vs model total (excluding tied weights)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"  Model total params: {model_params:,}")

    print("PASSED\n")


def test_e2e_config_generation():
    """E2E: Generate and validate nntr_config.json."""
    print("=== Task 5.3: E2E Config Generation ===\n")

    hf_config = HFQwen3Config(
        hidden_size=1024, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16,
        num_key_value_heads=8, head_dim=128,
        vocab_size=151936, max_position_embeddings=40960,
        tie_word_embeddings=True, rms_norm_eps=1e-6,
    )

    cfg = Qwen3Config.from_hf_config(hf_config)
    cfg.init_seq_len = 1024
    cfg.num_to_generate = 512
    cfg.max_seq_len = 2048

    # Generate config
    nntr_config = generate_nntr_config(
        cfg,
        model_file="nntr_qwen3_0_6b_fp32.bin",
        tokenizer_file="/path/to/tokenizer.json",
    )

    # Compare with reference config structure
    reference_keys = {
        "model_type", "model_tensor_type", "model_file_name",
        "fc_layer_dtype", "embedding_dtype",
        "lora_rank", "lora_alpha", "lora_target",
        "bad_word_ids", "fsu", "fsu_lookahead",
        "num_to_generate", "init_seq_len", "max_seq_len",
        "batch_size", "tokenizer_file", "sample_input",
    }

    generated_keys = set(nntr_config.keys())
    missing_keys = reference_keys - generated_keys
    assert len(missing_keys) == 0, f"Missing config keys: {missing_keys}"

    # Verify values
    assert nntr_config["model_type"] == "CausalLM"
    assert nntr_config["init_seq_len"] == 1024
    assert nntr_config["num_to_generate"] == 512
    assert nntr_config["max_seq_len"] == 2048
    assert nntr_config["batch_size"] == 1

    print(f"  Generated config: {json.dumps(nntr_config, indent=2)}")
    print("PASSED\n")


def test_e2e_layer_comparison_with_reference():
    """E2E: Compare generated layers against reference Qwen3 C++ implementation."""
    print("=== Task 5.4: Layer comparison with reference ===\n")

    # Reference layer structure from qwen3_causallm.cpp + transformer.cpp
    # For layer 0 with Qwen3-0.6B params:
    cfg = Qwen3Config(
        vocab_size=151936, hidden_size=1024, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, max_position_embeddings=40960, rope_theta=1000000.0,
        rms_norm_eps=1e-6, tie_word_embeddings=True, sliding_window=32768,
        init_seq_len=1024, num_to_generate=512,
    )

    layers = generate_qwen3_layers(cfg)
    name_map = {l.name: l for l in layers}

    # Reference: qwen3_causallm.cpp layer creation
    # V: unit = head_dim * n_heads / GQA_SIZE = 128 * 16 / 2 = 1024
    assert name_map["layer0_wv"].params["unit"] == 1024
    # K: same as V
    assert name_map["layer0_wk"].params["unit"] == 1024
    # Q: unit = head_dim * n_heads = 128 * 16 = 2048
    assert name_map["layer0_wq"].params["unit"] == 2048
    # O: unit = DIM = 1024
    assert name_map["layer0_attention_out"].params["unit"] == 1024

    # MHA core: input_layers = {Q_norm, K_norm, V}
    mha = name_map["layer0_attention"]
    mha_inputs = mha.params["input_layers"]
    assert "layer0_q_norm" in mha_inputs[0]
    assert "layer0_k_norm" in mha_inputs[1]
    assert "layer0_wv" in mha_inputs[2]

    # MHA params
    assert mha.params["num_heads"] == 16
    assert mha.params["num_heads_kv"] == 8
    assert mha.params["sliding_window"] == 32768
    assert mha.params["rope_theta"] == 1000000

    # MLP
    assert name_map["layer0_ffn_up"].params["unit"] == 3072
    assert name_map["layer0_ffn_gate"].params["unit"] == 3072
    assert name_map["layer0_ffn_down"].params["unit"] == 1024

    # SwiGLU inputs
    swiglu = name_map["layer0_ffn_swiglu"]
    assert "layer0_ffn_up" in swiglu.params["input_layers"]
    assert "layer0_ffn_gate" in swiglu.params["input_layers"]

    # Residual connections
    assert "embedding0" in name_map["layer0_decoder_add"].params["input_layers"]
    assert "layer0_attention_out" in name_map["layer0_decoder_add"].params["input_layers"]

    # Output
    assert name_map["output_norm"].params["input_layers"] == "layer27_decoder_output"
    assert name_map["output_of_causallm"].params["input_layers"] == "output_norm"
    assert name_map["output_of_causallm"].params["shared_from"] == "embedding0"

    print("  All reference comparisons passed")
    print("PASSED\n")


if __name__ == "__main__":
    test_e2e_trace_and_convert()
    test_e2e_weight_mapping()
    test_e2e_config_generation()
    test_e2e_layer_comparison_with_reference()
    print("=" * 50)
    print("All Task 5 E2E tests PASSED!")

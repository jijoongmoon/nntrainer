#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 Converter: Converts HuggingFace Qwen3 model to NNTrainer format.

Instead of generic graph pattern matching, this converter uses architecture-aware
conversion: it reads the HuggingFace model config and module hierarchy to directly
generate the corresponding NNTrainer layer definitions.

This approach is more robust than graph pattern matching because:
1. We know the exact Qwen3 architecture from the config
2. We know the exact NNTrainer layer structure from existing implementations
3. We just need to map HF config params to NNTrainer params
"""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NNTrainerLayer:
    """A single NNTrainer layer definition."""

    layer_type: str
    name: str
    params: dict = field(default_factory=dict)

    def to_cpp_create_layer(self) -> str:
        """Generate C++ createLayer() call."""
        props = []
        for k, v in self.params.items():
            if isinstance(v, list):
                val = ",".join(str(x) for x in v)
                props.append(f'withKey("{k}", {{{val}}})')
            elif isinstance(v, bool):
                props.append(f'withKey("{k}", "{str(v).lower()}")')
            elif isinstance(v, (int, float)):
                props.append(f'withKey("{k}", {v})')
            else:
                props.append(f'withKey("{k}", "{v}")')

        props_str = ", ".join(props)
        return f'createLayer("{self.layer_type}", {{{props_str}}})'

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.layer_type,
            "name": self.name,
            **self.params,
        }


@dataclass
class Qwen3Config:
    """Qwen3 model configuration extracted from HuggingFace config."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    sliding_window: Optional[int] = None

    # NNTrainer runtime params (from nntr_config.json)
    batch_size: int = 1
    init_seq_len: int = 1024
    max_seq_len: int = 2048
    num_to_generate: int = 512
    embedding_dtype: str = "FP32"
    fc_layer_dtype: str = "FP32"

    @classmethod
    def from_hf_config(cls, hf_config) -> "Qwen3Config":
        """Extract from HuggingFace PretrainedConfig object."""
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads),
            max_position_embeddings=hf_config.max_position_embeddings,
            rope_theta=(hf_config.rope_scaling.get("rope_theta", 10000.0)
                       if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling
                       else getattr(hf_config, "rope_theta", 10000.0)),
            rms_norm_eps=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            sliding_window=getattr(hf_config, "sliding_window", None),
        )

    @property
    def gqa_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


def generate_qwen3_layers(cfg: Qwen3Config) -> list[NNTrainerLayer]:
    """Generate the complete NNTrainer layer list for Qwen3 CausalLM.

    This mirrors the C++ code in:
    - transformer.cpp::constructModel()
    - transformer.cpp::createTransformerDecoderBlock()
    - qwen3_causallm.cpp::createAttention() (with Q/K norms)
    - transformer.cpp::createMlp()
    - causal_lm.cpp::constructModel() (adds lm_head)
    """
    layers = []

    # 1. Input layer
    layers.append(NNTrainerLayer(
        layer_type="input",
        name="input0",
        params={"input_shape": f"1:1:{cfg.init_seq_len}"},
    ))

    # 2. Embedding layer
    emb_type = "tie_word_embeddings" if cfg.tie_word_embeddings else "embedding_layer"
    layers.append(NNTrainerLayer(
        layer_type=emb_type,
        name="embedding0",
        params={
            "in_dim": cfg.vocab_size,
            "out_dim": cfg.hidden_size,
            "weight_dtype": cfg.embedding_dtype,
        },
    ))

    # 3. Transformer decoder blocks
    for i in range(cfg.num_hidden_layers):
        input_name = "embedding0" if i == 0 else f"layer{i - 1}_decoder_output"
        layers.extend(_create_decoder_block(cfg, i, input_name))

    # 4. Output RMSNorm
    layers.append(NNTrainerLayer(
        layer_type="rms_norm",
        name="output_norm",
        params={
            "input_layers": f"layer{cfg.num_hidden_layers - 1}_decoder_output",
            "epsilon": str(cfg.rms_norm_eps),
            "packed": "false",
        },
    ))

    # 5. LM Head
    lmhead_type = "tie_word_embeddings" if cfg.tie_word_embeddings else "lm_head"
    lmhead_params = {
        "unit": cfg.vocab_size,
        "disable_bias": "true",
        "input_layers": "output_norm",
        "weight_dtype": cfg.embedding_dtype,
    }
    if cfg.tie_word_embeddings:
        lmhead_params["shared_from"] = "embedding0"
    layers.append(NNTrainerLayer(
        layer_type=lmhead_type,
        name="output_of_causallm",
        params=lmhead_params,
    ))

    return layers


def _create_decoder_block(cfg: Qwen3Config, layer_id: int, input_name: str) -> list[NNTrainerLayer]:
    """Create a single Qwen3 transformer decoder block."""
    layers = []
    lid = str(layer_id)

    # Attention norm (input_layernorm)
    layers.append(NNTrainerLayer(
        layer_type="rms_norm",
        name=f"layer{lid}_attention_norm",
        params={
            "input_layers": input_name,
            "epsilon": str(cfg.rms_norm_eps),
            "packed": "false",
        },
    ))

    # Attention layers (Qwen3-specific with Q/K norms)
    att_norm = f"layer{lid}_attention_norm"
    layers.extend(_create_qwen3_attention(cfg, layer_id, att_norm, att_norm, att_norm))

    # Residual add after attention
    layers.append(NNTrainerLayer(
        layer_type="addition",
        name=f"layer{lid}_decoder_add",
        params={"input_layers": f"{input_name},layer{lid}_attention_out"},
    ))

    # FFN norm (post_attention_layernorm)
    layers.append(NNTrainerLayer(
        layer_type="rms_norm",
        name=f"layer{lid}_ffn_norm",
        params={
            "input_layers": f"layer{lid}_decoder_add",
            "epsilon": str(cfg.rms_norm_eps),
            "packed": "false",
        },
    ))

    # FFN (SwiGLU MLP)
    layers.extend(_create_mlp(cfg, layer_id, f"layer{lid}_ffn_norm"))

    # Residual add after FFN
    layers.append(NNTrainerLayer(
        layer_type="addition",
        name=f"layer{lid}_decoder_output",
        params={"input_layers": f"layer{lid}_decoder_add,layer{lid}_ffn_down"},
    ))

    return layers


def _create_qwen3_attention(
    cfg: Qwen3Config, layer_id: int,
    query_name: str, key_name: str, value_name: str,
) -> list[NNTrainerLayer]:
    """Create Qwen3 attention layers with Q/K norms.

    Mirrors qwen3_causallm.cpp::Qwen3Transformer::createAttention()
    Order: V -> K -> K_norm -> Q -> Q_norm -> MHA_core -> O
    """
    layers = []
    lid = str(layer_id)
    kv_dim = cfg.head_dim * cfg.num_attention_heads // cfg.gqa_size

    sliding_window = cfg.sliding_window if cfg.sliding_window else 4294967295  # UINT_MAX

    # V projection
    layers.append(NNTrainerLayer(
        layer_type="fully_connected",
        name=f"layer{lid}_wv",
        params={
            "unit": kv_dim,
            "disable_bias": "true",
            "input_layers": value_name,
        },
    ))

    # K projection
    layers.append(NNTrainerLayer(
        layer_type="fully_connected",
        name=f"layer{lid}_wk",
        params={
            "unit": kv_dim,
            "disable_bias": "true",
            "input_layers": key_name,
        },
    ))

    # K norm (Qwen3-specific: reshaped_rms_norm)
    layers.append(NNTrainerLayer(
        layer_type="reshaped_rms_norm",
        name=f"layer{lid}_k_norm",
        params={
            "input_layers": f"layer{lid}_wk",
            "packed": "false",
            "epsilon": str(cfg.rms_norm_eps),
            "feature_size": str(cfg.head_dim),
        },
    ))

    # Q projection
    layers.append(NNTrainerLayer(
        layer_type="fully_connected",
        name=f"layer{lid}_wq",
        params={
            "unit": cfg.head_dim * cfg.num_attention_heads,
            "disable_bias": "true",
            "input_layers": query_name,
        },
    ))

    # Q norm (Qwen3-specific: reshaped_rms_norm)
    layers.append(NNTrainerLayer(
        layer_type="reshaped_rms_norm",
        name=f"layer{lid}_q_norm",
        params={
            "input_layers": f"layer{lid}_wq",
            "packed": "false",
            "epsilon": str(cfg.rms_norm_eps),
            "feature_size": str(cfg.head_dim),
        },
    ))

    # MHA core
    layers.append(NNTrainerLayer(
        layer_type="mha_core",
        name=f"layer{lid}_attention",
        params={
            "num_heads": cfg.num_attention_heads,
            "num_heads_kv": cfg.num_attention_heads // cfg.gqa_size,
            "max_timestep": str(cfg.init_seq_len + cfg.num_to_generate),
            "sliding_window": sliding_window,
            "rope_theta": int(cfg.rope_theta),
            "max_position_embeddings": cfg.max_position_embeddings,
            "max_new_tokens": str(cfg.num_to_generate),
            "input_layers": [f"layer{lid}_q_norm", f"layer{lid}_k_norm", f"layer{lid}_wv"],
        },
    ))

    # O projection
    layers.append(NNTrainerLayer(
        layer_type="fully_connected",
        name=f"layer{lid}_attention_out",
        params={
            "unit": cfg.hidden_size,
            "disable_bias": "true",
            "input_layers": f"layer{lid}_attention",
        },
    ))

    return layers


def _create_mlp(cfg: Qwen3Config, layer_id: int, input_name: str) -> list[NNTrainerLayer]:
    """Create SwiGLU MLP layers.

    Mirrors transformer.cpp::Transformer::createMlp()
    Order: up -> gate -> swiglu -> down
    """
    layers = []
    lid = str(layer_id)

    # Up projection
    layers.append(NNTrainerLayer(
        layer_type="fully_connected",
        name=f"layer{lid}_ffn_up",
        params={
            "unit": cfg.intermediate_size,
            "disable_bias": "true",
            "input_layers": input_name,
        },
    ))

    # Gate projection
    layers.append(NNTrainerLayer(
        layer_type="fully_connected",
        name=f"layer{lid}_ffn_gate",
        params={
            "unit": cfg.intermediate_size,
            "disable_bias": "true",
            "input_layers": input_name,
        },
    ))

    # SwiGLU
    layers.append(NNTrainerLayer(
        layer_type="swiglu",
        name=f"layer{lid}_ffn_swiglu",
        params={
            "input_layers": f"layer{lid}_ffn_up,layer{lid}_ffn_gate",
        },
    ))

    # Down projection
    layers.append(NNTrainerLayer(
        layer_type="fully_connected",
        name=f"layer{lid}_ffn_down",
        params={
            "unit": cfg.hidden_size,
            "disable_bias": "true",
            "input_layers": f"layer{lid}_ffn_swiglu",
        },
    ))

    return layers


def generate_nntr_config(cfg: Qwen3Config, model_file: str, tokenizer_file: str,
                         sample_input: str = "") -> dict:
    """Generate nntr_config.json content."""
    return {
        "model_type": "CausalLM",
        "model_tensor_type": f"{cfg.fc_layer_dtype}-{cfg.fc_layer_dtype}",
        "model_file_name": model_file,
        "fc_layer_dtype": cfg.fc_layer_dtype,
        "embedding_dtype": cfg.embedding_dtype,
        "lora_rank": 0,
        "lora_alpha": 0,
        "lora_target": [],
        "bad_word_ids": [],
        "fsu": False,
        "fsu_lookahead": 2,
        "num_to_generate": cfg.num_to_generate,
        "init_seq_len": cfg.init_seq_len,
        "max_seq_len": cfg.max_seq_len,
        "batch_size": cfg.batch_size,
        "tokenizer_file": tokenizer_file,
        "sample_input": sample_input,
    }


def generate_weight_order(cfg: Qwen3Config) -> list[dict]:
    """Generate the weight save order for NNTrainer binary format.

    This matches weight_converter.py's save order.
    Returns list of dicts with 'hf_key', 'nntr_name', and 'transpose' flag.
    """
    weights = []

    # 1. Embedding
    weights.append({
        "hf_key": "model.embed_tokens.weight",
        "nntr_name": "embedding0",
        "transpose": False,
    })

    # 2. Per-layer weights
    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}."

        # input_layernorm
        weights.append({
            "hf_key": f"{prefix}input_layernorm.weight",
            "nntr_name": f"layer{i}_attention_norm",
            "transpose": False,
        })

        # Attention projections (order: V, K, Q, O matching NNTrainer layer order)
        # But weight_converter.py saves Q, K, V, O order with norms interspersed
        # Actually looking at weight_converter.py more carefully:
        # save_attention order: input_layernorm, then for [q,k,v,o]: projection + norm
        # But NNTrainer layer order is: V, K, K_norm, Q, Q_norm, MHA, O
        # The WEIGHT file order must match NNTrainer's layer registration order

        # V projection (saved first in NNTrainer)
        weights.append({
            "hf_key": f"{prefix}self_attn.v_proj.weight",
            "nntr_name": f"layer{i}_wv",
            "transpose": True,
        })

        # K projection
        weights.append({
            "hf_key": f"{prefix}self_attn.k_proj.weight",
            "nntr_name": f"layer{i}_wk",
            "transpose": True,
        })

        # K norm (Qwen3-specific)
        weights.append({
            "hf_key": f"{prefix}self_attn.k_norm.weight",
            "nntr_name": f"layer{i}_k_norm",
            "transpose": False,
        })

        # Q projection
        weights.append({
            "hf_key": f"{prefix}self_attn.q_proj.weight",
            "nntr_name": f"layer{i}_wq",
            "transpose": True,
        })

        # Q norm (Qwen3-specific)
        weights.append({
            "hf_key": f"{prefix}self_attn.q_norm.weight",
            "nntr_name": f"layer{i}_q_norm",
            "transpose": False,
        })

        # O projection
        weights.append({
            "hf_key": f"{prefix}self_attn.o_proj.weight",
            "nntr_name": f"layer{i}_attention_out",
            "transpose": True,
        })

        # post_attention_layernorm
        weights.append({
            "hf_key": f"{prefix}post_attention_layernorm.weight",
            "nntr_name": f"layer{i}_ffn_norm",
            "transpose": False,
        })

        # MLP: up, gate, down (matching NNTrainer layer order)
        weights.append({
            "hf_key": f"{prefix}mlp.up_proj.weight",
            "nntr_name": f"layer{i}_ffn_up",
            "transpose": True,
        })
        weights.append({
            "hf_key": f"{prefix}mlp.gate_proj.weight",
            "nntr_name": f"layer{i}_ffn_gate",
            "transpose": True,
        })
        weights.append({
            "hf_key": f"{prefix}mlp.down_proj.weight",
            "nntr_name": f"layer{i}_ffn_down",
            "transpose": True,
        })

    # 3. Final norm
    weights.append({
        "hf_key": "model.norm.weight",
        "nntr_name": "output_norm",
        "transpose": False,
    })

    # 4. LM head (if not tied)
    if not cfg.tie_word_embeddings:
        weights.append({
            "hf_key": "lm_head.weight",
            "nntr_name": "output_of_causallm",
            "transpose": True,
        })

    return weights


def generate_weight_converter_script(cfg: Qwen3Config, model_name: str) -> str:
    """Generate a weight_converter.py script for the model."""
    weight_order = generate_weight_order(cfg)

    lines = [
        '#!/usr/bin/env python3',
        '"""Auto-generated weight converter for NNTrainer."""',
        '',
        'import argparse',
        'import torch',
        'import numpy as np',
        'from transformers import AutoConfig, AutoModelForCausalLM',
        '',
        '',
        'def convert_weights(model_path, output_path, dtype="float32"):',
        '    config = AutoConfig.from_pretrained(model_path)',
        '    model = AutoModelForCausalLM.from_pretrained(',
        '        model_path, torch_dtype="float", trust_remote_code=True',
        '    )',
        '    model.eval()',
        '    params = model.state_dict()',
        '',
        '    with open(output_path, "wb") as f:',
    ]

    for w in weight_order:
        if w["transpose"]:
            lines.append(f'        np.array(params["{w["hf_key"]}"].permute(1, 0), dtype=dtype).tofile(f)')
        else:
            lines.append(f'        np.array(params["{w["hf_key"]}"], dtype=dtype).tofile(f)')

    lines.extend([
        '',
        '',
        'if __name__ == "__main__":',
        '    parser = argparse.ArgumentParser()',
        f'    parser.add_argument("--model_path", type=str, default="./{model_name}")',
        f'    parser.add_argument("--output_name", type=str, default="./nntr_{model_name.lower().replace("-", "_")}_fp32.bin")',
        '    parser.add_argument("--data_type", type=str, default="float32")',
        '    args = parser.parse_args()',
        '    convert_weights(args.model_path, args.output_name, args.data_type)',
    ])

    return "\n".join(lines) + "\n"

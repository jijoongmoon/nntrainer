#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generic Model Inspector: Converts any HuggingFace CausalLM model to NNTrainer format.

Instead of hardcoding per-model converters, this module inspects the HF model's
module hierarchy and config to automatically generate the correct NNTrainer layers.

Supported patterns (auto-detected):
- Attention: q/k/v/o projections, optional q_norm/k_norm
- FFN: gate+up+down with SwiGLU or GELU+multiply
- Decoder block: with or without post-attention/post-ffn norms
- Embedding: regular or scaled, with optional tie_word_embeddings
"""

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class NNTrainerLayer:
    """A single NNTrainer layer definition."""

    layer_type: str
    name: str
    params: dict = field(default_factory=dict)

    def to_cpp_create_layer(self) -> str:
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
        return f'createLayer("{self.layer_type}", {{{", ".join(props)}}})'

    def to_dict(self) -> dict:
        return {"type": self.layer_type, "name": self.name, **self.params}


# ---------- Module classification ----------

def _classify_module(name: str, module: nn.Module) -> str:
    """Classify a HF module by inspecting its type name and attributes."""
    type_name = type(module).__name__

    if isinstance(module, nn.Linear):
        return "linear"
    if isinstance(module, nn.Embedding) or "Embedding" in type_name:
        return "embedding"
    if "RMSNorm" in type_name or "RMSNorm" in type(module).__mro__[0].__name__:
        return "rmsnorm"
    if "LayerNorm" in type_name:
        return "layernorm"

    # Activation modules
    acti_map = {
        "SiLU": "swish", "SiLUActivation": "swish",
        "GELUTanh": "tanh_gelu", "NewGELUActivation": "tanh_gelu",
        "GELU": "gelu", "GELUActivation": "gelu",
        "ReLU": "relu", "Sigmoid": "sigmoid", "Tanh": "tanh",
    }
    for cls_name, acti in acti_map.items():
        if type_name == cls_name:
            return f"activation:{acti}"

    # Containers/structural
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        return "container"
    if "RotaryEmbedding" in type_name:
        return "rotary_emb"
    if "Attention" in type_name:
        return "attention_block"
    if "MLP" in type_name:
        return "mlp_block"
    if "DecoderLayer" in type_name:
        return "decoder_layer"
    if "Model" in type_name or "Transformer" in type_name:
        return "model_wrapper"

    return f"unknown:{type_name}"


# ---------- Structure detection ----------

@dataclass
class DecoderLayerInfo:
    """Detected structure of one decoder layer."""
    prefix: str  # e.g. "model.layers.0"

    # Attention components
    q_proj: str = ""
    k_proj: str = ""
    v_proj: str = ""
    o_proj: str = ""
    q_norm: str = ""  # empty if not present
    k_norm: str = ""  # empty if not present

    # Norms
    input_layernorm: str = ""
    post_attention_layernorm: str = ""
    pre_feedforward_layernorm: str = ""  # Gemma3 only
    post_feedforward_layernorm: str = ""  # Gemma3 only

    # MLP components
    gate_proj: str = ""
    up_proj: str = ""
    down_proj: str = ""
    mlp_activation: str = "swish"  # default SwiGLU

    # Detected flags
    has_qk_norm: bool = False
    has_pre_ffn_norm: bool = False
    has_post_ffn_norm: bool = False
    has_post_attn_norm: bool = False  # Gemma3: norm after attn, before residual add
    has_bias: bool = False


@dataclass
class ModelStructure:
    """Detected structure of the full model."""
    # Embedding
    embed_module: str = ""
    embed_dim: int = 0
    vocab_size: int = 0
    embedding_scale: float = 1.0

    # Decoder layers
    layers: list = field(default_factory=list)  # list[DecoderLayerInfo]
    num_layers: int = 0

    # Output
    final_norm: str = ""
    lm_head: str = ""

    # Global config
    hidden_size: int = 0
    intermediate_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048
    tie_word_embeddings: bool = False
    sliding_window: Optional[int] = None
    layer_types: list = field(default_factory=list)
    attn_logit_softcapping: float = 0.0

    # NNTrainer runtime params
    init_seq_len: int = 1024
    num_to_generate: int = 512
    max_seq_len: int = 2048
    batch_size: int = 1
    fc_layer_dtype: str = "FP32"
    embedding_dtype: str = "FP32"


def inspect_model(model: nn.Module) -> ModelStructure:
    """Inspect a HuggingFace CausalLM model and detect its structure.

    This works for any model that follows the standard HF pattern:
    model.embed_tokens → model.layers[N] → model.norm → lm_head
    """
    structure = ModelStructure()
    modules = dict(model.named_modules())

    # Extract config
    config = getattr(model, "config", None)
    if config is not None:
        # Try text_config for multimodal models
        text_config = getattr(config, "text_config", None) or config
        structure.hidden_size = getattr(text_config, "hidden_size", 0)
        structure.intermediate_size = getattr(text_config, "intermediate_size", 0)
        structure.num_attention_heads = getattr(text_config, "num_attention_heads", 0)
        structure.num_key_value_heads = getattr(text_config, "num_key_value_heads",
                                                 structure.num_attention_heads)
        structure.head_dim = getattr(text_config, "head_dim",
                                     structure.hidden_size // max(structure.num_attention_heads, 1))
        structure.vocab_size = getattr(text_config, "vocab_size", 0)
        structure.rms_norm_eps = getattr(text_config, "rms_norm_eps",
                                         getattr(text_config, "layer_norm_eps", 1e-6))
        structure.max_position_embeddings = getattr(text_config, "max_position_embeddings", 2048)
        structure.tie_word_embeddings = getattr(text_config, "tie_word_embeddings", False)

        # Rope theta
        rope_scaling = getattr(text_config, "rope_scaling", None)
        if rope_scaling and isinstance(rope_scaling, dict):
            structure.rope_theta = rope_scaling.get("rope_theta", 10000.0)
        else:
            structure.rope_theta = getattr(text_config, "rope_theta", 10000.0)

        # Sliding window
        sw = getattr(text_config, "sliding_window", None)
        if sw is not None and sw != 0:
            structure.sliding_window = sw

        # Model-specific
        structure.layer_types = getattr(text_config, "layer_types", [])
        structure.attn_logit_softcapping = getattr(text_config, "attn_logit_softcapping", 0.0) or 0.0

    # Detect embedding scale (Gemma uses sqrt(hidden_size))
    for name, mod in model.named_modules():
        type_name = type(mod).__name__
        if "ScaledWordEmbedding" in type_name or "ScaledEmbedding" in type_name:
            structure.embedding_scale = math.sqrt(structure.hidden_size)
            break

    # Find embedding, final norm, lm_head
    for name, mod in model.named_modules():
        cls = _classify_module(name, mod)
        if cls == "embedding" and "embed_tokens" in name:
            structure.embed_module = name
            if isinstance(mod, nn.Embedding):
                structure.embed_dim = mod.embedding_dim
                structure.vocab_size = structure.vocab_size or mod.num_embeddings
        elif cls == "rmsnorm" and name.endswith(".norm") and "layers" not in name:
            structure.final_norm = name
        elif cls == "linear" and "lm_head" in name:
            structure.lm_head = name

    # Find decoder layers
    decoder_layers = []
    for name, mod in model.named_modules():
        if _classify_module(name, mod) == "decoder_layer":
            decoder_layers.append(name)

    structure.num_layers = len(decoder_layers)

    # Inspect each decoder layer
    for layer_prefix in decoder_layers:
        info = DecoderLayerInfo(prefix=layer_prefix)

        for name, mod in model.named_modules():
            if not name.startswith(layer_prefix + "."):
                continue
            relative = name[len(layer_prefix) + 1:]
            cls = _classify_module(name, mod)

            # Attention projections
            if relative.endswith("q_proj") and cls == "linear":
                info.q_proj = name
                info.has_bias = mod.bias is not None
            elif relative.endswith("k_proj") and cls == "linear":
                info.k_proj = name
            elif relative.endswith("v_proj") and cls == "linear":
                info.v_proj = name
            elif relative.endswith("o_proj") and cls == "linear":
                info.o_proj = name

            # Q/K norms
            elif relative.endswith("q_norm") and "rmsnorm" in cls:
                info.q_norm = name
                info.has_qk_norm = True
            elif relative.endswith("k_norm") and "rmsnorm" in cls:
                info.k_norm = name

            # Layer norms
            elif relative == "input_layernorm":
                info.input_layernorm = name
            elif relative == "post_attention_layernorm":
                info.post_attention_layernorm = name
            elif relative == "pre_feedforward_layernorm":
                info.pre_feedforward_layernorm = name
                info.has_pre_ffn_norm = True
            elif relative == "post_feedforward_layernorm":
                info.post_feedforward_layernorm = name
                info.has_post_ffn_norm = True

            # MLP projections
            elif relative.endswith("gate_proj") and cls == "linear":
                info.gate_proj = name
            elif relative.endswith("up_proj") and cls == "linear":
                info.up_proj = name
            elif relative.endswith("down_proj") and cls == "linear":
                info.down_proj = name

            # MLP activation
            elif cls.startswith("activation:"):
                info.mlp_activation = cls.split(":")[1]

        # Detect post-attention norm pattern (Gemma3 style)
        if info.post_attention_layernorm and info.pre_feedforward_layernorm:
            info.has_post_attn_norm = True

        structure.layers.append(info)

    return structure


# ---------- NNTrainer layer generation ----------

def generate_layers(structure: ModelStructure) -> list[NNTrainerLayer]:
    """Generate NNTrainer layers from detected model structure."""
    layers = []
    cfg = structure

    # 1. Input
    layers.append(NNTrainerLayer(
        layer_type="input", name="input0",
        params={"input_shape": f"1:1:{cfg.init_seq_len}"},
    ))

    # 2. Embedding
    emb_type = "tie_word_embeddings" if cfg.tie_word_embeddings else "embedding_layer"
    emb_params = {
        "in_dim": cfg.vocab_size,
        "out_dim": cfg.hidden_size,
        "weight_dtype": cfg.embedding_dtype,
    }
    if cfg.embedding_scale != 1.0:
        emb_params["scale"] = str(cfg.embedding_scale)
    layers.append(NNTrainerLayer(layer_type=emb_type, name="embedding0", params=emb_params))

    # 3. Decoder blocks
    for i, layer_info in enumerate(cfg.layers):
        input_name = "embedding0" if i == 0 else f"layer{i - 1}_decoder_output"
        layers.extend(_create_decoder_block(cfg, i, layer_info, input_name))

    # 4. Final norm
    layers.append(NNTrainerLayer(
        layer_type="rms_norm", name="output_norm",
        params={
            "input_layers": f"layer{cfg.num_layers - 1}_decoder_output",
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
    layers.append(NNTrainerLayer(layer_type=lmhead_type, name="output_of_causallm",
                                 params=lmhead_params))

    return layers


def _create_decoder_block(cfg: ModelStructure, layer_id: int,
                          info: DecoderLayerInfo, input_name: str) -> list[NNTrainerLayer]:
    """Create a decoder block from detected structure."""
    layers = []
    lid = str(layer_id)

    # --- Attention norm ---
    layers.append(NNTrainerLayer(
        layer_type="rms_norm", name=f"layer{lid}_attention_norm",
        params={"input_layers": input_name, "epsilon": str(cfg.rms_norm_eps), "packed": "false"},
    ))

    # --- Attention layers ---
    att_norm = f"layer{lid}_attention_norm"
    layers.extend(_create_attention(cfg, layer_id, info, att_norm, att_norm, att_norm))

    if info.has_post_attn_norm:
        # Gemma3 pattern: attn_out → post_attention_norm → residual add
        layers.append(NNTrainerLayer(
            layer_type="rms_norm", name=f"layer{lid}_post_attention_norm",
            params={
                "input_layers": f"layer{lid}_attention_out",
                "epsilon": str(cfg.rms_norm_eps), "packed": "false",
            },
        ))
        layers.append(NNTrainerLayer(
            layer_type="addition", name=f"layer{lid}_post_attention",
            params={"input_layers": f"{input_name},layer{lid}_post_attention_norm"},
        ))
        ffn_input = f"layer{lid}_post_attention"
    else:
        # Standard pattern: attn_out → residual add
        layers.append(NNTrainerLayer(
            layer_type="addition", name=f"layer{lid}_decoder_add",
            params={"input_layers": f"{input_name},layer{lid}_attention_out"},
        ))
        ffn_input = f"layer{lid}_decoder_add"

    # --- FFN norm ---
    if info.has_pre_ffn_norm:
        ffn_norm_name = f"layer{lid}pre_ffn_norm"
    else:
        ffn_norm_name = f"layer{lid}_ffn_norm"
    layers.append(NNTrainerLayer(
        layer_type="rms_norm", name=ffn_norm_name,
        params={"input_layers": ffn_input, "epsilon": str(cfg.rms_norm_eps), "packed": "false"},
    ))

    # --- FFN layers ---
    layers.extend(_create_mlp(cfg, layer_id, info, ffn_norm_name))

    # --- Post-FFN norm + residual (Gemma3) or just residual ---
    if info.has_post_ffn_norm:
        layers.append(NNTrainerLayer(
            layer_type="rms_norm", name=f"layer{lid}post_ffn_norm",
            params={"epsilon": str(cfg.rms_norm_eps), "packed": "false"},
        ))
        layers.append(NNTrainerLayer(
            layer_type="addition", name=f"layer{lid}_decoder_output",
            params={"input_layers": f"{ffn_input},layer{lid}post_ffn_norm"},
        ))
    else:
        layers.append(NNTrainerLayer(
            layer_type="addition", name=f"layer{lid}_decoder_output",
            params={"input_layers": f"{ffn_input},layer{lid}_ffn_down"},
        ))

    return layers


def _create_attention(cfg: ModelStructure, layer_id: int, info: DecoderLayerInfo,
                      query_name: str, key_name: str, value_name: str) -> list[NNTrainerLayer]:
    """Create attention layers based on detected structure."""
    layers = []
    lid = str(layer_id)
    gqa_size = cfg.num_attention_heads // max(cfg.num_key_value_heads, 1)
    kv_dim = cfg.head_dim * cfg.num_attention_heads // gqa_size
    sliding_window_val = 4294967295  # UINT_MAX = no sliding window

    # Layer-type aware sliding window (Gemma3)
    if cfg.layer_types and layer_id < len(cfg.layer_types):
        if cfg.layer_types[layer_id] == "sliding_attention" and cfg.sliding_window:
            sliding_window_val = cfg.sliding_window
    elif cfg.sliding_window:
        sliding_window_val = cfg.sliding_window

    # Layer-type aware rope theta (Gemma3: 10000 for sliding, global for full)
    rope_theta = cfg.rope_theta
    if cfg.layer_types and layer_id < len(cfg.layer_types):
        if cfg.layer_types[layer_id] == "sliding_attention":
            rope_theta = 10000.0

    # V projection
    layers.append(NNTrainerLayer(
        layer_type="fully_connected", name=f"layer{lid}_wv",
        params={"unit": kv_dim, "disable_bias": "true", "input_layers": value_name},
    ))

    # K projection
    layers.append(NNTrainerLayer(
        layer_type="fully_connected", name=f"layer{lid}_wk",
        params={"unit": kv_dim, "disable_bias": "true", "input_layers": key_name},
    ))

    # K norm (if detected)
    if info.has_qk_norm:
        layers.append(NNTrainerLayer(
            layer_type="reshaped_rms_norm", name=f"layer{lid}_k_norm",
            params={
                "input_layers": f"layer{lid}_wk", "packed": "false",
                "epsilon": str(cfg.rms_norm_eps),
                "feature_size": str(cfg.head_dim),
            },
        ))

    # Q projection
    layers.append(NNTrainerLayer(
        layer_type="fully_connected", name=f"layer{lid}_wq",
        params={
            "unit": cfg.head_dim * cfg.num_attention_heads,
            "disable_bias": "true", "input_layers": query_name,
        },
    ))

    # Q norm (if detected)
    if info.has_qk_norm:
        layers.append(NNTrainerLayer(
            layer_type="reshaped_rms_norm", name=f"layer{lid}_q_norm",
            params={
                "input_layers": f"layer{lid}_wq", "packed": "false",
                "epsilon": str(cfg.rms_norm_eps),
                "feature_size": str(cfg.head_dim),
            },
        ))

    # MHA core
    q_input = f"layer{lid}_q_norm" if info.has_qk_norm else f"layer{lid}_wq"
    k_input = f"layer{lid}_k_norm" if info.has_qk_norm else f"layer{lid}_wk"
    mha_params = {
        "num_heads": cfg.num_attention_heads,
        "num_heads_kv": cfg.num_key_value_heads,
        "max_timestep": str(cfg.init_seq_len + cfg.num_to_generate),
        "sliding_window": sliding_window_val,
        "rope_theta": int(rope_theta),
        "max_new_tokens": str(cfg.num_to_generate),
        "input_layers": [q_input, k_input, f"layer{lid}_wv"],
    }
    if cfg.max_position_embeddings:
        mha_params["max_position_embeddings"] = cfg.max_position_embeddings
    if cfg.attn_logit_softcapping > 0:
        mha_params["attn_logit_softcapping"] = str(cfg.attn_logit_softcapping)
    layers.append(NNTrainerLayer(layer_type="mha_core", name=f"layer{lid}_attention", params=mha_params))

    # O projection
    layers.append(NNTrainerLayer(
        layer_type="fully_connected", name=f"layer{lid}_attention_out",
        params={"unit": cfg.hidden_size, "disable_bias": "true",
                "input_layers": f"layer{lid}_attention"},
    ))

    return layers


def _create_mlp(cfg: ModelStructure, layer_id: int, info: DecoderLayerInfo,
                input_name: str) -> list[NNTrainerLayer]:
    """Create MLP layers based on detected activation type."""
    layers = []
    lid = str(layer_id)

    if info.mlp_activation == "swish":
        # SwiGLU pattern: up + gate → swiglu → down
        layers.append(NNTrainerLayer(
            layer_type="fully_connected", name=f"layer{lid}_ffn_up",
            params={"unit": cfg.intermediate_size, "disable_bias": "true",
                    "input_layers": input_name},
        ))
        layers.append(NNTrainerLayer(
            layer_type="fully_connected", name=f"layer{lid}_ffn_gate",
            params={"unit": cfg.intermediate_size, "disable_bias": "true",
                    "input_layers": input_name},
        ))
        layers.append(NNTrainerLayer(
            layer_type="swiglu", name=f"layer{lid}_ffn_swiglu",
            params={"input_layers": f"layer{lid}_ffn_up,layer{lid}_ffn_gate"},
        ))
        layers.append(NNTrainerLayer(
            layer_type="fully_connected", name=f"layer{lid}_ffn_down",
            params={"unit": cfg.hidden_size, "disable_bias": "true",
                    "input_layers": f"layer{lid}_ffn_swiglu"},
        ))
    else:
        # GELU/tanh_gelu pattern: gate → activation → up → multiply → down
        layers.append(NNTrainerLayer(
            layer_type="fully_connected", name=f"layer{lid}_ffn_gate",
            params={"unit": cfg.intermediate_size, "disable_bias": "true",
                    "input_layers": input_name},
        ))
        layers.append(NNTrainerLayer(
            layer_type="activation", name=f"layer{lid}_ffn_gate_gelu",
            params={"activation": info.mlp_activation,
                    "input_layers": f"layer{lid}_ffn_gate"},
        ))
        layers.append(NNTrainerLayer(
            layer_type="fully_connected", name=f"layer{lid}_ffn_up",
            params={"unit": cfg.intermediate_size, "disable_bias": "true",
                    "input_layers": input_name},
        ))
        layers.append(NNTrainerLayer(
            layer_type="multiply", name=f"layer{lid}_ffn_geglu",
            params={"input_layers": f"layer{lid}_ffn_gate_gelu,layer{lid}_ffn_up"},
        ))
        layers.append(NNTrainerLayer(
            layer_type="fully_connected", name=f"layer{lid}_ffn_down",
            params={"unit": cfg.hidden_size, "disable_bias": "true",
                    "input_layers": f"layer{lid}_ffn_geglu"},
        ))

    return layers


# ---------- Weight order generation ----------

def generate_weight_order(structure: ModelStructure) -> list[dict]:
    """Generate weight save order matching NNTrainer's layer creation order."""
    weights = []
    cfg = structure

    # 1. Embedding
    weights.append({"hf_key": f"{cfg.embed_module}.weight", "nntr_name": "embedding0", "transpose": False})

    # 2. Per-layer weights
    for i, info in enumerate(cfg.layers):
        # Attention norm
        weights.append({"hf_key": f"{info.input_layernorm}.weight",
                        "nntr_name": f"layer{i}_attention_norm", "transpose": False})

        # Attention: V, K, (K_norm), Q, (Q_norm), O
        weights.append({"hf_key": f"{info.v_proj}.weight", "nntr_name": f"layer{i}_wv", "transpose": True})
        weights.append({"hf_key": f"{info.k_proj}.weight", "nntr_name": f"layer{i}_wk", "transpose": True})
        if info.has_qk_norm:
            weights.append({"hf_key": f"{info.k_norm}.weight", "nntr_name": f"layer{i}_k_norm", "transpose": False})
        weights.append({"hf_key": f"{info.q_proj}.weight", "nntr_name": f"layer{i}_wq", "transpose": True})
        if info.has_qk_norm:
            weights.append({"hf_key": f"{info.q_norm}.weight", "nntr_name": f"layer{i}_q_norm", "transpose": False})
        weights.append({"hf_key": f"{info.o_proj}.weight", "nntr_name": f"layer{i}_attention_out", "transpose": True})

        # Post-attention norm (Gemma3)
        if info.has_post_attn_norm:
            weights.append({"hf_key": f"{info.post_attention_layernorm}.weight",
                            "nntr_name": f"layer{i}_post_attention_norm", "transpose": False})

        # FFN norm
        if info.has_pre_ffn_norm:
            weights.append({"hf_key": f"{info.pre_feedforward_layernorm}.weight",
                            "nntr_name": f"layer{i}pre_ffn_norm", "transpose": False})
        elif info.post_attention_layernorm and not info.has_post_attn_norm:
            weights.append({"hf_key": f"{info.post_attention_layernorm}.weight",
                            "nntr_name": f"layer{i}_ffn_norm", "transpose": False})

        # MLP weights
        if info.mlp_activation == "swish":
            # SwiGLU: up, gate, down
            weights.append({"hf_key": f"{info.up_proj}.weight", "nntr_name": f"layer{i}_ffn_up", "transpose": True})
            weights.append({"hf_key": f"{info.gate_proj}.weight", "nntr_name": f"layer{i}_ffn_gate", "transpose": True})
        else:
            # GELU: gate, up, down
            weights.append({"hf_key": f"{info.gate_proj}.weight", "nntr_name": f"layer{i}_ffn_gate", "transpose": True})
            weights.append({"hf_key": f"{info.up_proj}.weight", "nntr_name": f"layer{i}_ffn_up", "transpose": True})
        weights.append({"hf_key": f"{info.down_proj}.weight", "nntr_name": f"layer{i}_ffn_down", "transpose": True})

        # Post-FFN norm (Gemma3)
        if info.has_post_ffn_norm:
            weights.append({"hf_key": f"{info.post_feedforward_layernorm}.weight",
                            "nntr_name": f"layer{i}post_ffn_norm", "transpose": False})

    # 3. Final norm
    weights.append({"hf_key": f"{cfg.final_norm}.weight", "nntr_name": "output_norm", "transpose": False})

    # 4. LM head (if not tied)
    if not cfg.tie_word_embeddings and cfg.lm_head:
        weights.append({"hf_key": f"{cfg.lm_head}.weight", "nntr_name": "output_of_causallm", "transpose": True})

    return weights


# ---------- Config generation ----------

def generate_nntr_config(structure: ModelStructure, model_file: str,
                         tokenizer_file: str, sample_input: str = "") -> dict:
    """Generate nntr_config.json."""
    cfg = structure
    return {
        "model_type": "CausalLM",
        "model_tensor_type": f"{cfg.fc_layer_dtype}-{cfg.fc_layer_dtype}",
        "model_file_name": model_file,
        "fc_layer_dtype": cfg.fc_layer_dtype,
        "embedding_dtype": cfg.embedding_dtype,
        "lora_rank": 0, "lora_alpha": 0, "lora_target": [],
        "bad_word_ids": [], "fsu": False, "fsu_lookahead": 2,
        "num_to_generate": cfg.num_to_generate,
        "init_seq_len": cfg.init_seq_len,
        "max_seq_len": cfg.max_seq_len,
        "batch_size": cfg.batch_size,
        "tokenizer_file": tokenizer_file,
        "sample_input": sample_input,
    }


def generate_weight_converter_script(structure: ModelStructure, model_name: str) -> str:
    """Generate a weight_converter.py script."""
    weight_order = generate_weight_order(structure)
    lines = [
        '#!/usr/bin/env python3',
        '"""Auto-generated weight converter for NNTrainer."""',
        '', 'import argparse', 'import torch', 'import numpy as np',
        'from transformers import AutoConfig, AutoModelForCausalLM',
        '', '',
        'def convert_weights(model_path, output_path, dtype="float32"):',
        '    config = AutoConfig.from_pretrained(model_path)',
        '    model = AutoModelForCausalLM.from_pretrained(',
        '        model_path, torch_dtype="float", trust_remote_code=True)',
        '    model.eval()',
        '    params = model.state_dict()',
        '    with open(output_path, "wb") as f:',
    ]
    for w in weight_order:
        if w["transpose"]:
            lines.append(f'        np.array(params["{w["hf_key"]}"].permute(1, 0), dtype=dtype).tofile(f)')
        else:
            lines.append(f'        np.array(params["{w["hf_key"]}"], dtype=dtype).tofile(f)')
    safe_name = model_name.lower().replace("-", "_").replace("/", "_")
    lines.extend([
        '', '', 'if __name__ == "__main__":',
        '    parser = argparse.ArgumentParser()',
        f'    parser.add_argument("--model_path", type=str, default="./{model_name}")',
        f'    parser.add_argument("--output_name", type=str, default="./nntr_{safe_name}_fp32.bin")',
        '    parser.add_argument("--data_type", type=str, default="float32")',
        '    args = parser.parse_args()',
        '    convert_weights(args.model_path, args.output_name, args.data_type)',
    ])
    return "\n".join(lines) + "\n"

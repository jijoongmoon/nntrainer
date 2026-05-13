## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
##
## @file weight_converter.py
## @brief weight conversion script for qwen3 model
## @author SeungBaek Hong <sb92.hong@samsung.com>

# pylint: skip-file

import argparse
from pathlib import Path
import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM

total_size = 0
def save_gemma3_for_nntrainer(
    params,
    config,
    dtype,
    file,
    prefix="",
    include_lm_head=True,
    dense_weight_keys=None,
):
    """Convert and save weights as nntrainer format for multi-head attention model"""  
    n_layers = config.num_hidden_layers
    hidden_size = config.hidden_size

    def save_weight(weight, is_rms=False):
        if is_rms:
            weight = weight + 1.0
        if hasattr(weight, "detach"):
            weight = weight.detach().cpu().numpy()
        np.asarray(weight, dtype=dtype).tofile(file)

    def save_projection(layer_name, proj_name):  
        """Save projection layer weights (with LoRA support)"""  
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"  
        if lora_key in params:  
            save_weight(params[f"{layer_name}{proj_name}.base_layer.weight"].permute(1, 0))  
            save_weight(params[f"{layer_name}{proj_name}.lora_A.default.weight"].permute(1, 0))  
            save_weight(params[f"{layer_name}{proj_name}.lora_B.default.weight"].permute(1, 0))  
        else:  
            save_weight(params[f"{layer_name}{proj_name}.weight"].permute(1, 0))  

    def save_attention(layer_name):  
        """Save attention layer weights"""  
        save_weight(params[f"{layer_name}input_layernorm.weight"], is_rms=True)  
          
        # Save in NNTrainer graph order:
        # attention_norm -> Q -> K -> V -> q_norm -> k_norm -> O
        save_projection(layer_name, "self_attn.q_proj")
        save_projection(layer_name, "self_attn.k_proj")
        save_projection(layer_name, "self_attn.v_proj")
        if f"{layer_name}self_attn.q_norm.weight" in params:
            save_weight(params[f"{layer_name}self_attn.q_norm.weight"], is_rms=True)
        if f"{layer_name}self_attn.k_norm.weight" in params:
            save_weight(params[f"{layer_name}self_attn.k_norm.weight"], is_rms=True)
        save_projection(layer_name, "self_attn.o_proj")

    def save_feed_forward(layer_name):  
        """Save feed forward layer weights"""  
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"], is_rms=True)
        save_weight(params[f"{layer_name}pre_feedforward_layernorm.weight"], is_rms=True)
        # Save in NNTrainer graph order:
        # post_attention_norm -> pre_ffn_norm -> gate -> up -> down -> post_ffn_norm
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")
        save_weight(params[f"{layer_name}post_feedforward_layernorm.weight"], is_rms=True)

    save_weight(params[f"{prefix}model.embed_tokens.weight"])
 
    for layer_idx in range(n_layers):  
        layer_prefix = f"{prefix}model.layers.{layer_idx}."
        save_attention(layer_prefix)  
        save_feed_forward(layer_prefix)  

    save_weight(params[f"{prefix}model.norm.weight"], is_rms=True)
    if include_lm_head:
        save_weight(params["lm_head.weight"].permute(1, 0))
    if dense_weight_keys:
        for key in dense_weight_keys:
            save_weight(params[key].permute(1, 0))


def find_sentence_transformer_dense_keys(params):
    dense_keys = []
    for key in params.keys():
        parts = key.split(".")
        if (
            len(parts) >= 3
            and parts[0].isdigit()
            and key.endswith("linear.weight")
        ):
            dense_keys.append((int(parts[0]), key))
    return [key for _, key in sorted(dense_keys)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./270m")
    parser.add_argument(
        "--output_name", type=str, default="./nntr_gemma3_270m_fp32.bin"
    )
    parser.add_argument("--data_type", type=str, default="float32")
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Load a SentenceTransformer Gemma embedding model and append dense module weights.",
    )
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = Path(args.model_path)
    output_name = args.output_name
    is_sentence_transformer = args.embedding or (
        model_path / "modules.json"
    ).exists()

    if is_sentence_transformer:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(model_path), trust_remote_code=True)
        config = model[0].model.config
        prefix = "0."
        include_lm_head = False
        dense_weight_keys = find_sentence_transformer_dense_keys(model.state_dict())
    else:
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, trust_remote_code=True
        )
        prefix = ""
        include_lm_head = True
        dense_weight_keys = None

    model.eval()

    with open(output_name, "wb") as f_model :
        save_gemma3_for_nntrainer(
            model.state_dict(),
            config,
            data_dtype,
            f_model,
            prefix=prefix,
            include_lm_head=include_lm_head,
            dense_weight_keys=dense_weight_keys,
        )

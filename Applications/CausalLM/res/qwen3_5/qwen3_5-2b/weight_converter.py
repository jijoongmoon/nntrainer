## @file weight_converter.py
## @brief weight conversion script for Qwen3.5 model
## @note  Qwen3.5 has a hybrid architecture with two layer types:
##        - linear_attention (GatedDeltaNet): layers where (i+1) % interval != 0
##        - full_attention: layers where (i+1) % interval == 0
##
##        GGUF tensor name mapping:
##          Linear attention layers:
##            attn_norm     -> input_layernorm
##            attn_qkv      -> in_proj_qkv (GatedDeltaNet)
##            attn_gate     -> in_proj_z (gate projection)
##            ssm_a         -> A_log
##            ssm_alpha     -> in_proj_b (beta projection)
##            ssm_beta      -> in_proj_a (alpha/decay projection)
##            ssm_conv1d    -> conv1d.weight
##            ssm_dt        -> dt_bias
##            ssm_norm      -> norm.weight (RMSNormGated)
##            ssm_out       -> out_proj
##
##          Full attention layers:
##            attn_norm     -> input_layernorm
##            attn_qkv      -> fused q_proj + k_proj + v_proj
##            attn_gate     -> o_proj
##            (no ssm_* tensors)

import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def is_full_attention_layer(layer_idx, interval=4):
    """Check if a layer uses full attention (every interval-th layer)"""
    return ((layer_idx + 1) % interval) == 0


def save_qwen3_5_for_nntrainer(params, config, dtype, file):
    """Convert and save Qwen3.5 weights in nntrainer format"""

    def save_weight(weight):
        np.array(weight.cpu().detach(), dtype=dtype).tofile(file)

    n_layers = config.num_hidden_layers
    interval = getattr(config, 'full_attention_interval', 4)

    # 1. Save embedding layer
    save_weight(params["model.embed_tokens.weight"])

    # 2. Process each layer
    for layer_idx in range(n_layers):
        prefix = f"model.layers.{layer_idx}."

        # Input layernorm (attention_norm)
        save_weight(params[f"{prefix}input_layernorm.weight"])

        if is_full_attention_layer(layer_idx, interval):
            # Full attention layer
            # Q projection (outputs 2x: half query, half gate)
            save_weight(params[f"{prefix}self_attn.q_proj.weight"].permute(1, 0))
            # K projection
            save_weight(params[f"{prefix}self_attn.k_proj.weight"].permute(1, 0))
            # V projection
            save_weight(params[f"{prefix}self_attn.v_proj.weight"].permute(1, 0))
            # Q norm
            save_weight(params[f"{prefix}self_attn.q_norm.weight"])
            # K norm
            save_weight(params[f"{prefix}self_attn.k_norm.weight"])
            # O projection (attn_gate in GGUF)
            save_weight(params[f"{prefix}self_attn.o_proj.weight"].permute(1, 0))
        else:
            # Linear attention layer (GatedDeltaNet)
            # in_proj_qkv
            save_weight(params[f"{prefix}linear_attn.in_proj_qkv.weight"].permute(1, 0))
            # conv1d weight [out_ch, 1, kernel] -> [kernel, out_ch]
            conv_w = params[f"{prefix}linear_attn.conv1d.weight"]
            save_weight(conv_w.squeeze(1).permute(1, 0))
            # A_log
            save_weight(params[f"{prefix}linear_attn.A_log"])
            # in_proj_b (beta/alpha projection)
            save_weight(params[f"{prefix}linear_attn.in_proj_b.weight"].permute(1, 0))
            # in_proj_a (decay projection)
            save_weight(params[f"{prefix}linear_attn.in_proj_a.weight"].permute(1, 0))
            # dt_bias
            save_weight(params[f"{prefix}linear_attn.dt_bias"])
            # norm weight (RMSNormGated)
            save_weight(params[f"{prefix}linear_attn.norm.weight"])
            # out_proj
            save_weight(params[f"{prefix}linear_attn.out_proj.weight"].permute(1, 0))
            # in_proj_z (gate projection, mapped to attn_gate in GGUF)
            save_weight(params[f"{prefix}linear_attn.in_proj_z.weight"].permute(1, 0))

        # Post-attention layernorm
        save_weight(params[f"{prefix}post_attention_layernorm.weight"])

        # MLP (SwiGLU)
        save_weight(params[f"{prefix}mlp.up_proj.weight"].permute(1, 0))
        save_weight(params[f"{prefix}mlp.gate_proj.weight"].permute(1, 0))
        save_weight(params[f"{prefix}mlp.down_proj.weight"].permute(1, 0))

    # 3. Final norm
    save_weight(params["model.norm.weight"])

    # 4. LM Head
    save_weight(params["lm_head.weight"].permute(1, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.5 model to nntrainer format")
    parser.add_argument("--model_path", type=str, default="./Qwen3.5-2B")
    parser.add_argument("--output_name", type=str,
                        default="./nntr_qwen3_5_2b_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="float", trust_remote_code=True)
    model.eval()

    print(f"Model: {model_path}")
    print(f"Layers: {config.num_hidden_layers}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Full attention interval: {getattr(config, 'full_attention_interval', 4)}")
    print(f"Output: {output_name}")

    with open(output_name, "wb") as f_model:
        save_qwen3_5_for_nntrainer(
            model.state_dict(), config, data_dtype, f_model)

    print("Done!")

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>

# @file talker_converter.py
# @brief Convert the Qwen2.5-Omni Talker (codec-token LM) into an nntrainer
#        FP32 .bin plus the config files the CausalLM application expects.
#
#        The Talker is a Qwen2-style decoder (hidden 896, 24 layers, 14 heads,
#        2 kv-heads, head_dim 64, inter 4864, q/k/v bias, SwiGLU, RMSNorm
#        eps 1e-6, rope_theta 1e6, mrope_section [16,16,0]) that emits codec
#        token ids (vocab 8448) conditioned on the Thinker's reply trajectory.
#
#        nntrainer Talker graph (host computes fused inputs_embeds, dim 2048):
#          input0[seq,2048] -> thinker_to_talker_proj (2048->896,+bias)
#            -> 24 decoder blocks (M-RoPE on q/k, mha_core theta=0)
#            -> output_norm (RMSNorm) -> codec_head (896->8448, no bias)
#        The codec embed_tokens table is NOT in the graph; it is emitted as a
#        separate raw fp32 bin (codec_embed.bin) for the host-side lookup that
#        builds inputs_embeds.
#
#        On-disk weight order == nntrainer's symbolic-graph DFS-from-output
#        load order (Model::compile), FC weights transposed (in,out):
#          thinker_to_talker_proj (w^T, b)
#          per layer: input_layernorm,
#                     q_proj (w^T, b), k_proj (w^T, b), v_proj (w^T, b),
#                     o_proj (w^T),
#                     post_attention_layernorm,
#                     gate_proj (w^T), up_proj (w^T), down_proj (w^T)   # gate first
#          output_norm
#          codec_head (w^T, no bias)
#
# @usage
#   python talker_converter.py --model_path Qwen/Qwen2.5-Omni-3B \
#       --output_dir ./qwen2.5-omni-3b-talker
#
# @author Jijoong Moon <jijoong.moon@samsung.com>

import argparse
import json
import os
import shutil

import numpy as np

from weight_converter import ShardedSafetensors, resolve_model_dir


def main():
    ap = argparse.ArgumentParser(
        description="Convert the Qwen2.5-Omni Talker to an nntrainer FP32 .bin")
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-3B",
                    help="Local checkpoint dir or HuggingFace repo id")
    ap.add_argument("--output_dir", default="./qwen2.5-omni-3b-talker")
    ap.add_argument("--output_name", default="nntr_qwen2.5_omni_talker_fp32.bin")
    ap.add_argument("--codec_embed_name", default="codec_embed.bin")
    ap.add_argument("--thinker_dir", default="",
                    help="nntrainer Thinker model dir (for end-to-end Stage C)")
    ap.add_argument("--speaker_bos", type=int, default=151872,
                    help="speaker bos token (Chelsie=151872, Ethan=151870)")
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(model_dir, "config.json")) as f:
        full_cfg = json.load(f)
    tcfg = full_cfg["talker_config"]

    n_layers = tcfg["num_hidden_layers"]
    hidden = tcfg["hidden_size"]          # 896
    inter = tcfg["intermediate_size"]      # 4864
    vocab = tcfg["vocab_size"]             # 8448 (codec)
    emb_size = tcfg["embedding_size"]      # 2048 (codec embed dim)
    n_heads = tcfg["num_attention_heads"]  # 14
    n_kv = tcfg["num_key_value_heads"]     # 2
    head_dim = tcfg["head_dim"]            # 64
    q_size = n_heads * head_dim            # 896
    kv_size = n_kv * head_dim              # 128

    print("Qwen2.5-Omni talker config:")
    print(f"  layers={n_layers} hidden={hidden} inter={inter} vocab={vocab} "
          f"emb_size={emb_size}")
    print(f"  heads={n_heads} kv_heads={n_kv} head_dim={head_dim} "
          f"mrope_section={tcfg['rope_scaling']['mrope_section']}")

    weights = ShardedSafetensors(model_dir)
    prefix = "talker." if "talker.model.embed_tokens.weight" in weights else ""

    def fetch(name):
        import torch
        return weights.get(prefix + name).to(torch.float32).numpy()

    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "wb") as out:

        def save(arr, expected_shape):
            assert arr.shape == tuple(expected_shape), \
                f"shape {arr.shape} != expected {tuple(expected_shape)}"
            np.ascontiguousarray(arr, dtype=np.float32).tofile(out)

        def save_fc(name, out_features, in_features):
            """nntrainer FC stores FP32 weights as (in, out): transpose HF."""
            w = fetch(name)
            assert w.shape == (out_features, in_features), \
                f"{name}: {w.shape} != ({out_features},{in_features})"
            np.ascontiguousarray(w.T, dtype=np.float32).tofile(out)

        # 1. thinker_to_talker_proj (first consumer of input0; w^T then bias)
        save_fc("thinker_to_talker_proj.weight", hidden, emb_size)
        save(fetch("thinker_to_talker_proj.bias"), (hidden,))

        # 2. decoder layers (gate before up — graph DFS order)
        for i in range(n_layers):
            lp = f"model.layers.{i}."
            save(fetch(lp + "input_layernorm.weight"), (hidden,))
            save_fc(lp + "self_attn.q_proj.weight", q_size, hidden)
            save(fetch(lp + "self_attn.q_proj.bias"), (q_size,))
            save_fc(lp + "self_attn.k_proj.weight", kv_size, hidden)
            save(fetch(lp + "self_attn.k_proj.bias"), (kv_size,))
            save_fc(lp + "self_attn.v_proj.weight", kv_size, hidden)
            save(fetch(lp + "self_attn.v_proj.bias"), (kv_size,))
            save_fc(lp + "self_attn.o_proj.weight", hidden, q_size)
            save(fetch(lp + "post_attention_layernorm.weight"), (hidden,))
            save_fc(lp + "mlp.gate_proj.weight", inter, hidden)
            save_fc(lp + "mlp.up_proj.weight", inter, hidden)
            save_fc(lp + "mlp.down_proj.weight", hidden, inter)
            print(f"  layer {i + 1:2d}/{n_layers} written")

        # 3. final norm
        save(fetch("model.norm.weight"), (hidden,))

        # 4. codec_head (896 -> 8448, no bias)
        save_fc("codec_head.weight", vocab, hidden)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.1f} MiB)")

    # codec embed_tokens table [vocab, emb_size] — host lookup, NOT transposed.
    codec_embed = fetch("model.embed_tokens.weight")
    assert codec_embed.shape == (vocab, emb_size)
    ce_path = os.path.join(args.output_dir, args.codec_embed_name)
    np.ascontiguousarray(codec_embed, dtype=np.float32).tofile(ce_path)
    print(f"Wrote {ce_path} "
          f"({os.path.getsize(ce_path) / (1024 * 1024):.1f} MiB)")

    # ------------------------------------------------------------------
    # Companion config files. The Talker class reads talker_config promoted
    # to the top level (so the common Transformer setup consumes it as-is).
    # ------------------------------------------------------------------
    cfg_out = dict(tcfg)
    cfg_out["tie_word_embeddings"] = False        # talker has no tying
    cfg_out["architectures"] = ["Qwen25OmniTalker"]
    cfg_out["model_type"] = "qwen2_5_omni_talker"
    # keep the full thinker_config too, so end-to-end (Stage C) can build the
    # Thinker from the same dir.
    cfg_out["thinker_config"] = full_cfg["thinker_config"]
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)

    # NOTE: copy tokenizer.json (base Transformer ctor + Stage C thinker need
    # it) but NOT tokenizer_config.json — its presence makes main.cpp apply the
    # chat template to the raw arg, which would mangle the "stageA:" prefix.
    src = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(args.output_dir, "tokenizer.json"))

    generation_cfg = {
        "bos_token_id": tcfg["tts_codec_start_token_id"],   # 8293
        "pad_token_id": tcfg["tts_codec_pad_token_id"],     # 8292
        "eos_token_id": [tcfg["tts_codec_pad_token_id"],
                         tcfg["tts_codec_end_token_id"]],   # [8292, 8294]
        "do_sample": False,
    }
    with open(os.path.join(args.output_dir, "generation_config.json"),
              "w") as f:
        json.dump(generation_cfg, f, indent=4)

    nntr_cfg = {
        "model_type": "CausalLM",
        "model_tensor_type": "FP32-FP32",
        "model_file_name": args.output_name,
        "codec_embed_path":
            os.path.abspath(os.path.join(args.output_dir, args.codec_embed_name)),
        "thinker_model_path": os.path.abspath(args.thinker_dir)
            if args.thinker_dir else "",
        "speaker_bos_token": args.speaker_bos,
        "thinker_max_new_tokens": 16,
        "talker_max_new_tokens": 128,
        "fc_layer_dtype": "FP32",
        "embedding_dtype": "FP32",
        "lmhead_dtype": "FP32",
        "lora_rank": 0,
        "lora_alpha": 0,
        "lora_target": [],
        "bad_word_ids": [],
        "fsu": False,
        "fsu_lookahead": 2,
        "num_to_generate": 1024,
        "init_seq_len": 1024,
        "max_seq_len": 2048,
        "batch_size": 1,
        "tokenizer_file":
            os.path.abspath(os.path.join(args.output_dir, "tokenizer.json")),
        "sample_input": "What is the capital of France? Answer in one word.",
    }
    with open(os.path.join(args.output_dir, "nntr_config.json"), "w") as f:
        json.dump(nntr_cfg, f, indent=4)

    print(f"Wrote config/generation_config/nntr_config to {args.output_dir}")


if __name__ == "__main__":
    main()

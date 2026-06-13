# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>

# @file weight_converter.py
# @brief HuggingFace -> nntrainer FP32 weight converter for Qwen3 dense
#        models (e.g. Qwen/Qwen3-0.6B). Streams weights from the safetensors
#        shards; no full-model RAM or bleeding-edge transformers needed.
#
#        The on-disk order is the symbolic graph's DFS-from-output order
#        (Model::compile), NOT layer creation order. For Qwen3 that is, per
#        decoder block:
#          input_layernorm,
#          q_proj(w^T), q_norm, k_proj(w^T), k_norm, v_proj(w^T),
#          o_proj(w^T),
#          post_attention_layernorm,
#          gate_proj(w^T), up_proj(w^T), down_proj(w^T)
#        i.e. gate BEFORE up (swiglu({gate, up}) DFS), and the q/k RMSNorms
#        sit right after their projections. Matches the (corrected) GGUF
#        converter gguf_to_nntrainer.py.
#
#        Qwen3 attention has NO q/k/v/o bias; q_norm/k_norm are RMSNorm over
#        head_dim. tie_word_embeddings shares lm_head with the embedding, so
#        no separate lm_head weight is written when tied.
#
# @usage
#   python weight_converter.py --model_path Qwen/Qwen3-0.6B \
#       --output_dir ./qwen3-0.6b-fp32
#
# @author Jijoong Moon <jijoong.moon@samsung.com>

import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open


def resolve_model_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo_id=model_path,
        allow_patterns=["config.json", "generation_config.json",
                        "tokenizer.json", "tokenizer_config.json",
                        "*.safetensors", "*.safetensors.index.json"])


class ShardedSafetensors:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        index = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            single = os.path.join(model_dir, "model.safetensors")
            with safe_open(single, framework="pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self._handles = {}

    def __contains__(self, key):
        return key in self.weight_map

    def get(self, key):
        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                os.path.join(self.model_dir, shard), framework="pt")
        return self._handles[shard].get_tensor(key)


def main():
    ap = argparse.ArgumentParser(
        description="Convert a HuggingFace Qwen3 dense model to nntrainer FP32")
    ap.add_argument("--model_path", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--output_dir", type=str, default="./qwen3-0.6b-fp32")
    ap.add_argument("--output_name", type=str,
                    default="nntr_qwen3_0.6b_fp32.bin")
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    n_layers = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", hidden // n_heads)
    tied = cfg.get("tie_word_embeddings", False)
    q_size, kv_size = n_heads * head_dim, n_kv * head_dim
    print(f"Qwen3: layers={n_layers} hidden={hidden} inter={inter} "
          f"vocab={vocab} heads={n_heads} kv={n_kv} head_dim={head_dim} "
          f"tied={tied}")

    w = ShardedSafetensors(model_dir)

    def fetch(name):
        return w.get(name).to(torch.float32).numpy()

    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "wb") as out:
        def save(arr, shape):
            assert arr.shape == tuple(shape), \
                f"{arr.shape} != {tuple(shape)}"
            np.ascontiguousarray(arr, dtype=np.float32).tofile(out)

        def save_fc(name, out_f, in_f):
            x = fetch(name)
            assert x.shape == (out_f, in_f), f"{name}: {x.shape}"
            np.ascontiguousarray(x.T, dtype=np.float32).tofile(out)

        save(fetch("model.embed_tokens.weight"), (vocab, hidden))
        for i in range(n_layers):
            p = f"model.layers.{i}."
            save(fetch(p + "input_layernorm.weight"), (hidden,))
            save_fc(p + "self_attn.q_proj.weight", q_size, hidden)
            save(fetch(p + "self_attn.q_norm.weight"), (head_dim,))
            save_fc(p + "self_attn.k_proj.weight", kv_size, hidden)
            save(fetch(p + "self_attn.k_norm.weight"), (head_dim,))
            save_fc(p + "self_attn.v_proj.weight", kv_size, hidden)
            save_fc(p + "self_attn.o_proj.weight", hidden, q_size)
            save(fetch(p + "post_attention_layernorm.weight"), (hidden,))
            save_fc(p + "mlp.gate_proj.weight", inter, hidden)  # gate first!
            save_fc(p + "mlp.up_proj.weight", inter, hidden)
            save_fc(p + "mlp.down_proj.weight", hidden, inter)
            print(f"  layer {i + 1:2d}/{n_layers}")
        save(fetch("model.norm.weight"), (hidden,))
        if not tied:
            save_fc("lm_head.weight", vocab, hidden)

    print(f"Wrote {out_path} "
          f"({os.path.getsize(out_path) / (1024 * 1024):.1f} MiB)")


if __name__ == "__main__":
    main()

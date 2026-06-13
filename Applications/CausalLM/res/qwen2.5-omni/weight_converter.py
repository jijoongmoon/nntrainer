# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>

# @file weight_converter.py
# @brief Convert the Qwen2.5-Omni Thinker text model (e.g.
#        Qwen/Qwen2.5-Omni-3B) into an nntrainer FP32 .bin weight file plus
#        the config/tokenizer files the CausalLM application expects.
#
#        Only the Thinker's text decoder is exported (text in / text out).
#        The audio/vision encoders, the Talker and Token2Wav are skipped.
#
#        Weights are streamed tensor-by-tensor straight from the safetensors
#        shards, so neither a bleeding-edge transformers version nor enough
#        RAM for the full multimodal model is required.
#
#        The on-disk order must match nntrainer's weight load order, which
#        is the symbolic graph's DFS-from-output order (Model::compile in
#        api/ccapi/src/tensor_api_graph.cpp), NOT layer creation order:
#          embedding
#          per layer: input_layernorm,
#                     q_proj (w^T, b), k_proj (w^T, b), v_proj (w^T, b),
#                     o_proj (w^T),
#                     post_attention_layernorm,
#                     gate_proj (w^T), up_proj (w^T), down_proj (w^T)
#          final norm
#          lm_head (w^T, only when tie_word_embeddings is false)
#        Note: ffn_gate comes BEFORE ffn_up. createMlp() creates the up FC
#        first, but the graph wires swiglu({gate, up}) and the DFS visits the
#        gate branch first, so the gate weights are loaded first. Verified
#        empirically: a synthetic random-weight checkpoint reproduces HF
#        greedy tokens exactly (12/12) with gate-first and diverges with
#        up-first. Same ordering as the qwen2 converter (commit de8f981cf).
#
# @usage
#   python weight_converter.py \
#       --model_path Qwen/Qwen2.5-Omni-3B \
#       --output_dir ./qwen2.5-omni-3b
#
#   Then quantize the FP32 output to Q4_0 with the nntr_quantize tool:
#   nntr_quantize ./qwen2.5-omni-3b --fc_dtype Q4_0
#
# @author Jijoong Moon <jijoong.moon@samsung.com>

import argparse
import json
import os
import shutil

import numpy as np
import torch
from safetensors import safe_open


def resolve_model_dir(model_path: str) -> str:
    """Return a local directory for the checkpoint, downloading if needed."""
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_path,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "*.safetensors",
            "*.safetensors.index.json",
        ],
    )


class ShardedSafetensors:
    """Lazy tensor reader over (possibly sharded) safetensors files."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            single = os.path.join(model_dir, "model.safetensors")
            if not os.path.exists(single):
                raise FileNotFoundError(
                    f"no model.safetensors(.index.json) under {model_dir}")
            with safe_open(single, framework="pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self._handles = {}

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                os.path.join(self.model_dir, shard), framework="pt")
        return self._handles[shard].get_tensor(key)


def main():
    parser = argparse.ArgumentParser(
        description="Convert the Qwen2.5-Omni Thinker text model to an "
                    "nntrainer FP32 .bin")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen2.5-Omni-3B",
                        help="Local checkpoint dir or HuggingFace repo id")
    parser.add_argument("--output_dir", type=str,
                        default="./qwen2.5-omni-3b",
                        help="Directory for the .bin and config files")
    parser.add_argument("--output_name", type=str,
                        default="nntr_qwen2.5_omni_3b_fp32.bin",
                        help="Output .bin filename")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    thinker_cfg = cfg.get("thinker_config", cfg)
    text_cfg = thinker_cfg.get("text_config", thinker_cfg)

    n_layers = text_cfg["num_hidden_layers"]
    hidden = text_cfg["hidden_size"]
    inter = text_cfg["intermediate_size"]
    vocab = text_cfg["vocab_size"]
    n_heads = text_cfg["num_attention_heads"]
    n_kv = text_cfg["num_key_value_heads"]
    head_dim = text_cfg.get("head_dim", hidden // n_heads)
    tied = text_cfg.get("tie_word_embeddings", False)
    q_size = n_heads * head_dim
    kv_size = n_kv * head_dim

    print("Qwen2.5-Omni thinker text config:")
    print(f"  layers={n_layers} hidden={hidden} inter={inter} vocab={vocab}")
    print(f"  heads={n_heads} kv_heads={n_kv} head_dim={head_dim} tied={tied}")

    weights = ShardedSafetensors(model_dir)

    # Full Omni checkpoints prefix the thinker; thinker-only ones do not.
    prefix = "thinker." if "thinker.model.embed_tokens.weight" in weights \
        else ""

    def fetch(name: str) -> np.ndarray:
        return weights.get(prefix + name).to(torch.float32).numpy()

    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "wb") as out:

        def save(arr: np.ndarray, expected_shape):
            assert arr.shape == tuple(expected_shape), \
                f"shape {arr.shape} != expected {tuple(expected_shape)}"
            np.ascontiguousarray(arr, dtype=np.float32).tofile(out)

        def save_fc(name: str, out_features: int, in_features: int):
            """nntrainer FC stores FP32 weights as (in, out): transpose HF's
            (out, in)."""
            w = fetch(name)
            assert w.shape == (out_features, in_features), \
                f"{name}: {w.shape} != ({out_features},{in_features})"
            np.ascontiguousarray(w.T, dtype=np.float32).tofile(out)

        # 1. embedding (no transpose)
        save(fetch("model.embed_tokens.weight"), (vocab, hidden))

        # 2. decoder layers
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
            # graph DFS loads ffn_gate before ffn_up — keep this order!
            save_fc(lp + "mlp.gate_proj.weight", inter, hidden)
            save_fc(lp + "mlp.up_proj.weight", inter, hidden)
            save_fc(lp + "mlp.down_proj.weight", hidden, inter)
            print(f"  layer {i + 1:2d}/{n_layers} written")

        # 3. final norm
        save(fetch("model.norm.weight"), (hidden,))

        # 4. lm_head (Omni-3B is untied)
        if not tied:
            save_fc("lm_head.weight", vocab, hidden)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.1f} MiB)")

    # ------------------------------------------------------------------
    # Companion files: config.json (as-is; the app flattens thinker_config
    # itself), generation_config.json (the HF one is empty — build one with
    # the token ids the runtime needs), tokenizer files, nntr_config.json.
    # ------------------------------------------------------------------
    shutil.copyfile(os.path.join(model_dir, "config.json"),
                    os.path.join(args.output_dir, "config.json"))
    for fname in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(args.output_dir, fname))

    eos = thinker_cfg.get("eos_token_id", 151645)
    generation_cfg = {
        "bos_token_id": thinker_cfg.get("bos_token_id", 151644),
        "pad_token_id": thinker_cfg.get("pad_token_id", 151643),
        "eos_token_id": eos if isinstance(eos, list) else [eos],
        "do_sample": False,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
    }
    with open(os.path.join(args.output_dir, "generation_config.json"),
              "w") as f:
        json.dump(generation_cfg, f, indent=4)

    nntr_cfg = {
        "model_type": "CausalLM",
        "model_tensor_type": "FP32-FP32",
        "model_file_name": args.output_name,
        "fc_layer_dtype": "FP32",
        "embedding_dtype": "FP32",
        "lmhead_dtype": "FP32",
        "lora_rank": 0,
        "lora_alpha": 0,
        "lora_target": [],
        "bad_word_ids": [],
        "fsu": False,
        "fsu_lookahead": 2,
        "num_to_generate": 512,
        "init_seq_len": 1024,
        "max_seq_len": 2048,
        "batch_size": 1,
        "tokenizer_file":
            os.path.abspath(os.path.join(args.output_dir, "tokenizer.json")),
        "sample_input":
            "<|im_start|>user\nGive me a short introduction to large "
            "language model.<|im_end|>\n<|im_start|>assistant\n",
    }
    with open(os.path.join(args.output_dir, "nntr_config.json"), "w") as f:
        json.dump(nntr_cfg, f, indent=4)

    print(f"Wrote config/generation_config/nntr_config to {args.output_dir}")
    print("Next: quantize with "
          f"`nntr_quantize {args.output_dir} --fc_dtype Q4_0`")


if __name__ == "__main__":
    main()

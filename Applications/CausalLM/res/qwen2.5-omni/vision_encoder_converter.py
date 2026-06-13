# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>

# @file vision_encoder_converter.py
# @brief Convert the Qwen2.5-Omni vision tower (thinker.visual.*) into an
#        nntrainer FP32 .bin for Qwen25OmniVisionEncoder.
#
#        Weight order = symbolic graph DFS-from-output order:
#          patch_embed (Conv3d[out,in,t,h,w] reshaped to Linear[in*t*h*w,out]),
#          per block: norm1, q(w^T,b), k(w^T,b), v(w^T,b), proj(w^T,b),
#                     norm2, gate(w^T,b), up(w^T,b), down(w^T,b),
#          merger: ln_q, mlp0(w^T,b), mlp2(w^T,b)
#        (gate before up; q/k/v/proj/mlp carry bias; norms are RMSNorm weight
#        only). The 2D-RoPE table is computed in the layer, not stored.
#
# @usage
#   python vision_encoder_converter.py --model_path Qwen/Qwen2.5-Omni-3B \
#       --output_dir ./qwen2.5-omni-3b-vision --grid_h 8 --grid_w 8
#
# @author Jijoong Moon <jijoong.moon@samsung.com>

import argparse
import json
import os

import numpy as np
import torch

from weight_converter import ShardedSafetensors, resolve_model_dir


def main():
    ap = argparse.ArgumentParser(
        description="Convert the Qwen2.5-Omni vision tower to nntrainer FP32")
    ap.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--output_dir", type=str, default="./qwen2.5-omni-3b-vision")
    ap.add_argument("--output_name", type=str,
                    default="nntr_qwen2.5_omni_3b_vision_fp32.bin")
    ap.add_argument("--head_output_name", type=str,
                    default="nntr_qwen2.5_omni_3b_vision_head_fp32.bin")
    ap.add_argument("--grid_h", type=int, default=8,
                    help="raw patch rows for the compiled graph (<= 8)")
    ap.add_argument("--grid_w", type=int, default=8,
                    help="raw patch cols for the compiled graph (<= 8)")
    ap.add_argument("--grid_t", type=int, default=1,
                    help="temporal patches (1 for images; videos use >1)")
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    vc = cfg["thinker_config"]["vision_config"]
    dim = vc.get("hidden_size", vc.get("embed_dim", 1280))
    depth = vc.get("depth", 32)
    inter = vc.get("intermediate_size", 3420)
    out_hidden = vc.get("out_hidden_size", 2048)
    merge = vc.get("spatial_merge_size", 2)
    patch = vc.get("patch_size", 14)
    tpatch = vc.get("temporal_patch_size", 2)
    in_ch = vc.get("in_channels", vc.get("in_chans", 3))
    patch_dim = in_ch * tpatch * patch * patch
    merge_hidden = dim * merge * merge

    w = ShardedSafetensors(model_dir)
    pre = "thinker.visual."

    def fetch(name):
        return w.get(pre + name).to(torch.float32).numpy()

    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "wb") as out:
        def save(arr, shape):
            assert arr.shape == tuple(shape), f"{arr.shape} != {tuple(shape)}"
            np.ascontiguousarray(arr, dtype=np.float32).tofile(out)

        def save_fc(name, out_f, in_f, bias):
            x = fetch(name + ".weight")
            assert x.shape == (out_f, in_f), f"{name}: {x.shape}"
            np.ascontiguousarray(x.T, dtype=np.float32).tofile(out)
            if bias:
                save(fetch(name + ".bias"), (out_f,))

        # patch_embed: Conv3d weight [dim, in, t, ph, pw] -> Linear[patch_dim,dim]
        pe = fetch("patch_embed.proj.weight").reshape(dim, patch_dim)
        np.ascontiguousarray(pe.T, dtype=np.float32).tofile(out)  # no bias

        for i in range(depth):
            b = f"blocks.{i}."
            save(fetch(b + "norm1.weight"), (dim,))
            save_fc(b + "attn.q", dim, dim, True)
            save_fc(b + "attn.k", dim, dim, True)
            save_fc(b + "attn.v", dim, dim, True)
            save_fc(b + "attn.proj", dim, dim, True)
            save(fetch(b + "norm2.weight"), (dim,))
            save_fc(b + "mlp.gate_proj", inter, dim, True)  # gate first
            save_fc(b + "mlp.up_proj", inter, dim, True)
            save_fc(b + "mlp.down_proj", dim, inter, True)
            print(f"  block {i + 1:2d}/{depth}")

        save(fetch("merger.ln_q.weight"), (dim,))  # ln_q ends the main graph

    # merger MLP runs as a separate head graph (the 2x2 reshape changes the
    # row count, which the main graph's incremental slicing can't express)
    head_path = os.path.join(args.output_dir, args.head_output_name)
    with open(head_path, "wb") as out:
        def save_fc_h(name, out_f, in_f):
            x = fetch(name + ".weight")
            assert x.shape == (out_f, in_f), f"{name}: {x.shape}"
            np.ascontiguousarray(x.T, dtype=np.float32).tofile(out)
            np.ascontiguousarray(fetch(name + ".bias"),
                                 dtype=np.float32).tofile(out)
        save_fc_h("merger.mlp.0", merge_hidden, merge_hidden)
        save_fc_h("merger.mlp.2", out_hidden, merge_hidden)

    print(f"Wrote {out_path} "
          f"({os.path.getsize(out_path) / (1024 * 1024):.1f} MiB)")
    print(f"Wrote {head_path} "
          f"({os.path.getsize(head_path) / (1024 * 1024):.1f} MiB)")

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"architectures": ["Qwen25OmniVisionEncoder"],
                   "model_type": "qwen2_5_omni_vision_encoder",
                   "vision_config": vc}, f, indent=4)
    with open(os.path.join(args.output_dir, "nntr_config.json"), "w") as f:
        json.dump({
            "model_type": "Model", "skip_tokenizer": True,
            "model_tensor_type": "FP32-FP32",
            "model_file_name": args.output_name,
            "vision_head_file_name": args.head_output_name,
            "fc_layer_dtype": "FP32", "embedding_dtype": "FP32",
            "batch_size": 1, "num_to_generate": 0,
            "grid_h": args.grid_h, "grid_w": args.grid_w,
            "grid_t": args.grid_t,
            "init_seq_len": args.grid_t * args.grid_h * args.grid_w,
            "max_seq_len": args.grid_t * args.grid_h * args.grid_w,
            "sample_input": "./patches.bin",
        }, f, indent=4)
    print(f"Wrote config/nntr_config to {args.output_dir} "
          f"(grid t={args.grid_t} {args.grid_h}x{args.grid_w})")


if __name__ == "__main__":
    main()

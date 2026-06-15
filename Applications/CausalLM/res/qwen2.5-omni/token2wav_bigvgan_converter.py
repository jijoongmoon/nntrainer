# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file token2wav_bigvgan_converter.py
# @brief Convert Qwen2.5-Omni Token2Wav BigVGAN weights to an nntrainer .bin.
#
# Writes weights in DFS-from-output order to match the nntrainer BigVGAN graph
# load order (see [[weight-bin-load-order-dfs]]). Layout rules (all FP32):
#   - plain Conv1d weight [out,in,k]  -> [out,in,1,k]  (byte-identical, no transpose)
#   - ConvTranspose1d  [in,out,k]     -> [out,in,1,k]  (transpose dims (0,1))
#   - SnakeBeta alpha/beta [C]        -> [C] raw log-domain (NO exp; runtime exp)
#   - conv_post has NO bias
#
# BigVGAN config: mel 80, up_init_ch 1536, rates [5,3,2,2,2,2], up_kernels
# [11,7,4,4,4,4], resblock kernels [3,7,11], dilations [[1,3,5]]x3.
#
#   python token2wav_bigvgan_converter.py --model_path <snapshot> \
#       --output_dir <model dir> [--output_name bigvgan.bin]

import argparse
import json
import os

import numpy as np
import torch

from weight_converter import ShardedSafetensors, resolve_model_dir

PREFIX = "token2wav.code2wav_bigvgan_model."

UP_RATES = [5, 3, 2, 2, 2, 2]
UP_KERNELS = [11, 7, 4, 4, 4, 4]
UP_INIT_CH = 1536
RESBLOCK_KERNELS = [3, 7, 11]
RESBLOCK_DILATIONS = [1, 3, 5]
MEL_DIM = 80
CONV_PRE_K = 7
CONV_POST_K = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--output_name", default="bigvgan.bin")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_dir = resolve_model_dir(args.model_path)
    weights = ShardedSafetensors(model_dir)

    def fetch(name):
        return weights.get(PREFIX + name).to(torch.float32).numpy()

    stats = {"tensors": 0, "floats": 0}
    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "wb") as out:

        def write(arr, expected_shape, label):
            assert arr.shape == tuple(expected_shape), \
                f"{label}: {arr.shape} != {tuple(expected_shape)}"
            a = np.ascontiguousarray(arr, dtype=np.float32)
            a.tofile(out)
            stats["tensors"] += 1
            stats["floats"] += a.size

        def save_conv(name, out_ch, in_ch, k, bias):
            # plain Conv1d [out,in,k] -> [out,in,1,k] (no transpose)
            w = fetch(name + ".weight")
            write(w.reshape(out_ch, in_ch, 1, k), (out_ch, in_ch, 1, k),
                  name + ".weight")
            if bias:
                write(fetch(name + ".bias"), (out_ch,), name + ".bias")

        def save_convT(name, in_ch, out_ch, k):
            # ConvTranspose1d [in,out,k] -> [out,in,1,k] (transpose (0,1))
            w = fetch(name + ".weight")
            assert w.shape == (in_ch, out_ch, k), \
                f"{name}: {w.shape} != ({in_ch},{out_ch},{k})"
            wt = np.ascontiguousarray(w.transpose(1, 0, 2)).reshape(
                out_ch, in_ch, 1, k)
            write(wt, (out_ch, in_ch, 1, k), name + ".weight")
            write(fetch(name + ".bias"), (out_ch,), name + ".bias")

        def save_snake(name, ch):
            write(fetch(name + ".alpha"), (ch,), name + ".alpha")
            write(fetch(name + ".beta"), (ch,), name + ".beta")

        # ---- DFS-from-output order (must match the nntrainer graph) ----
        # 1) conv_pre
        save_conv("conv_pre", UP_INIT_CH, MEL_DIM, CONV_PRE_K, bias=True)

        ch = UP_INIT_CH
        for i in range(6):
            out_ch = ch // 2
            # 2a) upsample i (ConvTranspose1d)
            save_convT(f"ups.{i}.0", ch, out_ch, UP_KERNELS[i])
            # 2b) the 3 AMPBlocks at this stage (resblocks i*3 + b)
            for b in range(3):
                r = i * 3 + b
                kb = RESBLOCK_KERNELS[b]
                for kk in range(3):  # 3 sub-blocks per AMPBlock
                    save_snake(f"resblocks.{r}.activations.{2 * kk}.act", out_ch)
                    save_conv(f"resblocks.{r}.convs1.{kk}", out_ch, out_ch, kb,
                              bias=True)
                    save_snake(f"resblocks.{r}.activations.{2 * kk + 1}.act",
                               out_ch)
                    save_conv(f"resblocks.{r}.convs2.{kk}", out_ch, out_ch, kb,
                              bias=True)
            ch = out_ch
        # 3) activation_post
        save_snake("activation_post.act", ch)
        # 4) conv_post (NO bias)
        save_conv("conv_post", 1, ch, CONV_POST_K, bias=False)

    # sanity: every bigvgan tensor consumed exactly once
    all_keys = [k for k in weights.keys() if k.startswith(PREFIX)] \
        if hasattr(weights, "keys") else None
    print(f"wrote {out_path}: {stats['tensors']} tensors, {stats['floats']} floats "
          f"({os.path.getsize(out_path) / (1024 * 1024):.1f} MiB)")
    if all_keys is not None:
        print(f"checkpoint bigvgan tensors: {len(all_keys)} (expect == {stats['tensors']})")

    cfg = {
        "architectures": ["Qwen25OmniBigVGAN"],
        "model_type": "qwen2_5_omni_bigvgan",
        "mel_dim": MEL_DIM,
        "upsample_initial_channel": UP_INIT_CH,
        "upsample_rates": UP_RATES,
        "upsample_kernel_sizes": UP_KERNELS,
        "resblock_kernel_sizes": RESBLOCK_KERNELS,
        "resblock_dilation_sizes": [RESBLOCK_DILATIONS] * 3,
        "conv_pre_kernel": CONV_PRE_K,
        "conv_post_kernel": CONV_POST_K,
        "model_file_name": args.output_name,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote config.json")


if __name__ == "__main__":
    main()

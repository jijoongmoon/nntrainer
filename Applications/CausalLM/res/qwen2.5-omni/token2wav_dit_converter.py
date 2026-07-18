# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file token2wav_dit_converter.py
# @brief Convert Qwen2.5-Omni Token2Wav DiT weights to nntrainer .bin files.
#
# Emits dit.bin in the COMPILED GRAPH ORDER dumped via NNTR_DIT_SUMMARY=1
# (see [[weight-bin-load-order-dfs]]): proj, time_mlp.0, time_mlp.2, then per
# block i: attn_norm.linear, to_q, to_k, to_v, to_out.0, ff.ff.0, ff.ff.3,
# then norm_out.linear, proj_out — 159 FC layers x (weight, bias) = 318
# tensors, all FP32. FC weights transpose [out,in] -> [in,out]
# (weight_converter.py convention); biases raw.
#
# EXCLUDED from dit.bin (dit-2B-confirmed.md / HANDOFF §6.3.3):
#   - 40 input_embed.spk_encoder.* (ECAPA; bring-up injects host-side)
#   - text_embed.codec_embed.weight -> separate raw codec_embed.bin
#     (host gather; the CFG null branch needs row 0, C8)
#   - rotary_embed.inv_freq (host recompute)
#
#   python token2wav_dit_converter.py --model_path <snapshot> \
#       --output_dir <model dir> [--output_name dit.bin]

import argparse
import json
import os

import numpy as np
import torch

from weight_converter import ShardedSafetensors, resolve_model_dir

PREFIX = "token2wav.code2wav_dit_model."

HIDDEN = 1024
DEPTH = 22
FF_INNER = 2048
MEL_DIM = 80
COND_W = 912  # mel 80 + ecapa 128 + code 512 + speaker 192
TIME_FREQ = 256
CODEC_VOCAB = 8194
CODEC_DIM = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--output_name", default="dit.bin")
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
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            out.write(arr.tobytes())
            stats["tensors"] += 1
            stats["floats"] += arr.size

        def save_fc(name, in_dim, out_dim):
            w = fetch(name + ".weight")  # [out, in]
            write(w.T, (in_dim, out_dim), name + ".weight")
            b = fetch(name + ".bias")
            write(b, (out_dim,), name + ".bias")

        # 1) input_embed.proj (the host assembles the 912-wide concat)
        save_fc("input_embed.proj", COND_W, HIDDEN)
        # 2) time_mlp: Linear(256->1024), SiLU, Linear(1024->1024)
        save_fc("time_embed.time_mlp.0", TIME_FREQ, HIDDEN)
        save_fc("time_embed.time_mlp.2", HIDDEN, HIDDEN)
        # 3) blocks 0..21 (graph order: adaln, q, k, v, o, ff0, ff3)
        for i in range(DEPTH):
            p = f"transformer_blocks.{i}."
            save_fc(p + "attn_norm.linear", HIDDEN, 6 * HIDDEN)
            save_fc(p + "attn.to_q", HIDDEN, HIDDEN)
            save_fc(p + "attn.to_k", HIDDEN, HIDDEN)
            save_fc(p + "attn.to_v", HIDDEN, HIDDEN)
            save_fc(p + "attn.to_out.0", HIDDEN, HIDDEN)
            save_fc(p + "ff.ff.0", HIDDEN, FF_INNER)
            save_fc(p + "ff.ff.3", FF_INNER, HIDDEN)
        # 4) final AdaLN (chunk-2 [scale, shift], C4) + projection
        save_fc("norm_out.linear", HIDDEN, 2 * HIDDEN)
        save_fc("proj_out", HIDDEN, MEL_DIM)

    assert stats["tensors"] == 318, f"wrote {stats['tensors']} tensors != 318"
    print(f"wrote {out_path}: {stats['tensors']} tensors, {stats['floats']} "
          f"floats ({os.path.getsize(out_path) / (1024 * 1024):.1f} MiB)")

    # codec embed: raw [8194, 512], host-gathered (row 0 = CFG null branch)
    ce = fetch("text_embed.codec_embed.weight")
    assert ce.shape == (CODEC_VOCAB, CODEC_DIM), ce.shape
    ce_path = os.path.join(args.output_dir, "codec_embed.bin")
    with open(ce_path, "wb") as f:
        f.write(np.ascontiguousarray(ce, dtype=np.float32).tobytes())
    print(f"wrote {ce_path}: {ce.shape}")

    # rotary inv_freq: the checkpoint values are bf16-rounded (0.75^j-like),
    # NOT the 10000^(-2j/64) formula (4.4e-4 rel diff -> 5e-2 cos error at
    # s=127). The host MUST use these, not a recompute.
    inv = fetch("rotary_embed.inv_freq")
    assert inv.shape == (32,), inv.shape
    inv_path = os.path.join(args.output_dir, "inv_freq.bin")
    with open(inv_path, "wb") as f:
        f.write(np.ascontiguousarray(inv, dtype=np.float32).tobytes())
    print(f"wrote {inv_path}: {inv.shape}")

    # ECAPA-TDNN speaker encoder: 40 tensors, raw [C_out, C_in, K] conv
    # layout, in the fixed order the C++ EcapaTdnn::load consumes
    # (see ecapa_tdnn.h; validated spec in docs/omni-speech).
    ecapa_names = ["blocks.0.conv"]
    for i in (1, 2, 3):
        ecapa_names += [
            f"blocks.{i}.tdnn1.conv",
            f"blocks.{i}.res2net_block.blocks.0.conv",
            f"blocks.{i}.tdnn2.conv",
            f"blocks.{i}.se_block.conv1",
            f"blocks.{i}.se_block.conv2",
        ]
    ecapa_names += ["mfa.conv", "asp.tdnn.conv", "asp.conv", "fc"]
    ec_path = os.path.join(args.output_dir, "ecapa.bin")
    n_ecapa = 0
    with open(ec_path, "wb") as f:
        for name in ecapa_names:
            for suffix in (".weight", ".bias"):
                arr = fetch("input_embed.spk_encoder." + name + suffix)
                f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
                n_ecapa += 1
    assert n_ecapa == 40, n_ecapa
    print(f"wrote {ec_path}: {n_ecapa} tensors")

    cfg = {
        "architectures": ["Qwen25OmniDiT"],
        "model_type": "qwen2_5_omni_dit",
        "hidden_size": HIDDEN,
        "depth": DEPTH,
        "num_heads": 16,
        "head_dim": 64,
        "ff_inner": FF_INNER,
        "mel_dim": MEL_DIM,
        "repeats": 2,
        "codec_vocab": CODEC_VOCAB,
        "codec_dim": CODEC_DIM,
        "enc_dim": 128,
        "enc_emb_dim": 192,
        "time_freq": TIME_FREQ,
        "block_size": 24,
        "guidance_scale": 0.5,
        "rope_theta": 10000.0,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    nntr_cfg = {
        "model_file_name": args.output_name,
        "model_tensor_type": "FP32-FP32",
        "model_type": "Model",
        "skip_tokenizer": True,
    }
    with open(os.path.join(args.output_dir, "nntr_config.json"), "w") as f:
        json.dump(nntr_cfg, f, indent=2)
    print("wrote config.json / nntr_config.json")


if __name__ == "__main__":
    main()

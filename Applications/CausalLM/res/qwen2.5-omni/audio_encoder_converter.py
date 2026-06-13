# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>

# @file audio_encoder_converter.py
# @brief Convert the Qwen2.5-Omni audio tower (thinker.audio_tower.*) into
#        the two nntrainer FP32 .bin files used by Qwen25OmniAudioEncoder:
#          - encoder bin: conv front-end + sinusoid pos-embed + 32 blocks
#          - head bin:    ln_post + proj (runs after host-side AvgPool)
#
#        Weight order follows the symbolic graph's DFS-from-output order
#        (NOT creation order; see res/qwen2.5-omni/weight_converter.py):
#          conv1 (w,b), conv2 (w,b), pos_embed[100,1280],
#          per layer: attn_ln (g,b), q (w^T,b), k (w^T, NO bias), v (w^T,b),
#                     out (w^T,b), ffn_ln (g,b), fc1 (w^T,b), fc2 (w^T,b)
#        head bin: ln_post (g,b), proj (w^T,b)
#
#        The sinusoidal positional table is NOT in the checkpoint (a
#        non-persistent buffer); it is regenerated here exactly as HF's
#        SinusoidsPositionEmbedding (sin||cos halves, max_timescale 1e4) and
#        baked into the encoder bin. Positions restart per 200-mel window, so
#        only the first n_window (=100) rows are needed.
#
# @usage
#   python audio_encoder_converter.py \
#       --model_path Qwen/Qwen2.5-Omni-3B --output_dir ./qwen2.5-omni-3b-audio
#
# @author Jijoong Moon <jijoong.moon@samsung.com>

import argparse
import json
import os
import sys

import numpy as np
import torch

# repack_q4_0 / quantize_q4_0 live in the qwen3 GGUF converter
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "qwen3", "qwen3-0.6b"))
from gguf_to_nntrainer import quantize_q4_0, repack_q4_0  # noqa: E402
from weight_converter import ShardedSafetensors, resolve_model_dir  # noqa: E402


def sinusoid_table(n_positions: int, channels: int,
                   max_timescale: float = 10000.0) -> np.ndarray:
    """HF Qwen2_5Omni SinusoidsPositionEmbedding: concat[sin, cos] halves."""
    assert channels % 2 == 0
    half = channels // 2
    log_inc = np.log(max_timescale) / (half - 1)
    inv_timescales = np.exp(-log_inc * np.arange(half, dtype=np.float64))
    scaled = np.arange(n_positions, dtype=np.float64)[:, None] * \
        inv_timescales[None, :]
    table = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
    return table.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Convert the Qwen2.5-Omni audio tower to nntrainer .bin")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--output_dir", type=str,
                        default="./qwen2.5-omni-3b-audio")
    parser.add_argument("--output_name", type=str,
                        default="nntr_qwen2.5_omni_3b_audio_fp32.bin")
    parser.add_argument("--head_output_name", type=str,
                        default="nntr_qwen2.5_omni_3b_audio_head_fp32.bin")
    parser.add_argument("--fc-dtype", choices=["fp32", "q4_0"], default="fp32",
                        help="dtype for the 32 encoder blocks' FC weights "
                             "(q/k/v/out/fc1/fc2). conv, layernorm, pos-embed "
                             "and the head proj always stay FP32.")
    parser.add_argument("--target", choices=["x86", "arm"], default="x86",
                        help="Q4_0 repack layout (x86 -> q4_0x8, arm -> "
                             "q4_0x4); ignored for fp32")
    args = parser.parse_args()
    interleave = 8 if args.target == "x86" else 4

    model_dir = resolve_model_dir(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # tag the encoder bin with its FC dtype (the head bin stays FP32)
    if args.fc_dtype == "q4_0" and "fp32" in args.output_name:
        args.output_name = args.output_name.replace(
            "fp32", "q4_0_" + args.target)

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    thinker_cfg = cfg.get("thinker_config", cfg)
    audio_cfg = thinker_cfg.get("audio_config", thinker_cfg)

    d_model = audio_cfg.get("d_model", 1280)
    n_layers = audio_cfg.get("encoder_layers", 32)
    n_heads = audio_cfg.get("encoder_attention_heads", 20)
    ffn = audio_cfg.get("encoder_ffn_dim", 5120)
    n_mels = audio_cfg.get("num_mel_bins", 128)
    n_window = audio_cfg.get("n_window", 100)
    out_dim = audio_cfg.get("output_dim", 2048)

    print("Qwen2.5-Omni audio tower config:")
    print(f"  d_model={d_model} layers={n_layers} heads={n_heads} ffn={ffn}")
    print(f"  n_mels={n_mels} n_window={n_window} output_dim={out_dim}")

    weights = ShardedSafetensors(model_dir)
    prefix = "thinker.audio_tower." \
        if "thinker.audio_tower.conv1.weight" in weights else "audio_tower."

    def fetch(name: str) -> np.ndarray:
        return weights.get(prefix + name).to(torch.float32).numpy()

    enc_path = os.path.join(args.output_dir, args.output_name)
    with open(enc_path, "wb") as out:

        def save(arr: np.ndarray, expected_shape):
            assert arr.shape == tuple(expected_shape), \
                f"shape {arr.shape} != expected {tuple(expected_shape)}"
            np.ascontiguousarray(arr, dtype=np.float32).tofile(out)

        def save_fc(name: str, out_features: int, in_features: int,
                    bias: bool):
            w = fetch(name + ".weight")
            assert w.shape == (out_features, in_features), \
                f"{name}: {w.shape} != ({out_features},{in_features})"
            if args.fc_dtype == "q4_0":
                # quantize the [out, in] weight per 32-element row then repack
                # into nntrainer's q4_0x{interleave} layout (matches the FC
                # layer's weight_dtype=Q4_0 on the loader side)
                assert in_features % 32 == 0 and out_features % interleave == 0
                raw = quantize_q4_0(np.ascontiguousarray(w, dtype=np.float32))
                out.write(repack_q4_0(raw, out_features, in_features,
                                      interleave))
            else:
                np.ascontiguousarray(w.T, dtype=np.float32).tofile(out)
            if bias:  # bias always stays FP32
                save(fetch(name + ".bias"), (out_features,))

        def save_ln(name: str, dim: int):
            save(fetch(name + ".weight"), (dim,))  # gamma
            save(fetch(name + ".bias"), (dim,))    # beta

        # conv kernels are byte-identical to PyTorch [out, in, k]: no transpose
        save(fetch("conv1.weight"), (d_model, n_mels, 3))
        save(fetch("conv1.bias"), (d_model,))
        save(fetch("conv2.weight"), (d_model, d_model, 3))
        save(fetch("conv2.bias"), (d_model,))

        save(sinusoid_table(n_window, d_model), (n_window, d_model))

        for i in range(n_layers):
            lp = f"layers.{i}."
            save_ln(lp + "self_attn_layer_norm", d_model)
            save_fc(lp + "self_attn.q_proj", d_model, d_model, bias=True)
            save_fc(lp + "self_attn.k_proj", d_model, d_model, bias=False)
            save_fc(lp + "self_attn.v_proj", d_model, d_model, bias=True)
            save_fc(lp + "self_attn.out_proj", d_model, d_model, bias=True)
            save_ln(lp + "final_layer_norm", d_model)
            save_fc(lp + "fc1", ffn, d_model, bias=True)
            save_fc(lp + "fc2", d_model, ffn, bias=True)
            print(f"  layer {i + 1:2d}/{n_layers} written")

    head_path = os.path.join(args.output_dir, args.head_output_name)
    with open(head_path, "wb") as out:

        def savh(arr, shape):
            assert arr.shape == tuple(shape)
            np.ascontiguousarray(arr, dtype=np.float32).tofile(out)

        savh(fetch("ln_post.weight"), (d_model,))
        savh(fetch("ln_post.bias"), (d_model,))
        savh(fetch("proj.weight").T, (d_model, out_dim))
        savh(fetch("proj.bias"), (out_dim,))

    print(f"Wrote {enc_path} "
          f"({os.path.getsize(enc_path) / (1024 * 1024):.1f} MiB)")
    print(f"Wrote {head_path} "
          f"({os.path.getsize(head_path) / (1024 * 1024):.1f} MiB)")

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({
            "architectures": ["Qwen25OmniAudioEncoder"],
            "model_type": "qwen2_5_omni_audio_encoder",
            "d_model": d_model,
            "encoder_layers": n_layers,
            "encoder_attention_heads": n_heads,
            "encoder_ffn_dim": ffn,
            "num_mel_bins": n_mels,
            "n_window": n_window,
            "output_dim": out_dim,
        }, f, indent=4)

    fc_cfg = "Q4_0" if args.fc_dtype == "q4_0" else "FP32"
    with open(os.path.join(args.output_dir, "nntr_config.json"), "w") as f:
        json.dump({
            "model_type": "Model",
            "skip_tokenizer": True,
            "model_tensor_type": "FP32-FP32",
            "model_file_name": args.output_name,
            "audio_head_file_name": args.head_output_name,
            "fc_layer_dtype": fc_cfg,
            "embedding_dtype": "FP32",
            "batch_size": 1,
            "init_seq_len": n_window,
            "max_seq_len": n_window,
            "num_to_generate": 0,
            "bad_word_ids": [],
            "sample_input": "./mel_input.bin",
        }, f, indent=4)

    print(f"Wrote config.json / nntr_config.json to {args.output_dir}")


if __name__ == "__main__":
    main()

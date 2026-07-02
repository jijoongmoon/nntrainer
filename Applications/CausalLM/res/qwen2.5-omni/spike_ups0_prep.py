# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file spike_ups0_prep.py
# @brief Prepare raw-float32 inputs for the BigVGAN ups0 (ConvTranspose1d)
#        micro-spike that validates the conv2d_transpose width-bug fix + the
#        converter (0,1) transpose against the HF dump.
#
#   Reads ups.0 weight/bias from the checkpoint and conv_pre/ups0 from the
#   Token2Wav dump; writes raw float32 .bin files the C++ spike consumes:
#     ups0_weight.bin   : nntrainer load order [kernel, bias]
#                         kernel = ConvT1d weight [in,out,k] -> [out,in,1,k]
#                                  (transpose dims (0,1), unsqueeze kh=1)
#                         bias   = [out]
#     conv_pre.bin      : input  [1,1536,128] (row-major)
#     ups0_expected.bin : expect [1, 768,640] (row-major)
#
#   python spike_ups0_prep.py --model_path <local snapshot> \
#       --dump /tmp/omni_t2w_dump --outdir /tmp/ups0_spike

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="local HF snapshot dir (offline)")
    ap.add_argument("--dump", default="/tmp/omni_t2w_dump")
    ap.add_argument("--outdir", default="/tmp/ups0_spike")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from safetensors import safe_open
    shard = os.path.join(args.model_path, "model-00003-of-00003.safetensors")
    pre = "token2wav.code2wav_bigvgan_model."
    with safe_open(shard, framework="np") as f:
        w = f.get_tensor(pre + "ups.0.0.weight")  # [in=1536, out=768, k=11]
        b = f.get_tensor(pre + "ups.0.0.bias")     # [768]
    print(f"ups0 weight {w.shape} {w.dtype}; bias {b.shape}", flush=True)
    assert w.shape == (1536, 768, 11), w.shape

    # PyTorch ConvTranspose1d [in,out,k] -> nntrainer Conv2DTranspose [out,in,1,k]
    w_nntr = np.ascontiguousarray(w.transpose(1, 0, 2)).astype(np.float32)  # [768,1536,11]
    b_nntr = np.ascontiguousarray(b).astype(np.float32)

    # nntrainer load order for Conv2DTransposeLayer: kernel then bias
    with open(os.path.join(args.outdir, "ups0_weight.bin"), "wb") as fo:
        fo.write(w_nntr.tobytes())
        fo.write(b_nntr.tobytes())

    conv_pre = np.load(os.path.join(args.dump, "conv_pre.npy")).astype(np.float32)
    ups0 = np.load(os.path.join(args.dump, "ups0.npy")).astype(np.float32)
    print(f"conv_pre {conv_pre.shape}; ups0 {ups0.shape}", flush=True)
    assert conv_pre.shape == (1, 1536, 128), conv_pre.shape
    assert ups0.shape == (1, 768, 640), ups0.shape

    np.ascontiguousarray(conv_pre).tofile(
        os.path.join(args.outdir, "conv_pre.bin"))
    np.ascontiguousarray(ups0).tofile(
        os.path.join(args.outdir, "ups0_expected.bin"))

    meta = {
        "in_ch": 1536, "out_ch": 768, "kernel": 11, "stride": 5, "pad": 3,
        "T_in": 128, "T_out": 640,
        "weight_floats": int(w_nntr.size), "bias_floats": int(b_nntr.size),
    }
    print("prepared ups0 spike ->", args.outdir, meta, flush=True)


if __name__ == "__main__":
    main()

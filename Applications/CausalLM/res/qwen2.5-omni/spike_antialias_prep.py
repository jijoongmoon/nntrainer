# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file spike_antialias_prep.py
# @brief Prepare raw-float32 inputs for the antialiased_snake C++ spike, which
#        runs HF activation_post (TorchActivation1d) on the dumped stage5 and
#        compares to the dumped activation_post.
#
#   weight.bin   : [alpha[24], beta[24]] (nntrainer load order alpha-then-beta)
#   input.bin    : stage5            [1,24,30720]
#   expected.bin : activation_post   [1,24,30720]
#
#   python spike_antialias_prep.py --model_path <snapshot> --dump /tmp/omni_t2w_dump \
#       --outdir /tmp/antialias_spike

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dump", default="/tmp/omni_t2w_dump")
    ap.add_argument("--outdir", default="/tmp/antialias_spike")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from safetensors import safe_open
    shard = os.path.join(args.model_path, "model-00003-of-00003.safetensors")
    pre = "token2wav.code2wav_bigvgan_model.activation_post.act."
    with safe_open(shard, framework="np") as f:
        alpha = f.get_tensor(pre + "alpha").astype(np.float32)  # [24]
        beta = f.get_tensor(pre + "beta").astype(np.float32)
    with open(os.path.join(args.outdir, "weight.bin"), "wb") as fo:
        fo.write(np.ascontiguousarray(alpha).tobytes())
        fo.write(np.ascontiguousarray(beta).tobytes())

    stage5 = np.load(os.path.join(args.dump, "stage5.npy")).astype(np.float32)
    ap_out = np.load(os.path.join(args.dump, "activation_post.npy")).astype(np.float32)
    assert stage5.shape == (1, 24, 30720), stage5.shape
    assert ap_out.shape == (1, 24, 30720), ap_out.shape
    np.ascontiguousarray(stage5).tofile(os.path.join(args.outdir, "input.bin"))
    np.ascontiguousarray(ap_out).tofile(os.path.join(args.outdir, "expected.bin"))

    print("prepared antialias spike ->", args.outdir,
          {"C": 24, "T": 30720, "alpha0": float(alpha[0]), "beta0": float(beta[0])},
          flush=True)


if __name__ == "__main__":
    main()

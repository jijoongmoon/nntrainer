# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file spike_bigvgan_prep.py
# @brief Raw-float32 inputs for the full BigVGAN graph spike (Stage C):
#   input.bin    : processed_mel  [1,80,128]   (conv_pre input; host process_mel already applied)
#   expected.bin : wav            [30720]       (final, post clamp[-1,1])
#   stageN.bin   : stage{N}/conv_pre/ups{N}/activation_post for bisecting
#
#   python spike_bigvgan_prep.py --dump /tmp/omni_t2w_dump --outdir /tmp/bigvgan_spike

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/omni_t2w_dump")
    ap.add_argument("--outdir", default="/tmp/bigvgan_spike")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def dump(name, src):
        a = np.load(os.path.join(args.dump, src)).astype(np.float32)
        np.ascontiguousarray(a).tofile(os.path.join(args.outdir, name))
        print(f"  {name} <- {src} {a.shape}")

    dump("input.bin", "processed_mel.npy")   # [1,80,128]
    dump("expected.bin", "wav.npy")          # [30720]
    # intermediates for bisecting a mismatch
    dump("conv_pre.bin", "conv_pre.npy")     # [1,1536,128]
    for i in range(6):
        dump(f"ups{i}.bin", f"ups{i}.npy")
        dump(f"stage{i}.bin", f"stage{i}.npy")
    dump("activation_post.bin", "activation_post.npy")
    print("prepared bigvgan spike ->", args.outdir)


if __name__ == "__main__":
    main()

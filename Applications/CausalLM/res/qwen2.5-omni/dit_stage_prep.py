# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file dit_stage_prep.py
# @brief Convert /tmp/omni_{dit,t2w}_dump .npy refs into the raw side-input
#        .bin files consumed by Qwen25OmniDiT::run() (Stage A/B bring-up).
#
#   python dit_stage_prep.py --dit_dump /tmp/omni_dit_dump \
#       --t2w_dump /tmp/omni_t2w_dump --outdir <dir>
#
# Emits: codes.bin (i32[64]), ecapa_pos/neg.bin (f32[128]), spk.bin (f32[192]),
# x_in.bin + t.bin (Stage A), noise.bin (f32[128*80], Stage B) and reference
# copies velocity_ref.bin / guided_ref.bin / null_ref.bin / dit_mel_ref.bin.

import argparse
import os

import numpy as np


def save(path, arr, dtype):
    np.ascontiguousarray(arr, dtype=dtype).tofile(path)
    print(f"wrote {path} {arr.shape} {dtype.__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit_dump", default="/tmp/omni_dit_dump")
    ap.add_argument("--t2w_dump", default="/tmp/omni_t2w_dump")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    d, t2w, o = args.dit_dump, args.t2w_dump, args.outdir

    codes = np.load(os.path.join(d, "codes.npy")).reshape(-1)
    save(os.path.join(o, "codes.bin"), codes, np.int32)

    ecapa = np.load(os.path.join(d, "ecapa_out.npy"))  # [2,128] row0=pos
    save(os.path.join(o, "ecapa_pos.bin"), ecapa[0], np.float32)
    save(os.path.join(o, "ecapa_neg.bin"), ecapa[1], np.float32)

    spk = np.load(os.path.join(d, "cond192.npy")).reshape(-1)  # [192]
    save(os.path.join(o, "spk.bin"), spk, np.float32)

    # Stage A: one-step input + time + per-branch/combined velocity refs
    x_in = np.load(os.path.join(d, "x_in.npy")).reshape(-1)  # [1,128,80]
    save(os.path.join(o, "x_in.bin"), x_in, np.float32)
    t = np.load(os.path.join(d, "t_value.npy")).reshape(-1)  # [1]
    save(os.path.join(o, "t.bin"), t, np.float32)
    proj = np.load(os.path.join(d, "proj_out.npy"))  # [2,128,80] rows=cond/null
    save(os.path.join(o, "guided_ref.bin"), proj[0].reshape(-1), np.float32)
    save(os.path.join(o, "null_ref.bin"), proj[1].reshape(-1), np.float32)
    vel = np.load(os.path.join(d, "velocity_cfg.npy")).reshape(-1)
    save(os.path.join(o, "velocity_ref.bin"), vel, np.float32)

    # Stage B: HF noise slice + final mel target
    noise = np.load(os.path.join(t2w, "initial_state_full.npy"))[:, :128, :]
    save(os.path.join(o, "noise.bin"), noise.reshape(-1), np.float32)
    mel = np.load(os.path.join(t2w, "dit_mel.npy")).reshape(-1)  # [1,80,128]
    save(os.path.join(o, "dit_mel_ref.bin"), mel, np.float32)


if __name__ == "__main__":
    main()

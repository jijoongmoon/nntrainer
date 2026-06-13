# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file verify_talker_stageA.py
# @brief Stage A: feed the HF-dumped per-step talker inputs_embeds into the
#        nntrainer Talker graph and check the codec token ids match HF exactly.
#
#   1) run test_talker.py first to produce the .npy ground truth in <dump>.
#   2) python verify_talker_stageA.py --dump <dump> --talker_dir <talker> \
#          --binary <nntrainer_causallm_binary>
#
# Writes prefill.f32 / steps.f32 (raw [int32 n][int32 d][float...]) the C++
# Stage A path reads, runs the binary with "stageA:<dump>", parses "CODES: ...".

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np


def write_f32(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, d = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, d))
        arr.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/omni_talker_dump")
    ap.add_argument("--talker_dir", required=True)
    ap.add_argument("--binary", required=True)
    args = ap.parse_args()

    prefill = np.load(os.path.join(args.dump, "talker_prefill_embeds.npy"))
    steps = np.load(os.path.join(args.dump, "talker_step_embeds.npy"))
    codes = np.load(os.path.join(args.dump, "talker_codes.npy")).reshape(-1)
    with open(os.path.join(args.dump, "meta.json")) as f:
        meta = json.load(f)
    print("meta:", meta)
    print(f"prefill {prefill.shape}, steps {steps.shape}, "
          f"HF codes ({len(codes)}): {codes.tolist()}")

    write_f32(os.path.join(args.dump, "prefill.f32"), prefill)
    write_f32(os.path.join(args.dump, "steps.f32"),
              steps if steps.size else np.zeros((0, prefill.shape[1]), np.float32))

    res = subprocess.run([args.binary, args.talker_dir, "stageA:" + args.dump],
                         capture_output=True, text=True, env=dict(os.environ))
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-3000:])
        sys.stderr.write(res.stderr[-3000:])
        sys.exit("nntr Talker Stage A failed")

    line = next((ln for ln in res.stdout.splitlines()
                 if ln.startswith("CODES:")), None)
    if line is None:
        sys.stderr.write(res.stdout[-3000:])
        sys.exit("no CODES: line in nntr output")
    produced = [int(x) for x in line[len("CODES:"):].split()]
    print(f"nntr produced ({len(produced)}): {produced}")

    # produced = [c0, c1, ..., cN]; HF codes = [c0..c_{N-1}] (eos dropped).
    n = len(codes)
    head = produced[:n]
    tail = produced[n:]
    match = head == codes.tolist()
    first_div = next((i for i in range(min(n, len(produced)))
                      if produced[i] != int(codes[i])), None)
    print(f"\n=== Stage A result ===")
    print(f"codec ids match HF (first {n}): {match}")
    if not match:
        print(f"first divergence at step {first_div}: "
              f"nntr={produced[first_div] if first_div is not None else '?'} "
              f"hf={int(codes[first_div]) if first_div is not None else '?'}")
    print(f"nntr tail (expect one eos in {{{meta['codec_eos']},"
          f"{meta['codec_pad']}}}): {tail}")
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()

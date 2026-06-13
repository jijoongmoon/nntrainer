# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file verify_talker_stageC.py
# @brief Stage B/C: drive the nntrainer Thinker -> capture -> assemble -> Talker
#        and check it reproduces HF.
#
#   Stage B: the Talker writes assembled_prefill.f32 / assembled_steps.f32 from
#            its OWN thinker capture; we diff them against the HF dumps
#            (talker_prefill_embeds.npy / talker_step_embeds.npy).
#   Stage C: the Talker's codec ids (feedback decode) must match talker_codes.npy
#            and its REPLY_IDS must match reply_ids.npy.
#
#   python verify_talker_stageC.py --dump <dump> --talker_dir <talker> \
#       --binary <nntr_causallm>

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np


def write_i32(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr).reshape(-1), dtype=np.int32)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", arr.size))
        arr.tofile(f)


def read_f32(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        v = np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/omni_talker_dump")
    ap.add_argument("--talker_dir", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    prompt_ids = np.load(os.path.join(args.dump, "prompt_ids.npy"))
    reply_ids = np.load(os.path.join(args.dump, "reply_ids.npy"))
    hf_codes = np.load(os.path.join(args.dump, "talker_codes.npy")).reshape(-1)
    hf_prefill = np.load(os.path.join(args.dump, "talker_prefill_embeds.npy"))
    hf_steps = np.load(os.path.join(args.dump, "talker_step_embeds.npy"))

    write_i32(os.path.join(args.dump, "prompt_ids.i32"), prompt_ids)
    write_i32(os.path.join(args.dump, "reply_ids.i32"), reply_ids)
    write_i32(os.path.join(args.dump, "codes.i32"), hf_codes)

    res = subprocess.run([args.binary, args.talker_dir, "stageBC:" + args.dump],
                         capture_output=True, text=True, env=dict(os.environ),
                         timeout=args.timeout)
    sys.stdout.write(res.stdout[-1500:])
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-3000:])
        sys.exit("nntr Talker stageBC failed")

    def grab(tag):
        ln = next((l for l in res.stdout.splitlines() if l.startswith(tag)),
                  None)
        return [int(x) for x in ln[len(tag):].split()] if ln else None

    nntr_reply = grab("REPLY_IDS:")
    produced = grab("CODES:")

    print("\n=== Stage B: assembled embeds vs HF ===")
    for name, hf in [("assembled_prefill.f32", hf_prefill),
                     ("assembled_steps.f32", hf_steps)]:
        p = os.path.join(args.dump, name)
        if not os.path.exists(p):
            print(f"  {name}: MISSING")
            continue
        mine = read_f32(p)
        if mine.shape != hf.shape:
            print(f"  {name}: shape {mine.shape} != HF {hf.shape}")
            continue
        d = np.abs(mine - hf)
        print(f"  {name}: shape {mine.shape}  max|Δ|={d.max():.3e}  "
              f"mean|Δ|={d.mean():.3e}")

    print("\n=== Stage C: codec ids vs HF ===")
    print(f"  REPLY_IDS match: {nntr_reply == reply_ids.tolist()}")
    if nntr_reply != reply_ids.tolist():
        print(f"    nntr reply: {nntr_reply}")
        print(f"    HF   reply: {reply_ids.tolist()}")
    n = len(hf_codes)
    head = produced[:n] if produced else []
    match = head == hf_codes.tolist()
    print(f"  codec ids match HF (first {n}): {match}")
    if not match and head:
        div = next((i for i in range(min(n, len(head)))
                    if head[i] != int(hf_codes[i])), None)
        if div is not None:
            print(f"    first divergence at {div}: nntr={head[div]} "
                  f"hf={int(hf_codes[div])}")
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()

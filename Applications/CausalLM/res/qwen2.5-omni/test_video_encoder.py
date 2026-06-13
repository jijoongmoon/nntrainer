# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file test_video_encoder.py
# @brief Verify the nntrainer vision encoder on a multi-frame (video) grid
#        (grid_t > 1) against HF, using synthetic patches (no video decode).

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np
import torch
from weight_converter import ShardedSafetensors, resolve_model_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--vision_model_dir", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--grid_t", type=int, default=2)
    ap.add_argument("--gh", type=int, default=8)
    ap.add_argument("--gw", type=int, default=8)
    ap.add_argument("--workdir", default="/tmp/omni_video_test")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    model_dir = resolve_model_dir(args.model_path)
    with open(os.path.join(model_dir, "config.json")) as f:
        vc = json.load(f)["thinker_config"]["vision_config"]
    patch_dim = (vc.get("in_channels", 3) * vc["temporal_patch_size"] *
                 vc["patch_size"] ** 2)
    t, gh, gw = args.grid_t, args.gh, args.gw
    seq = t * gh * gw

    rng = np.random.default_rng(11)
    patches = rng.standard_normal((seq, patch_dim)).astype(np.float32) * 0.5
    print(f"grid t,h,w = {t},{gh},{gw}; patches {patches.shape}")

    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniVisionEncoderConfig)
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniVisionEncoder)
    config = Qwen2_5OmniVisionEncoderConfig(**{
        k: v for k, v in vc.items()
        if k in Qwen2_5OmniVisionEncoderConfig().to_dict()})
    vis = Qwen2_5OmniVisionEncoder(config).eval()
    w = ShardedSafetensors(model_dir)
    pre = "thinker.visual."
    state = {k[len(pre):]: w.get(k).to(torch.float32)
             for k in w.weight_map if k.startswith(pre)}
    vis.load_state_dict(state, strict=False)
    with torch.no_grad():
        ref = vis(torch.from_numpy(patches),
                  grid_thw=torch.tensor([[t, gh, gw]])).numpy()
    print(f"HF reference: {ref.shape}")

    patch_path = os.path.join(args.workdir, "vpatches.bin")
    with open(patch_path, "wb") as f:
        f.write(struct.pack("<ii", gh, gw))  # per-frame grid; grid_t from cfg
        np.ascontiguousarray(patches, dtype=np.float32).tofile(f)
    res = subprocess.run([args.binary, args.vision_model_dir, patch_path],
                         capture_output=True, text=True, env=dict(os.environ))
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-2000:])
        sys.exit("nntr video encoder failed")
    with open(patch_path + ".embd", "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        got = np.fromfile(f, dtype=np.float32).reshape(n, d)
    assert got.shape == ref.shape, f"{got.shape} != {ref.shape}"
    cos = (got * ref).sum(1) / (
        np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1) + 1e-9)
    print(f"tokens={n} max|diff|={np.abs(got - ref).max():.5f} "
          f"cos[min/mean]={cos.min():.6f}/{cos.mean():.6f}")
    print("PASS" if cos.min() > 0.999 else "FAIL")
    sys.exit(0 if cos.min() > 0.999 else 1)


if __name__ == "__main__":
    main()

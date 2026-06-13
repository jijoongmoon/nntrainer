# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>

# @file test_vision_encoder.py
# @brief Verify the nntrainer Qwen2.5-Omni vision encoder against HF for a
#        single-window image (<= 112x112 px -> 8x8 patches -> 4x4 merged,
#        where windowed attention == full attention so no reordering applies).
#
#        Uses the HF image processor to flatten patches (guaranteeing identical
#        patch ordering), runs the HF Qwen2_5OmniVisionEncoder (fp32) and the
#        nntr_causallm vision encoder on the same patches, compares embeddings.
#
# @usage python test_vision_encoder.py --vision_model_dir ./qwen2.5-omni-3b-vision \
#           --binary <build>/Applications/CausalLM/nntr_causallm
#
# @author Jijoong Moon <jijoong.moon@samsung.com>

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
    ap.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--vision_model_dir", type=str, required=True)
    ap.add_argument("--binary", type=str, required=True)
    ap.add_argument("--px", type=int, default=112, help="square image size")
    ap.add_argument("--workdir", type=str, default="/tmp/omni_vision_test")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    model_dir = resolve_model_dir(args.model_path)

    with open(os.path.join(model_dir, "config.json")) as f:
        vc = json.load(f)["thinker_config"]["vision_config"]
    patch = vc["patch_size"]

    # deterministic RGB image
    rng = np.random.default_rng(7)
    img = (rng.uniform(0, 255, (args.px, args.px, 3))).astype(np.uint8)
    from PIL import Image
    pil = Image.fromarray(img)

    from transformers import Qwen2VLImageProcessor
    proc = Qwen2VLImageProcessor(
        patch_size=patch, temporal_patch_size=vc["temporal_patch_size"],
        merge_size=vc["spatial_merge_size"], min_pixels=56 * 56,
        max_pixels=args.px * args.px)
    enc = proc(images=pil, return_tensors="np")
    pixel_values = enc["pixel_values"].astype(np.float32)   # [seq, 1176]
    grid = enc["image_grid_thw"][0]                          # [t, h, w]
    t, gh, gw = int(grid[0]), int(grid[1]), int(grid[2])
    print(f"grid t,h,w = {t},{gh},{gw}; patches {pixel_values.shape}")
    assert t == 1, "single frame only"

    # HF reference vision tower (fp32)
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
    missing, unexpected = vis.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected: {unexpected[:5]}"
    with torch.no_grad():
        ref = vis(torch.from_numpy(pixel_values),
                  grid_thw=torch.tensor([[t, gh, gw]])).numpy()
    print(f"HF reference: {ref.shape}")

    # nntr binary
    patch_path = os.path.join(args.workdir, "patches.bin")
    with open(patch_path, "wb") as f:
        f.write(struct.pack("<ii", gh, gw))
        np.ascontiguousarray(pixel_values, dtype=np.float32).tofile(f)
    res = subprocess.run([args.binary, args.vision_model_dir, patch_path],
                         capture_output=True, text=True, env=dict(os.environ))
    sys.stdout.write(res.stdout[-400:])
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-2000:])
        sys.exit("nntr_causallm vision failed")

    with open(patch_path + ".embd", "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        got = np.fromfile(f, dtype=np.float32).reshape(n, d)
    assert got.shape == ref.shape, f"{got.shape} != {ref.shape}"
    cos = (got * ref).sum(1) / (
        np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1) + 1e-9)
    print(f"tokens={n} max|diff|={np.abs(got - ref).max():.5f} "
          f"cos[min/mean]={cos.min():.6f}/{cos.mean():.6f}")
    ok = cos.min() > 0.999
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

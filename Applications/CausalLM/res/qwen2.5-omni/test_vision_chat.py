# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file test_vision_chat.py
# @brief Verify image+text -> text (Qwen25OmniVisionChat) vs HF thinker greedy.

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np
import torch
from weight_converter import resolve_model_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--vchat_dir", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--px", type=int, default=112)
    ap.add_argument("--question", default="What is in this image? Answer briefly.")
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--workdir", default="/tmp/omni_vchat_test")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    model_dir = resolve_model_dir(args.model_path)
    with open(os.path.join(model_dir, "config.json")) as f:
        vc = json.load(f)["thinker_config"]["vision_config"]

    # structured (low-entropy) image: red / green / blue vertical thirds
    img = np.zeros((args.px, args.px, 3), dtype=np.uint8)
    w3 = args.px // 3
    img[:, :w3, 0] = 220
    img[:, w3:2 * w3, 1] = 220
    img[:, 2 * w3:, 2] = 220
    from PIL import Image
    from transformers import Qwen2VLImageProcessor
    proc = Qwen2VLImageProcessor(patch_size=vc["patch_size"],
                                 temporal_patch_size=vc["temporal_patch_size"],
                                 merge_size=vc["spatial_merge_size"],
                                 min_pixels=56 * 56, max_pixels=args.px * args.px)
    enc = proc(images=Image.fromarray(img), return_tensors="np")
    pv = enc["pixel_values"].astype(np.float32)
    t, gh, gw = [int(v) for v in enc["image_grid_thw"][0]]
    n_img = (gh // 2) * (gw // 2)
    print(f"grid {t},{gh},{gw}; {n_img} image tokens; patches {pv.shape}")

    patch_path = os.path.join(args.workdir, "patches.bin")
    with open(patch_path, "wb") as f:
        f.write(struct.pack("<ii", gh, gw))
        np.ascontiguousarray(pv, dtype=np.float32).tofile(f)

    # identical prompt to the nntr builder
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n<|vision_bos|>" + "<|IMAGE|>" * n_img +
              "<|vision_eos|>" + args.question +
              "<|im_end|>\n<|im_start|>assistant\n")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    ids = tok.encode(prompt, add_special_tokens=False).ids

    from transformers import Qwen2_5OmniThinkerForConditionalGeneration
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            pixel_values=torch.from_numpy(pv),
            image_grid_thw=torch.tensor([[t, gh, gw]]),
            max_new_tokens=args.max_new, do_sample=False)
    hf_ids = out[0][len(ids):].tolist()
    hf_text = tok.decode(hf_ids, skip_special_tokens=True)
    print("HF :", repr(hf_text))

    res = subprocess.run(
        [args.binary, args.vchat_dir, "image:" + patch_path + " " + args.question],
        capture_output=True, text=True, env=dict(os.environ))
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-2000:])
        sys.exit("nntr vision chat failed")
    # extract assistant text
    out_txt = res.stdout
    nntr_text = ""
    if "assistant" in out_txt:
        tail = out_txt.split("assistant", 1)[1]
        for line in tail.splitlines():
            if "====" in line:
                break
            nntr_text += line + "\n"
    print("nntr:", repr(nntr_text.strip()[:200]))
    sys.exit(0)


if __name__ == "__main__":
    main()

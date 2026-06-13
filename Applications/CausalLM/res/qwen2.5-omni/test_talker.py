# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file test_talker.py
# @brief Dump HF Qwen2.5-Omni Talker ground truth (codec token ids + per-step
#        talker inputs_embeds + thinker_reply_part) for nntrainer verification.
#
#   Stage A (decoder+converter): nntrainer feeds the dumped per-step
#       talker inputs_embeds (talker_prefill_embeds.npy + talker_step_embeds.npy)
#       straight into its Talker graph and must reproduce talker_codes.npy
#       (exact id match under greedy).
#   Stage B (thinker capture): nntrainer's own captured thinker_reply_part must
#       match thinker_reply_part.npy.
#   Stage C (end-to-end): nntrainer thinker+talker must reproduce talker_codes.
#
# Everything runs greedy (do_sample=False) on CPU fp32 for determinism. The
# Token2Wav decoder is stubbed out (Phase 2), so only codec ids are produced.
#
#   python test_talker.py --model_path Qwen/Qwen2.5-Omni-3B \
#       --prompt "What is the capital of France? Answer in one word." \
#       --outdir /tmp/omni_talker_dump

import argparse
import json
import os

import numpy as np
import torch
from weight_converter import resolve_model_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--prompt",
                    default="What is the capital of France? Answer in one word.")
    ap.add_argument("--speaker", default="Chelsie")
    ap.add_argument("--thinker_max_new", type=int, default=16)
    ap.add_argument("--talker_max_new", type=int, default=128)
    ap.add_argument("--outdir", default="/tmp/omni_talker_dump")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(0)

    model_dir = resolve_model_dir(args.model_path)

    # spk_dict.pt is a torch pickle; transformers blocks torch.load on
    # torch < 2.6 (CVE-2025-32434). It is the official Qwen checkpoint, so
    # bypass the guard for this local-only load.
    import transformers.utils.import_utils as _iu
    _iu.check_torch_load_is_safe = lambda *a, **k: None
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _m
    if hasattr(_m, "check_torch_load_is_safe"):
        _m.check_torch_load_is_safe = lambda *a, **k: None

    from transformers import Qwen2_5OmniForConditionalGeneration

    print("loading model (CPU fp32) ...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float32).eval()

    talker = model.talker
    print("speakers:", list(model.speaker_map.keys()),
          "| Chelsie bos:", model.speaker_map[args.speaker]["bos_token"],
          flush=True)

    # ---- build a text-only prompt with the omni chat template (no processor:
    #      the image processor needs preprocessor_config.json we don't ship).
    system = ("You are Qwen, a virtual human developed by the Qwen Team, "
              "Alibaba Group, capable of perceiving auditory and visual inputs, "
              "as well as generating text and speech.")
    prompt_str = (f"<|im_start|>system\n{system}<|im_end|>\n"
                  f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    ids = tok.encode(prompt_str, add_special_tokens=False).ids
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones(1, len(ids), dtype=torch.long)
    prompt_len = int(input_ids.shape[1])
    print(f"prompt tokens: {prompt_len}", flush=True)

    # ---- capture per-step talker inputs_embeds (the proj input) -------------
    proj_inputs = []

    def proj_pre_hook(module, inp):
        proj_inputs.append(inp[0].detach().to(torch.float32).cpu().numpy())

    talker.thinker_to_talker_proj.register_forward_pre_hook(proj_pre_hook)

    # ---- capture thinker_reply_part on the first talker.forward -------------
    reply_part_holder = {}

    def talker_pre_hook(module, args_, kwargs_):
        if "thinker_reply_part" not in reply_part_holder:
            trp = kwargs_.get("thinker_reply_part", None)
            if trp is not None:
                reply_part_holder["thinker_reply_part"] = \
                    trp.detach().to(torch.float32).cpu().numpy()

    talker.register_forward_pre_hook(talker_pre_hook, with_kwargs=True)

    # ---- stub Token2Wav (Phase 2 out of scope); capture codec ids -----------
    captured = {}

    import torch.nn as nn

    class FakeToken2Wav(nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.float32

        def forward(self, codes, **kw):
            captured["codes"] = codes.detach().cpu().numpy()
            return torch.zeros(1, 1, 16, dtype=torch.float32)

    model.token2wav = FakeToken2Wav()

    # ---- run the real omni generate path, greedy on both stages -------------
    with torch.no_grad():
        thinker_seq, _wav = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speaker=args.speaker,
            return_audio=True,
            thinker_max_new_tokens=args.thinker_max_new,
            talker_max_new_tokens=args.talker_max_new,
            do_sample=False,            # shared -> thinker greedy
            talker_do_sample=False,     # talker greedy
        )

    reply_ids = thinker_seq[0][prompt_len:].tolist()
    codes = captured["codes"].reshape(-1).astype(np.int64)
    print(f"thinker reply ids ({len(reply_ids)}):", reply_ids, flush=True)
    print(f"talker codec codes ({len(codes)}):", codes.tolist(), flush=True)

    # proj_inputs[0] = prefill [1, L0, 2048]; proj_inputs[1:] = [1,1,2048] steps
    prefill = proj_inputs[0][0]                      # [L0, 2048]
    steps = np.concatenate([p[0] for p in proj_inputs[1:]], axis=0) \
        if len(proj_inputs) > 1 else np.zeros((0, prefill.shape[1]), np.float32)
    L0 = prefill.shape[0]
    print(f"prefill embeds: {prefill.shape}; step embeds: {steps.shape}; "
          f"L0={L0} (expect prompt_len+2={prompt_len + 2})", flush=True)

    np.save(os.path.join(args.outdir, "talker_prefill_embeds.npy"),
            prefill.astype(np.float32))
    np.save(os.path.join(args.outdir, "talker_step_embeds.npy"),
            steps.astype(np.float32))
    np.save(os.path.join(args.outdir, "talker_codes.npy"), codes)
    if "thinker_reply_part" in reply_part_holder:
        np.save(os.path.join(args.outdir, "thinker_reply_part.npy"),
                reply_part_holder["thinker_reply_part"][0].astype(np.float32))
    np.save(os.path.join(args.outdir, "prompt_ids.npy"),
            np.array(input_ids[0].tolist(), np.int64))
    np.save(os.path.join(args.outdir, "reply_ids.npy"),
            np.array(reply_ids, np.int64))

    meta = {
        "prompt_len": prompt_len,
        "L0": int(L0),
        "num_steps": int(steps.shape[0]),
        "num_codes": int(len(codes)),
        "speaker_bos": int(model.speaker_map[args.speaker]["bos_token"]),
        "codec_bos": int(talker.codec_bos_token),
        "codec_eos": int(talker.codec_eos_token),
        "codec_pad": int(talker.codec_pad_token),
        "codec_mask": int(talker.codec_mask_token),
        "embedding_size": int(prefill.shape[1]),
        "prompt": args.prompt,
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("dumped ground truth to", args.outdir, "->", meta, flush=True)


if __name__ == "__main__":
    main()

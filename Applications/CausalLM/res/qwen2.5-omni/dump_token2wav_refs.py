# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file dump_token2wav_refs.py
# @brief Dump HF Qwen2.5-Omni Token2Wav ground truth for nntrainer Phase 2.
#
#   Stage C (BigVGAN): feeds the dumped `mel` into nntrainer BigVGAN and compares
#     per-stage tensors (conv_pre, each of 6 upsample stages, activation_post,
#     wav) against HF.
#   Stage A/B (DiT, later): initial_state + per-step DiT I/O + final mel.
#
# Runs the real HF Token2Wav (CPU fp32) on the Phase-1 talker codes + Chelsie
# speaker conditioning. Greedy/deterministic via a fixed seed; for Stage C the
# nntrainer side consumes the dumped `mel`, so the noise need not be re-matched.
#
#   python dump_token2wav_refs.py --talker_dump /tmp/omni_talker_dump \
#       --outdir /tmp/omni_t2w_dump --model_path <local snapshot>

import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--talker_dump", default="/tmp/omni_talker_dump")
    ap.add_argument("--speaker", default="Chelsie")
    ap.add_argument("--outdir", default="/tmp/omni_t2w_dump")
    ap.add_argument("--max_codes", type=int, default=64,
                    help="truncate codes to keep the dump fast")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(0)

    from weight_converter import resolve_model_dir
    model_dir = resolve_model_dir(args.model_path)

    # spk_dict.pt is a torch pickle; transformers blocks torch.load on <2.6.
    import transformers.utils.import_utils as _iu
    _iu.check_torch_load_is_safe = lambda *a, **k: None
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _m
    if hasattr(_m, "check_torch_load_is_safe"):
        _m.check_torch_load_is_safe = lambda *a, **k: None

    from transformers import Qwen2_5OmniForConditionalGeneration
    print("loading model (CPU fp32) ...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float32).eval()
    t2w = model.token2wav
    bigvgan = t2w.code2wav_bigvgan_model
    spk = model.speaker_map[args.speaker]
    cond = spk["cond"].to(torch.float32)            # [1, 192]
    ref_mel = spk["ref_mel"].to(torch.float32)      # [1, 400, 80]

    codes = np.load(os.path.join(args.talker_dump, "talker_codes.npy")).reshape(-1)
    codes = codes[:args.max_codes].astype(np.int64)
    code_t = torch.tensor(codes, dtype=torch.long).unsqueeze(0)
    print(f"codes: {codes.shape} -> mel T = {len(codes)} * repeats", flush=True)

    cap = {}

    # ---- instrumented BigVGAN forward capturing per-stage tensors ----
    def instrumented_bigvgan(mel_spectrogram):
        cap["mel"] = mel_spectrogram.detach().float().cpu().numpy()
        x = bigvgan.process_mel_spectrogram(mel_spectrogram)
        cap["processed_mel"] = x.detach().float().cpu().numpy()
        x = bigvgan.conv_pre(x)
        cap["conv_pre"] = x.detach().float().cpu().numpy()
        for i in range(bigvgan.num_upsample_layers):
            x = bigvgan.ups[i][0](x)
            cap[f"ups{i}"] = x.detach().float().cpu().numpy()
            res = sum(
                bigvgan.resblocks[i * bigvgan.num_residual_blocks + b](x)
                for b in range(bigvgan.num_residual_blocks)) / bigvgan.num_residual_blocks
            x = res
            cap[f"stage{i}"] = x.detach().float().cpu().numpy()
        x = bigvgan.activation_post(x)
        cap["activation_post"] = x.detach().float().cpu().numpy()
        x = bigvgan.conv_post(x)
        wav = torch.clamp(x, -1.0, 1.0).squeeze().cpu()
        cap["wav"] = wav.detach().float().numpy()
        return wav

    # ---- capture DiT sampler initial_state + final mel (for Stage A/B) ----
    dit = t2w.code2wav_dit_model
    orig_randn = torch.randn

    def patched_randn(*a, **k):
        out = orig_randn(*a, **k)
        if len(a) and isinstance(a[0], (list, tuple)) and len(a[0]) == 3 \
           and a[0][0] == 1 and a[0][2] == 80:
            cap["initial_state_full"] = out.detach().float().cpu().numpy()
        return out

    torch.randn = patched_randn
    with torch.no_grad():
        mel = dit.sample(cond, ref_mel, code_t, num_steps=10,
                         guidance_scale=0.5, sway_coefficient=-1.0)
    torch.randn = orig_randn
    cap["dit_mel"] = mel.detach().float().cpu().numpy()  # [1, 80, T]
    print(f"dit mel: {cap['dit_mel'].shape}", flush=True)

    with torch.no_grad():
        wav = instrumented_bigvgan(mel)
    print(f"wav: {cap['wav'].shape} range [{cap['wav'].min():.3f},"
          f"{cap['wav'].max():.3f}]", flush=True)

    for k, v in cap.items():
        np.save(os.path.join(args.outdir, f"{k}.npy"), np.ascontiguousarray(v))
    np.save(os.path.join(args.outdir, "codes.npy"), codes)
    np.save(os.path.join(args.outdir, "cond.npy"), cond.numpy())
    np.save(os.path.join(args.outdir, "ref_mel.npy"), ref_mel.numpy())

    bcfg = json.loads(model.config.token2wav_config.bigvgan_config.to_json_string()) \
        if hasattr(model.config.token2wav_config, "bigvgan_config") else {}
    meta = {
        "num_codes": int(len(codes)),
        "mel_T": int(cap["dit_mel"].shape[-1]),
        "mel_dim": int(cap["dit_mel"].shape[1]),
        "wav_len": int(cap["wav"].shape[0]),
        "upsample_rates": bcfg.get("upsample_rates", [5, 3, 2, 2, 2, 2]),
        "upsample_kernel_sizes": bcfg.get("upsample_kernel_sizes",
                                          [11, 7, 4, 4, 4, 4]),
        "speaker": args.speaker,
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("dumped token2wav refs to", args.outdir, "->", meta, flush=True)
    print("stages:", sorted(cap.keys()), flush=True)


if __name__ == "__main__":
    main()

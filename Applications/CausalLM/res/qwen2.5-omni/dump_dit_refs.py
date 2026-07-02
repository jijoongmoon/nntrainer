# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file dump_dit_refs.py
# @brief Dump HF Qwen2.5-Omni Token2Wav DiT Stage-A references (one ODE step +
#        intermediates) so the nntrainer per-step DiT graph can be validated.
#
# The existing /tmp/omni_t2w_dump has only the final dit_mel; the DiT per-step
# graph needs intermediate ground truth. This runs ONE dit.forward(apply_cfg)
# at a fixed (x=initial_state[:, :T], t) reusing the dumped sample() inputs,
# hooking submodules to capture: time_emb, code_embed (cond/uncond),
# ECAPA(ref_mel)/ECAPA(0), input_embed out, rotary cos/sin, block0 in/out,
# norm_out out, proj_out (per-row velocity), and the post-CFG velocity.
#
#   python dump_dit_refs.py --model_path <snapshot> --t2w_dump /tmp/omni_t2w_dump \
#       --outdir /tmp/omni_dit_dump [--step 1]

import argparse
import math
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--t2w_dump", default="/tmp/omni_t2w_dump")
    ap.add_argument("--outdir", default="/tmp/omni_dit_dump")
    ap.add_argument("--step", type=int, default=1,
                    help="which sway time index to evaluate (0..9)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(0)

    from weight_converter import resolve_model_dir
    model_dir = resolve_model_dir(args.model_path)

    import transformers.utils.import_utils as _iu
    _iu.check_torch_load_is_safe = lambda *a, **k: None
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _m
    if hasattr(_m, "check_torch_load_is_safe"):
        _m.check_torch_load_is_safe = lambda *a, **k: None

    from transformers import Qwen2_5OmniForConditionalGeneration
    print("loading model (CPU fp32) ...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float32).eval()
    dit = model.token2wav.code2wav_dit_model

    # reuse the exact sample() inputs from the Token2Wav dump
    d = args.t2w_dump
    cond = torch.tensor(np.load(os.path.join(d, "cond.npy")), dtype=torch.float32)        # [1,192]
    ref_mel = torch.tensor(np.load(os.path.join(d, "ref_mel.npy")), dtype=torch.float32)  # [1,400,80]
    codes = torch.tensor(np.load(os.path.join(d, "codes.npy")).reshape(1, -1), dtype=torch.long)  # [1,64]
    init = torch.tensor(np.load(os.path.join(d, "initial_state_full.npy")), dtype=torch.float32)  # [1,30000,80]

    repeats = dit.repeats
    T = codes.shape[1] * repeats
    x = init[:, :T].contiguous()                       # [1,T,80]
    speaker = cond.unsqueeze(1).repeat(1, T, 1)         # [1,T,192]

    # sway schedule (num_steps=10, sway=-1.0)
    t = torch.linspace(0, 1, 10, dtype=torch.float32)
    t = t + (-1.0) * (torch.cos(math.pi / 2 * t) - 1 + t)
    ts = t[args.step].reshape(())
    print(f"T={T}, step={args.step}, t={float(ts):.6f}", flush=True)

    cap = {}

    def save(name, x):
        cap[name] = x.detach().float().cpu().numpy()

    hooks = []

    def hook(name):
        def fn(mod, inp, out):
            if isinstance(out, (tuple, list)):
                for i, o in enumerate(out):
                    if torch.is_tensor(o):
                        save(f"{name}_{i}", o)
            else:
                save(name, out)
        return fn

    hooks.append(dit.time_embed.register_forward_hook(hook("time_emb")))
    hooks.append(dit.input_embed.register_forward_hook(hook("input_embed_out")))
    hooks.append(dit.input_embed.spk_encoder.register_forward_hook(hook("ecapa_out")))
    hooks.append(dit.rotary_embed.register_forward_hook(hook("rotary")))
    hooks.append(dit.transformer_blocks[0].register_forward_hook(hook("block0_out")))
    hooks.append(dit.norm_out.register_forward_hook(hook("norm_out")))
    hooks.append(dit.proj_out.register_forward_hook(hook("proj_out")))

    # capture code_embed (cond + uncond) directly
    with torch.no_grad():
        save("code_embed", dit.text_embed(codes, drop_code=False))
        save("code_embed_uncond", dit.text_embed(codes, drop_code=True))

    with torch.no_grad():
        out = dit(hidden_states=x, condition_vector=ref_mel,
                  speaker_embedding=speaker, quantized_code=codes,
                  time_step=ts, apply_cfg=True)        # [2,T,80]
    guided, null = torch.chunk(out, 2, dim=0)
    velocity = guided + (guided - null) * 0.5
    save("dit_out_batched", out)
    save("velocity_cfg", velocity)

    for h in hooks:
        h.remove()

    # also persist the exact inputs the nntrainer Stage-A spike must feed
    save("x_in", x)
    save("ref_mel", ref_mel)
    save("cond192", cond)
    save("codes", codes.float())
    cap["t_value"] = np.array([float(ts)], dtype=np.float32)

    for k, v in cap.items():
        np.save(os.path.join(args.outdir, f"{k}.npy"), np.ascontiguousarray(v))
    meta = {k: list(np.asarray(v).shape) for k, v in cap.items()}
    print("dumped DiT Stage-A refs to", args.outdir, flush=True)
    for k in sorted(meta): print(f"  {k}: {meta[k]}")


if __name__ == "__main__":
    main()

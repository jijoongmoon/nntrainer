#!/usr/bin/env python3
"""Quantify the group-128 int4 -> per-channel int4 requantization loss of the
GaussA repack (convert_qwen3_5_moe_gaussa.py) on sampled expert matrices.

For each sampled packed expert mat:
  W_g  = GaussA group-128 dequant (what HF/vLLM computes with)
  W_pc = per-channel absmax/7 requant of W_g, dequantized with the fp16 scale
         (exactly what nntrainer will compute with)
Reports relative Frobenius error ||W_pc - W_g||_F / ||W_g||_F and max |delta|
relative to the channel absmax. Layer 0 (bf16 experts) is also sampled to show
the bf16 -> int4 first-quantization loss there.
"""
import sys
import importlib.util
import numpy as np

GAUSSA = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/aisjetson/workspace/models/GaussA-Qwen3.6-35B-A3B-v0.3-INT"
HERE = "/home/aisjetson/jijoongmoon/nntrainer/Applications/CausalLM/models/qwen3_5_moe"
spec = importlib.util.spec_from_file_location(
    "conv", HERE + "/convert_qwen3_5_moe_gaussa.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

src = conv.Source(GAUSSA)


def per_channel_roundtrip(w):
    absmax = np.abs(w).max(axis=1).astype(np.float32)
    scale = (absmax / np.float32(7.0)).astype(np.float32)
    scale[scale == 0.0] = 1.0
    q = np.clip(np.rint(w / scale[:, None]), -8, 7).astype(np.float32)
    scale16 = np.float16(scale).astype(np.float32)  # runtime dequant scale
    return q * scale16[:, None]


def report(tag, w):
    wpc = per_channel_roundtrip(w)
    d = wpc - w
    rel_f = np.linalg.norm(d) / max(np.linalg.norm(w), 1e-30)
    ch_absmax = np.abs(w).max(axis=1, keepdims=True)
    rel_max = np.abs(d / np.maximum(ch_absmax, 1e-30)).max()
    print(f"{tag:55s} relF={rel_f:.4f} max|d|/absmax={rel_max:.4f}", flush=True)
    return rel_f


rels = []
P = "model.language_model.layers.{}.mlp.experts.{}.{}_proj"
for li in (1, 5, 20, 39):
    for e in (0, 77, 200):
        for proj in ("gate", "up", "down"):
            base = P.format(li, e, proj)
            rels.append(report(f"L{li} E{e} {proj} (packed g128->pc)",
                               src.packed_linear(base)))
print(f"== packed experts: mean relF = {np.mean(rels):.4f} "
      f"max relF = {np.max(rels):.4f}", flush=True)

rels0 = []
for e in (0, 77, 200):
    for proj in ("gate", "up", "down"):
        base = P.format(0, e, proj)
        rels0.append(report(f"L0 E{e} {proj} (bf16->pc int4)",
                            src.fp32(base + ".weight")))
print(f"== layer-0 experts: mean relF = {np.mean(rels0):.4f} "
      f"max relF = {np.max(rels0):.4f}", flush=True)

# attention + shared (bf16 -> per-channel int4, first quantization)
rels_a = []
A = "model.language_model.layers.3.self_attn.{}_proj.weight"
for p in ("q", "k", "v", "o"):
    rels_a.append(report(f"L3 attn {p}_proj (bf16->pc int4)", src.fp32(A.format(p))))
S = "model.language_model.layers.5.mlp.shared_expert.{}_proj.weight"
for p in ("gate", "up", "down"):
    rels_a.append(report(f"L5 shared {p}_proj (bf16->pc int4)", src.fp32(S.format(p))))
print(f"== attn/shared: mean relF = {np.mean(rels_a):.4f} "
      f"max relF = {np.max(rels_a):.4f}", flush=True)

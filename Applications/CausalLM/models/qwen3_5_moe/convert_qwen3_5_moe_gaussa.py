#!/usr/bin/env python3
"""Streaming GaussA-Qwen3.6-35B-A3B-v0.3-INT -> nntrainer weights.bin repacker.

Reads the weight manifest dumped by gdn_dump_manifest (the positional load
order + dtype + dims the real Qwen3_5MoeCausalLM graph requests) and emits the
bin tensor-by-tensor from the GaussA safetensors checkpoint (~1-tensor RAM).

Transforms (validated bit-exact on the tiny model, convert_tiny_gdn.py):
  - FC weights: HF [out,in]; FP16/FP32 records store nntrainer [in,out]
    (transpose); QINT4 plain records store [out,in] rows directly.
  - (1+w) RMSNorm bake: +1 on input_layernorm / post_attention_layernorm /
    final norm / q_norm / k_norm gammas (NOT the GDN internal gated norm).
  - Full-attn q_proj [.,nH,2*hd] de-interleaved into wq | w_gate.
  - Partial-RoPE q/k(+norm) head_dim row permutation (rope_inv_perm) so
    nntrainer's split-half RoPE pairing reproduces HF's contiguous pairing.
  - conv1d [conv,1,K] -> flat [conv,K].

QINT4 records use the PER_CHANNEL_AFFINE "plain" container (qscheme 0x0001):
  uint16 0x0001 | N rows x ceil(K/2) bytes row-major nibbles (low nibble =
  even k, uint4 = int4+8) | fp32 scales (fp16 precision) at byte N*(K+1)/2 |
  zero pad to ceil(N/8)*8 * (roundup(K,32)/2 + 12).  Per-output-channel
  symmetric scale = absmax/7 (nntrainer Int4Utils default); the loader
  repacks to the in-memory KAI Section-A form at read time.

GaussA packed experts (layers 1..39) are group-128 symmetric int4
(compressed-tensors pack-quantized, low-nibble-first, offset +8): unpack ->
dequant -> requant per-channel (lossy group->per-channel; measure logit KL).

Usage:
  convert_qwen3_5_moe_gaussa.py <gaussa_dir> <model_dir> [--manifest M] [--out B]
  (defaults: manifest = <model_dir>/weights_manifest.txt,
             out = <model_dir>/<model_file_name from nntr_config.json>)
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

# ---------------------------------------------------------------- dims (35B)
HID = 2048
N_HEADS, N_KV, HEAD_DIM = 16, 2, 256
ROTARY_DIM = HEAD_DIM // 4  # partial_rotary_factor 0.25
GROUP_SIZE = 128
NUM_BITS = 4

HFP = "model.language_model."  # HF text-stack prefix


# ------------------------------------------------------------- HF tensor I/O
class Source:
    def __init__(self, gaussa_dir):
        from safetensors import safe_open  # lazy: emitters usable without torch

        self.f = safe_open(os.path.join(gaussa_dir, "model.safetensors"), "pt")
        self.keys = set(self.f.keys())

    def fp32(self, name):
        """bf16 (or any float) tensor as float32 numpy."""
        return self.f.get_tensor(name).float().numpy()

    def has(self, name):
        return name in self.keys

    def packed_linear(self, base):
        """Unpack a compressed-tensors pack-quantized Linear -> fp32 [out,in].
        Recipe verified bit-identical to compressed_tensors unpack_from_int32."""
        packed = self.f.get_tensor(base + ".weight_packed").numpy()  # int32 [out, in/8]
        scale = self.f.get_tensor(base + ".weight_scale").float().numpy()  # [out, in/128]
        out_f, in_f = [int(x) for x in self.f.get_tensor(base + ".weight_shape")]
        pf = 32 // NUM_BITS
        u = packed.view(np.uint32)
        rows, cols = u.shape
        unpacked = np.empty((rows, cols * pf), dtype=np.uint8)
        for i in range(pf):  # LOW-nibble-first: element i at bit 4*i
            unpacked[:, i::pf] = ((u >> (NUM_BITS * i)) & 0xF).astype(np.uint8)
        q = unpacked[:, :in_f].astype(np.int16) - (1 << (NUM_BITS - 1))  # -> [-8,7]
        ng = in_f // GROUP_SIZE
        w = (
            q.reshape(out_f, ng, GROUP_SIZE).astype(np.float32)
            * scale[:, :, None]
        ).reshape(out_f, in_f)
        return w


# ------------------------------------------------------------- transforms
def rope_inv_perm(hd, rd):
    """inv[nntr_dim] = hf_dim (see convert_tiny_gdn.py; bit-exact validated)."""
    h_rd, h_hd = rd // 2, hd // 2
    P = [0] * hd  # P[hf] = nntr
    for k in range(h_rd):
        P[k] = k
        P[h_rd + k] = h_hd + k
    nntr_pass = list(range(h_rd, h_hd)) + list(range(h_hd + h_rd, hd))
    for i, hf in enumerate(range(rd, hd)):
        P[hf] = nntr_pass[i]
    inv = [0] * hd
    for hf, nt in enumerate(P):
        inv[nt] = hf
    return inv


ROPE_INV = rope_inv_perm(HEAD_DIM, ROTARY_DIM)


def perm_rows(w, nheads):
    """permute head_dim rows within each head of a [nheads*hd, in] matrix."""
    return w.reshape(nheads, HEAD_DIM, -1)[:, ROPE_INV, :].reshape(
        nheads * HEAD_DIM, -1
    )


# ------------------------------------------------------------- emitters
def emit_fp32(out, w, dims):
    a = np.ascontiguousarray(w, dtype="<f4")
    assert a.size == int(np.prod(dims)), f"fp32 size {a.shape} vs dims {dims}"
    out.write(a.tobytes())
    return a.nbytes


def emit_fp16(out, w, dims):
    a = np.ascontiguousarray(w, dtype=np.float32).astype("<f2")
    assert a.size == int(np.prod(dims)), f"fp16 size {a.shape} vs dims {dims}"
    out.write(a.tobytes())
    return a.nbytes


def plain_record_payload_bytes(n, k):
    # Int4Utils::plainRecordPayloadBytes: ceil(N/8)*8 * (roundup(K,32)/2 + 12)
    n8 = (n + 7) // 8 * 8
    k32 = (k + 31) // 32 * 32
    return n8 * (k32 // 2 + 12)


def emit_qint4(out, w_out_in, dims):
    """w_out_in: fp32 [out=N, in=K] (HF orientation). dims: manifest (b,c,h,w)
    with h=K(in), w=N(out). Emits the PER_CHANNEL_AFFINE plain container."""
    N, K = w_out_in.shape
    assert dims[2] == K and dims[3] == N, f"qint4 {w_out_in.shape} vs dims {dims}"
    assert N % 4 == 0 and K % 32 == 0, f"qint4 constraint violated N={N} K={K}"
    absmax = np.abs(w_out_in).max(axis=1).astype(np.float32)
    scale = (absmax / np.float32(7.0)).astype(np.float32)
    scale[scale == 0.0] = 1.0
    # nntrainer quantizes with the FP32 scale (Int4Utils::quantizeToInt4);
    # only the STORED scale is fp16-rounded (compute_fp32_to_fp16).
    q = np.clip(np.rint(w_out_in / scale[:, None]), -8, 7).astype(np.int8)
    scale = np.float16(scale).astype(np.float32)  # fp16 precision, fp32 slot
    u = (q + 8).astype(np.uint8)  # offset-binary uint4
    nib = (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)  # low nibble = even k
    payload = np.zeros(plain_record_payload_bytes(N, K), dtype=np.uint8)
    nb = nib.reshape(-1)
    payload[: nb.size] = nb
    soff = N * (K + 1) // 2
    payload[soff : soff + 4 * N] = np.frombuffer(
        scale.astype("<f4").tobytes(), dtype=np.uint8
    )
    out.write(np.uint16(0x0001).tobytes())
    out.write(payload.tobytes())
    return 2 + payload.nbytes


# ------------------------------------------------------------- HF resolvers
def gamma_plus1(src, name):
    return src.fp32(name) + 1.0


def resolve(src, lname, wname, layer_types):
    """Map an nntrainer weight name -> fp32 numpy array in HF [out,in] (FC) /
    natural (vector) orientation, transforms applied. Returns (array, kind)
    where kind is 'fc' (needs [in,out] transpose for FP16/FP32 records) or
    'raw' (emit as-is)."""
    if lname == "embedding0":
        return src.fp32(HFP + "embed_tokens.weight"), "raw"  # [vocab, hid]
    if lname == "output_norm":
        return gamma_plus1(src, HFP + "norm.weight"), "raw"
    if lname == "output_of_causallm":
        return src.fp32("lm_head.weight"), "fc"  # [vocab, hid] -> T

    m = re.match(r"layer(\d+)_(.+)", lname)
    if not m:
        raise KeyError(f"unmapped layer name: {lname}")
    li, rest = int(m.group(1)), m.group(2)
    L = f"{HFP}layers.{li}."
    is_gdn = layer_types[li] == "linear_attention"

    if rest == "attention_norm":
        return gamma_plus1(src, L + "input_layernorm.weight"), "raw"
    if rest == "ffn_norm":
        return gamma_plus1(src, L + "post_attention_layernorm.weight"), "raw"

    if rest == "attention" and is_gdn:  # GDN mixer internal weights
        g = L + "linear_attn."
        if wname == "in_proj_qkv":
            return src.fp32(g + "in_proj_qkv.weight"), "fc"
        if wname == "in_proj_z":
            return src.fp32(g + "in_proj_z.weight"), "fc"
        if wname == "in_proj_b":
            return src.fp32(g + "in_proj_b.weight"), "fc"
        if wname == "in_proj_a":
            return src.fp32(g + "in_proj_a.weight"), "fc"
        if wname == "conv1d":
            return src.fp32(g + "conv1d.weight").reshape(-1, 4), "raw"  # [conv,K]
        if wname == "A_log":
            return src.fp32(g + "A_log"), "raw"
        if wname == "dt_bias":
            return src.fp32(g + "dt_bias"), "raw"
        if wname == "norm":
            return src.fp32(g + "norm.weight"), "raw"  # gated norm: plain w
        if wname == "out_proj":
            return src.fp32(g + "out_proj.weight"), "fc"
        raise KeyError(f"unmapped GDN weight: {wname}")

    a = L + "self_attn."
    if rest == "wq":
        qp = src.fp32(a + "q_proj.weight").reshape(N_HEADS, 2 * HEAD_DIM, HID)
        return perm_rows(qp[:, :HEAD_DIM, :].reshape(N_HEADS * HEAD_DIM, HID),
                         N_HEADS), "fc"
    if rest == "w_gate":
        qp = src.fp32(a + "q_proj.weight").reshape(N_HEADS, 2 * HEAD_DIM, HID)
        return qp[:, HEAD_DIM:, :].reshape(N_HEADS * HEAD_DIM, HID), "fc"
    if rest == "q_norm":
        return gamma_plus1(src, a + "q_norm.weight")[ROPE_INV], "raw"
    if rest == "wk":
        return perm_rows(src.fp32(a + "k_proj.weight"), N_KV), "fc"
    if rest == "k_norm":
        return gamma_plus1(src, a + "k_norm.weight")[ROPE_INV], "raw"
    if rest == "wv":
        return src.fp32(a + "v_proj.weight"), "fc"
    if rest == "attention_out":
        return src.fp32(a + "o_proj.weight"), "fc"

    mlp = L + "mlp."
    if rest == "ffn_down":  # qwen_moe layer: router + experts
        if wname == "gate":
            return src.fp32(mlp + "gate.weight"), "fc"  # router [E, hid]
        em = re.match(r"expert_(up|gate|down)_(\d+)", wname)
        if em:
            proj, e = em.group(1), int(em.group(2))
            base = f"{mlp}experts.{e}.{proj}_proj"
            if src.has(base + ".weight_packed"):
                return src.packed_linear(base), "fc"
            return src.fp32(base + ".weight"), "fc"  # layer 0: bf16 experts
        raise KeyError(f"unmapped MoE weight: {wname}")
    if rest == "shared_gate":
        return src.fp32(mlp + "shared_expert.gate_proj.weight"), "fc"
    if rest == "shared_up":
        return src.fp32(mlp + "shared_expert.up_proj.weight"), "fc"
    if rest == "shared_down":
        return src.fp32(mlp + "shared_expert.down_proj.weight"), "fc"
    if rest == "shared_gate_lin":
        return src.fp32(mlp + "shared_expert_gate.weight"), "fc"  # [1, hid]

    raise KeyError(f"unmapped: {lname}:{wname}")


# ------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gaussa_dir")
    ap.add_argument("model_dir")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    manifest = args.manifest or os.path.join(args.model_dir, "weights_manifest.txt")
    nntr_cfg = json.load(open(os.path.join(args.model_dir, "nntr_config.json")))
    out_path = args.out or os.path.join(args.model_dir, nntr_cfg["model_file_name"])
    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    layer_types = cfg["layer_types"]

    entries = []
    for line in open(manifest):
        if not line.startswith("W|"):
            continue
        _, idx, name, dtype, b, c, h, w = line.rstrip("\n").split("|")
        entries.append((int(idx), name, dtype, (int(b), int(c), int(h), int(w))))
    assert entries and [e[0] for e in entries] == list(range(len(entries)))
    print(f"[repack] {len(entries)} weights -> {out_path}", flush=True)

    src = Source(args.gaussa_dir)
    t0 = time.time()
    total = 0
    with open(out_path, "wb") as out:
        for idx, name, dtype, dims in entries:
            lname, wname = name.split(":", 1)
            arr, kind = resolve(src, lname, wname, layer_types)
            if dtype == "QINT4":
                assert kind == "fc", f"{name}: QINT4 must be an FC matrix"
                total += emit_qint4(out, arr, dims)
            else:
                if kind == "fc" and arr.ndim == 2:
                    arr = np.ascontiguousarray(arr.T)  # [out,in] -> [in,out]
                exp = (dims[2] * dims[3]) if arr.ndim == 2 else arr.size
                assert arr.size == int(np.prod(dims)), (
                    f"{name}: size {arr.shape} vs dims {dims}")
                if arr.ndim == 2:
                    assert arr.shape == (dims[2], dims[3]), (
                        f"{name}: shape {arr.shape} vs dims {dims}")
                total += (emit_fp16 if dtype == "FP16" else emit_fp32)(
                    out, arr, dims)
                if dtype not in ("FP16", "FP32"):
                    raise ValueError(f"{name}: unsupported dtype {dtype}")
            if idx % 500 == 0 or idx == len(entries) - 1:
                print(f"  [{idx}/{len(entries)}] {name} {dtype} "
                      f"{total/2**30:.2f}GiB {time.time()-t0:.0f}s", flush=True)
    print(f"[repack] DONE {total} bytes ({total/2**30:.2f} GiB) "
          f"in {time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

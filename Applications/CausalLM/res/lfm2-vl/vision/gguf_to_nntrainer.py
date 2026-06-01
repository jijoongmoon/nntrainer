#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file  gguf_to_nntrainer.py
# @brief Convert a GGUF vision-tower (LFM2.5-VL / SigLIP2-86M style) into
#        either:
#          (a) the layer-ordered raw .bin format the nntrainer
#              ClipVitTransformer consumes via Model::load(MODEL_FORMAT_BIN),
#          (b) a HuggingFace .safetensors file with nntrainer-style names, or
#          (c) both, side-by-side.
#        Selected via `--format {bin,safetensors,both}`.
#
# The nntrainer .bin format is not self-describing: it is a sequence of raw
# tensor bytes laid out in the exact order the layers are constructed (and
# within a layer, in the order each weight was requested). This script knows
# the order ClipVitTransformer wires up — see
# Applications/CausalLM/models/clip_vit/clip_vit_transformer.cpp — and writes
# tensors out in that exact sequence.
#
# The .safetensors output stores the same tensors as named entries:
#   "v_patch_embd.filter", "v_patch_embd.bias",
#   "v_blk_{i}_ln1.gamma", "v_blk_{i}_ln1.beta",
#   "v_blk_{i}_attn_q.weight", "v_blk_{i}_attn_q.bias",
#   ... and so on, mirroring the C++ layer names assigned in
#   clip_vit_transformer.cpp. All tensors are stored as FP32. Useful as an
#   interchange format and for diffing converters; nntrainer's runtime does
#   not yet have a safetensors loader.
#
# Expected GGUF tensor names (HuggingFace convert_clip.py / mmproj convention):
#
#     v.patch_embd.weight                          # [DIM, C, P, P]   F16/F32
#     v.patch_embd.bias                            # [DIM]            F32
#     v.blk.{i}.ln1.weight                         # [DIM]            F32
#     v.blk.{i}.ln1.bias                           # [DIM]            F32
#     v.blk.{i}.attn_q.weight                      # [DIM, DIM]       F16/F32
#     v.blk.{i}.attn_q.bias                        # [DIM]            F32
#     v.blk.{i}.attn_k.weight                      # [DIM, DIM]       F16/F32
#     v.blk.{i}.attn_k.bias                        # [DIM]            F32
#     v.blk.{i}.attn_v.weight                      # [DIM, DIM]       F16/F32
#     v.blk.{i}.attn_v.bias                        # [DIM]            F32
#     v.blk.{i}.attn_out.weight                    # [DIM, DIM]       F16/F32
#     v.blk.{i}.attn_out.bias                      # [DIM]            F32
#     v.blk.{i}.ln2.weight                         # [DIM]            F32
#     v.blk.{i}.ln2.bias                           # [DIM]            F32
#     v.blk.{i}.ffn_up.weight                      # [HID, DIM]       F16/F32
#     v.blk.{i}.ffn_up.bias                        # [HID]            F32
#     v.blk.{i}.ffn_down.weight                    # [DIM, HID]       F16/F32
#     v.blk.{i}.ffn_down.bias                      # [DIM]            F32
#     v.post_ln.weight                             # [DIM]            F32
#     v.post_ln.bias                               # [DIM]            F32
#
# Some GGUFs use slightly different tensor name conventions; the script
# accepts both common variants:
#
#     ffn_up   <-> mlp.fc1   /  ffn_fc1
#     ffn_down <-> mlp.fc2   /  ffn_fc2
#     ln1      <-> norm1     /  attn_norm
#     ln2      <-> norm2     /  ffn_norm
#     post_ln  <-> output_norm / pre_proj_norm
#
# This converter targets the ClipVitTransformer parameter layout. The current
# implementation does NOT yet emit a learnable position embedding tensor
# because the matching graph wiring is still TODO on the C++ side (see
# clip_vit_transformer.cpp). When that is wired up, add a write_norm call for
# v.position_embd.weight right after the patch_embd block here.
#
# Weight orientation note:
#
#     - Conv2D filter: GGUF stores [filter_out, in_channel, kH, kW] which is
#       the same layout nntrainer's conv2d_layer requests, so the bytes are
#       passed through verbatim (only dtype-cast to FP32).
#     - FullyConnected: GGUF stores [out, in] (PyTorch nn.Linear convention)
#       but nntrainer's fc_layer requests weight as [1, 1, in, out], so
#       every FC weight is transposed before being written.
#     - LayerNorm gamma/beta: 1-D, no reshape needed.
#
# Quantised GGUF tensors (Q4_0 / Q8_0 / Q6_K) are dequantised to FP32 on read.
# This keeps the script small; vision towers at this scale (~86M params) are
# almost always shipped F16 anyway. Other dtypes will raise a clear error.

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# GGUF constants
# ---------------------------------------------------------------------------
GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

GGUF_U8, GGUF_I8 = 0, 1
GGUF_U16, GGUF_I16 = 2, 3
GGUF_U32, GGUF_I32 = 4, 5
GGUF_F32 = 6
GGUF_BOOL = 7
GGUF_STR = 8
GGUF_ARR = 9
GGUF_U64, GGUF_I64 = 10, 11
GGUF_F64 = 12

GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q8_0 = 8
GGML_Q6_K = 14
GGML_BF16 = 30  # llama.cpp >= b3000

GGML_TYPE_NAMES = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_Q4_0: "Q4_0",
    GGML_Q4_1: "Q4_1",
    GGML_Q8_0: "Q8_0",
    GGML_Q6_K: "Q6_K",
    GGML_BF16: "BF16",
}

QK4_0 = 32
QK8_0 = 32
QK_K = 256
Q4_0_BLOCK_BYTES = 2 + QK4_0 // 2          # fp16 scale + 16 packed bytes
Q4_1_BLOCK_BYTES = 2 + 2 + QK4_0 // 2      # fp16 scale + fp16 min + 16 packed
Q8_0_BLOCK_BYTES = 2 + QK8_0               # fp16 scale + 32 int8
Q6_K_BLOCK_BYTES = QK_K // 2 + QK_K // 4 + QK_K // 16 + 2  # ql + qh + scales + d


# ---------------------------------------------------------------------------
# GGUF reader
# ---------------------------------------------------------------------------
class GGUFReader:
    """Minimal GGUF v2 / v3 reader.

    Loads the full kv-store and tensor info table eagerly, then exposes
    read_tensor_fp32(name) to deliver any tensor as a numpy F32 array.
    """

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        self._read_header()
        self._read_metadata()
        self._read_tensor_info()

    def close(self):
        self.f.close()

    # ---- raw io ----
    def _u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def _read_string(self):
        n = self._u64()
        return self.f.read(n).decode("utf-8", errors="replace")

    def _read_value(self, typ):
        if typ == GGUF_U8:   return struct.unpack("<B", self.f.read(1))[0]
        if typ == GGUF_I8:   return struct.unpack("<b", self.f.read(1))[0]
        if typ == GGUF_U16:  return struct.unpack("<H", self.f.read(2))[0]
        if typ == GGUF_I16:  return struct.unpack("<h", self.f.read(2))[0]
        if typ == GGUF_U32:  return struct.unpack("<I", self.f.read(4))[0]
        if typ == GGUF_I32:  return struct.unpack("<i", self.f.read(4))[0]
        if typ == GGUF_F32:  return struct.unpack("<f", self.f.read(4))[0]
        if typ == GGUF_BOOL: return struct.unpack("<B", self.f.read(1))[0] != 0
        if typ == GGUF_STR:  return self._read_string()
        if typ == GGUF_ARR:
            arr_typ = self._u32()
            n = self._u64()
            return [self._read_value(arr_typ) for _ in range(n)]
        if typ == GGUF_U64:  return struct.unpack("<Q", self.f.read(8))[0]
        if typ == GGUF_I64:  return struct.unpack("<q", self.f.read(8))[0]
        if typ == GGUF_F64:  return struct.unpack("<d", self.f.read(8))[0]
        raise ValueError(f"unknown GGUF metadata type: {typ}")

    # ---- parse ----
    def _read_header(self):
        magic = self._u32()
        if magic != GGUF_MAGIC:
            raise ValueError(
                f"not a GGUF file (magic=0x{magic:08x}): {self.path}")
        self.version = self._u32()
        self.n_tensors = self._u64()
        self.n_kv = self._u64()

    def _read_metadata(self):
        self.metadata = {}
        for _ in range(self.n_kv):
            key = self._read_string()
            typ = self._u32()
            self.metadata[key] = self._read_value(typ)

    def _read_tensor_info(self):
        self.tensors = {}
        for _ in range(self.n_tensors):
            name = self._read_string()
            n_dim = self._u32()
            # GGUF stores dims innermost-first; convert to outer-first.
            dims = [self._u64() for _ in range(n_dim)]
            shape = tuple(reversed(dims))
            typ = self._u32()
            offset = self._u64()
            self.tensors[name] = {
                "shape": shape,
                "type": typ,
                "offset": offset,
            }
        alignment = self.metadata.get("general.alignment", 32)
        pos = self.f.tell()
        pad = (alignment - (pos % alignment)) % alignment
        self.data_start = pos + pad

    # ---- byte sizes ----
    def _bytes_size(self, info):
        numel = int(np.prod(info["shape"])) if info["shape"] else 0
        t = info["type"]
        if t == GGML_F32:  return numel * 4
        if t == GGML_F16:  return numel * 2
        if t == GGML_BF16: return numel * 2
        if t == GGML_Q4_0:
            assert numel % QK4_0 == 0
            return (numel // QK4_0) * Q4_0_BLOCK_BYTES
        if t == GGML_Q8_0:
            assert numel % QK8_0 == 0
            return (numel // QK8_0) * Q8_0_BLOCK_BYTES
        if t == GGML_Q6_K:
            assert numel % QK_K == 0
            return (numel // QK_K) * Q6_K_BLOCK_BYTES
        raise ValueError(
            f"unsupported GGML type {t} "
            f"({GGML_TYPE_NAMES.get(t, '?')}) in vision converter")

    def _raw(self, name):
        info = self.tensors[name]
        size = self._bytes_size(info)
        self.f.seek(self.data_start + info["offset"])
        return self.f.read(size), info

    def has(self, name):
        return name in self.tensors

    # ---- public: read any tensor as FP32 numpy in outer-first shape ----
    def read_tensor_fp32(self, name) -> np.ndarray:
        if name not in self.tensors:
            raise KeyError(f"tensor '{name}' not in GGUF")
        buf, info = self._raw(name)
        shape = info["shape"]
        t = info["type"]
        if t == GGML_F32:
            arr = np.frombuffer(buf, dtype=np.float32).copy()
        elif t == GGML_F16:
            arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        elif t == GGML_BF16:
            # bf16 = top 16 bits of fp32; widen by zero-padding the lower 16.
            u16 = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
            arr = (u16 << 16).view(np.float32).copy()
        elif t == GGML_Q8_0:
            arr = _dequant_q8_0(buf, int(np.prod(shape)))
        elif t == GGML_Q4_0:
            arr = _dequant_q4_0(buf, int(np.prod(shape)))
        elif t == GGML_Q6_K:
            arr = _dequant_q6_k(buf, int(np.prod(shape)))
        else:
            raise ValueError(
                f"tensor '{name}' has unsupported GGML type "
                f"{GGML_TYPE_NAMES.get(t, t)}")
        return arr.reshape(shape)


# ---------------------------------------------------------------------------
# Minimal dequantisation routines (copied from the qwen3 converter and pared
# down to the dtypes we are likely to see in a vision tower).
# ---------------------------------------------------------------------------
def _dequant_q8_0(buf: bytes, numel: int) -> np.ndarray:
    n_blocks = numel // QK8_0
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(n_blocks, Q8_0_BLOCK_BYTES)
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = raw[:, 2:].view(np.int8).astype(np.float32)
    return (qs * scales[:, None]).reshape(-1)


def _dequant_q4_0(buf: bytes, numel: int) -> np.ndarray:
    n_blocks = numel // QK4_0
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(n_blocks, Q4_0_BLOCK_BYTES)
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    packed = raw[:, 2:]
    # low and high nibble interleaved into 32 fp32 values per block.
    lo = (packed & 0x0F).astype(np.int8) - 8
    hi = (packed >> 4).astype(np.int8) - 8
    out = np.empty((n_blocks, QK4_0), dtype=np.float32)
    out[:, :QK4_0 // 2] = lo
    out[:, QK4_0 // 2:] = hi
    out *= scales[:, None]
    return out.reshape(-1)


def _dequant_q6_k(buf: bytes, numel: int) -> np.ndarray:
    n_blocks = numel // QK_K
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(n_blocks, Q6_K_BLOCK_BYTES)
    ql = raw[:, : QK_K // 2]            # 4-bit lower nibbles (128 bytes)
    qh = raw[:, QK_K // 2: QK_K // 2 + QK_K // 4]  # 2-bit upper (64 bytes)
    scales = raw[:, QK_K // 2 + QK_K // 4: QK_K // 2 + QK_K // 4 + QK_K // 16]
    d_bytes = raw[:, -2:].copy()
    d = d_bytes.view(np.float16).astype(np.float32).reshape(-1)

    out = np.zeros((n_blocks, QK_K), dtype=np.float32)
    for j in range(QK_K // 128):
        # 128 elements per "half" - matches llama.cpp's dequantize_row_q6_K.
        base_o = j * 128
        base_ql = j * 64
        base_qh = j * 32
        base_sc = j * 8
        for l in range(32):
            is0 = ql[:, base_ql + l] & 0x0F
            is1 = ql[:, base_ql + l + 32] & 0x0F
            is2 = (ql[:, base_ql + l] >> 4) & 0x0F
            is3 = (ql[:, base_ql + l + 32] >> 4) & 0x0F
            qh_l = qh[:, base_qh + l]
            q0 = is0 | ((qh_l & 0x03) << 4)
            q1 = is1 | (((qh_l >> 2) & 0x03) << 4)
            q2 = is2 | (((qh_l >> 4) & 0x03) << 4)
            q3 = is3 | (((qh_l >> 6) & 0x03) << 4)
            for k, q in enumerate((q0, q1, q2, q3)):
                pos = base_o + k * 32 + l
                sc = scales[:, base_sc + k].astype(np.int8).astype(np.float32)
                out[:, pos] = d * sc * (q.astype(np.float32) - 32)
    return out.reshape(-1)


# ---------------------------------------------------------------------------
# Tensor-name resolution: tolerate a handful of common naming conventions.
# ---------------------------------------------------------------------------
def _pick(reader: GGUFReader, *candidates: str) -> str:
    for n in candidates:
        if reader.has(n):
            return n
    raise KeyError(
        "none of these tensors found in GGUF: " + ", ".join(candidates))


def vision_tensor_names(i: int):
    """Return a dict of (logical_name -> [candidate names]) for block i."""
    def b(suffix):
        return [f"v.blk.{i}.{suffix}"]

    return {
        "ln1.weight":      b("ln1.weight")      + b("norm1.weight") + b("attn_norm.weight"),
        "ln1.bias":        b("ln1.bias")        + b("norm1.bias")   + b("attn_norm.bias"),
        "attn_q.weight":   b("attn_q.weight"),
        "attn_q.bias":     b("attn_q.bias"),
        "attn_k.weight":   b("attn_k.weight"),
        "attn_k.bias":     b("attn_k.bias"),
        "attn_v.weight":   b("attn_v.weight"),
        "attn_v.bias":     b("attn_v.bias"),
        "attn_out.weight": b("attn_out.weight") + b("attn_o.weight") + b("attn_output.weight"),
        "attn_out.bias":   b("attn_out.bias")   + b("attn_o.bias")   + b("attn_output.bias"),
        "ln2.weight":      b("ln2.weight")      + b("norm2.weight") + b("ffn_norm.weight"),
        "ln2.bias":        b("ln2.bias")        + b("norm2.bias")   + b("ffn_norm.bias"),
        "ffn_up.weight":   b("ffn_up.weight")   + b("mlp.fc1.weight") + b("ffn_fc1.weight"),
        "ffn_up.bias":     b("ffn_up.bias")     + b("mlp.fc1.bias")   + b("ffn_fc1.bias"),
        "ffn_down.weight": b("ffn_down.weight") + b("mlp.fc2.weight") + b("ffn_fc2.weight"),
        "ffn_down.bias":   b("ffn_down.bias")   + b("mlp.fc2.bias")   + b("ffn_fc2.bias"),
    }


def top_level_names(reader: GGUFReader):
    return {
        "patch_embd.weight": ["v.patch_embd.weight"],
        "patch_embd.bias":   ["v.patch_embd.bias"],
        "post_ln.weight":    ["v.post_ln.weight", "v.output_norm.weight", "v.pre_proj_norm.weight"],
        "post_ln.bias":      ["v.post_ln.bias",   "v.output_norm.bias",   "v.pre_proj_norm.bias"],
    }


# ---------------------------------------------------------------------------
# Tensor collection helpers
# ---------------------------------------------------------------------------
# Each helper appends one or more (name, fp32_array) entries to `tensors`.
# `name` is the nntrainer-style fully-qualified weight key:
#   "<layer_name>.<weight_role>"
# (matches the names ClipVitTransformer assigns in clip_vit_transformer.cpp).
# Writing the array contents in `tensors` order reproduces the byte stream
# nntrainer's Model::load(MODEL_FORMAT_BIN) expects; the same array list is
# also what we hand to write_safetensors().


def _check_shape(name: str, arr: np.ndarray, expected):
    if tuple(arr.shape) != tuple(expected):
        raise ValueError(
            f"{name} shape {tuple(arr.shape)} != expected {tuple(expected)}")


def _check_len(name: str, arr: np.ndarray, expected: int):
    if arr.size != expected:
        raise ValueError(f"{name} length {arr.size} != expected {expected}")


def collect_conv2d(tensors, reader: GGUFReader,
                   nntr_layer: str, name_w: str, name_b: str,
                   expected_shape):
    w = reader.read_tensor_fp32(name_w)
    _check_shape(name_w, w, expected_shape)
    tensors.append((f"{nntr_layer}.filter", w.astype(np.float32, copy=False)))

    b = reader.read_tensor_fp32(name_b)
    _check_len(name_b, b, expected_shape[0])
    tensors.append((f"{nntr_layer}.bias", b.astype(np.float32, copy=False)))


def collect_fc(tensors, reader: GGUFReader,
               nntr_layer: str, name_w: str, name_b: str,
               N_out: int, N_in: int):
    """GGUF stores nn.Linear weight as [out, in]; nntrainer FC expects
    [in, out]. Transpose on the way out."""
    w = reader.read_tensor_fp32(name_w)
    _check_shape(name_w, w, (N_out, N_in))
    w_t = np.ascontiguousarray(w.T).astype(np.float32, copy=False)
    tensors.append((f"{nntr_layer}.weight", w_t))

    b = reader.read_tensor_fp32(name_b)
    _check_len(name_b, b, N_out)
    tensors.append((f"{nntr_layer}.bias", b.astype(np.float32, copy=False)))


def collect_norm(tensors, reader: GGUFReader,
                 nntr_layer: str, name_w: str, name_b: str,
                 expected_len: int):
    g = reader.read_tensor_fp32(name_w)
    _check_len(name_w, g, expected_len)
    tensors.append((f"{nntr_layer}.gamma", g.astype(np.float32, copy=False)))

    b = reader.read_tensor_fp32(name_b)
    _check_len(name_b, b, expected_len)
    tensors.append((f"{nntr_layer}.beta", b.astype(np.float32, copy=False)))


# ---------------------------------------------------------------------------
# Serialisers
# ---------------------------------------------------------------------------
def write_bin(out_path: str, tensors):
    """nntrainer .bin: raw FP32 bytes concatenated in layer order, no header,
    no names. Consumed by Model::load(MODEL_FORMAT_BIN)."""
    with open(out_path, "wb") as out:
        for _name, arr in tensors:
            np.ascontiguousarray(arr).astype(np.float32, copy=False).tofile(out)


def write_safetensors(out_path: str, tensors):
    """HuggingFace safetensors v0.4 format:
       <u64 LE header_size> <UTF-8 JSON header> <raw bytes ...>
    The JSON header maps each tensor name to its dtype, shape and
    (start, end) byte offsets inside the data block. All tensors are
    written contiguous, FP32 little-endian."""
    # First pass: compute offsets.
    header = {}
    offset = 0
    payloads = []
    for name, arr in tensors:
        arr_f32 = np.ascontiguousarray(arr).astype(np.float32, copy=False)
        nbytes = arr_f32.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr_f32.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        payloads.append(arr_f32.tobytes())

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad the header to 8-byte alignment so the data section is aligned too.
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad

    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for payload in payloads:
            out.write(payload)


def _layer_name_for_block(i: int, suffix: str) -> str:
    """Mirror the C++ blkName(i, suffix) helper in clip_vit_transformer.cpp."""
    return f"v_blk_{i}_{suffix}"


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------
def _infer_config(md: dict, reader: GGUFReader):
    """Read the vision-tower hyperparameters from the GGUF KV store. Falls
    back to LFM2.5-VL / SigLIP2 defaults when a key is missing."""
    image_size = int(md.get("clip.vision.image_size", 256))
    patch_size = int(md.get("clip.vision.patch_size", 16))
    hidden     = int(md.get("clip.vision.embedding_length", 768))
    n_heads    = int(md.get("clip.vision.attention.head_count", 12))
    n_layers   = int(md.get("clip.vision.block_count", 12))
    ff_dim     = int(md.get("clip.vision.feed_forward_length", 3072))
    channels   = int(md.get("clip.vision.num_channels", 3))

    # Fall back to counting blocks in the tensor table if block_count is absent.
    if n_layers == 0:
        n_layers = 1 + max(
            int(name.split(".")[2])
            for name in reader.tensors
            if name.startswith("v.blk.") and name.split(".")[2].isdigit()
        )

    return dict(
        image_size=image_size,
        patch_size=patch_size,
        hidden=hidden,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        channels=channels,
    )


def gather_tensors(reader: GGUFReader, cfg: dict):
    """Walk the ClipVitTransformer layer order and collect (name, fp32_array)
    tuples. Order matches clip_vit_transformer.cpp exactly: writing these out
    sequentially as raw FP32 reproduces the byte layout
    Model::load(MODEL_FORMAT_BIN) expects."""
    tensors = []
    top = top_level_names(reader)
    expected_patch_shape = (cfg["hidden"], cfg["channels"],
                            cfg["patch_size"], cfg["patch_size"])

    # 1. patch embedding (Conv2D).
    collect_conv2d(tensors, reader, "v_patch_embd",
                   _pick(reader, *top["patch_embd.weight"]),
                   _pick(reader, *top["patch_embd.bias"]),
                   expected_patch_shape)

    # 1b. learnable position embedding [NUM_PATCHES, DIM]. Added right after
    #     the patch permute and before the encoder blocks, matching the graph
    #     weight-load order in lfm2_vl_vision_transformer.cpp.
    n_patches = (cfg["image_size"] // cfg["patch_size"]) ** 2
    pos = reader.read_tensor_fp32("v.position_embd.weight")
    _check_shape("v.position_embd.weight", pos, (n_patches, cfg["hidden"]))
    tensors.append(("v_pos_embd", pos.astype(np.float32, copy=False)))

    # 2. encoder blocks (Pre-LN order matches clip_vit_transformer.cpp:
    #    ln1, attn_q, attn_k, attn_v, attn_out, ln2, ffn_up, ffn_down).
    for i in range(cfg["n_layers"]):
        blk = vision_tensor_names(i)
        layer = lambda s: _layer_name_for_block(i, s)

        collect_norm(tensors, reader, layer("ln1"),
                     _pick(reader, *blk["ln1.weight"]),
                     _pick(reader, *blk["ln1.bias"]),
                     cfg["hidden"])
        collect_fc(tensors, reader, layer("attn_q"),
                   _pick(reader, *blk["attn_q.weight"]),
                   _pick(reader, *blk["attn_q.bias"]),
                   cfg["hidden"], cfg["hidden"])
        collect_fc(tensors, reader, layer("attn_k"),
                   _pick(reader, *blk["attn_k.weight"]),
                   _pick(reader, *blk["attn_k.bias"]),
                   cfg["hidden"], cfg["hidden"])
        collect_fc(tensors, reader, layer("attn_v"),
                   _pick(reader, *blk["attn_v.weight"]),
                   _pick(reader, *blk["attn_v.bias"]),
                   cfg["hidden"], cfg["hidden"])
        collect_fc(tensors, reader, layer("attn_out"),
                   _pick(reader, *blk["attn_out.weight"]),
                   _pick(reader, *blk["attn_out.bias"]),
                   cfg["hidden"], cfg["hidden"])
        collect_norm(tensors, reader, layer("ln2"),
                     _pick(reader, *blk["ln2.weight"]),
                     _pick(reader, *blk["ln2.bias"]),
                     cfg["hidden"])
        collect_fc(tensors, reader, layer("ffn_up"),
                   _pick(reader, *blk["ffn_up.weight"]),
                   _pick(reader, *blk["ffn_up.bias"]),
                   cfg["ff_dim"], cfg["hidden"])
        collect_fc(tensors, reader, layer("ffn_down"),
                   _pick(reader, *blk["ffn_down.weight"]),
                   _pick(reader, *blk["ffn_down.bias"]),
                   cfg["hidden"], cfg["ff_dim"])

        print(f"  block {i:2d}/{cfg['n_layers']} collected")

    # 3. final post_ln.
    collect_norm(tensors, reader, "v_post_ln",
                 _pick(reader, *top["post_ln.weight"]),
                 _pick(reader, *top["post_ln.bias"]),
                 cfg["hidden"])

    return tensors


def convert(args):
    reader = GGUFReader(args.gguf)
    cfg = _infer_config(reader.metadata, reader)

    print("Vision tower config from GGUF:")
    for k, v in cfg.items():
        print(f"  {k:14s} : {v}")
    print()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)

    # Gather the full tensor list once; reuse for both serializers.
    tensors = gather_tensors(reader, cfg)
    total_params = sum(t[1].size for t in tensors)
    print(f"\n  total params : {total_params:,} "
          f"({total_params * 4 / (1024 * 1024):.2f} MiB raw FP32)")

    fmt = args.format
    base, _ext = os.path.splitext(args.output)

    if fmt in ("bin", "both"):
        bin_path = args.output if fmt == "bin" else (base + ".bin")
        write_bin(bin_path, tensors)
        size_mb = os.path.getsize(bin_path) / (1024 * 1024)
        print(f"Wrote {bin_path} ({size_mb:.2f} MiB)  [nntrainer .bin]")

    if fmt in ("safetensors", "both"):
        st_path = args.output if fmt == "safetensors" else (base + ".safetensors")
        write_safetensors(st_path, tensors)
        size_mb = os.path.getsize(st_path) / (1024 * 1024)
        print(f"Wrote {st_path} ({size_mb:.2f} MiB)  [safetensors]")

    if args.emit_nntr_config:
        # nntrainer's CausalLM runtime loads MODEL_FORMAT_BIN, so point
        # model_file_name at the .bin variant when both are written.
        bin_name = base + ".bin" if fmt == "both" else (
            os.path.basename(args.output) if fmt == "bin" else
            os.path.basename(base + ".bin"))
        out_cfg = {
            "model_type": "embedding",
            "architectures": "ClipVitTransformer",
            "model_tensor_type": "FP32-FP32",
            "model_file_name": os.path.basename(bin_name),
            "fc_layer_dtype": "FP32",
            "embedding_dtype": "FP32",
            "batch_size": 1,
            "init_seq_len": (cfg["image_size"] // cfg["patch_size"]) ** 2,
            "max_seq_len":  (cfg["image_size"] // cfg["patch_size"]) ** 2,
            "num_to_generate": 0,
            "image_size":     cfg["image_size"],
            "patch_size":     cfg["patch_size"],
            "num_channels":   cfg["channels"],
            "hidden_size":    cfg["hidden"],
            "num_attention_heads": cfg["n_heads"],
            "num_hidden_layers":   cfg["n_layers"],
            "intermediate_size":   cfg["ff_dim"],
            "layer_norm_eps": 1e-6,
        }
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(args.output)) or ".",
            "nntr_config.json")
        with open(cfg_path, "w") as f:
            json.dump(out_cfg, f, indent=4)
        print(f"Wrote {cfg_path}")

    reader.close()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Convert a GGUF vision tower (LFM2.5-VL / SigLIP2 86M) "
                    "into either the nntrainer .bin layout that "
                    "ClipVitTransformer consumes (Model::load(MODEL_FORMAT_BIN)) "
                    "or a HuggingFace .safetensors file with nntrainer-style "
                    "tensor names, or both.")
    ap.add_argument("gguf", help="input GGUF file")
    ap.add_argument("-o", "--output", default=None,
                    help="output path. Default: vision.bin (for --format bin), "
                         "vision.safetensors (for --format safetensors), or "
                         "vision.{bin,safetensors} when --format both")
    ap.add_argument("--format", default="bin",
                    choices=["bin", "safetensors", "both"],
                    help="output format. bin = raw layer-ordered nntrainer "
                         ".bin (loadable today); safetensors = named-tensor "
                         "HuggingFace .safetensors (FP32); both = write both "
                         "alongside one another. Default: bin")
    ap.add_argument("--emit-nntr-config", action="store_true",
                    help="also write a matching nntr_config.json next to "
                         "the output (always points at the .bin variant)")
    args = ap.parse_args()

    # Resolve the default output path based on --format.
    if args.output is None:
        if args.format == "safetensors":
            args.output = "vision.safetensors"
        else:
            args.output = "vision.bin"
    return args


def main():
    args = parse_args()
    if not os.path.isfile(args.gguf):
        print(f"error: input file not found: {args.gguf}", file=sys.stderr)
        return 1
    convert(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

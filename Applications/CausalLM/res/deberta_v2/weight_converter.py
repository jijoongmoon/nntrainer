# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

# @file weight_converter.py
# @brief Weight conversion script for DeBERTa V2 sentence embedding models.
# @author Samsung Electronics Co., Ltd.

import argparse
import json
import struct

import numpy as np
import torch
from transformers import AutoConfig, AutoModel


SAFETENSORS_DTYPE_MAP = {
    "float32": "F32",
}


def tensor_to_numpy(tensor, dtype, transpose=False):
    """Convert torch tensor to contiguous numpy array."""
    if transpose:
        tensor = tensor.transpose(0, 1)

    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(dtype))


def get_safetensors_output_name(output_name):
    """Return safetensors output path based on the given output name."""
    if output_name.endswith(".bin"):
        return output_name[:-4] + ".safetensors"

    if output_name.endswith(".safetensors"):
        return output_name

    return output_name + ".safetensors"


def strip_encoder_prefix(params):
    """Return a state dict keyed from the DeBERTa encoder root."""
    if "embeddings.word_embeddings.weight" in params:
        return params

    prefixes = (
        "deberta.",
        "0.auto_model.",
        "0.auto_model.deberta.",
        "auto_model.",
        "auto_model.deberta.",
    )
    for prefix in prefixes:
        if prefix + "embeddings.word_embeddings.weight" in params:
            return {
                name[len(prefix):]: value
                for name, value in params.items()
                if name.startswith(prefix)
            }

    raise KeyError("Cannot find DeBERTa V2 encoder weights in state_dict")


def _uses_relative_bias(config):
    pos_att_type = getattr(config, "pos_att_type", None) or []
    if isinstance(pos_att_type, str):
        pos_att_type = pos_att_type.lower().replace("|", " ").split()
    return getattr(config, "relative_attention", False) and (
        "c2p" in pos_att_type or "p2c" in pos_att_type
    )


def _norm_rel_ebd(config):
    norm_rel_ebd = getattr(config, "norm_rel_ebd", "none") or "none"
    return "layer_norm" in norm_rel_ebd.lower()


def _entries(params, config):
    """Yield (nntrainer_name, hf_key, transpose) in nntrainer weight order.

    Mirrors DebertaV2::constructTransformerModule / createDebertaLayer in
    models/deberta_v2/deberta_v2.cpp. Note that rel_embeddings (and its
    optional LayerNorm) are emitted *before* the encoder layers, matching the
    nntrainer graph construction order. Linear weights are transposed; biases
    and LayerNorm gamma/beta are stored as-is.
    """
    params = strip_encoder_prefix(params)

    uses_rel = _uses_relative_bias(config)
    norm_rel = _norm_rel_ebd(config)

    # --- Embeddings ---
    yield "embedding0:Embedding", "embeddings.word_embeddings.weight", False
    yield "embeddings_norm:gamma", "embeddings.LayerNorm.weight", False
    yield "embeddings_norm:beta", "embeddings.LayerNorm.bias", False

    # --- Encoder layers ---
    for i in range(config.num_hidden_layers):
        hf = f"encoder.layer.{i}."
        nn = f"layer{i}"
        self_attn = f"{hf}attention.self."

        yield f"{nn}_wq:weight", f"{self_attn}query_proj.weight", True
        yield f"{nn}_wq:bias", f"{self_attn}query_proj.bias", False
        yield f"{nn}_wk:weight", f"{self_attn}key_proj.weight", True
        yield f"{nn}_wk:bias", f"{self_attn}key_proj.bias", False
        yield f"{nn}_wv:weight", f"{self_attn}value_proj.weight", True
        yield f"{nn}_wv:bias", f"{self_attn}value_proj.bias", False

        # nntrainer's graph emits the shared relative-position embeddings (and
        # their optional LayerNorm) right after the first layer's Q/K/V, since
        # the deberta_attention layer that consumes them is created there.
        if i == 0 and uses_rel:
            yield "rel_embeddings:weight", "encoder.rel_embeddings.weight", False
            if norm_rel:
                yield "rel_embeddings_norm:gamma", "encoder.LayerNorm.weight", False
                yield "rel_embeddings_norm:beta", "encoder.LayerNorm.bias", False

        yield f"{nn}_attention_out:weight", f"{hf}attention.output.dense.weight", True
        yield f"{nn}_attention_out:bias", f"{hf}attention.output.dense.bias", False
        yield f"{nn}_attention_norm:gamma", f"{hf}attention.output.LayerNorm.weight", False
        yield f"{nn}_attention_norm:beta", f"{hf}attention.output.LayerNorm.bias", False

        yield f"{nn}_intermediate:weight", f"{hf}intermediate.dense.weight", True
        yield f"{nn}_intermediate:bias", f"{hf}intermediate.dense.bias", False
        yield f"{nn}_output_dense:weight", f"{hf}output.dense.weight", True
        yield f"{nn}_output_dense:bias", f"{hf}output.dense.bias", False
        # nntrainer names the post-FFN LayerNorm "layer{i}_output".
        yield f"{nn}_output:gamma", f"{hf}output.LayerNorm.weight", False
        yield f"{nn}_output:beta", f"{hf}output.LayerNorm.bias", False


def save_deberta_v2_for_nntrainer(params, config, dtype, file):
    """Save DeBERTa V2 encoder weights in nntrainer binary layer order."""
    for _, hf_key, transpose in _entries(params, config):
        arr = tensor_to_numpy(params_view(params)[hf_key], dtype, transpose=transpose)
        arr.tofile(file)


def collect_deberta_v2_for_nntrainer(params, config, dtype):
    """Collect weights as ordered (nntrainer_name, ndarray) pairs."""
    view = params_view(params)
    weights = []
    for nntr_name, hf_key, transpose in _entries(params, config):
        arr = tensor_to_numpy(view[hf_key], dtype, transpose=transpose)
        weights.append((nntr_name, arr))

    return weights


def params_view(params):
    """Memoize the prefix-stripped state dict view."""
    return strip_encoder_prefix(params)


def save_safetensors(weights, output_path, dtype):
    """Write weights to a safetensors file."""
    if dtype not in SAFETENSORS_DTYPE_MAP:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")

    safetensors_dtype = SAFETENSORS_DTYPE_MAP[dtype]
    metadata = {"format": "pt"}

    offset = 0
    tensor_meta = {}
    raw_buffers = []

    for name, arr in weights:
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        nbytes = arr.nbytes
        tensor_meta[name] = {
            "dtype": safetensors_dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }

        raw_buffers.append(arr.tobytes(order="C"))
        offset += nbytes

    header = {"__metadata__": metadata}
    header.update(tensor_meta)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    with open(output_path, "wb") as output_file:
        output_file.write(struct.pack("<Q", len(header_bytes)))
        output_file.write(header_bytes)

        for buffer in raw_buffers:
            output_file.write(buffer)

    print(f"Saved safetensors: {output_path}")
    print(f"Tensor data size: {offset / 1e9:.2f} GB")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/deberta-v3-small",
        help="Hugging Face model directory or hub id",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="./nntr_deberta_v2_fp32.bin",
        help="Output nntrainer weight path",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="float32",
        choices=["float32"],
        help="Output data type",
    )
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help="Save weights in safetensors format instead of binary format",
    )
    return parser.parse_args()


def main():
    """Convert DeBERTa V2 Hugging Face weights to nntrainer weight format."""
    args = parse_args()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    params = model.state_dict()

    if args.safetensors:
        output_name = get_safetensors_output_name(args.output_name)
        weights = collect_deberta_v2_for_nntrainer(params, config, args.data_type)
        save_safetensors(weights, output_name, args.data_type)
        return

    with torch.no_grad(), open(args.output_name, "wb") as output_file:
        save_deberta_v2_for_nntrainer(params, config, args.data_type, output_file)

    print(f"Saved binary: {args.output_name}")


if __name__ == "__main__":
    main()

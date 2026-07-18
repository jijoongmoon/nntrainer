# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>

## @file weight_converter.py
## @brief weight conversion script for multilingual tiny-BERT embedding model
## @author Seunghui Lee <shsh1004.lee@samsung.com>

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
        tensor = tensor.permute(1, 0)

    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(dtype))


def get_safetensors_output_name(output_name):
    """Return safetensors output path based on the given output name."""
    if output_name.endswith(".bin"):
        return output_name[:-4] + ".safetensors"

    if output_name.endswith(".safetensors"):
        return output_name

    return output_name + ".safetensors"


def resolve_prefix(params):
    """Resolve the BERT encoder key prefix inside the state dict.

    ``AutoModel`` (BertModel) yields ``embeddings.word_embeddings.weight``;
    ``BertForMaskedLM`` yields ``bert.embeddings.word_embeddings.weight``.
    """
    for prefix in ["", "bert."]:
        if f"{prefix}embeddings.word_embeddings.weight" in params:
            return prefix

    raise KeyError("Cannot find BERT encoder weights in state_dict")


def _entries(params, n_layers):
    """Yield (nntrainer_name, hf_key, transpose) in nntrainer weight order.

    Mirrors BertTransformer in models/bert/bert_transformer.cpp:
      embeddings -> per encoder layer [attn q/k/v, attn out, attn norm,
      ffn fc1, ffn down, ffn norm]. FC weights are transposed; biases and
      LayerNorm gamma/beta are stored as-is.
    """
    p = resolve_prefix(params)

    # --- Embeddings (embedding_layer weight name suffix is "Embedding") ---
    yield "embedding0:Embedding", f"{p}embeddings.word_embeddings.weight", False
    yield "position_embedding:Embedding", f"{p}embeddings.position_embeddings.weight", False
    yield "token_type_embedding:Embedding", f"{p}embeddings.token_type_embeddings.weight", False
    yield "embedding_norm:gamma", f"{p}embeddings.LayerNorm.weight", False
    yield "embedding_norm:beta", f"{p}embeddings.LayerNorm.bias", False

    # --- Encoder layers ---
    for i in range(n_layers):
        hf = f"{p}encoder.layer.{i}."
        nn = f"layer{i}"
        self_attn = f"{hf}attention.self."

        yield f"{nn}_wq:weight", f"{self_attn}query.weight", True
        yield f"{nn}_wq:bias", f"{self_attn}query.bias", False
        yield f"{nn}_wk:weight", f"{self_attn}key.weight", True
        yield f"{nn}_wk:bias", f"{self_attn}key.bias", False
        yield f"{nn}_wv:weight", f"{self_attn}value.weight", True
        yield f"{nn}_wv:bias", f"{self_attn}value.bias", False

        yield f"{nn}_attention_out:weight", f"{hf}attention.output.dense.weight", True
        yield f"{nn}_attention_out:bias", f"{hf}attention.output.dense.bias", False
        yield f"{nn}_attention_norm:gamma", f"{hf}attention.output.LayerNorm.weight", False
        yield f"{nn}_attention_norm:beta", f"{hf}attention.output.LayerNorm.bias", False

        yield f"{nn}_ffn_fc1:weight", f"{hf}intermediate.dense.weight", True
        yield f"{nn}_ffn_fc1:bias", f"{hf}intermediate.dense.bias", False
        yield f"{nn}_ffn_down:weight", f"{hf}output.dense.weight", True
        yield f"{nn}_ffn_down:bias", f"{hf}output.dense.bias", False
        yield f"{nn}_ffn_norm:gamma", f"{hf}output.LayerNorm.weight", False
        yield f"{nn}_ffn_norm:beta", f"{hf}output.LayerNorm.bias", False


def save_tinybert_for_nntrainer(params, n_layers, dtype, file):
    """Convert and save weights as nntrainer binary format for tiny-BERT."""
    for _, hf_key, transpose in _entries(params, n_layers):
        arr = tensor_to_numpy(params[hf_key], dtype, transpose=transpose)
        arr.tofile(file)


def collect_tinybert_for_nntrainer(params, n_layers, dtype):
    """Collect weights as ordered (nntrainer_name, ndarray) pairs."""
    weights = []
    for nntr_name, hf_key, transpose in _entries(params, n_layers):
        arr = tensor_to_numpy(params[hf_key], dtype, transpose=transpose)
        weights.append((nntr_name, arr))

    return weights


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
        default="zl369/multilingual-tinyBERT-16MB",
        help="Hugging Face model path or local model directory",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="./tinybert_fp32.bin",
        help="Output weight file path",
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
    """Convert tiny-BERT Hugging Face weights to nntrainer weight format."""
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
        weights = collect_tinybert_for_nntrainer(
            params, config.num_hidden_layers, args.data_type
        )
        save_safetensors(weights, output_name, args.data_type)
        return

    with open(args.output_name, "wb") as output_file:
        save_tinybert_for_nntrainer(
            params, config.num_hidden_layers, args.data_type, output_file
        )

    print(f"Saved binary: {args.output_name}")


if __name__ == "__main__":
    main()

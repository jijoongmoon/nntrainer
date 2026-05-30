# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

# @file weight_converter.py
# @brief Weight conversion script for DeBERTa V2 sentence embedding models.
# @author Samsung Electronics Co., Ltd.

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import DebertaV2Model


def _to_numpy(weight, dtype):
    return weight.detach().cpu().numpy().astype(dtype, copy=False)


def _save_weight(file, weight, dtype):
    _to_numpy(weight, dtype).tofile(file)


def _save_linear(file, params, prefix, dtype):
    _save_weight(file, params[f"{prefix}.weight"].transpose(0, 1), dtype)
    _save_weight(file, params[f"{prefix}.bias"], dtype)


def save_deberta_v2_for_nntrainer(params, config, dtype, file):
    """Save DeBERTa V2 encoder weights in nntrainer layer order."""
    _save_weight(file, params["embeddings.word_embeddings.weight"], dtype)
    _save_weight(file, params["embeddings.LayerNorm.weight"], dtype)
    _save_weight(file, params["embeddings.LayerNorm.bias"], dtype)

    norm_rel_ebd = getattr(config, "norm_rel_ebd", "none").lower().split("|")
    norm_rel_ebd = [item.strip() for item in norm_rel_ebd]
    pos_att_type = getattr(config, "pos_att_type", None) or []
    uses_relative_bias = getattr(config, "relative_attention", False) and (
        "c2p" in pos_att_type or "p2c" in pos_att_type
    )
    saved_relative_embeddings = False

    for layer_idx in range(config.num_hidden_layers):
        layer_prefix = f"encoder.layer.{layer_idx}"
        attn_prefix = f"{layer_prefix}.attention"
        self_prefix = f"{attn_prefix}.self"

        _save_linear(file, params, f"{self_prefix}.query_proj", dtype)
        _save_linear(file, params, f"{self_prefix}.key_proj", dtype)
        _save_linear(file, params, f"{self_prefix}.value_proj", dtype)
        if uses_relative_bias and not saved_relative_embeddings:
            _save_weight(file, params["encoder.rel_embeddings.weight"], dtype)
            if "layer_norm" in norm_rel_ebd:
                _save_weight(file, params["encoder.LayerNorm.weight"], dtype)
                _save_weight(file, params["encoder.LayerNorm.bias"], dtype)
            saved_relative_embeddings = True

        _save_linear(file, params, f"{attn_prefix}.output.dense", dtype)
        _save_weight(file, params[f"{attn_prefix}.output.LayerNorm.weight"], dtype)
        _save_weight(file, params[f"{attn_prefix}.output.LayerNorm.bias"], dtype)
        _save_linear(file, params, f"{layer_prefix}.intermediate.dense", dtype)
        _save_linear(file, params, f"{layer_prefix}.output.dense", dtype)
        _save_weight(file, params[f"{layer_prefix}.output.LayerNorm.weight"], dtype)
        _save_weight(file, params[f"{layer_prefix}.output.LayerNorm.bias"], dtype)


def _encoder_state_dict(model):
    if hasattr(model, "deberta"):
        return model.deberta.state_dict()
    return model.state_dict()


def _strip_encoder_prefix(params):
    if "embeddings.word_embeddings.weight" in params:
        return params

    prefixes = (
        "0.auto_model.",
        "0.auto_model.deberta.",
        "auto_model.",
        "auto_model.deberta.",
        "deberta.",
    )
    for prefix in prefixes:
        key = prefix + "embeddings.word_embeddings.weight"
        if key in params:
            return {
                name[len(prefix) :]: value
                for name, value in params.items()
                if name.startswith(prefix)
            }

    raise KeyError("Cannot find DeBERTa V2 encoder weights in state_dict")


def _load_model_config_and_params(model_path):
    try:
        model = DebertaV2Model.from_pretrained(model_path)
        return model.config, _strip_encoder_prefix(_encoder_state_dict(model))
    except Exception as deberta_error:
        try:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(model_path, trust_remote_code=True)
            auto_model = getattr(st_model[0], "auto_model", None)
            if auto_model is None:
                raise AttributeError(
                    "first SentenceTransformer module has no auto_model"
                )
            return auto_model.config, _strip_encoder_prefix(st_model.state_dict())
        except Exception as st_error:
            raise RuntimeError(
                "Failed to load as DebertaV2Model or SentenceTransformer"
            ) from st_error


def main():
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
        help="Output nntrainer binary weight path",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Weight dtype written to the binary file",
    )
    args = parser.parse_args()

    dtype = np.float16 if args.data_type == "float16" else np.float32
    config, params = _load_model_config_and_params(args.model_path)

    output_path = Path(args.output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), output_path.open("wb") as output_file:
        save_deberta_v2_for_nntrainer(params, config, dtype, output_file)


if __name__ == "__main__":
    main()

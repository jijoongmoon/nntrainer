# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file generate_deberta_v2_reference.py
## @brief Generate golden fixtures for the DebertaV2 differential reference test.
##
## Builds a tiny DeBERTa V2 encoder (matching the nntrainer DebertaV2 graph),
## saves its weights in nntrainer graph order, runs a HuggingFace DebertaV2Model
## forward pass, applies mean pooling + L2 normalization (matching nntrainer's
## SentenceTransformer module pipeline), and writes the embedding fixture.
##
## nntrainer weight order (graph construction order in deberta_v2.cpp):
##   word_emb, embedding LayerNorm(gamma,beta),
##   rel_embeddings.weight                         <- before any encoder layer
##   (optional encoder.LayerNorm if norm_rel_ebd contains "layer_norm")
##   per layer: Wq(w,b) Wk(w,b) Wv(w,b)
##              Wo(w,b) attn-LN(g,b)
##              ffn_fc1(w,b) ffn_down(w,b) ffn-LN(g,b)
## Fully-connected weights are transposed ([out,in] -> [in,out]).
##
## This ordering differs from the existing res/deberta_v2/weight_converter.py
## (which saved rel_embeddings inside layer 0 after Q/K/V).  The generator here
## follows the C++ graph construction order so tests are self-contained.
##
## Usage:
##   python3 generate_deberta_v2_reference.py [--out <dir>] [--seed <int>]
##
## Requirements: torch >= 2.0, transformers >= 4.40 (DeBERTa V2 support)

import argparse
import json
import pathlib

import numpy as np
import torch
from transformers import DebertaV2Config, DebertaV2Model

THIS_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = THIS_DIR.parent / "deberta_v2_tiny"

# Tiny model dimensions
HIDDEN = 64
N_LAYERS = 1
N_HEADS = 4          # head_dim = HIDDEN / N_HEADS = 16
INTERMEDIATE = 128
VOCAB = 32
MAX_POS = 16
MAX_REL_POS = 8      # rel_embed_size = MAX_REL_POS * 2 = 16
LN_EPS = 1e-7
SEQ_LEN = 4

INPUT_IDS = [1, 4, 2, 3]
PROMPT = "hello tok4 world tok3"

TINY_TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {"id": 31, "content": "<eos>", "single_word": False, "lstrip": False,
         "rstrip": False, "normalized": False, "special": True}
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {
            "<unk>": 0, "hello": 1, "world": 2,
            **{f"tok{i}": i for i in range(3, 31)},
            "<eos>": 31,
        },
        "unk_token": "<unk>",
    },
}

MODULES_JSON = [
    {"idx": 0, "name": "0", "path": "",
     "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling",
     "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2", "path": "",
     "type": "sentence_transformers.models.Normalize"},
]

POOLING_CONFIG = {
    "word_embedding_dimension": HIDDEN,
    "pooling_mode_cls_token": False,
    "pooling_mode_mean_tokens": True,
    "pooling_mode_max_tokens": False,
    "pooling_mode_mean_sqrt_len_tokens": False,
    "pooling_mode_weightedmean_tokens": False,
    "pooling_mode_lasttoken": False,
    "include_prompt": True,
}


def build_model(seed: int):
    torch.manual_seed(seed)
    config = DebertaV2Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        intermediate_size=INTERMEDIATE,
        max_position_embeddings=MAX_POS,
        # Disable absolute position and token-type embeddings so HF and
        # nntrainer match: nntrainer DebertaV2 has only word embeddings + LN.
        type_vocab_size=0,
        position_biased_input=False,
        layer_norm_eps=LN_EPS,
        # Disentangled relative attention with both C2P and P2C enabled.
        relative_attention=True,
        max_relative_positions=MAX_REL_POS,
        pos_att_type=["p2c", "c2p"],
        share_att_key=True,
        norm_rel_ebd="none",
        position_buckets=-1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        pad_token_id=0,
    )
    model = DebertaV2Model(config)
    model.eval()
    return model, config


def save_deberta_v2_nntrainer_order(params: dict, config, dtype, file) -> None:
    """Save weights in the order nntrainer reads them from binary (graph order).

    nntrainer builds layers in this sequence (deberta_v2.cpp):
      embedding0        -> word_embeddings.weight
      embeddings_norm   -> embeddings.LayerNorm.{weight,bias}
      rel_embeddings    -> encoder.rel_embeddings.weight   (before any layer)
      [rel_embeddings_norm -> encoder.LayerNorm.{weight,bias}] (if norm_rel_ebd)
      per layer i:
        layer{i}_wq          -> attention.self.query_proj.{weight.T, bias}
        layer{i}_wk          -> attention.self.key_proj.{weight.T, bias}
        layer{i}_wv          -> attention.self.value_proj.{weight.T, bias}
        (wq_rel, wk_rel: shared from wq/wk, no new params)
        (deberta_attention: no trainable params)
        layer{i}_attention_out -> attention.output.dense.{weight.T, bias}
        layer{i}_attention_norm -> attention.output.LayerNorm.{weight,bias}
        layer{i}_intermediate -> intermediate.dense.{weight.T, bias}
        layer{i}_output_dense -> output.dense.{weight.T, bias}
        layer{i}_output       -> output.LayerNorm.{weight,bias}
    """
    def save(t):
        np.array(t.detach().float().numpy(), dtype=dtype).tofile(file)

    def save_fc(prefix):  # transposed weight then bias
        save(params[f"{prefix}.weight"].T)
        save(params[f"{prefix}.bias"])

    def save_ln(prefix):  # weight then bias as-is
        save(params[f"{prefix}.weight"])
        save(params[f"{prefix}.bias"])

    # Embedding + embedding LayerNorm
    save(params["embeddings.word_embeddings.weight"])
    save_ln("embeddings.LayerNorm")

    # Relative position embeddings (global, before any encoder layer)
    save(params["encoder.rel_embeddings.weight"])

    # Optional relative embeddings LayerNorm
    norm_rel_ebd = getattr(config, "norm_rel_ebd", "none") or "none"
    if "layer_norm" in norm_rel_ebd.lower():
        save_ln("encoder.LayerNorm")

    # Per-layer weights
    for i in range(config.num_hidden_layers):
        p = f"encoder.layer.{i}."
        save_fc(p + "attention.self.query_proj")
        save_fc(p + "attention.self.key_proj")
        save_fc(p + "attention.self.value_proj")
        # wq_rel and wk_rel share weights with wq/wk — no new parameters
        # deberta_attention has no trainable parameters
        save_fc(p + "attention.output.dense")
        save_ln(p + "attention.output.LayerNorm")
        save_fc(p + "intermediate.dense")
        save_fc(p + "output.dense")
        save_ln(p + "output.LayerNorm")


def run_embedding(model: DebertaV2Model, input_ids: list) -> list:
    """Run HuggingFace DebertaV2Model, mean-pool and L2-normalise."""
    ids = torch.tensor([input_ids], dtype=torch.long)
    # Full attention mask: all tokens visible to all tokens (no padding).
    attention_mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model(ids, attention_mask=attention_mask)
    hidden = out.last_hidden_state[0]   # [seq_len, hidden]
    pooled = hidden.mean(dim=0)          # mean pooling over all tokens
    norm = pooled.norm(p=2)
    emb = pooled / norm if norm > 0 else pooled
    return emb.float().tolist()


def write_configs(out_dir: pathlib.Path, bin_name: str,
                  tokenizer_path: str) -> None:
    config_json = {
        "architectures": ["DebertaV2ForMaskedLM"],
        "model_type": "deberta-v2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "intermediate_size": INTERMEDIATE,
        "hidden_act": "gelu",
        "max_position_embeddings": MAX_POS,
        "vocab_size": VOCAB,
        "type_vocab_size": 0,
        "position_biased_input": False,
        "layer_norm_eps": LN_EPS,
        "relative_attention": True,
        "max_relative_positions": MAX_REL_POS,
        "pos_att_type": ["p2c", "c2p"],
        "share_att_key": True,
        "norm_rel_ebd": "none",
        "position_buckets": -1,
        "attention_probs_dropout_prob": 0.0,
        "hidden_dropout_prob": 0.0,
        # Extra fields consumed by nntrainer's Transformer base class
        "rope_theta": 0,
        "rms_norm_eps": LN_EPS,
        "tie_word_embeddings": False,
        "num_key_value_heads": N_HEADS,
        "is_causal": False,
    }
    generation_config_json = {
        "bos_token_id": 0,
        "eos_token_id": 31,
        "do_sample": False,
        "top_k": 1,
        "top_p": 1.0,
        "temperature": 1.0,
    }
    nntr_config_json = {
        "bad_word_ids": [],
        "batch_size": 1,
        "embedding_dtype": "FP32",
        "fc_layer_dtype": "FP32",
        "init_seq_len": SEQ_LEN,
        "lmhead_dtype": "FP32",
        "max_seq_len": MAX_POS,
        "model_file_name": bin_name,
        "model_tensor_type": "FP32-FP32",
        "model_type": "Embedding",
        "module_config_path": "modules.json",
        "num_to_generate": 1,
        "tokenizer_file": tokenizer_path,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)
    with open(out_dir / "generation_config.json", "w") as f:
        json.dump(generation_config_json, f, indent=2)
    with open(out_dir / "nntr_config.json", "w") as f:
        json.dump(nntr_config_json, f, indent=2)


def write_modules(out_dir: pathlib.Path) -> None:
    with open(out_dir / "modules.json", "w") as f:
        json.dump(MODULES_JSON, f, indent=2)
    pooling_dir = out_dir / "1_Pooling"
    pooling_dir.mkdir(parents=True, exist_ok=True)
    with open(pooling_dir / "config.json", "w") as f:
        json.dump(POOLING_CONFIG, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DebertaV2 tiny reference fixtures")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate] output dir: {out_dir}")

    model, config = build_model(args.seed)
    print(f"[generate] model params: {sum(p.numel() for p in model.parameters()):,}")

    bin_name = "nntr_deberta_v2_tiny_fp32.bin"
    with open(out_dir / bin_name, "wb") as bf:
        save_deberta_v2_nntrainer_order(model.state_dict(), config, np.float32, bf)
    size_kb = (out_dir / bin_name).stat().st_size / 1024
    print(f"[converter] saved {out_dir / bin_name} ({size_kb:.1f} KB)")

    tokenizer_path = out_dir / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(TINY_TOKENIZER, f, indent=2)

    write_configs(out_dir, bin_name, str(tokenizer_path))
    write_modules(out_dir)
    print("[generate] configs + modules.json written")

    ref_embedding = run_embedding(model, INPUT_IDS)
    with open(out_dir / "reference_embedding.json", "w") as f:
        json.dump(ref_embedding, f)
    norm_check = sum(v * v for v in ref_embedding) ** 0.5
    print(f"[generate] reference embedding: {len(ref_embedding)} values, "
          f"norm={norm_check:.4f}")

    with open(out_dir / "input_ids.json", "w") as f:
        json.dump(INPUT_IDS, f)

    import transformers
    meta = {
        "seed": args.seed,
        "input_ids": INPUT_IDS,
        "prompt": PROMPT,
        "embedding_atol": 5e-2,
        "cosine_min": 0.995,
        "embedding_atol_q40": 0.1,
        "cosine_min_q40": 0.99,
        "pooling": "mean",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[generate] meta.json written (transformers={transformers.__version__})")

    print(f"\n[generate] done. Commit: {out_dir}")


if __name__ == "__main__":
    main()

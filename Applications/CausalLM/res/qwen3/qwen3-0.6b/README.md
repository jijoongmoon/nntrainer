# gguf_to_nntrainer.py

Convert a **Qwen3 GGUF** model into an **nntrainer `.bin` weight file**.

This script is mainly intended for **Qwen3-0.6B** style GGUF files and writes:

- **Embedding** as `Q6_K`
- **FC weights** as `Q4_0` by default
- **Norm weights** as `FP32`

It can also emit a matching `nntr_config.json`.

---

## What this script does

The converter reads tensors directly from GGUF and writes them in the layout expected by nntrainer.

By default:

- `token_embd.weight` -> `Q6_K`
- FC weights such as attention / FFN linear weights -> `Q4_0`
- RMSNorm weights -> `FP32`

For `Q4_0` FC weights, the script also repacks them into nntrainer’s internal layout:

- `q4_0x8` for **x86**
- `q4_0x4` for **ARM**

---

## Supported tensor mapping

The script expects Qwen3 tensor names like:

- `token_embd.weight`
- `blk.{i}.attn_norm.weight`
- `blk.{i}.attn_q.weight`
- `blk.{i}.attn_q_norm.weight`
- `blk.{i}.attn_k.weight`
- `blk.{i}.attn_k_norm.weight`
- `blk.{i}.attn_v.weight`
- `blk.{i}.attn_output.weight`
- `blk.{i}.ffn_norm.weight`
- `blk.{i}.ffn_up.weight`
- `blk.{i}.ffn_gate.weight`
- `blk.{i}.ffn_down.weight`
- `output_norm.weight`
- `output.weight` (only if the model is **not tied**)

---

## Basic usage

### 1. Convert a GGUF file for x86

```bash
python3 gguf_to_nntrainer.py \
  /path/to/qwen3-0.6b.gguf \
  -o nntr_qwen3_0.6b_q40_embdq6k.bin \
  --target x86 \
  --emit-nntr-config
```

### 2. Convert a GUUF file for ARM

```bash
python3 gguf_to_nntrainer.py \
  /path/to/qwen3-0.6b.gguf \
  -o nntr_qwen3_0.6b_q40_arm.bin \
  --target arm \
  --emit-nntr-config
```

---

# QINT4-FP16 path (GPU 1K benchmark model)

The GPU benchmark model `qwen3_lg_q6k` uses a different recipe: **embedding
Q6_K, FC QINT4, lm_head Q6_K, FP16 activations** (`model_tensor_type:
QINT4-FP16`). It is produced from the HF checkpoint (not GGUF) via a two-step
pipeline:

1. `weight_converter_layergraph.py` — numpy-only HF Qwen3 → FP32-weights /
   FP16-norms nntrainer layer-graph `.bin` (the quantize source). This is the
   numpy sibling of the torch `weight_converter.py`; it needs no torch/
   transformers.
2. `nntr_quantize ... --fc_dtype QINT4 --embd_dtype Q6_K --lmhead_dtype Q6_K`.

```bash
# one-shot (auto-detects nntr_quantize under <repo>/build*/):
./build_qint4.sh /path/to/hf/Qwen3-0.6B /path/to/out_qwen3_qint4
```

`weight_converter.py` (torch) remains for FP32 / safetensors exports;
`gguf_to_nntrainer.py` above is the separate GGUF→Q4_0 (x86/ARM) path.

# Qwen3-1.7B (Q4_0 FC + Q6_K embedding)

Resources and converter for running Qwen3-1.7B inside `Applications/CausalLM`
from a HuggingFace GGUF file, without the usual FP32 round-trip.

## Files

- `gguf_to_nntrainer.py` — converts a GGUF to an nntrainer `.bin`:
  - Q6_K `token_embd.weight` is copied byte-for-byte.
  - Each Q4_0 linear weight is repacked into nntrainer's interleaved
    `q4_0x8` (x86, default) or `q4_0x4` (ARM) layout with the `0x88` nibble
    XOR the C backends apply at save time.
  - RMSNorm scales are emitted as FP32.
  - When the GGUF has `tie_word_embeddings=true` (the default for 1.7B),
    `output.weight` is omitted so the shared embedding weight is only written
    once, matching nntrainer's save path.
- `nntr_config.json` — matching config (`Q4_0-FP32`, Q6_K embedding).

## Usage

```bash
# Grab a Qwen3-1.7B GGUF with Q4_0 FC + Q6_K embedding
# (e.g. Qwen/Qwen3-1.7B-GGUF, file qwen3-1.7b-q4_0.gguf).
python3 gguf_to_nntrainer.py \
    /path/to/qwen3-1.7b-q4_0.gguf \
    -o nntr_qwen3_1.7b_q40_embdq6k.bin \
    --target x86          # or 'arm' for Android builds
    --strict              # fail unless embedding=Q6_K and FC=Q4_0
```

For GGUFs that don't match exactly (e.g. an FFN stored as Q6_K), drop
`--strict` and the converter will dequantise then re-quantise that tensor.

Then copy `config.json`, `generation_config.json`, and `tokenizer.json`
from the HuggingFace `Qwen/Qwen3-1.7B` repo into this directory (alongside
`nntr_config.json`) and run `nntr_causallm` against the directory.

## Layout the script produces

The binary is a concatenation of raw tensor bytes in the order the
`Qwen3CausalLM` graph traverses them:

```
embedding0                      (Q6_K)
for each layer i:
  layer{i}_attention_norm       (FP32)
  layer{i}_wq                   (Q4_0x8 / Q4_0x4)
  layer{i}_q_norm               (FP32)
  layer{i}_wk                   (Q4_0x8 / Q4_0x4)
  layer{i}_k_norm               (FP32)
  layer{i}_wv                   (Q4_0x8 / Q4_0x4)
  layer{i}_attention_out        (Q4_0x8 / Q4_0x4)
  layer{i}_ffn_norm             (FP32)
  layer{i}_ffn_up               (Q4_0x8 / Q4_0x4)
  layer{i}_ffn_gate             (Q4_0x8 / Q4_0x4)
  layer{i}_ffn_down             (Q4_0x8 / Q4_0x4)
output_norm                     (FP32)
# output_of_causallm is shared with embedding when tie_word_embeddings=true
```

`N` (rows = `out_features`) must be divisible by the interleave group size
(8 for x86, 4 for ARM) — this is true for every linear in Qwen3-1.7B.

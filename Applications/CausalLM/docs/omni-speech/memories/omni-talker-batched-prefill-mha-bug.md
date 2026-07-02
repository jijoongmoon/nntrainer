---
name: omni-talker-batched-prefill-mha-bug
description: "nntrainer Talker: a single batched multi-row prefill gives a wrong LAST-ROW output; processing the prefill row-by-row (incremental) matches HF exactly"
metadata: 
  node_type: memory
  type: project
  originSessionId: ccf3254c-6df0-4720-919c-83cc763b069c
---

In the Qwen2.5-Omni **Talker** (`Applications/CausalLM/models/qwen25_omni/qwen25_omni_talker_causallm.cpp`), a single batched `incremental_inference(.., L0, 0, L0)` over the whole prefill writes a **correct KV cache** (autoregressive generation that reads it reproduces HF codec ids exactly) but the **last row's output logits diverge from HF** — it picked a different first codec token (7653 vs HF's 8028; 8028 wasn't even in nntr's batched top-6, so NOT a near-tie).

**Fix (used in Phase 1):** feed the prefill rows **one at a time** via `incremental_inference(.., L0, r, r+1)` for r=0..L0-1 — the same incremental path generation uses — and take the last row's logits. This reproduces HF's first codec token exactly (8028, logit 18.0, clear top-1). The prompt-context rows' outputs are unused (only their cached K/V matter), so row-by-row prefill is exact and is the chosen approach.

**Why:** root cause not yet isolated in core nntrainer. The Talker is the first model combining causal=true + **external M-RoPE** (`mrope_apply`, so mha_core runs `rope_theta=0`) + persistent KV cache + **batched** multi-row prefill. The text model (causal+cache+batched but mha's built-in rope) and the vision encoder (external mrope+batched but causal=false, no cache, verified cosine 1.0 over all rows) each work; only their combination here exposes it. The batched output[0] IS the last row (the text model proves that convention), so it's a genuine batched-attention/output computation difference for that config, not a return-row mismatch.

**Follow-up:** worth a focused look at `mha_core` batched forwarding when `rope_theta=0` + causal + external KV cache, to see whether the last-row (or all-row) attention OUTPUT is mis-masked while the cached K/V stay correct. Until then, row-by-row prefill is the correct workaround. See [[omni-talker-speech-output]].

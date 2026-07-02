---
name: weight-bin-load-order-dfs
description: "CausalLM .bin weight order is graph DFS-from-output order, not layer creation order — ffn_gate loads before ffn_up"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0bfecd2f-f705-494e-9a95-33ff10cccdb3
---

In Applications/CausalLM, the sequential `.bin` weight load order is the symbolic graph's DFS-from-output order (`Model::compile` in `api/ccapi/src/tensor_api_graph.cpp`), NOT the layer creation order, since the symbolic-graph migration (commit 9159ec1cc, 2026-04-26).

**Why:** `Transformer::createMlp` creates the `ffn_up` FC before `ffn_gate`, but wires `swiglu({gate, up})`; the DFS visits the gate branch first, so **gate_proj weights must be written before up_proj** in converters. Commit de8f981cf (2026-05-18) fixed qwen2's converter accordingly. Verified empirically (2026-06-12) with a synthetic random-weight Qwen2.5-Omni tiny checkpoint: gate-first reproduces HF greedy tokens 12/12 exactly; up-first diverges.

**How to apply:** Any new HF→nntrainer weight converter must write per-layer MLP weights as gate, up, down. Also beware false-positive validation: with small-init random weights swiglu is nearly symmetric (silu(x)≈x/2), so a swapped order can still match several greedy tokens — compare against HF argmax at near-tie steps or use trained weights.

**Full qwen3 DFS order (verified 2026-06-13 vs HF Qwen3-0.6B, coherent):** per decoder block = `input_layernorm, q_proj(w^T), q_norm, k_proj(w^T), k_norm, v_proj(w^T), o_proj(w^T), post_attention_layernorm, gate(w^T), up(w^T), down(w^T)`; embedding first, output_norm last, NO lm_head when tie_word_embeddings (shared from embedding0). The q/k RMSNorms sit right after their projections (mha_core consumes q_normed/k_normed). Only the MLP gate/up order changed in the migration — the attention sub-order already matches creation order.

**Fixed (2026-06-13):** `res/qwen3/qwen3-0.6b/gguf_to_nntrainer.py` swapped to gate-first (its attention order was already correct). Added `res/qwen3/qwen3-0.6b/weight_converter.py` (HF→nntrainer, gate-first). NOTE: the pre-existing `models/qwen3_0.6b/*.bin` and `res/qwen3/qwen3-0.6b/nntr_qwen3_0.6b_fp32.bin` (April, untracked) are STALE — they produce garbage on the current build (more than just gate/up differs) and need regeneration with the new converter.

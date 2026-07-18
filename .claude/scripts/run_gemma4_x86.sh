#!/bin/bash
# Gemma4-E2B (QINT4-FP16, untied lm_head int4) on x86 Intel Arc [0x7d55] via OpenCL.
# Verified COHERENT 2026-06-19 (HEAD ef7cfcdf, build_cl):
#   "The capital of South Korea is **Seoul**."  (M=16 and M=45 GPU-rope path)
# Intel buffer path = NNTR_V8C_BUF (NEO can't do image read_imageui); NO KV_IMG_ATTN (Adreno-only).
# GPU-rope prefill (M>=32) coherent => commit 24abaeb5 (RoPE-LUT cap + half_d key) works on Intel too.
# REQUIRES rebuilt nntr_causallm: ninja -C build_cl Applications/CausalLM/nntr_causallm
# Model dir must have nntr_config.json with "model_type":"CausalLM" + local tokenizer_file.
set -e
ROOT=/home/nntrainer/nntrainer
MODEL=${MODEL:-/home/nntrainer/qwen3_e2e/gemma4_e2b_qint4fp16_lmint4}
cd "$ROOT"
# no extra arg -> uses chat-templated sample_input from nntr_config.json (coherence test)
# pass a prompt as $1 for a raw (non-chat-templated) prompt
NNTR_GPU_SVM_POOL=1 NNTR_V8C_BUF=1 NNTR_MHA_GPU=1 NNTR_FC_GPU=1 \
  NNTR_FC_INT8_GPU=1 NNTR_GPU_CLMEM_POOL=1 \
  ./build_cl/Applications/CausalLM/nntr_causallm "$MODEL" "$@"

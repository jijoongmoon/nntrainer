#!/bin/bash
# Run Gemma2-2B (QINT4-FP16, Q6_K lm_head) on x86 Intel Arc via OpenCL.
# Model pulled from S26U: models/gemma2_lg_q6k -> ~/qwen3_e2e/gemma2_lg_q6k
# REQUIRES build_cl at HEAD (BOS fix 99b8c7b5 + Q6_K lm_head GEMV 83ecdaca).
# NNTR_GPU_CLMEM_POOL=1 is mandatory for coherence (without it -> greedy-collapse).
# NNTR_V8C_BUF=1 is the Intel buffer FC path (Intel NEO can't do the image read path).
set -e
ROOT=/home/nntrainer/nntrainer
MODEL=${MODEL:-/home/nntrainer/qwen3_e2e/gemma2_lg_q6k}
PROMPT="${1:-The capital of France is}"
cd "$ROOT"
NNTR_GPU_SVM_POOL=1 NNTR_V8C_BUF=1 NNTR_MHA_GPU=1 NNTR_FC_GPU=1 \
  NNTR_FC_INT8_GPU=1 NNTR_GPU_CLMEM_POOL=1 \
  ./build_cl/Applications/CausalLM/nntr_causallm "$MODEL" "$PROMPT"

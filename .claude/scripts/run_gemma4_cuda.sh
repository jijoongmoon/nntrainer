#!/bin/bash
# Gemma4-E2B (QINT4-FP16, untied lm_head int4) on NVIDIA CUDA (build_cuda).
# README_GPU.md §8.2 — discrete (RTX) env: device-only activations + decode CUDA-graph.
# Usage: run_gemma4_cuda.sh ["prompt"]   (no arg -> chat-templated sample_input coherence test)
#        SAFE=1 run_gemma4_cuda.sh ...    -> integrated/safe-set (no DEV_ACT/ASYNC/M2B/VCOPY)
set -e
ROOT=/home/nntrainer/nntrainer
MODEL=${MODEL:-/home/nntrainer/qwen3_e2e/gemma4_e2b_qint4fp16_lmint4}
BD="$ROOT/build_cuda"
export LD_LIBRARY_PATH="$BD/Applications/CausalLM:$BD/Applications/CausalLM/layers:$BD/nntrainer:$BD/api/ccapi:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
# common GPU-op flags (all models, both HW classes)
COMMON="NNTR_ENGINE=cuda NNTR_CUDA_ROPE=1 NNTR_CUDA_ATTN=1 NNTR_CUDA_KV_UVM=1 NNTR_CUDA_GEGLU=1 NNTR_CUDA_ELTWISE=1 NNTR_CUDA_QKNORM=1 NNTR_CUDA_FLASH_DECODE=64 NNTR_CUDA_BLOCKQ=1 NNTR_FC_CUDA_CUBLAS=1 NNTR_CUDA_PREWARM=1"
if [ "${SAFE:-0}" = 1 ]; then
  ENV="$COMMON"   # integrated/safe: managed activations, no discrete tricks
else
  # discrete (RTX) residency add-ons + gemma4 decode CUDA-graph
  ENV="$COMMON NNTR_CUDA_DEV_ACT=1 NNTR_CUDA_VCOPY_PREFILL=1 NNTR_RMSNORM_CUDA_OFF=all NNTR_CUDA_M2B=1 NNTR_CUDA_ASYNC=1"
fi
echo "[run_gemma4_cuda] SAFE=${SAFE:-0}  MODEL=$MODEL"
env $ENV "$BD/Applications/CausalLM/nntr_causallm" "$MODEL" "$@"

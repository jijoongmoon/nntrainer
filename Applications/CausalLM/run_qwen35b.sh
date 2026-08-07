#!/usr/bin/env bash
# Qwen3.6-35B-A3B on this base. FIRST CUDA RUN -- the CPU graph is validated
# (e2e 3.8e-4 vs HF) and the 20.8 GB bin loads positionally, but nothing of this
# model has ever executed on the device here.
#
# NOTE the retired lane's recipe required NNTR_CUDA_MOE=1 ("CPU path would
# SIGILL": Orin has no i8mm, so a host QINT4 dot is an illegal instruction).
# This base has NO GPU MoE at all, so that lever does not exist yet -- which is
# exactly what this run is meant to demonstrate.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
MODEL=${1:-$HOME/jijoongmoon/models/qwen35b}
BIN=./build/Applications/CausalLM/nntr_causallm
export LD_LIBRARY_PATH=$ROOT/build/nntrainer:$ROOT/build/Applications/CausalLM:$ROOT/build/Applications/CausalLM/layers:$ROOT/build/Applications/CausalLM/models/qwen3_5_moe:$ROOT/build/api/ccapi:$ROOT/subprojects/OpenBLAS/build/lib:/usr/local/cuda/lib64
export NNTR_ENGINE=cuda
# The integrated profile now auto-arms rope/attn/kv-uvm/eltwise/qknorm/cublas/
# rmsnorm-all/vcopy-prefill, so only the graph levers and diagnostics are here.
export NNTR_CUDA_FC_DBG=1       # count host-dot FC fallbacks (want 0)
export NNTR_CUDA_CAP_AUDIT=1    # count host ops inside a capture (want 0)
echo "==== qwen3.6-35B-A3B: first CUDA run on this base ===="
"$BIN" "$MODEL" ${2:+"$2"}
echo "==== EXIT: $? ===="

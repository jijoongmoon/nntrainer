#!/usr/bin/env bash
# =============================================================================
# run_llm.sh — HW(backend) x model 통합 러너 (2026-07-08 검증 매트릭스 기준)
#
# Usage:
#   run_llm.sh <backend> <model> [perf | "질문 텍스트"]
#
#   backend:
#     xmx        Intel Xe3 GPU, XMX/DPAS GEMM        (build_cl)
#     intel      Intel Xe3 GPU, dp4a (XMX off)       (build_cl)
#     wrap       Intel XMX, uint16-Half wrapper 빌드  (build_wrap)
#     cuda       NVIDIA RTX, SAFE env(지원 구성)      (build_cuda)
#     cuda-fast  NVIDIA RTX, discrete 가속 세트       (build_cuda)
#     adreno     S26U(R3CY70LV96T) Adreno 840 GPU    (단말 배포본)
#     adreno-cpu S26U ARM CPU (KAI, fp32-act) — quant CPU 골든 (x86엔 int4 CPU 경로 없음/NYI)
#
#   model: gauss4 | gemma4 | gemma4-lmint4 | gemma2 | qwen3
#
#   3번째 인자:
#     (없음)      coherence 테스트 — 표준 질문을 모델별 chat template로 래핑해 실행
#     perf        1.2K 프롬프트(prompt_1p2k.txt, ~999-1024tok로 절단됨) 성능 측정
#     그 외 문자열  해당 텍스트를 질문으로 템플릿 래핑해 실행
#
# 기대 결과(2026-07-08 실측, 자세한 표는 README_RUN.md):
#   coherence: gauss4/gemma4 → "Seoul" 포함, gemma2(base) → 프랑스 관련 이어쓰기,
#              qwen3 → CUDA/CPU는 "Seoul", Intel/Adreno는 fp16 마진 오답 가능(기지 사항)
#   perf @~1K: gemma4 CUDA 5808/64 · XMX ~2500/16+ · Adreno 2373/21.9(qs4cx)
#              gauss4 Adreno 1249/19.2 · gemma2 Adreno 835/16 · qwen3 Adreno 2198/33
#
# 주의:
#  - Adreno image attn(NNTR_KV_IMG_ATTN)은 값-체크(=0이 실제로 끔, ca7e36a9 이후).
#    sliding window는 커널이 마스킹하므로 window<max_seq 모델도 image 경로 안전.
#  - 단말은 폰 2대 연결 가능성 → 항상 -s $ADRENO_SERIAL.
#  - 842tok 완결 지문(prompt_1k.txt)은 chat 모델이 1토큰 EOS를 내는 게 정상.
# =============================================================================
set -euo pipefail

REF_ROOT=/home/nntrainer/nntrainer            # reference repo (x86 빌드들)
PROMPT_1P2K=/home/nntrainer/migrate_stage/device_aux/prompt_1p2k.txt
ADRENO_SERIAL=R3CY70LV96T                     # Galaxy S26 Ultra (SM8850/Adreno 840)
DEVDIR=/data/local/tmp/nntrainer/causallm

BACKEND=${1:?usage: run_llm.sh <backend> <model> [perf|\"질문\"]}
MODEL=${2:?usage: run_llm.sh <backend> <model> [perf|\"질문\"]}
ARG3=${3:-}

# ---------- 모델 테이블 ----------------------------------------------------
# LOCAL_DIR: x86/CUDA용 모델 디렉토리, DEV_DIR: 단말 모델 디렉토리(models/ 하위)
# TMPL: 질문 → 프롬프트 래핑 방식
case "$MODEL" in
  gauss4)
    LOCAL_DIR=/home/nntrainer/GAUSS4/nntr_model_untie2_qs4cx
    DEV_DIR=gauss4_packed
    TMPL=gauss4 ;;                # gemma4 턴 마커 + no-think 고스트 채널
  gauss4-ple)                     # PLE sidecar 분리(mmap on-demand dequant) — 출력은 gauss4와 bit-exact
    LOCAL_DIR=/home/nntrainer/GAUSS4/nntr_model_untie2_qs4cx_ple
    DEV_DIR=gauss4_ple
    TMPL=gauss4 ;;
  gemma4-ple)                     # PLE sidecar 분리 — 출력은 gemma4와 bit-exact, peak mem -1.8GB
    LOCAL_DIR=/home/nntrainer/qwen3_e2e/gemma4_qs4cx_fp16_ple
    DEV_DIR=gemma4_ple
    TMPL=gemma4 ;;
  gemma4)
    LOCAL_DIR=/home/nntrainer/qwen3_e2e/gemma4_qs4cx_fp16
    DEV_DIR=gemma4_qs4cx_fp16
    TMPL=gemma4 ;;
  gemma4-lmint4)                  # README known-good 매트릭스/CUDA 5550-급 기준 모델
    LOCAL_DIR=/home/nntrainer/qwen3_e2e/gemma4_lmint4
    DEV_DIR=gemma4_lmint4
    TMPL=gemma4 ;;
  gemma2)                         # base 모델 — 템플릿 없음(raw 이어쓰기)
    LOCAL_DIR=/home/nntrainer/qwen3_e2e/gemma2_lg_q6k_qs4cx
    DEV_DIR=gemma2_qs4cx_regen    # bin md5는 LOCAL_DIR와 동일
    TMPL=raw ;;
  qwen3)                          # ChatML + 빈 <think> 필수
    LOCAL_DIR=/home/nntrainer/qwen3_e2e/qwen3_lg_q6k_qs4cx
    DEV_DIR=qwen3_qs4cx_regen     # bin md5는 LOCAL_DIR와 동일
    TMPL=chatml ;;
  *) echo "unknown model: $MODEL (gauss4|gauss4-ple|gemma4|gemma4-ple|gemma4-lmint4|gemma2|qwen3)"; exit 2 ;;
esac

# ---------- 프롬프트 구성 ---------------------------------------------------
wrap_prompt() { # $1=question
  local q="$1"
  case "$TMPL" in
    gemma4) printf '<bos><|turn>user\n%s<turn|>\n<|turn>model\n' "$q" ;;
    gauss4) printf '<bos><|turn>user\n%s<turn|>\n<|turn>model\n<|channel>thought\n<channel|>' "$q" ;;
    chatml) printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' "$q" ;;
    raw)    printf '%s' "$q" ;;
  esac
}

MODE=coh
if [ "$ARG3" = "perf" ]; then
  MODE=perf
elif [ -n "$ARG3" ]; then
  MODE=custom
fi

# 주의: $()는 끝 개행을 제거하므로 sentinel로 보존한다 — gemma4 템플릿은
# '<|turn>model\n'의 마지막 개행이 없으면 토큰이 하나 줄어 답이 훼손된다
# (", South Korea." 류). gauss4 템플릿은 원래 개행 없이 <channel|>로 끝난다.
case "$MODE" in
  coh)
    if [ "$TMPL" = raw ]; then Q="The capital of France is"
    else Q="What is the capital of South Korea?"; fi
    PROMPT=$(wrap_prompt "$Q"; printf x); PROMPT=${PROMPT%x} ;;
  custom)
    PROMPT=$(wrap_prompt "$ARG3"; printf x); PROMPT=${PROMPT%x} ;;
  perf)
    PROMPT="" ;;                  # 백엔드별로 파일에서 읽음(아래)
esac

# ---------- 백엔드 실행 ------------------------------------------------------
run_local() { # $1=builddir  $2...=env k=v들
  local BD="$REF_ROOT/$1"; shift
  local BIN="$BD/Applications/CausalLM/nntr_causallm"
  [ -x "$BIN" ] || { echo "binary not found: $BIN (ninja -C $BD Applications/CausalLM/nntr_causallm)"; exit 3; }
  local P="$PROMPT"
  [ "$MODE" = perf ] && P="$(cat "$PROMPT_1P2K")"
  cd "$REF_ROOT"
  env -u LD_LIBRARY_PATH "$@" timeout 600 "$BIN" "$LOCAL_DIR" "$P"
}

run_cuda() { # $1=extra env 유무(fast)
  local BD="$REF_ROOT/build_cuda"
  local BIN="$BD/Applications/CausalLM/nntr_causallm"
  [ -x "$BIN" ] || { echo "binary not found: $BIN"; exit 3; }
  local P="$PROMPT"
  [ "$MODE" = perf ] && P="$(cat "$PROMPT_1P2K")"
  local SAFE=(NNTR_ENGINE=cuda NNTR_CUDA_ROPE=1 NNTR_CUDA_ATTN=1 NNTR_CUDA_KV_UVM=1
              NNTR_CUDA_GEGLU=1 NNTR_CUDA_ELTWISE=1 NNTR_CUDA_QKNORM=1
              NNTR_CUDA_FLASH_DECODE=64 NNTR_CUDA_BLOCKQ=1 NNTR_FC_CUDA_CUBLAS=1
              NNTR_CUDA_PREWARM=1)
  local FAST=(NNTR_CUDA_DEV_ACT=1 NNTR_CUDA_VCOPY_PREFILL=1 NNTR_RMSNORM_CUDA_OFF=all
              NNTR_CUDA_M2B=1 NNTR_CUDA_ASYNC=1)
  cd "$REF_ROOT"
  export LD_LIBRARY_PATH="$BD/Applications/CausalLM:$BD/Applications/CausalLM/layers:$BD/nntrainer:$BD/api/ccapi:/usr/local/cuda/lib64"
  if [ "$1" = fast ]; then env "${SAFE[@]}" "${FAST[@]}" timeout 600 "$BIN" "$LOCAL_DIR" "$P"
  else env "${SAFE[@]}" timeout 600 "$BIN" "$LOCAL_DIR" "$P"; fi
}

run_adreno() { # $1=gpu|cpu
  adb -s "$ADRENO_SERIAL" get-state >/dev/null || { echo "device $ADRENO_SERIAL not connected"; exit 3; }
  local ENVV
  if [ "$1" = cpu ]; then
    ENVV="NNTR_ENGINE=cpu NNTR_NUM_THREADS=4 LD_LIBRARY_PATH=$DEVDIR"
  else
    ENVV="NNTR_NUM_THREADS=4 NNTR_FC_INT8_GPU=1 NNTR_MHA_GPU=1 NNTR_GPU_SVM_POOL=1 NNTR_KV_IMG_ATTN=1 NNTR_GPU_CLMEM_POOL=1 LD_LIBRARY_PATH=$DEVDIR"
  fi
  if [ "$MODE" = perf ]; then
    adb -s "$ADRENO_SERIAL" shell "[ -f $DEVDIR/prompt_1p2k.txt ]" \
      || adb -s "$ADRENO_SERIAL" push "$PROMPT_1P2K" "$DEVDIR/prompt_1p2k.txt" >/dev/null
    timeout 900 adb -s "$ADRENO_SERIAL" shell \
      "cd $DEVDIR && $ENVV ./nntrainer_causallm models/$DEV_DIR \"\$(cat prompt_1p2k.txt)\""
  else
    # 프롬프트를 임시 파일로 푸시(셸 이스케이프/개행 안전)
    local TF; TF=$(mktemp); printf '%s' "$PROMPT" > "$TF"
    adb -s "$ADRENO_SERIAL" push "$TF" "$DEVDIR/.run_llm_prompt.txt" >/dev/null; rm -f "$TF"
    timeout 900 adb -s "$ADRENO_SERIAL" shell \
      "cd $DEVDIR && $ENVV ./nntrainer_causallm models/$DEV_DIR \"\$(cat .run_llm_prompt.txt)\""
  fi
}

CL_ENV_COMMON=(NNTR_GPU_SVM_POOL=1 NNTR_V8C_BUF=1 NNTR_MHA_GPU=1 NNTR_FC_GPU=1
               NNTR_FC_INT8_GPU=1 NNTR_GPU_CLMEM_POOL=1)

case "$BACKEND" in
  xmx)        run_local build_cl   "${CL_ENV_COMMON[@]}" NNTR_FC_XMX=1 ;;
  intel)      run_local build_cl   "${CL_ENV_COMMON[@]}" NNTR_FC_XMX=0 ;;
  wrap)       run_local build_wrap "${CL_ENV_COMMON[@]}" NNTR_FC_XMX=1 ;;
  cuda)       run_cuda safe ;;
  cuda-fast)  run_cuda fast ;;
  adreno)     run_adreno gpu ;;
  adreno-cpu) run_adreno cpu ;;
  *) echo "unknown backend: $BACKEND (xmx|intel|wrap|cuda|cuda-fast|adreno|adreno-cpu)"; exit 2 ;;
esac

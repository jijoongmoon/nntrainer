#!/usr/bin/env bash
# measure_llm_mem.sh — run_llm.sh 래퍼: 호스트 RSS + Intel Xe GEM(per-process
# DRM fdinfo) + NVIDIA VRAM(per-process)을 함께 샘플링해 peak을 보고한다.
#
#   ./measure_llm_mem.sh <backend> <model> [perf|"question"]
#
# 배경: 매트릭스의 "peak memory"(호스트 RSS)는 GPU측 상주를 안 센다 —
# Intel iGPU의 GEM 버퍼(shmem-backed, cl_mem/v8c backing)와 NVIDIA VRAM
# (cudaMalloc 파생 캐시)은 여기 안 잡힌다. iGPU는 물리적으로 같은 RAM을
# 쓰므로 "시스템이 진짜 얼마나 쓰나"는 RSS+GEM을 같이 봐야 한다.
# (Windows WDDM은 반대로 이것들을 WS에 합산해서 보여준다 — 크로스 OS 비교
# 금지, same-box A/B 전용.)
#
# 샘플 소스 (전부 비루트):
#   RSS  : /proc/$PID/status VmRSS (+VmHWM 최종)
#   GEM  : /proc/$PID/fdinfo/* 의 drm-driver:xe 항목의 drm-total-system /
#          drm-resident-system (xe DRM fdinfo memory stats)
#   VRAM : nvidia-smi --query-compute-apps (해당 PID 행)
set -u
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN="$SELF_DIR/run_llm.sh"
OUT="${MEASURE_OUT:-/tmp/measure_llm_mem.$$}"
mkdir -p "$OUT"

"$RUN" "$@" > "$OUT/run.log" 2>&1 &
RUNNER=$!

# 실제 바이너리 PID 대기 (최대 30초)
PID=""
for i in $(seq 1 300); do
  for c in $(pgrep -f "nntr_causallm|nntrainer_causallm" 2>/dev/null); do
    case "$(cat /proc/$c/comm 2>/dev/null)" in
      nntr_causallm|nntrainer_causal*) PID=$c; break;;
    esac
  done
  [ -n "$PID" ] && break
  sleep 0.1
done
if [ -z "$PID" ]; then
  echo "binary not found; runner output:" >&2
  tail -5 "$OUT/run.log" >&2
  wait $RUNNER
  exit 3
fi

gem_kb() { # per-process xe GEM(KB): xe drm fd의 drm-<field>-{system,gtt,vram*} 합.
  # 값은 "N KiB|MiB|GiB" 형식 — KB로 정규화. fd가 여럿이면 최대값(동일 클라이언트 중복).
  local pid=$1 field=$2 best=0 v
  for f in /proc/$pid/fdinfo/*; do
    grep -q "drm-driver:.*xe" "$f" 2>/dev/null || continue
    v=$(awk -v k="^drm-$field-" 'BEGIN{s=0}
         $1 ~ k {n=$2; u=$3;
                 if (u=="MiB") n*=1024; else if (u=="GiB") n*=1048576;
                 else if (u!="KiB" && u!="") n=0;
                 s+=n}
         END{printf "%d", s}' "$f" 2>/dev/null)
    [ -n "$v" ] && [ "$v" -gt "$best" ] 2>/dev/null && best=$v
  done
  echo $best
}

RSS_PEAK=0; GEMR_PEAK=0; VRAM_PEAK=0
while kill -0 $PID 2>/dev/null; do
  R=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null) || break
  [ -n "$R" ] && [ "$R" -gt "$RSS_PEAK" ] && RSS_PEAK=$R
  GR=$(gem_kb $PID resident); [ "$GR" -gt "$GEMR_PEAK" ] && GEMR_PEAK=$GR
  V=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v p=$PID '$1==p{print $2; exit}')
  [ -n "${V:-}" ] && [ "$V" -gt "$VRAM_PEAK" ] 2>/dev/null && VRAM_PEAK=$V
  sleep 0.2
done
wait $RUNNER

echo "==================[ measure_llm_mem ]=================="
grep -E "prefill:|generation:|peak memory|Seoul" "$OUT/run.log" | head -6
printf "sampler peaks : RSS %d KB (%.0f MB) | XeGEM resident %d KB (%.0f MB) | NV VRAM %d MiB | RSS+GEM %.0f MB\n" \
  "$RSS_PEAK" "$(echo "$RSS_PEAK/1024" | bc -l)" \
  "$GEMR_PEAK" "$(echo "$GEMR_PEAK/1024" | bc -l)" \
  "$VRAM_PEAK" "$(echo "($RSS_PEAK+$GEMR_PEAK)/1024" | bc -l)"
echo "run log: $OUT/run.log"

#!/usr/bin/env bash
# =============================================================================
# nntrainer 이사 패킹 (2026-06-19, "코드/메모리만" 슬림판)
#   repo는 git(이미 push됨), x86 모델은 단말 adb-pull → 패키지엔 안 넣음.
#   결과: $OUT/  (migrate_setup.sh + 두 tar + MANIFEST + MD5SUMS), 수십 MB.
#
# 사용: .claude/scripts/migrate_pack.sh [출력디렉토리=/home/myungjoo/migrate]
#   INCLUDE_TRANSCRIPT=<session-uuid> 주면 해당 transcript도 동봉(literal resume용).
#
# 새 머신: 이 디렉토리째 옮긴 뒤  bash migrate_setup.sh [NEW_BASE] [NDK] [SERIAL]
# =============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-/home/myungjoo/migrate}"
PROJ_BASE="$HOME/.claude/projects"
PROJ="-home-myungjoo-nntrainer"
mkdir -p "$OUT"

echo "[1/5] repo .claude 툴링 (presentation/대형로그 제외) ..."
tar -C "$REPO" -czf "$OUT/claude_tooling.tar.gz" \
  --exclude='.claude/presentation' \
  --exclude='.claude/salvage' \
  --exclude='.claude/*.log' \
  --exclude='.claude/scheduled_tasks.lock' \
  .claude

echo "[2/5] Claude 세션 메모리 (memory/ 만 — 매 세션 자동 로드분) ..."
# leading-dash 경로($PROJ)는 tar가 옵션으로 오해 → ./ 접두
tar -C "$PROJ_BASE" -czf "$OUT/claude_memory.tar.gz" "./$PROJ/memory"
if [ -n "${INCLUDE_TRANSCRIPT:-}" ] && [ -f "$PROJ_BASE/$PROJ/$INCLUDE_TRANSCRIPT.jsonl" ]; then
  echo "      + transcript $INCLUDE_TRANSCRIPT (literal resume용)"
  if [ -d "$PROJ_BASE/$PROJ/$INCLUDE_TRANSCRIPT" ]; then
    tar -C "$PROJ_BASE" -czf "$OUT/claude_transcript.tar.gz" \
      "./$PROJ/$INCLUDE_TRANSCRIPT.jsonl" "./$PROJ/$INCLUDE_TRANSCRIPT"
  else
    tar -C "$PROJ_BASE" -czf "$OUT/claude_transcript.tar.gz" "./$PROJ/$INCLUDE_TRANSCRIPT.jsonl"
  fi
fi

echo "[3/5] migrate_setup.sh + MIGRATION.md 동봉 ..."
cp "$REPO/.claude/scripts/migrate_setup.sh" "$OUT/"
cp "$REPO/.claude/MIGRATION.md" "$OUT/" 2>/dev/null || true

# git bundle: carries the LOCAL commits that are not on origin/main (the unpushed
# session work — e.g. the CUDA backend). The new machine clones origin/main from
# github (the prereq) then applies this delta, so jijoongmoon push access is NOT
# required to continue. Tiny (delta only). session.info records the tip to reset.
echo "      + session.bundle (unpushed commits, delta over origin/main) ..."
SESS_TIP="$(git -C "$REPO" rev-parse HEAD)"
SESS_BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
SESS_PREREQ="$(git -C "$REPO" rev-parse origin/main 2>/dev/null || echo '')"
if git -C "$REPO" bundle create "$OUT/session.bundle" HEAD --not origin/main 2>/dev/null; then
  { echo "SESS_TIP=$SESS_TIP"; echo "SESS_BRANCH=$SESS_BRANCH"; echo "SESS_PREREQ=$SESS_PREREQ"; \
    echo "AHEAD=$(git -C "$REPO" rev-list --count origin/main..HEAD 2>/dev/null)"; } > "$OUT/session.info"
  echo "        $(du -h "$OUT/session.bundle" | cut -f1), tip=$(git -C "$REPO" rev-parse --short HEAD), ahead=$(git -C "$REPO" rev-list --count origin/main..HEAD 2>/dev/null)"
else
  echo "        ⚠ bundle 생성 실패(origin/main 부재?) — 새 머신은 github clone에 의존" >&2
fi

echo "[4/5] MANIFEST.txt 생성 ..."
{
  echo "# nntrainer GPU CausalLM 이사 MANIFEST"
  echo
  echo "[git]"
  echo "branch  : $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  echo "HEAD    : $(git -C "$REPO" rev-parse HEAD)"
  echo "on top of origin/main: $(git -C "$REPO" rev-parse --short origin/main 2>/dev/null || echo '?')"
  echo "remotes :"
  git -C "$REPO" remote -v | sed 's/^/  /'
  echo
  echo "[device]  serial=R3CY70LV96T (S26U/Adreno840), root=/data/local/tmp/nntrainer/causallm"
  echo "[toolchain] NDK 27.2.12479018 (~/Android/Sdk/ndk/), meson+ninja, Intel NEO ICD(libOpenCL.so.1)"
  echo
  echo "[models]  device_dir : local_dir (setup가 adb-pull + tokenizer 경로 보정)"
  echo "  gemma4_lmint4 : gemma4_e2b_qint4fp16_lmint4   (QINT4-FP16, untied int4 lm_head)"
  echo "  gemma2_lg_q6k : gemma2_lg_q6k                 (QINT4-FP16, Q6_K lm_head)"
  echo "  qwen3_lg_q6k  : qwen3_lg_q6k                  (QINT4-FP16, Q6_K lm_head)"
  echo
  echo "[Intel env]  NNTR_GPU_SVM_POOL=1 NNTR_V8C_BUF=1 NNTR_MHA_GPU=1 NNTR_FC_GPU=1 NNTR_FC_INT8_GPU=1 NNTR_GPU_CLMEM_POOL=1"
  echo "[Adreno env] NNTR_FC_INT8_GPU=1 NNTR_MHA_GPU=1 NNTR_GPU_SVM_POOL=1 NNTR_KV_IMG_ATTN=1 NNTR_GPU_CLMEM_POOL=1"
  echo
  echo "[deploy 6 artifact -> device]  (libccapi-nntrainer.so 가 #1 자주 누락)"
  echo "  nntrainer_causallm, libcausallm_core.so      (jni/obj/local/arm64-v8a)"
  echo "  libccapi-nntrainer.so, libnntrainer.so       (builddir/android_build_result/lib/arm64-v8a)"
  echo "  libOpenCL.so                                 (builddir/opencl/lib/arm64-v8a)"
  echo "  libc++_shared.so                             (NDK)"
  echo
  echo "[build 함정]"
  echo "  Intel: meson에 -Denable-transformer=true 필수 (없으면 nntr_causallm 타깃 부재)"
  echo "  Android lib: package_android.sh -Denable-opencl=true (기본 off면 GPU 심볼 없음)"
  echo "  검증 1K perf: Adreno gemma4 2473/15.6 gemma2 819/14.0 qwen3 2151/22.0"
  echo "               Intel  gemma4 ~1600/5.17 gemma2 690/7.79 qwen3 1936/9.35"
  echo "  상세: Applications/CausalLM/README_GPU.md"
} > "$OUT/MANIFEST.txt"

echo "[5/6] MD5SUMS ..."
( cd "$OUT" && md5sum claude_tooling.tar.gz claude_memory.tar.gz \
    $( [ -f claude_transcript.tar.gz ] && echo claude_transcript.tar.gz ) \
    $( [ -f session.bundle ] && echo session.bundle ) \
    $( [ -f session.info ] && echo session.info ) \
    migrate_setup.sh > MD5SUMS )

echo "[6/6] 단말 적재 (운반체) ..."
# S26U(R3CY70LV96T) 우선 — Note20(R3CN80CW3FY)이 같이 꽂혀 있어도 안전
DEV="${DEVICE:-}"
if [ -z "$DEV" ]; then
  if adb devices 2>/dev/null | grep -qw "R3CY70LV96T"; then DEV="R3CY70LV96T"
  else DEV="$(adb devices 2>/dev/null | awk 'NR>1 && $2=="device"{print $1; exit}')"; fi
fi
DEV="${DEV:-R3CY70LV96T}"
DEV_PKG="/data/local/tmp/nntrainer/migrate"
if adb -s "$DEV" get-state >/dev/null 2>&1; then
  adb -s "$DEV" shell "mkdir -p $DEV_PKG" >/dev/null 2>&1
  for f in claude_tooling.tar.gz claude_memory.tar.gz migrate_setup.sh MANIFEST.txt MD5SUMS \
           $( [ -f "$OUT/session.bundle" ] && echo session.bundle ) \
           $( [ -f "$OUT/session.info" ] && echo session.info ) \
           $( [ -f "$OUT/claude_transcript.tar.gz" ] && echo claude_transcript.tar.gz ); do
    adb -s "$DEV" push "$OUT/$f" "$DEV_PKG/$f" >/dev/null 2>&1 && echo "  → $DEV_PKG/$f"
  done
  echo "단말($DEV) 적재 완료: $DEV_PKG"
else
  echo "⚠ 단말 미연결 — 단말 적재 생략. 수동: adb push $OUT/* $DEV_PKG/"
fi

echo
echo "패키지 완료: $OUT  (단말: $DEV_PKG)"
ls -lh "$OUT"
cat <<EOF

새 머신에서 (둘 중 하나):
  (A) 스크립트만 pull → 나머지는 스크립트가 단말서 자동 pull:
      adb pull $DEV_PKG/migrate_setup.sh .  &&  bash migrate_setup.sh
  (B) 패키지 통째로 pull 후 실행:
      adb pull $DEV_PKG ~/migrate  &&  bash ~/migrate/migrate_setup.sh
  인자: bash migrate_setup.sh [NEW_BASE=\$HOME] [NDK경로] [단말serial]
EOF

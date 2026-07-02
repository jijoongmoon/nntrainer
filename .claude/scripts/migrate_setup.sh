#!/usr/bin/env bash
# =============================================================================
# nntrainer GPU CausalLM — 새 머신 단일 setup 스크립트 (2026-06-19)
#
#   하나만 실행하면: repo clone → .claude 툴링/세션메모리 복원 → 단말서 모델 pull
#   + config 경로 보정 → Intel(build_cl) 빌드 → Android 빌드 → 단말 6-artifact 배포
#   → 양 플랫폼 smoke test.  ※ 경로는 NEW_BASE에서 전부 파생 (dir path 변경 OK).
#
# 사용법:
#   bash migrate_setup.sh [NEW_BASE] [ANDROID_NDK] [DEVICE_SERIAL]
#     NEW_BASE      : repo와 qwen3_e2e를 둘 상위 디렉토리 (기본: $HOME)
#                     → repo = $NEW_BASE/nntrainer , 모델 = $NEW_BASE/qwen3_e2e
#     ANDROID_NDK   : NDK 경로 (기본: ~/Android/Sdk/ndk/* 자동탐지)
#     DEVICE_SERIAL : adb 단말 (기본: 자동, 없으면 R3CY70LV96T)
#
#   플래그(환경변수=1): SKIP_CLONE SKIP_MEMORY SKIP_MODELS SKIP_INTEL
#                       SKIP_ANDROID SKIP_DEPLOY SKIP_TEST SKIP_CLAUDE
#
#   ⭐ 새 머신 최단 경로 (단말만 연결돼 있으면):
#       adb pull /data/local/tmp/nntrainer/migrate/migrate_setup.sh .
#       bash migrate_setup.sh            # 나머지 tar·모델은 단말서 자동 pull
#
# 이 스크립트는 자신이 들어있는 migration 패키지 디렉토리(=$PKG)에서
#   claude_tooling.tar.gz, claude_memory.tar.gz, MANIFEST.txt 를 찾는다.
# =============================================================================
set -uo pipefail

PKG="$(cd "$(dirname "$0")" && pwd)"
NEW_BASE="${1:-$HOME}"
ANDROID_NDK_IN="${2:-}"
DEV="${3:-}"

REPO="$NEW_BASE/nntrainer"
E2E="$NEW_BASE/qwen3_e2e"
GIT_URL="${GIT_URL:-https://github.com/jijoongmoon/nntrainer.git}"
BRANCH="${BRANCH:-gpu/v8c-on-main}"
PIN="${PIN:-f5a5d2d6}"          # 검증된 커밋 (이사 시점). 비우면 브랜치 tip.
DEV_ROOT="/data/local/tmp/nntrainer/causallm"
DEV_PKG="/data/local/tmp/nntrainer/migrate"   # 단말에 적재된 마이그레이션 패키지
OLD_BASE="/home/myungjoo"        # 패키지가 만들어진 원래 경로 (치환 대상)

# device_model_dir : local_model_dir  (단말→로컬, 이름이 다름에 주의)
MODELS=(
  "gemma4_lmint4:gemma4_e2b_qint4fp16_lmint4"
  "gemma2_lg_q6k:gemma2_lg_q6k"
  "qwen3_lg_q6k:qwen3_lg_q6k"
)

INTEL_ENV="NNTR_GPU_SVM_POOL=1 NNTR_V8C_BUF=1 NNTR_MHA_GPU=1 NNTR_FC_GPU=1 NNTR_FC_INT8_GPU=1 NNTR_GPU_CLMEM_POOL=1"
ADRENO_ENV="NNTR_NUM_THREADS=4 NNTR_FC_INT8_GPU=1 NNTR_MHA_GPU=1 NNTR_GPU_SVM_POOL=1 NNTR_KV_IMG_ATTN=1 NNTR_GPU_CLMEM_POOL=1"

c(){ printf '\n\033[1;36m===== %s =====\033[0m\n' "$*"; }
ok(){ printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn(){ printf '  \033[33m⚠\033[0m %s\n' "$*"; }
die(){ printf '\n\033[31m✗ %s\033[0m\n' "$*"; exit 1; }
flag(){ eval "[ \"\${$1:-0}\" = 1 ]"; }

# ---- 0. preflight --------------------------------------------------------
c "[0] preflight"
need(){ command -v "$1" >/dev/null 2>&1 && ok "$1: $(command -v "$1")" || warn "$1 없음 — $2"; }
need git    "sudo apt install git"
need meson  "sudo apt install meson (Intel 빌드 필수)"
need ninja  "sudo apt install ninja-build"
need adb    "platform-tools (단말 모델 pull/배포 필수)"
need python3 "config 경로 보정에 사용"
# NDK 자동탐지
if [ -z "$ANDROID_NDK_IN" ]; then
  ANDROID_NDK_IN="$(ls -d "$HOME"/Android/Sdk/ndk/* 2>/dev/null | sort -V | tail -1)"
fi
[ -n "$ANDROID_NDK_IN" ] && [ -x "$ANDROID_NDK_IN/ndk-build" ] \
  && ok "NDK: $ANDROID_NDK_IN" || warn "NDK 미탐지 — Android 빌드는 SKIP_ANDROID=1 또는 인자2로 지정"
# 단말 자동탐지 — S26U(R3CY70LV96T) 우선 (Note20 R3CN80CW3FY 혼선 방지)
if [ -z "$DEV" ]; then
  if adb devices 2>/dev/null | grep -qw "R3CY70LV96T"; then DEV="R3CY70LV96T"
  else DEV="$(adb devices 2>/dev/null | awk 'NR>1 && $2=="device"{print $1; exit}')"; fi
  [ -z "$DEV" ] && DEV="R3CY70LV96T"
fi
ok "NEW_BASE=$NEW_BASE  REPO=$REPO  E2E=$E2E"
ok "DEVICE=$DEV"

# ---- 0.5 bootstrap: tar가 스크립트 옆에 없으면 단말에서 패키지 통째 pull ----
if [ ! -f "$PKG/claude_memory.tar.gz" ]; then
  c "[0.5] 패키지 bootstrap (단말 $DEV_PKG → 로컬)"
  if adb -s "$DEV" get-state >/dev/null 2>&1 && \
     adb -s "$DEV" shell "[ -f $DEV_PKG/claude_memory.tar.gz ]" 2>/dev/null; then
    LOCALPKG="${MIG_DIR:-$HOME/migrate}"; mkdir -p "$LOCALPKG"
    adb -s "$DEV" pull "$DEV_PKG/." "$LOCALPKG/" >/dev/null 2>&1 || die "단말 패키지 pull 실패"
    PKG="$LOCALPKG"; ok "패키지 pull 완료: $PKG"
  else
    die "패키지 tar 없음 & 단말($DEV) $DEV_PKG 에도 없음 — 원본 머신서 migrate_pack.sh 먼저 실행"
  fi
else ok "패키지 발견: $PKG"; fi
[ -f "$PKG/MD5SUMS" ] && ( cd "$PKG" && md5sum -c MD5SUMS >/dev/null 2>&1 \
  && ok "MD5 검증 OK" || warn "MD5 불일치/일부 누락(계속 진행)" )

# ---- 1. repo clone -------------------------------------------------------
c "[1] repo clone"
if flag SKIP_CLONE; then warn "SKIP_CLONE"
elif [ -d "$REPO/.git" ]; then ok "이미 존재: $REPO (clone 건너뜀)"
else
  git clone "$GIT_URL" "$REPO" || die "clone 실패: $GIT_URL"
  git -C "$REPO" checkout "$BRANCH" || die "checkout $BRANCH 실패"
  [ -n "$PIN" ] && { git -C "$REPO" rev-parse --short HEAD | grep -q "^$PIN" \
    || warn "tip != $PIN (브랜치가 앞서감 — 의도면 OK)"; }
  ok "clone+checkout $BRANCH @ $(git -C "$REPO" rev-parse --short HEAD)"
fi

# ---- 1.5 session bundle 적용 (github clone엔 없는 unpushed 커밋) --------
if [ -f "$PKG/session.bundle" ] && [ -d "$REPO/.git" ] && ! flag SKIP_CLONE; then
  c "[1.5] session bundle 적용 (unpushed 커밋)"
  SESS_TIP=""; SESS_PREREQ=""; AHEAD=""
  [ -f "$PKG/session.info" ] && . "$PKG/session.info" 2>/dev/null || true
  if [ -n "$SESS_PREREQ" ] && ! git -C "$REPO" cat-file -e "$SESS_PREREQ" 2>/dev/null; then
    warn "bundle prereq $SESS_PREREQ 부재 — origin fetch 시도"
    git -C "$REPO" fetch --quiet origin 2>/dev/null || true
  fi
  if git -C "$REPO" fetch --quiet "$PKG/session.bundle" HEAD 2>/dev/null; then
    git -C "$REPO" reset --hard "${SESS_TIP:-FETCH_HEAD}" >/dev/null 2>&1 || die "bundle reset 실패"
    NOW="$(git -C "$REPO" rev-parse HEAD)"
    { [ -z "$SESS_TIP" ] || [ "$NOW" = "$SESS_TIP" ]; } \
      && ok "bundle 적용 OK: HEAD=$(git -C "$REPO" rev-parse --short HEAD) (+${AHEAD:-?} commits over origin/main)" \
      || warn "bundle 적용 후 HEAD($NOW) != 기대($SESS_TIP)"
  else warn "bundle fetch 실패 — github clone 상태로 진행(unpushed 커밋 누락)"; fi
fi

# ---- 2. .claude 툴링 복원 + 경로 치환 -----------------------------------
c "[2] .claude 툴링 복원"
if [ -f "$PKG/claude_tooling.tar.gz" ]; then
  mkdir -p "$REPO"
  tar -xzf "$PKG/claude_tooling.tar.gz" -C "$REPO"   # .claude/ 를 repo 루트에 풂
  # 호스트 스크립트의 구 경로 → 새 경로 (NDK 경로도)
  for f in "$REPO"/.claude/scripts/*.sh; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in migrate_setup.sh|migrate_pack.sh) continue;; esac
    [ -n "$ANDROID_NDK_IN" ] && sed -i "s|$OLD_BASE/Android/Sdk/ndk/[^\"' ]*|$ANDROID_NDK_IN|g" "$f"
    sed -i "s|$OLD_BASE|$NEW_BASE|g" "$f"
  done
  ok ".claude 복원 + 경로 치환 ($OLD_BASE → $NEW_BASE)"
else warn "claude_tooling.tar.gz 없음 — .claude 복원 건너뜀"; fi

# ---- 3. Claude 세션 메모리 설치 (프로젝트 키 = repo 경로 파생) -----------
c "[3] Claude 세션 메모리"
if flag SKIP_MEMORY; then warn "SKIP_MEMORY"
elif [ -f "$PKG/claude_memory.tar.gz" ]; then
  KEY="$(echo "$REPO" | tr '/' '-')"     # Claude Code의 프로젝트 디렉토리 키
  DEST="$HOME/.claude/projects/$KEY"
  mkdir -p "$HOME/.claude/projects"
  if [ -d "$DEST" ]; then warn "이미 존재: $DEST (메모리 병합 안 함)"; else
    tmp="$(mktemp -d)"; tar -xzf "$PKG/claude_memory.tar.gz" -C "$tmp"
    src="$(ls -d "$tmp"/-home-myungjoo-nntrainer "$tmp"/* 2>/dev/null | head -1)"
    mv "$src" "$DEST"; rm -rf "$tmp"
    ok "메모리 설치: $DEST ($(ls "$DEST/memory"/*.md 2>/dev/null | wc -l) memory 파일)"
  fi
else warn "claude_memory.tar.gz 없음"; fi

# ---- 4. 단말서 x86 모델 pull + config 경로 보정 -------------------------
c "[4] 모델 pull (단말 → $E2E) + tokenizer 경로 보정"
if flag SKIP_MODELS; then warn "SKIP_MODELS"
elif ! adb -s "$DEV" get-state >/dev/null 2>&1; then warn "단말 $DEV 미연결 — 모델 pull 건너뜀"
else
  mkdir -p "$E2E"
  for m in "${MODELS[@]}"; do
    dev_d="${m%%:*}"; loc_d="${m##*:}"
    if [ -d "$E2E/$loc_d" ]; then ok "이미 존재: $E2E/$loc_d"; continue; fi
    if ! adb -s "$DEV" shell "[ -d $DEV_ROOT/models/$dev_d ]" 2>/dev/null; then
      warn "단말에 $dev_d 없음 — 건너뜀"; continue; fi
    echo "  pull $dev_d → $loc_d ..."
    adb -s "$DEV" pull "$DEV_ROOT/models/$dev_d" "$E2E/$loc_d" >/dev/null 2>&1 \
      || { warn "pull 실패: $dev_d"; continue; }
    # adb pull 이 한 단계 더 만들면 평탄화
    [ -d "$E2E/$loc_d/$dev_d" ] && { mv "$E2E/$loc_d/$dev_d"/* "$E2E/$loc_d/" 2>/dev/null; rmdir "$E2E/$loc_d/$dev_d" 2>/dev/null; }
    cfg="$E2E/$loc_d/nntr_config.json"
    if [ -f "$cfg" ] && command -v python3 >/dev/null; then
      python3 - "$cfg" "$E2E/$loc_d/tokenizer.json" <<'PY'
import json,sys
cfg,tok=sys.argv[1],sys.argv[2]
d=json.load(open(cfg))
if "tokenizer_file" in d: d["tokenizer_file"]=tok
json.dump(d,open(cfg,"w"))
PY
      ok "$loc_d 받음 + tokenizer_file → 로컬"
    else warn "$loc_d config/python 보정 생략"; fi
  done
fi

# ---- 5. Intel(x86) build_cl --------------------------------------------
c "[5] Intel build_cl"
if flag SKIP_INTEL; then warn "SKIP_INTEL"
elif command -v meson >/dev/null && command -v ninja >/dev/null; then
  cd "$REPO"
  [ -d build_cl ] || meson setup build_cl . -Denable-opencl=true -Denable-fp16=true \
      -Denable-transformer=true -Denable-clblast=true -Dwerror=false --buildtype=release \
      || die "meson setup 실패"
  meson configure build_cl -Denable-transformer=true >/dev/null 2>&1 || true
  ninja -C build_cl Applications/CausalLM/nntr_causallm \
    && ok "build_cl OK: $(ls build_cl/Applications/CausalLM/nntr_causallm)" \
    || warn "build_cl 실패 (의존성: openblas/jsoncpp/libcurl/opencl-headers + Intel NEO ICD)"
else warn "meson/ninja 없음 — Intel 빌드 건너뜀"; fi

# ---- 6. Android 빌드 (lib + app) ---------------------------------------
c "[6] Android 빌드"
if flag SKIP_ANDROID; then warn "SKIP_ANDROID"
elif [ -z "$ANDROID_NDK_IN" ] || [ ! -x "$ANDROID_NDK_IN/ndk-build" ]; then warn "NDK 없음 — Android 빌드 건너뜀"
else
  cd "$REPO"
  if [ ! -f builddir/android_build_result/lib/arm64-v8a/libnntrainer.so ]; then
    echo "  lib 빌드 (package_android, opencl on) ..."
    ANDROID_NDK="$ANDROID_NDK_IN" ./tools/package_android.sh \
      -Denable-opencl=true -Denable-clblast=false -Dwerror=false \
      >/tmp/mig_android_lib.log 2>&1 \
      && ok "lib OK (libnntrainer+libccapi+libOpenCL)" \
      || { warn "lib 빌드 실패 — tail /tmp/mig_android_lib.log"; tail -5 /tmp/mig_android_lib.log; }
  else ok "lib 이미 존재"; fi
  echo "  app 빌드 (ndk-build) ..."
  ( cd Applications/CausalLM/jni && rm -rf libs obj && \
    PATH="$ANDROID_NDK_IN:$PATH" ndk-build NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj \
      APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk \
      causallm_core nntrainer_causallm -j"$(nproc)" >/tmp/mig_android_app.log 2>&1 ) \
    && ok "app OK (libcausallm_core + nntrainer_causallm)" \
    || { warn "app 빌드 실패 — tail /tmp/mig_android_app.log"; tail -5 /tmp/mig_android_app.log; }
fi

# ---- 7. 단말 배포 (6 artifact — libccapi 잊지 말 것) --------------------
c "[7] 단말 배포 (6 artifact)"
if flag SKIP_DEPLOY; then warn "SKIP_DEPLOY"
elif ! adb -s "$DEV" get-state >/dev/null 2>&1; then warn "단말 미연결 — 배포 건너뜀"
else
  OBJ="$REPO/Applications/CausalLM/jni/obj/local/arm64-v8a"
  LIB="$REPO/builddir/android_build_result/lib/arm64-v8a"
  CLLIB="$(find "$ANDROID_NDK_IN" -path '*aarch64-linux-android/libc++_shared.so' 2>/dev/null | head -1)"
  adb -s "$DEV" shell "mkdir -p $DEV_ROOT" 2>/dev/null
  push(){ [ -f "$1" ] && adb -s "$DEV" push "$1" "$DEV_ROOT/$2" >/dev/null 2>&1 && ok "push $2" || warn "없음/실패: $2 ($1)"; }
  push "$OBJ/nntrainer_causallm"      nntrainer_causallm
  push "$OBJ/libcausallm_core.so"     libcausallm_core.so
  push "$LIB/libccapi-nntrainer.so"   libccapi-nntrainer.so        # ⚠ #1 자주 누락
  push "$LIB/libnntrainer.so"         libnntrainer.so
  push "$REPO/builddir/opencl/lib/arm64-v8a/libOpenCL.so" libOpenCL.so
  push "$CLLIB"                        libc++_shared.so
  adb -s "$DEV" shell "chmod 755 $DEV_ROOT/nntrainer_causallm" 2>/dev/null
fi

# ---- 8. smoke test -------------------------------------------------------
c "[8] smoke test (coherence)"
if flag SKIP_TEST; then warn "SKIP_TEST"; else
  BIN="$REPO/build_cl/Applications/CausalLM/nntr_causallm"
  if [ -x "$BIN" ] && [ -d "$E2E/gemma4_e2b_qint4fp16_lmint4" ]; then
    echo "  Intel gemma4 ..."
    env $INTEL_ENV "$BIN" "$E2E/gemma4_e2b_qint4fp16_lmint4" 2>/dev/null \
      | grep -i "Seoul\|capital" | head -1 && ok "Intel coherent" || warn "Intel 출력 확인 필요"
  else warn "Intel smoke 건너뜀 (바이너리/모델 미비)"; fi
  if adb -s "$DEV" get-state >/dev/null 2>&1; then
    echo "  Adreno gemma4 ..."
    adb -s "$DEV" shell "cd $DEV_ROOT && export LD_LIBRARY_PATH=\$PWD && $ADRENO_ENV ./nntrainer_causallm models/gemma4_lmint4 2>/dev/null" \
      | grep -i "Seoul" | head -1 && ok "Adreno coherent" || warn "Adreno 출력 확인 필요"
  fi
fi

# ---- 9. Claude Code CLI 설치 (새 머신엔 없음) ---------------------------
c "[9] Claude Code CLI"
if flag SKIP_CLAUDE; then warn "SKIP_CLAUDE"
elif command -v claude >/dev/null 2>&1; then ok "이미 설치됨: $(claude --version 2>/dev/null || command -v claude)"
else
  installed=0
  if command -v curl >/dev/null 2>&1; then
    echo "  native installer (claude.ai/install.sh) ..."
    curl -fsSL https://claude.ai/install.sh | bash >/tmp/mig_claude_install.log 2>&1 && installed=1 \
      || warn "native installer 실패 (tail /tmp/mig_claude_install.log)"
  fi
  if [ "$installed" = 0 ] && command -v npm >/dev/null 2>&1; then
    echo "  npm install -g @anthropic-ai/claude-code ..."
    npm install -g @anthropic-ai/claude-code >/tmp/mig_claude_install.log 2>&1 && installed=1 || warn "npm 설치 실패"
  fi
  export PATH="$HOME/.local/bin:$PATH"
  if command -v claude >/dev/null 2>&1; then
    ok "설치됨: $(claude --version 2>/dev/null)"
    grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null \
      || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  elif [ "$installed" = 1 ]; then
    warn "설치됨(PATH 미반영) — 새 셸: export PATH=\$HOME/.local/bin:\$PATH"
  else
    warn "curl/npm 둘 다 실패 — 수동: curl -fsSL https://claude.ai/install.sh | bash"
  fi
fi

c "완료"
cat <<EOF
repo     : $REPO  ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null))
models   : $E2E
device   : $DEV ($DEV_ROOT)
Intel run: cd $REPO && $INTEL_ENV ./build_cl/Applications/CausalLM/nntr_causallm <model_dir> "프롬프트"
Adreno   : adb -s $DEV shell 'cd $DEV_ROOT && LD_LIBRARY_PATH=\$PWD $ADRENO_ENV ./nntrainer_causallm models/<dir> "프롬프트"'
문서      : $REPO/Applications/CausalLM/README_GPU.md

▶ 세션 이어서:  cd $REPO && claude
   - 첫 실행 시 인증 필요(자격증명은 이사 안 됨): 브라우저 로그인,
     또는 헤드리스 → export CLAUDE_CODE_OAUTH_TOKEN=...  (원본 머신서 'claude setup-token')
     또는 export ANTHROPIC_API_KEY=sk-ant-...
   - 메모리 자동 로드: ~/.claude/projects/$(echo "$REPO" | tr '/' '-')/memory/MEMORY.md
EOF

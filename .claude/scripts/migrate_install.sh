#!/usr/bin/env bash
# ============================================================
# nntrainer 작업환경 자동 이주 설치기 (2026-06-12)
#
# 새 PC에서:
#   adb pull /sdcard/migrate ~/migrate
#   bash ~/migrate/migrate_install.sh <NEW_BASE> [NDK_PATH] [--no-build]
#
# 예) repo를 /home/jijoongmoon/work/nntrainer 로 두려면:
#   bash ~/migrate/migrate_install.sh /home/jijoongmoon/work
#
# 하는 일: md5 검증 → tar 해제 → Claude 메모리 설치(+키 변경) →
#          경로 치환(스크립트/config) → 죽은 빌드트리 정리(+opencl 샐비지) →
#          x86 빌드 재구성+핵심 타깃 빌드 → 잔여 수동단계 출력
# ============================================================
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"   # tar들이 있는 디렉토리 (~/migrate)
NEW=${1:?"사용법: migrate_install.sh <NEW_BASE> [NDK_PATH] [--no-build]"}
NDK=${2:-}
NOBUILD=0
for a in "$@"; do [ "$a" = "--no-build" ] && NOBUILD=1; done
[ -n "$NDK" ] && [ "$NDK" = "--no-build" ] && NDK=""
OLD=/home/nntrainer
REPO="$NEW/nntrainer"
E2E="$NEW/qwen3_e2e"

step() { echo; echo "===== $* ====="; }

step "[0/7] 무결성 검증"
cd "$SRC"
md5sum -c MD5SUMS

step "[1/7] tar 해제 → $NEW"
mkdir -p "$NEW"
if [ -d "$REPO" ]; then
  echo "⚠ $REPO 가 이미 존재 — 해제 건너뜀 (직접 정리 후 재실행)"
else
  tar -xf nntrainer_repo.tar -C "$NEW"
fi
if [ -d "$E2E" ]; then
  echo "⚠ $E2E 가 이미 존재 — 해제 건너뜀"
else
  tar -xf qwen3_e2e.tar -C "$NEW"
fi

step "[2/7] 최신 이주 파일을 repo에 반영 (tar보다 새 버전)"
mkdir -p "$REPO/.claude/scripts"
for f in migrate_install.sh migrate_fixup.sh MIGRATION.md; do
  if [ -f "$SRC/$f" ]; then
    case "$f" in
      MIGRATION.md) cp "$SRC/$f" "$REPO/.claude/$f" ;;
      *) cp "$SRC/$f" "$REPO/.claude/scripts/$f" ;;
    esac
  fi
done

step "[3/7] Claude 메모리 설치 + 프로젝트 키 변경"
KEY=$(echo "$REPO" | tr '/' '-')
mkdir -p "$HOME/.claude/projects"
if [ -d "$HOME/.claude/projects/$KEY" ]; then
  echo "⚠ $HOME/.claude/projects/$KEY 이미 존재 — 메모리 설치 건너뜀"
else
  tar -xf "$SRC/claude_memory.tar" -C "$HOME/.claude/projects/"
  mv "$HOME/.claude/projects/-home-myungjoo-nntrainer" "$HOME/.claude/projects/$KEY"
  echo "메모리 설치됨: ~/.claude/projects/$KEY ($(ls "$HOME/.claude/projects/$KEY/memory" 2>/dev/null | wc -l) 파일 in memory/)"
fi

step "[4/7] 경로 치환 (구 $OLD → $NEW)"
for f in "$REPO"/.claude/scripts/*.sh; do
  base=$(basename "$f")
  [ "$base" = "migrate_install.sh" ] && continue
  [ "$base" = "migrate_fixup.sh" ] && continue
  if [ -n "$NDK" ]; then
    sed -i "s|$OLD/Android/Sdk/ndk/27.2.12479018|$NDK|g" "$f"
  fi
  sed -i "s|$OLD|$NEW|g" "$f"
done
[ -d "$E2E" ] && find "$E2E" -maxdepth 2 -name 'nntr_config.json' -exec sed -i "s|$OLD|$NEW|g" {} +
git -C "$REPO" worktree prune 2>/dev/null || true
echo "치환 완료"

step "[5/7] 죽은 meson 빌드트리 정리 (절대경로 박힘 → 새 경로에서 무용)"
# 샐비지: 단말서 뽑았던 libOpenCL + android 설치 헤더 스냅샷
SAL="$REPO/.claude/salvage"
mkdir -p "$SAL"
[ -d "$REPO/builddir/opencl" ] && cp -r "$REPO/builddir/opencl" "$SAL/" && echo "샐비지: builddir/opencl (libOpenCL.so + CL 헤더)"
[ -d "$REPO/builddir/android_build_result" ] && cp -r "$REPO/builddir/android_build_result" "$SAL/" && echo "샐비지: android_build_result (설치 헤더+lib)"
[ -d "$REPO/builddir/jni" ] && cp -r "$REPO/builddir/jni" "$SAL/builddir_jni" && echo "샐비지: builddir/jni (Android.mk/Application.mk)"
for b in build build_cl build_x86_cl builddir; do
  [ -d "$REPO/$b" ] && rm -rf "$REPO/$b" && echo "삭제: $b/"
done

step "[6/7] x86 빌드 재구성"
if [ "$NOBUILD" = "1" ]; then
  echo "--no-build 지정 — 건너뜀"
elif command -v meson >/dev/null && command -v ninja >/dev/null; then
  cd "$REPO"
  meson setup build_cl -Denable-opencl=true -Denable-fp16=true \
    -Denable-app=true -Denable-ccapi=true -Denable-transformer=true \
    -Dbuildtype=release -Dwerror=false \
    && ninja -C build_cl Applications/CausalLM/nntr_quantize Applications/CausalLM/nntr_causallm \
    && echo "x86 핵심 타깃 빌드 OK (전체는 ninja -C build_cl)" \
    || echo "⚠ x86 빌드 실패 — meson-logs 확인 (의존성: openblas/jsoncpp/libcurl/opencl-headers 등)"
else
  echo "⚠ meson/ninja 미설치 — sudo apt install meson ninja-build 후:"
  echo "  cd $REPO && meson setup build_cl -Denable-opencl=true -Denable-fp16=true -Denable-app=true -Denable-ccapi=true -Denable-transformer=true -Dbuildtype=release -Dwerror=false"
fi

step "[7/7] 잔여 수동 단계"
cat <<EOF
1. NDK 설치: ${NDK:-"<미지정 — 27.2.12479018 권장>"}
   → .claude/scripts/build_lib.sh 의 ANDROID_NDK 경로 확인
2. Android builddir 재구성 (MIGRATION.md 'Adreno 빌드 흐름' / 설치헤더 동기화 함정 참조):
   meson setup builddir (platform=android) → ndk-build
   샐비지 복원: cp -r $SAL/opencl $REPO/builddir/ ; cp -r $SAL/android_build_result $REPO/builddir/
   (libOpenCL.so 재취득: adb -s R3CY70LV96T pull /vendor/lib64/libOpenCL.so)
3. adb 단말 인증: adb devices 에 R3CY70LV96T 보이는지 (단말엔 모델/스크립트/KS기준 모두 보존됨)
4. git: jijoongmoon 리모트 인증 설정 시 push 가능 (커밋은 --no-gpg-sign --author="Jijoong Moon <jijoong.moon@samsung.com>")
5. pr3978 worktree 필요 시: git fetch origin pull/3978/head:pr3978 && git worktree add ../nntrainer-pr3978 pr3978
6. Claude Code를 $REPO 에서 실행 → 메모리 자동 로드 확인 (MEMORY.md 인덱스)
EOF
echo; echo "설치 완료: $REPO"
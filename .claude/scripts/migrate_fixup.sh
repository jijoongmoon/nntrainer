#!/usr/bin/env bash
# 이주 후 경로 수정 (구경로 /home/nntrainer → 새 위치).
# 새 PC에서 tar를 푼 "직후", repo 루트에서 실행:
#   .claude/scripts/migrate_fixup.sh <NEW_BASE> [NDK_PATH]
# 예: repo가 /home/jijoongmoon/work/nntrainer 에 풀렸으면
#   cd /home/jijoongmoon/work/nntrainer
#   .claude/scripts/migrate_fixup.sh /home/jijoongmoon/work /home/jijoongmoon/Android/Sdk/ndk/27.2.12479018
set -euo pipefail
OLD=/home/nntrainer
NEW=${1:?"사용법: migrate_fixup.sh <NEW_BASE> [NDK_PATH] — NEW_BASE = nntrainer/와 qwen3_e2e/가 들어있는 상위 디렉토리"}
NDK=${2:-$NEW/Android/Sdk/ndk/27.2.12479018}
REPO="$NEW/nntrainer"
E2E="$NEW/qwen3_e2e"

echo "== 1. 호스트 스크립트 경로 치환 (.claude/scripts/*.sh)"
for f in "$REPO"/.claude/scripts/*.sh; do
  [ "$(basename "$f")" = "migrate_fixup.sh" ] && continue
  sed -i "s|$OLD/Android/Sdk/ndk/27.2.12479018|$NDK|g; s|$OLD|$NEW|g" "$f"
done

echo "== 2. qwen3_e2e config 절대경로 치환 (tokenizer_file 등)"
if [ -d "$E2E" ]; then
  find "$E2E" -maxdepth 2 -name 'nntr_config.json' \
    -exec sed -i "s|$OLD|$NEW|g" {} +
fi

echo "== 3. 죽은 meson 빌드트리 정리 (절대경로 박혀서 새 경로에선 무용)"
for b in build build_cl build_x86_cl builddir; do
  if [ -d "$REPO/$b" ]; then echo "   rm 후보: $REPO/$b (재구성 필요)"; fi
done
echo "   → 확인 후 직접 삭제하고 MIGRATION.md의 재구성 절차 수행"
echo "     x86:    meson setup build_cl -Denable-opencl=true -Denable-fp16=true \\"
echo "             -Denable-app=true -Denable-ccapi=true -Denable-transformer=true \\"
echo "             -Dbuildtype=release -Dwerror=false"
echo "     android: meson setup builddir (platform=android 옵션) 후 ndk-build — MIGRATION.md 참조"
echo "     ⚠ builddir/android_build_result/* (설치 헤더·opencl lib)는 삭제 전 백업:"
echo "       opencl/lib의 libOpenCL.so는 단말에서 다시 pull 가능"

echo "== 4. git worktree 메타 정리"
git -C "$REPO" worktree prune || true

echo "== 5. Claude 메모리 디렉토리명"
KEY=$(echo "$REPO" | tr '/' '-')
echo "   메모리 tar를 푼 뒤 디렉토리명을 다음으로 변경해야 자동 로드됨:"
echo "   mv ~/.claude/projects/-home-myungjoo-nntrainer ~/.claude/projects/$KEY"

echo "== 완료. 남은 수동 단계: NDK 설치 확인($NDK), adb 인증(S26 R3CY70LV96T)."

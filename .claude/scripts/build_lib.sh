#!/usr/bin/env bash
set -e -o pipefail
export ANDROID_NDK=/home/nntrainer/Android/Sdk/ndk/27.2.12479018
ROOT=/home/nntrainer/nntrainer
STRIP=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip

# CRITICAL: ndk-build does NOT run meson's .cl -> .cpp codegen. If a kernel .cl
# was edited (e.g. geglu row_off, 52544bf7) the embedded string in builddir's
# generated .cpp goes stale -> HEAD host code dispatches a mismatched kernel ->
# silently GARBAGE output (Adreno "is is" greedy-collapse) while it still builds
# and runs. Regenerate any .cl that is newer than its generated .cpp first.
CLK=$ROOT/nntrainer/tensor/cl_operations/cl_kernels
GEN=$ROOT/builddir/nntrainer/tensor/cl_operations/cl_kernels
for cl in "$CLK"/*.cl; do
  base=$(basename "$cl" .cl); cpp="$GEN/$base.cpp"; hdr="$GEN/$base.h"
  if [ -f "$cpp" ] && [ "$cl" -nt "$cpp" ]; then
    var=$(grep -oE 'extern const std::string [A-Za-z0-9_]+' "$hdr" 2>/dev/null | awk '{print $4}')
    [ -z "$var" ] && var="${base}_kernel"
    python3 "$ROOT/.claude/regen_cl.py" "$base" "$var"
  fi
done

$ANDROID_NDK/ndk-build NDK_PROJECT_PATH=$ROOT/builddir \
  APP_BUILD_SCRIPT=$ROOT/builddir/jni/Android.mk \
  NDK_APPLICATION_MK=$ROOT/builddir/jni/Application.mk \
  -j$(nproc) nntrainer 2>&1 | grep -iE "Compile|SharedLibrary|error" | tail -20
# ndk-build links to obj/local/ but does NOT strip+install to libs/ for a named
# target -> strip the fresh unstripped link ourselves (the libs/ copy goes stale).
$STRIP --strip-unneeded $ROOT/builddir/obj/local/arm64-v8a/libnntrainer.so \
  -o $ROOT/builddir/libs/arm64-v8a/libnntrainer.so
cp $ROOT/builddir/libs/arm64-v8a/libnntrainer.so $ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so
echo "OK: $(ls -la --time-style=+%H:%M:%S $ROOT/builddir/libs/arm64-v8a/libnntrainer.so | awk '{print $6,$7}')"

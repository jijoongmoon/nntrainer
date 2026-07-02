#!/usr/bin/env bash
# Build nntrainer lib + gpu_native binary, stage, and push to device.
# Usage: build_gpu_native.sh [lib]   ("lib" => also rebuild libnntrainer.so)
set -e
cd /home/myungjoo/nntrainer
export PATH=/home/myungjoo/Android/Sdk/ndk/27.2.12479018:$PATH
export ANDROID_NDK=/home/myungjoo/Android/Sdk/ndk/27.2.12479018
export NNTRAINER_ROOT=/home/myungjoo/nntrainer
ROOT=/home/myungjoo/nntrainer
DEV=R3CY205ZMND
DST=/data/local/tmp/nntrainer/causallm

if [ "$1" = "lib" ]; then
  echo "=== [1/4] ndk-build nntrainer lib ==="
  ndk-build -C builddir/jni -j$(nproc) 2>&1 | tail -3
  echo "=== [2/4] stage headers + lib ==="
  cp nntrainer/opencl/opencl_command_queue_manager.h \
     builddir/android_build_result/include/nntrainer/opencl_command_queue_manager.h
  cp nntrainer/opencl/opencl_loader.h \
     builddir/android_build_result/include/nntrainer/opencl_loader.h
  cp builddir/libs/arm64-v8a/libnntrainer.so \
     builddir/android_build_result/lib/arm64-v8a/libnntrainer.so
  cp builddir/libs/arm64-v8a/libnntrainer.so \
     Applications/CausalLM/jni/libs/arm64-v8a/libnntrainer.so
fi

echo "=== [3/4] ndk-build gpu_native (absolute paths) ==="
ndk-build NDK_PROJECT_PATH=$ROOT/Applications/CausalLM/jni \
          APP_BUILD_SCRIPT=$ROOT/Applications/CausalLM/jni/Android.mk \
          NDK_APPLICATION_MK=$ROOT/Applications/CausalLM/jni/Application.mk \
          nntrainer_qwen3_gpu -j$(nproc) 2>&1 | tail -4

echo "=== [4/4] push (obj binary is the fresh link) ==="
PUSH="Applications/CausalLM/jni/obj/local/arm64-v8a/nntrainer_qwen3_gpu"
if [ "$1" = "lib" ]; then
  adb -s $DEV push $PUSH Applications/CausalLM/jni/libs/arm64-v8a/libnntrainer.so $DST/ 2>&1 | tail -1
else
  adb -s $DEV push $PUSH $DST/nntrainer_qwen3_gpu 2>&1 | tail -1
fi
echo "=== BUILD+PUSH DONE ==="

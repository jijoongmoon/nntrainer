#!/usr/bin/env bash
#
# tools/gpu_bench/package_android.sh
#
# Slim Android packager for GPU baseline measurement. Builds only the OpenCL
# unit tests (unittest_opencl_kernels_blas / int4 / qk_k) with profiling on,
# then tars the binaries plus shared libs for adb push.
#
# Usage:
#   tools/gpu_bench/package_android.sh [PROJECT_ROOT] [--arm-arch=armv8.2-a]
#
# Output:
#   <PROJECT_ROOT>/nntrainer_gpu_bench_android.tar.gz
#
# Compared to tools/package_android.sh (full CausalLM build), this script:
#   * passes -Denable-profile=true so M0-1 per-kernel profiler emits per-kernel
#     timings during the unit test run.
#   * disables tflite / fp16-only paths that are not needed for GPU kernel
#     measurement.
#   * builds into builddir_gpu_bench/ so it never conflicts with the regular
#     android build tree.

set -e

TARGET=$1
[ -z "$1" ] && TARGET=$(pwd)

if [ ! -d "$TARGET" ]; then
    if [[ $1 == -D* ]] || [[ $1 == --arm-arch* ]]; then
        TARGET=$(pwd)
    else
        echo "$TARGET is not a directory. please put project root of nntrainer"
        exit 1
    fi
fi

pushd "$TARGET"

filtered_args=()
arm_arch=""

for arg in "$@"; do
    if [[ $arg == -D* ]]; then
        filtered_args+=("$arg")
    fi
    if [[ $arg == --arm-arch=* ]]; then
        arm_arch="${arg#*=}"
    fi
done

if [[ -z "$arm_arch" ]]; then
    arm_arch="armv8.2-a"
fi

arch_filename=$(echo "$arm_arch" | sed 's/\./-/g')
json_file="${TARGET}/tools/cross/android_${arch_filename}.json"
if [[ -f "$json_file" ]]; then
    eval "$(python3 -c "
import json
data = json.load(open('$json_file'))
print(f'arm_march=\"{data.get(\"arm_march\", \"\")}\"')
print(f'enable_fp16={data.get(\"enable_fp16\", \"True\")}')
")"
    filtered_args+=("-Darm-arch=${arm_arch}")
    filtered_args+=("-Darm-march=-march=${arm_march}")
    if [[ "$enable_fp16" == "False" ]]; then
        filtered_args+=("-Denable-fp16=false")
    fi
else
    echo "Warning: JSON config file not found: $json_file"
fi

BUILDDIR="builddir_gpu_bench"

# GPU bench specific options. The profile flag wires M0-1's per-kernel timing
# collector (see commit 468dece / ClContext::~ClContext report).
GPU_BENCH_ARGS=(
    -Dplatform=android
    -Dopenblas-num-threads=1
    -Denable-tflite-interpreter=false
    -Denable-tflite-backbone=false
    -Denable-opencl=true
    -Denable-profile=true
    -Denable-fp16=true
    -Dnntr-num-threads=4
    -Dhgemm-experimental-kernel=false
)

if [ ! -d "$BUILDDIR" ]; then
    meson "$BUILDDIR" "${GPU_BENCH_ARGS[@]}" "${filtered_args[@]}"
else
    echo "warning: $TARGET/$BUILDDIR already exists, reconfiguring"
    pushd "$BUILDDIR"
        meson configure "${GPU_BENCH_ARGS[@]}" "${filtered_args[@]}"
        meson --wipe
    popd
fi

pushd "$BUILDDIR"
# Only the three OpenCL unittests + their library deps. Ninja figures the rest.
ninja test/unittest/unittest_opencl_kernels_blas \
      test/unittest/unittest_opencl_kernels_int4 \
      test/unittest/unittest_opencl_kernels_qk_k

# Stage binaries + shared libs into a flat layout for adb push.
STAGING="$TARGET/$BUILDDIR/gpu_bench_staging"
rm -rf "$STAGING"
mkdir -p "$STAGING"

find test/unittest -maxdepth 1 -type f -name 'unittest_opencl_kernels_*' \
    -not -name '*.o' -not -name '*.p' -exec cp {} "$STAGING/" \;

# Pull in any .so produced by the build (libnntrainer etc.).
find . -maxdepth 4 -name '*.so' -exec cp {} "$STAGING/" \; 2>/dev/null || true

# Bundle the run script so the tarball is self-contained on device.
cp "$TARGET/tools/gpu_bench/run_baseline.sh" "$STAGING/"
chmod +x "$STAGING/run_baseline.sh"

tar -czvf "$TARGET/nntrainer_gpu_bench_android.tar.gz" \
    --directory="$STAGING" .

popd
popd

echo "Packaged: $TARGET/nntrainer_gpu_bench_android.tar.gz"

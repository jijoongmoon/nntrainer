#!/usr/bin/env bash
# Build the public CausalLM Android native libraries and QuickDotAI AAR.
# With no options, build the CPU-only Android.mk targets. App modes build the
# standalone app and AAR.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./build_android.sh [options] [engine Meson options]

Options:
  --cache                 Reuse a compatible engine build; build on cache miss
  --skip-engine           Require and reuse an existing compatible engine build
  --clean                 Recreate the selected CausalLM build outputs
  --enable-qnn            Opt in to the QNN backend and QNN AAR libraries
  --skip-qnn              Explicitly select CPU-only mode (default)
  --nntr-threads=N        Set the nntrainer compute thread count (default: 7)
  --legacy-ndk            Also build Android.mk targets in an app/AAR mode
  --assemble-aar, --aar   Build the standalone app and assemble AAR/APK
  --native-only           Build/stage the standalone app; skip Gradle
  --skip-gradle           Alias for --native-only
  --skip-install          Build/stage standalone native libs; skip Gradle/install
  --install               Assemble, install the sample APK, and push the CLIs
  --help, -h              Show this help

Engine Meson options beginning with -D, and --arm-arch=..., are forwarded to
tools/package_android.sh. They cannot be used together with --skip-engine.

Environment:
  ANDROID_NDK / NDK_ROOT  Android NDK root (required)
  QNN_SDK_ROOT            Qualcomm QNN SDK root (required with --enable-qnn)
  ANDROID_SERIAL          Device selected for --install (optional if exactly
                          one authorized device is connected)
EOF
}

CLEAN=false
USE_BUILD_CACHE=false
SKIP_ENGINE=false
ENABLE_QNN=false
QNN_MODE_SET=""
INSTALL=false
INSTALL_REQUESTED=false
SKIP_INSTALL_REQUESTED=false
NATIVE_ONLY=false
ASSEMBLE_AAR=false
APP_MODE_REQUESTED=false
LEGACY_NDK_REQUESTED=false
NNTR_THREADS="${NNTR_THREADS:-7}"
ENGINE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache)
            USE_BUILD_CACHE=true
            ;;
        --skip-engine)
            SKIP_ENGINE=true
            ;;
        --clean)
            CLEAN=true
            ;;
        --enable-qnn)
            if [[ "$QNN_MODE_SET" == "cpu" ]]; then
                echo "Error: --enable-qnn and --skip-qnn are mutually exclusive." >&2
                exit 2
            fi
            ENABLE_QNN=true
            QNN_MODE_SET="qnn"
            APP_MODE_REQUESTED=true
            ASSEMBLE_AAR=true
            ;;
        --skip-qnn)
            if [[ "$QNN_MODE_SET" == "qnn" ]]; then
                echo "Error: --enable-qnn and --skip-qnn are mutually exclusive." >&2
                exit 2
            fi
            ENABLE_QNN=false
            QNN_MODE_SET="cpu"
            ;;
        --nntr-threads=*)
            NNTR_THREADS="${1#*=}"
            ;;
        --legacy-ndk)
            LEGACY_NDK_REQUESTED=true
            ;;
        --assemble-aar|--aar)
            APP_MODE_REQUESTED=true
            ASSEMBLE_AAR=true
            ;;
        --native-only|--skip-gradle)
            APP_MODE_REQUESTED=true
            NATIVE_ONLY=true
            ;;
        --skip-install)
            APP_MODE_REQUESTED=true
            NATIVE_ONLY=true
            SKIP_INSTALL_REQUESTED=true
            ;;
        --install)
            APP_MODE_REQUESTED=true
            ASSEMBLE_AAR=true
            INSTALL_REQUESTED=true
            INSTALL=true
            ;;
        -D*|--arm-arch=*)
            ENGINE_ARGS+=("$1")
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if ! [[ "$NNTR_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --nntr-threads must be a positive integer." >&2
    exit 2
fi
if [[ "$USE_BUILD_CACHE" == true && "$SKIP_ENGINE" == true ]]; then
    echo "Error: --cache and --skip-engine are mutually exclusive." >&2
    exit 2
fi
if [[ "$SKIP_ENGINE" == true && ${#ENGINE_ARGS[@]} -ne 0 ]]; then
    echo "Error: engine Meson options cannot be used with --skip-engine." >&2
    exit 2
fi
if [[ "$INSTALL_REQUESTED" == true && "$SKIP_INSTALL_REQUESTED" == true ]]; then
    echo "Error: --install and --skip-install are mutually exclusive." >&2
    exit 2
fi
if [[ "$INSTALL" == true && "$NATIVE_ONLY" == true ]]; then
    echo "Error: --install and --native-only are mutually exclusive." >&2
    exit 2
fi

if [[ "$NATIVE_ONLY" == true ]]; then
    ASSEMBLE_AAR=false
fi
if [[ "$APP_MODE_REQUESTED" == true ]]; then
    LEGACY_NDK="$LEGACY_NDK_REQUESTED"
else
    LEGACY_NDK=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAUSALLM_ROOT="$SCRIPT_DIR"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
XGRAMMAR_ROOT="$CAUSALLM_ROOT/xgrammar"
APP_BUILD="$NNTRAINER_ROOT/builddir_app"
APP_MACHINE_DIR="$NNTRAINER_ROOT/builddir_app_machine"

if command -v nproc >/dev/null 2>&1; then
    BUILD_JOBS="$(nproc)"
else
    BUILD_JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 1)"
fi

ANDROID_NDK="${ANDROID_NDK:-${NDK_ROOT:-}}"
if [[ -z "$ANDROID_NDK" ]]; then
    echo "Error: ANDROID_NDK (or NDK_ROOT) is not set." >&2
    exit 1
fi
if [[ ! -d "$ANDROID_NDK/toolchains/llvm/prebuilt" ]]; then
    echo "Error: invalid Android NDK root: $ANDROID_NDK" >&2
    exit 1
fi
export ANDROID_NDK
export PATH="$ANDROID_NDK:$PATH"

case "$(uname -s)" in
    Linux*)
        ANDROID_NDK_HOST="linux-x86_64"
        ANDROID_CLANG_SUFFIX=""
        ANDROID_EXE_SUFFIX=""
        NDK_BUILD="$ANDROID_NDK/ndk-build"
        ;;
    Darwin*)
        ANDROID_NDK_HOST="darwin-x86_64"
        ANDROID_CLANG_SUFFIX=""
        ANDROID_EXE_SUFFIX=""
        NDK_BUILD="$ANDROID_NDK/ndk-build"
        ;;
    CYGWIN*|MINGW*|MSYS*)
        ANDROID_NDK_HOST="windows-x86_64"
        ANDROID_CLANG_SUFFIX=".cmd"
        ANDROID_EXE_SUFFIX=".exe"
        NDK_BUILD="$ANDROID_NDK/ndk-build.cmd"
        ;;
    *)
        echo "Error: unsupported NDK host: $(uname -s)" >&2
        exit 1
        ;;
esac
if [[ ! -d "$ANDROID_NDK/toolchains/llvm/prebuilt/$ANDROID_NDK_HOST" ]]; then
    echo "Error: NDK host toolchain not found: $ANDROID_NDK_HOST" >&2
    exit 1
fi

# Meson is a native Windows process under Git Bash, so paths embedded inside
# its cross file must use the C:/... spelling rather than MSYS /c/....
ANDROID_NDK_MESON="$ANDROID_NDK"
if [[ "$ANDROID_NDK_HOST" == "windows-x86_64" ]] && command -v cygpath >/dev/null 2>&1; then
    ANDROID_NDK_MESON="$(cygpath -m "$ANDROID_NDK")"
fi

if [[ "$ENABLE_QNN" == true && -z "${QNN_SDK_ROOT:-}" ]]; then
    echo "Error: QNN_SDK_ROOT is required with --enable-qnn." >&2
    exit 1
fi

echo "=== nntrainer CausalLM Android build ==="
echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK:    $ANDROID_NDK"
echo "Mode:           $([[ "$ENABLE_QNN" == true ]] && echo QNN || echo CPU)"
echo "Install:        $INSTALL"
echo "App/AAR mode:   $APP_MODE_REQUESTED"
echo "Assemble AAR:   $ASSEMBLE_AAR"
echo "Native only:    $NATIVE_ONLY"
echo "Legacy NDK:     $LEGACY_NDK"
echo "Engine cache:   $USE_BUILD_CACHE"
echo "Skip engine:    $SKIP_ENGINE"

# Initialize missing build dependencies.
if [[ ! -f "$XGRAMMAR_ROOT/cpp/compiled_grammar.cc" ]]; then
    echo "[0] Initializing xgrammar..."
    git -C "$NNTRAINER_ROOT" submodule update --init Applications/CausalLM/xgrammar
fi
if [[ ! -d "$XGRAMMAR_ROOT/3rdparty/dlpack/include" ]]; then
    echo "[0] Initializing xgrammar/dlpack..."
    git -C "$XGRAMMAR_ROOT" submodule update --init 3rdparty/dlpack
fi
if [[ ! -f "$NNTRAINER_ROOT/subprojects/iniparser/src/iniparser.h" ]]; then
    echo "[0] Initializing nntrainer nested submodules..."
    git -C "$NNTRAINER_ROOT" submodule update --init --recursive --depth 1
fi

restore_cached_json_header() {
    local candidate
    for candidate in \
        "$NNTRAINER_ROOT/builddir_x86/json.hpp" \
        "$NNTRAINER_ROOT/builddir/json.hpp" \
        "$NNTRAINER_ROOT/builddir/encoder/json.hpp"; do
        if [[ -f "$candidate" ]]; then
            cp "$candidate" "$CAUSALLM_ROOT/json.hpp"
            return 0
        fi
    done
    return 1
}

if [[ ! -f "$CAUSALLM_ROOT/json.hpp" ]]; then
    restore_cached_json_header || true
fi
if [[ ! -f "$CAUSALLM_ROOT/json.hpp" ]]; then
    echo "[1] Preparing json.hpp..."
    # Clear a stale encoder marker if its generated header was removed.
    rm -rf "$NNTRAINER_ROOT/builddir/encoder"
    "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir" "0.2"
    restore_cached_json_header || true
fi
if [[ ! -f "$CAUSALLM_ROOT/json.hpp" ]]; then
    echo "Error: failed to prepare $CAUSALLM_ROOT/json.hpp" >&2
    exit 1
fi

TOKENIZER="$CAUSALLM_ROOT/lib/libtokenizers_android_c.a"
if [[ ! -f "$TOKENIZER" ]]; then
    echo "[2] Building the Android tokenizer library..."
    "$CAUSALLM_ROOT/build_tokenizer_android.sh"
fi
if [[ ! -f "$TOKENIZER" ]]; then
    echo "Error: tokenizer build did not produce $TOKENIZER" >&2
    exit 1
fi

NNTRAINER_ANDROID_RESULT="$NNTRAINER_ROOT/builddir/android_build_result"
NNTRAINER_ANDROID_LIBDIR="$NNTRAINER_ANDROID_RESULT/lib/arm64-v8a"
NNTRAINER_ABI_FILE="$NNTRAINER_ANDROID_RESULT/nntrainer-abi.ini"

engine_cache_valid() {
    local required=(
        "$NNTRAINER_ANDROID_LIBDIR/libnntrainer.so"
        "$NNTRAINER_ANDROID_LIBDIR/libccapi-nntrainer.so"
        "$NNTRAINER_ABI_FILE"
    )
    if [[ "$ENABLE_QNN" == true ]]; then
        required+=("$NNTRAINER_ANDROID_LIBDIR/libqnn_context.so")
    fi
    local file
    for file in "${required[@]}"; do
        [[ -f "$file" ]] || return 1
    done
}

describe_missing_engine_cache() {
    local required=(
        "$NNTRAINER_ANDROID_LIBDIR/libnntrainer.so"
        "$NNTRAINER_ANDROID_LIBDIR/libccapi-nntrainer.so"
        "$NNTRAINER_ABI_FILE"
    )
    if [[ "$ENABLE_QNN" == true ]]; then
        required+=("$NNTRAINER_ANDROID_LIBDIR/libqnn_context.so")
    fi
    local file
    for file in "${required[@]}"; do
        [[ -f "$file" ]] || echo "  missing: $file" >&2
    done
}

if [[ "$SKIP_ENGINE" == true ]]; then
    if ! engine_cache_valid; then
        echo "Error: --skip-engine requires a compatible engine build:" >&2
        describe_missing_engine_cache
        exit 1
    fi
    echo "[3] Reusing the existing engine (--skip-engine)."
elif [[ "$USE_BUILD_CACHE" == true && ${#ENGINE_ARGS[@]} -eq 0 ]] && engine_cache_valid; then
    echo "[3] Reusing a compatible engine build (--cache)."
else
    if [[ "$USE_BUILD_CACHE" == true ]]; then
        if [[ ${#ENGINE_ARGS[@]} -ne 0 ]]; then
            echo "[3] Explicit engine options bypass --cache; rebuilding."
        else
            echo "[3] Engine cache miss or incompatible metadata; rebuilding."
            describe_missing_engine_cache
        fi
    else
        echo "[3] Building the Android engine."
    fi
    (
        cd "$NNTRAINER_ROOT"
        ./tools/package_android.sh \
            -Dmmap-read=false \
            -Dnntr-num-threads="$NNTR_THREADS" \
            -Denable-npu="$ENABLE_QNN" \
            "${ENGINE_ARGS[@]}"
    )
    if ! engine_cache_valid; then
        echo "Error: engine build did not produce all required artifacts." >&2
        describe_missing_engine_cache
        exit 1
    fi
fi

build_legacy_ndk_targets() {
    echo "[4] Building the Android.mk targets."
    local legacy_jni_dir="$CAUSALLM_ROOT/jni"
    if [[ "$CLEAN" == true ]]; then
        rm -rf "$legacy_jni_dir/libs" "$legacy_jni_dir/obj"
    fi
    (
        cd "$legacy_jni_dir"
        "$NDK_BUILD" \
            NDK_PROJECT_PATH=. \
            NDK_LIBS_OUT=./libs \
            NDK_OUT=./obj \
            APP_BUILD_SCRIPT=./Android.mk \
            NDK_APPLICATION_MK=./Application.mk \
            causallm_core nntrainer_causallm nntr_quantize \
            nntr_safetensors_info causallm_api test_api \
            -j "$BUILD_JOBS"
    )
    local file
    for file in \
        libcausallm_core.so nntrainer_causallm nntr_quantize \
        nntr_safetensors_info libquick_dot_ai_api.so quick_dot_ai_test; do
        if [[ ! -f "$legacy_jni_dir/libs/arm64-v8a/$file" ]]; then
            echo "Error: expected Android.mk artifact missing: $file" >&2
            exit 1
        fi
    done
}

if [[ "$LEGACY_NDK" == true ]]; then
    build_legacy_ndk_targets
fi
if [[ "$APP_MODE_REQUESTED" == false ]]; then
    echo "=== Done (Android.mk build) ==="
    echo "Artifacts: $CAUSALLM_ROOT/jni/libs/arm64-v8a"
    exit 0
fi

# Keep machine files outside APP_BUILD so Meson can wipe it safely.
mkdir -p "$APP_MACHINE_DIR"
CROSS_FILE="$APP_MACHINE_DIR/android-aarch64.cross"
ABI_CROSS_FILE="$APP_MACHINE_DIR/nntrainer-abi.ini"
CROSS_FILE_IN="$CAUSALLM_ROOT/app_build/android-aarch64.cross.in"
MACHINE_CHANGED=false

rendered_cross="$(mktemp)"
rendered_abi="$(mktemp)"
cleanup_temp_files() {
    rm -f "$rendered_cross" "$rendered_abi"
}
trap cleanup_temp_files EXIT

sed -e "s|@ANDROID_NDK@|$ANDROID_NDK_MESON|g" \
    -e "s|@ANDROID_NDK_HOST@|$ANDROID_NDK_HOST|g" \
    -e "s|@ANDROID_CLANG_SUFFIX@|$ANDROID_CLANG_SUFFIX|g" \
    -e "s|@ANDROID_EXE_SUFFIX@|$ANDROID_EXE_SUFFIX|g" \
    "$CROSS_FILE_IN" > "$rendered_cross"
cp "$NNTRAINER_ABI_FILE" "$rendered_abi"

if [[ ! -f "$CROSS_FILE" ]] || ! cmp -s "$rendered_cross" "$CROSS_FILE"; then
    MACHINE_CHANGED=true
    cp "$rendered_cross" "$CROSS_FILE"
fi
if [[ ! -f "$ABI_CROSS_FILE" ]] || ! cmp -s "$rendered_abi" "$ABI_CROSS_FILE"; then
    MACHINE_CHANGED=true
    cp "$rendered_abi" "$ABI_CROSS_FILE"
fi

if [[ "$CLEAN" == true || "$MACHINE_CHANGED" == true ]]; then
    if [[ -d "$APP_BUILD" ]]; then
        reason="machine configuration changed"
        [[ "$CLEAN" == true ]] && reason="--clean requested"
        echo "[4] Recreating app build directory ($reason)."
        rm -rf "$APP_BUILD"
    fi
fi

MESON_APP_OPTS=(
    -Dplatform=android
    -Denable-qnn="$ENABLE_QNN"
    -Denable-api=true
    -Denable-api-test=true
)
MESON_CROSS_OPTS=(
    --cross-file "$CROSS_FILE"
    --cross-file "$ABI_CROSS_FILE"
)

echo "[4] Configuring the standalone app build."
if [[ ! -f "$APP_BUILD/build.ninja" ]]; then
    meson setup "$APP_BUILD" "$CAUSALLM_ROOT/app_build" \
        "${MESON_CROSS_OPTS[@]}" "${MESON_APP_OPTS[@]}"
else
    meson setup "$APP_BUILD" "$CAUSALLM_ROOT/app_build" --reconfigure \
        "${MESON_CROSS_OPTS[@]}" "${MESON_APP_OPTS[@]}"
fi

echo "[5] Building libcausallm, the public API, and CLIs."
ninja -C "$APP_BUILD" -j "$BUILD_JOBS"
for file in libcausallm.so libquick_dot_ai_api.so quick_dot_ai_test nntr_causallm; do
    if [[ ! -f "$APP_BUILD/$file" ]]; then
        echo "Error: expected app artifact missing: $APP_BUILD/$file" >&2
        exit 1
    fi
done

PREBUILT_DIR="$CAUSALLM_ROOT/Android/QuickDotAI/prebuilt_libs"
echo "[6] Staging public native libraries in $PREBUILT_DIR."
mkdir -p "$PREBUILT_DIR"
find "$PREBUILT_DIR" -maxdepth 1 -type f -name '*.so' -delete

copy_required() {
    local source="$1"
    if [[ ! -f "$source" ]]; then
        echo "Error: required native library missing: $source" >&2
        exit 1
    fi
    cp "$source" "$PREBUILT_DIR/"
}

copy_required "$NNTRAINER_ANDROID_LIBDIR/libnntrainer.so"
copy_required "$NNTRAINER_ANDROID_LIBDIR/libccapi-nntrainer.so"
copy_required "$APP_BUILD/libcausallm.so"
copy_required "$APP_BUILD/libquick_dot_ai_api.so"

if [[ "$ENABLE_QNN" == true ]]; then
    copy_required "$NNTRAINER_ANDROID_LIBDIR/libqnn_context.so"
    QNN_AARCH64_LIB_DIR="$QNN_SDK_ROOT/lib/aarch64-android"
    QNN_AARCH64_LIBS=(
        libQnnHtp.so
        libQnnHtpNetRunExtensions.so
        libQnnHtpPrepare.so
        libQnnHtpProfilingReader.so
        libQnnHtpOptraceProfilingReader.so
        libQnnSaver.so
        libQnnSystem.so
        libQnnHtpV75Stub.so
        libQnnHtpV75CalculatorStub.so
        libQnnHtpV79Stub.so
        libQnnHtpV79CalculatorStub.so
        libQnnHtpV81Stub.so
        libQnnHtpV81CalculatorStub.so
    )
    for file in "${QNN_AARCH64_LIBS[@]}"; do
        if [[ -f "$QNN_AARCH64_LIB_DIR/$file" ]]; then
            cp "$QNN_AARCH64_LIB_DIR/$file" "$PREBUILT_DIR/"
        else
            echo "Warning: optional QNN library not found: $file" >&2
        fi
    done
    QNN_SKEL_LIBS=(
        hexagon-v75/unsigned/libQnnHtpV75Skel.so
        hexagon-v79/unsigned/libQnnHtpV79Skel.so
        hexagon-v81/unsigned/libQnnHtpV81Skel.so
    )
    for relative_path in "${QNN_SKEL_LIBS[@]}"; do
        if [[ -f "$QNN_SDK_ROOT/lib/$relative_path" ]]; then
            cp "$QNN_SDK_ROOT/lib/$relative_path" "$PREBUILT_DIR/"
        else
            echo "Warning: optional QNN skel not found: $relative_path" >&2
        fi
    done
fi

LIBCXX="$ANDROID_NDK/toolchains/llvm/prebuilt/$ANDROID_NDK_HOST/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
copy_required "$LIBCXX"

if [[ "$ENABLE_QNN" == false ]]; then
    if find "$PREBUILT_DIR" -maxdepth 1 -type f \
        \( -name 'libqnn_context.so' -o -name 'libQnn*.so' \) | grep -q .; then
        echo "Error: a CPU-only stage contains QNN libraries." >&2
        exit 1
    fi
fi
if [[ -f "$PREBUILT_DIR/libquick_dot_ai.so" ]]; then
    echo "Error: proprietary model overlay leaked into the public stage." >&2
    exit 1
fi

if [[ "$ASSEMBLE_AAR" == false ]]; then
    echo "=== Done (native libraries staged; Gradle skipped) ==="
    exit 0
fi

echo "[7] Assembling the QuickDotAI AAR and sample APK."
(
    cd "$CAUSALLM_ROOT/Android"
    ./gradlew :QuickDotAI:assembleDebug :SampleTestAPP:assembleDebug
)
AAR="$CAUSALLM_ROOT/Android/QuickDotAI/build/outputs/aar/QuickDotAI-debug.aar"
APK="$CAUSALLM_ROOT/Android/SampleTestAPP/build/outputs/apk/debug/SampleTestAPP-debug.apk"
for file in "$AAR" "$APK"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: Gradle artifact missing: $file" >&2
        exit 1
    fi
done

if [[ "$INSTALL" == false ]]; then
    echo "=== Done (AAR/APK assembled; no device modified) ==="
    echo "AAR: $AAR"
    echo "APK: $APK"
    exit 0
fi

if ! command -v adb >/dev/null 2>&1; then
    echo "Error: adb is required with --install." >&2
    exit 1
fi
if [[ -z "${ANDROID_SERIAL:-}" ]]; then
    connected_devices="$(adb devices | awk '$2 == "device" { print $1 }')"
    device_count="$(printf '%s\n' "$connected_devices" | awk 'NF { count++ } END { print count + 0 }')"
    if [[ "$device_count" -ne 1 ]]; then
        echo "Error: --install needs ANDROID_SERIAL or exactly one authorized device." >&2
        exit 1
    fi
    export ANDROID_SERIAL="$connected_devices"
fi
ADB=(adb -s "$ANDROID_SERIAL")
if [[ "$("${ADB[@]}" get-state 2>/dev/null)" != "device" ]]; then
    echo "Error: Android device is not ready: $ANDROID_SERIAL" >&2
    exit 1
fi

echo "[8] Installing the sample APK and pushing command-line tools."
(
    cd "$CAUSALLM_ROOT/Android"
    ./gradlew :SampleTestAPP:installDebug
)

DEVICE_DIR="/data/local/tmp/Quick.AI"
"${ADB[@]}" shell "mkdir -p $DEVICE_DIR"
"${ADB[@]}" push "$APP_BUILD/quick_dot_ai_test" "$DEVICE_DIR/"
"${ADB[@]}" push "$APP_BUILD/nntr_causallm" "$DEVICE_DIR/"
"${ADB[@]}" push "$APP_BUILD/libcausallm.so" "$DEVICE_DIR/"
"${ADB[@]}" push "$APP_BUILD/libquick_dot_ai_api.so" "$DEVICE_DIR/"
"${ADB[@]}" push "$NNTRAINER_ANDROID_LIBDIR/libnntrainer.so" "$DEVICE_DIR/"
"${ADB[@]}" push "$NNTRAINER_ANDROID_LIBDIR/libccapi-nntrainer.so" "$DEVICE_DIR/"
"${ADB[@]}" push "$LIBCXX" "$DEVICE_DIR/"
if [[ "$ENABLE_QNN" == true ]]; then
    "${ADB[@]}" push "$NNTRAINER_ANDROID_LIBDIR/libqnn_context.so" "$DEVICE_DIR/"
fi

"${ADB[@]}" shell "cat > $DEVICE_DIR/run_test.sh << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=$DEVICE_DIR:\$LD_LIBRARY_PATH
export NNTR_NUM_THREADS=$NNTR_THREADS
cd $DEVICE_DIR
./quick_dot_ai_test \"\$@\"
EOF
chmod 755 $DEVICE_DIR/run_test.sh $DEVICE_DIR/quick_dot_ai_test"

"${ADB[@]}" shell "cat > $DEVICE_DIR/run_causallm.sh << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=$DEVICE_DIR:\$LD_LIBRARY_PATH
export NNTR_NUM_THREADS=$NNTR_THREADS
cd $DEVICE_DIR
./nntr_causallm \"\$@\"
EOF
chmod 755 $DEVICE_DIR/run_causallm.sh $DEVICE_DIR/nntr_causallm"

echo "=== Done (installed on $ANDROID_SERIAL) ==="
echo "Device CLI: $DEVICE_DIR"

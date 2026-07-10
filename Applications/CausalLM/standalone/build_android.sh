#!/bin/bash
# nntrainer-native Android orchestration script (Slice S3-後).
#
# Builds + installs the CausalLM Android APP end-to-end using ONLY the
# nntrainer submodule -- no QuickAI root meson project, no QUICKAI_ROOT.
# It drives the S3-先 standalone meson project (Applications/CausalLM/
# standalone/meson.build) instead of QuickAI's root meson.build, and stages
# a public-only set of libraries into the QuickDotAI AAR's prebuilt_libs/
# (no libquick_dot_ai.so -- the standalone project never builds the
# proprietary overlay plugin, so the result contains public models only,
# by construction).
#
# Usage:
#   ./build_android.sh                      # engine + app build + stage + install
#   ./build_android.sh --skip-engine        # reuse existing builddir/android_build_result
#   ./build_android.sh --clean              # wipe the app builddir (builddir_standalone_app) first
#   ./build_android.sh --nntr-threads=4     # override nntrainer compute thread count (default 7)
#
# Environment:
#   ANDROID_NDK / NDK_ROOT  - required (either name accepted)
#   QNN_SDK_ROOT            - optional; enables staging the QNN vendor runtime libs
#   ANDROID_SERIAL          - optional; select a specific adb/gradle target device
set -e

# ── Parse arguments ─────────────────────────────────────────────────────
CLEAN=false
SKIP_ENGINE=false
NNTR_THREADS="${NNTR_THREADS:-7}"

for arg in "$@"; do
    case "$arg" in
        --clean)          CLEAN=true ;;
        --skip-engine)     SKIP_ENGINE=true ;;
        --nntr-threads=*) NNTR_THREADS="${arg#*=}" ;;
        --help|-h)
            sed -n '2,/^set -e$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Warning: Unknown option: $arg" ;;
    esac
done

# ── Path configuration (self-contained inside the nntrainer submodule) ──
# This script lives at Applications/CausalLM/standalone/build_android.sh:
#   SCRIPT_DIR      = .../nntrainer/Applications/CausalLM/standalone
#   CAUSALLM_ROOT   = .../nntrainer/Applications/CausalLM        (SCRIPT_DIR/..)
#   NNTRAINER_ROOT  = .../nntrainer                              (SCRIPT_DIR/../../..)
# There is intentionally NO QUICKAI_ROOT anywhere in this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAUSALLM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
XGRAMMAR_ROOT="$CAUSALLM_ROOT/xgrammar"

# The app build gets a NEW build directory, separate from nntrainer/build,
# nntrainer/builddir (android engine) and nntrainer/builddir_x86 -- none of
# those are touched by this script.
APP_BUILD="$NNTRAINER_ROOT/builddir_standalone_app"

# Accept either ANDROID_NDK or NDK_ROOT.
if [ -z "$ANDROID_NDK" ] && [ -n "$NDK_ROOT" ]; then
    ANDROID_NDK="$NDK_ROOT"
fi
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK (or NDK_ROOT) is not set."
    echo "Example: export ANDROID_NDK=/path/to/android-ndk-r28c"
    exit 1
fi

echo "=== nntrainer standalone Android app build (public-only) ==="
echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "CAUSALLM_ROOT:  $CAUSALLM_ROOT"
echo "APP_BUILD:      $APP_BUILD"
echo "ANDROID_NDK:    $ANDROID_NDK"
echo "NNTR_THREADS:   $NNTR_THREADS"
echo "SKIP_ENGINE:    $SKIP_ENGINE"
echo "CLEAN:          $CLEAN"
echo ""

# ── Step 0: Submodule init guard ────────────────────────────────────────
# xgrammar lives inside the nntrainer submodule at
# Applications/CausalLM/xgrammar. A missing checkout makes the standalone
# meson configure fail with "File .../xgrammar/cpp/compiled_grammar.cc does
# not exist." Only the dlpack nested submodule is required by the build
# (cpptrace is compiled out; googletest is test-only).
if [ ! -f "$XGRAMMAR_ROOT/cpp/compiled_grammar.cc" ]; then
    echo "[0] Initializing xgrammar submodule..."
    git -C "$NNTRAINER_ROOT" submodule update --init Applications/CausalLM/xgrammar
fi

if [ ! -d "$XGRAMMAR_ROOT/3rdparty/dlpack/include" ]; then
    echo "[0] Initializing xgrammar nested submodule (dlpack)..."
    git -C "$XGRAMMAR_ROOT" submodule update --init 3rdparty/dlpack
fi

# nntrainer's own nested submodule (iniparser), required by the engine
# build (tools/package_android.sh). Guarded the same way build.sh does it,
# for a truly fresh checkout.
if [ ! -f "$NNTRAINER_ROOT/subprojects/iniparser/src/iniparser.h" ]; then
    echo "[0] Initializing nntrainer nested submodules (iniparser)..."
    git -C "$NNTRAINER_ROOT" submodule update --init --recursive --depth 1
fi

# ── Step 1: json.hpp guard ───────────────────────────────────────────────
if [ ! -f "$CAUSALLM_ROOT/json.hpp" ]; then
    echo "[1] Preparing json.hpp..."
    "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir" "0.2" || true

    if [ ! -f "$CAUSALLM_ROOT/json.hpp" ]; then
        for candidate in "$NNTRAINER_ROOT/builddir_x86/json.hpp" "$NNTRAINER_ROOT/builddir/json.hpp"; do
            if [ -f "$candidate" ]; then
                cp "$candidate" "$CAUSALLM_ROOT/"
                break
            fi
        done
    fi

    if [ ! -f "$CAUSALLM_ROOT/json.hpp" ]; then
        echo "Error: Failed to prepare json.hpp"
        exit 1
    fi
else
    echo "[1] json.hpp already present, skipping."
fi

# ── Step 2: Tokenizer check ──────────────────────────────────────────────
TOKENIZER="$CAUSALLM_ROOT/lib/libtokenizers_android_c.a"
if [ ! -f "$TOKENIZER" ]; then
    echo "Error: Tokenizer library missing. Place it at: $TOKENIZER"
    exit 1
fi
echo "[2] Tokenizer library present: $TOKENIZER"

# ── Step 3: Engine build (nntrainer core + libqnn_context.so) ───────────
NNTRAINER_ANDROID_LIB="$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so"

if [ "$SKIP_ENGINE" = true ]; then
    if [ ! -f "$NNTRAINER_ANDROID_LIB" ]; then
        echo "Error: --skip-engine given but no existing engine build found at:"
        echo "  $NNTRAINER_ANDROID_LIB"
        echo "Run without --skip-engine at least once."
        exit 1
    fi
    echo "[3] --skip-engine: reusing existing engine build ($NNTRAINER_ANDROID_LIB)"
else
    echo "[3] Building nntrainer engine for Android (mmap-read=false, nntr-num-threads=$NNTR_THREADS, enable-npu=true)..."
    (
        cd "$NNTRAINER_ROOT"
        ./tools/package_android.sh -Dmmap-read=false -Dnntr-num-threads="$NNTR_THREADS" -Denable-npu=true
    )
    if [ ! -f "$NNTRAINER_ANDROID_LIB" ]; then
        echo "Error: engine build did not produce $NNTRAINER_ANDROID_LIB"
        exit 1
    fi
fi

# ── Step 4: Generate the android cross file ─────────────────────────────
CROSS_FILE_IN="$SCRIPT_DIR/android-aarch64.cross.in"
CROSS_FILE="$APP_BUILD/android-aarch64.cross"
mkdir -p "$APP_BUILD"
sed "s|@ANDROID_NDK@|$ANDROID_NDK|g" "$CROSS_FILE_IN" > "$CROSS_FILE"
echo "[4] Generated cross file: $CROSS_FILE"

# ── Step 5: App meson build (standalone project, public-only) ───────────
MESON_APP_OPTS=(
    -Dplatform=android
    -Denable-qnn=true
    -Denable-qnn-models=true
    -Denable-api=true
    -Denable-api-test=true
)

echo "[5] Configuring app meson build..."
if [ "$CLEAN" = true ] || [ ! -f "$APP_BUILD/build.ninja" ]; then
    if [ "$CLEAN" = true ]; then
        echo "    --clean: wiping $APP_BUILD"
        rm -rf "$APP_BUILD"
        mkdir -p "$APP_BUILD"
        sed "s|@ANDROID_NDK@|$ANDROID_NDK|g" "$CROSS_FILE_IN" > "$CROSS_FILE"
    fi
    meson setup "$APP_BUILD" "$SCRIPT_DIR" --cross-file "$CROSS_FILE" "${MESON_APP_OPTS[@]}"
else
    meson setup "$APP_BUILD" "$SCRIPT_DIR" --reconfigure --cross-file "$CROSS_FILE" "${MESON_APP_OPTS[@]}" || true
fi

echo "[5] Building app (libcausallm.so, libquick_dot_ai_api.so, quick_dot_ai_test)..."
ninja -C "$APP_BUILD" -j "$(nproc)"

for f in libcausallm.so libquick_dot_ai_api.so quick_dot_ai_test; do
    if [ ! -f "$APP_BUILD/$f" ]; then
        echo "Error: expected app build artifact missing: $APP_BUILD/$f"
        exit 1
    fi
done
echo "[5] App build artifacts present in: $APP_BUILD"

# ── Step 6: Stage into the AAR prebuilt_libs/ (public-only) ─────────────
NNTRAINER_ANDROID_LIBDIR="$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a"
PREBUILT_DIR="$CAUSALLM_ROOT/Android/QuickDotAI/prebuilt_libs"

echo "[6] Staging public-only libraries into $PREBUILT_DIR ..."
mkdir -p "$PREBUILT_DIR"
# Clear first so no stale libquick_dot_ai.so (or anything else) survives from
# a previous QuickAI-driven build -- this project never builds the
# proprietary overlay plugin, so a fresh directory contains public models
# only, by construction.
rm -f "$PREBUILT_DIR"/*.so

# nntrainer engine libs
for f in libnntrainer.so libccapi-nntrainer.so libqnn_context.so; do
    if [ -f "$NNTRAINER_ANDROID_LIBDIR/$f" ]; then
        cp "$NNTRAINER_ANDROID_LIBDIR/$f" "$PREBUILT_DIR/"
    else
        echo "Warning: engine lib not found, skipping: $NNTRAINER_ANDROID_LIBDIR/$f"
    fi
done

# app libs (public: causallm + api only -- no proprietary overlay plugin exists to copy)
for f in libcausallm.so libquick_dot_ai_api.so; do
    cp "$APP_BUILD/$f" "$PREBUILT_DIR/"
done

# QNN vendor runtime libs from the QNN SDK (mirrors apk_install_android.sh)
if [ -n "$QNN_SDK_ROOT" ]; then
    echo "    Copying QNN vendor libraries from QNN_SDK_ROOT ($QNN_SDK_ROOT)..."
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
    for lib in "${QNN_AARCH64_LIBS[@]}"; do
        if [ -f "$QNN_AARCH64_LIB_DIR/$lib" ]; then
            cp "$QNN_AARCH64_LIB_DIR/$lib" "$PREBUILT_DIR/"
        else
            echo "Warning: QNN lib not found, skipping: $QNN_AARCH64_LIB_DIR/$lib"
        fi
    done

    QNN_SKEL_LIBS=(
        "hexagon-v75/unsigned/libQnnHtpV75Skel.so"
        "hexagon-v79/unsigned/libQnnHtpV79Skel.so"
        "hexagon-v81/unsigned/libQnnHtpV81Skel.so"
    )
    for rel in "${QNN_SKEL_LIBS[@]}"; do
        if [ -f "$QNN_SDK_ROOT/lib/$rel" ]; then
            cp "$QNN_SDK_ROOT/lib/$rel" "$PREBUILT_DIR/"
        else
            echo "Warning: QNN lib not found, skipping: $QNN_SDK_ROOT/lib/$rel"
        fi
    done
else
    echo "Warning: QNN_SDK_ROOT not set; QNN vendor libs (libQnnHtp*.so etc.) not staged."
fi

# libc++_shared.so from the NDK
LIBCXX=$(find "$ANDROID_NDK" -name "libc++_shared.so" -path "*/aarch64*" 2>/dev/null | head -1)
if [ -n "$LIBCXX" ]; then
    cp "$LIBCXX" "$PREBUILT_DIR/"
else
    echo "Warning: libc++_shared.so not found under $ANDROID_NDK"
fi

echo ""
echo "[6] prebuilt_libs/ staged (public-only). Contents:"
ls -la "$PREBUILT_DIR"
if ls "$PREBUILT_DIR"/libquick_dot_ai.so >/dev/null 2>&1; then
    echo "Error: libquick_dot_ai.so present in prebuilt_libs/ -- staging is NOT public-only!"
    exit 1
fi

# ── Step 7: gradle build + install ───────────────────────────────────────
echo ""
echo "[7] Building + installing APK (SampleTestAPP)..."
(
    cd "$CAUSALLM_ROOT/Android"
    ./gradlew ":SampleTestAPP:installDebug"
)

# ── Step 8: Stage the CLI on-device for CPU verification ────────────────
DEVICE_DIR="/data/local/tmp/Quick.AI"
echo ""
echo "[8] Pushing CLI + fresh libraries to $DEVICE_DIR (device CPU verification)..."
if command -v adb >/dev/null 2>&1 && adb devices | grep -q "device$"; then
    adb shell "mkdir -p $DEVICE_DIR"
    adb push "$APP_BUILD/quick_dot_ai_test"                 "$DEVICE_DIR/"
    adb push "$APP_BUILD/libcausallm.so"                    "$DEVICE_DIR/"
    adb push "$APP_BUILD/libquick_dot_ai_api.so"             "$DEVICE_DIR/"
    adb push "$NNTRAINER_ANDROID_LIBDIR/libnntrainer.so"     "$DEVICE_DIR/"
    adb push "$NNTRAINER_ANDROID_LIBDIR/libccapi-nntrainer.so" "$DEVICE_DIR/"
    [ -f "$NNTRAINER_ANDROID_LIBDIR/libqnn_context.so" ] && \
        adb push "$NNTRAINER_ANDROID_LIBDIR/libqnn_context.so" "$DEVICE_DIR/"
    [ -n "$LIBCXX" ] && adb push "$LIBCXX" "$DEVICE_DIR/"
    adb shell "chmod 755 $DEVICE_DIR/quick_dot_ai_test"
    echo "    CLI staged. Example CPU run:"
    echo "      adb shell \"cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR NNTR_NUM_THREADS=$NNTR_THREADS ./quick_dot_ai_test <model> [prompt]\""
else
    echo "    (skipped: no adb device connected)"
fi

echo ""
echo "=== Done ==="
echo "App build:     $APP_BUILD"
echo "prebuilt_libs: $PREBUILT_DIR (public-only)"
echo "Device CLI:    $DEVICE_DIR"

#!/bin/bash
# APK 빌드용 라이브러리 설치 스크립트
# install_libs/ 디렉토리에 라이브러리만 복사 (adb push 없음)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR is nntrainer/Applications/CausalLM; QUICKAI_ROOT is the Quick.AI
# superproject root, 3 levels up.
QUICKAI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$QUICKAI_ROOT/builddir_android"
NNTRAINER_ROOT="$QUICKAI_ROOT/nntrainer"
NNTRAINER_ANDROID="$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a"
INSTALL_LIBS_DIR="$QUICKAI_ROOT/install_libs"

# ── Validate ────────────────────────────────────────────────────────────
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    echo "Run './build.sh --platform=android' first."
    exit 1
fi

echo "=== Installing libraries for APK build ==="
echo "Install dir: $INSTALL_LIBS_DIR"
echo ""

mkdir -p "$INSTALL_LIBS_DIR"

# ── Copy nntrainer runtime libraries ────────────────────────────────────
echo "Copying nntrainer libraries..."
[ -f "$NNTRAINER_ANDROID/libnntrainer.so" ] && cp "$NNTRAINER_ANDROID/libnntrainer.so" "$INSTALL_LIBS_DIR/"
[ -f "$NNTRAINER_ANDROID/libccapi-nntrainer.so" ] && cp "$NNTRAINER_ANDROID/libccapi-nntrainer.so" "$INSTALL_LIBS_DIR/"

# ── Copy built artifacts ────────────────────────────────────────────────
echo "Copying built artifacts..."

# src targets
for f in libcausallm.so libquick_dot_ai.so; do
    [ -f "$BUILD_DIR/src/$f" ] && cp "$BUILD_DIR/src/$f" "$INSTALL_LIBS_DIR/"
done

[ -f "$BUILD_DIR/src/quick_dot_ai" ] && cp "$BUILD_DIR/src/quick_dot_ai" "$INSTALL_LIBS_DIR/"

# api target
[ -f "$BUILD_DIR/api/libquick_dot_ai_api.so" ] && cp "$BUILD_DIR/api/libquick_dot_ai_api.so" "$INSTALL_LIBS_DIR/"

# api-test target
[ -f "$BUILD_DIR/api-app/quick_dot_ai_test" ] && cp "$BUILD_DIR/api-app/quick_dot_ai_test" "$INSTALL_LIBS_DIR/"

# qnn target
QNN_SO="$NNTRAINER_ANDROID/libqnn_context.so"
[ -f "$QNN_SO" ] && cp "$QNN_SO" "$INSTALL_LIBS_DIR/"

# ── Copy QNN vendor libraries from the QNN SDK ──────────────────────────
# libqnn_context.so only exists when the build ran with --enable-qnn. In
# that case, stage the QNN vendor runtime libs directly from $QNN_SDK_ROOT
# instead of relying on them being manually pre-staged in install_libs/ —
# this is what makes the AAR's prebuilt_libs/ self-sufficient from a clean
# checkout. Missing individual libs only warn (SDK layouts
# can vary slightly across versions); a missing QNN_SDK_ROOT also only
# warns, since install_libs/ may already have these from a previous run.
if [ -f "$QNN_SO" ]; then
    if [ -n "$QNN_SDK_ROOT" ]; then
        echo "Copying QNN vendor libraries from QNN_SDK_ROOT ($QNN_SDK_ROOT)..."

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
                cp "$QNN_AARCH64_LIB_DIR/$lib" "$INSTALL_LIBS_DIR/"
            else
                echo "Warning: QNN lib not found, skipping: $QNN_AARCH64_LIB_DIR/$lib"
            fi
        done

        # Hexagon DSP skel libraries (one per HTP arch version), unsigned
        # variant — matches what the rest of the vendored QNN infra expects.
        QNN_SKEL_LIBS=(
            "hexagon-v75/unsigned/libQnnHtpV75Skel.so"
            "hexagon-v79/unsigned/libQnnHtpV79Skel.so"
            "hexagon-v81/unsigned/libQnnHtpV81Skel.so"
        )
        for rel in "${QNN_SKEL_LIBS[@]}"; do
            if [ -f "$QNN_SDK_ROOT/lib/$rel" ]; then
                cp "$QNN_SDK_ROOT/lib/$rel" "$INSTALL_LIBS_DIR/"
            else
                echo "Warning: QNN lib not found, skipping: $QNN_SDK_ROOT/lib/$rel"
            fi
        done
    else
        echo "Warning: QNN enabled (libqnn_context.so present) but QNN_SDK_ROOT is not set;"
        echo "         QNN vendor libs (libQnnHtp*.so etc.) must already be present in $INSTALL_LIBS_DIR"
    fi
fi

# ── Copy libc++_shared.so from NDK ──────────────────────────────────────
if [ -n "$ANDROID_NDK" ]; then
    LIBCXX=$(find "$ANDROID_NDK" -name "libc++_shared.so" -path "*/aarch64*" 2>/dev/null | head -1)
    if [ -n "$LIBCXX" ]; then
        echo "Copying libc++_shared.so..."
        cp "$LIBCXX" "$INSTALL_LIBS_DIR/"
    fi
fi

echo ""
echo "=== Installation completed ==="
echo "Libraries copied to: $INSTALL_LIBS_DIR"
echo ""
echo "Copied files:"
ls -la "$INSTALL_LIBS_DIR/"
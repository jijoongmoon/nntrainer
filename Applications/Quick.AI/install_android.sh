#!/bin/bash
# Unified installation script for Quick-Dot-AI on Android device
# Installs all built artifacts from builddir_android/ to /data/local/tmp/Quick.AI/
set -e

INSTALL_DIR="/data/local/tmp/Quick.AI"
MODEL_DIR="$INSTALL_DIR/models"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/builddir_android"
NNTRAINER_ROOT="$SCRIPT_DIR/nntrainer"
NNTRAINER_ANDROID="$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a"

# ── Validate ────────────────────────────────────────────────────────────
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    echo "Run './build.sh --platform=android' first."
    exit 1
fi

if ! adb devices | grep -q "device$"; then
    echo "Error: No Android device connected."
    exit 1
fi

INSTALL_LIBS_DIR="$SCRIPT_DIR/install_libs"

echo "=== Installing Quick-Dot-AI to Android device ==="
echo "Install dir: $INSTALL_DIR"
echo ""

mkdir -p "$INSTALL_LIBS_DIR"

adb shell "mkdir -p $INSTALL_DIR $MODEL_DIR"

# ── Push nntrainer runtime libraries ────────────────────────────────────
echo "Pushing nntrainer libraries..."
adb push "$NNTRAINER_ANDROID/libnntrainer.so" $INSTALL_DIR/ && cp "$NNTRAINER_ANDROID/libnntrainer.so" "$INSTALL_LIBS_DIR/"
adb push "$NNTRAINER_ANDROID/libccapi-nntrainer.so" $INSTALL_DIR/ && cp "$NNTRAINER_ANDROID/libccapi-nntrainer.so" "$INSTALL_LIBS_DIR/"

# ── Push built artifacts ────────────────────────────────────────────────
echo "Pushing built artifacts..."

# src targets
for f in libcausallm.so libquick_dot_ai.so; do
    [ -f "$BUILD_DIR/src/$f" ] && adb push "$BUILD_DIR/src/$f" $INSTALL_DIR/ && cp "$BUILD_DIR/src/$f" "$INSTALL_LIBS_DIR/"
done

if [ -f "$BUILD_DIR/src/quick_dot_ai" ]; then
    adb push "$BUILD_DIR/src/quick_dot_ai" $INSTALL_DIR/ && cp "$BUILD_DIR/src/quick_dot_ai" "$INSTALL_LIBS_DIR/"
    adb shell "chmod 755 $INSTALL_DIR/quick_dot_ai"
fi

# api target
[ -f "$BUILD_DIR/api/libquick_dot_ai_api.so" ] && \
    adb push "$BUILD_DIR/api/libquick_dot_ai_api.so" $INSTALL_DIR/ && cp "$BUILD_DIR/api/libquick_dot_ai_api.so" "$INSTALL_LIBS_DIR/"

# api-test target
if [ -f "$BUILD_DIR/api-app/quick_dot_ai_test" ]; then
    adb push "$BUILD_DIR/api-app/quick_dot_ai_test" $INSTALL_DIR/ && cp "$BUILD_DIR/api-app/quick_dot_ai_test" "$INSTALL_LIBS_DIR/"
    adb shell "chmod 755 $INSTALL_DIR/quick_dot_ai_test"
fi

[ -f "$BUILD_DIR/qnn/libqnn_context.so" ] && \
    adb push "$BUILD_DIR/qnn/libqnn_context.so" $INSTALL_DIR/ && cp "$BUILD_DIR/qnn/libqnn_context.so" "$INSTALL_LIBS_DIR/"

# ── Push libc++_shared.so from NDK ──────────────────────────────────────
if [ -n "$ANDROID_NDK" ]; then
    LIBCXX=$(find "$ANDROID_NDK" -name "libc++_shared.so" -path "*/aarch64*" 2>/dev/null | head -1)
    if [ -n "$LIBCXX" ]; then
        echo "Pushing libc++_shared.so..."
        adb push "$LIBCXX" $INSTALL_DIR/ && cp "$LIBCXX" "$INSTALL_LIBS_DIR/"
    fi
fi

# ── Push model config files (res/) ──────────────────────────────────────
RES_DIR="$SCRIPT_DIR/src/res"
if [ -d "$RES_DIR" ]; then
    echo "Pushing model configs..."
    for model_dir in "$RES_DIR"/*/; do
        model_name=$(basename "$model_dir")
        adb shell "mkdir -p $MODEL_DIR/$model_name"
        adb push "$model_dir." "$MODEL_DIR/$model_name/"
        mkdir -p "$INSTALL_LIBS_DIR/models/$model_name"
        cp -r "$model_dir." "$INSTALL_LIBS_DIR/models/$model_name/"
    done
fi

# ── Create run scripts on device ────────────────────────────────────────
adb shell "cat > $INSTALL_DIR/run.sh << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=/data/local/tmp/Quick.AI:\$LD_LIBRARY_PATH
cd /data/local/tmp/Quick.AI
./quick_dot_ai \$@
EOF"
adb shell "chmod 755 $INSTALL_DIR/run.sh"

adb shell "cat > $INSTALL_DIR/run_test.sh << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=/data/local/tmp/Quick.AI:\$LD_LIBRARY_PATH
cd /data/local/tmp/Quick.AI
./quick_dot_ai_test \$@
EOF"
adb shell "chmod 755 $INSTALL_DIR/run_test.sh"

echo ""
echo "=== Installation completed ==="
echo ""
echo "To run on device:"
echo "  adb shell $INSTALL_DIR/run.sh $MODEL_DIR/<model_name>"
echo ""
echo "To run API test:"
echo "  adb shell $INSTALL_DIR/run_test.sh <model_name> [prompt] [chat_template] [quant] [verbose]"

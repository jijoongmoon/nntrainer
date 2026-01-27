#!/bin/bash

# Installation script for CausalLM Android application
set -e

# Configuration
INSTALL_DIR="/data/local/tmp/nntrainer/causallm"
MODEL_DIR="$INSTALL_DIR/models"

# Set SCRIPT_DIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Install CausalLM to Android Device"
echo "========================================"
echo "INSTALL_DIR: $INSTALL_DIR"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo ""

# Check if device is connected
echo "[Step 1/3] Check device connection"
echo "----------------------------------------"
if ! adb devices | grep -q "device$"; then
    echo "Error: No Android device connected. Please connect a device and try again."
    exit 1
fi

DEVICE_ID=$(adb devices | grep "device$" | head -1 | cut -f1)
echo "[SUCCESS] Device connected: $DEVICE_ID"
echo ""

# Check if all required files exist
echo "[Step 2/3] Check build artifacts"
echo "----------------------------------------"
REQUIRED_FILES=(
    "$SCRIPT_DIR/jni/libs/arm64-v8a/nntrainer_causallm"
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm.so"
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libnntrainer.so"
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libccapi-nntrainer.so"
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libc++_shared.so"
)

ALL_FOUND=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  [OK] $(basename $file) ($size)"
    else
        echo "  [MISSING] $file"
        ALL_FOUND=false
    fi
done

if [ "$ALL_FOUND" = false ]; then
    echo ""
    echo "Error: Some required files are missing"
    echo "Please run: ./build_android.sh"
    exit 1
fi

echo "[SUCCESS] All build artifacts found"
echo ""

# Create directories on device
echo "[Step 3/3] Push files to device"
echo "----------------------------------------"
echo "Creating directories on device..."
adb shell "mkdir -p $INSTALL_DIR"
adb shell "mkdir -p $MODEL_DIR"
echo "[SUCCESS] Directories created"
echo ""

# Push executable
echo "Pushing executable..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/nntrainer_causallm" "$INSTALL_DIR/" 2>&1 | tail -1
adb shell "chmod 755 $INSTALL_DIR/nntrainer_causallm"
echo "[SUCCESS] nntrainer_causallm pushed"
echo ""

# Push shared libraries
echo "Pushing shared libraries..."
echo "  [1/5] libcausallm.so (CausalLM API library)..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm.so" "$INSTALL_DIR/" 2>&1 | tail -1

echo "  [2/5] libnntrainer.so (nntrainer library)..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libnntrainer.so" "$INSTALL_DIR/" 2>&1 | tail -1

echo "  [3/5] libccapi-nntrainer.so (nntrainer C/C API)..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libccapi-nntrainer.so" "$INSTALL_DIR/" 2>&1 | tail -1

echo "  [4/5] libc++_shared.so (C++ runtime)..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libc++_shared.so" "$INSTALL_DIR/" 2>&1 | tail -1

echo "  [5/5] libomp.so (OpenMP runtime)..."
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libomp.so" ]; then
    adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libomp.so" "$INSTALL_DIR/" 2>&1 | tail -1
else
    echo "  [SKIP] libomp.so not found"
fi

echo "[SUCCESS] All libraries pushed"
echo ""

# Create run script on device
echo "Creating run script on device..."
adb shell "cat > $INSTALL_DIR/run_causallm.sh << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=$INSTALL_DIR:\$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4
cd $INSTALL_DIR
./nntrainer_causallm \$@
EOF
"

adb shell "chmod 755 $INSTALL_DIR/run_causallm.sh"
echo "[SUCCESS] Run script created"
echo ""

# Summary
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Device: $DEVICE_ID"
echo "Install directory: $INSTALL_DIR"
echo ""
echo "Installed files:"
echo "  - nntrainer_causallm (executable)"
echo "  - libcausallm.so (CausalLM API library)"
echo "  - libnntrainer.so"
echo "  - libccapi-nntrainer.so"
echo "  - libc++_shared.so"
echo "  - libomp.so (if available)"
echo ""
echo "To run CausalLM on the device:"
echo "1. Push your model files to: $MODEL_DIR/"
echo "   Example:"
echo "   adb push res/qwen3-0.6b-w32a32/ $MODEL_DIR/qwen3-0.6b-w32a32/"
echo ""
echo "2. Run the application:"
echo "   adb shell $INSTALL_DIR/run_causallm.sh $MODEL_DIR/qwen3-0.6b-w32a32"
echo ""
echo "For interactive shell:"
echo "   adb shell"
echo "   cd $INSTALL_DIR"
echo "   ./run_causallm.sh $MODEL_DIR/qwen3-0.6b-w32a32"
echo ""

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
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_core.so"
)

# Optional dependency files (might not be in libs/arm64-v8a depending on build)
DEP_FILES=(
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libnntrainer.so"
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libccapi-nntrainer.so"
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libc++_shared.so"
)

# Check main build artifacts
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

# Check dependencies with fallback to obj/local/arm64-v8a
for file in "${DEP_FILES[@]}"; do
    filename=$(basename "$file")
    
    # Special handling for libc++_shared.so (Try copy from NDK)
    if [[ "$filename" == "libc++_shared.so" ]] && [ ! -f "$file" ]; then
        if [ -n "$ANDROID_NDK" ]; then
            # Attempt to find it in typical NDK locations for aarch64
            NDK_LIBCXX=$(find "$ANDROID_NDK" -name "libc++_shared.so" 2>/dev/null | grep "aarch64" | head -n 1)
            
            if [ -n "$NDK_LIBCXX" ] && [ -f "$NDK_LIBCXX" ]; then
                echo "  [WARN] libc++_shared.so not found in build dir, copying from NDK..."
                cp "$NDK_LIBCXX" "$file"
                # Fall through to standard check to confirm copy success
            fi
        fi
    fi

    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  [OK] $filename ($size)"
    else
        # Try to find in obj directory
        obj_path="$SCRIPT_DIR/jni/obj/local/arm64-v8a/$filename"
        if [ -f "$obj_path" ]; then
            echo "  [WARN] $filename found in obj, copying to libs..."
            cp "$obj_path" "$file"
            size=$(du -h "$file" | cut -f1)
            echo "  [OK] $filename ($size) (Copied)"
        else
            echo "  [MISSING] $filename"
            echo "Error: Required dependency not found"
            exit 1
        fi
    fi
done

echo "[SUCCESS] All required build artifacts found"

# Check optional files (API and test app)
OPTIONAL_FILES=(
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_api.so"
    "$SCRIPT_DIR/jni/libs/arm64-v8a/test_api"
)

for file in "${OPTIONAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  [OK] $(basename $file) ($size) (Optional)"
    fi
done

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

# Push optional test_api if exists
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/test_api" ]; then
    echo "Pushing test_api..."
    adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/test_api" "$INSTALL_DIR/" 2>&1 | tail -1
    adb shell "chmod 755 $INSTALL_DIR/test_api"
    echo "[SUCCESS] test_api pushed"
    echo ""
fi

# Push shared libraries
echo "Pushing shared libraries..."
echo "  [1/6] libcausallm_core.so (CausalLM Core library)..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_core.so" "$INSTALL_DIR/" 2>&1 | tail -1

echo "  [2/6] libnntrainer.so (nntrainer library)..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libnntrainer.so" "$INSTALL_DIR/" 2>&1 | tail -1

echo "  [3/6] libccapi-nntrainer.so (nntrainer C/C API)..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libccapi-nntrainer.so" "$INSTALL_DIR/" 2>&1 | tail -1

echo "  [4/6] libc++_shared.so (C++ runtime)..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libc++_shared.so" "$INSTALL_DIR/" 2>&1 | tail -1

echo "  [5/6] libomp.so (OpenMP runtime)..."
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libomp.so" ]; then
    adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libomp.so" "$INSTALL_DIR/" 2>&1 | tail -1
else
    echo "  [SKIP] libomp.so not found"
fi

echo "  [6/6] libcausallm_api.so (CausalLM API library)..."
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_api.so" ]; then
    adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_api.so" "$INSTALL_DIR/" 2>&1 | tail -1
else
    echo "  [SKIP] libcausallm_api.so not found (Optional)"
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

# Create test script on device if API lib exists
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/test_api" ]; then
    adb shell "cat > $INSTALL_DIR/run_test_api.sh << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=$INSTALL_DIR:\$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4
cd $INSTALL_DIR
./test_api \$@
EOF
"
    adb shell "chmod 755 $INSTALL_DIR/run_test_api.sh"
    echo "Run script for test_api created"
fi

echo "[SUCCESS] Run scripts created"
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
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/test_api" ]; then
    echo "  - test_api (executable)"
fi
echo "  - libcausallm_core.so (CausalLM Core library)"
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_api.so" ]; then
    echo "  - libcausallm_api.so (CausalLM API library)"
fi
echo "  - libnntrainer.so"
echo "  - libccapi-nntrainer.so"
echo "  - libc++_shared.so"
echo "  - libomp.so (if available)"
echo ""
echo "To run CausalLM on the device:"
echo "  adb shell $INSTALL_DIR/run_causallm.sh [ARGS]"
echo ""
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/test_api" ]; then
    echo "To run API Test on the device:"
    echo "  adb shell $INSTALL_DIR/run_test_api.sh [ARGS]"
    echo ""
fi

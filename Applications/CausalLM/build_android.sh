#!/bin/bash

# Build script for CausalLM Android application
# This script builds both libcausallm.so and nntrainer_causallm executable
set -e

# Check if NDK path is set
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK is not set. Please set it to your Android NDK path."
    echo "Example: export ANDROID_NDK=/path/to/android-ndk-r21d"
    exit 1
fi

# Set NNTRAINER_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export NNTRAINER_ROOT

echo "========================================"
echo "Build CausalLM Android Application"
echo "========================================"
echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK: $ANDROID_NDK"
echo "Working directory: $(pwd)"
echo ""

# Step 1: Build nntrainer for Android if not already built
echo "[Step 1/4] Build nntrainer for Android"
echo "----------------------------------------"
if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    echo "Building nntrainer for Android..."
    cd "$NNTRAINER_ROOT"
    if [ -d "$NNTRAINER_ROOT/builddir" ]; then
        rm -rf builddir
    fi
    ./tools/package_android.sh
else
    echo "nntrainer for Android already built (skipping)"
fi

# Check if build was successful
if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    echo "Error: nntrainer build failed. Please check the build logs."
    exit 1
fi
echo "[SUCCESS] nntrainer ready"
echo ""

# Step 2: Build tokenizer library if not present
echo "[Step 2/4] Build Tokenizer Library"
echo "----------------------------------------"
cd "$SCRIPT_DIR"
if [ ! -f "lib/libtokenizers_android_c.a" ]; then
    echo "Warning: libtokenizers_android_c.a not found in lib directory."
    echo "Attempting to build tokenizer library..."
    if [ -f "build_tokenizer_android.sh" ]; then
        ./build_tokenizer_android.sh
    else
        echo "Error: tokenizer library not found and build script is missing."
        echo "Please build or download the tokenizer library for Android arm64-v8a"
        echo "and place it in: $SCRIPT_DIR/lib/libtokenizers_android_c.a"
        exit 1
    fi
else
    echo "Tokenizer library already built (skipping)"
fi
echo "[SUCCESS] Tokenizer library ready"
echo ""

# Step 3: Prepare json.hpp if not present
echo "[Step 3/4] Prepare json.hpp"
echo "----------------------------------------"
if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
    echo "json.hpp not found. Downloading..."
    # prepare_encoder.sh expects target directory as first argument and version as second
    # It copies json.hpp to ../Applications/CausalLM/ if version is 0.2
    "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir" "0.2"
    
    if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
        echo "Error: Failed to download json.hpp"
        exit 1
    fi
else
    echo "json.hpp already exists (skipping)"
fi
echo "[SUCCESS] json.hpp ready"
echo ""

# Step 4: Build CausalLM (both libcausallm.so and nntrainer_causallm)
echo "[Step 4/4] Build CausalLM (library + executable)"
echo "----------------------------------------"
cd "$SCRIPT_DIR/jni"

# Clean previous builds
rm -rf obj libs

echo "Building with ndk-build (builds both libcausallm.so and nntrainer_causallm)..."
if ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j $(nproc); then
    echo "[SUCCESS] Build completed successfully"
else
    echo "Error: Build failed"
    exit 1
fi

# Verify outputs
echo ""
echo "Build artifacts:"
if [ -f "libs/arm64-v8a/libcausallm.so" ]; then
    size=$(ls -lh "libs/arm64-v8a/libcausallm.so" | awk '{print $5}')
    echo "  [OK] libcausallm.so ($size)"
else
    echo "  [ERROR] libcausallm.so not found!"
    exit 1
fi

if [ -f "libs/arm64-v8a/nntrainer_causallm" ]; then
    size=$(ls -lh "libs/arm64-v8a/nntrainer_causallm" | awk '{print $5}')
    echo "  [OK] nntrainer_causallm ($size)"
else
    echo "  [ERROR] nntrainer_causallm not found!"
    exit 1
fi
echo ""

# Summary
echo "========================================"
echo "Build Summary"
echo "========================================"
echo "Build completed successfully!"
echo ""
echo "Output files are in: $SCRIPT_DIR/jni/libs/arm64-v8a/"
echo ""
echo "Executables:"
echo "  - nntrainer_causallm (main application)"
echo ""
echo "Libraries:"
echo "  - libcausallm.so (CausalLM API library)"
echo "  - libnntrainer.so (nntrainer library)"
echo "  - libccapi-nntrainer.so (nntrainer C/C API)"
echo "  - libc++_shared.so (C++ runtime)"
echo ""
echo "To install and run:"
echo "  ./install_android.sh"
echo ""

#!/bin/bash
echo "=========================================="
echo "  Android Build & Install Script"
echo "=========================================="

# Exit immediately if any command fails
set -e

# ==========================================================
# Configuration
# ==========================================================
NDK_ROOT="/home/junbong/progra/Android/Sdk/ndk/26.3.11579264"
APK_APPLICATION="SampleTestApp"

# ==========================================================
# 1. Configure Environment
# ==========================================================
echo "[1/6] Configuring environment variables..."
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NDK_ROOT}"
export PATH="${PATH}:${NDK_ROOT}"
export ANDROID_NDK="${NDK_ROOT}"
echo "      ANDROID_NDK set to: ${ANDROID_NDK}"

# ==========================================================
# 2. Build NNTrainer for Android with QNN support
# ==========================================================
echo "[2/6] Building project for Android (with QNN, clean build)..."
./build.sh --platform=android --enable-qnn --clean

# ==========================================================
# 3. Install Android Libraries
# ==========================================================
echo "[3/6] Installing Android libraries..."
./install_android.sh

# ==========================================================
# 4. Deploy Prebuilt Libraries
# ==========================================================
echo "[4/6] Copying prebuilt libraries to QuickDotAI project..."
PREBUILT_DIR="./nntrainer/Applications/QuickAI/QuickDotAI/prebuilt_libs"

# Ensure destination directory exists
mkdir -p "${PREBUILT_DIR}"

# Copy all shared libraries to the project's prebuilt directory
cp ./install_libs/*.so "${PREBUILT_DIR}/"
echo "      Libraries copied to: ${PREBUILT_DIR}"

# ==========================================================
# 5. Build and Install APK
# ==========================================================
echo "[5/6] Building and installing APK..."
cd ./nntrainer/Applications/QuickAI/
./gradlew ":${APK_APPLICATION}:installDebug"

# ==========================================================
# 6. Completion
# ==========================================================
echo "[6/6] Build and installation complete!"
echo "=========================================="
echo "  Success!"
echo "=========================================="

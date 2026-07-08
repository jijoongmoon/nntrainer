#!/bin/bash
echo "=========================================="
echo "  Android Build & Install Script"
echo "=========================================="

# Exit immediately if any command fails
set -e

# ==========================================================
# Configuration
# ==========================================================
APK_APPLICATION="SampleTestApp"

# Default nntrainer compute thread count.
# Passed through to build.sh as --nntr-threads=N → -Dnntr-num-threads=N.
# For APK builds, only the compile-time flag applies (apps can't inherit shell env vars).
NNTR_THREADS="${NNTR_THREADS:-7}"

# ── Parse arguments ─────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --nntr-threads=*)
            NNTR_THREADS="${arg#*=}"
            ;;
        --help|-h)
            echo "Usage: ./apk-build-install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --nntr-threads=N  Set nntrainer compute thread count (default: 7)"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  NNTR_THREADS      Default thread count if --nntr-threads is not given (default: 7)"
            exit 0
            ;;
        *)
            echo "Warning: Unknown option: $arg"
            ;;
    esac
done

# SCRIPT_DIR is nntrainer/Applications/CausalLM (Slice E move); QUICKAI_ROOT
# is the Quick.AI superproject root, 3 levels up (CausalLM -> Applications ->
# nntrainer -> Quick.AI). The QuickAI meson project stays at QUICKAI_ROOT.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUICKAI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ==========================================================
# 1. Configure Environment
# ==========================================================
echo "[1/6] Configuring environment variables..."
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NDK_ROOT}"
export PATH="${PATH}:${NDK_ROOT}"

if [ -z "$NDK_ROOT" ]; then
    echo "Error: NDK_ROOT environment variable is not set"
    echo "Please set NDK_ROOT to your Android NDK installation path"
    echo "Example: export NDK_ROOT=/path/to/android-ndk"
    exit 1
fi
export ANDROID_NDK="${NDK_ROOT}"
echo "      ANDROID_NDK set to: ${ANDROID_NDK}"

# ==========================================================
# 2. Build NNTrainer for Android with QNN support
# ==========================================================
echo "[2/6] Building project for Android (with QNN + QNN models, nntr_threads=${NNTR_THREADS}, clean build)..."
# --qnn-models is required: the API layer (api/quick_dot_ai_api.cpp) registers
# the product QNN models (e.g. gemma4-e2b-qnn), so their sources under
# src/models/qnn/ must be compiled too — otherwise libquick_dot_ai_api.so fails
# to link (undefined Quick_Dot_AI_QNN vtable/dtor). --enable-qnn alone only
# builds the libqnn_context.so QNN infra without the product models.
"$SCRIPT_DIR/build.sh" --platform=android --enable-qnn --qnn-models --enable-experimental-multimodal --nntr-threads="${NNTR_THREADS}" --clean

# ==========================================================
# 3. Install Android Libraries for APK
# ==========================================================
echo "[3/6] Installing Android libraries for APK..."
"$SCRIPT_DIR/apk_install_android.sh"

# ==========================================================
# 4. Deploy Prebuilt Libraries
# ==========================================================
echo "[4/6] Copying prebuilt libraries to QuickDotAI project..."
PREBUILT_DIR="$SCRIPT_DIR/Android/QuickDotAI/prebuilt_libs"

# Ensure destination directory exists
mkdir -p "${PREBUILT_DIR}"

# Copy all shared libraries to the project's prebuilt directory
cp "$QUICKAI_ROOT"/install_libs/*.so "${PREBUILT_DIR}/"
echo "      Libraries copied to: ${PREBUILT_DIR}"

# ==========================================================
# 5. Build and Install APK
# ==========================================================
echo "[5/6] Building and installing APK..."
cd "$SCRIPT_DIR/Android"
./gradlew ":${APK_APPLICATION}:installDebug"

# ==========================================================
# 6. Completion
# ==========================================================
echo "[6/6] Build and installation complete!"
echo "=========================================="
echo "  Success!"
echo "=========================================="

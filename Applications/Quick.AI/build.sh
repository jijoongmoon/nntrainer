#!/bin/bash
# Unified build script for causallm-extension
#
# Usage:
#   ./build.sh                                          # x86, all targets
#   ./build.sh --platform=android                       # android, all targets
#   ./build.sh --platform=android --target=src          # android, src only
#   ./build.sh --platform=x86 --target=src,api          # x86, src + api
#   ./build.sh --platform=android --enable-qnn          # android with QNN
#   ./build.sh --clean                                  # clean rebuild
#
# Environment:
#   ANDROID_NDK  - required for --platform=android builds
set -e

# ── Parse arguments ─────────────────────────────────────────────────────
PLATFORM="x86"
CLEAN=false
TARGETS="all"
ENABLE_QNN=false

for arg in "$@"; do
    case "$arg" in
        --platform=*) PLATFORM="${arg#*=}" ;;
        --target=*)   TARGETS="${arg#*=}" ;;
        --clean)      CLEAN=true ;;
        --enable-qnn) ENABLE_QNN=true ;;
        --help|-h)
            sed -n '2,/^set -e$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
    esac
done

# Validate platform
if [[ "$PLATFORM" != "x86" && "$PLATFORM" != "android" ]]; then
    echo "Error: --platform must be x86 or android (got: $PLATFORM)"
    exit 1
fi

# QNN is android-only
if [ "$ENABLE_QNN" = true ] && [ "$PLATFORM" != "android" ]; then
    echo "Error: --enable-qnn is only supported with --platform=android"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$SCRIPT_DIR/nntrainer"
CAUSALLM_ROOT="$NNTRAINER_ROOT/Applications/CausalLM"

# Build directory
if [ "$PLATFORM" = "android" ]; then
    BUILD_DIR="$SCRIPT_DIR/builddir_android"
else
    BUILD_DIR="$SCRIPT_DIR/builddir_x86"
fi

echo "=== Quick-Dot-AI Unified Build ==="
echo "PLATFORM:   $PLATFORM"
echo "TARGETS:    $TARGETS"
echo "ENABLE_QNN: $ENABLE_QNN"
echo "BUILD_DIR:  $BUILD_DIR"
echo ""

# ── Step 0: Check submodules ────────────────────────────────────────────
if [ ! -f "$NNTRAINER_ROOT/meson.build" ]; then
    echo "[0] Initializing nntrainer submodule..."
    git -C "$SCRIPT_DIR" submodule update --init --recursive --depth 1
fi

# Check iniparser submodule
if [ ! -f "$NNTRAINER_ROOT/subprojects/iniparser/src/iniparser.h" ]; then
    echo "[0] Initializing nntrainer nested submodules..."
    cd "$NNTRAINER_ROOT"
    git submodule update --init --recursive --depth 1
    cd "$SCRIPT_DIR"
fi

# ── Step 1: Pre-build nntrainer ─────────────────────────────────────────
if [ "$PLATFORM" = "android" ]; then
    if [ -z "$ANDROID_NDK" ]; then
        echo "Error: ANDROID_NDK is not set."
        echo "Example: export ANDROID_NDK=/path/to/android-ndk-r21d"
        exit 1
    fi

    NNTRAINER_ANDROID_LIB="$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so"
    if [ "$CLEAN" = true ] || [ ! -f "$NNTRAINER_ANDROID_LIB" ]; then
        echo "[1] Building nntrainer for Android..."
        cd "$NNTRAINER_ROOT"
        [ "$CLEAN" = true ] && rm -rf builddir

        # Build nntrainer with QNN support if requested
        NNTRAINER_EXTRA_OPTS="-Dmmap-read=false"
        if [ "$ENABLE_QNN" = true ]; then
            NNTRAINER_EXTRA_OPTS="$NNTRAINER_EXTRA_OPTS -Denable-npu=true"
        fi
        ./tools/package_android.sh $NNTRAINER_EXTRA_OPTS
        cd "$SCRIPT_DIR"
    else
        echo "[1] nntrainer (android) already built. (use --clean to rebuild)"
    fi
else
    NNTRAINER_BUILD="$NNTRAINER_ROOT/builddir_x86"
    NNTRAINER_X86_LIB="$NNTRAINER_BUILD/nntrainer/libnntrainer.so"
    if [ "$CLEAN" = true ] || [ ! -f "$NNTRAINER_X86_LIB" ]; then
        echo "[1] Building nntrainer for x86..."
        cd "$NNTRAINER_ROOT"
        if [ "$CLEAN" = true ]; then
            rm -rf "$NNTRAINER_BUILD"
        fi
        if [ ! -f "$NNTRAINER_BUILD/build.ninja" ]; then
            meson setup "$NNTRAINER_BUILD" . \
                --buildtype=release \
                -Denable-app=false \
                -Denable-test=false \
                -Denable-transformer=false \
                -Denable-tflite-backbone=false \
                -Denable-tflite-interpreter=false
        fi
        ninja -C "$NNTRAINER_BUILD" -j $(nproc)
        cd "$SCRIPT_DIR"
    else
        echo "[1] nntrainer (x86) already built. (use --clean to rebuild)"
    fi
fi

# ── Step 2: Prepare json.hpp ────────────────────────────────────────────
if [ ! -f "$CAUSALLM_ROOT/json.hpp" ]; then
    echo "[2] Preparing json.hpp..."
    pushd "$NNTRAINER_ROOT" > /dev/null

    if [ "$PLATFORM" = "android" ]; then
        "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir" "0.2" || true
    else
        "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir_x86" "0.2" || true
    fi

    # Fallback: manual copy
    if [ ! -f "$CAUSALLM_ROOT/json.hpp" ]; then
        for candidate in "$NNTRAINER_ROOT/builddir_x86/json.hpp" "$NNTRAINER_ROOT/builddir/json.hpp"; do
            if [ -f "$candidate" ]; then
                cp "$candidate" "$CAUSALLM_ROOT/"
                break
            fi
        done
    fi
    popd > /dev/null

    if [ ! -f "$CAUSALLM_ROOT/json.hpp" ]; then
        echo "Error: Failed to prepare json.hpp"
        exit 1
    fi
fi

# ── Step 3: Tokenizer check ────────────────────────────────────────────
if [ "$PLATFORM" = "android" ]; then
    TOKENIZER="$CAUSALLM_ROOT/lib/libtokenizers_android_c.a"
else
    TOKENIZER="$CAUSALLM_ROOT/lib/libtokenizers_c.a"
fi

if [ ! -f "$TOKENIZER" ]; then
    echo "Warning: Tokenizer library not found: $TOKENIZER"
    if [ -f "$CAUSALLM_ROOT/build_tokenizer_android.sh" ] && [ "$PLATFORM" = "android" ]; then
        echo "Building tokenizer..."
        cd "$CAUSALLM_ROOT" && ./build_tokenizer_android.sh && cd "$SCRIPT_DIR"
    else
        echo "Error: Tokenizer library missing. Place it at: $TOKENIZER"
        exit 1
    fi
fi

# ── Step 4: Generate cross file (android) ───────────────────────────────
CROSS_ARGS=""
if [ "$PLATFORM" = "android" ]; then
    CROSS_FILE="$SCRIPT_DIR/cross/android-aarch64.cross"
    sed "s|@ANDROID_NDK@|$ANDROID_NDK|g" \
        "$SCRIPT_DIR/cross/android-aarch64.cross.in" > "$CROSS_FILE"
    CROSS_ARGS="--cross-file $CROSS_FILE"
fi

# ── Step 5: Parse targets into meson options ────────────────────────────
ENABLE_API=false
ENABLE_API_TEST=false

if [ "$TARGETS" = "all" ]; then
    ENABLE_API=true
    ENABLE_API_TEST=true
else
    IFS=',' read -ra T <<< "$TARGETS"
    for t in "${T[@]}"; do
        case "$(echo "$t" | tr -d ' ')" in
            api)      ENABLE_API=true ;;
            api-test) ENABLE_API=true; ENABLE_API_TEST=true ;;
            src)      ;; # src is always built
            qnn)      ENABLE_QNN=true ;;
        esac
    done
fi

# ── Step 6: Meson setup ────────────────────────────────────────────────
MESON_OPTS=(
    --buildtype=release
    -Denable-qnn=$ENABLE_QNN
    -Denable-api=$ENABLE_API
    -Denable-api-test=$ENABLE_API_TEST
)

if [ "$PLATFORM" = "android" ]; then
    MESON_OPTS+=(-Dplatform=android)
else
    MESON_OPTS+=(-Dnntrainer_builddir=builddir_x86)
fi

echo "[3] Configuring meson..."
if [ "$CLEAN" = true ] || [ ! -f "$BUILD_DIR/build.ninja" ]; then
    rm -rf "$BUILD_DIR"
    meson setup "$BUILD_DIR" "$SCRIPT_DIR" $CROSS_ARGS "${MESON_OPTS[@]}"
else
    meson setup "$BUILD_DIR" "$SCRIPT_DIR" --reconfigure $CROSS_ARGS "${MESON_OPTS[@]}" || true
fi

# ── Step 7: Build ───────────────────────────────────────────────────────
echo "[4] Building..."
ninja -C "$BUILD_DIR" -j $(nproc)

echo ""
echo "=== Build completed ==="
echo "Artifacts in: $BUILD_DIR"

if [ "$PLATFORM" = "x86" ]; then
    NNTRAINER_BUILD="${NNTRAINER_BUILD:-$NNTRAINER_ROOT/builddir_x86}"
    echo ""
    echo "Run executable:"
    echo "  LD_LIBRARY_PATH=$NNTRAINER_BUILD/nntrainer:$NNTRAINER_BUILD/api/ccapi:$BUILD_DIR/src:$BUILD_DIR/api \\"
    echo "    $BUILD_DIR/src/quick_dot_ai <model_path> [input_prompt]"
fi

if [ "$PLATFORM" = "android" ]; then
    echo ""
    echo "Install to device:"
    echo "  ./install_android.sh"
fi

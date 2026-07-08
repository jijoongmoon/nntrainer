#!/bin/bash
# Unified build script for causallm-extension
#
# Usage:
#   ./build.sh                                          # x86, all targets
#   ./build.sh --platform=android                       # android, all targets
#   ./build.sh --platform=android --target=src          # android, src only
#   ./build.sh --platform=x86 --target=src,api          # x86, src + api
#   ./build.sh --platform=android --enable-qnn          # android with QNN
#   ./build.sh --platform=android --enable-qnn --qnn-models  # + QuickAI QNN models
#   ./build.sh --platform=android --nntr-threads=4      # override nntrainer thread count
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
# Build the QuickAI product QNN models (src/models/qnn: gauss-*/gemma4-*).
# OFF by default; enable with --qnn-models. The libqnn_context.so QNN infra
# still builds with --enable-qnn alone. Multimodal QNN models
# (gauss-3.8-vit-qnn, vjepa2-qnn) stay excluded — see
# docs/qnn-model-main-adaptation-todo.ko.md.
ENABLE_QNN_MODELS=false
ENABLE_EXPERIMENTAL_MULTIMODAL=false
# nntrainer compute thread count.
#   Compile-time: -Dnntr-num-threads=N passed to nntrainer meson build
#   Runtime:      NNTR_NUM_THREADS env var overrides compile default
# Priority: NNTR_NUM_THREADS env > compile flag > hardware_concurrency/2
NNTR_THREADS="${NNTR_THREADS:-7}"

for arg in "$@"; do
    case "$arg" in
        --platform=*)  PLATFORM="${arg#*=}" ;;
        --target=*)    TARGETS="${arg#*=}" ;;
        --clean)       CLEAN=true ;;
        --enable-qnn)  ENABLE_QNN=true ;;
        --qnn-models)  ENABLE_QNN_MODELS=true ;;
        --enable-experimental-multimodal) ENABLE_EXPERIMENTAL_MULTIMODAL=true ;;
        --nntr-threads=*) NNTR_THREADS="${arg#*=}" ;;
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
# SCRIPT_DIR is nntrainer/Applications/CausalLM (this script's home since the
# Slice E move). QUICKAI_ROOT is the Quick.AI superproject root, 3 levels up
# (CausalLM -> Applications -> nntrainer -> Quick.AI). The QuickAI meson
# project (meson.build, meson_options.txt, src/, api/, api-app/) stays at
# QUICKAI_ROOT -- only this shell script (and its siblings) moved under
# CausalLM.
QUICKAI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
NNTRAINER_ROOT="$QUICKAI_ROOT/nntrainer"
XGRAMMAR_ROOT="$SCRIPT_DIR/xgrammar"
CAUSALLM_ROOT="$SCRIPT_DIR"

# Build directory (stays under the QuickAI project root, alongside meson.build)
if [ "$PLATFORM" = "android" ]; then
    BUILD_DIR="$QUICKAI_ROOT/builddir_android"
else
    BUILD_DIR="$QUICKAI_ROOT/builddir_x86"
fi

echo "=== Quick-Dot-AI Unified Build ==="
echo "PLATFORM:      $PLATFORM"
echo "TARGETS:       $TARGETS"
echo "ENABLE_QNN:    $ENABLE_QNN"
echo "NNTR_THREADS:  $NNTR_THREADS"
echo "BUILD_DIR:     $BUILD_DIR"
echo ""

# ── Step 0: Check submodules ────────────────────────────────────────────
if [ ! -f "$NNTRAINER_ROOT/meson.build" ]; then
    echo "[0] Initializing nntrainer submodule..."
    git -C "$QUICKAI_ROOT" submodule update --init --recursive --depth 1
fi

# xgrammar submodule: since the Slice A relocation it lives INSIDE the nntrainer
# submodule at Applications/CausalLM/xgrammar (src/meson.build compiles
# xgrammar/cpp/*.cc from causallm_root; root meson.build adds its include dirs).
# A missing checkout makes meson configuration fail with:
#   "ERROR: File .../xgrammar/cpp/compiled_grammar.cc does not exist."
# Init it from the nntrainer submodule, not the QuickAI superproject.
if [ ! -f "$XGRAMMAR_ROOT/cpp/compiled_grammar.cc" ]; then
    echo "[0] Initializing xgrammar submodule..."
    git -C "$NNTRAINER_ROOT" submodule update --init Applications/CausalLM/xgrammar
fi

# xgrammar nested submodule: only dlpack is required by the build
# (xgrammar/3rdparty/dlpack/include, used e.g. by grammar_matcher.cc).
# cpptrace is compiled out (guarded by XGRAMMAR_ENABLE_CPPTRACE != 1) and
# googletest is test-only, so we deliberately avoid --recursive to skip those
# large, unnecessary clones.
if [ ! -d "$XGRAMMAR_ROOT/3rdparty/dlpack/include" ]; then
    echo "[0] Initializing xgrammar nested submodule (dlpack)..."
    git -C "$XGRAMMAR_ROOT" submodule update --init 3rdparty/dlpack
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
    NNTRAINER_OPTS_MARKER="$NNTRAINER_ROOT/builddir/.qai_build_opts"


    # nntrainer build options.
    #   mmap-read=false : load weights into a contiguous heap buffer. The
    #     upstream default is true (weights mmap'd), which makes the
    #     memory-bandwidth-bound decode loop ~10x+ slower. ALWAYS force false.
    NNTRAINER_EXTRA_OPTS="-Dmmap-read=false -Dnntr-num-threads=$NNTR_THREADS"
    if [ "$ENABLE_QNN" = true ]; then
        echo "      nntrainer QNN backend ON (vendored)"
        NNTRAINER_EXTRA_OPTS="$NNTRAINER_EXTRA_OPTS -Denable-npu=true"
    fi

    # Rebuild when: --clean, the lib is missing, OR the requested options differ
    # from the last build. The options check prevents a stale lib built with
    # different flags (e.g. the slow mmap-read=true default) from silently
    # persisting — build.sh otherwise reuses any existing libnntrainer.so.
    NNTR_PREV_OPTS="$(cat "$NNTRAINER_OPTS_MARKER" 2>/dev/null || true)"
    if [ "$CLEAN" = true ] || [ ! -f "$NNTRAINER_ANDROID_LIB" ] || \
       [ "$NNTR_PREV_OPTS" != "$NNTRAINER_EXTRA_OPTS" ]; then
        echo "[1] Building nntrainer for Android ($NNTRAINER_EXTRA_OPTS)..."
        cd "$NNTRAINER_ROOT"
        # Force a clean reconfigure on --clean or when options changed so the
        # new flags actually take effect.
        if [ "$CLEAN" = true ] || [ "$NNTR_PREV_OPTS" != "$NNTRAINER_EXTRA_OPTS" ]; then
            rm -rf builddir
        fi
        ./tools/package_android.sh $NNTRAINER_EXTRA_OPTS
        mkdir -p "$(dirname "$NNTRAINER_OPTS_MARKER")"
        printf '%s' "$NNTRAINER_EXTRA_OPTS" > "$NNTRAINER_OPTS_MARKER"
        cd "$SCRIPT_DIR"
    else
        echo "[1] nntrainer (android) already built with matching options ($NNTRAINER_EXTRA_OPTS)."
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
    CROSS_FILE="$QUICKAI_ROOT/cross/android-aarch64.cross"
    sed "s|@ANDROID_NDK@|$ANDROID_NDK|g" \
        "$QUICKAI_ROOT/cross/android-aarch64.cross.in" > "$CROSS_FILE"
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
    -Denable-qnn-models=$ENABLE_QNN_MODELS
    -Denable-experimental-multimodal=$ENABLE_EXPERIMENTAL_MULTIMODAL
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
    meson setup "$BUILD_DIR" "$QUICKAI_ROOT" $CROSS_ARGS "${MESON_OPTS[@]}"
else
    meson setup "$BUILD_DIR" "$QUICKAI_ROOT" --reconfigure $CROSS_ARGS "${MESON_OPTS[@]}" || true
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
    # Stage freshly built libraries into install_libs/ and sync them into the
    # QuickDotAI AAR's prebuilt_libs/ so the AAR never ships a stale .so (the
    # AAR consumes prebuilt libs only — it does not compile native code).
    echo ""
    echo "[5] Staging libraries and syncing AAR prebuilt_libs/..."
    "$SCRIPT_DIR/apk_install_android.sh"
    PREBUILT_DIR="$SCRIPT_DIR/Android/QuickDotAI/prebuilt_libs"
    if [ -d "$PREBUILT_DIR" ]; then
        cp "$QUICKAI_ROOT"/install_libs/*.so "$PREBUILT_DIR"/
        echo "      Synced $(ls "$QUICKAI_ROOT"/install_libs/*.so 2>/dev/null | wc -l) .so into prebuilt_libs/"
    else
        echo "      (skipped: $PREBUILT_DIR not found)"
    fi

    echo ""
    echo "Install to device:"
    echo "  ./install_android.sh"
fi

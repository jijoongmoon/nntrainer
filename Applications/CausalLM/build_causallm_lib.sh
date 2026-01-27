#!/bin/bash

# Build script for libcausallm.so library
# This script builds the CausalLM API library with all model sources
set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    local timestamp=$(date "+%H:%M:%S")
    echo -e "${YELLOW}[$timestamp] [INFO]${NC} $1"
}

log_success() {
    local timestamp=$(date "+%H:%M:%S")
    echo -e "${GREEN}[$timestamp] [SUCCESS]${NC} $1"
}

log_warning() {
    local timestamp=$(date "+%H:%M:%S")
    echo -e "${CYAN}[$timestamp] [WARNING]${NC} $1"
}

log_error() {
    local timestamp=$(date "+%H:%M:%S")
    echo -e "${RED}[$timestamp] [ERROR]${NC} $1"
}

log_step() {
    local step="$1"
    local total="$2"
    local desc="$3"
    echo ""
    echo "[$step/$total] $desc"
    echo "----------------------------------------"
}

# Start time measurement
START_TIME=$(date +%s)

log_info "Build CausalLM Library (libcausallm.so)"
log_info "========================================"
log_info "Working directory: $(pwd)"
echo ""

# Check if NDK path is set
log_info "Checking environment..."
if [ -z "$ANDROID_NDK" ]; then
    log_error "ANDROID_NDK is not set"
    log_error "Please set it to your Android NDK path"
    log_error "Example: export ANDROID_NDK=/path/to/android-ndk-r21d"
    exit 1
fi
log_success "ANDROID_NDK found: $ANDROID_NDK"

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export NNTRAINER_ROOT

log_info "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo ""

# Step 1: Check nntrainer library
log_step "1" "5" "Check nntrainer library"
log_info "Checking: $NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so"
if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    log_warning "nntrainer library not found"
    log_info "Building nntrainer for Android..."
    log_info "Changing directory to: $NNTRAINER_ROOT"
    cd "$NNTRAINER_ROOT"

    log_info "Cleaning old builddir..."
    if [ -d "$NNTRAINER_ROOT/builddir" ]; then
        rm -rf builddir
        log_success "Old builddir removed"
    fi

    log_info "Running ./tools/package_android.sh -Dmmap-read=false"
    ./tools/package_android.sh -Dmmap-read=false

    log_success "nntrainer build completed"
else
    log_success "nntrainer already built (skipping build)"
fi

# Verify nntrainer library
if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    log_error "nntrainer library not found after build"
    exit 1
fi
log_info "nntrainer library verified"

# Step 2: Check tokenizer library
log_step "2" "5" "Check tokenizer library"
cd "$SCRIPT_DIR"
log_info "Checking: $SCRIPT_DIR/lib/libtokenizers_android_c.a"
if [ ! -f "lib/libtokenizers_android_c.a" ]; then
    log_warning "libtokenizers_android_c.a not found in lib directory"
    log_info "Attempting to build tokenizer library..."
    if [ -f "build_tokenizer_android.sh" ]; then
        log_info "Running: ./build_tokenizer_android.sh"
        ./build_tokenizer_android.sh
        log_success "tokenizer library built"
    else
        log_error "tokenizer library not found and build script is missing"
        log_error "Please build or download the tokenizer library for Android arm64-v8a"
        log_error "and place it in: $SCRIPT_DIR/lib/libtokenizers_android_c.a"
        exit 1
    fi
else
    log_success "tokenizer library found"
fi

# Step 3: Prepare json.hpp
log_step "3" "5" "Check json.hpp"
log_info "Checking: $SCRIPT_DIR/json.hpp"
if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
    log_warning "json.hpp not found"
    log_info "Downloading json.hpp..."
    log_info "Running: $NNTRAINER_ROOT/jni/prepare_encoder.sh $NNTRAINER_ROOT/builddir 0.2"
    "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir" "0.2"
    if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
        log_error "Failed to download json.hpp"
        exit 1
    fi
    log_success "json.hpp downloaded"
else
    log_success "json.hpp found"
fi

# Step 4: Clean previous build
log_step "4" "5" "Clean previous build"
log_info "Build directory: $SCRIPT_DIR/jni"
cd "$SCRIPT_DIR/jni"

if [ -f "libs/arm64-v8a/libcausallm.so" ]; then
    log_info "Removing: libs/arm64-v8a/libcausallm.so"
    rm -f libs/arm64-v8a/libcausallm.so
    log_success "Clean complete"
else
    log_info "No previous libcausallm.so found (skipping clean)"
fi

# Step 5: Build libcausallm.so
log_step "5" "5" "Build libcausallm.so"
log_info "Build script: Android.mk.lib"
log_info "Application.mk: Application.mk"
log_info "Start ndk-build..."

# Run ndk-build and capture output
BUILD_START=$(date +%s)
ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk.lib NDK_APPLICATION_MK=./Application.mk -j $(nproc) 2>&1 | while IFS= read -r line; do
    log_info "  $line"
done
BUILD_STATUS=${PIPESTATUS[0]}
BUILD_END=$(date +%s)
BUILD_DURATION=$((BUILD_END - BUILD_START))

if [ $BUILD_STATUS -eq 0 ]; then
    log_info "ndk-build completed successfully (${BUILD_DURATION}s)"
else
    log_error "ndk-build failed with status: $BUILD_STATUS"
    exit 1
fi

# Verify output
if [ -f "libs/arm64-v8a/libcausallm.so" ]; then
    local file_size=$(du -h "libs/arm64-v8a/libcausallm.so" | cut -f1)
    log_success "libcausallm.so built successfully"
    log_info "   Location: $SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm.so"
    log_info "   Size: $file_size"
else
    log_error "libcausallm.so not found after build"
    exit 1
fi

# End time measurement
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

# Summary
echo ""
log_info "========================================"
log_info "Build Complete"
log_info "========================================"
log_info "Total time: ${TOTAL_DURATION}s"
log_info "Output: $SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm.so"
log_info "Size: $file_size"
echo ""
log_success "All steps completed successfully!"

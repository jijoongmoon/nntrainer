#!/bin/bash

# Build script for test_causal_lm test app
# This script builds the test application that uses libcausallm.so
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

log_info "Build Test App (test_causal_lm)"
log_info "========================================"
log_info "Working directory: $(pwd)"
echo ""

# Check if NDK path is set
log_info "Checking environment..."
if [ -z "$ANDROID_NDK" ]; then
    log_error "ANDROID_NDK is not set"
    log_error "Please set it to your Android NDK path"
    exit 1
fi
log_success "ANDROID_NDK found: $ANDROID_NDK"

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info "SCRIPT_DIR: $SCRIPT_DIR"
echo ""

# Step 1: Check libcausallm.so
log_step "1" "3" "Check libcausallm.so dependency"
lib_path="$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm.so"
log_info "Checking: $lib_path"

if [ ! -f "$lib_path" ]; then
    log_error "libcausallm.so not found"
    log_error "Please run: ./build_causallm_lib.sh first"
    log_error "Location expected: $lib_path"
    exit 1
fi

lib_size=$(du -h "$lib_path" | cut -f1)
lib_time=$(stat -c %y "$lib_path" 2>/dev/null | cut -d' ' -f1,2 | cut -d. -f1)
log_success "libcausallm.so found"
log_info "   Size: $lib_size"
log_info "   Modified: $lib_time"

# Step 2: Check test_api.cpp source
log_step "2" "3" "Check test source file"
src_path="$SCRIPT_DIR/api/test_api.cpp"
log_info "Checking: $src_path"

if [ ! -f "$src_path" ]; then
    log_error "test_api.cpp not found"
    exit 1
fi
log_success "test_api.cpp found"

# Step 3: Build test_causal_lm
log_step "3" "3" "Build test_causal_lm executable"
log_info "Build directory: $SCRIPT_DIR/jni"
cd "$SCRIPT_DIR/jni"

# Clean previous test build
if [ -f "libs/arm64-v8a/test_causal_lm" ]; then
    log_info "Removing: libs/arm64-v8a/test_causal_lm"
    rm -f libs/arm64-v8a/test_causal_lm
fi

log_info "Build script: Android.mk.test"
log_info "Application.mk: Application.mk"
log_info "Start ndk-build..."

# Run ndk-build and capture output
BUILD_START=$(date +%s)
ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk.test NDK_APPLICATION_MK=./Application.mk -j $(nproc) 2>&1 | while IFS= read -r line; do
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
output_path="$SCRIPT_DIR/jni/libs/arm64-v8a/test_causal_lm"
log_info "Checking output: $output_path"

if [ -f "$output_path" ]; then
    file_size=$(du -h "$output_path" | cut -f1)
    log_success "test_causal_lm built successfully"
    log_info "   Location: $output_path"
    log_info "   Size: $file_size"

    # Check file type
    file_type=$(file "$output_path" | cut -d: -f2-)
    log_info "   Type: $file_type"
else
    log_error "test_causal_lm not found after build"
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
log_info "Output: $output_path"
log_info "Size: $file_size"
echo ""
log_success "All steps completed successfully!"
log_info ""
log_info "To run the test app:"
log_info "  ./run_test.sh"

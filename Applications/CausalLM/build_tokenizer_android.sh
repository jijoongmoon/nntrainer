#!/bin/bash

# Script to build the CausalLM tokenizers_c static library for Android.
#
# Source of truth is the in-tree crate Applications/CausalLM/tokenizers_c_win,
# NOT the external mlc-ai/tokenizers-cpp checkout this script used to clone.
# That external repo only exports the original C ABI; every entry point added
# to the C ABI since then (currently tokenizers_snapshot_from_json /
# tokenizers_new_from_snapshot / tokenizers_snapshot_free, the persistent
# tokenizer snapshot cache) lives in the in-tree crate. Building Android from
# the external repo therefore produced an archive whose symbol set silently
# lagged the tracked x86 archive lib/libtokenizers_c.a, and the next ndk-build
# failed with undefined references that looked like an application bug.
#
# The in-tree crate exports a strict superset of the external ABI -- its symbol
# set is identical to the tracked x86 archive -- so all lanes (x86 archive,
# build_tokenizer_windows.ps1, this script) now come from one source.
#
# usage: ./build_tokenizer_android.sh [abi]      (default: arm64-v8a)

set -e

# Default target ABI
TARGET_ABI="${1:-arm64-v8a}"

echo "Building tokenizers_c library for Android $TARGET_ABI..."

# Check prerequisites
if [ -z "$ANDROID_NDK" ]; then
    for candidate in "$HOME"/Android/Sdk/ndk/*; do
        [ -d "$candidate" ] && ANDROID_NDK="$candidate"
    done
fi

if [ ! -d "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK is not set and no NDK was found under ~/Android/Sdk/ndk."
    exit 1
fi

# Check if rust is installed
if ! command -v rustc &> /dev/null || ! command -v cargo &> /dev/null; then
    if [ -x "$HOME/.cargo/bin/cargo" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        echo "Error: Rust is not installed. Please install Rust from https://rustup.rs/"
        exit 1
    fi
fi

# Map Android ABI to Rust target and to the NDK cross-compiler prefix.
case "$TARGET_ABI" in
    "arm64-v8a")
        RUST_TARGET="aarch64-linux-android"
        CLANG_PREFIX="aarch64-linux-android"
        ;;
    "armeabi-v7a")
        RUST_TARGET="armv7-linux-androideabi"
        CLANG_PREFIX="armv7a-linux-androideabi"
        ;;
    "x86")
        RUST_TARGET="i686-linux-android"
        CLANG_PREFIX="i686-linux-android"
        ;;
    "x86_64")
        RUST_TARGET="x86_64-linux-android"
        CLANG_PREFIX="x86_64-linux-android"
        ;;
    *)
        echo "Error: Invalid target ABI: $TARGET_ABI"
        echo "Supported ABIs: arm64-v8a, armeabi-v7a, x86, x86_64"
        exit 1
        ;;
esac
echo "Target ABI: $TARGET_ABI"

# Must track APP_PLATFORM in jni/Application.mk: this archive is linked into
# modules built against that API level.
ANDROID_API=29

# Install Rust target if not already installed
echo "Checking Rust target: $RUST_TARGET"
if ! rustup target list --installed | grep -q "^$RUST_TARGET$"; then
    echo "Installing Rust target: $RUST_TARGET"
    rustup target add "$RUST_TARGET"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$SCRIPT_DIR/tokenizers_c_win"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/tokenizers-cpp-build}"
TARGET_DIR="$BUILD_DIR/target-$TARGET_ABI"

# Detect platform for NDK paths
if [[ "$OSTYPE" == "darwin"* ]]; then
    NDK_HOST="darwin-x86_64"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    NDK_HOST="linux-x86_64"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    NDK_HOST="windows-x86_64"
else
    echo "Warning: Unknown platform $OSTYPE, assuming linux-x86_64"
    NDK_HOST="linux-x86_64"
fi

TOOLCHAIN="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin"
TARGET_CC="$TOOLCHAIN/${CLANG_PREFIX}${ANDROID_API}-clang"
if [ ! -x "$TARGET_CC" ]; then
    echo "Error: $TARGET_CC not found."
    exit 1
fi

# The crate is a staticlib, so rustc only archives -- but the tokenizers
# dependency pulls onig_sys and esaxx-rs, which compile C/C++ and so need a
# cross compiler for the target triple.
RUST_TARGET_UPPER=$(echo "$RUST_TARGET" | tr 'a-z-' 'A-Z_')
RUST_TARGET_LOWER=$(echo "$RUST_TARGET" | tr '-' '_')
export CARGO_TARGET_${RUST_TARGET_UPPER}_LINKER="$TARGET_CC"
export CC_${RUST_TARGET_LOWER}="$TARGET_CC"
export CXX_${RUST_TARGET_LOWER}="${TARGET_CC}++"
export AR_${RUST_TARGET_LOWER}="$TOOLCHAIN/llvm-ar"

echo "Building crate: $CRATE_DIR"
echo "Target dir:     $TARGET_DIR"

cargo build \
    --manifest-path "$CRATE_DIR/Cargo.toml" \
    --target-dir "$TARGET_DIR" \
    --target "$RUST_TARGET" \
    --release \
    --locked

BUILT="$TARGET_DIR/$RUST_TARGET/release/libtokenizers_c.a"
if [ ! -f "$BUILT" ]; then
    echo "Error: libtokenizers_c.a was not produced at $BUILT"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/lib/$TARGET_ABI"
cp -f "$BUILT" "$SCRIPT_DIR/lib/$TARGET_ABI/libtokenizers_android_c.a"

# jni/Android.mk links ../lib/libtokenizers_android_c.a for the default ABI.
if [ "$TARGET_ABI" = "arm64-v8a" ]; then
    cp -f "$BUILT" "$SCRIPT_DIR/lib/libtokenizers_android_c.a"
fi

echo "Build completed successfully!"
echo "Library copied to: $SCRIPT_DIR/lib/$TARGET_ABI/libtokenizers_android_c.a"
if [ "$TARGET_ABI" = "arm64-v8a" ]; then
    echo "Also copied to: $SCRIPT_DIR/lib/libtokenizers_android_c.a"
fi

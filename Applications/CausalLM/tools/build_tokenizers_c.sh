#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
#
# @file build_tokenizers_c.sh
# @brief Build libtokenizers_c.a from the in-tree crate, reproducibly.
#
# The CausalLM tokenizer is a C ABI (tokenizers_c.h) over the Rust
# `tokenizers` crate. Its only source is the in-tree crate
# Applications/CausalLM/tokenizers_c_win, pinned by its Cargo.lock; this
# script turns that crate into the static archive the application links.
# Applications/CausalLM/lib/libtokenizers_c.a is a convenience copy of what
# this script produces for x86_64 Linux, so a checkout builds without Rust.
#
# Reproducibility: `cargo build --locked` resolves exactly the versions in
# Cargo.lock, local paths (cargo home, crate dir) are remapped out of the
# panic-location strings and the C objects of the dependencies, and the
# debug sections the precompiled Rust standard library carries are stripped
# with a deterministic archive writer. Two runs with the same rustc and the
# same C compiler give the same archive bytes; the script prints the rustc
# version and the sha256 of the result so a refresh can be checked.
#
# Size: the precompiled standard library objects also carry LLVM bitcode
# (.llvmbc/.llvmcmd, kept by rustup for cross-crate LTO). A C/C++ link never
# reads it, and it is about a third of the archive, so it is always removed.
# The result must stay below MAX_ARCHIVE_BYTES (30 MiB, the size limit for a
# file in a pull request); the script fails instead of producing a larger
# archive.
#
# usage: tools/build_tokenizers_c.sh [options]
#   --target=TARGET    host (default), x86_64-linux, aarch64-linux, android
#   --abi=ABI          Android ABI: arm64-v8a (default), armeabi-v7a, x86,
#                      x86_64. Implies --target=android.
#   --api=LEVEL        Android API level (default 29, = jni/Application.mk)
#   --out-dir=DIR      output root (default Applications/CausalLM/tokenizers-build)
#                      The archive lands in DIR/<rust triple>/libtokenizers_c.a
#   --update-prebuilt  also copy the result over the archive the build uses by
#                      default: lib/libtokenizers_c.a (x86_64 Linux) or
#                      lib/libtokenizers_android_c.a (Android arm64-v8a)
#   --no-strip         keep the debug sections (LLVM bitcode is still removed;
#                      the result may exceed the size limit, which then only
#                      warns, and cannot be used with --update-prebuilt)
#   --offline          do not touch the network (crates must be in cargo home)
#   -h, --help         this text
#
# Windows (MSVC) builds the same crate from meson through
# build_tokenizer_windows.ps1; there is nothing to prebuild for it.
#
# Use a freshly built archive without replacing the tracked one:
#   meson setup build -Dcausallm-tokenizer-lib=<out-dir>/<triple>/libtokenizers_c.a

set -euo pipefail

usage() {
  sed -n '/^# usage:/,/^# Windows/p' "$0" | sed '$d; s/^# \{0,1\}//'
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAUSALLM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CRATE_DIR="$CAUSALLM_ROOT/tokenizers_c_win"

TARGET=host
ABI=""
API=29
OUT_DIR="$CAUSALLM_ROOT/tokenizers-build"
UPDATE_PREBUILT=0
STRIP=1
OFFLINE=()
MAX_ARCHIVE_BYTES=$((30 * 1024 * 1024))

for arg in "$@"; do
  case "$arg" in
    --target=*) TARGET="${arg#*=}" ;;
    --abi=*) ABI="${arg#*=}"; TARGET=android ;;
    --api=*) API="${arg#*=}" ;;
    --out-dir=*) OUT_DIR="${arg#*=}" ;;
    --update-prebuilt) UPDATE_PREBUILT=1 ;;
    --no-strip) STRIP=0 ;;
    --offline) OFFLINE=(--offline) ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: unknown option $arg" >&2; usage >&2; exit 1 ;;
  esac
done

if [ "$STRIP" = 0 ] && [ "$UPDATE_PREBUILT" = 1 ]; then
  echo "Error: --no-strip cannot be combined with --update-prebuilt" >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  if [ -x "$HOME/.cargo/bin/cargo" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  else
    echo "Error: cargo not found. Install Rust from https://rustup.rs/" >&2
    exit 1
  fi
fi

HOST_TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
STRIP_TOOL=""
PREBUILT=""

case "$TARGET" in
  host) RUST_TARGET="$HOST_TRIPLE" ;;
  x86_64-linux) RUST_TARGET=x86_64-unknown-linux-gnu ;;
  aarch64-linux) RUST_TARGET=aarch64-unknown-linux-gnu ;;
  android)
    ABI="${ABI:-arm64-v8a}"
    case "$ABI" in
      arm64-v8a) RUST_TARGET=aarch64-linux-android; CLANG_PREFIX=aarch64-linux-android ;;
      armeabi-v7a) RUST_TARGET=armv7-linux-androideabi; CLANG_PREFIX=armv7a-linux-androideabi ;;
      x86) RUST_TARGET=i686-linux-android; CLANG_PREFIX=i686-linux-android ;;
      x86_64) RUST_TARGET=x86_64-linux-android; CLANG_PREFIX=x86_64-linux-android ;;
      *) echo "Error: unsupported Android ABI $ABI" >&2; exit 1 ;;
    esac
    ;;
  *windows*)
    echo "Error: Windows builds the crate through build_tokenizer_windows.ps1" >&2
    exit 1
    ;;
  *) echo "Error: unknown target $TARGET" >&2; exit 1 ;;
esac

if command -v rustup >/dev/null 2>&1 &&
  ! rustup target list --installed | grep -qx "$RUST_TARGET"; then
  echo "Installing Rust target $RUST_TARGET (rustup)"
  rustup target add "$RUST_TARGET"
fi

TRIPLE_ENV="$(echo "$RUST_TARGET" | tr '-' '_')"
TRIPLE_ENV_UPPER="$(echo "$TRIPLE_ENV" | tr 'a-z' 'A-Z')"

if [ "$TARGET" = android ]; then
  if [ -z "${ANDROID_NDK:-}" ]; then
    echo "Error: ANDROID_NDK is not set" >&2
    exit 1
  fi
  case "$(uname -s)" in
    Darwin) NDK_HOST=darwin-x86_64 ;;
    *) NDK_HOST=linux-x86_64 ;;
  esac
  TOOLCHAIN="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin"
  TARGET_CC="$TOOLCHAIN/${CLANG_PREFIX}${API}-clang"
  if [ ! -x "$TARGET_CC" ]; then
    echo "Error: $TARGET_CC not found" >&2
    exit 1
  fi
  # The crate itself is only archived, but onig_sys and esaxx-rs compile
  # C/C++ and need the cross compiler of the target.
  export "CARGO_TARGET_${TRIPLE_ENV_UPPER}_LINKER=$TARGET_CC"
  export "CC_${TRIPLE_ENV}=$TARGET_CC"
  export "CXX_${TRIPLE_ENV}=${TARGET_CC}++"
  export "AR_${TRIPLE_ENV}=$TOOLCHAIN/llvm-ar"
  STRIP_TOOL="$TOOLCHAIN/llvm-strip"
  [ "$ABI" = arm64-v8a ] && PREBUILT="$CAUSALLM_ROOT/lib/libtokenizers_android_c.a"
else
  if [ "$RUST_TARGET" != "$HOST_TRIPLE" ]; then
    # Cross Linux build: let the caller name the compilers, as cargo expects
    # (CC_<triple>, CXX_<triple>, AR_<triple>).
    echo "Note: cross build $HOST_TRIPLE -> $RUST_TARGET uses CC_${TRIPLE_ENV}"
  fi
  if command -v llvm-strip >/dev/null 2>&1; then
    STRIP_TOOL=llvm-strip
  else
    STRIP_TOOL=strip
  fi
  [ "$RUST_TARGET" = x86_64-unknown-linux-gnu ] &&
    PREBUILT="$CAUSALLM_ROOT/lib/libtokenizers_c.a"
fi

# Keep build-machine paths out of the archive.
CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
REMAP="--remap-path-prefix=$CARGO_HOME_DIR=/cargo --remap-path-prefix=$CRATE_DIR=/tokenizers_c"
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$REMAP"
PREFIX_MAP="-ffile-prefix-map=$CARGO_HOME_DIR=/cargo"
export "CFLAGS_${TRIPLE_ENV}=${CFLAGS:+$CFLAGS }$PREFIX_MAP"
export "CXXFLAGS_${TRIPLE_ENV}=${CXXFLAGS:+$CXXFLAGS }$PREFIX_MAP"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"

TARGET_DIR="$OUT_DIR/cargo-target"
DEST_DIR="$OUT_DIR/$RUST_TARGET"
mkdir -p "$DEST_DIR"

echo "crate:  $CRATE_DIR"
echo "target: $RUST_TARGET"
echo "rustc:  $(rustc -V)"

cargo build \
  --manifest-path "$CRATE_DIR/Cargo.toml" \
  --target-dir "$TARGET_DIR" \
  --target "$RUST_TARGET" \
  --release \
  --locked \
  "${OFFLINE[@]}"

BUILT="$TARGET_DIR/$RUST_TARGET/release/libtokenizers_c.a"
if [ ! -f "$BUILT" ]; then
  echo "Error: cargo did not produce $BUILT" >&2
  exit 1
fi

OUT="$DEST_DIR/libtokenizers_c.a"
# The embedded bitcode comes from the precompiled standard library, so no
# RUSTFLAGS setting removes it; drop the sections from the archive instead.
STRIP_ARGS=(-D --remove-section=.llvmbc --remove-section=.llvmcmd)
[ "$STRIP" = 1 ] && STRIP_ARGS+=(--strip-debug)
"$STRIP_TOOL" "${STRIP_ARGS[@]}" -o "$OUT" "$BUILT"

SIZE="$(wc -c <"$OUT" | tr -d ' ')"
echo "size:   $SIZE bytes (limit $MAX_ARCHIVE_BYTES)"
if [ "$SIZE" -gt "$MAX_ARCHIVE_BYTES" ]; then
  if [ "$STRIP" = 1 ]; then
    echo "Error: $OUT is larger than $MAX_ARCHIVE_BYTES bytes" >&2
    exit 1
  fi
  echo "Warning: $OUT is larger than $MAX_ARCHIVE_BYTES bytes" >&2
fi

# Every entry point tokenizers_c.h declares must be exported.
if command -v nm >/dev/null 2>&1 || [ -x "${TOOLCHAIN:-}/llvm-nm" ]; then
  NM=nm
  [ "$TARGET" = android ] && NM="$TOOLCHAIN/llvm-nm"
  EXPORTED="$("$NM" -g --defined-only "$OUT" 2>/dev/null | awk '$2 == "T" { print $3 }')"
  for sym in $(sed -n 's/^.*[ *]\([a-z_]*tokenizers_[a-z_]*\)(.*$/\1/p' \
    "$CAUSALLM_ROOT/tokenizers_c.h"); do
    if ! grep -qx "$sym" <<<"$EXPORTED"; then
      echo "Error: $OUT does not export $sym" >&2
      exit 1
    fi
  done
fi

echo "built:  $OUT"
echo "sha256: $(sha256sum "$OUT" | cut -d' ' -f1)"

if [ "$UPDATE_PREBUILT" = 1 ]; then
  if [ -z "$PREBUILT" ]; then
    echo "Error: no default archive is linked for $RUST_TARGET" >&2
    exit 1
  fi
  cp -f "$OUT" "$PREBUILT"
  echo "updated: $PREBUILT"
fi

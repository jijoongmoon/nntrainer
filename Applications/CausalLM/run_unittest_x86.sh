#!/bin/bash
# Run CausalLM model unit tests on x86 (Ubuntu host).
# Usage: ./run_unittest_x86.sh [--rebuild] [--filter <gtest_filter>]
#   --rebuild        Wipe and recreate builddir
#   --filter <pat>   Pass --gtest_filter=<pat> to the test binary (e.g. '*Gemma4*')
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Dedicated builddir for x86 tests — avoids conflicting with the Android builddir.
BUILDDIR="$NNTRAINER_ROOT/builddir_x86_test"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "\n${YELLOW}[Step $1]${NC} $2\n${YELLOW}----------------------------------------${NC}"; }
log_header()  { echo -e "\n${CYAN}========================================\n $1\n========================================${NC}"; }

REBUILD=0
GTEST_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild) REBUILD=1; shift ;;
        --filter)  GTEST_FILTER="$2"; shift 2 ;;
        *) echo "Usage: $0 [--rebuild] [--filter <gtest_filter>]"; exit 1 ;;
    esac
done

log_header "CausalLM x86 Unit Tests"
log_info "NNTRAINER_ROOT: $NNTRAINER_ROOT"
log_info "BUILDDIR:       $BUILDDIR"
[[ -n "$GTEST_FILTER" ]] && log_info "Filter:         $GTEST_FILTER"

# ---------------------------------------------------------------------------
log_step "1/3" "Configure and build"
# ---------------------------------------------------------------------------

if [[ "$REBUILD" -eq 1 && -d "$BUILDDIR" ]]; then
    log_warning "--rebuild requested: removing $BUILDDIR"
    rm -rf "$BUILDDIR"
fi

cd "$NNTRAINER_ROOT"

MESON_OPTS=(
    -Denable-test=true
    -Denable-app=true
    -Denable-transformer=true
    -Denable-fp16=false
    -Dopenblas-num-threads=1
)

if [[ ! -d "$BUILDDIR" ]]; then
    log_info "Running meson setup (builddir_x86_test)..."
    meson setup "$BUILDDIR" "${MESON_OPTS[@]}"
else
    log_info "Reusing $BUILDDIR (pass --rebuild to recreate)"
fi

log_info "Building..."
ninja -C "$BUILDDIR" \
    Applications/CausalLM/unittest_causallm_models \
    Applications/CausalLM/nntr_quantize

# ---------------------------------------------------------------------------
log_step "2/3" "Check fixtures"
# ---------------------------------------------------------------------------

FIXTURE_DIR="$NNTRAINER_ROOT/test/unittest/models/causallm_reference"
if [[ ! -d "$FIXTURE_DIR" ]]; then
    log_info "Extracting fixture tarballs..."
    tar xzf "$NNTRAINER_ROOT/packaging/causallm_reference_generation.tar.gz" \
        -C "$NNTRAINER_ROOT/test/unittest/models/"
    tar xzf "$NNTRAINER_ROOT/packaging/causallm_reference_embedding.tar.gz" \
        -C "$NNTRAINER_ROOT/test/unittest/models/"
    log_success "Fixtures extracted to $FIXTURE_DIR"
else
    log_info "Fixtures already present"
fi

# ---------------------------------------------------------------------------
log_step "3/3" "Run tests"
# ---------------------------------------------------------------------------

TEST_BIN="$BUILDDIR/Applications/CausalLM/unittest_causallm_models"
QUANTIZE_BIN="$BUILDDIR/Applications/CausalLM/nntr_quantize"

if [[ ! -x "$TEST_BIN" ]]; then
    log_error "Test binary not found: $TEST_BIN"
    exit 1
fi

ARGS=()
[[ -n "$GTEST_FILTER" ]] && ARGS+=("--gtest_filter=$GTEST_FILTER")

log_info "Running unittest_causallm_models..."
NNTR_QUANTIZE_BIN="$QUANTIZE_BIN" "$TEST_BIN" "${ARGS[@]}"
EXIT=$?

echo ""
if [[ $EXIT -eq 0 ]]; then
    log_success "All tests passed."
else
    log_error "Some tests failed (exit code $EXIT)."
fi
exit $EXIT

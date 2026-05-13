#!/usr/bin/env sh
#
# tools/gpu_bench/run_baseline.sh
#
# Device-side OpenCL baseline runner. Designed to execute inside the tarball
# produced by tools/gpu_bench/package_android.sh after adb push + adb shell.
#
# Runs each unittest_opencl_kernels_* binary with the M0-1 profiler enabled
# (PROFILE build) and parses the per-kernel report into a JSON file conforming
# to docs/gpu/baseline_schema.json.
#
# Usage on device:
#   cd /data/local/tmp/gpu_bench
#   LD_LIBRARY_PATH=. ./run_baseline.sh [OUT_DIR]
#
# Output:
#   <OUT_DIR>/meta.json       device + build identification
#   <OUT_DIR>/<test>.log      raw stdout per binary
#   <OUT_DIR>/baseline.json   merged per-kernel records (schema v1)
#
# /bin/sh chosen for portability — Android shell is not bash.

set -e

OUT_DIR=${1:-./baseline_out}
mkdir -p "$OUT_DIR"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export LD_LIBRARY_PATH="$SCRIPT_DIR:${LD_LIBRARY_PATH:-}"

# Device + build metadata. getprop is only present on Android; gate it.
{
    echo "{"
    echo "  \"schema_version\": 1,"
    echo "  \"timestamp_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    if command -v getprop >/dev/null 2>&1; then
        echo "  \"device_model\": \"$(getprop ro.product.model)\","
        echo "  \"device_hardware\": \"$(getprop ro.hardware)\","
        echo "  \"android_release\": \"$(getprop ro.build.version.release)\","
        echo "  \"abi\": \"$(getprop ro.product.cpu.abi)\","
    fi
    echo "  \"uname\": \"$(uname -a | sed 's/"/\\"/g')\""
    echo "}"
} >"$OUT_DIR/meta.json"

TESTS="unittest_opencl_kernels_blas unittest_opencl_kernels_int4 unittest_opencl_kernels_qk_k"

# Header of M0-1 profiler report. Used as the awk anchor.
PROFILE_HEADER="=== OpenCL GPU Profile ==="

# Parse the per-kernel table out of a unittest log and append JSON records.
# Columns (from opencl_profiler.cpp::report):
#   kernel  calls  exec_total_us  avg_us  min_us  max_us  queued_us submit_us  %exec
# Note the kernel column is left-aligned width 48 and may contain spaces, so we
# count from the right: last 8 fields are the numbers; the rest is the name.
emit_records() {
    test_name=$1
    log_file=$2
    awk -v test_name="$test_name" -v anchor="$PROFILE_HEADER" '
    BEGIN { in_table = 0; sep_seen = 0; }
    index($0, anchor) { in_table = 1; next }
    in_table && /^-+$/ { sep_seen += 1; next }
    in_table && sep_seen == 0 { next }       # column header line
    in_table && sep_seen == 1 {
        # Row line; trim trailing % if present.
        line = $0
        # Stop when we reach the post-table separator (handled by sep_seen==2).
        # Split into fields. Last 8 numeric columns => fields NF-7 .. NF.
        if (NF < 9) next
        pct_field = $NF; sub(/%$/, "", pct_field)
        submit_us = $(NF-1)
        queued_us = $(NF-2)
        max_us    = $(NF-3)
        min_us    = $(NF-4)
        avg_us    = $(NF-5)
        exec_us   = $(NF-6)
        calls     = $(NF-7)
        # Kernel name = everything before NF-7 (joined with single space).
        name = $1
        for (i = 2; i <= NF-8; i++) name = name " " $i
        # Skip TOTAL row (has empty calls column rendering).
        if (name == "TOTAL") next
        if (printed) printf(",\n")
        printed = 1
        printf("    {")
        printf("\"binary\": \"%s\", ", test_name)
        printf("\"kernel\": \"%s\", ", name)
        printf("\"calls\": %d, ", calls)
        printf("\"exec_total_us\": %d, ", exec_us)
        printf("\"avg_us\": %s, ", avg_us)
        printf("\"min_us\": %d, ", min_us)
        printf("\"max_us\": %d, ", max_us)
        printf("\"queued_us\": %d, ", queued_us)
        printf("\"submit_us\": %d, ", submit_us)
        printf("\"pct_exec\": %s", pct_field)
        printf("}")
    }
    in_table && sep_seen == 2 { in_table = 0 }
    END { if (printed) printf("\n") }
    ' "$log_file"
}

{
    echo "{"
    echo "  \"schema_version\": 1,"
    echo "  \"records\": ["
    first_bin=1
    for t in $TESTS; do
        bin="$SCRIPT_DIR/$t"
        log="$OUT_DIR/$t.log"
        if [ ! -x "$bin" ]; then
            echo "[run_baseline] skipping missing binary: $t" >&2
            continue
        fi
        echo "[run_baseline] running $t" >&2
        # Profiler report goes to stdout via ClContext dtor. Capture both.
        "$bin" >"$log" 2>&1 || echo "[run_baseline] $t exited non-zero (see $log)" >&2
        records=$(emit_records "$t" "$log")
        if [ -n "$records" ]; then
            if [ $first_bin -eq 0 ]; then echo ","; fi
            first_bin=0
            printf "%s" "$records"
        fi
    done
    echo ""
    echo "  ]"
    echo "}"
} >"$OUT_DIR/baseline.json"

echo "[run_baseline] done. results in $OUT_DIR" >&2

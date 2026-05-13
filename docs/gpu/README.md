# GPU baseline measurement (M0-pre / M0-2)

This directory holds the validation infrastructure for the OpenCL GPU stack
rework toward [ML Drift](https://arxiv.org/abs/2505.00232) parity. See the
project plan for milestone context.

## Files

- `baseline_schema.json` — JSON Schema (draft 2020-12) describing the
  per-kernel records produced by `tools/gpu_bench/run_baseline.sh`. One record
  per `(binary, kernel)` pair with aggregated `calls / exec_total_us / avg_us /
  min_us / max_us / queued_us / submit_us / pct_exec`.

## Pipeline

```
host (Linux dev box / CI)            device (Android, e.g. Adreno)
-----------------------------        -----------------------------
package_android.sh   ------>  adb push nntrainer_gpu_bench_android.tar.gz
  -> tar.gz                          adb shell tar xzf
                                     adb shell ./run_baseline.sh
                                       -> meta.json
                                       -> *.log (raw stdout)
                                       -> baseline.json (schema v1)
                              <----  adb pull baseline_out/
                              (compare against parity targets)
```

### Step 1 — package on host

```sh
tools/gpu_bench/package_android.sh "$(pwd)" --arm-arch=armv8.2-a
```

Builds the OpenCL unit tests (`unittest_opencl_kernels_blas`, `..._int4`,
`..._qk_k`) with `-Denable-profile=true` so the M0-1 profiler captures every
`clEnqueueNDRangeKernel` via `clGetEventProfilingInfo`. Stages binaries +
`.so` deps + `run_baseline.sh` into `nntrainer_gpu_bench_android.tar.gz`.

Build tree is `builddir_gpu_bench/`, separate from the normal android build.

### Step 2 — push and run on device

```sh
adb push nntrainer_gpu_bench_android.tar.gz /data/local/tmp/
adb shell "mkdir -p /data/local/tmp/gpu_bench && \
           cd /data/local/tmp/gpu_bench && \
           tar xzf /data/local/tmp/nntrainer_gpu_bench_android.tar.gz && \
           LD_LIBRARY_PATH=. ./run_baseline.sh ./out"
adb pull /data/local/tmp/gpu_bench/out ./baseline_out
```

### Step 3 — inspect

```sh
cat baseline_out/meta.json       # device + build identity
cat baseline_out/baseline.json   # schema v1, machine-readable
less baseline_out/unittest_opencl_kernels_int4.log   # raw profiler table
```

Validate against the schema with any JSON Schema validator, e.g.

```sh
python3 -c "import json,jsonschema as j; \
  j.validate(json.load(open('baseline_out/baseline.json')), \
             json.load(open('docs/gpu/baseline_schema.json')))"
```

## POCL (host-side correctness)

For numerical correctness checks on the dev box without GPU hardware, install
POCL and re-run the same unittests against the CPU ICD:

```sh
sudo apt-get install -y pocl-opencl-icd clinfo
clinfo | head -20             # confirm POCL platform is visible
meson builddir_pocl -Denable-opencl=true -Denable-profile=false
ninja -C builddir_pocl test/unittest/unittest_opencl_kernels_int4
./builddir_pocl/test/unittest/unittest_opencl_kernels_int4
```

POCL profiling timestamps are wall-clock on CPU, not device-side GPU timing,
so use POCL only for correctness (`assert` / value comparison). Use a real
Adreno or Mali device for performance numbers.

## Schema versioning

`schema_version` is currently `1`. Bump on any breaking change to the record
shape; consumers should reject unknown versions rather than silently parse.

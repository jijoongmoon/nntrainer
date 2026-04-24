# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NNTrainer is a C++17 software framework for training and inference of neural networks on resource-constrained devices (Tizen, Android, Linux, Windows; arm64 and x86_64). Core use cases:

- On-device training / personalization / transfer learning
- LLM inference with aggressive memory optimization (FSU — Flash Storage Utilization, MoE expert caching, proactive weight loading)

The project ships a core C++ library (`nntrainer/`), a C API for Tizen (`api/capi`), a C++ API (`api/ccapi`), example applications (`Applications/`), an NNStreamer sub-filter (`nnstreamer/`), and JNI/Android build glue (`jni/`).

## Build System (Meson + Ninja)

Warnings are treated as errors (`werror=true`), C++17 is mandatory (`MIN_CPP_VERSION=201703L`), and default `buildtype=release`. Do not weaken warnings or disable checks.

Common commands:

```bash
# Configure (default: Linux/Ubuntu, tests on, apps on)
meson setup build

# With commonly-tweaked options
meson setup build -Denable-fp16=true -Denable-opencl=true -Denable-fsu=true

# Reconfigure an existing build dir
meson setup --reconfigure build -Denable-fp16=true

# Build
ninja -C build

# Install (respects --prefix from meson setup)
ninja -C build install

# Clean
ninja -C build clean
```

Key `meson_options.txt` flags (see file for full list):

- `platform` — `none|tizen|yocto|android|windows`
- `enable-fp16`, `enable-blas`, `enable-openmp`, `enable-opencl`, `enable-cublas`
- `enable-fsu` (flash storage utilization), `fsu-path`
- `enable-tflite-interpreter`, `enable-onnx-interpreter`, `enable-tflite-backbone`
- `enable-test`, `enable-long-test`, `reduce-tolerance`, `test-timeout`
- `arm-arch` (`armv7l|armv8.2-a|armv9.2-a`) for ARM-specific ISA tuning
- `enable-profile`, `enable-trace`, `enable-debug`, `enable-logging`
- `nntr-num-threads`, `omp-num-threads`, `openblas-num-threads`

Platform-specific builds:

```bash
# Tizen (runs unit tests as part of gbs)
gbs build

# Debian/Ubuntu package
debuild -us -uc

# Android (NDK) — uses jni/ + prepare_*.sh scripts
# See docs/how-to-run-example-android.md
```

## Testing

Tests are GoogleTest-based and discovered by Meson. Test timeout defaults to 90s; set `-Dtest-timeout=…` to override.

```bash
# Run full test suite
ninja -C build test

# Run a single test binary directly (faster feedback, full gtest filtering)
./build/test/unittest/unittest_layers
./build/test/unittest/unittest_layers --gtest_filter='FullyConnected*'
./build/test/unittest/unittest_nntrainer_tensor --gtest_filter='*Quantizer*'

# Run one suite via meson
meson test -C build unittest_layers
meson test -C build --list               # enumerate suites
meson test -C build -v unittest_models   # verbose
```

Test layout (`test/`):

- `unittest/` — core library unit tests (tensors, layers, memory, compiler, datasets, models, integration)
- `unittest/layers/` — per-layer tests plus shared harness (`layers_common_tests.h`, `layers_golden_tests.cpp`). Layer tests compare against golden data in `test/input_gen/` — generate with the scripts there.
- `unittest/integration_tests/` — end-to-end (FSU, loss, mixed precision)
- `unittest/models/` — golden-value model tests
- `ccapi/`, `tizen_capi/` — API-layer conformance
- `nnstreamer/` — sub-filter tests
- `jni/` — Android build harness for test binaries

Android testing: `./tools/android_test.sh` builds, `adb push`es to `/data/local/tmp/nntr_android_test`, and the test binaries (`unittest_layers`, `unittest_nntrainer_tensor`, …) run under `adb shell`. Golden data in `build/res/` must be pushed for layer tests.

Coverage: `test/unittestcoverage.py` post-processes lcov output.

## Lint / Format

Follow `.clang-format` exactly for `.cpp`/`.c`. Header files (`.h`) may deviate from clang-format indentation and 80-column rules — match surrounding style rather than re-imposing clang-format. Keep diffs minimal; do not mass-reformat.

CI runs `cpp_linter` and the scripts in `.github/workflows/static.check.scripts/`.

## Architecture

### Top-level layout

```
api/            C and C++ public APIs (entry points for users)
  capi/         Tizen-style C API (ml_train_*)
  ccapi/        C++ API (ml::train namespace) — model.h, layer.h, optimizer.h, dataset.h, tensor_api.h
nntrainer/      Core library — everything here is the implementation
Applications/   Runnable examples (CausalLM, LLaMA, Resnet, MNIST, YOLOv2/3, Transfer Learning, …)
nnstreamer/     NNStreamer tensor_filter / tensor_trainer sub-plugins
test/           gtest suites; mirrors nntrainer/ structure
jni/            Android NDK build (Android.mk, Application.mk, prepare_*.sh)
tools/          Dev scripts (android_test.sh, cross-compile, pyutils)
subprojects/    Meson wraps for dependencies (iniparser, ggml, ruy, etc.)
debian/, packaging/   Packaging for Ubuntu PPA and Tizen RPM
docs/           User docs (getting-started.md, components.md, how-to-*.md)
```

### Core library (`nntrainer/`) — reading order

1. **`engine.{h,cpp}`, `app_context.{h,cpp}`, `cl_context.{h,cpp}`, `context.h`** — process-wide registries. `Engine` owns device contexts (CPU, GPU/OpenCL). `AppContext` (singleton) registers factory functions for layers, optimizers, LR schedulers, datasets; user plugins register here to be discoverable by string keyword.

2. **`tensor/`** — the numeric core. `tensor_base` is the abstract interface; concrete types (`float_tensor`, `half_tensor`, `char_tensor`, `short_tensor`, `int4_tensor`, `q4_0_tensor`, `q4_k_tensor`, `q6_k_tensor`, `bcq_tensor`) implement it. Operations dispatch through `cpu_backend/` (see §Tensor backends below) and `cl_operations/` (OpenCL).
   - **Memory**: `manager.{h,cpp}` + `tensor_pool.{cpp}` are the top-level allocators. `memory_pool` is the byte arena. `basic_planner` / `optimized_v1_planner` / `optimized_v2_planner` / `optimized_v3_planner` decide in-memory layout across training/inference execution order.
   - **FSU / caching**: `cache_pool`, `cache_loader`, `cache_elem`, `swap_device`, `task_executor`, `task.h` implement the flash-backed tensor pool used for large-LLM inference. See `integration_test_fsu.cpp` for end-to-end usage.
   - **Quantization**: `quantizer.{h,cpp}` + per-dtype `*_tensor.cpp` + `int4_utils` / `q4_0_utils` handle weight quantization.

3. **`layers/`** — every layer (fc_layer, conv2d_layer, lstm, multi_head_attention, batch_normalization, …). Pattern for all layers:
   - Declared by deriving from `Layer` (in `layer_devel.h`) via `layer_impl.{h,cpp}`.
   - Each layer is wrapped by a `LayerNode` (`layer_node.{h,cpp}`) inside the graph.
   - `layer_context.{h,cpp}` gives layers access to inputs/outputs/weights during init and run.
   - Properties parsed through `common_properties.{h,cpp}` and `utils/base_properties.{h,cpp}`.
   - `cl_layers/` holds GPU variants; `loss/` holds loss layers; `plugged_layer.h` is the plugin interface.
   - When adding a layer: implement both the header/source in `layers/`, register it in `app_context.cpp`, add to `nntrainer/meson.build`, write a `test/unittest/layers/unittest_layers_<name>.cpp`, and provide golden data via `test/input_gen/`.

4. **`graph/`** — `network_graph.{h,cpp}` is the DAG of `LayerNode`s. `graph_core` handles topological ordering and execution order; `connection.{h,cpp}` represents edges. Execution order feeds into the memory planner.

5. **`compiler/`** — model construction and interchange. `ini_interpreter` loads INI config files, `tflite_interpreter` / `tflite_export_realizer` handle TFLite I/O, `onnx_interpreter` handles ONNX. The `*_realizer.{h,cpp}` classes are graph-rewriting passes (activation folding, BN fusion, recurrent unrolling, flatten/slice/multiout/remap normalization, input/previous_input wiring) applied before the graph is finalized.

6. **`models/`** — `neuralnet.{h,cpp}` is the top-level `NeuralNetwork` class that owns the graph, optimizer, dataset, and training/inference loop. `model_loader.{h,cpp}` parses INI/model files. `dynamic_training_optimization.{h,cpp}` prunes updates during training.

7. **`optimizers/`** — `adam`, `adamw`, `lion`, SGD, plus LR schedulers. Registered through `app_context`.

8. **`dataset/`** — `databuffer.{h,cpp}` is the async producer/consumer queue. `data_producer.h` is the interface; `dir_data_producers`, `func_data_producer`, `random_data_producers`, `raw_file_data_producer` are concrete producers. `iteration_queue.{h,cpp}` decouples data prep from training steps.

9. **`opencl/`** — thin wrappers over the OpenCL loader, context, command queue, buffers, kernels, programs. Used by `tensor/cl_operations/` and `layers/cl_layers/`.

10. **`utils/`** — `base_properties` (typed property framework with string parsing), `profiler`, `tracer`, `nntr_threads` + `bs_thread_pool` (threading), `ini_wrapper`, `fp16` helpers, `util_simd*` (SIMD utility), `node_exporter`, `singleton`/`noncopyable`/`nonmovable` mixins.

### Tensor backends (`nntrainer/tensor/cpu_backend/`)

`cpu_backend.h` is the single external interface for tensor ops. Implementations are selected at build time per arch:

```
arm/      armv7_neon, neon_impl, neon_impl_fp16 (armv8.2+), arm_compute_backend
x86/      avx2_impl, x86_compute_backend
fallback/ pure-C reference (no SIMD) — used when target ISA has no custom impl
cblas_interface/   OpenBLAS/cblas shim
ggml_interface/    ggml quantized kernel shim
```

When adding a new SIMD ISA or external math lib: create a new folder next to `arm/` / `x86/`, mirror the `cpu_backend.h` surface, and route through the existing selection logic (see `nntrainer/tensor/cpu_backend/README.md`).

### LLM inference stack

`Applications/CausalLM/` is the reference LLM runner (Qwen3, Qwen3-MoE, GPT-OSS). It wires together: tokenizer (`huggingface_tokenizer.cpp`), model graph assembled via ccapi, FSU-enabled tensor pool (`enable-fsu`), and the MoE expert cache. Use `Applications/CausalLM/models/*-slim` variants to test on-the-fly expert loading.

### How a training run flows

1. User creates `ml::train::Model`, adds layers, sets optimizer and dataset (via ccapi) — or loads an INI file (via `ini_interpreter`).
2. `compile()` → `NetworkGraph` is built, realizer passes run, topological order computed.
3. `initialize()` → `Manager`/`TensorPool` plan memory using execution order; weights allocated; `AppContext` resolves factory strings to concrete types.
4. `train()` → `databuffer` fills `iteration_queue`; graph runs forward → loss → backward; optimizer updates weights; optional `dynamic_training_optimization` skips updates.
5. `save_path` → serialized weights written.

Inference follows the same path minus backward/optimizer; with FSU enabled, tensor accesses hit the cache pool which pages from `fsu-path`.

## Conventions & Review Priorities

From `.cursorrules` and project norms — apply these when editing:

- **Correctness & ABI safety first.** Public headers (`api/ccapi/include/**`, `api/capi/include/**`), exported symbols, virtual tables, and struct layouts are ABI-sensitive. Don't break them silently. Watch for UB, lifetime issues, alignment, strict-aliasing, signed overflow, and concurrency hazards.
- **Portability.** Code must build across Tizen, Ubuntu 22.04/24.04, Android NDK, and Windows (MSVC + clang). Don't use compiler-/libc-specific behavior without guards. Don't add dependencies without matching wraps in `subprojects/` and debian/packaging updates.
- **RAII, explicit ownership, no new leaks.** Error paths must release resources. Match the error propagation style already in the surrounding file (status codes in capi, exceptions in ccapi/internals).
- **Performance in hot paths.** Tensor ops, training loops, and LLM inference are hot — avoid unnecessary allocations/copies, prefer const-correctness and move semantics. Call out runtime impact in PR descriptions; a microbench or reasoning beats hand-waving.
- **Tests required for behavior changes.** GoogleTest. Prefer narrow tests pinning edge cases / regressions. For layers, add both standalone and dependent tests with golden data.
- **Minimal diffs.** Don't reformat unrelated code. Don't invent new conventions; follow the nearby file.
- **Strict warnings.** `werror=true` is intentional. Fix warnings at the root; don't add `-Wno-*` to silence them.
- **Commit messages.** Describe the what-and-why (bug + fix, or feature summary). Include a sign-off (`Signed-off-by:`). See `CONTRIBUTING.md`.
- **Commit author & committer.** Both the author and the committer of every commit must be `Jijoong Moon <jijoong.moon@samsung.com>`. Never commit as `Claude <noreply@anthropic.com>` or any other identity. Before the first commit in a session set `git config user.name "Jijoong Moon"` and `git config user.email "jijoong.moon@samsung.com"` (this controls the committer), and pass `--author="Jijoong Moon <jijoong.moon@samsung.com>"` explicitly on each commit. The `Signed-off-by` trailer must match. This applies equally to commits, amends, cherry-picks, and rebases — if any commit slips through with a different identity, rewrite it with `git filter-branch --env-filter` to fix both `GIT_AUTHOR_*` and `GIT_COMMITTER_*` before pushing.

## Key Files for Common Tasks

| Task | Start here |
| --- | --- |
| Add a new layer | `nntrainer/layers/<name>_layer.{h,cpp}`, register in `app_context.cpp`, update `nntrainer/layers/meson.build`, test in `test/unittest/layers/` |
| Add a new optimizer | `nntrainer/optimizers/`, register in `app_context.cpp` |
| Add a tensor dtype / quantization scheme | `nntrainer/tensor/<name>_tensor.{h,cpp}` deriving from `tensor_base`, update `tensor_pool`/`manager`, `quantizer.cpp` if needed |
| Add a SIMD kernel | `nntrainer/tensor/cpu_backend/<arch>/` — follow layout in `cpu_backend/README.md` |
| Add a compiler pass | `nntrainer/compiler/*_realizer.{h,cpp}`, hook into the realizer pipeline used by `NeuralNetwork::compile()` |
| Add a dataset producer | `nntrainer/dataset/` deriving from `data_producer.h`, register factory in `databuffer_factory.cpp` |
| Wire a new public API call | `api/ccapi/include/` + `api/ccapi/src/` for C++; `api/capi/` for the Tizen C API |
| Add an example app | `Applications/<name>/jni/main.cpp` + `Applications/<name>/jni/meson.build`, then `subdir('<name>/jni')` in `Applications/meson.build` |

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Minimal code impact.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.



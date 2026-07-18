# Building nntrainer with the QNN (Qualcomm AI Engine Direct) backend

The QNN backend lets nntrainer offload supported subgraphs to the
Qualcomm Hexagon NPU on Snapdragon devices via Qualcomm's *AI Engine
Direct* (QNN) runtime.

The Qualcomm QNN SDK itself is **proprietary** ("Confidential and
Proprietary - Qualcomm Technologies, Inc."). It cannot be redistributed
inside this repository. nntrainer therefore follows the standard
**bring-your-own-SDK** pattern used by CUDA, ROCm, and oneAPI: the user
installs the SDK locally and points the build at it.

The SDK headers and compatibility shims needed at compile time are
**not** committed under `nntrainer/qnn/jni/vendor/`. That directory is
gitignored and is **auto-generated at `meson setup` time** by the script
`nntrainer/qnn/jni/tools/update_qnn_version.py`, which copies the
required files out of the locally-installed SDK and applies a small
set of compatibility patches. You do not need to run the script by hand
in the normal build flow.

This document describes how to install the SDK and wire it into the
build.

## 1. Obtain the QNN SDK

The SDK is distributed by Qualcomm via the *Qualcomm Software Center* /
*Qualcomm Package Manager*:

  https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk

Search for **Qualcomm AI Engine Direct SDK** (a.k.a. QNN SDK). Download
requires a (free) Qualcomm developer account and acceptance of
Qualcomm's license terms.

### Tested versions

nntrainer's QNN backend has been verified against:

| QNN SDK package | QNN API version | Status |
| --------------- | --------------- | ------ |
| 2.47.0          | 2.36.0          | OK     |

> **API-version pin:** the build is gated on **QNN API version 2.36**
> (i.e. `QNN_API_VERSION_MINOR == 36`). The `update_qnn_version.py`
> script asserts this at configure time and fails with a clear message
> if the installed SDK ships a different API version. The SDK *package*
> version string (2.47.x) may vary across patch releases while the API
> version remains 2.36; what matters for compatibility is the API
> version, not the package string.

Older 1.x SDKs and SDKs shipping a QNN API version other than 2.36 are
not supported. Newer SDK packages that retain QNN API 2.36 should work;
please report regressions if you encounter them.

## 2. Extract the SDK

After unpacking, the SDK root contains roughly this layout:

```
qnn-2.x.x.x/
├── bin/                    # qnn-net-run, qnn-context-binary-generator, ...
├── include/
│   ├── QNN/                # QnnInterface.h, QnnTensor.h, QnnTypes.h, ...
│   ├── HTP/                # HTP-specific headers
│   └── System/             # QnnSystemInterface.h ...
├── lib/
│   ├── x86_64-linux-clang/ # Host libraries (libQnnCpu.so, libQnnHtp.so ...)
│   └── aarch64-android/    # Device libraries
└── share/
```

nntrainer's build only consumes headers under `include/`; the runtime
shared libraries (`libQnnHtp.so`, `libQnnCpu.so`, ...) are dlopen'd at
runtime — see §5.

## 3. Configure the build

The SDK root can be supplied in either of two ways — the meson option
takes priority when both are set:

**Option A — meson option (recommended for reproducible CI):**

```bash
meson setup build \
    -Denable-npu=true \
    -Dqnn-sdk-root=/opt/qcom/aistack/qnn-2.47.0
ninja -C build
```

**Option B — environment variable (convenient for interactive dev):**

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qnn-2.47.0
meson setup build -Denable-npu=true
ninja -C build
```

When `-Denable-npu=true`, `meson setup` automatically runs
`nntrainer/qnn/jni/tools/update_qnn_version.py` to copy the required
SDK headers and apply compatibility patches into
`nntrainer/qnn/jni/vendor/`. This happens once at configure time; you
do not need to run the script by hand unless you later update the SDK
and want to refresh `vendor/` without reconfiguring from scratch.

The configure step validates the SDK root early and produces a clear
error if neither `-Dqnn-sdk-root` nor `QNN_SDK_ROOT` is set, or if the
path does not point at a valid SDK:

```
nntrainer/qnn/meson.build:9:0: ERROR: enable-npu=true requires
-Dqnn-sdk-root=<path-to-qcom-qnn-sdk> or QNN_SDK_ROOT env var.
Obtain the QNN SDK from https://qpm.qualcomm.com/ and pass its root
directory. See docs/backend_guide/QNN_BUILD.md.
```

When `enable-npu=false` (the default) the `qnn-sdk-root` option and
`QNN_SDK_ROOT` are both ignored — default builds need nothing from
Qualcomm and are unaffected by all of this.

## 4. What gets built

With `enable-npu=true`, the QNN integration is compiled into a
**separate plugin shared library** (`libqnn_context.so`). The main
`libnntrainer.so` is unchanged. The plugin is loaded dynamically by
`Engine::registerContext` only when the user actually requests the
`"qnn"` backend; this keeps the QNN runtime dependency from leaking
into binaries that do not need it.

The plugin contains:

- `nntrainer/qnn/jni/qnn_rpc_manager.{h,cpp}` — RPC memory wiring for
  shared-buffer DMA between the host and the HTP.
- `nntrainer/qnn/jni/qnn_context_var.h` — `QNNBackendVar` (the typed
  payload reachable through `ContextData::as<QNNBackendVar>()`).
- `nntrainer/qnn/jni/iotensor_wrapper.hpp` — adapter around the QNN
  sample-app `IOTensor` / `DataUtil` utilities (these utilities live
  inside the SDK, *not* in this repo).
- `nntrainer/qnn/jni/qnn/op/QNN{Linear,Graph}.{h,cpp}` — Layer-level
  ops that capture into a QNN graph (see ARCHITECTURE.md §5 for why
  QNN integrates at Layer granularity rather than at ComputeOps op
  granularity).
- `nntrainer/qnn/jni/qnn/qnn_properties.{h,cpp}` — string ⇄ struct
  converters for QNN-specific layer properties (quantization params,
  tensor shapes, ...).

Everything else under `nntrainer/qnn/jni/qnn/{Log,PAL,Utils,WrapperUtils}`
that previously sat in-tree was Qualcomm sample code. It is now
expected to come from the SDK (or, where the SDK does not ship a
particular utility, to be replaced by an equivalent from elsewhere in
nntrainer). Issues uncovered while doing this should be filed against
this repo, not Qualcomm.

## 5. Runtime: locating libQnnHtp.so

At run time the QNN runtime libraries (`libQnnHtp.so`,
`libQnnHtpV73Stub.so`, `libQnnSystem.so`, ...) need to be reachable by
`dlopen`. The simplest option is `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/opt/qcom/aistack/qnn-2.47.0/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
./build/test/unittest/unittest_nntrainer_qnn   # or any binary using the qnn ctx
```

On Android / device builds, place the matching `aarch64-android/`
libraries in `/data/local/tmp/qnn/` (or whichever path your APK / shell
binary expects) and adjust `LD_LIBRARY_PATH` accordingly.

We deliberately do *not* burn an `rpath` into the nntrainer binaries —
the SDK install path is environment-specific and an rpath would either
be wrong on every other developer's machine or force everyone to use
the same layout.

## 6. Verifying the build

Quick sanity check that the QNN context registers without exercising
the NPU itself:

```bash
ninja -C build test
build/test/unittest/unittest_nntrainer_engine --gtest_filter='*QNN*'
```

Tests that exercise the actual HTP runtime are gated behind the
presence of the SDK shared libraries and a Snapdragon device with HTP
support. If you only have the headers (no `libQnnHtp.so` runtime,
no device), expect those tests to be skipped.

## 7. Troubleshooting

- **`fatal error: 'QnnInterface.h' file not found`** — `qnn-sdk-root`
  (or `QNN_SDK_ROOT`) points at the wrong directory, the SDK root was
  not set before `meson setup` ran, or `update_qnn_version.py` did not
  populate `vendor/` successfully. Confirm
  `<sdk-root>/include/QNN/QnnInterface.h` exists in the SDK, then
  re-run `meson setup` (or run `update_qnn_version.py` directly) to
  re-generate `nntrainer/qnn/jni/vendor/`.
- **`undefined reference to QnnInterface_getProviders`** — the QNN
  runtime is `dlopen`'d, not link-time linked, so this should not
  occur during build. If it does, the build is mis-configured: file
  an issue.
- **`libQnnHtp.so: cannot open shared object file`** at run time —
  see §5 (`LD_LIBRARY_PATH`).
- **Wrong HTP architecture stub** (`HtpV68Stub` vs `HtpV73Stub` etc.) —
  pick the stub matching your SoC. The QNN SDK README has a table.

## 8. Why the SDK is not committed to this repository

The QNN SDK is "Confidential and Proprietary - Qualcomm Technologies,
Inc." and cannot be redistributed. Qualcomm distributes it exclusively
through the Software Center as versioned tarballs that require a free
developer account and license acceptance — there is no public git
mirror. Committing the SDK headers directly (or as a submodule pointing
at a private mirror) would either violate Qualcomm's license or prevent
most contributors from accessing the repo at all.

The auto-generation approach (`vendor/` is gitignored; populated at
configure time by `update_qnn_version.py`) keeps proprietary code off
the repository while still giving every developer a reproducible,
patch-applied copy of the headers the moment they run `meson setup`.
This is the same BYO-SDK rationale used by CUDA / ROCm / oneAPI
backends in this and other open-source ML frameworks.

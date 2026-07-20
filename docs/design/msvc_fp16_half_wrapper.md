# nntrainer on Windows/MSVC — fp16 via a uint16-backed `Half` wrapper

**Author:** Fable (design)  ·  **Implementer:** Opus  ·  **Date:** 2026-07-04
**Target:** same HW as Linux (Intel Panther Lake Xe3 iGPU + NVIDIA RTX 5060 dGPU), but built on **Windows with MSVC cl.exe (x64)** for **Intel OpenCL + NVIDIA CUDA inference**, with **fp16** working.

---

## 한글 요약 (TL;DR)

- **질문에 대한 답: 된다. uint16에 담아 계산하는 방식이 정확히 맞고, 업계 표준이다** (PyTorch `c10::Half`, Eigen::half, CUTLASS `half_t`, half.hpp 전부 동일 패턴).
- MSVC는 `_Float16`/`__fp16`/`std::float16_t`를 **전혀 지원하지 않는다** (VS2022 17.14 / 2026-03 기준 확인). 하지만 **F16C 변환 intrinsic(`_mm_cvtph_ps`/`_mm_cvtps_ph`)은 VS2012부터 지원**되고 `/arch` 플래그도 필요 없다.
- 해법: `_FP16` 매크로를 **native FP16이 없을 때만**(capability 탐지, 예: MSVC) `struct Half { uint16_t bits; }` 로 치환. "FP16 있으면 FP16, 없으면 uint16"이 meson 컴파일 프로브로 자동 선택됨(§6.4). 연산자는 전부 "float로 풀어서 계산 후 다시 half로 반올림" — GCC/clang의 `_Float16` 스칼라 산술이 내부적으로 하는 것과 의미상 동일. **native FP16을 쓰는 GCC/clang/ARM 빌드는 한 글자도 안 바뀌고 byte-identical.** 두 빌드의 fp16 데이터는 비트 동일이라 모델/체크포인트/GPU 버퍼 호환.
- **핵심 유리점**: 사용자 말대로 계산은 GPU에서 한다. GPU 스테이징 코드(`cl_operations/*`, `cuda/*`)는 호스트 half **산술을 안 한다** — `getData<_FP16>()` 포인터 + `sizeof` 바이트계산 + 커널 인자뿐. 실제 호스트 half 산술은 CPU 텐서 경로(`half_tensor.cpp`, `fallback_internal_fp16.cpp`)에만 있고, 이건 GPU 추론에선 cold path다. 그래도 **컴파일은 돼야** 하므로 **완전한 연산자 집합을 갖춘 Half를 권장**한다(연산자가 전부 1줄짜리라 부담 없음).
- **CUDA는 `.cu` 파일이 0개** — 전부 NVRTC 런타임 컴파일. nvcc 불필요, MSVC가 `.cpp`만 컴파일. cuBLAS fp16도 `CUDA_R_16F` enum 태그로만 쓰여 호스트 `__half` 산술 없음. → CUDA 백엔드는 MSVC로 그냥 컴파일된다(라이브러리 경로만 Windows용으로 수정).
- **2단계 롤아웃 권장**: Phase 0 = `enable-fp16=false`로 먼저 OpenCL+CUDA MSVC 빌드를 세운다(툴체인/링크 리스크 격리) → Phase 1 = Half 래퍼 추가 후 `enable-fp16=true`.

---

## 1. The problem, precisely

`_FP16` is a **preprocessor macro** defined only when `ENABLE_FP16` is set:

```cpp
// api/ccapi/include/tensor_dim.h:25-31
#ifdef ENABLE_FP16
#ifdef USE__FP16
#define _FP16 __fp16      // ARM/Android (Clang/GCC extension)
#else
#define _FP16 _Float16    // x86_64 GCC (GCC/Clang builtin)
#endif
#endif
```

Both `__fp16` and `_Float16` are **GCC/Clang-only language extensions**. Verified against Microsoft's own C/C++ conformance table (msvc-170, dated 2026-03-10, through VS 2022 17.14):

- **`_Float16` / `__fp16`**: not a keyword in cl.exe at all. `const _Float16 *p` fails to *parse* (unknown identifier) — it is not even an error-path, it simply does not exist.
- **`std::float16_t` (`<stdfloat>`, C++23)**: listed as **"No"** (P1467R9), footnote: *"Extended floating-point types … won't be implemented until C++23 standardization is finalized."* `__STDCPP_FLOAT16_T__` is undefined. **Do not rely on it.**
- **F16C conversion intrinsics** (`_mm_cvtph_ps`, `_mm_cvtps_ph`, `_mm256_cvtph_ps`, `_mm256_cvtps_ph`): **available on MSVC since VS2012**, via `<immintrin.h>`, and **cl.exe emits the F16C instruction on the intrinsic with no `/arch:` flag required** (unlike GCC/Clang which need `-mf16c`). Runtime CPUID gating is still the caller's responsibility. Note: the *scalar* names `_cvtsh_ss`/`_cvtss_sh` are GCC/Clang builtins — MSVC does **not** ship those bare names; wrap the 128-bit `_mm_cvtph_ps`/`_mm_cvtps_ph` for a scalar helper.
- **`<cuda_fp16.h>` `__half`**: safe to `#include` in an MSVC host `.cpp` (device pieces guarded by `__CUDACC__`), gives storage + `__half2float`/`__float2half` host conversions — but its `+ - * /` operators are **device-only** (`__CUDA_NO_HALF_OPERATORS__` semantics). Never usable for host arithmetic without nvcc. (We don't need it — see §5.)

**Conclusion:** the only viable MSVC path is a **software half type**: `uint16_t` storage + float round-trip conversions (F16C-accelerated, software fallback) + hand-rolled operators. Exactly the user's instinct.

---

## 2. Design: `_FP16` → `Half` only when native FP16 is unavailable, byte-identical everywhere else

Introduce a self-contained leaf header defining `nntrainer::Half`, and add a **third branch** to the `_FP16` macro selected by a new define `USE_HALF_WRAPPER`. That define is emitted by a **capability probe** (§6.4), not a compiler-name check: if the compiler can do native `_Float16` arithmetic it is used; otherwise (e.g. MSVC) the wrapper is selected. "FP16 있으면 FP16, 없으면 uint16" — automatic.

```cpp
// api/ccapi/include/tensor_dim.h (extend lines 25-31)
#ifdef ENABLE_FP16
#ifdef USE__FP16
#define _FP16 __fp16
#elif defined(USE_HALF_WRAPPER)
#include "half_fp16.h"          // new self-contained leaf header
#define _FP16 ::nntrainer::Half
#else
#define _FP16 _Float16
#endif
#endif
```

**Byte-identical guarantee for existing platforms:** GCC/Clang/ARM builds never define `USE_HALF_WRAPPER`, so they take the exact same `__fp16`/`_Float16` branch as today. Zero behavioral change on Adreno / Intel-Linux / CUDA-Linux. This satisfies the project's byte-identical-across-platforms invariant and the "migrate before removing" rule (we *add* a branch; we remove nothing).

`USE_HALF_WRAPPER` is a brand-new token (grep-confirmed absent from the tree), orthogonal to `USE__FP16` (which is only ever set on aarch64/arm/android).

**Circular-include safety:** `tensor_dim.h` lives in `api/ccapi/include/` and is included by ~82 files. The new `half_fp16.h` must be a **leaf**: include only `<cstdint>`, `<cstring>`, `<cmath>`, and (on MSVC) `<immintrin.h>`. It must **not** include `tensor_dim.h`, `tensor.h`, or anything nntrainer-specific. (`nntrainer/utils/fp16.h` — the existing pure-software `compute_fp32_to_fp16`/`compute_fp16_to_fp32` bit-twiddlers — is already installed publicly alongside `tensor_dim.h` and has no nntrainer includes, so it *could* be reused as the conversion backend; but to keep the foundational header free of any link-order coupling, **inline the bit-trick conversions directly** in `half_fp16.h` with an F16C fast path. Your call; both are cycle-free.)

---

## 3. The `Half` class specification

### 3.1 ABI / layout (load-bearing — this is what the GPU path actually needs)

```cpp
struct Half {
  uint16_t bits_;
  Half() = default;                 // trivial: `new Half[n]` (no parens) leaves bits_ uninitialized,
                                     // matching _Float16 exactly; needed for trivial-copyability
  // ... ctors / operators below
};
static_assert(sizeof(Half) == 2, "GPU/CL ABI: half must be 2 bytes");
static_assert(std::is_trivially_copyable_v<Half>, "memcpy/memset/vector need trivial copy");
static_assert(std::is_standard_layout_v<Half>,   "reinterpret_cast<unsigned short*> needs standard layout");
```

Requirements exercised by **Group B (GPU staging)** — all satisfied by the above:
- `sizeof(_FP16) == 2` used for buffer byte-size math throughout `cl_operations/*` and `cuda/*`.
- `reinterpret_cast<unsigned short*>` / `<uint16_t*>` of `_FP16*` for CUDA kernel launch + OpenCL SVM args (`cuda_compute_ops.cpp:99-104`, `cl_compute_ops.cpp:181-182`, `cuda_rmsnorm_layer.cpp:95-99`). Standard-layout + first member `uint16_t` ⇒ pointer-interconvertible, valid.
- `memset(p, 0, sizeof(_FP16)*n)` (`half_tensor.cpp:157`) — bit-zero must be the zero half (it is: 0x0000 = +0.0).
- `std::vector<_FP16>`, `new _FP16[n]` / `new _FP16[n]()`, default-construction, `_FP16 dot_cl(...)` return-by-value (copyable/movable).
- **Keep `Half() = default;` (trivial).** Do *not* write `Half(): bits_(0){}` — that would break trivial-copyability and change `new Half[n]` semantics vs `_Float16`.

### 3.2 Conversions

```cpp
// float -> half: F16C when available, software otherwise
explicit-or-implicit Half(float f);   // see §3.4 on explicit vs implicit
Half(double d) : Half(static_cast<float>(d)) {}
Half(int i)    : Half(static_cast<float>(i)) {}
operator float() const;               // implicit — matches _Float16 promotion
```

Conversion body (recommended):
```cpp
static inline uint16_t f32_to_f16_bits(float f) {
#if defined(_MSC_VER)   // MSVC F16C: no /arch flag needed, VS2012+
  return (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0);
#else
  return ::nntrainer::compute_fp32_to_fp16(f);   // software fallback
#endif
}
static inline float f16_bits_to_f32(uint16_t h) {
#if defined(_MSC_VER)
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
#else
  return ::nntrainer::compute_fp16_to_fp32(h);
#endif
}
```
The GGML path already ships exactly this `#ifdef _MSC_VER` F16C idiom and compiles on MSVC today (`nntr_ggml_impl/nntr_ggml_impl_fp16_fp32.cpp:71`) — mirror it. Consider a runtime CPUID check for F16C, or (simpler for a first cut) always use the software `compute_*` path and add F16C later as an optimization; the GPU inference target barely touches host conversion, so software is fine to start.

### 3.3 Arithmetic & comparison operators (needed by Group A — CPU tensor path)

**Every operator computes in float then rounds back**, matching `_Float16` semantics (GCC promotes `_Float16` ops to float and rounds back too):

- Homogeneous, **result `Half`** (must stay Half to preserve chained rounding parity like `a*b*c` at `fallback_internal_fp16.cpp:378`):
  `operator+ - * /` for `(Half,Half)`; compound `operator+= -= *= /=`; unary `operator-`.
- Comparison, **result `bool`**: `operator< > <= >= == !=` for `(Half,Half)` — used by `std::max_element`/`std::min_element`/`isamax`/`std::clamp` and the `*X != *X` NaN self-check.
- Mixed **`Half`/`float`** → **result `float`** (matches `_Float16 op float` promoting to float): `operator+ - * /` and all comparisons for `(Half,float)` and `(float,Half)`. Needed by e.g. `half_tensor.cpp:393` (`Half * beta`), `:819` (`Half >= dropout`), `fallback_internal_fp16.cpp:72` (`Half * beta`).

Once the above exist, **all STL functor instantiations fall out for free** — `std::multiplies<Half>`, `std::plus<Half>`, `std::minus<Half>`, `std::divides<Half>`, `std::clamp<Half>` need no separate specialization.

**Not required anywhere** (grep-confirmed): `operator<<` streaming, `std::numeric_limits<Half>`, `std::to_string`, printf/varargs of half, `std::isnan(Half)` direct (code always casts to float first). Don't bother implementing them.

### 3.4 The one real design decision: implicit `operator float` + ambiguity

With an implicit `operator float()` **and** a converting `Half(float)` ctor, `Half + float` is ambiguous between `operator+(Half,Half)` (float→Half) and built-in `float+float` (Half→float). Two clean resolutions:

- **Recommended — non-explicit ctors + explicit mixed operators.** Provide the `(Half,float)`/`(float,Half)` operators from §3.3 (returning float). They are exact matches, so no ambiguity, and copy-init like `_FP16 ret = 0;` (`x86_compute_backend_fp16.cpp:173`) still compiles. Cost: ~40 one-line operators. **Zero source churn** beyond the type swap. This is what half.hpp/Eigen::half effectively do.
- **Alternative — `explicit Half(float)`.** Kills the ambiguity automatically (float→Half needs explicit construction, so `Half op float` always goes through `operator float()` → float). But it breaks copy-init sites (`_FP16 ret = 0;`, `_FP16 x = someFloat;`), which then need one-line source touch-ups (`static_cast<_FP16>(0)`) — safe/byte-identical on GCC, but you must find them all.

Go with the **recommended** option. Reference implementations to mirror the operator design: **Eigen::half** (`Eigen/src/Core/arch/Default/Half.h`, MPL2, MSVC-clean, battle-tested) and **Christian Rau's half.hpp** (MIT, single header, purpose-built for MSVC). Do **not** vendor a whole library — hand-roll ~200 lines over the existing `fp16.h` conversions for full ABI control, using those two as the operator-semantics reference.

---

## 4. Two-phase rollout (do these in order)

The Windows build has **two independent risk surfaces**: (a) does the OpenCL+CUDA MSVC toolchain link and run at all, and (b) does fp16 compile. Separate them.

### Phase 0 — `enable-fp16=false`: prove the GPU toolchain on MSVC first

Get a **fp32-only** Windows build with `enable-opencl=true` + `enable-cuda=true` working end-to-end. No `_FP16` is compiled in this configuration (it's fully `#ifdef`-gated; FP16-dtype tensor construction throws `std::invalid_argument` at runtime but nothing references the macro). This isolates and validates: OpenCL.lib linking, CUDA `lib/x64` paths, NVRTC runtime compile, and MSVC C++20 compilation of every non-fp16 CL/CUDA host `.cpp`. Fix everything in §6.1–§6.3 here. **Ship/verify this before touching fp16.**

Note: **fp32 activation is a first-class supported configuration**, not a throwaway — nntrainer supports both FP16 and FP32 activation dtypes. So a Phase-0 fp32 build is a genuinely shippable Windows configuration (larger memory / slower than fp16, but correct and usable), and simultaneously the toolchain gate that de-risks Phase 1. Two wins from one build. Ship/verify it before touching fp16.

### Phase 1 — add `Half`, flip `enable-fp16=true`

Land the `Half` header + macro branch (§2–§3), the meson MSVC fp16 branch, and the `avx2_impl.h` hardcoded-`_Float16` fix (§6.4). Now FP16-dtype tensors construct, `half_tensor.cpp` + CPU fp16 kernels compile against `Half`, and the GPU fp16 path runs. Iterate on the operator set using the CPU fp16 files as the acceptance test (they're where the full operator inventory bites).

---

## 5. Group A vs Group B — where host half work actually is

The recon split every `_FP16` operation into two groups. This is *why* the user's "compute is on the GPU" intuition is correct and load-bearing:

**Group B — GPU staging (`cl_operations/*`, `cuda/*`, `layers/cl_layers/*`, `layers/cuda_layers/*`):** the code that runs on the critical path for this goal. It does **no host half arithmetic on the normal path** — only pointer plumbing (`getData<_FP16>()`, element-offset pointer arithmetic), `sizeof(_FP16)` byte math, `reinterpret_cast` to `unsigned short`/`uint16_t` for kernel args, `const_cast` for SVM args, and a couple of `static_cast<_FP16>(epsilon)` scalar-arg conversions. The only half *arithmetic* is inside `NNTR_RMSN_VERIFY` / `NNTR_GEGLU_VERIFY` **debug env-gated** branches, and even those cast to float first. **CUDA fp16 GEMM uses `CUDA_R_16F` enum tags + `CUBLAS_COMPUTE_32F` with `float` alpha/scale (`cuda_attention.cpp:701-719`, `cuda_blas_manager.cpp`) — no host `__half` value ever.** ⇒ Group B needs only the §3.1 + §3.2 subset (2-byte layout + conversions).

**Group A — CPU tensor path (`half_tensor.cpp`, `fallback_internal_fp16.cpp`, `x86_compute_backend_fp16.cpp`):** real scalar half arithmetic, chains, comparisons, STL functors, template specializations. Cold for GPU inference, **but it is in the build when `enable-fp16=true`** (meson adds these sources unconditionally on `enable-fp16`), so it must compile ⇒ requires the **full** §3.3 operator set.

**Decision: implement the full `Half` (Group A + B).** Rationale: (1) the operators are trivial one-liners; (2) it avoids surgery to exclude `half_tensor.cpp` from the build and the risk of proving a negative about tensor dispatch (FP16-dtype `Tensor` instantiates `HalfTensor` as its `itensor_`, so `half_tensor.cpp` is genuinely reachable); (3) correctness-first — a fully-working CPU fp16 fallback on Windows costs nothing at GPU-inference runtime.

---

## 6. Exhaustive file-edit checklist

Ordered. Items §6.1–§6.3 are Phase 0; §6.4–§6.6 are Phase 1.

### 6.1 CUDA Windows build (`nntrainer/meson.build` ~296-324)
- `cuda_libdir` is hardcoded `cuda_path / 'lib64'`. Windows CUDA Toolkit uses `cuda_path\lib\x64`:
  ```meson
  if host_machine.system() == 'windows'
    cuda_libdir = cuda_path / 'lib' / 'x64'
  else
    cuda_libdir = cuda_path / 'lib64'
  endif
  ```
- Library base names (`cudart`, `nvrtc`, `cublas`, `cublasLt`, `cuda`) already match the Windows import-lib names — no change.
- The `stubs` fallback for the driver lib (lines 314-317) is Linux-only. Windows ships a real `cuda.lib` in `lib/x64` (no `stubs/`). Branch it:
  ```meson
  cudadrv_dep = cxx.find_library('cuda', required: false)
  if not cudadrv_dep.found()
    if host_machine.system() == 'windows'
      cudadrv_dep = cxx.find_library('cuda', dirs: cuda_libdir, required: true)
    else
      cudadrv_dep = cxx.find_library('cuda', dirs: cuda_libdir / 'stubs', required: true)
    endif
  endif
  ```
- Confirmed: **no `.cu` files** (`nntrainer/cuda/meson.build` is all `.cpp`), kernels compiled at runtime via NVRTC (`cuda_module.cpp:71,90` `nvrtcCreateProgram`/`nvrtcCompileProgram`), NVRTC option strings are host-compiler-agnostic. No POSIX-only includes in any `nntrainer/cuda/*`. No host nvcc/gcc flags to port.

### 6.2 OpenCL link on Windows (`meson.build` ~281) — **the real Phase-0 blocker**
- `opencl_loader_dep = cxx.find_library('OpenCL', required: true)` has **no `dirs:`**. On Windows there is no vendored `OpenCL.lib` (the `nntrainer-windows-resource/x64/` bundle ships googletest/iniparser/benchmark/CLBlast/OpenBLAS only — no OpenCL).
- Note: nntrainer's own CL code loads `OpenCL.dll` dynamically at runtime (`opencl_loader.cpp` via `LoadLibraryA`/`GetProcAddress`, all calls go through `PFN_cl*` function-pointer globals in `namespace nntrainer::opencl`) — so nntrainer proper does **not** need `OpenCL.lib` at link time. The **link-time need comes from CLBlast** (`enable-clblast` defaults true), which calls the real global `cl*` symbols. Setting `enable-clblast=false` does **not** remove the need because the `find_library('OpenCL')` call is unconditional and outside the clblast block.
- **Fix (choose one):**
  - (a) User adds the dir holding `OpenCL.lib` to the `LIB` env var before meson/ninja. Sources: Intel oneAPI / "Intel SDK for OpenCL Applications", a Khronos/vcpkg ICD loader, **or the CUDA Toolkit itself ships `<cuda_path>/lib/x64/OpenCL.lib`** (convenient since CUDA is already a dep). Zero code change.
  - (b) Add a new `opencl-lib-path` meson option (mirror `cuda-path`) and pass `dirs: get_option('opencl-lib-path')` into the `find_library('OpenCL', ...)` call, guarded by `host_machine.system() == 'windows'`. Cleaner, reproducible.
- CL **headers** are fully vendored (`nntrainer/opencl/CL/*`, include dir added via the generic `nntrainer_elements` loop) — no external SDK needed, no change.

### 6.3 Windows config (`configurations/windows-native.ini`, and `-clang.ini` if targeted)
Currently `enable-opencl=false`, no cuda/fp16 keys. Add under `[project options]`:
```ini
enable-opencl = true
enable-cuda   = true
cuda-path     = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vXX.X'
# opencl-lib-path = '...'   ; only if using §6.2(b)
# enable-fp16   = true      ; add in Phase 1
```

### 6.4 fp16 meson branch (`meson.build` ~214-220) — Phase 1

**Selection is capability-based, not compiler-name-based:** "if native FP16 is usable, use it; otherwise fall back to the uint16 `Half` wrapper." Probe whether the compiler can actually *do half arithmetic* (not just parse the type, and not "is it MSVC") with `cxx.compiles(...)`. This auto-corrects for any toolchain: a future MSVC that gains `_Float16`, clang-cl, or old GCC<12.1 all land on the correct branch automatically.

The current branch checks `cc.version() >= 12.1.0` (a GCC-version proxy) and never probes real usability — on MSVC it wrongly defines `ENABLE_FP16` expecting `_Float16`, which then fails to compile. Replace with a compile probe:

```meson
elif arch == 'x86_64'
  # Does the C++ compiler support native _Float16 *arithmetic* (not just the
  # keyword)? Probe real usability; fall back to the uint16 Half wrapper if not.
  fp16_probe = '''_Float16 mul(_Float16 a, _Float16 b){ return a * b; }
                  int main(){ _Float16 x = (_Float16)1, y = (_Float16)2; return (int)mul(x, y); }'''
  if cxx.compiles(fp16_probe, name: 'native _Float16 arithmetic')
    extra_defines += '-DENABLE_FP16=1'          # native _Float16 (GCC/clang, unchanged, byte-identical)
  else
    extra_defines += '-DENABLE_FP16=1'
    extra_defines += '-DUSE_HALF_WRAPPER=1'      # uint16-backed nntrainer::Half fallback (e.g. MSVC)
  endif
```

Notes:
- **Strict improvement over the version check:** old GCC<12.1 (which fails the probe) previously got a "warning + probably fails" build; now it gets a working (software-converted) `Half` wrapper. No compiler is left in a broken state.
- **ABI-identical across both branches:** native and wrapper are both IEEE754 binary16, `sizeof==2`, same bit layout — so a wrapper-built binary and a native-built binary produce **byte-identical fp16 data**. Model files, checkpoints, and GPU kernel buffers are interchangeable between a native-fp16 build and a wrapper build; only the host *scalar* arithmetic path differs (and only in speed, not bits — see §3, R5).
- Keep the ARM branches (`USE__FP16` → `__fp16`) exactly as-is; they already select native half by arch and are untouched.
- Optional manual override: if you ever need to force one path (e.g. to test the wrapper on GCC), add a `-Dfp16-impl=auto|native|wrapper` option gating the probe. Not required; auto-probe is the right default.

### 6.5 New header + macro — Phase 1
- **New** `api/ccapi/include/half_fp16.h`: the self-contained `class Half` from §3 (includes only `<cstdint>/<cstring>/<cmath>`, and `<immintrin.h>` under `_MSC_VER`).
- **Edit** `api/ccapi/include/tensor_dim.h:25-31`: add the `#elif defined(USE_HALF_WRAPPER)` branch (§2). Install `half_fp16.h` alongside `tensor_dim.h` (add to the public ccapi header list — `api/ccapi/meson.build`).

### 6.6 The hidden second blocker — hardcoded `_Float16` in AVX2 files — Phase 1
`nntrainer/tensor/cpu_backend/x86/avx2_impl.h` (lines 34,44,53) and `avx2_impl_fp16.cpp` (lines 24,30,60,66,100,107) **hardcode the literal `_Float16`** in signatures — they bypass the `_FP16` macro entirely. Fixing `tensor_dim.h` alone is **not enough**; MSVC hits the bare unknown `_Float16` keyword and fails to parse. Replace `_Float16` → `_FP16` in these files (add a macro-only include). Byte-identical on GCC/clang-x86_64 (`_FP16` already expands to `_Float16` there). The F16C intrinsic bulk paths (`_mm256_cvtph_ps` on `__m128i`-loaded raw bits) are unaffected — they operate on bit patterns; only the scalar tail conversions (`static_cast<float>(*data)`, `static_cast<_FP16>(*input)`) use `Half`'s conversion operators. Also `x86_compute_backend_fp16.cpp:134,144` passes `_FP16*` into these `_Float16*` params — silently fine on GCC (same type), breaks the moment `_FP16` is a class; the rename fixes both sides.

### 6.7 Latent test bug (fix opportunistically)
`test/unittest/unittest_cl_residency.cpp` uses `_FP16` **unguarded** (lines 329-331, 413) but is gated in `test/unittest/meson.build:90` on `enable-opencl` only, **not** `enable-fp16`. This is a **pre-existing bug** on *any* `enable-opencl=true, enable-fp16=false` build (all platforms). It won't trigger the Windows target (which sets both true), but wrap the two `TEST(...)` bodies in `#ifdef ENABLE_FP16 … #endif` (mirroring `unittest_opencl_kernels_blas.cpp:70,737`). No other opencl/cuda-gated test uses `_FP16` unguarded.

---

## 7. Sources that instantiate templates on `_FP16` — audit during bring-up

The CPU fp16 files carry explicit `_FP16` template specializations that must compile against `Half`: `__fallback_gemm_q4_0`, `dequantize_row_q8_K`, `quantize_row_q8_K`, `__fallback_gemm_q6_K`, `rms_norm_wrt_width_fp16_intrinsic`, `__fallback_clamp` (`fallback_internal_fp16.cpp:467-510`, `x86_compute_backend_fp16.cpp:303-340`). Several delegate into **`ggml_interface.h` templates on `_FP16`** — this header was **not** opened by recon; **audit it** for any op not covered by `Half`'s operator set (it does q4_0/q8_K/q6_K quant math, all reducible to float arithmetic ⇒ should be covered, but verify). Same for `blas_kernels.h`'s cl-side `*_cl_internal` templates on `_FP16` and `setActiFunc<_FP16>` (`cl_compute_ops.cpp:251`). These are the three honestly-flagged unknowns; treat the CPU fp16 files' successful compile as the completeness test for `Half`.

---

## 8. Verification plan

1. **Phase 0 gate:** `meson setup` with the Windows-native ini + opencl/cuda true, `ninja`. Run a small fp32 gemma/qwen inference on both Intel CL (`NNTR_ENGINE`/engine=gpu) and CUDA (engine=cuda). Confirms toolchain + linking + NVRTC + Intel ICD.
2. **`Half` unit sanity:** a tiny host test asserting `sizeof(Half)==2`, round-trip `float→Half→float` matches `compute_fp16_to_fp32(compute_fp32_to_fp16(x))` for a spread of values incl. denormals/inf/nan, and that `Half(a)*Half(b)` equals `static_cast<Half>(float(a)*float(b))`.
3. **Byte-identity regression (critical):** rebuild the **Linux GCC** targets (Adreno/Intel/CUDA) and confirm binaries/outputs are unchanged — the macro's GCC branch must be untouched. This is the project's standing invariant.
4. **Phase 1 gate:** flip `enable-fp16=true`, rebuild on MSVC, run the **same** gemma4-E2B / qwen3 fp16 inference on Intel CL and CUDA on Windows; compare token output coherence + perf against the Linux fp16 baseline (see `project_3platform_9cell_bench` memory). Watch for the known Xe3 in-order-queue SVM coherence issue (`NNTR_XE3_SYNC`) — orthogonal to fp16 but relevant on the same Intel HW.
5. **CPU fp16 fallback:** exercise one op that routes through `half_tensor.cpp` on host (e.g. a CPU-dtype FP16 tensor add) to confirm the wrapper arithmetic is correct, not just compilable.

---

## 9. Risk register

| # | Risk | Likelihood | Mitigation |
|---|------|-----------|------------|
| R1 | `OpenCL.lib` not found at link (Windows) | High | §6.2 — use CUDA Toolkit's `lib/x64/OpenCL.lib` or Intel SDK; add `dirs:`/`LIB`. Caught in Phase 0. |
| R2 | Overload ambiguity from implicit `operator float` + `Half(float)` | Medium | §3.4 — non-explicit ctor + explicit mixed `(Half,float)` operators; mirror Eigen::half/half.hpp. |
| R3 | `avx2_impl.h` hardcoded `_Float16` missed (parse error) | Medium | §6.6 — explicitly enumerated; grep `_Float16` across `x86/` before declaring done. |
| R4 | `ggml_interface.h` / `blas_kernels.h` templates hit an uncovered op | Low-Med | §7 — audit; CPU-fp16 compile is the completeness test. |
| R5 | Rounding parity: chained `Half*Half*Half` vs native `_Float16` | Low | Operators keep intermediate as `Half` (round each step) — matches GCC's promote-per-op-then-round. Validate in test #2. |
| R6 | F16C not present on some target CPU | Very Low (target is modern) | Software `compute_*` fallback under `#else`; or CPUID-gate. Panther Lake has F16C. |
| R7 | Accidentally changing GCC/clang codegen | Low | All edits are `#ifdef _MSC_VER` / `cxx_compiler_id=='msvc'` / new-define gated. Regression test #3. |

---

## 10. One-paragraph handoff for Opus

MSVC has no native half type, so make `_FP16` resolve to a `uint16_t`-backed `nntrainer::Half` **only under MSVC** (new `USE_HALF_WRAPPER` define, third branch in `tensor_dim.h`; GCC/clang/ARM stay byte-identical). `Half` = 2-byte standard-layout POD + implicit `operator float()` + non-explicit `Half(float/int/double)` + full `+ - * / += -= *= /=`, unary `-`, comparisons, and mixed `Half/float` operators returning float — every op computed in float and rounded back (matches `_Float16`), using F16C intrinsics (MSVC-native, no `/arch` flag) with a software `compute_fp32_to_fp16`/`compute_fp16_to_fp32` fallback. Do Phase 0 first (`enable-fp16=false`, opencl+cuda MSVC build — fixes the `OpenCL.lib` link blocker §6.2, CUDA `lib/x64` paths §6.1), then Phase 1 (add `Half`, meson MSVC fp16 branch §6.4, and the hardcoded-`_Float16`→`_FP16` rename in `avx2_impl.h`/`avx2_impl_fp16.cpp` §6.6, which is a separate parse-blocker beyond the macro). CUDA is easy: no `.cu` files (NVRTC runtime compile), cuBLAS fp16 via `CUDA_R_16F` enum tags with float compute — MSVC never sees a host `__half`. The user's premise holds: all real math is on the GPU, so the wrapper's per-op float conversion is off the hot path; implement the full operator set anyway for a correct CPU fallback and a clean compile of `half_tensor.cpp`. Full inventory + file:line evidence in the recon journals; the CPU fp16 files are the acceptance test for wrapper completeness.


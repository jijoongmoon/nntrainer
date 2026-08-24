// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 */
/**
 * @file   half_fp16.h
 * @date   04 Jul 2026
 * @brief  uint16-backed IEEE754 binary16 `Half` wrapper for toolchains with no
 *         native `_Float16` / `__fp16` scalar type (notably MSVC cl.exe)
 * @see    https://github.com/nntrainer/nntrainer
 * @see    docs/design/msvc_fp16_half_wrapper.md
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @note   This is a SELF-CONTAINED LEAF header. It must include only the C++
 *         standard library (<cstdint>/<cstring>/<cmath>) plus <immintrin.h>
 *         under MSVC. It must NOT include any nntrainer header (in particular
 *         tensor_dim.h), so that tensor_dim.h can pull it in without an include
 *         cycle.
 *
 * @note   Selected only when the build defines USE_HALF_WRAPPER (emitted by the
 *         meson capability probe when native _Float16 arithmetic is
 *         unavailable, e.g. MSVC). On GCC/clang/ARM the `_FP16` macro keeps
 *         resolving to the native `_Float16`/`__fp16` and this header is not
 *         used -- those builds stay byte-identical.
 *
 * @note   Layout is IEEE754 binary16 (`uint16_t bits_`), bit-for-bit identical
 *         to native `_Float16`, so model files, checkpoints and GPU kernel
 *         buffers are interchangeable between a native-fp16 build and a
 *         wrapper build. Every arithmetic operator computes in `float` and
 *         rounds the result back to half, so a single operation is
 *         bit-identical to native `_Float16`; see the note on chained
 *         expressions above the homogeneous operators below.
 */

#ifndef __HALF_FP16_H__
#define __HALF_FP16_H__
#ifdef __cplusplus

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(_MSC_VER)
#include <immintrin.h>
#endif

namespace nntrainer {

/**
 * @brief Internal bit-level fp16<->fp32 conversion helpers.
 * @note  The software path is a byte-for-byte copy of
 *        nntrainer::compute_fp32_to_fp16 / compute_fp16_to_fp32
 *        (nntrainer/utils/fp16.cpp, Marat Dukhan's algorithm) so that a
 *        wrapper build produces the exact same fp16 bit patterns as the
 *        software converter used elsewhere. Kept private to this header to
 *        avoid clashing with the nntrainer::fp32_from_bits declarations in
 *        fp16.h.
 */
namespace half_fp16_detail {

static inline float bits_to_f32(uint32_t w) {
  float f;
  std::memcpy(&f, &w, sizeof(f));
  return f;
}

static inline uint32_t f32_to_bits(float f) {
  uint32_t w;
  std::memcpy(&w, &f, sizeof(w));
  return w;
}

/** software float -> half (round to nearest even), matches compute_fp32_to_fp16
 */
static inline uint16_t f32_to_f16_bits_sw(float f) {
  const float scale_to_inf = bits_to_f32(UINT32_C(0x77800000));
  const float scale_to_zero = bits_to_f32(UINT32_C(0x08800000));
  float base = (std::fabs(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = f32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = bits_to_f32((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = f32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (
    uint16_t)((sign >> 16) |
              (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

/** software half -> float, matches compute_fp16_to_fp32 */
static inline float f16_bits_to_f32_sw(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;
  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
  const float exp_scale = bits_to_f32(UINT32_C(0x7800000));
  const float normalized_value =
    bits_to_f32((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
    bits_to_f32((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
    sign | (two_w < denormalized_cutoff ? f32_to_bits(denormalized_value)
                                        : f32_to_bits(normalized_value));
  return bits_to_f32(result);
}

/**
 * @brief Is the F16C conversion instruction pair known to be present at
 *        compile time?
 *
 * MSVC's x64 baseline is SSE2 and it has no per-function target attribute, so
 * an unguarded _mm_cvtps_ph emitted into a default-baseline binary is an
 * illegal instruction on any pre-Ivy-Bridge / pre-Piledriver CPU rather than a
 * slow path. Use the intrinsics only when the translation unit is already being
 * compiled for them: cl.exe defines __AVX2__ under /arch:AVX2, and F16C is
 * present on every x86 part that has AVX2. Otherwise the software converter
 * runs. GCC/clang keep the software converter unconditionally, so the non-MSVC
 * behaviour is exactly what it was; unittest_half_fp16 pins the two paths to
 * the same bits over the whole half domain either way.
 */
#if defined(_MSC_VER) && defined(__AVX2__)
#define NNTR_HALF_HAS_F16C 1
#endif

/** float -> half bits (F16C where the baseline has it; software elsewhere) */
static inline uint16_t f32_to_f16_bits(float f) {
#if defined(NNTR_HALF_HAS_F16C)
  // Rounding immediate must be in 0-7 on MSVC (C4556); _MM_FROUND_NO_EXC (0x08)
  // is rejected. _MM_FROUND_TO_NEAREST_INT (0x00) selects round-to-nearest-even
  // explicitly -- it does not defer to MXCSR, which is why the result matches
  // the software converter's fixed round-to-nearest-even regardless of the
  // caller's rounding mode. Only exception masking is given up.
  return (uint16_t)_mm_extract_epi16(
    _mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0);
#else
  return f32_to_f16_bits_sw(f);
#endif
}

/** half bits -> float (F16C where the baseline has it; software elsewhere) */
static inline float f16_bits_to_f32(uint16_t h) {
#if defined(NNTR_HALF_HAS_F16C)
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
#else
  return f16_bits_to_f32_sw(h);
#endif
}

} // namespace half_fp16_detail

/**
 * @brief 2-byte IEEE754 binary16 value type. Drop-in stand-in for native
 *        `_Float16` on toolchains without one.
 */
struct Half {
  uint16_t bits_;

  /**
   * @brief Trivial default ctor: `new Half[n]` (no parens) leaves bits_
   *        uninitialized, exactly like `_Float16`. Required for trivial
   *        copyability / standard layout. Do NOT initialize bits_ here.
   */
  Half() = default;

  /**
   * @brief non-explicit construction from any built-in arithmetic type
   *        (float/double/int/unsigned/...), routed through float and rounded to
   *        half. A single templated ctor (rather than separate float/double/int
   *        overloads) so that `static_cast<Half>(x)` is never ambiguous for
   *        integral x -- matching native `static_cast<_Float16>(x)`. SFINAE
   *        keeps it from shadowing the trivial copy/move ctors (Half is not
   *        arithmetic), so Half stays trivially copyable.
   */
  template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value>::type>
  Half(T v) : bits_(half_fp16_detail::f32_to_f16_bits(static_cast<float>(v))) {}

  /** @brief implicit Half -> float (matches _Float16 promotion to float) */
  operator float() const { return half_fp16_detail::f16_bits_to_f32(bits_); }

  /** @brief compound assignment (compute in float, round back) */
  Half &operator+=(Half o) {
    *this = Half(static_cast<float>(*this) + static_cast<float>(o));
    return *this;
  }
  Half &operator-=(Half o) {
    *this = Half(static_cast<float>(*this) - static_cast<float>(o));
    return *this;
  }
  Half &operator*=(Half o) {
    *this = Half(static_cast<float>(*this) * static_cast<float>(o));
    return *this;
  }
  Half &operator/=(Half o) {
    *this = Half(static_cast<float>(*this) / static_cast<float>(o));
    return *this;
  }

  /**
   * @brief compound assignment against a wider arithmetic type.
   *
   * Without these, `h += f` (f a float) converts f to Half first and rounds
   * twice: once narrowing f, once narrowing the sum. Native `_Float16` promotes
   * the other way -- it widens the half, computes in float and rounds once --
   * so a wrapper build would disagree with a native build by more than the
   * documented excess-precision difference, and an f above the half range would
   * become inf before the addition instead of after it. Live call sites exist:
   * `C[...] += beta * c_old` in fallback_internal_fp16.cpp's SGEMM macros.
   * Compute in float and round exactly once, matching the native promotion.
   */
#define NNTR_HALF_MIXED_COMPOUND(OP)                                           \
  template <typename T,                                                        \
            typename =                                                         \
              typename std::enable_if<std::is_arithmetic<T>::value>::type>     \
  Half &operator OP##=(T o) {                                                  \
    using C = typename std::common_type<float, T>::type;                       \
    *this =                                                                    \
      Half(static_cast<C>(static_cast<float>(*this)) OP static_cast<C>(o));    \
    return *this;                                                              \
  }
  NNTR_HALF_MIXED_COMPOUND(+)
  NNTR_HALF_MIXED_COMPOUND(-)
  NNTR_HALF_MIXED_COMPOUND(*)
  NNTR_HALF_MIXED_COMPOUND(/)
#undef NNTR_HALF_MIXED_COMPOUND

  /** @brief increment / decrement, as `_Float16` supports them */
  Half &operator++() {
    *this = Half(static_cast<float>(*this) + 1.0f);
    return *this;
  }
  Half &operator--() {
    *this = Half(static_cast<float>(*this) - 1.0f);
    return *this;
  }
  Half operator++(int) {
    Half prev = *this;
    ++*this;
    return prev;
  }
  Half operator--(int) {
    Half prev = *this;
    --*this;
    return prev;
  }
};

/**
 * Homogeneous (Half, Half) arithmetic -> Half. Each operation is computed in
 * float and rounded back, so a single operation is bit-identical to native
 * `_Float16`.
 *
 * Note on chained expressions: because the result type is Half, an expression
 * such as `a * b + c` rounds the intermediate to binary16 (the IEEE result).
 * GCC/Clang evaluate native `_Float16` arithmetic with excess precision by
 * default, keeping intermediates in float and rounding once at the end
 * (`-fexcess-precision=16` disables that and makes the two agree bit-for-bit).
 * ARM `__fp16` behaves like the excess-precision case as well, since it is a
 * storage format whose arithmetic promotes to float. The residual difference
 * is at most a couple of binary16 ULP and only in multi-operation expressions;
 * it is inherent to a value-type half and does not affect stored data, which
 * is bit-identical in every build.
 */
inline Half operator+(Half a, Half b) {
  return Half(static_cast<float>(a) + static_cast<float>(b));
}
inline Half operator-(Half a, Half b) {
  return Half(static_cast<float>(a) - static_cast<float>(b));
}
inline Half operator*(Half a, Half b) {
  return Half(static_cast<float>(a) * static_cast<float>(b));
}
inline Half operator/(Half a, Half b) {
  return Half(static_cast<float>(a) / static_cast<float>(b));
}

/** @brief unary negation -> Half */
inline Half operator-(Half a) { return Half(-static_cast<float>(a)); }

/* Homogeneous (Half, Half) comparisons -> bool */
inline bool operator<(Half a, Half b) {
  return static_cast<float>(a) < static_cast<float>(b);
}
inline bool operator>(Half a, Half b) {
  return static_cast<float>(a) > static_cast<float>(b);
}
inline bool operator<=(Half a, Half b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}
inline bool operator>=(Half a, Half b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}
inline bool operator==(Half a, Half b) {
  return static_cast<float>(a) == static_cast<float>(b);
}
inline bool operator!=(Half a, Half b) {
  return static_cast<float>(a) != static_cast<float>(b);
}

/**
 * Mixed (Half, arithmetic) / (arithmetic, Half) operators, templated + SFINAE
 * so they never apply to (Half, Half) and each is an EXACT match on both
 * arguments (disambiguating `Half op T`, otherwise ambiguous between T->Half
 * construction and Half->float conversion). Return types reproduce native
 * `_Float16 op T` promotion EXACTLY:
 *   - T integral  -> `Half`  (the integer converts to half; keeps e.g.
 *                             `(cond ? Half : -1 * Half)` single-typed)
 *   - T floating  -> `common_type<float,T>` (float, or double/long double)
 * Every op is computed in float and (for the integral case) rounded back.
 */
#define NNTR_HALF_MIXED_ARITH(OP)                                              \
  template <typename T>                                                        \
  inline typename std::enable_if<std::is_integral<T>::value, Half>::type       \
  operator OP(Half a, T b) {                                                   \
    return Half(static_cast<float>(a) OP b);                                   \
  }                                                                            \
  template <typename T>                                                        \
  inline typename std::enable_if<std::is_integral<T>::value, Half>::type       \
  operator OP(T a, Half b) {                                                   \
    return Half(a OP static_cast<float>(b));                                   \
  }                                                                            \
  template <typename T>                                                        \
  inline                                                                       \
    typename std::enable_if<std::is_floating_point<T>::value,                  \
                            typename std::common_type<float, T>::type>::type   \
    operator OP(Half a, T b) {                                                 \
    return static_cast<float>(a) OP b;                                         \
  }                                                                            \
  template <typename T>                                                        \
  inline                                                                       \
    typename std::enable_if<std::is_floating_point<T>::value,                  \
                            typename std::common_type<float, T>::type>::type   \
    operator OP(T a, Half b) {                                                 \
    return a OP static_cast<float>(b);                                         \
  }

NNTR_HALF_MIXED_ARITH(+)
NNTR_HALF_MIXED_ARITH(-)
NNTR_HALF_MIXED_ARITH(*)
NNTR_HALF_MIXED_ARITH(/)
#undef NNTR_HALF_MIXED_ARITH

/* Mixed comparisons -> bool for any arithmetic T (both operand orders). */
#define NNTR_HALF_MIXED_CMP(OP)                                                \
  template <typename T>                                                        \
  inline typename std::enable_if<std::is_arithmetic<T>::value, bool>::type     \
  operator OP(Half a, T b) {                                                   \
    return static_cast<float>(a) OP b;                                         \
  }                                                                            \
  template <typename T>                                                        \
  inline typename std::enable_if<std::is_arithmetic<T>::value, bool>::type     \
  operator OP(T a, Half b) {                                                   \
    return a OP static_cast<float>(b);                                         \
  }

NNTR_HALF_MIXED_CMP(<)
NNTR_HALF_MIXED_CMP(>)
NNTR_HALF_MIXED_CMP(<=)
NNTR_HALF_MIXED_CMP(>=)
NNTR_HALF_MIXED_CMP(==)
NNTR_HALF_MIXED_CMP(!=)
#undef NNTR_HALF_MIXED_CMP

static_assert(sizeof(Half) == 2, "GPU/CL ABI: half must be 2 bytes");
static_assert(std::is_trivially_copyable<Half>::value,
              "memcpy/memset/vector need trivial copy");
static_assert(std::is_standard_layout<Half>::value,
              "reinterpret_cast<unsigned short*> needs standard layout");

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __HALF_FP16_H__ */

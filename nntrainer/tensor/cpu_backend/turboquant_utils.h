// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   turboquant_utils.h
 * @date   28 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Utilities for TurboQuant 4-bit packing: 3-bit quantized data + 1-bit
 *         QJL signature packed into 4-bit per element (2 elements per byte).
 *
 * Packing layout (per byte):
 *   Lower nibble (bits 0-3): element[2i]   → [sign(1) | data(3)]
 *   Upper nibble (bits 4-7): element[2i+1] → [sign(1) | data(3)]
 *
 * Each 4-bit element:
 *   bits [2:0] = 3-bit quantized value (0-7)
 *   bit  [3]   = 1-bit QJL signature (0 or 1)
 */

#ifndef __TURBOQUANT_UTILS_H__
#define __TURBOQUANT_UTILS_H__
#ifdef __cplusplus

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nntrainer {

/** Group size for per-group scale factor computation */
constexpr int TURBOQUANT_GROUP_SIZE = 32;

/**
 * @brief Pack 3-bit quantized data + 1-bit QJL signature into 4-bit packed
 *        bytes.
 * @param[in]  q_vals   3-bit quantized values (0-7), num_elements entries
 * @param[in]  signs    1-bit QJL signatures (0 or 1), num_elements entries
 * @param[in]  num_elements total number of elements (must be even)
 * @param[out] out_packed packed output buffer, size = num_elements / 2
 */
inline void pack_turboquant_4bit(const uint8_t *q_vals, const uint8_t *signs,
                                 size_t num_elements, uint8_t *out_packed) {
  for (size_t i = 0; i < num_elements; i += 2) {
    uint8_t elem0 = (q_vals[i] & 0x07) | ((signs[i] & 0x01) << 3);
    uint8_t elem1 = (q_vals[i + 1] & 0x07) | ((signs[i + 1] & 0x01) << 3);
    out_packed[i / 2] = (elem1 << 4) | elem0;
  }
}

/**
 * @brief Unpack 4-bit packed byte into two 3-bit values and two 1-bit signs.
 * @param[in]  packed single packed byte
 * @param[out] val0   3-bit value of lower nibble element
 * @param[out] sign0  1-bit sign of lower nibble element
 * @param[out] val1   3-bit value of upper nibble element
 * @param[out] sign1  1-bit sign of upper nibble element
 */
inline void unpack_turboquant_4bit(uint8_t packed, uint8_t &val0,
                                   uint8_t &sign0, uint8_t &val1,
                                   uint8_t &sign1) {
  val0 = packed & 0x07;
  sign0 = (packed >> 3) & 0x01;
  val1 = (packed >> 4) & 0x07;
  sign1 = (packed >> 7) & 0x01;
}

/**
 * @brief Quantize FP32 values to 3-bit (0-7) with per-group absmax scale,
 *        compute QJL sign, and pack into 4-bit format.
 *
 * @param[in]  input        FP32 input values, length = num_elements
 * @param[in]  num_elements total number of elements (must be even)
 * @param[out] out_packed   packed 4-bit output, size = num_elements / 2
 * @param[out] out_scales   per-group scale factors, size = ceil(num_elements /
 *                          TURBOQUANT_GROUP_SIZE)
 */
inline void quantize_and_pack_turboquant(const float *input,
                                         size_t num_elements,
                                         uint8_t *out_packed,
                                         float *out_scales) {
  int num_groups =
    (num_elements + TURBOQUANT_GROUP_SIZE - 1) / TURBOQUANT_GROUP_SIZE;

  for (int g = 0; g < num_groups; ++g) {
    size_t start = g * TURBOQUANT_GROUP_SIZE;
    size_t end = start + TURBOQUANT_GROUP_SIZE;
    if (end > num_elements)
      end = num_elements;

    // Find absmax for this group
    float absmax = 0.0f;
    for (size_t i = start; i < end; ++i) {
      float av = std::fabs(input[i]);
      if (av > absmax)
        absmax = av;
    }

    // Scale: map [-absmax, absmax] to [0, 7] with zero_point = 4
    // dequant: val = scale * (q - 4), so scale = absmax / 3.0
    // (q ranges 0-7, centered at 4, so effective range is [-3, 3] * scale)
    float scale = (absmax > 0.0f) ? (absmax / 3.0f) : 1.0f;
    out_scales[g] = scale;
    float inv_scale = 1.0f / scale;

    // Quantize and pack pairs
    for (size_t i = start; i < end; i += 2) {
      // Quantize: q = clamp(round(val / scale) + 4, 0, 7)
      auto quantize_one = [inv_scale](float val) -> uint8_t {
        int q = (int)std::round(val * inv_scale) + 4;
        if (q < 0)
          q = 0;
        if (q > 7)
          q = 7;
        return (uint8_t)q;
      };

      uint8_t q0 = quantize_one(input[i]);
      uint8_t s0 = (input[i] >= 0.0f) ? 1 : 0; // QJL sign: 1=positive

      uint8_t q1 = 4, s1 = 1; // neutral defaults: zero_point=4, sign=positive
      if (i + 1 < end) {
        q1 = quantize_one(input[i + 1]);
        s1 = (input[i + 1] >= 0.0f) ? 1 : 0;
      }

      uint8_t elem0 = (q0 & 0x07) | ((s0 & 0x01) << 3);
      uint8_t elem1 = (q1 & 0x07) | ((s1 & 0x01) << 3);
      out_packed[(i - start) / 2 + start / 2] = (elem1 << 4) | elem0;
    }
  }
}

/**
 * @brief Dequantize a single 3-bit value with scale and zero_point=4.
 *        dequant_val = scale * (q_val - 4)
 * @param[in] q_val 3-bit quantized value (0-7)
 * @param[in] scale per-group scale factor
 * @return dequantized float value
 */
inline float dequantize_turboquant(uint8_t q_val, float scale) {
  return scale * ((float)q_val - 4.0f);
}

/**
 * @brief In-place Walsh-Hadamard Transform (WHT).
 *        Transforms x[] of length n (must be power of 2).
 *        Normalizes by 1/sqrt(n) so the transform is orthogonal.
 * @param[in,out] x  data array, length n
 * @param[in]     n  length (must be power of 2)
 */
inline void hadamard_transform(float *x, int n) {
  for (int len = 1; len < n; len <<= 1) {
    for (int i = 0; i < n; i += len << 1) {
      for (int j = 0; j < len; ++j) {
        float u = x[i + j];
        float v = x[i + j + len];
        x[i + j] = u + v;
        x[i + j + len] = u - v;
      }
    }
  }
  float inv_sqrt_n = 1.0f / std::sqrt((float)n);
  for (int i = 0; i < n; ++i)
    x[i] *= inv_sqrt_n;
}

/**
 * @brief Generate deterministic random sign vector (±1) for rotation.
 *        Uses a simple LCG seeded by seed value.
 * @param[out] signs  output array of +1.0f / -1.0f, length n
 * @param[in]  n      length
 * @param[in]  seed   seed for deterministic generation
 */
inline void generate_random_signs(float *signs, int n,
                                  uint32_t seed = 0x5EED1234) {
  uint32_t state = seed;
  for (int i = 0; i < n; ++i) {
    // LCG: state = state * 1664525 + 1013904223
    state = state * 1664525u + 1013904223u;
    signs[i] = (state & 0x80000000u) ? -1.0f : 1.0f;
  }
}

/**
 * @brief Apply PolarQuant rotation: R = (1/sqrt(n)) * H * D
 *        where H = Hadamard, D = diag(random_signs).
 *        Equivalent to: element-wise multiply by signs, then Hadamard.
 * @param[in]  input   input vector, length n
 * @param[out] output  rotated output, length n
 * @param[in]  signs   random sign vector (±1), length n
 * @param[in]  n       length (must be power of 2)
 */
inline void apply_rotation(const float *input, float *output,
                           const float *signs, int n) {
  for (int i = 0; i < n; ++i)
    output[i] = input[i] * signs[i];
  hadamard_transform(output, n);
}

/**
 * @brief Apply inverse rotation: R^T = D * (1/sqrt(n)) * H
 *        Since H is self-inverse (up to scaling, already handled),
 *        inverse = Hadamard then multiply by signs.
 * @param[in,out] data   data to inverse-rotate in place, length n
 * @param[in]     signs  same random sign vector used for forward rotation
 * @param[in]     n      length (must be power of 2)
 */
inline void apply_inverse_rotation(float *data, const float *signs, int n) {
  hadamard_transform(data, n);
  for (int i = 0; i < n; ++i)
    data[i] *= signs[i];
}

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TURBOQUANT_UTILS_H__ */

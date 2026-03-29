// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   turboquant_utils.h
 * @date   28 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  TurboQuant KV cache compression utilities.
 *
 *         v1: 3-bit uniform quantization + 1-bit sign, per-group scale
 *         v2: Norm normalization + Hadamard rotation + Lloyd-Max codebook
 *             (paper Algorithm 1: MSE-optimal)
 *
 * Packing layout (per byte):
 *   Lower nibble (bits 0-3): element[2i]   → [sign(1) | data(3)]
 *   Upper nibble (bits 4-7): element[2i+1] → [sign(1) | data(3)]
 */

#ifndef __TURBOQUANT_UTILS_H__
#define __TURBOQUANT_UTILS_H__
#ifdef __cplusplus

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nntrainer {

/** Group size for per-group scale factor computation (v1) */
constexpr int TURBOQUANT_GROUP_SIZE = 32;

/***********************************************************************
 * Hadamard transform and rotation utilities
 ***********************************************************************/

/**
 * @brief In-place Walsh-Hadamard Transform (WHT).
 *        Normalizes by 1/sqrt(n) so the transform is orthogonal.
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

/** Generate deterministic random sign vector (±1) via LCG. */
inline void generate_random_signs(float *signs, int n,
                                  uint32_t seed = 0x5EED1234) {
  uint32_t state = seed;
  for (int i = 0; i < n; ++i) {
    state = state * 1664525u + 1013904223u;
    signs[i] = (state & 0x80000000u) ? -1.0f : 1.0f;
  }
}

/** Forward rotation: R = (1/sqrt(n)) * H * D */
inline void apply_rotation(const float *input, float *output,
                           const float *signs, int n) {
  for (int i = 0; i < n; ++i)
    output[i] = input[i] * signs[i];
  hadamard_transform(output, n);
}

/** Inverse rotation: R^T = D * (1/sqrt(n)) * H */
inline void apply_inverse_rotation(float *data, const float *signs, int n) {
  hadamard_transform(data, n);
  for (int i = 0; i < n; ++i)
    data[i] *= signs[i];
}

/***********************************************************************
 * Lloyd-Max optimal codebooks (precomputed for Beta distribution)
 ***********************************************************************/

struct LloydMaxCodebook {
  float centroids[8];
  float boundaries[7];
};

/** d=64, 3-bit (8 levels) */
static constexpr LloydMaxCodebook CODEBOOK_D64 = {
  {-0.26391393f, -0.16616786f, -0.09383226f, -0.03046918f,
    0.03046918f,  0.09383226f,  0.16616786f,  0.26391393f},
  {-0.21504089f, -0.13000006f, -0.06215072f, 0.0f,
    0.06215072f,  0.13000006f,  0.21504089f}
};

/** d=128, 3-bit (8 levels) */
static constexpr LloydMaxCodebook CODEBOOK_D128 = {
  {-0.18839719f, -0.11813977f, -0.06658561f, -0.02160431f,
    0.02160431f,  0.06658561f,  0.11813977f,  0.18839719f},
  {-0.15326848f, -0.09236269f, -0.04409496f, 0.0f,
    0.04409496f,  0.09236269f,  0.15326848f}
};

inline const LloydMaxCodebook &get_codebook(int head_dim) {
  if (head_dim == 64)
    return CODEBOOK_D64;
  return CODEBOOK_D128;
}

/** Lloyd-Max quantize: boundary search for optimal bin index. */
inline uint8_t lloydmax_quantize(float val, const LloydMaxCodebook &cb) {
  int idx = 0;
  for (int i = 0; i < 7; ++i) {
    if (val > cb.boundaries[i])
      idx = i + 1;
  }
  return (uint8_t)idx;
}

/***********************************************************************
 * v2: Norm + Rotation + Lloyd-Max (paper Algorithm 1)
 ***********************************************************************/

/**
 * @brief Full TurboQuant v2 quantize pipeline (paper Algorithm 1):
 *        1. Compute norm, normalize to unit vector
 *        2. Apply Hadamard rotation
 *        3. Lloyd-Max quantize each coordinate (3-bit)
 *        4. Pack into 4-bit (3-bit index + 1-bit sign, sign unused in v2)
 */
inline void turboquant_quantize_head(const float *input, int head_dim,
                                     uint8_t *out_packed, float *out_norm,
                                     const float *rot_signs,
                                     const LloydMaxCodebook &cb) {
  float norm_sq = 0.0f;
  for (int i = 0; i < head_dim; ++i)
    norm_sq += input[i] * input[i];
  float norm = std::sqrt(norm_sq);
  *out_norm = norm;

  std::vector<float> rotated(head_dim);
  float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
  for (int i = 0; i < head_dim; ++i)
    rotated[i] = input[i] * inv_norm * rot_signs[i];
  hadamard_transform(rotated.data(), head_dim);

  for (int d = 0; d < head_dim; d += 2) {
    uint8_t q0 = lloydmax_quantize(rotated[d], cb);
    uint8_t q1 = 4;
    if (d + 1 < head_dim)
      q1 = lloydmax_quantize(rotated[d + 1], cb);

    // Pack: 3-bit index in lower bits, bit[3] unused (set to 0)
    out_packed[d / 2] = (q1 << 4) | q0;
  }
}

/**
 * @brief Dequantize one head: centroid lookup + inverse rotation + norm rescale.
 */
inline void turboquant_dequantize_head(const uint8_t *packed, float norm,
                                       int head_dim, float *output,
                                       const float *rot_signs,
                                       const LloydMaxCodebook &cb) {
  for (int d = 0; d < head_dim; d += 2) {
    uint8_t byte = packed[d / 2];
    uint8_t q0 = byte & 0x07;
    uint8_t q1 = (byte >> 4) & 0x07;
    output[d] = cb.centroids[q0];
    if (d + 1 < head_dim)
      output[d + 1] = cb.centroids[q1];
  }
  hadamard_transform(output, head_dim);
  for (int i = 0; i < head_dim; ++i)
    output[i] *= rot_signs[i] * norm;
}

/***********************************************************************
 * v1: Original uniform quantization (kept for backward compatibility)
 ***********************************************************************/

inline void pack_turboquant_4bit(const uint8_t *q_vals, const uint8_t *signs,
                                 size_t num_elements, uint8_t *out_packed) {
  for (size_t i = 0; i < num_elements; i += 2) {
    uint8_t elem0 = (q_vals[i] & 0x07) | ((signs[i] & 0x01) << 3);
    uint8_t elem1 = (q_vals[i + 1] & 0x07) | ((signs[i + 1] & 0x01) << 3);
    out_packed[i / 2] = (elem1 << 4) | elem0;
  }
}

inline void unpack_turboquant_4bit(uint8_t packed, uint8_t &val0,
                                   uint8_t &sign0, uint8_t &val1,
                                   uint8_t &sign1) {
  val0 = packed & 0x07;
  sign0 = (packed >> 3) & 0x01;
  val1 = (packed >> 4) & 0x07;
  sign1 = (packed >> 7) & 0x01;
}

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

    float absmax = 0.0f;
    for (size_t i = start; i < end; ++i) {
      float av = std::fabs(input[i]);
      if (av > absmax)
        absmax = av;
    }

    float scale = (absmax > 0.0f) ? (absmax / 3.0f) : 1.0f;
    out_scales[g] = scale;
    float inv_scale = 1.0f / scale;

    for (size_t i = start; i < end; i += 2) {
      auto quantize_one = [inv_scale](float val) -> uint8_t {
        int q = (int)std::round(val * inv_scale) + 4;
        if (q < 0)
          q = 0;
        if (q > 7)
          q = 7;
        return (uint8_t)q;
      };

      uint8_t q0 = quantize_one(input[i]);
      uint8_t s0 = (input[i] >= 0.0f) ? 1 : 0;

      uint8_t q1 = 4, s1 = 1;
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

inline float dequantize_turboquant(uint8_t q_val, float scale) {
  return scale * ((float)q_val - 4.0f);
}

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TURBOQUANT_UTILS_H__ */

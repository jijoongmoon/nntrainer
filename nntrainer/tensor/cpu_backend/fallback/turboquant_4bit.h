// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   turboquant_4bit.h
 * @date   28 March 2026
 * @brief  TurboQuant 4-bit packing utilities for KV cache quantization.
 *         Packs 3-bit quantized data + 1-bit QJL signature into 4-bit format.
 *         Two 4-bit elements are stored per uint8_t byte.
 *
 * @note   Bit layout per 4-bit element:
 *         [bit3: QJL sign | bit2-bit0: 3-bit quantized value (0~7)]
 *
 *         Byte layout (two elements packed):
 *         [upper 4-bit: elem1 | lower 4-bit: elem0]
 */

#ifndef __TURBOQUANT_4BIT_H__
#define __TURBOQUANT_4BIT_H__

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nntrainer {

/**
 * @brief Quantize FP32 values to 3-bit (0~7) with per-group min-max scaling.
 *        Also computes QJL 1-bit sign (sign of quantization residual) and
 *        the QJL correction scale (mean absolute residual).
 *
 * @param[in]  input         FP32 input values
 * @param[out] q_vals        3-bit quantized output (0~7), stored as int8_t
 * @param[out] signs         1-bit QJL signature (0 or 1), stored as uint8_t
 * @param[out] scale         quantization scale: (max - min) / 7
 * @param[out] zero_point    quantization zero point: min
 * @param[out] qjl_scale     QJL correction scale: mean(|residual|)
 * @param[in]  num_elements  number of elements to quantize
 */
inline void quantize_fp32_to_3bit(const float *input, int8_t *q_vals,
                                  uint8_t *signs, float *scale,
                                  float *zero_point, float *qjl_scale,
                                  size_t num_elements) {
  if (num_elements == 0)
    return;

  // Step 1: Find min/max for scale computation
  float min_val = input[0];
  float max_val = input[0];
  for (size_t i = 1; i < num_elements; ++i) {
    if (input[i] < min_val)
      min_val = input[i];
    if (input[i] > max_val)
      max_val = input[i];
  }

  // Step 2: Compute scale and zero point
  float range = max_val - min_val;
  float s = (range > 1e-10f) ? range / 7.0f : 1e-10f;
  float zp = min_val;

  *scale = s;
  *zero_point = zp;

  // Step 3: Quantize and compute residuals
  float abs_residual_sum = 0.0f;
  for (size_t i = 0; i < num_elements; ++i) {
    // Quantize: q = clamp(round((x - zp) / scale), 0, 7)
    float normalized = (input[i] - zp) / s;
    int q = static_cast<int>(normalized + 0.5f); // round
    if (q < 0)
      q = 0;
    if (q > 7)
      q = 7;
    q_vals[i] = static_cast<int8_t>(q);

    // Dequantize to compute residual
    float dequantized = static_cast<float>(q) * s + zp;
    float residual = input[i] - dequantized;

    // QJL sign: 1 if residual >= 0, 0 if residual < 0
    signs[i] = (residual >= 0.0f) ? 1 : 0;

    abs_residual_sum += std::fabs(residual);
  }

  // Step 4: QJL correction scale = mean(|residual|)
  *qjl_scale = abs_residual_sum / static_cast<float>(num_elements);
}

/**
 * @brief Pack 3-bit quantized data + 1-bit QJL sign into 4-bit packed format.
 *        Two 4-bit elements per uint8_t byte.
 *
 * @param[in]  q_vals        3-bit quantized values (0~7)
 * @param[in]  signs         1-bit QJL signatures (0 or 1)
 * @param[in]  num_elements  number of elements (must be even)
 * @param[out] out_packed    packed output buffer (size = num_elements / 2)
 */
inline void pack_turboquant_4bit(const int8_t *q_vals, const uint8_t *signs,
                                 size_t num_elements, uint8_t *out_packed) {
  for (size_t i = 0; i < num_elements; i += 2) {
    uint8_t elem0 = (q_vals[i] & 0x07) | ((signs[i] & 0x01) << 3);
    uint8_t elem1 = (q_vals[i + 1] & 0x07) | ((signs[i + 1] & 0x01) << 3);
    out_packed[i / 2] = (elem1 << 4) | elem0;
  }
}

/**
 * @brief Unpack 4-bit packed data back to 3-bit values and 1-bit signs.
 *
 * @param[in]  packed        packed input buffer (size = num_elements / 2)
 * @param[out] q_vals        3-bit quantized values (0~7)
 * @param[out] signs         1-bit QJL signatures (0 or 1)
 * @param[in]  num_elements  number of elements to unpack (must be even)
 */
inline void unpack_turboquant_4bit(const uint8_t *packed, int8_t *q_vals,
                                   uint8_t *signs, size_t num_elements) {
  for (size_t i = 0; i < num_elements; i += 2) {
    uint8_t byte = packed[i / 2];
    // Lower 4 bits -> elem0
    q_vals[i] = static_cast<int8_t>(byte & 0x07);
    signs[i] = (byte >> 3) & 0x01;
    // Upper 4 bits -> elem1
    q_vals[i + 1] = static_cast<int8_t>((byte >> 4) & 0x07);
    signs[i + 1] = (byte >> 7) & 0x01;
  }
}

/**
 * @brief Dequantize 3-bit value with QJL correction to FP32.
 *
 * @param[in] q_val     3-bit quantized value (0~7)
 * @param[in] sign      1-bit QJL sign (0 or 1)
 * @param[in] scale     quantization scale
 * @param[in] zp        zero point
 * @param[in] qjl_scale QJL correction scale
 * @return    float     dequantized + corrected value
 */
inline float dequantize_4bit_qjl(int8_t q_val, uint8_t sign, float scale,
                                 float zp, float qjl_scale) {
  float dequantized = static_cast<float>(q_val) * scale + zp;
  float correction = (sign == 1) ? qjl_scale : -qjl_scale;
  return dequantized + correction;
}

/**
 * @brief Combined: quantize FP32 -> pack to 4-bit in one pass.
 *        Operates per-head: quantizes head_dim elements, writes
 *        head_dim/2 packed bytes plus scale/zp/qjl_scale metadata.
 *
 * @param[in]  input         FP32 input (head_dim elements)
 * @param[out] out_packed    packed output (head_dim/2 bytes)
 * @param[out] scale         output scale
 * @param[out] zero_point    output zero point
 * @param[out] qjl_scale     output QJL correction scale
 * @param[in]  head_dim      number of elements per head (must be even)
 * @param[in]  q_buf         temp buffer for q_vals (head_dim int8_t)
 * @param[in]  s_buf         temp buffer for signs (head_dim uint8_t)
 */
inline void quantize_and_pack_4bit(const float *input, uint8_t *out_packed,
                                   float *scale, float *zero_point,
                                   float *qjl_scale, size_t head_dim,
                                   int8_t *q_buf, uint8_t *s_buf) {
  quantize_fp32_to_3bit(input, q_buf, s_buf, scale, zero_point, qjl_scale,
                        head_dim);
  pack_turboquant_4bit(q_buf, s_buf, head_dim, out_packed);
}

/**
 * @brief Combined: unpack 4-bit + dequantize with QJL correction in one pass.
 *
 * @param[in]  packed        packed input (head_dim/2 bytes)
 * @param[out] output        FP32 output (head_dim elements)
 * @param[in]  scale         quantization scale
 * @param[in]  zero_point    zero point
 * @param[in]  qjl_scale     QJL correction scale
 * @param[in]  head_dim      number of elements (must be even)
 */
inline void unpack_and_dequantize_4bit(const uint8_t *packed, float *output,
                                       float scale, float zero_point,
                                       float qjl_scale, size_t head_dim) {
  for (size_t i = 0; i < head_dim; i += 2) {
    uint8_t byte = packed[i / 2];

    int8_t q0 = static_cast<int8_t>(byte & 0x07);
    uint8_t s0 = (byte >> 3) & 0x01;
    output[i] = dequantize_4bit_qjl(q0, s0, scale, zero_point, qjl_scale);

    int8_t q1 = static_cast<int8_t>((byte >> 4) & 0x07);
    uint8_t s1 = (byte >> 7) & 0x01;
    output[i + 1] =
      dequantize_4bit_qjl(q1, s1, scale, zero_point, qjl_scale);
  }
}

/**
 * @brief Scalar (fallback) implementation of compute_kcaches for 4-bit packed
 *        KV cache. Computes Q · K^T attention scores.
 *
 * For each KV head n, for each GQA group g, for each cached row:
 *   output[row, n*gqa_size+g] = dot(query[n*gqa+g], key_cache[row,n]) /
 * sqrt(head_dim)
 *
 * @param[in]  in               Query: float[num_heads_Q * head_dim]
 * @param[in]  kcache_packed    4-bit packed key cache:
 *                              uint8[num_rows * num_cache_head * head_dim/2]
 * @param[out] output           Attention scores: float[row_cnt *
 *                              num_cache_head * gqa_size]
 * @param[in]  num_rows         Total cached rows (current timestep)
 * @param[in]  num_cache_head   Number of KV heads
 * @param[in]  head_dim         Dimension per head (must be even)
 * @param[in]  gqa_size         GQA group size (num_heads_Q / num_cache_head)
 * @param[in]  tile_size        Tile size for loop blocking
 * @param[in]  scales           Per-head per-row scale: float[num_rows *
 *                              num_cache_head]
 * @param[in]  zero_points      Per-head per-row zero point
 * @param[in]  qjl_scales       Per-head per-row QJL correction scale
 * @param[in]  local_window_size  Sliding window size (UINT_MAX = no window)
 * @param[in]  head_start       First head to process
 * @param[in]  head_end         Past-the-end head (-1 = all heads)
 */
inline void compute_kcaches_4bit(
  const float *in, const uint8_t *kcache_packed, float *output, int num_rows,
  int num_cache_head, int head_dim, int gqa_size, int tile_size,
  const float *scales, const float *zero_points, const float *qjl_scales,
  size_t local_window_size = UINT_MAX, int head_start = 0,
  int head_end = -1) {

  int actual_head_end = (head_end < 0) ? num_cache_head : head_end;

  int start_row = (static_cast<size_t>(num_rows) < local_window_size)
                    ? 0
                    : num_rows - static_cast<int>(local_window_size);
  int row_cnt = (static_cast<size_t>(num_rows) < local_window_size)
                  ? num_rows
                  : static_cast<int>(local_window_size);
  int tile_count = (row_cnt + tile_size - 1) / tile_size;

  int packed_head_dim = head_dim / 2;
  float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Temp buffer for dequantized key row
  // Using stack allocation for typical head_dim (128~256)
  std::vector<float> tmp_fp32(head_dim);

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int t = 0; t < tile_count; ++t) {
      int row_tile_start = t * tile_size;
      int tile_rows = std::min(tile_size, row_cnt - row_tile_start);

      for (int g = 0; g < gqa_size; ++g) {
        const float *in_ptr = in + n * gqa_size * head_dim + g * head_dim;

        for (int t_row = 0; t_row < tile_rows; ++t_row) {
          int row = start_row + row_tile_start + t_row;

          // Get packed key data for this row and head
          const uint8_t *kptr =
            kcache_packed + (row * num_cache_head + n) * packed_head_dim;

          // Get quantization metadata for this row and head
          float s = scales[row * num_cache_head + n];
          float zp = zero_points[row * num_cache_head + n];
          float qs = qjl_scales[row * num_cache_head + n];

          // Unpack + dequantize
          unpack_and_dequantize_4bit(kptr, tmp_fp32.data(), s, zp, qs,
                                     head_dim);

          // Dot product
          float sum = 0.0f;
          for (int i = 0; i < head_dim; ++i) {
            sum += in_ptr[i] * tmp_fp32[i];
          }

          output[(row - start_row) * num_cache_head * gqa_size +
                 n * gqa_size + g] = sum * inv_sqrt_hd;
        }
      }
    }
  }
}

/**
 * @brief Scalar (fallback) implementation of compute_vcaches for 4-bit packed
 *        value cache. Computes weighted sum: output = attn_weights * V_cache.
 *
 * @param[in]  row_num          Current row (timestep - 1, 0-indexed)
 * @param[in]  in               Attention weights after softmax:
 *                              float[row_cnt * num_cache_head * gqa_size]
 * @param[in]  vcache_packed    4-bit packed value cache
 * @param[out] output           float[num_heads_Q * head_dim]
 * @param[in]  num_cache_head   Number of KV heads
 * @param[in]  gqa_size         GQA group size
 * @param[in]  head_dim         Dimension per head
 * @param[in]  scales           Per-head per-row scale
 * @param[in]  zero_points      Per-head per-row zero point
 * @param[in]  qjl_scales       Per-head per-row QJL correction scale
 * @param[in]  local_window_size  Sliding window size
 * @param[in]  head_start       First head to process
 * @param[in]  head_end         Past-the-end head (-1 = all heads)
 */
inline void compute_vcaches_4bit(
  int row_num, const float *in, const uint8_t *vcache_packed, float *output,
  int num_cache_head, int gqa_size, int head_dim, const float *scales,
  const float *zero_points, const float *qjl_scales,
  size_t local_window_size = UINT_MAX, int head_start = 0,
  int head_end = -1) {

  int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  int packed_head_dim = head_dim / 2;

  std::vector<float> tmp_fp32(head_dim);

  for (int n = head_start; n < actual_head_end; ++n) {
    // Zero output for all GQA groups of this head
    for (int h = 0; h < gqa_size; ++h) {
      for (int d = 0; d < head_dim; ++d) {
        output[(n * gqa_size + h) * head_dim + d] = 0.0f;
      }
    }

    int j_start = (static_cast<size_t>(row_num) < local_window_size)
                    ? 0
                    : row_num + 1 - static_cast<int>(local_window_size);

    for (int j = j_start; j <= row_num; ++j) {
      const uint8_t *vptr =
        vcache_packed + (j * num_cache_head + n) * packed_head_dim;

      float s = scales[j * num_cache_head + n];
      float zp = zero_points[j * num_cache_head + n];
      float qs = qjl_scales[j * num_cache_head + n];

      // Unpack + dequantize value row
      unpack_and_dequantize_4bit(vptr, tmp_fp32.data(), s, zp, qs, head_dim);

      for (int h = 0; h < gqa_size; ++h) {
        int attn_idx =
          (static_cast<size_t>(row_num) < local_window_size
             ? j
             : j - (row_num + 1 - static_cast<int>(local_window_size)));
        float a_val =
          in[attn_idx * gqa_size * num_cache_head + n * gqa_size + h];

        // Scale-accumulate: output[head] += attn_weight * value
        for (int d = 0; d < head_dim; ++d) {
          output[(n * gqa_size + h) * head_dim + d] += a_val * tmp_fp32[d];
        }
      }
    }
  }
}

} // namespace nntrainer

#endif // __TURBOQUANT_4BIT_H__

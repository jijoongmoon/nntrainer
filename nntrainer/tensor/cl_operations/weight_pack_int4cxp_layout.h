// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    weight_pack_int4cxp_layout.h
 * @date    15 May 2026
 * @brief   Host helpers for the per-channel signed-int4 weight layout
 *          (B1-int4 of the ML Drift parity work, arXiv:2505.00232 §4.2).
 *          Pairs with the GPU dequant kernel in
 *          cl_kernels/weight_pack_int4cxp_dequant.cl — the offset macro and
 *          dequant formula must stay in lockstep.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * Storage:
 *   * Bytes — O * H * W * ceil(I/4) * 2 bytes.
 *     byte_offset(o, i, h, w) = ((o*H + h)*W + w) * slice_I*2 + (i >> 1)
 *     Low nibble  = i with (i & 1) == 0
 *     High nibble = next i
 *   * Each nibble holds  uint4 = (signed_int4 + 8)  so the bit pattern is
 *     0..15. The first iteration is NOT byte-compatible with KAI qsi4cxp.
 *   * Scale buffer — O fp32 values, one per output channel.
 */
#ifndef __NNTRAINER_WEIGHT_PACK_INT4CXP_LAYOUT_H__
#define __NNTRAINER_WEIGHT_PACK_INT4CXP_LAYOUT_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nntrainer::weight_pack_int4cxp {

inline std::size_t num_bytes(int O, int I, int H, int W) {
  const std::size_t slice_I = (I + 3) / 4;
  return static_cast<std::size_t>(O) * H * W * slice_I * 2;
}

inline std::size_t num_scales(int O) { return static_cast<std::size_t>(O); }

inline std::size_t byte_offset(int o, int i, int h, int w, int O, int I, int H,
                               int W) {
  (void)O;
  const std::size_t slice_I = (I + 3) / 4;
  return (((static_cast<std::size_t>(o) * H + h) * W + w) * slice_I * 2) +
         (i >> 1);
}

inline float dequant(std::uint8_t byte_val, int i, float scale_o) {
  const std::uint8_t nibble = (byte_val >> ((i & 1) * 4)) & 0xF;
  return (static_cast<float>(nibble) - 8.0f) * scale_o;
}

/**
 * Quantize an OIHW fp32 tensor to per-channel symmetric int4 + scale.
 *
 *   scale[o] = max_i,h,w |src[o, i, h, w]| / 7
 *   q        = round(src / scale[o])  clipped to [-8, +7]
 *   stored   = q + 8                   (uint4 in [0, 15])
 *
 * @param[in]  src       fp32 buffer of length O*I*H*W
 * @param[out] dst_bytes uchar buffer of length num_bytes(O,I,H,W); existing
 *                       content is zeroed first so padding nibbles are 0
 *                       (uint4=0 -> int4=-8; harmless because the masked
 *                       slice_I padding is never read in dequant or matmul)
 * @param[out] dst_scale fp32 buffer of length num_scales(O)
 */
inline void pack_fp32_to_int4cxp(const float *src, std::uint8_t *dst_bytes,
                                 float *dst_scale, int O, int I, int H,
                                 int W) {
  std::fill_n(dst_bytes, num_bytes(O, I, H, W), std::uint8_t{0});
  const std::size_t ihw = static_cast<std::size_t>(I) * H * W;
  for (int o = 0; o < O; ++o) {
    float amax = 0.0f;
    for (std::size_t k = 0; k < ihw; ++k) {
      const float a = std::fabs(src[o * ihw + k]);
      if (a > amax)
        amax = a;
    }
    const float scale = (amax > 0.0f) ? (amax / 7.0f) : 1.0f;
    dst_scale[o] = scale;
    const float inv = 1.0f / scale;
    for (int i = 0; i < I; ++i) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          const std::size_t src_idx =
            ((static_cast<std::size_t>(o) * I + i) * H + h) * W + w;
          int q = static_cast<int>(std::lrintf(src[src_idx] * inv));
          if (q < -8)
            q = -8;
          if (q > 7)
            q = 7;
          const std::uint8_t nibble = static_cast<std::uint8_t>(q + 8);
          const std::size_t boff = byte_offset(o, i, h, w, O, I, H, W);
          const int shift = (i & 1) * 4;
          // The byte was zero-cleared above; for even i this writes the low
          // nibble (high nibble is still 0), for odd i it ORs in the high
          // nibble on top of the even-i write.
          dst_bytes[boff] = static_cast<std::uint8_t>(
            (dst_bytes[boff] & ~(0xF << shift)) | (nibble << shift));
        }
      }
    }
  }
}

/**
 * Inverse of pack_fp32_to_int4cxp. Reads (bytes, scale) and writes OIHW fp32
 * — deterministic, lossless w.r.t. the quantized int4 representation.
 * Useful as the reference for the GPU dequant kernel.
 */
inline void unpack_int4cxp_to_fp32(const std::uint8_t *src_bytes,
                                   const float *src_scale, float *dst, int O,
                                   int I, int H, int W) {
  for (int o = 0; o < O; ++o) {
    for (int i = 0; i < I; ++i) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          const std::size_t boff = byte_offset(o, i, h, w, O, I, H, W);
          const std::size_t dst_idx =
            ((static_cast<std::size_t>(o) * I + i) * H + h) * W + w;
          dst[dst_idx] = dequant(src_bytes[boff], i, src_scale[o]);
        }
      }
    }
  }
}

} // namespace nntrainer::weight_pack_int4cxp
#endif /* __NNTRAINER_WEIGHT_PACK_INT4CXP_LAYOUT_H__ */

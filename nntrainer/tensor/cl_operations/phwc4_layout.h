// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    phwc4_layout.h
 * @date    15 May 2026
 * @brief   Host-side helpers for the GPU PHWC4 (4-channel slice) tensor
 *          layout introduced by the ML Drift parity work (arXiv:2505.00232).
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * The OpenCL kernel side keeps the same offset formula in PHWC4_OFFSET
 * (see cl_kernels/phwc4_identity.cl). These host helpers must stay in
 * lockstep with that macro: the round-trip unit test verifies they do.
 */
#ifndef __NNTRAINER_PHWC4_LAYOUT_H__
#define __NNTRAINER_PHWC4_LAYOUT_H__

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace nntrainer::phwc4 {

/**
 * Number of PHWC4 elements (after slice padding) for a logical (B, C, H, W)
 * tensor. The 4-channel slice padding rounds C up to the next multiple of 4.
 */
inline std::size_t num_elements(int B, int C, int H, int W) {
  const std::size_t slices = (C + 3) / 4;
  return slices * static_cast<std::size_t>(H) * static_cast<std::size_t>(W) *
         static_cast<std::size_t>(B) * 4;
}

/**
 * Physical offset within a PHWC4 buffer for logical (b, c, h, w). Mirrors
 * PHWC4_OFFSET in the OpenCL header.
 */
inline std::size_t offset(int b, int c, int h, int w, int B, int C, int H,
                          int W) {
  const std::size_t slice = static_cast<std::size_t>(c >> 2);
  const std::size_t c4 = static_cast<std::size_t>(c & 3);
  return ((((slice * static_cast<std::size_t>(H) + h) *
              static_cast<std::size_t>(W) +
            w) *
             static_cast<std::size_t>(B) +
           b) *
            4 +
          c4);
}

/**
 * Pack an NCHW source into a PHWC4 destination. The destination must be
 * sized for num_elements() — the tail channels of the last slice (when C is
 * not a multiple of 4) are zero-padded.
 */
template <typename T>
void pack_nchw_to_phwc4(const T *src, T *dst, int B, int C, int H, int W) {
  std::fill_n(dst, num_elements(B, C, H, W), T{});
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          const std::size_t src_idx =
            ((static_cast<std::size_t>(b) * C + c) * H + h) * W + w;
          dst[offset(b, c, h, w, B, C, H, W)] = src[src_idx];
        }
      }
    }
  }
}

/**
 * Inverse of pack_nchw_to_phwc4. Destination is sized B*C*H*W.
 */
template <typename T>
void unpack_phwc4_to_nchw(const T *src, T *dst, int B, int C, int H, int W) {
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          const std::size_t dst_idx =
            ((static_cast<std::size_t>(b) * C + c) * H + h) * W + w;
          dst[dst_idx] = src[offset(b, c, h, w, B, C, H, W)];
        }
      }
    }
  }
}

} // namespace nntrainer::phwc4

#endif /* __NNTRAINER_PHWC4_LAYOUT_H__ */

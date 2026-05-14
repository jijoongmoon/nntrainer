// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    weight_pack_layout.h
 * @date    15 May 2026
 * @brief   Host-side helpers for the GPU weight repack layout introduced by
 *          B1 of the ML Drift parity work (arXiv:2505.00232 §3.1). Pairs with
 *          B0's PHWC4 activation layout: the 4-input-channel slice on weights
 *          aligns with the 4-channel slice on the activation so the upcoming
 *          1x1-conv-as-matmul kernel (B4) can vload4 across input channels.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * The OpenCL kernel side keeps the same offset formula in
 * WEIGHT_PACK_OFFSET (see cl_kernels/weight_pack_identity.cl). These host
 * helpers must stay in lockstep with that macro: the round-trip unit test
 * verifies they do.
 *
 * Layout: [slice_O][O4][H][W][slice_I][I4]
 *   slice_O = ceil(O/4), O4 = o & 3
 *   slice_I = ceil(I/4), I4 = i & 3
 *
 * Algebraically the layout collapses to:
 *   offset(o, i, h, w) = ((o*H + h)*W + w) * (slice_I*4) + i
 * i.e. an OIHW row-major layout where the input-channel stride is padded up
 * to the next multiple of 4 so vload4 stays in bounds.
 */
#ifndef __NNTRAINER_WEIGHT_PACK_LAYOUT_H__
#define __NNTRAINER_WEIGHT_PACK_LAYOUT_H__

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace nntrainer::weight_pack {

/**
 * Number of elements in the weight-pack buffer (after padding I up to a
 * multiple of 4). The buffer is exactly O * H * W * slice_I * 4 elements.
 */
inline std::size_t num_elements(int O, int I, int H, int W) {
  const std::size_t slice_I = (I + 3) / 4;
  return static_cast<std::size_t>(O) * H * W * slice_I * 4;
}

/**
 * Physical offset of logical (o, i, h, w) in the weight-pack buffer. Mirrors
 * WEIGHT_PACK_OFFSET in the OpenCL header.
 */
inline std::size_t offset(int o, int i, int h, int w, int O, int I, int H,
                          int W) {
  (void)O;
  const std::size_t slice_I_x4 = static_cast<std::size_t>((I + 3) / 4) * 4;
  return ((static_cast<std::size_t>(o) * H + h) * W + w) * slice_I_x4 + i;
}

/**
 * Pack an OIHW source into a weight-pack destination. The destination must
 * be sized for num_elements() — the tail of each slice_I row (when I is not
 * a multiple of 4) is zero-padded so vload4 across input channels never
 * reads stale memory.
 */
template <typename T>
void pack_oihw_to_weight_pack(const T *src, T *dst, int O, int I, int H,
                              int W) {
  std::fill_n(dst, num_elements(O, I, H, W), T{});
  for (int o = 0; o < O; ++o) {
    for (int i = 0; i < I; ++i) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          const std::size_t src_idx =
            ((static_cast<std::size_t>(o) * I + i) * H + h) * W + w;
          dst[offset(o, i, h, w, O, I, H, W)] = src[src_idx];
        }
      }
    }
  }
}

/**
 * Inverse of pack_oihw_to_weight_pack. Destination is sized O*I*H*W.
 */
template <typename T>
void unpack_weight_pack_to_oihw(const T *src, T *dst, int O, int I, int H,
                                int W) {
  for (int o = 0; o < O; ++o) {
    for (int i = 0; i < I; ++i) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          const std::size_t dst_idx =
            ((static_cast<std::size_t>(o) * I + i) * H + h) * W + w;
          dst[dst_idx] = src[offset(o, i, h, w, O, I, H, W)];
        }
      }
    }
  }
}

} // namespace nntrainer::weight_pack

#endif /* __NNTRAINER_WEIGHT_PACK_LAYOUT_H__ */

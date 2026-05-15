// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    phwc4_int8_quant_layout.h
 * @date    15 May 2026
 * @brief   Host helpers for the per-token symmetric INT8 activation layout
 *          (B5 of the ML Drift parity work, arXiv:2505.00232 §4.2). Pairs
 *          with cl_kernels/activation_quant_int8_per_token.cl — quantization
 *          formula and PHWC4 offset must stay in lockstep.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * Storage:
 *   * int8 PHWC4 buffer — same offset formula as the B0 fp32 PHWC4 buffer
 *     (one byte per channel instead of one float). Channels beyond C in
 *     the last slice are quantized but undefined; readers must mask them.
 *   * per-token scale buffer — fp32, length B*H*W, indexed by
 *     (h*W + w)*B + b matching the kernel's work-item id decomposition.
 *
 * Quantization (deterministic, matches the kernel bit-for-bit):
 *   scale[b,h,w] = max_c |X| / 127   (1.0f if all zero)
 *   q[b,c,h,w]   = clamp( rint(X / scale),  -127, +127 )
 *
 * For the in-range case |X| <= 127*scale we never hit the saturating -128
 * value — the dequant therefore stays symmetric around 0.
 */
#ifndef __NNTRAINER_PHWC4_INT8_QUANT_LAYOUT_H__
#define __NNTRAINER_PHWC4_INT8_QUANT_LAYOUT_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "phwc4_layout.h"

namespace nntrainer::phwc4_int8 {

inline std::size_t num_bytes(int B, int C, int H, int W) {
  // Same element count as the fp32 PHWC4 buffer; each element is 1 byte.
  return nntrainer::phwc4::num_elements(B, C, H, W);
}

inline std::size_t num_scales(int B, int H, int W) {
  return static_cast<std::size_t>(B) * H * W;
}

inline std::size_t scale_index(int b, int h, int w, int B, int W) {
  return ((static_cast<std::size_t>(h) * W) + w) * B + b;
}

/**
 * Per-token symmetric quantize. src is fp32 PHWC4 (use the B0 packer first if
 * the caller has NCHW). dst is int8 PHWC4 sized num_bytes(); scale is fp32 of
 * length num_scales(). Padded channels (c >= C in the last slice) are written
 * as zero — same as the GPU kernel's behaviour, since the fp32 buffer was
 * zero-padded by pack_nchw_to_phwc4.
 */
inline void quantize_int8_per_token(const float *src_phwc4,
                                    std::int8_t *dst_phwc4, float *dst_scale,
                                    int B, int C, int H, int W) {
  const int slice_c = (C + 3) / 4;
  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      for (int b = 0; b < B; ++b) {
        // amax over slices.
        float amax = 0.0f;
        for (int s = 0; s < slice_c; ++s) {
          const std::size_t off_f4 =
            ((static_cast<std::size_t>(s) * H + h) * W + w) * B + b;
          const float *p = src_phwc4 + off_f4 * 4;
          for (int k = 0; k < 4; ++k) {
            const float v = std::fabs(p[k]);
            if (v > amax)
              amax = v;
          }
        }
        const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        const float inv = 1.0f / scale;
        for (int s = 0; s < slice_c; ++s) {
          const std::size_t off_f4 =
            ((static_cast<std::size_t>(s) * H + h) * W + w) * B + b;
          const float *p = src_phwc4 + off_f4 * 4;
          std::int8_t *q = dst_phwc4 + off_f4 * 4;
          for (int k = 0; k < 4; ++k) {
            float xv = p[k] * inv;
            if (xv > 127.0f)
              xv = 127.0f;
            if (xv < -127.0f)
              xv = -127.0f;
            const long qi = std::lrintf(xv);
            q[k] = static_cast<std::int8_t>(qi);
          }
        }
        dst_scale[scale_index(b, h, w, B, W)] = scale;
      }
    }
  }
}

/**
 * Dequantize back to fp32 PHWC4. Inverse of quantize_int8_per_token within
 * the int8 quantization grid. The padded channels stay 0 (q=0 * any scale).
 */
inline void dequantize_int8_per_token(const std::int8_t *src_phwc4,
                                      const float *src_scale, float *dst_phwc4,
                                      int B, int C, int H, int W) {
  const int slice_c = (C + 3) / 4;
  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      for (int b = 0; b < B; ++b) {
        const float s = src_scale[scale_index(b, h, w, B, W)];
        for (int sl = 0; sl < slice_c; ++sl) {
          const std::size_t off_f4 =
            ((static_cast<std::size_t>(sl) * H + h) * W + w) * B + b;
          const std::int8_t *q = src_phwc4 + off_f4 * 4;
          float *p = dst_phwc4 + off_f4 * 4;
          for (int k = 0; k < 4; ++k) {
            p[k] = static_cast<float>(q[k]) * s;
          }
        }
      }
    }
  }
}

} // namespace nntrainer::phwc4_int8
#endif /* __NNTRAINER_PHWC4_INT8_QUANT_LAYOUT_H__ */

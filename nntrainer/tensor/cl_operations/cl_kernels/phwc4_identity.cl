// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
//
// PHWC4 storage layout — B0 of GPU stack ML Drift parity (arXiv:2505.00232).
//
// Logical layer interface stays NCHW; only GPU physical storage is reorganised
// into PHWC4 (4-channel slice tile, paper §3.1). This file pins the offset
// formula in macros so every future GPU kernel can address logical (b, c, h, w)
// without knowing the slice math.
//
// Physical layout — slice = ceil(C/4), c4 = c & 3:
//   buffer is shaped [slice][h][w][b][c4]
//   offset(b,c,h,w) = (((slice * H + h) * W + w) * B + b) * 4 + c4
//
// Each (h, w) position therefore holds a contiguous 4-element chunk along c,
// which is the alignment vload4 / image2d sampling expects.
#define PHWC4_OFFSET(b, c, h, w, B, C, H, W)                                   \
  ((((((c) >> 2) * (H) + (h)) * (W) + (w)) * (B) + (b)) * 4 + ((c) & 3))

#define READ_ACT_F32(buf, b, c, h, w, B, C, H, W)                              \
  ((buf)[PHWC4_OFFSET((b), (c), (h), (w), (B), (C), (H), (W))])

// Identity round-trip: read a PHWC4-packed source via logical coordinates and
// write the same value to an NCHW-ordered destination. The unit test on the
// host fills NCHW with a known pattern, packs to PHWC4, runs this kernel, and
// expects the output to match the original NCHW buffer bit-for-bit. This is
// the correctness gate for the PHWC4_OFFSET formula before any real compute
// kernel is layered on top.
__kernel void phwc4_identity_f32(__global const float *src_phwc4,
                                 __global float *dst_nchw, const int B,
                                 const int C, const int H, const int W) {
  const int idx = get_global_id(0);
  const int hw = H * W;
  const int chw = C * hw;
  if (idx >= B * chw)
    return;
  const int b = idx / chw;
  const int rem = idx - b * chw;
  const int c = rem / hw;
  const int rem2 = rem - c * hw;
  const int h = rem2 / W;
  const int w = rem2 - h * W;
  dst_nchw[idx] = READ_ACT_F32(src_phwc4, b, c, h, w, B, C, H, W);
}

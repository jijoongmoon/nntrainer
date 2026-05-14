// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
//
// B1 of GPU stack ML Drift parity (arXiv:2505.00232 §3.1) — weight repack.
//
// Weight tensor logically OIHW. Physical layout pairs slice_O with the PHWC4
// activation (B0) so the convolution-as-matmul kernel (B4) can vload4 across
// input channels naturally:
//
//   buffer shape : [slice_O][O4][H][W][slice_I][I4]
//   slice_O = ceil(O/4), O4 = o & 3
//   slice_I = ceil(I/4), I4 = i & 3
//
// Offset formula (logical (o, i, h, w) -> 1D physical index):
//   ((((slice_O*4 + O4)*H + h)*W + w)*slice_I + slice_I_)*4 + I4
//
// For 1x1 conv this collapses to plain row-major with I padded to 4:
//   offset(o, i, 0, 0) = o * (slice_I * 4) + i
//
// Round-trip identity kernel for B1 — read a weight-packed buffer back into
// OIHW order via the macro. The unit test fills OIHW with a known pattern,
// packs to weight-pack on the host, runs this kernel, and expects the output
// to match the original OIHW bit-for-bit.

#define WEIGHT_PACK_SLICE_I(I) (((I) + 3) >> 2)

#define WEIGHT_PACK_OFFSET(o, i, h, w, O, I, H, W)                             \
  (((((((o) >> 2) * 4 + ((o) & 3)) * (H) + (h)) * (W) + (w)) *                 \
      WEIGHT_PACK_SLICE_I(I) +                                                 \
    ((i) >> 2)) *                                                              \
     4 +                                                                       \
   ((i) & 3))

#define READ_WEIGHT_F32(buf, o, i, h, w, O, I, H, W)                           \
  ((buf)[WEIGHT_PACK_OFFSET((o), (i), (h), (w), (O), (I), (H), (W))])

__kernel void weight_pack_identity_f32(__global const float *src_pack,
                                       __global float *dst_oihw, const int O,
                                       const int I, const int H, const int W) {
  const int idx = get_global_id(0);
  const int hw = H * W;
  const int ihw = I * hw;
  if (idx >= O * ihw)
    return;
  const int o = idx / ihw;
  const int rem = idx - o * ihw;
  const int i = rem / hw;
  const int rem2 = rem - i * hw;
  const int h = rem2 / W;
  const int w = rem2 - h * W;
  dst_oihw[idx] = READ_WEIGHT_F32(src_pack, o, i, h, w, O, I, H, W);
}


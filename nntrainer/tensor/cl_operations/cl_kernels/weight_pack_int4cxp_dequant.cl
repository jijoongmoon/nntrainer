// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
//
// B1-int4 of GPU stack ML Drift parity (arXiv:2505.00232 §4.2). Per-channel
// signed-int4 weight layout mirroring the B1 fp32 weight_pack design.
//
// Storage:
//   * Packed bytes — O * H * W * slice_I * 2 bytes (slice_I = ceil(I/4)).
//     Layout matches B1 fp32 logically: byte_offset(o, i, h, w) =
//     ((o*H + h)*W + w) * slice_I * 2 + (i >> 1)
//     The low nibble holds i with (i & 1) == 0; the high nibble holds the
//     next i. Each nibble stores  uint4 = (signed_int4 + 8)  so the bit
//     pattern is 0..15.
//   * Scale buffer — O fp32 values (one per output channel; H/W shared).
//
// Dequant:  fp = (uint4_value - 8) * scale[o]
//
// This file is the correctness gate (B0/B1 pattern): a deterministic dequant
// kernel that emits OIHW fp32 from (packed, scale). The unit test on the
// host computes the same dequant in C++ and compares the two outputs bit-
// for-bit — isolating layout/dequant correctness from any quantization noise
// in the original quantizer.

// Byte offset of the *byte* containing the nibble for logical (o, i, h, w).
// Note: this is a byte address into a __global uchar buffer.
#define INT4CXP_BYTE_OFFSET(o, i, h, w, H, W, SLICE_I)                         \
  ((((o) * (H) + (h)) * (W) + (w)) * (SLICE_I) * 2 + ((i) >> 1))

// Extract the int4 nibble for input channel i from the byte. Returns the raw
// uint4 (0..15) — caller subtracts 8 to recover the signed int4 value.
#define INT4CXP_NIBBLE(byte_val, i) (((byte_val) >> (((i) & 1) * 4)) & 0xF)

#define DEQUANT_INT4CXP(byte_val, i, scale_o)                                  \
  ((((float)INT4CXP_NIBBLE((byte_val), (i))) - 8.0f) * (scale_o))

__kernel void weight_pack_int4cxp_dequant_f32(
  __global const uchar *packed, __global const float *scale,
  __global float *dst_oihw, const int O, const int I, const int H,
  const int W) {
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
  const int slice_i = (I + 3) >> 2;
  const int byte_off = INT4CXP_BYTE_OFFSET(o, i, h, w, H, W, slice_i);
  const uchar b = packed[byte_off];
  dst_oihw[idx] = DEQUANT_INT4CXP(b, i, scale[o]);
}


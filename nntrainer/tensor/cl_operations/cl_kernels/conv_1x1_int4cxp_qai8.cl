// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
//
// B4-int4-int8 of GPU stack ML Drift parity (arXiv:2505.00232 §4.2 — the
// "embed / FFN" matmul path). int4 weight x int8 activation, both with
// per-channel / per-token symmetric scales. The inner accumulator is plain
// 32-bit int; final fp32 scale-multiply happens once per output element.
//
//   Y[b][o][h][w] = Sx[b,h,w] * Sw[o]
//                   * sum_i  int_dot( X[b,i,h,w], (uint4(W[o,i]) - 8) )
//
// Storage:
//   X  : int8 PHWC4 (B5 quantizer output)
//   Sx : fp32 per-token scale, length B*H*W
//   Wq : int4cxp packed bytes (B1-int4)
//   Sw : fp32 per-channel scale, length slice_O*4 (padded)
//   Y  : fp32 PHWC4 (consumer can downstream-quantize via B5 again)
//
// Device requirement: cl_khr_integer_dot_product (verified on Intel Arc
// Meteor Lake-P, OpenCL 3.0 2.0.0, and on Adreno 830, driver 0800.40.1).
// The host build-time probe is the gate; this kernel will refuse to compile
// on devices that lack the extension. Vendor-specific fast paths
// (cl_qcom_dot_product8 / cl_arm_matrix_multiply) come later via plan §C1.
//
// Per work-item: 4 output channels at one (b, h, w).
//   - vload4 of 4 int8 input channels (X is char-typed -> char4)
//   - For each o4 (4 outputs in this slice_O):
//       2 bytes from Wq -> 4 int4 nibbles -> char4 of signed int4 in [-8,7]
//       int dot_acc = dot(x_char4, w_char4)        [KHR-accelerated]
//   - After inner loop: convert int accumulator to float, multiply by
//     Sx * Sw, vstore4 the resulting float4 into Y in PHWC4.

#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

#define PHWC4_OFFSET_F4(b, slice_c, h, w, B, H, W)                             \
  ((((slice_c) * (H) + (h)) * (W) + (w)) * (B) + (b))

// Expand 2 packed-int4 bytes into a char4 of signed int4 values in [-8, +7].
//   byte0 holds inputs i, i+1   (low nibble = i, high nibble = i+1)
//   byte1 holds inputs i+2, i+3
#define UNPACK_INT4_PAIR(byte0, byte1)                                         \
  (char4)((char)((byte0) & 0xF) - (char)8,                                     \
          (char)(((byte0) >> 4) & 0xF) - (char)8,                              \
          (char)((byte1) & 0xF) - (char)8,                                     \
          (char)(((byte1) >> 4) & 0xF) - (char)8)

__kernel void conv_1x1_int4cxp_qai8(
  __global const char *X, __global const float *Sx,
  __global const uchar *Wq, __global const float *Sw, __global float *Y,
  const int B, const int C_in, const int C_out, const int H, const int W_dim) {
  const int slice_i_total = (C_in + 3) >> 2;
  const int slice_o_total = (C_out + 3) >> 2;
  const int hw = H * W_dim;
  const int bhw = B * hw;
  const int gid = get_global_id(0);
  if (gid >= slice_o_total * bhw)
    return;

  const int slice_o = gid / bhw;
  const int rem = gid - slice_o * bhw;
  const int h_idx = rem / (W_dim * B);
  const int rem2 = rem - h_idx * (W_dim * B);
  const int w_idx = rem2 / B;
  const int b_idx = rem2 - w_idx * B;

  const int o_base = slice_o * 4;
  const int row_bytes = slice_i_total * 2; // bytes per (o, h_w=0, w_w=0) row

  // Pre-load 4 per-channel weight scales and the single per-token act scale.
  // Sw and Sx were padded by their respective host packers so out-of-range
  // o indices read zero.
  const float ws0 = Sw[o_base + 0];
  const float ws1 = Sw[o_base + 1];
  const float ws2 = Sw[o_base + 2];
  const float ws3 = Sw[o_base + 3];
  const float xs = Sx[(h_idx * W_dim + w_idx) * B + b_idx];

  int4 iacc = (int4)(0, 0, 0, 0);
  for (int slice_i = 0; slice_i < slice_i_total; ++slice_i) {
    const int x_off_f4 =
      PHWC4_OFFSET_F4(b_idx, slice_i, h_idx, w_idx, B, H, W_dim);
    const char4 x4 = vload4(x_off_f4, X);

    // Read 2 bytes per output row at this slice_i. Each row holds the same
    // slice_i for one output channel; different o4 are slice_i_total*2 apart.
    const int byte_base = o_base * row_bytes + slice_i * 2;
    const uchar2 b0 = vload2(0, Wq + byte_base + 0 * row_bytes);
    const uchar2 b1 = vload2(0, Wq + byte_base + 1 * row_bytes);
    const uchar2 b2 = vload2(0, Wq + byte_base + 2 * row_bytes);
    const uchar2 b3 = vload2(0, Wq + byte_base + 3 * row_bytes);

    const char4 w0 = UNPACK_INT4_PAIR(b0.s0, b0.s1);
    const char4 w1 = UNPACK_INT4_PAIR(b1.s0, b1.s1);
    const char4 w2 = UNPACK_INT4_PAIR(b2.s0, b2.s1);
    const char4 w3 = UNPACK_INT4_PAIR(b3.s0, b3.s1);

    // KHR integer dot product: 4-way char*char -> int, summed.
    iacc.s0 += dot(x4, w0);
    iacc.s1 += dot(x4, w1);
    iacc.s2 += dot(x4, w2);
    iacc.s3 += dot(x4, w3);
  }

  const float4 scales = (float4)(ws0, ws1, ws2, ws3) * (float4)(xs);
  const float4 out = convert_float4(iacc) * scales;

  const int y_off_f4 =
    PHWC4_OFFSET_F4(b_idx, slice_o, h_idx, w_idx, B, H, W_dim);
  vstore4(out, y_off_f4, Y);
}


// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
//
// B4-int4 of GPU stack ML Drift parity (arXiv:2505.00232 §3.7/§3.8 + §4.2).
//
// 1x1 conv-as-matmul with per-channel int4 weight and fp32 activation:
//   Y[b][o][h][w] = sum_i  X[b][i][h][w] * dequant(W[o][i])
//   dequant(W[o][i]) = ((uint4_value(o,i) - 8) * scale[o])
//
// Activation X / output Y use B0 PHWC4 fp32; weight uses B1-int4 packed
// bytes + per-channel fp32 scale. Each work-item produces 4 output channels
// (one slice_O group) at one (b, h, w):
//   - vload4 of 4 input channels (fp32, contiguous in PHWC4)
//   - 4 separate 2-byte reads give weight nibbles for the 4 outputs at
//     this slice_I step (slice_O is the outermost dim so different o4 are
//     slice_I*2 bytes apart — not contiguous, hence 4 reads)
//   - each 2-byte read expands to a float4 via dequant; 4 dot() into
//     the accumulator
//   - vstore4 writes the slice_O group to the output PHWC4 buffer
//
// Layout macros are duplicated inline (each .cl is a separate program).
// Keep in lockstep with B0 / B1 / B1-int4 canonical definitions.

#define PHWC4_OFFSET_F4(b, slice_c, h, w, B, H, W)                             \
  ((((slice_c) * (H) + (h)) * (W) + (w)) * (B) + (b))

// Dequant one i4 group (4 input channels) into a float4. byte0 holds inputs
// (i, i+1), byte1 holds (i+2, i+3). All four share the same scale_o.
#define DEQUANT_INT4_GROUP(byte0, byte1, scale_o)                              \
  ((float4)(((float)((byte0) & 0xF) - 8.0f) * (scale_o),                       \
            ((float)((byte0) >> 4) - 8.0f) * (scale_o),                        \
            ((float)((byte1) & 0xF) - 8.0f) * (scale_o),                       \
            ((float)((byte1) >> 4) - 8.0f) * (scale_o)))

__kernel void conv_1x1_int4cxp_fp32(
  __global const float *X, __global const uchar *Wq, __global const float *Ws,
  __global float *Y, const int B, const int C_in, const int C_out, const int H,
  const int W_dim) {
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
  // Stride between adjacent output rows in the packed byte buffer.
  // (((o*H_w + h_w)*W_w + w_w)*slice_I*2 + (i>>1) for 1x1 conv  collapses to
  //  o * slice_I * 2.  H_w = W_w = 1.)
  const int row_bytes = slice_i_total * 2;

  // Pre-load 4 per-channel scales for this slice_O group.
  const float s0 = Ws[o_base + 0];
  const float s1 = (o_base + 1 < C_out) ? Ws[o_base + 1] : 0.0f;
  const float s2 = (o_base + 2 < C_out) ? Ws[o_base + 2] : 0.0f;
  const float s3 = (o_base + 3 < C_out) ? Ws[o_base + 3] : 0.0f;

  float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  for (int slice_i = 0; slice_i < slice_i_total; ++slice_i) {
    // 4 input channels in one float4 vload.
    const int x_off_f4 =
      PHWC4_OFFSET_F4(b_idx, slice_i, h_idx, w_idx, B, H, W_dim);
    const float4 x4 = vload4(x_off_f4, X);

    // For each of the 4 output channels in this slice_O group, read 2 bytes
    // (one i4 group), dequant to float4, MAC into the accumulator slot.
    const int byte_base = o_base * row_bytes + slice_i * 2;
    const uchar b0a = Wq[byte_base + 0 * row_bytes + 0];
    const uchar b0b = Wq[byte_base + 0 * row_bytes + 1];
    const uchar b1a = Wq[byte_base + 1 * row_bytes + 0];
    const uchar b1b = Wq[byte_base + 1 * row_bytes + 1];
    const uchar b2a = Wq[byte_base + 2 * row_bytes + 0];
    const uchar b2b = Wq[byte_base + 2 * row_bytes + 1];
    const uchar b3a = Wq[byte_base + 3 * row_bytes + 0];
    const uchar b3b = Wq[byte_base + 3 * row_bytes + 1];

    const float4 w0 = DEQUANT_INT4_GROUP(b0a, b0b, s0);
    const float4 w1 = DEQUANT_INT4_GROUP(b1a, b1b, s1);
    const float4 w2 = DEQUANT_INT4_GROUP(b2a, b2b, s2);
    const float4 w3 = DEQUANT_INT4_GROUP(b3a, b3b, s3);

    acc.s0 += dot(x4, w0);
    acc.s1 += dot(x4, w1);
    acc.s2 += dot(x4, w2);
    acc.s3 += dot(x4, w3);
  }

  const int y_off_f4 =
    PHWC4_OFFSET_F4(b_idx, slice_o, h_idx, w_idx, B, H, W_dim);
  vstore4(acc, y_off_f4, Y);
}


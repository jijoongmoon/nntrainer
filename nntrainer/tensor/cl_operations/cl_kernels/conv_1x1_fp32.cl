// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
//
// B4 of GPU stack ML Drift parity (arXiv:2505.00232 §3.7/§3.8) — first real
// compute kernel on the new PHWC4 + weight_pack layout pair.
//
// 1x1 conv-as-matmul:
//   Y[b][o][h][w] = sum_i  X[b][i][h][w] * W[o][i][1][1]
//
// Activation X and output Y use the B0 PHWC4 layout (innermost C4); weight W
// uses the B1 weight_pack layout (innermost I4). Each work-item produces 4
// output channels (one slice_O group) at one (b, h, w):
//   - vload4 over input C4 gives 4 input channels at once
//   - 4 separate vload4s give weights for the 4 output channels
//   - 4 dot() calls produce the 4 accumulator entries
//   - vstore4 writes the slice_O group to the output PHWC4 buffer
//
// Layout macros are duplicated inline (each .cl is a separate OpenCL program;
// see B0/B1 for the canonical definitions). Keep them in lockstep.

#define PHWC4_OFFSET_F4(b, slice_c, h, w, B, H, W)                             \
  ((((slice_c) * (H) + (h)) * (W) + (w)) * (B) + (b))

#define WEIGHT_PACK_OFFSET_F4(o, slice_i, h, w, H, W, SLICE_I)                 \
  ((((o) * (H) + (h)) * (W) + (w)) * (SLICE_I) + (slice_i))

// Compute Y = W * X under the layout pair above.
//   X  : __global PHWC4 (B0)  shape logical (B, C_in,  H, W)
//   W  : __global weight_pack (B1) shape logical (C_out, C_in, 1, 1)
//   Y  : __global PHWC4 (B0)  shape logical (B, C_out, H, W)
// Work-item id selects (slice_o, b, h, w); the kernel produces a float4 of
// output channels [slice_o*4 .. slice_o*4+3].
__kernel void conv_1x1_fp32(__global const float *X, __global const float *W,
                            __global float *Y, const int B, const int C_in,
                            const int C_out, const int H, const int W_dim) {
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
  // Weight spatial dims are (1, 1) for 1x1 conv, so the float4 stride between
  // adjacent output channels is just slice_i_total. Do NOT use the activation
  // H/W here — those are the input/output spatial size, not the weight's.
  const int weight_o_stride = slice_i_total;

  float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  for (int slice_i = 0; slice_i < slice_i_total; ++slice_i) {
    // X[b][slice_i*4..slice_i*4+3][h][w] — 4 input channels in one vload4.
    const int x_off_f4 =
      PHWC4_OFFSET_F4(b_idx, slice_i, h_idx, w_idx, B, H, W_dim);
    const float4 x4 = vload4(x_off_f4, X);

    // W[o_base+o4][slice_i*4..slice_i*4+3][0][0] for o4 = 0..3.
    // For 1x1 conv, h=w=0 in the weight; we still pass them through the macro.
    const int w_off_f4 = WEIGHT_PACK_OFFSET_F4(o_base + 0, slice_i, 0, 0, 1, 1,
                                               slice_i_total);
    const float4 w0 = vload4(w_off_f4 + 0 * weight_o_stride, W);
    const float4 w1 = vload4(w_off_f4 + 1 * weight_o_stride, W);
    const float4 w2 = vload4(w_off_f4 + 2 * weight_o_stride, W);
    const float4 w3 = vload4(w_off_f4 + 3 * weight_o_stride, W);

    acc.s0 += dot(x4, w0);
    acc.s1 += dot(x4, w1);
    acc.s2 += dot(x4, w2);
    acc.s3 += dot(x4, w3);
  }

  const int y_off_f4 = PHWC4_OFFSET_F4(b_idx, slice_o, h_idx, w_idx, B, H, W_dim);
  vstore4(acc, y_off_f4, Y);
}


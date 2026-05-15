// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
//
// Tier S2 of GPU stack ML Drift parity. Tile widening on top of B2:
// each work-item now produces 4 outputs (one slice_O group) x 4 tokens
// (W positions) = 16 outputs total, instead of 4 outputs x 1 token in
// the B2 baseline. Reuses each weight image2d texel across 4 tokens
// without re-reading, which lifts arithmetic intensity per memory
// transaction and was the main bottleneck the B2 baseline left on the
// table (Adreno 830 plateaued at ~655 GOps/s ~= 19% of peak int8).
//
// Constraints:
//   * W (sequence / spatial) must be a multiple of 4. Prefill / FFN
//     shapes are naturally aligned. Decode (W=1) uses the non-tiled
//     B2 kernel — covered by plan §B4 stage-aware kernel split.
//   * Weight image2d, sampler-less read_imageui — same as B2.
//   * Inner accumulator: 4 int4 vectors (one per token), each holding
//     4 output channels.

#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define PHWC4_OFFSET_F4(b, slice_c, h, w, B, H, W)                             \
  ((((slice_c) * (H) + (h)) * (W) + (w)) * (B) + (b))

#define UNPACK_INT4_TEXEL(v)                                                   \
  (char4)((char)((v) & 0xF) - (char)8,                                         \
          (char)(((v) >> 4) & 0xF) - (char)8,                                  \
          (char)(((v) >> 8) & 0xF) - (char)8,                                  \
          (char)(((v) >> 12) & 0xF) - (char)8)

__kernel void conv_1x1_int4cxp_qai8_img2d_w_tile_w4_fp16o(
  __global const char *X, __global const float *Sx,
  __read_only image2d_t Wq_img, __global const float *Sw,
  __global half *Y, const int B, const int C_in, const int C_out, const int H,
  const int W_dim) {
  const int slice_i_total = (C_in + 3) >> 2;
  const int slice_o_total = (C_out + 3) >> 2;
  // Caller must pass W_dim divisible by 4. The host-side launcher rounds up
  // and any tail tokens beyond W_dim are masked on store below.
  const int w_tiles = W_dim >> 2;
  const int htile = H * w_tiles;
  const int bhwtile = B * htile;
  const int gid = get_global_id(0);
  if (gid >= slice_o_total * bhwtile)
    return;

  const int slice_o = gid / bhwtile;
  const int rem = gid - slice_o * bhwtile;
  const int h_idx = rem / (w_tiles * B);
  const int rem2 = rem - h_idx * (w_tiles * B);
  const int w_tile = rem2 / B;
  const int b_idx = rem2 - w_tile * B;
  const int w_base = w_tile * 4;

  const int o_base = slice_o * 4;

  // Per-output weight scales (same for all 4 tokens).
  const float4 ws =
    (float4)(Sw[o_base + 0], Sw[o_base + 1], Sw[o_base + 2], Sw[o_base + 3]);

  // Per-token activation scales.
  const float xs0 = Sx[(h_idx * W_dim + (w_base + 0)) * B + b_idx];
  const float xs1 = Sx[(h_idx * W_dim + (w_base + 1)) * B + b_idx];
  const float xs2 = Sx[(h_idx * W_dim + (w_base + 2)) * B + b_idx];
  const float xs3 = Sx[(h_idx * W_dim + (w_base + 3)) * B + b_idx];

  // 4 accumulators — one int4 per token, holding the 4 output channels.
  int4 acc_t0 = (int4)(0, 0, 0, 0);
  int4 acc_t1 = (int4)(0, 0, 0, 0);
  int4 acc_t2 = (int4)(0, 0, 0, 0);
  int4 acc_t3 = (int4)(0, 0, 0, 0);

  const int slice_i_pairs = (slice_i_total + 1) >> 1;
  for (int pair = 0; pair < slice_i_pairs; ++pair) {
    // 4 weight texels, shared across all 4 tokens.
    const uint4 t0 = read_imageui(Wq_img, (int2)(pair, o_base + 0));
    const uint4 t1 = read_imageui(Wq_img, (int2)(pair, o_base + 1));
    const uint4 t2 = read_imageui(Wq_img, (int2)(pair, o_base + 2));
    const uint4 t3 = read_imageui(Wq_img, (int2)(pair, o_base + 3));

    // ----- slice_i = 2*pair (lower bytes of each texel) -----
    {
      const int slice_i = pair * 2;
      const int off_t0 =
        PHWC4_OFFSET_F4(b_idx, slice_i, h_idx, w_base + 0, B, H, W_dim);
      const int off_t1 =
        PHWC4_OFFSET_F4(b_idx, slice_i, h_idx, w_base + 1, B, H, W_dim);
      const int off_t2 =
        PHWC4_OFFSET_F4(b_idx, slice_i, h_idx, w_base + 2, B, H, W_dim);
      const int off_t3 =
        PHWC4_OFFSET_F4(b_idx, slice_i, h_idx, w_base + 3, B, H, W_dim);
      const char4 x_t0 = vload4(off_t0, X);
      const char4 x_t1 = vload4(off_t1, X);
      const char4 x_t2 = vload4(off_t2, X);
      const char4 x_t3 = vload4(off_t3, X);

      const char4 w0 = UNPACK_INT4_TEXEL(t0.x | (t0.y << 8));
      const char4 w1 = UNPACK_INT4_TEXEL(t1.x | (t1.y << 8));
      const char4 w2 = UNPACK_INT4_TEXEL(t2.x | (t2.y << 8));
      const char4 w3 = UNPACK_INT4_TEXEL(t3.x | (t3.y << 8));

      acc_t0.s0 += dot(x_t0, w0); acc_t0.s1 += dot(x_t0, w1);
      acc_t0.s2 += dot(x_t0, w2); acc_t0.s3 += dot(x_t0, w3);
      acc_t1.s0 += dot(x_t1, w0); acc_t1.s1 += dot(x_t1, w1);
      acc_t1.s2 += dot(x_t1, w2); acc_t1.s3 += dot(x_t1, w3);
      acc_t2.s0 += dot(x_t2, w0); acc_t2.s1 += dot(x_t2, w1);
      acc_t2.s2 += dot(x_t2, w2); acc_t2.s3 += dot(x_t2, w3);
      acc_t3.s0 += dot(x_t3, w0); acc_t3.s1 += dot(x_t3, w1);
      acc_t3.s2 += dot(x_t3, w2); acc_t3.s3 += dot(x_t3, w3);
    }

    // ----- slice_i = 2*pair + 1 (upper bytes) -----
    const int slice_i_hi = pair * 2 + 1;
    if (slice_i_hi < slice_i_total) {
      const int off_t0 =
        PHWC4_OFFSET_F4(b_idx, slice_i_hi, h_idx, w_base + 0, B, H, W_dim);
      const int off_t1 =
        PHWC4_OFFSET_F4(b_idx, slice_i_hi, h_idx, w_base + 1, B, H, W_dim);
      const int off_t2 =
        PHWC4_OFFSET_F4(b_idx, slice_i_hi, h_idx, w_base + 2, B, H, W_dim);
      const int off_t3 =
        PHWC4_OFFSET_F4(b_idx, slice_i_hi, h_idx, w_base + 3, B, H, W_dim);
      const char4 x_t0 = vload4(off_t0, X);
      const char4 x_t1 = vload4(off_t1, X);
      const char4 x_t2 = vload4(off_t2, X);
      const char4 x_t3 = vload4(off_t3, X);

      const char4 w0 = UNPACK_INT4_TEXEL(t0.z | (t0.w << 8));
      const char4 w1 = UNPACK_INT4_TEXEL(t1.z | (t1.w << 8));
      const char4 w2 = UNPACK_INT4_TEXEL(t2.z | (t2.w << 8));
      const char4 w3 = UNPACK_INT4_TEXEL(t3.z | (t3.w << 8));

      acc_t0.s0 += dot(x_t0, w0); acc_t0.s1 += dot(x_t0, w1);
      acc_t0.s2 += dot(x_t0, w2); acc_t0.s3 += dot(x_t0, w3);
      acc_t1.s0 += dot(x_t1, w0); acc_t1.s1 += dot(x_t1, w1);
      acc_t1.s2 += dot(x_t1, w2); acc_t1.s3 += dot(x_t1, w3);
      acc_t2.s0 += dot(x_t2, w0); acc_t2.s1 += dot(x_t2, w1);
      acc_t2.s2 += dot(x_t2, w2); acc_t2.s3 += dot(x_t2, w3);
      acc_t3.s0 += dot(x_t3, w0); acc_t3.s1 += dot(x_t3, w1);
      acc_t3.s2 += dot(x_t3, w2); acc_t3.s3 += dot(x_t3, w3);
    }
  }

  // Final scale + store. 4 separate vstore4 (one per token).
  const float4 out_t0 = convert_float4(acc_t0) * (ws * (float4)(xs0));
  const float4 out_t1 = convert_float4(acc_t1) * (ws * (float4)(xs1));
  const float4 out_t2 = convert_float4(acc_t2) * (ws * (float4)(xs2));
  const float4 out_t3 = convert_float4(acc_t3) * (ws * (float4)(xs3));

  // Down-cast to fp16 at the final write. Halves the output bandwidth and
  // lets a downstream layer load 8-byte slice tiles in one go.
  vstore4(convert_half4(out_t0),
          PHWC4_OFFSET_F4(b_idx, slice_o, h_idx, w_base + 0, B, H, W_dim), Y);
  vstore4(convert_half4(out_t1),
          PHWC4_OFFSET_F4(b_idx, slice_o, h_idx, w_base + 1, B, H, W_dim), Y);
  vstore4(convert_half4(out_t2),
          PHWC4_OFFSET_F4(b_idx, slice_o, h_idx, w_base + 2, B, H, W_dim), Y);
  vstore4(convert_half4(out_t3),
          PHWC4_OFFSET_F4(b_idx, slice_o, h_idx, w_base + 3, B, H, W_dim), Y);
}


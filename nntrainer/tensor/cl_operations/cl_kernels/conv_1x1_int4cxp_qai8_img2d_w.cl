// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
//
// B2 of GPU stack ML Drift parity (arXiv:2505.00232 §3.3 + Figure 2). Variant
// of the B4-int4-int8 1x1 conv kernel that reads the int4 weight through an
// image2d_t (texel format CL_R + CL_UNSIGNED_INT16) instead of a plain
// __global uchar* buffer. Activation X stays as a char* buffer for this
// iteration — the experiment is "does Adreno's texture cache close the gap
// for weight reads", which the baseline measurements identified as the main
// Adreno bottleneck on this kernel.
//
// Weight image2d layout:
//   image dimensions : ((slice_I + 1) / 2, O)
//   image texel      : CL_RGBA + CL_UNSIGNED_INT8 — 1 texel = 4 bytes = 2 i4
//                      groups. The 4 byte channels map to:
//                        .x = byte0 of slice_i=2*pair   (i = 8*pair+0..+1)
//                        .y = byte1 of slice_i=2*pair   (i = 8*pair+2..+3)
//                        .z = byte0 of slice_i=2*pair+1 (i = 8*pair+4..+5)
//                        .w = byte1 of slice_i=2*pair+1 (i = 8*pair+6..+7)
//                      CL_RGBA + CL_UNSIGNED_INT8 is a mandatory minimum
//                      format in OpenCL; CL_R + UINT16 and CL_R + UINT32
//                      both produced SPIR-V "undefined ImageRead" errors on
//                      Intel Meteor Lake-P even with the sampler-less form.
//   coordinate       : read_imageui(W_img, (int2)(slice_i_pair, o))
//
// The bytes in the texel match the B1-int4 byte layout exactly: bits
//   [0..3]   = nibble for i = slice_i*4 + 0  (low nibble of byte0)
//   [4..7]   = nibble for i = slice_i*4 + 1  (high nibble of byte0)
//   [8..11]  = nibble for i = slice_i*4 + 2  (low nibble of byte1)
//   [12..15] = nibble for i = slice_i*4 + 3  (high nibble of byte1)
// so dequant follows the same (nibble - 8) * scale[o] formula.

#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

#define PHWC4_OFFSET_F4(b, slice_c, h, w, B, H, W)                             \
  ((((slice_c) * (H) + (h)) * (W) + (w)) * (B) + (b))

// Expand one 16-bit texel (4 packed nibbles) into a char4 of signed int4
// values in [-8, +7], correcting for the +8 storage offset.
#define UNPACK_INT4_TEXEL(v)                                                   \
  (char4)((char)((v) & 0xF) - (char)8,                                         \
          (char)(((v) >> 4) & 0xF) - (char)8,                                  \
          (char)(((v) >> 8) & 0xF) - (char)8,                                  \
          (char)(((v) >> 12) & 0xF) - (char)8)

__kernel void conv_1x1_int4cxp_qai8_img2d_w(
  __global const char *X, __global const float *Sx,
  __read_only image2d_t Wq_img, __global const float *Sw,
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

  const float ws0 = Sw[o_base + 0];
  const float ws1 = Sw[o_base + 1];
  const float ws2 = Sw[o_base + 2];
  const float ws3 = Sw[o_base + 3];
  const float xs = Sx[(h_idx * W_dim + w_idx) * B + b_idx];

  // We use the sampler-less read_imageui overload (no sampler argument).
  // Integer image reads on Intel Meteor Lake-P fail with a SPIR-V backend
  // error when a sampler is supplied — the sampler-less form maps to the
  // ImageRead opcode instead of ImageSampleExplicitLod and is available on
  // all OpenCL 1.2+ devices for unnormalised coords + nearest filtering,
  // which is exactly what we need here.

  // Walk slice_i in pairs since each texel covers 2 i4 groups.
  const int slice_i_pairs = (slice_i_total + 1) >> 1;

  int4 iacc = (int4)(0, 0, 0, 0);
  for (int pair = 0; pair < slice_i_pairs; ++pair) {
    // Read 4 texels at (pair, o_base + 0..3). Each is uchar4 in a uint4:
    // .x/.y are the byte pair for slice_i=2*pair, .z/.w for slice_i=2*pair+1.
    const uint4 t0 = read_imageui(Wq_img, (int2)(pair, o_base + 0));
    const uint4 t1 = read_imageui(Wq_img, (int2)(pair, o_base + 1));
    const uint4 t2 = read_imageui(Wq_img, (int2)(pair, o_base + 2));
    const uint4 t3 = read_imageui(Wq_img, (int2)(pair, o_base + 3));

    // Process slice_i = 2*pair  (lo bytes).
    {
      const int slice_i = pair * 2;
      const int x_off_f4 =
        PHWC4_OFFSET_F4(b_idx, slice_i, h_idx, w_idx, B, H, W_dim);
      const char4 x4 = vload4(x_off_f4, X);
      const uint v0 = t0.x | (t0.y << 8);
      const uint v1 = t1.x | (t1.y << 8);
      const uint v2 = t2.x | (t2.y << 8);
      const uint v3 = t3.x | (t3.y << 8);
      const char4 w0 = UNPACK_INT4_TEXEL(v0);
      const char4 w1 = UNPACK_INT4_TEXEL(v1);
      const char4 w2 = UNPACK_INT4_TEXEL(v2);
      const char4 w3 = UNPACK_INT4_TEXEL(v3);
      iacc.s0 += dot(x4, w0);
      iacc.s1 += dot(x4, w1);
      iacc.s2 += dot(x4, w2);
      iacc.s3 += dot(x4, w3);
    }
    // Process slice_i = 2*pair + 1  (hi bytes) — only if in range.
    const int slice_i_hi = pair * 2 + 1;
    if (slice_i_hi < slice_i_total) {
      const int x_off_f4 =
        PHWC4_OFFSET_F4(b_idx, slice_i_hi, h_idx, w_idx, B, H, W_dim);
      const char4 x4 = vload4(x_off_f4, X);
      const uint v0 = t0.z | (t0.w << 8);
      const uint v1 = t1.z | (t1.w << 8);
      const uint v2 = t2.z | (t2.w << 8);
      const uint v3 = t3.z | (t3.w << 8);
      const char4 w0 = UNPACK_INT4_TEXEL(v0);
      const char4 w1 = UNPACK_INT4_TEXEL(v1);
      const char4 w2 = UNPACK_INT4_TEXEL(v2);
      const char4 w3 = UNPACK_INT4_TEXEL(v3);
      iacc.s0 += dot(x4, w0);
      iacc.s1 += dot(x4, w1);
      iacc.s2 += dot(x4, w2);
      iacc.s3 += dot(x4, w3);
    }
  }

  const float4 scales = (float4)(ws0, ws1, ws2, ws3) * (float4)(xs);
  const float4 out = convert_float4(iacc) * scales;

  const int y_off_f4 =
    PHWC4_OFFSET_F4(b_idx, slice_o, h_idx, w_idx, B, H, W_dim);
  vstore4(out, y_off_f4, Y);
}


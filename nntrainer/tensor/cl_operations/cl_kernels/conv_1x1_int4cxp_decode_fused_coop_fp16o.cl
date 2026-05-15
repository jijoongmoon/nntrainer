// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
//
// §3.6 op-fused decode kernel (paper §3.7 "decode integrates input
// quantization directly within the operational kernel").
//
// Combines into a single kernel:
//   1. per-token INT8 quantization of the fp32 activation X (paper §4.2)
//   2. int4-weight × int8-activation matmul (B4-int4-int8)
//   3. fp16 down-cast at the output
//
// Designed for the decode regime: B = H = W = 1 (single token). The
// 1-token shape lets every work-item own one slice_O group's full output
// row, and the input scale is shared by every work-item (whole-token
// max). Each work-item scans X redundantly to recover the scale — cheap
// at decode scale (~1k vload4) compared to the matmul work, and
// eliminates the round-trip through the INT8 X buffer + scale buffer
// that B5 + B4-int4-int8 otherwise needs.
//
// Removed vs the 2-kernel pipeline:
//   * INT8 X buffer write + read (~8 MiB for a 4K-hidden token roundtrip
//     in cache lines)
//   * scale_x buffer write + read
//   * One kernel dispatch + queue submit (~50-100 us on Adreno)
//   * One pass over X (B5 reads it once for amax; we just read once and
//     reuse the result)
//
// Inputs:
//   X     : fp32 PHWC4 activation, sized (B=1, C_in, H=1, W=1)
//   Wq    : int4cxp packed bytes as image2d (CL_RGBA + UINT8, 2 i4
//           groups per texel — same as the B2 / Tier S2 kernels)
//   Sw    : fp32 per-output-channel weight scale, length slice_O * 4
//   Y     : fp16 PHWC4 output, sized (B=1, C_out, H=1, W=1)

#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define UNPACK_INT4_TEXEL(v)                                                   \
  (char4)((char)((v) & 0xF) - (char)8,                                         \
          (char)(((v) >> 4) & 0xF) - (char)8,                                  \
          (char)(((v) >> 8) & 0xF) - (char)8,                                  \
          (char)(((v) >> 12) & 0xF) - (char)8)

__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void conv_1x1_int4cxp_decode_fused_coop_fp16o(
  __global const float *X, __read_only image2d_t Wq_img,
  __global const float *Sw, __global half *Y, const int C_in,
  const int C_out) {
  const int slice_i_total = (C_in + 3) >> 2;
  const int slice_o_total = (C_out + 3) >> 2;
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  const int wg_size = 64;
  if (gid >= slice_o_total)
    return;
  const int slice_o = gid;
  const int o_base = slice_o * 4;

  // PHWC4 with B=H=W=1 collapses to a linear array of float4.

  // ----- Cooperative amax via sub-group reduction --------------------
  // The 64 lanes in this work-group together scan X exactly once and share
  // the result via sub_group_reduce_max. The naive (non-cooperative)
  // variant had every work-item rescan X independently, which produced a
  // 1.47x regression at decode 4K hidden — see baseline_*_fused_decode.json.
  // With a WG of 64, the redundant scan factor drops from ~1024 to 16
  // (16 work-groups, each scanning once), but within a single WG no work
  // is duplicated.
  float local_amax = 0.0f;
  for (int s = lid; s < slice_i_total; s += wg_size) {
    const float4 x4 = vload4(s, X);
    local_amax = fmax(local_amax, fmax(fmax(fabs(x4.s0), fabs(x4.s1)),
                                        fmax(fabs(x4.s2), fabs(x4.s3))));
  }
  const float amax = sub_group_reduce_max(local_amax);
  const float scale_x = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
  const float inv_x = 1.0f / scale_x;

  // ----- Pass 2 : matmul with inline quantization, fp16 output ------
  const float4 ws = (float4)(Sw[o_base + 0], Sw[o_base + 1],
                             Sw[o_base + 2], Sw[o_base + 3]);

  int4 iacc = (int4)(0, 0, 0, 0);
  const int slice_i_pairs = (slice_i_total + 1) >> 1;
  for (int pair = 0; pair < slice_i_pairs; ++pair) {
    const uint4 t0 = read_imageui(Wq_img, (int2)(pair, o_base + 0));
    const uint4 t1 = read_imageui(Wq_img, (int2)(pair, o_base + 1));
    const uint4 t2 = read_imageui(Wq_img, (int2)(pair, o_base + 2));
    const uint4 t3 = read_imageui(Wq_img, (int2)(pair, o_base + 3));

    // ----- slice_i = 2*pair (low bytes) -----
    {
      const int slice_i = pair * 2;
      const float4 x_fp = vload4(slice_i, X);
      int4 q;
      q.s0 = (int)rint(clamp(x_fp.s0 * inv_x, -127.0f, 127.0f));
      q.s1 = (int)rint(clamp(x_fp.s1 * inv_x, -127.0f, 127.0f));
      q.s2 = (int)rint(clamp(x_fp.s2 * inv_x, -127.0f, 127.0f));
      q.s3 = (int)rint(clamp(x_fp.s3 * inv_x, -127.0f, 127.0f));
      const char4 qx = convert_char4(q);
      const char4 w0 = UNPACK_INT4_TEXEL(t0.x | (t0.y << 8));
      const char4 w1 = UNPACK_INT4_TEXEL(t1.x | (t1.y << 8));
      const char4 w2 = UNPACK_INT4_TEXEL(t2.x | (t2.y << 8));
      const char4 w3 = UNPACK_INT4_TEXEL(t3.x | (t3.y << 8));
      iacc.s0 += dot(qx, w0);
      iacc.s1 += dot(qx, w1);
      iacc.s2 += dot(qx, w2);
      iacc.s3 += dot(qx, w3);
    }
    // ----- slice_i = 2*pair + 1 (high bytes) -----
    const int slice_i_hi = pair * 2 + 1;
    if (slice_i_hi < slice_i_total) {
      const float4 x_fp = vload4(slice_i_hi, X);
      int4 q;
      q.s0 = (int)rint(clamp(x_fp.s0 * inv_x, -127.0f, 127.0f));
      q.s1 = (int)rint(clamp(x_fp.s1 * inv_x, -127.0f, 127.0f));
      q.s2 = (int)rint(clamp(x_fp.s2 * inv_x, -127.0f, 127.0f));
      q.s3 = (int)rint(clamp(x_fp.s3 * inv_x, -127.0f, 127.0f));
      const char4 qx = convert_char4(q);
      const char4 w0 = UNPACK_INT4_TEXEL(t0.z | (t0.w << 8));
      const char4 w1 = UNPACK_INT4_TEXEL(t1.z | (t1.w << 8));
      const char4 w2 = UNPACK_INT4_TEXEL(t2.z | (t2.w << 8));
      const char4 w3 = UNPACK_INT4_TEXEL(t3.z | (t3.w << 8));
      iacc.s0 += dot(qx, w0);
      iacc.s1 += dot(qx, w1);
      iacc.s2 += dot(qx, w2);
      iacc.s3 += dot(qx, w3);
    }
  }

  // Final scale and fp16 store. PHWC4 with B=H=W=1 means slice_o is the
  // float4 index directly.
  const float4 out_fp = convert_float4(iacc) * ws * (float4)(scale_x);
  vstore4(convert_half4(out_fp), slice_o, Y);
}


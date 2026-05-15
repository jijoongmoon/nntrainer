// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
//
// §3.6 op fusion candidate (paper "automatic fusion: residual connections
// with RMS normalization"). One kernel folds:
//
//   tmp[b,c,h,w] = X[b,c,h,w] + R[b,c,h,w]              // residual add
//   rms          = sqrt( mean_c(tmp^2) + eps )
//   Y[b,c,h,w]   = tmp[b,c,h,w] * gamma[c] / rms        // norm + scale
//
// Inputs / output: fp16 PHWC4. gamma is a flat fp16 vector of length C.
// Intermediates (sum_sq, inv_rms, scale-multiply) stay in fp32 for
// numerical precision; the final cast to half happens at the vstore4.
//
// Work-item geometry: WG of 64 lanes; one WG per (b, h, w) token. The 64
// lanes cooperatively scan slice_C and reduce sum_sq via
// sub_group_reduce_add (cl_khr_subgroups, verified on Adreno 830). The
// inv_rms is broadcast across the WG implicitly because sub_group_reduce
// returns the same scalar to every lane. Pass 2 then has each lane write
// its share of normalized output.
//
// Replaces an unfused two-kernel sequence (element-wise add, then
// RMSNorm). For decode (one token) the unfused cost is dominated by
// dispatch overhead (~50 us per launch); fusing should roughly halve
// that for this stage of the transformer block.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define PHWC4_OFFSET_F4(b, slice_c, h, w, B, H, W)                             \
  ((((slice_c) * (H) + (h)) * (W) + (w)) * (B) + (b))

__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void rmsnorm_residual_fused_fp16(
  __global const half *X, __global const half *R,
  __global const half *gamma, __global half *Y, const int B, const int C,
  const int H, const int W, const float eps) {
  const int slice_C = (C + 3) >> 2;
  const int lid = get_local_id(0);
  const int wg_size = 64;
  const int wg_id = get_group_id(0);
  const int total_tokens = B * H * W;
  if (wg_id >= total_tokens)
    return;

  // (b, h, w) decomposition. wg_id ordering matches the PHWC4 token-slot
  // ordering (h*W*B + w*B + b) so adjacent tokens land in adjacent WGs.
  const int b_idx = wg_id % B;
  const int wh = wg_id / B;
  const int w_idx = wh % W;
  const int h_idx = wh / W;

  // ----- Pass 1 : cooperative sum_sq over all C channels -----
  float partial = 0.0f;
  for (int s = lid; s < slice_C; s += wg_size) {
    const int off_f4 = PHWC4_OFFSET_F4(b_idx, s, h_idx, w_idx, B, H, W);
    const half4 x4 = vload4(off_f4, X);
    const half4 r4 = vload4(off_f4, R);
    const float4 sum4 = convert_float4(x4) + convert_float4(r4);
    partial = fma(sum4.s0, sum4.s0, partial);
    partial = fma(sum4.s1, sum4.s1, partial);
    partial = fma(sum4.s2, sum4.s2, partial);
    partial = fma(sum4.s3, sum4.s3, partial);
  }
  const float total_sq = sub_group_reduce_add(partial);
  const float inv_rms = rsqrt(total_sq / (float)C + eps);

  // ----- Pass 2 : write normalized + scaled output -----
  for (int s = lid; s < slice_C; s += wg_size) {
    const int off_f4 = PHWC4_OFFSET_F4(b_idx, s, h_idx, w_idx, B, H, W);
    const half4 x4 = vload4(off_f4, X);
    const half4 r4 = vload4(off_f4, R);
    const float4 sum4 = convert_float4(x4) + convert_float4(r4);

    // Per-channel gamma. The PHWC4 input zero-pads tail channels (c >= C)
    // so we mirror that here for gamma — values beyond C are read as 0 and
    // their contribution to the output is 0 (multiplied by zero gamma).
    const int c_base = s * 4;
    float4 g4;
    g4.s0 = (c_base + 0 < C) ? convert_float(gamma[c_base + 0]) : 0.0f;
    g4.s1 = (c_base + 1 < C) ? convert_float(gamma[c_base + 1]) : 0.0f;
    g4.s2 = (c_base + 2 < C) ? convert_float(gamma[c_base + 2]) : 0.0f;
    g4.s3 = (c_base + 3 < C) ? convert_float(gamma[c_base + 3]) : 0.0f;

    const float4 y4f = sum4 * g4 * (float4)(inv_rms);
    vstore4(convert_half4(y4f), off_f4, Y);
  }
}

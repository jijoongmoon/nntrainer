// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
//
// B5 of GPU stack ML Drift parity (arXiv:2505.00232 §4.2). Per-token symmetric
// INT8 quantization on a fp32 PHWC4 activation tensor. The pair (int8 PHWC4,
// per-token fp32 scale) feeds the INT8 activation side of the INT4-weight ×
// INT8-activation matmul the paper prescribes for embed/FFN layers.
//
// Per-token = per (b, h, w) position; one scale shared across all C channels
// at that position. For LLM with (B=1, H=1, W=seq), this is the standard
// "per-token" scaling used in qint8 / qai8 quantization.
//
//   scale[b, h, w] = max_c |X[b, c, h, w]| / 127
//   Q[b, c, h, w]  = clamp( rint(X / scale),  -127, +127 )       (stored int8)
//   dequant        = Q * scale[b, h, w]
//
// Storage:
//   Input  X : fp32 PHWC4 (B0), shape (B, C, H, W)
//   Output Q : int8 PHWC4, same byte_offset(b, c, h, w) as the scalar fp32
//              offset (each c4 element is one byte instead of four)
//   Output S : fp32 buffer of length B * H * W; index((b,h,w)) =
//              (h*W + w) * B + b
//
// One work-item per token: scan all slices to find |X|max, then a second pass
// to quantize and store. The two-pass strategy keeps register pressure low —
// one float4 in flight at a time — which is friendlier to Adreno than
// caching the whole (b,h,w) row in registers.

#define PHWC4_OFFSET_F4(b, slice_c, h, w, B, H, W)                             \
  ((((slice_c) * (H) + (h)) * (W) + (w)) * (B) + (b))

__kernel void activation_quant_int8_per_token(__global const float *X,
                                              __global char *Q,
                                              __global float *S, const int B,
                                              const int C, const int H,
                                              const int W) {
  const int slice_c = (C + 3) >> 2;
  const int hwb = H * W * B;
  const int gid = get_global_id(0);
  if (gid >= hwb)
    return;

  // (b, h, w) decomposition. Index((b,h,w)) = (h*W + w)*B + b — same order as
  // the scale buffer indexing below.
  const int h_idx = gid / (W * B);
  const int rem = gid - h_idx * (W * B);
  const int w_idx = rem / B;
  const int b_idx = rem - w_idx * B;

  // Pass 1: amax over all C (over slices). Note slice padding (slice_c*4 - C
  // tail) holds zero in PHWC4, so reading those into the fmax is harmless.
  float amax = 0.0f;
  for (int s = 0; s < slice_c; ++s) {
    const int off = PHWC4_OFFSET_F4(b_idx, s, h_idx, w_idx, B, H, W);
    const float4 x4 = vload4(off, X);
    amax = fmax(amax, fmax(fmax(fabs(x4.s0), fabs(x4.s1)),
                           fmax(fabs(x4.s2), fabs(x4.s3))));
  }

  // Edge case: all-zero token. Use scale = 1 so dequant of any stored q is 0.
  const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
  const float inv = 1.0f / scale;

  // Pass 2: quantize each slice and store. clamp before convert so the result
  // is always in [-127, +127] — symmetric, never hits the asymmetric -128.
  for (int s = 0; s < slice_c; ++s) {
    const int off = PHWC4_OFFSET_F4(b_idx, s, h_idx, w_idx, B, H, W);
    const float4 x4 = vload4(off, X);
    int4 q4;
    q4.s0 = (int)rint(clamp(x4.s0 * inv, -127.0f, 127.0f));
    q4.s1 = (int)rint(clamp(x4.s1 * inv, -127.0f, 127.0f));
    q4.s2 = (int)rint(clamp(x4.s2 * inv, -127.0f, 127.0f));
    q4.s3 = (int)rint(clamp(x4.s3 * inv, -127.0f, 127.0f));
    vstore4(convert_char4(q4), off, Q);
  }

  S[gid] = scale;
}


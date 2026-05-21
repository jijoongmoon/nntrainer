// SPDX-License-Identifier: Apache-2.0
// v8c GEMM kernel set: paper-aligned int8×int4 prefill compute (8/4/4).
//
// Weight format (in cl_mem buffer, viewed as image2d via image2d_from_buffer):
//   - Layout: row-major [N output channels][K/2 bytes per row]
//   - Encoding: int4 offset (value + 8) in low 4 bits of each byte
//   - Image2D view: CL_RGBA + CL_UNSIGNED_INT32, width=K/32, height=N,
//                    row_pitch=K/2 (16 bytes per texel = 32 int4 K-channels)
// Activation format:
//   - Layout: row-major [M rows][K bytes per row]
//   - Encoding: signed int8
//   - Image2D view: CL_RGBA + CL_UNSIGNED_INT32, width=K/16, height=M,
//                    row_pitch=K (16 bytes per texel = 16 int8 K-channels)
// Math: acc(i,j) = Σ_k act_ik × (enc_kj − 8)
//                = Σ_k dot_4x8packed_su_int(act_packed, enc_masked) − 8·row_sum_act_i
// row_sum_act_i = Σ_k act_ik, precomputed in the act-quant kernel.

#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t SMP_v8c = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE |
                               CLK_FILTER_NEAREST;

// ============================================================
// fp→int8 activation quantizer + row_sum, paper §3.7 separate kernel.
//   Input:  fp16 activations [M, K] (row-major in cl_mem buffer)
//   Outputs: int8 acts [M, K] (row-major), per-row fp32 scale, per-row int32 row_sum
// One work-item per row (M work-items total). K assumed multiple of 4.
// ============================================================
__kernel void v8c_act_quant_f16(
    __global const half *act_fp16,           // [M, K] fp16
    __global       char *act_int8,           // [M, K] int8 (row-major buffer; image2d view used by GEMM)
    __global       float *scale_per_row,     // [M] fp32
    __global       int   *row_sum_act,       // [M] int32 (== sum_k int8_value)
    const int M, const int K) {
  int i = get_global_id(0);
  if (i >= M) return;
  // pass 1: amax for scale
  float amax = 0.0f;
  for (int k = 0; k < K; k++) {
    float v = (float)act_fp16[(long)i * K + k];
    float a = fabs(v);
    if (a > amax) amax = a;
  }
  float s = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
  scale_per_row[i] = s;
  // pass 2: quantize + accumulate row_sum
  float inv_s = 1.0f / s;
  int rs = 0;
  for (int k = 0; k < K; k++) {
    int q = (int)rint((float)act_fp16[(long)i * K + k] * inv_s);
    q = clamp(q, -127, 127);
    act_int8[(long)i * K + k] = (char)q;
    rs += q;
  }
  row_sum_act[i] = rs;
}

// fp32 variant (some paths feed fp32 acts)
__kernel void v8c_act_quant_f32(
    __global const float *act_fp32,
    __global       char  *act_int8,
    __global       float *scale_per_row,
    __global       int   *row_sum_act,
    const int M, const int K) {
  int i = get_global_id(0);
  if (i >= M) return;
  float amax = 0.0f;
  for (int k = 0; k < K; k++) {
    float a = fabs(act_fp32[(long)i * K + k]);
    if (a > amax) amax = a;
  }
  float s = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
  scale_per_row[i] = s;
  float inv_s = 1.0f / s;
  int rs = 0;
  for (int k = 0; k < K; k++) {
    int q = (int)rint(act_fp32[(long)i * K + k] * inv_s);
    q = clamp(q, -127, 127);
    act_int8[(long)i * K + k] = (char)q;
    rs += q;
  }
  row_sum_act[i] = rs;
}

// ============================================================
// v8c int8 × int4(offset) GEMM (signed-unsigned packed dot product).
// Output: fp16 [M, N] (row-major)
// Canonical work-item tile: TM=4, TN=8; LWS 4×16; tile %TM/%TN must divide M/N.
// (Configurable via -DTM= -DTN= at build time if needed; defaults match the
//  best-measured config on Adreno 830: 87% of HW peak.)
// ============================================================
#ifndef V8C_TM
#define V8C_TM 4
#endif
#ifndef V8C_TN
#define V8C_TN 8
#endif

__kernel void v8c_gemm_int8_int4(
    __read_only image2d_t  Ximg,            // act image view (RGBA UINT32, K/16 × M)
    __read_only image2d_t  Wimg,            // weight image view (RGBA UINT32, K/32 × N)
    __global const float  *scale_act,       // [M] per-row act scale
    __global const float  *scale_wgt,       // [N] per-channel weight scale
    __global const int    *row_sum_act,     // [M] sum_k(int8 act_ik)
    __global       half   *Y,               // [M, N] fp16 output
    const int M, const int N, const int K) {
  const int n0 = get_global_id(0) * V8C_TN;
  const int m0 = get_global_id(1) * V8C_TM;
  const int K32 = K >> 5;

  int acc[V8C_TM][V8C_TN];
  #pragma unroll
  for (int i = 0; i < V8C_TM; i++)
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) acc[i][j] = 0;

  for (int k32 = 0; k32 < K32; k32++) {
    // 2 activation texels (16 K each) cover 32 K of activation row
    uint4 a_lo[V8C_TM], a_hi[V8C_TM];
    #pragma unroll
    for (int i = 0; i < V8C_TM; i++) {
      a_lo[i] = read_imageui(Ximg, SMP_v8c, (int2)(2*k32  , m0 + i));
      a_hi[i] = read_imageui(Ximg, SMP_v8c, (int2)(2*k32+1, m0 + i));
    }
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) {
      // 1 weight texel = 32 int4 K-channels (offset-encoded)
      uint4 w = read_imageui(Wimg, SMP_v8c, (int2)(k32, n0 + j));
      // Mask-only unpack: each masked uint = 4 unsigned bytes in [0..15]
      // (= encoded values; real value = encoded - 8).
      const uint M4 = 0x0F0F0F0Fu;
      uint w0lo =  w.x        & M4;  // K = j_block*32 + [ 0.. 3]
      uint w0hi = (w.x >> 4)  & M4;  // K = j_block*32 + [ 4.. 7]
      uint w1lo =  w.y        & M4;  // K = j_block*32 + [ 8..11]
      uint w1hi = (w.y >> 4)  & M4;  // K = j_block*32 + [12..15]
      uint w2lo =  w.z        & M4;  // K = j_block*32 + [16..19]
      uint w2hi = (w.z >> 4)  & M4;  // K = j_block*32 + [20..23]
      uint w3lo =  w.w        & M4;  // K = j_block*32 + [24..27]
      uint w3hi = (w.w >> 4)  & M4;  // K = j_block*32 + [28..31]
      #pragma unroll
      for (int i = 0; i < V8C_TM; i++) {
        // dot_4x8packed_su_int: signed_int8 (act) × unsigned_uint8 (encoded)
        acc[i][j] += dot_4x8packed_su_int(a_lo[i].x, w0lo)
                   + dot_4x8packed_su_int(a_lo[i].y, w0hi)
                   + dot_4x8packed_su_int(a_lo[i].z, w1lo)
                   + dot_4x8packed_su_int(a_lo[i].w, w1hi)
                   + dot_4x8packed_su_int(a_hi[i].x, w2lo)
                   + dot_4x8packed_su_int(a_hi[i].y, w2hi)
                   + dot_4x8packed_su_int(a_hi[i].z, w3lo)
                   + dot_4x8packed_su_int(a_hi[i].w, w3hi);
      }
    }
  }
  // Bias correction: subtract 8·row_sum_act_i (per (i,j) at the end of K-loop)
  #pragma unroll
  for (int i = 0; i < V8C_TM; i++) {
    int rs = row_sum_act[m0 + i];
    float s_i = scale_act[m0 + i];
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) {
      int corrected = acc[i][j] - 8 * rs;
      float v = (float)corrected * s_i * scale_wgt[n0 + j];
      Y[(long)(m0 + i) * N + (n0 + j)] = (half)v;
    }
  }
}

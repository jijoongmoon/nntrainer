// SPDX-License-Identifier: Apache-2.0
// Fused Q+K+V projection + RoPE layout kernel (paper §3.6 #1 of arXiv:2505.00232).
//
// REPLACES: three separate dotCl_v8c dispatches (Q, K, V FCs) + CPU RoPE pass
//           + V copy, currently invoked from
//             Applications/CausalLM/layers/qkv_layer.cpp:178  (3 FCs via dot())
//             Applications/CausalLM/layers/mha_core.cpp:978/980/1031/1035 (RoPE)
//
// PAPER:    "We crafted a custom kernel to combine rotary embedding with the
//           layout transformations of query (Q), key (K), and value (V)
//           projections, transforming the query projection from an initial
//           layout of (B,1,S,hq·dh) to a resultant layout of
//           (B·hkv, S·hq/hkv, dh)." -- §3.6
//
// STATUS:   Step 2a skeleton. Stub kernel — fills outputs with zeros so the
//           build + dispatch + env-gate plumbing can be exercised without
//           introducing a correctness regression. Step 2b will fill in the
//           quant + 3-GEMM + RoPE + layout math.
//
// QWEN3-0.6B SHAPES (for reference; the kernel is parameterized):
//   hidden = 1024, hq = 16, hkv = 8, dh = 128, hq/hkv = 2
//   Q FC:  K=1024, N = hq * dh = 2048
//   K FC:  K=1024, N = hkv * dh = 1024
//   V FC:  K=1024, N = hkv * dh = 1024
//   RoPE: cos/sin tables [max_position, dh] in fp16, half_ = dh/2 = 64

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Skeleton entry. Step 2b will replace the body. The signature is intentionally
// final-form (real bindings) so the host dispatch can be wired now and not
// rewritten when the body lands.
__kernel void fused_qkv_rope_skeleton(
    // Inputs (activation + 3 weights + scales + RoPE tables)
    __global const half  *act_fp16,          // [S, hidden] FP16
    __global const char  *wq_packed,         // QINT4 packed Q weights
    __global const char  *wk_packed,         // QINT4 packed K weights
    __global const char  *wv_packed,         // QINT4 packed V weights
    __global const half  *wq_scales,         // [N_q] FP16 per-channel scales
    __global const half  *wk_scales,         // [N_kv]
    __global const half  *wv_scales,         // [N_kv]
    __global const half  *cos_table,         // [max_pos, dh] FP16
    __global const half  *sin_table,         // [max_pos, dh] FP16

    // Outputs
    __global       half  *q_out,             // [S, hq*dh] FP16, post-RoPE
    __global       half  *k_out,             // [S, hkv*dh] FP16, post-RoPE
    __global       half  *v_out,             // [S, hkv*dh] FP16, no RoPE

    // Geometry
    const int S,                              // sequence length this dispatch
    const int hidden,                         // input feature dim
    const int hq,                             // num Q heads
    const int hkv,                            // num KV heads
    const int dh,                             // head dim
    const int from_pos                        // RoPE position offset (cache index)
) {
  // Step 2a stub: zero outputs. The dispatch wiring + env gate are what's
  // being validated here; Step 2b replaces this body with real compute.
  const int s = get_global_id(0);   // row (token within this dispatch window)
  const int n = get_global_id(1);   // output column tile id

  if (s >= S) return;

  // q_out row stride = hq * dh; k_out / v_out stride = hkv * dh.
  // For the stub we write zeros across the full output width on this row
  // when n == 0, and return on other n's so we don't double-write.
  if (n != 0) return;

  for (int j = 0; j < hq * dh; ++j)  q_out[s * hq  * dh + j] = (half)0.0f;
  for (int j = 0; j < hkv * dh; ++j) k_out[s * hkv * dh + j] = (half)0.0f;
  for (int j = 0; j < hkv * dh; ++j) v_out[s * hkv * dh + j] = (half)0.0f;

  // Silence "unused parameter" warnings during skeleton phase.
  (void)act_fp16; (void)wq_packed; (void)wk_packed; (void)wv_packed;
  (void)wq_scales; (void)wk_scales; (void)wv_scales;
  (void)cos_table; (void)sin_table;
  (void)hidden; (void)from_pos;
}

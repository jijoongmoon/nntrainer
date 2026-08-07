// SPDX-License-Identifier: Apache-2.0
/**
 * @file    cuda_gdn.h
 * @brief   Eager GPU GatedDeltaNet single-token decode for the CUDA backend.
 *
 * Runs the whole GDN decode step on the backend stream with ONE drain at the
 * end (vs ~7 host<->GPU transitions on the host path):
 *   4x in_proj GEMV (fp16 W, fp32 acc) -> depthwise conv1d + SiLU with the
 *   persistent ring -> per-v-head decay-first delta recurrence + gated RMSNorm
 *   (fused, fp32, state updated in place) -> out_proj GEMV (fp16 out).
 * Prefill stays on the host (bit-exact reference path); the persistent
 * state/ring tensors are FP32 UVM so both sides stay coherent.
 */
#ifndef __CUDA_GDN_H__
#define __CUDA_GDN_H__

namespace nntrainer::cuda {

/**
 * @brief One GDN decode step (B==1) fully on the GPU.
 *
 * All pointers must be device-accessible (UVM ok) EXCEPT h_wconv/h_alog/
 * h_dtb/h_wnorm, which are HOST fp32 arrays; they are uploaded once per layer
 * into a device cache keyed by h_wconv (stable per-layer heap pointer).
 *
 * @param x       [H] fp16 input token
 * @param wqkv    [H, CONV] fp16 in_proj_qkv ([in,out] row-major)
 * @param wz      [H, VAL] fp16 in_proj_z
 * @param wb      [H, NVH] fp16 in_proj_b
 * @param wa      [H, NVH] fp16 in_proj_a
 * @param wout    [VAL, H] fp16 out_proj
 * @param h_wconv [CONV, KS] fp32 HOST depthwise conv weight
 * @param h_alog  [NVH] fp32 HOST A_log
 * @param h_dtb   [NVH] fp32 HOST dt_bias
 * @param h_wnorm [HVD] fp32 HOST gated-RMSNorm weight
 * @param state   [NVH, HKD, HVD] fp32 persistent recurrent state (in-place)
 * @param ring    [CONV, KS-1] fp32 persistent conv ring (in-place, advanced)
 * @param out     [H] fp16 output
 * @param H,NVH,NKH,HKD,HVD,KS  dims (HVD must be a power of two <= 1024;
 *                              HKD <= 128; H and VAL <= 4096 for the
 *                              static shared-memory GEMV x tile)
 * @param eps     rsqrt epsilon (l2norm and gated RMSNorm)
 * @return false on any registration/alloc/launch failure or unsupported dims
 *         (caller falls back to the host path)
 */
bool cuda_gdn_decode_fp16(const unsigned short *x, const unsigned short *wqkv,
                          const unsigned short *wz, const unsigned short *wb,
                          const unsigned short *wa, const unsigned short *wout,
                          const float *h_wconv, const float *h_alog,
                          const float *h_dtb, const float *h_wnorm,
                          float *state, float *ring, unsigned short *out,
                          unsigned int H, unsigned int NVH, unsigned int NKH,
                          unsigned int HKD, unsigned int HVD, unsigned int KS,
                          float eps);

/** @brief Pre-grow the GDN decode scratch (projection/conv/normed buffers). */
bool cuda_gdn_prewarm(unsigned int H, unsigned int NVH, unsigned int NKH,
                      unsigned int HKD, unsigned int HVD);

} // namespace nntrainer::cuda

#endif /* __CUDA_GDN_H__ */

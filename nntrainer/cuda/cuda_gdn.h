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
                          float eps, const float *qkv_pre = nullptr);

/** @brief Pre-grow the GDN decode scratch (projection/conv/normed buffers). */
bool cuda_gdn_prewarm(unsigned int H, unsigned int NVH, unsigned int NKH,
                      unsigned int HKD, unsigned int HVD);

/**
 * @brief Everything between the input projections and the layer output, for a
 *        whole prefill of T tokens, on the device.
 *
 * conv1d(+SiLU) -> l2norm(q,k) + q.k -> decay-first delta scan -> gated
 * RMSNorm -> out_proj. The caller supplies the four projection outputs (they
 * are plain dense GEMMs and already go through cuBLAS) and gets the layer
 * output back; nothing in between touches the host.
 *
 * The scan keeps the recurrent state S[HKD][HVD] entirely in REGISTERS -- 512
 * threads x 32 floats each covers 128x128 exactly -- so a token costs ~96 FMAs
 * per thread and NO memory traffic for S. Holding S in shared memory instead
 * would cost ~256 KB of shared traffic per token per head, which at T=20000
 * over 30 layers is ~1.9 s versus ~0.2 s. That is why HKD and HVD are pinned
 * to 128 below rather than made general.
 *
 * @param p_qkv [T,CONV] fp32 in_proj_qkv output (device-accessible)
 * @param p_z   [T,VAL]  fp32 in_proj_z output
 * @param p_b   [T,NVH]  fp32 in_proj_b output
 * @param p_a   [T,NVH]  fp32 in_proj_a output
 * @param wout  [VAL,H]  fp16 out_proj weight
 * @param h_wconv/h_alog/h_dtb/h_wnorm HOST fp32 per-layer params (uploaded
 *        once per layer, cached on h_wconv as in cuda_gdn_decode_fp16)
 * @param state [NVH,HKD,HVD] fp32 recurrent state; read when seed_state,
 *        written when save_state (in-place)
 * @param ring  [CONV,KS-1] fp32 conv ring; READ as the causal left-pad when
 *        seed_state. Writing it back is left to the caller, which already
 *        does so from p_qkv.
 * @param out   [T,H] fp16 layer output
 * @param seed_state resume a chunked prefill from the persistent state+ring
 *        instead of starting from zero
 * @return false on any unsupported dim / alloc / dispatch failure, having
 *         mutated nothing the caller cannot recompute -- state is written only
 *         on the success path.
 */
bool cuda_gdn_prefill_fp16(const float *p_qkv, const float *p_z,
                           const float *p_b, const float *p_a,
                           const unsigned short *wout, const float *h_wconv,
                           const float *h_alog, const float *h_dtb,
                           const float *h_wnorm, float *state,
                           const float *ring, unsigned short *out,
                           unsigned int T, unsigned int H, unsigned int NVH,
                           unsigned int NKH, unsigned int HKD, unsigned int HVD,
                           unsigned int KS, float eps, bool seed_state,
                           bool save_state);

/**
 * @brief The chunked (WY/UT-transform) prefill: the delta-rule scan
 *        re-expressed as chunk-of-64 matrix work (vLLM's fla decomposition),
 *        same signature, same algebraic contract, same buffers in and out.
 *        Not bit-identical to the sequential scan (exp-of-cumsum-differences
 *        replaces per-token products), fp32 throughout; validate with
 *        NNTR_CUDA_GDN_CHUNK=2 against the sequential device path.
 */
bool cuda_gdn_prefill_chunked_fp16(
  const float *p_qkv, const float *p_z, const float *p_b, const float *p_a,
  const unsigned short *wout, const float *h_wconv, const float *h_alog,
  const float *h_dtb, const float *h_wnorm, float *state, const float *ring,
  unsigned short *out, unsigned int T, unsigned int H, unsigned int NVH,
  unsigned int NKH, unsigned int HKD, unsigned int HVD, unsigned int KS,
  float eps, bool seed_state, bool save_state);

/**
 * @brief Device conv-ring rebuild -- the byte-exact replacement for the
 *        layer's host save_ring loop. Reads the RAW projections (pre-conv)
 *        exactly as the host lambda did; in-place safe (old ring is staged
 *        in registers first). has_prev = the old ring carries over when the
 *        chunk is shorter than the ring (seed_state).
 */
bool cuda_gdn_save_ring_dev(const float *p_qkv, float *ring, unsigned int T,
                            unsigned int CONV, unsigned int KS, bool has_prev);

} // namespace nntrainer::cuda

#endif /* __CUDA_GDN_H__ */

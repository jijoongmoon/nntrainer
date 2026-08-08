// SPDX-License-Identifier: Apache-2.0
/**
 * @file    cuda_moe.h
 * @brief   Grouped MoE expert FFN on the device.
 *
 * Replaces the per-expert loop of ops->fc calls with ~7 launches per layer.
 * Two separate problems made that loop expensive, and both are fixed here:
 *
 *  - DRAINS. Every ops->fc ended in maybeFinish(), a full cudaStreamSynchronize
 *    on an integrated GPU, and the host code between the GEMMs (gather, SwiGLU,
 *    scatter) is what made those drains necessary. Doing everything on the
 *    device removes the host reads and so removes the drains.
 *  - WORK PER LAUNCH. dp4a_gemm_reg's 64x64 tile gives ceil(N/64)*ceil(M/64) =
 *    EIGHT blocks for one expert at the prefill shape (M~42, N=512) on a 16-SM
 *    part, measured at 0.32 TOPS against a ~21 TOPS dp4a peak. Note the tile is
 *    NOT the problem: a 16x16 tile with 12x the blocks measured 1.4x SLOWER.
 *    The grouped kernel keeps the tile and puts every routed expert in one
 *    grid instead -- 2,048 blocks at prefill, 64 at decode.
 *
 * Written fresh against this base's PLAIN int4 / fp32-scale QS4CX layout. The
 * retired lane's cuda_moe.cpp is NOT a valid source: same signatures over a
 * Section-A payload with fp16 scales, so a port compiles, links and produces
 * garbage.
 */
#ifndef __CUDA_MOE_H__
#define __CUDA_MOE_H__

namespace nntrainer::cuda {

// ---------------------------------------------------------------------------
// Per-expert primitives. These are the SHIPPING path: they took the drains out
// (one per layer instead of one per fc) and are measured and validated. The
// grouped entry point below supersedes them but is not wired yet.
// ---------------------------------------------------------------------------

/** @brief dst[i,:] = src[rows[i],:] for i in [0,m). `rows` device-accessible. */
bool cuda_moe_gather_fp16(const unsigned short *src, unsigned short *dst,
                          const int *rows, unsigned int m, unsigned int width);

/**
 * @brief Router logits on the device: L[T,E] = X[T,H](fp16) * Wg[H,E](fp32),
 *        widened and accumulated in fp32.
 *
 * Replaces the host `input.clone(FP32)` + OpenBLAS sgemm that dominated the
 * MoE routing block (13.6 s of a 35 s prefill for the whole block). Kept in
 * fp32 rather than put on the fp16 Tensor Cores on purpose: the top-k pick is
 * DISCRETE, so a weight rounded to fp16 does not give a slightly wrong answer,
 * it gives a different expert.
 */
bool cuda_moe_router_gemm_fp16(const unsigned short *X, const float *Wg,
                               float *L, unsigned int T, unsigned int H,
                               unsigned int E);

/** @brief out = silu(gate)*up elementwise, fp32 math, fp16 storage. */
bool cuda_moe_swiglu_fp16(const unsigned short *gate, const unsigned short *up,
                          unsigned short *out, unsigned int n);

/**
 * @brief dst[rows[i],:] += wts[i]*src[i,:] for i in [0,m).
 * @note Safe without atomics ONLY because it is called once per expert and one
 *       expert never sees the same token twice (topK returns distinct indices
 *       within a row). Calling it over all experts at once would race.
 */
bool cuda_moe_scatter_add_fp16(const unsigned short *src, unsigned short *dst,
                               const int *rows, const float *wts,
                               unsigned int m, unsigned int width);

/** @brief Mapped staging for `m` row indices and routing weights. */
bool cuda_moe_stage(unsigned int m, int **rows_out, float **wts_out);

// ---------------------------------------------------------------------------
// Grouped path (written, NOT yet wired into the layer).
// ---------------------------------------------------------------------------

/**
 * @brief Host-written, device-read description of one layer's routing.
 *
 * All pointers are into one mapped allocation owned by cuda_moe_plan_stage().
 * Assignments are bucketed by expert: expert e owns rows [wl_r0, wl_r0+wl_n) of
 * the A-row activation block, and `slots` is the inverse map the combine step
 * needs.
 */
struct MoePlan {
  int *rows;    /**< [A] source token index per assignment (expert-major) */
  float *wts;   /**< [A] routing weight per assignment */
  int *slots;   /**< [T*topk] assignment index per (token, k), -1 if unused */
  int *wl_e;    /**< [W] expert of each work item */
  int *wl_r0;   /**< [W] first assignment row of each work item */
  int *wl_n;    /**< [W] row count of each work item (<= 64) */
  /**
   * [3E] weight payload pointers, PROJECTION-MAJOR: three contiguous blocks of
   * E, so wptr[off_gate + e] is expert e's gate. The kernel is handed
   * (wptr + off_X) and indexes it by expert alone, which is what lets one grid
   * cover every expert.
   */
  const unsigned char **wptr;
  const unsigned short **wsc; /**< [3E] fp16 scale pointers, same indexing */
  unsigned int off_up;        /**< start of the `up` block (= 0) */
  unsigned int off_gate;      /**< start of the `gate` block (= E) */
  unsigned int off_down;      /**< start of the `down` block (= 2E) */
};

/**
 * @brief Grow and hand back the mapped staging for one layer's plan.
 * @param A     total assignments (tokens * topk)
 * @param T     tokens, @param topk experts per token, @param E expert count
 * @param Wmax  upper bound on work items (sum of ceil(m_e/64))
 */
bool cuda_moe_plan_stage(unsigned int A, unsigned int T, unsigned int topk,
                         unsigned int E, unsigned int Wmax, MoePlan *out);

/**
 * @brief Allocate a per-LAYER mapped weight-pointer table of `n` entries each.
 *
 * Deliberately NOT part of cuda_moe_plan_stage's shared staging: that is reused
 * every forward by every layer, and every layer has different experts, so a
 * shared table would be overwritten by whichever layer ran last. This one is
 * filled once (weight pointers are stable for the run -- the weight arena is
 * allocate-once and there is no FSU here) and owned for the process lifetime;
 * 40 layers x 768 pointers x 2 tables is ~500 KB.
 */
bool cuda_moe_new_ptr_table(unsigned int n, const unsigned char ***wp,
                            const unsigned short ***ws);

/**
 * @brief gather -> quant -> grouped gate/up -> SwiGLU -> quant -> grouped down
 *        -> token-major weighted combine, entirely on the device.
 *
 * @param input  [T, H] fp16 layer input
 * @param output [T, H] fp16 layer output, OVERWRITTEN (not accumulated: the
 *               combine sums each token's own topk contributions in slot order)
 * @param W      number of work items the caller filled in the plan
 * @return false on any alloc / dispatch failure, having written nothing the
 *         caller cannot recompute.
 */
bool cuda_moe_expert_ffn_fp16(const unsigned short *input,
                              unsigned short *output, const MoePlan &p,
                              unsigned int A, unsigned int W, unsigned int T,
                              unsigned int topk, unsigned int H,
                              unsigned int I);

} // namespace nntrainer::cuda

#endif /* __CUDA_MOE_H__ */

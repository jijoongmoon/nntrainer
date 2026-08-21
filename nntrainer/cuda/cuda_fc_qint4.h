// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_fc_qint4.h
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Fused QS4CX dequant-GEMM for the CUDA FC layer:
 *          Y[M,N] = X[M,K] * dequant(W), where W is the QS4CX PLAIN payload
 *          (row-major [N][(K+1)/2] nibbles, uint4 = int4+8) with an
 *          N-entry per-channel fp16 scale. The int4 weight is read and
 *          dequantized inline in the kernel; float accumulation. Callers must
 *          pass device-accessible (UVM) pointers.
 */

#ifndef __CUDA_FC_QINT4_H__
#define __CUDA_FC_QINT4_H__

#include <cstddef>

namespace nntrainer::cuda {

/**
 * @brief Smallest GEMM M for which the cuBLAS int8-IMMA FC path is selected.
 *
 * Below this the dp4a int-ALU GEMM wins (the IMMA path's per-call activation
 * repack + cuBLAS launch overhead is not amortized), so the dispatcher only
 * reaches cuda_fc_qs4cx_cublas_i8_gemm_fp16() at M >= this. It is therefore
 * also the exact condition under which the i8 [K,N] weight cache can ever be
 * READ: a turn whose largest prefill M stays below it never touches that
 * cache, and building it eagerly is provably dead work. Single source of truth
 * for both the dispatcher (cuda_compute_ops.cpp) and the load-time prewarm
 * (causal_lm.cpp) -- they must not drift apart, or the prewarm builds caches
 * nothing reads (or skips caches something reads).
 */
constexpr unsigned int CUDA_FC_I8_PREFILL_MIN_M = 32u;

/**
 * @brief Build (and cache) the N-entry UVM fp16 per-channel scale buffer from
 *        the tensor's fp32 scales. The dequant kernels read the scale on device
 *        every call; the tensor stores fp32, so the fp16 copy is made once at
 *        first use and cached by the fp32-scale pointer (weights live for the
 *        process lifetime). @p out_sc receives the cached device pointer.
 * @return false on allocation failure (caller falls back to the host path).
 */
bool cuda_fc_qs4cx_scales_to_uvm_fp16(const float *fp32_scales, unsigned int N,
                                      const unsigned short **out_sc);

/**
 * @brief Y[M,N] = X[M,K] * dequant(QS4CX W) where W is the PLAIN QS4CX payload
 *        and @p scales_fp16 is the N-entry fp16 scale buffer (from
 *        cuda_fc_qs4cx_scales_to_uvm_fp16). FP32 activation, FP32 output.
 *        One thread per output element; float accumulation.
 * @return true on success.
 */
bool cuda_fc_qs4cx_gemm_fp32(const float *X, const unsigned char *plain_w,
                             const unsigned short *scales_fp16, float *Y,
                             unsigned int M, unsigned int N, unsigned int K);

/**
 * @brief fp16-activation variant of cuda_fc_qs4cx_gemm_fp32: fp16 in / fp16
 *        out, staged through fp32 for the plain-decode GEMM (float
 *        accumulation, no int8 activation quantization -- the accuracy
 *        reference for the int4 FC).
 * @return true on success.
 */
bool cuda_fc_qs4cx_gemm_fp16_naive(const unsigned short *Xh,
                                   const unsigned char *plain_w,
                                   const unsigned short *scales_fp16,
                                   unsigned short *Yh, unsigned int M,
                                   unsigned int N, unsigned int K);

/**
 * @brief w4a8 dp4a fast path: Y[M,N] = X[M,K] * dequant(QS4CX W), FP32
 *        activation. Per-row asymmetric int8 activation quant + symmetric int4
 *        weight, int8xint8 dot via __dp4a on the int ALU. The int4 weight is
 *        repacked to signed packed int4 once and cached on device (keyed by
 *        @p plain_w). The int32 accumulate is exact.
 * @return true on success.
 */
bool cuda_fc_qs4cx_dp4a_gemm_fp32(const float *X, const unsigned char *plain_w,
                                  const unsigned short *scales_fp16, float *Y,
                                  unsigned int M, unsigned int N,
                                  unsigned int K);

/** @brief fp16-activation variant of cuda_fc_qs4cx_dp4a_gemm_fp32: fp16 in /
 *  fp16 out (the conversion folded into the GEMM epilogue). */
bool cuda_fc_qs4cx_dp4a_gemm_fp16(const unsigned short *Xh,
                                  const unsigned char *plain_w,
                                  const unsigned short *scales_fp16,
                                  unsigned short *Yh, unsigned int M,
                                  unsigned int N, unsigned int K);

/**
 * @brief Grouped-MoE int4 GEMM on the Tensor Cores (imma_moe_grouped): one
 *        launch covers every expert via a padded per-expert block work list;
 *        block_expert[b] steers block b to its expert's weight through the
 *        wp_tab/ws_tab pointer tables (-1 discards). tokid maps gathered rows
 *        to source token rows of q8 (nullptr = direct/gathered input, the
 *        down projection). Bit-identical to per-expert imma_gemm_pipe calls.
 *        N and K must be multiples of 64; buffers are caller-owned.
 */
/**
 * @brief NNTR_MOE_G3 grouped GEMM (cp.async ring + packed fragment-order W +
 *        precomputed per-expert rowsum table). Payload must be repacked via
 *        cuda_fc_qs4cx_moe_repack_g3 first. Output bytes identical to
 *        cuda_fc_qs4cx_moe_grouped_gemm on the same inputs.
 */
bool cuda_fc_qs4cx_moe_grouped_gemm_g3(
  const signed char *q8, const int *tokid, const unsigned long long *wp_tab,
  const unsigned long long *ws_tab, const unsigned long long *wr_tab,
  const int *block_expert, const float *ascale, const int *azp, void *Y,
  unsigned int n_mblocks, unsigned int N, unsigned int K, int out_fp16,
                                       const int *wl_n = nullptr);

/** @brief G3 down variant (K <= 512): persistent-N, grid (1, W). */
bool cuda_fc_qs4cx_moe_grouped_gemm_g3d(
  const signed char *q8, const unsigned long long *wp_tab,
  const unsigned long long *ws_tab, const unsigned long long *wr_tab,
  const int *block_expert, const float *ascale, const int *azp, void *Y,
  unsigned int n_mblocks, unsigned int N, unsigned int K, int out_fp16,
                                        const int *wl_n = nullptr);

/** @brief In-place fragment repack of ALL E payloads via the pointer table. */
/**
 * @brief Slab-to-slab m4-order repack of ALL E expert payloads of one
 * projection (imma_moe_g4's fragment-chunk order). Requires the payloads to
 * be ONE contiguous device slab (stride N*K/2, wp_tab[0] = base). On success
 * the table entries are repointed to the new slab and the old slab is freed;
 * on failure the payloads are untouched (caller may fall back to the g3
 * repack). N % 128 == 0, K % 256 == 0.
 */
bool cuda_fc_qs4cx_moe_repack_m4(unsigned long long *wp_tab, unsigned int E,
                                 unsigned int N, unsigned int K);

/**
 * @brief Grouped gate/up GEMM on m4-order payloads (imma_moe_g4): same
 * steering and epilogue semantics as _g3, BN=128 grid. Payloads must have
 * been through cuda_fc_qs4cx_moe_repack_m4.
 */
bool cuda_fc_qs4cx_moe_grouped_gemm_g4(
  const signed char *q8, const int *tokid, const unsigned long long *wp_tab,
  const unsigned long long *ws_tab, const unsigned long long *wr_tab,
  const int *block_expert, const float *ascale, const int *azp, void *Y,
  unsigned int n_mblocks, unsigned int N, unsigned int K, int out_fp16,
  const int *wl_n);

bool cuda_fc_qs4cx_moe_repack_g3(const unsigned long long *wp_tab,
                                 unsigned int E, unsigned int N,
                                 unsigned int K);

/** @brief Batched per-channel int4 rowsum into rs[e*N + n], one projection. */
bool cuda_fc_qs4cx_moe_rowsum_g3(const unsigned long long *wp_tab,
                                 unsigned int E, unsigned int N,
                                 unsigned int K, int *rs);

bool cuda_fc_qs4cx_moe_grouped_gemm(
  const signed char *q8, const int *tokid, const unsigned long long *wp_tab,
  const unsigned long long *ws_tab, const int *block_expert,
  const float *ascale, const int *azp, void *Y, unsigned int n_mblocks,
  unsigned int N, unsigned int K, int out_fp16);

/**
 * @brief gate+up FUSED grouped GEMM: one A staging serves both projections'
 *        W tiles (16 mma per k-step against one barrier). Outputs Yg/Yu are
 *        written identically to two cuda_fc_qs4cx_moe_grouped_gemm calls.
 */
bool cuda_fc_qs4cx_moe_grouped_gemm2(
  const signed char *q8, const int *tokid, const unsigned long long *wpg_tab,
  const unsigned long long *wsg_tab, const unsigned long long *wpu_tab,
  const unsigned long long *wsu_tab, const int *block_expert,
  const float *ascale, const int *azp, void *Yg, void *Yu,
  unsigned int n_mblocks, unsigned int N, unsigned int K, int out_fp16);

/**
 * @brief Wide-N (64x128 block, 32x32 warp tiles) grouped GEMM: halves the
 *        B-fragment ldmatrix per mma. Same output as
 *        cuda_fc_qs4cx_moe_grouped_gemm; requires N % 128 == 0.
 */
bool cuda_fc_qs4cx_moe_grouped_gemm_w(
  const signed char *q8, const int *tokid, const unsigned long long *wp_tab,
  const unsigned long long *ws_tab, const int *block_expert,
  const float *ascale, const int *azp, void *Y, unsigned int n_mblocks,
  unsigned int N, unsigned int K, int out_fp16);

/**
 * @brief fp16 activation in, FP32 out (the GDN projection variant): the same
 *        act-quant + w4a8 ladder as the fp16 entry, writing float Y directly.
 */
bool cuda_fc_qs4cx_dp4a_gemm_fp16in_f32out(const unsigned short *Xh,
                                           const unsigned char *plain_w,
                                           const unsigned short *scales_fp16,
                                           float *Yf, unsigned int M,
                                           unsigned int N, unsigned int K);

/** @brief NNTR_CUDA_FUSED_NORMQ (default on, =0 opts out): whether the decode
 *  RMSNorm may fold in the int8 activation quant of the FC group it feeds. */
bool cuda_fc_qs4cx_fused_normq_enabled();

/**
 * @brief Shared-expert decode-chain fusion hooks (NNTR_SHEXP_FUSE; 0/unset =
 * off, 1 = full fusion, 2 = bit-identical mode that keeps the cuBLAS gate_lin
 * + sigmoid live). Each backend entry on the qwen3_5_moe shared-expert chain
 * calls its hook first; a true return means the dispatch was handled (fused
 * or skipped) and the caller must return success without launching anything.
 * The first two M=1 sightings of a layer's chain record and verify the
 * operand pointers; from the third token shared_gate launches the fused
 * gate/up/swiglu(+gate_lin+sigmoid) kernel and shared_mul launches the fused
 * down+gate-scale kernel. shexp_fc_qs4cx_hook must be called with the dp4a
 * cache mutex held (it resolves DevWeightQ planes).
 */
bool shexp_fc_qs4cx_hook(const unsigned short *Xh, const unsigned char *plain_w,
                         const unsigned short *scales_fp16, unsigned short *Yh,
                         unsigned int M, unsigned int N, unsigned int K);
/** @brief gate_lin (dense fp16 N=1) hook -- see shexp_fc_qs4cx_hook. */
bool shexp_dense_hook(const void *Xh, const void *Wh, void *Yh, unsigned int M,
                      unsigned int N, unsigned int K);
/** @brief shared_swiglu hook -- see shexp_fc_qs4cx_hook. */
bool shexp_swiglu_hook(const unsigned short *gate, const unsigned short *up,
                       unsigned short *out, unsigned int n);
/** @brief shared_gate_sig (in-place sigmoid, n==1) hook. */
bool shexp_sigmoid_hook(unsigned short *x, unsigned int n);
/** @brief shared_mul (row-broadcast multiply) hook; launches the fused down
 *  projection + gate scale into @p out when the chain is fused. */
bool shexp_bcast_hook(const unsigned short *a, const unsigned short *g,
                      unsigned short *out, int n, int W);

/**
 * @brief NNTR_DENSE_I8W (opt-in): one-time int8 [N,K] planes + w8a8
 * warp-per-row GEMV for huge M=1 fp16 dense FCs -- the decode lm_head, whose
 * cuBLAS fp16 form streams a ~1 GB weight plane per token at the DRAM
 * roofline. gdn-i8w family convention (per-channel absmax/127 scale, integer
 * rowsum, asym activation correction). True = dispatched here; false = caller
 * runs its normal (cuBLAS) path. The plane build happens on the first eager
 * M=1 sighting, never under capture.
 */
bool cuda_fc_dense_i8w_gemv(const void *Xh, const void *Wh, void *Yh,
                            unsigned int M, unsigned int N, unsigned int K);

/**
 * @brief RMSNorm fused with the int8 activation quant its consumer FC needs.
 *
 * Writes the normed fp16 rows to @p y exactly as cuda_rmsnorm_fp16 would, and
 * in the same launch stages the per-row asymmetric int8 quant of those rows in
 * the dp4a activation scratch. The next FC on @p y then runs its GEMM without
 * a quant launch of its own -- and so do its siblings (q/k/v share one norm,
 * gate/up share another), which is where the decode launch count comes down.
 * The staging is published under a pointer + width + stream-dispatch-sequence
 * stamp, so an unrelated kernel writing a recycled buffer at the same address
 * cannot be mistaken for it.
 *
 * Bit-identical to the split rmsnorm_fp16 + act_quant_i8_h pair (identical
 * reduction order and identical rounding), so it needs no numerical waiver.
 *
 * @param x     [rows, width] fp16 input (device-accessible)
 * @param gamma [width] fp16 per-feature scale, or nullptr
 * @param y     [rows, width] fp16 output (device-accessible)
 * @param eps   epsilon added to the mean of squares
 * @param rows  row count (decode: 1)
 * @param width feature size (== K of the consuming FC)
 * @return false if the lever is off or the staging could not be prepared --
 *         the caller must then run the plain norm (nothing was published).
 */
bool cuda_fc_qs4cx_rmsnorm_prequant_fp16(const unsigned short *x,
                                         const unsigned short *gamma,
                                         unsigned short *y, float eps,
                                         unsigned int rows, unsigned int width);

/**
 * @brief w4a8 on the INT8 Tensor Cores via cuBLAS (prefill FC). Same quant
 *        scheme as the dp4a path (per-row asym int8 activation x symmetric int4
 *        weight) but the int8xint8->int32 GEMM runs on the IMMA Tensor Cores
 *        (~10x the dp4a int-ALU GEMM at prefill M). The int32 accumulate is
 *        exact, so the result is bit-identical to dp4a; the int4->int8 weight
 *        unpack is cached once. Returns false (caller falls to dp4a) on any
 *        cuBLAS/runtime failure.
 */
bool cuda_fc_qs4cx_cublas_i8_gemm_fp16(const unsigned short *Xh,
                                       const unsigned char *plain_w,
                                       const unsigned short *scales_fp16,
                                       unsigned short *Yh, unsigned int M,
                                       unsigned int N, unsigned int K);

/**
 * @brief [wprefetch] Migrate a QS4CX weight's managed plain payload (+ its
 *        fp32 scale tail) to the device with cudaMemPrefetchAsync, so the FC
 *        bytes leave host RSS and the GEMM reads them from VRAM. Discrete GPU
 *        only (a no-op / false on integrated, where managed pages don't
 *        migrate). @p plain_w must be a managed (UVM) pointer.
 * @return true if the prefetch was issued.
 */
bool cuda_fc_qs4cx_prefetch_weight(const unsigned char *plain_w, unsigned int N,
                                   unsigned int K);

/**
 * @brief [pool-bypass] Drop the plain payload's fully-owned pages once every
 *        derived device cache exists (the forward only key-compares the
 *        pointer). Meaningful with NNTR_QS4CX_HEAP_BYPASS (heap pages);
 *        harmless EINVAL no-op on managed/pool memory. Refuses when
 *        NNTR_FC_CUDA_DP4A=0 (the naive path reads the payload). x86 only.
 * @return true if pages were dropped
 */
bool cuda_fc_qs4cx_drop_plain_pages(const unsigned char *plain_w,
                                    unsigned int N, unsigned int K);

/**
 * @brief [pool-bypass] True when the dp4a derived cache exists for this
 *        plain pointer -- dispatch may then treat the pointer as a pure key
 *        (no device access, no staging needed).
 * @note DP4A ONLY. The cuBLAS-i8 [K,N] cache is a separate map with its own
 *       existence condition, so a true here does NOT license the i8 path to
 *       assume a hit: on a miss that path binds the payload into
 *       repack_plain_i8_kn, and it therefore checks device-readability itself
 *       before building (and reports failure so the caller falls to dp4a).
 */
bool cuda_fc_qs4cx_has_cache(const unsigned char *plain_w);

/**
 * @brief [pool-bypass] True once cuda_fc_qs4cx_drop_plain_pages() has actually
 *        discarded this payload's pages. Reading those bytes afterwards yields
 *        zero-filled pages, so any path that would dereference the payload --
 *        the naive plain GEMM, or the host dot() fallback -- must refuse rather
 *        than compute against zeros.
 */
bool cuda_fc_qs4cx_plain_dropped(const unsigned char *plain_w);

/**
 * @brief Build the dp4a derived weight cache (packed int4 + rowsum) for one
 *        QS4CX plain payload at load time, off the first prefill. Host permute
 *        + row-sum fold in bounded chunks, then H2D; idempotent (the in-memory
 *        cache is pointer-keyed, process-local). Returns false only on a device
 *        allocation / dispatch failure (the lazy in-path build then remains the
 *        fallback).
 * @param cache_name stable per-weight name for the derive-once pack disk cache
 *        (cuda_pack_cache.h), or nullptr to derive without consulting/writing
 *        it. NEVER the pointer -- the pack is keyed by (file identity, name).
 */
bool cuda_fc_qs4cx_prewarm(const unsigned char *plain_w, unsigned int N,
                           unsigned int K, const char *cache_name = nullptr);

/**
 * @brief Split of the load-time prewarm cost into the part a persistent pack
 *        cache can remove (host derive + miss-path tee) and the part it cannot
 *        (the H2D upload), plus what pack HITs actually cost. Any pointer may
 *        be null. Milliseconds are summed over all prewarmed weights.
 */
void cuda_fc_qs4cx_prewarm_stats(double *derive_ms, double *upload_ms,
                                 double *tee_ms, double *hit_ms,
                                 size_t *derive_bytes, size_t *hit_bytes);

/**
 * @brief Mark a weight exempt from the eager load-time cuBLAS-i8 [K,N] cache
 *        build (skip_prefill towers / untied lm_head cannot reach the M>=32
 *        cuBLAS gate -- their int8 cache is dead VRAM). Lazy build self-heals.
 */
void cuda_fc_qs4cx_prewarm_exempt_i8(const void *plain_w);

/**
 * @brief Pre-grow the dp4a activation-quant scratch to the given decode
 *        bounds so the M=1 decode FC never cudaMallocs inside a CUDA-graph
 *        capture. maxN is accepted for signature stability; the decode path
 *        has no N-sized scratch.
 */
bool cuda_fc_qint4_dp4a_prewarm(unsigned int maxM, unsigned int maxK,
                                unsigned int maxN);

/**
 * @brief Free every pointer-keyed derived weight cache (dp4a packed int4 +
 *        cuBLAS int8) -- the model-reload teardown. The fp16-scale UVM side
 *        buffers are process-lifetime by design and are not freed.
 */
void cuda_fc_qs4cx_release_weight_caches();

} // namespace nntrainer::cuda

#endif // __CUDA_FC_QINT4_H__

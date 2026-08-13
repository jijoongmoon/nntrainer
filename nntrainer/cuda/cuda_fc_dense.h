// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_fc_dense.h
 * @date    06 Aug 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Dense (unquantized) FC GEMM on the device: Y[M,N] = X[M,K]*W[K,N]
 *          with an fp16 or fp32 weight, via cuBLAS.
 *
 *          The quantized FC dispatch (cuda_fc_qint4.h) only covers a QS4CX
 *          weight; a model that keeps ONE dense FC -- e.g. a square mixing
 *          matrix that is too small to be worth quantizing -- had no device
 *          arm at all and fell to the inherited host dot(). On the
 *          device-only activation pool that is not a slow path but a hard
 *          refusal (the operands are device memory the CPU cannot address),
 *          so a single dense FC forced the WHOLE model onto the
 *          host-coherent pool. These two entry points close that gap.
 *
 *          Declared with void* / float* only so cublas_v2.h stays out of the
 *          op-table translation unit (same separation as cuda_blas_manager.h).
 *          Callers must pass device-accessible pointers.
 */

#ifndef __CUDA_FC_DENSE_H__
#define __CUDA_FC_DENSE_H__

#include <cstddef>

namespace nntrainer::cuda {

/**
 * @brief Y[M,N] = X[M,K] * W[K,N], all three row-major fp16, fp32 accumulate.
 *
 * M == 1 (decode) is passed through to the same call: cuBLAS dispatches its
 * GEMV kernel for that shape, so no separate arm is needed.
 *
 * @param Xh device fp16 activation [M,K] row-major (ld=K)
 * @param Wh device fp16 weight [K,N] row-major (ld=N)
 * @param Yh device fp16 output [M,N] row-major (ld=N)
 * @return false when the handle is unavailable, the op-out is set, or cuBLAS
 *         reports failure -- the caller must then fall through to its own
 *         fallback chain.
 */
/**
 * @brief Lazy device mirror for a pinned-host dense WEIGHT plane (Tegra:
 *        the pinned pool is not GPU-L2-cached). Returns the device copy,
 *        or the original pointer when mirroring is off/unneeded/failed.
 */
const void *cuda_dense_w_dev(const void *w, size_t bytes);

bool cuda_fc_dense_gemm_fp16(const void *Xh, const void *Wh, void *Yh,
                             unsigned int M, unsigned int N, unsigned int K);

/**
 * @brief fp32 twin of cuda_fc_dense_gemm_fp16(). Same layout contract.
 */
bool cuda_fc_dense_gemm_fp32(const float *X, const float *W, float *Y,
                             unsigned int M, unsigned int N, unsigned int K);

/**
 * @brief fp16 in, fp32 OUT. Same layout contract as cuda_fc_dense_gemm_fp16().
 *
 * cublasGemmEx already accumulates these in fp32; this variant simply keeps
 * that accumulator instead of rounding it back to fp16 on the way out. It
 * exists for GDN, whose projections feed a conv1d / L2-norm / recurrence chain
 * that the host reference computes entirely in fp32 -- writing fp16 here would
 * make the device and host paths disagree for a reason unrelated to the GEMM.
 *
 * @param Xh device fp16 activation [M,K] row-major (ld=K)
 * @param Wh device fp16 weight [K,N] row-major (ld=N)
 * @param Y  device fp32 output [M,N] row-major (ld=N)
 */
bool cuda_fc_dense_gemm_fp16_f32out_acc(const void *Xh, const void *Wh,
                                        float *Y, unsigned int M,
                                        unsigned int N, unsigned int K,
                                        bool accumulate);
bool cuda_fc_dense_gemm_fp16_f32out(const void *Xh, const void *Wh, float *Y,
                                    unsigned int M, unsigned int N,
                                    unsigned int K);

/**
 * @brief Record that a dense weight of this dtype/shape exists, so a later
 *        cuda_fc_dense_warmup_run() can force cuBLAS to load its GEMM kernel
 *        libraries off the critical path. Does NO GPU work itself.
 *
 * cuBLAS loads kernel modules lazily, per shape class. MEASURED on the first
 * dense fp16 FC of a run: 14 cuLibraryLoadData calls, 106 ms, landing in the
 * middle of the first prefill and costing ~20% of its throughput -- a one-time
 * process cost charged entirely to the first request.
 *
 * Arming and running are SPLIT because the natural place to learn the shape
 * (the per-weight prebuild seam) runs inside the load workers, i.e. exactly
 * at the load-time RSS peak: warming there overlapped the ~170MB of cuBLAS
 * module residency with that peak and raised reported peak memory by the same
 * amount (measured 2259 -> 2430 MB) for no steady-state gain. Running it after
 * load keeps both the prefill win and the old peak.
 *
 * @param fp16 the dense weight is fp16 (else fp32)
 * @param N,K the weight shape; M a representative activation row count
 */
void cuda_fc_dense_warmup(bool fp16, unsigned int N, unsigned int K,
                          unsigned int M);

/**
 * @brief Execute any warm-up armed by cuda_fc_dense_warmup(). Call once after
 *        the model is loaded and before the first forward.
 *
 * A no-op when no dense weight was seen -- a fully quantized model must not
 * pay cuBLAS's fp16 module load for a path it never takes. Idempotent and
 * self-guarding: allocation failure is a silent skip, since this is an
 * optimization with no correctness role.
 */
void cuda_fc_dense_warmup_run();

} // namespace nntrainer::cuda

#endif // __CUDA_FC_DENSE_H__

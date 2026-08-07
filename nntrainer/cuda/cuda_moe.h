// SPDX-License-Identifier: Apache-2.0
/**
 * @file    cuda_moe.h
 * @brief   Device-side gather / SwiGLU / weighted scatter for the MoE layer.
 *
 * These exist for one reason: they are the HOST work sitting between the three
 * expert GEMMs, and while they are on the host every one of those GEMMs has to
 * drain the stream before the next line of C++ runs. On an integrated GPU that
 * drain is a full cudaStreamSynchronize. Measured on a 1,341-token prefill:
 * 61,440 drains, ~92% of the MoE layer's time, against ~8% of actual GEMM.
 *
 * Moving these three onto the device is what makes StreamManager's
 * deferred-drain region legal for the expert loop -- with no host read left
 * between the GEMMs, the whole loop can issue and drain once.
 *
 * Written fresh against this base's PLAIN int4 / fp32-scale QS4CX layout. The
 * retired lane's cuda_moe.cpp is NOT a valid source: its payload was Section-A
 * and its scales fp16, with identical signatures, so a port compiles, links and
 * produces garbage.
 */
#ifndef __CUDA_MOE_H__
#define __CUDA_MOE_H__

namespace nntrainer::cuda {

/**
 * @brief dst[i, :] = src[rows[i], :] for i in [0, m)
 * @param src   [*, width] fp16 source rows (the layer input)
 * @param dst   [m, width] fp16 contiguous gather buffer
 * @param rows  [m] device-accessible int32 source row indices
 */
bool cuda_moe_gather_fp16(const unsigned short *src, unsigned short *dst,
                          const int *rows, unsigned int m, unsigned int width);

/**
 * @brief out = silu(gate) * up, elementwise, fp32 math with fp16 storage.
 * @param n total element count (m * intermediate_size)
 */
bool cuda_moe_swiglu_fp16(const unsigned short *gate, const unsigned short *up,
                          unsigned short *out, unsigned int n);

/**
 * @brief dst[rows[i], :] += wts[i] * src[i, :] for i in [0, m)
 * @note No atomics: one expert never sees the same token twice, because topK
 *       returns distinct indices within a row. Two DIFFERENT experts do write
 *       the same token, which is why this is called once per expert rather
 *       than once for all of them.
 * @param wts [m] device-accessible fp32 routing weights
 */
bool cuda_moe_scatter_add_fp16(const unsigned short *src, unsigned short *dst,
                               const int *rows, const float *wts,
                               unsigned int m, unsigned int width);

/**
 * @brief Host-writable, device-readable staging for the per-expert row indices
 *        and routing weights.
 *
 * Returns pointers into a single mapped allocation grown to hold `m` of each.
 * Mapped rather than device memory so the host can fill them without a copy;
 * on Orin every pool is host-mapped anyway. Returns false if the grow fails or
 * would have to happen inside a graph capture.
 */
bool cuda_moe_stage(unsigned int m, int **rows_out, float **wts_out);

} // namespace nntrainer::cuda

#endif /* __CUDA_MOE_H__ */

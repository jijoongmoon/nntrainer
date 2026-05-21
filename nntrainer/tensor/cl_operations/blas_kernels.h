// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernels.h
 * @date	14 May 2024
 * @brief	Common blas OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNELS_H__
#define __BLAS_KERNELS_H__

#include <cl_context.h>
#include <engine.h>
#include <opencl_buffer.h>
#include <opencl_buffer_manager.h>
#include <opencl_kernel.h>

#include <string>

namespace nntrainer {

/**
 * @brief     signed 4-bit integer gemv async computation : C = A*B
 * @param[in] weight std::vector<void *> for int4 quantized weight
 * @param[in] scale std::vector<uint16_t *> for scales
 * @param[in] input uint16_t * for input
 * @param[in] output std::vector<uint16_t *> for output
 * @param[in] K hidden dimension
 * @param[in] Ns output dimensions
 */
void gemv_int4_async_cl(std::vector<void *> weights,
                        std::vector<uint16_t *> scales, uint16_t *input,
                        std::vector<uint16_t *> outputs, unsigned int K,
                        std::vector<unsigned int> Ns,
                        unsigned int quantization_group_size);

/**
 * @brief     signed 4-bit integer gemv async computation : C = A*B
 * @param[in] weight std::vector<void *> for int4 quantized weight
 * @param[in] scale std::vector<uint16_t *> for scales
 * @param[in] input float * for input
 * @param[in] output std::vector<float *> for output
 * @param[in] K hidden dimension
 * @param[in] Ns output dimensions
 */
void gemv_int4_async_cl(std::vector<void *> weights,
                        std::vector<uint16_t *> scales, float *input,
                        std::vector<float *> outputs, unsigned int K,
                        std::vector<unsigned int> Ns,
                        unsigned int quantization_group_size);

/**
 * @brief     signed 4-bit integer gemv computation : C = A*B
 * @param[in] weight char * for int4 quantized weight
 * @param[in] scale uint16_t * for scales
 * @param[in] input uint16_t * for input
 * @param[in] output uint16_t * for output
 * @param[in] K hidden dimension
 * @param[in] N output dimension
 */
void gemv_int4_cl(char *weight, uint16_t *scale, uint16_t *input,
                  uint16_t *output, unsigned int K, unsigned int N,
                  unsigned int quantization_group_size);

/**
 * @brief     signed 4-bit integer gemv computation : C = A*B
 * @param[in] weight char * for int4 quantized weight
 * @param[in] scale uint16_t * for scales
 * @param[in] input float * for input
 * @param[in] output float * for output
 * @param[in] K hidden dimension
 * @param[in] N output dimension
 */
void gemv_int4_cl(char *weight, uint16_t *scale, float *input, float *output,
                  unsigned int K, unsigned int N,
                  unsigned int quantization_group_size);

/**
 * @brief     Q4_0 gemm async computation : C = A*B
 * @param[in] matAdata std::vector<void *> for Matrix A
 * @param[in] matBdata float * for Matrix B
 * @param[in] matCdata std::vector<float *> for Matrix C
 * @param[in] M input dimension
 * @param[in] N output dimensions of As
 * @param[in] K hidden dimension
 */
void gemm_q4_0_async_cl(std::vector<void *> matAdata, float *matBdata,
                        std::vector<float *> matCdata, unsigned int M,
                        std::vector<unsigned int> N, unsigned int K);

/**
 * @brief     Q4_0 gemm computation : C = A*B
 * @param[in] matAdata void * for Matrix A
 * @param[in] matBdata float * for Matrix B
 * @param[in] matCdata float * for Matrix C
 * @param[in] M input dimension
 * @param[in] K hidden dimension
 * @param[in] N output dimension
 */
void gemm_q4_0_cl(void *matAdata, float *matBdata, float *matCdata,
                  unsigned int M, unsigned int N, unsigned int K);

/**
 * @brief INT4 GEMM computation for float input / output
 */
void sgemm_int4_cl(float *input, char *weight, uint16_t *scale, float *output,
                   unsigned int M, unsigned int N, unsigned int K,
                   unsigned int quantization_group_size);
/**
 * @brief INT4 GEMM computation for fp16 input / output
 */
void gemm_int4_cl(void *input, void *weights, void *scales, void *output,
                  unsigned int M, unsigned int N, unsigned int K,
                  unsigned int quantization_group_size);

/**
 * @brief INT4 GEMM async computation
 */
void gemm_int4_async_cl(float *input, std::vector<void *> weights,
                        std::vector<uint16_t *> scales,
                        std::vector<float *> matCdata, unsigned int M,
                        std::vector<unsigned int> Ns, unsigned int K,
                        unsigned int quantization_group_size);

/**
 * @brief     Q6_K sgemv computation : Y = A*X
 * @param[in] matAdata void * for Matrix A
 * @param[in] vecXdata float * for Vector X
 * @param[in] vecYdata float * for Vector Y
 * @param[in] M number of rows in matrix A
 * @param[in] N number of columns in matrix A
 */
void sgemv_q6_k_cl(void *matAdata, float *vecXdata, float *vecYdata,
                   unsigned int M, unsigned int N);

/**
 * @brief     sgemv computation : Y = A*X + Y
 * @param[in] matAdata float * for Matrix A
 * @param[in] vecXdata float * for Vector X
 * @param[in] vecYdata float * for Vector Y
 * @param[in] transA bool transpose
 * @param[in] dim1 number of A's columns
 * @param[in] dim2 number of A's rows
 * @param[in] lda number of X's columns
 * @param[in] context RunLayerContext reference
 */
void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda);

/**
 * @brief     dot computation : sum of all X * Y
 * @param[in] vecAdata float * for Vector A
 * @param[in] vecXdata float * for Vector X
 * @param[in] dim1 number of elements in both input vectors
 * @param[in] context RunLayerContext reference
 * @return    float dot product result
 */
float dot_cl(const float *vecAdata, const float *vecXdata, unsigned int dim1);

/**
 * @brief     sgemm computation : Y = op(A)*op(B) + C,
 * where op(X) is one of X or X**T
 * @param[in] transA bool transpose
 * @param[in] transB bool transpose
 * @param[in] A float * for Matrix A
 * @param[in] B float * for Matrix B
 * @param[in] C float * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] lda number of A's columns
 * @param[in] ldb number of B's columns
 * @param[in] ldc number of C's columns
 * @param[in] context RunLayerContext reference
 */
void sgemm_cl(bool TransA, bool TransB, const float *A, const float *B,
              float *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc);

/**
 * @brief     addition : sum of all input vectors
 * @param[in] input float * for input
 * @param[in] res float * for result/output
 * @param[in] size_input number of elements in input vector
 * @param[in] size_res number of elements in result vector
 */
void addition_cl(const float *input, float *res, unsigned int size_input,
                 unsigned int size_res);

/**
 * @brief rmsnorm each row of the tensor
 * @param[in] input float * for input
 * @param[in] gamma float * for gamma multiplier for each row
 * @param[in] result float * for result
 * @param[in] epsilon epsilon to add to each row sum to prevent division by zero
 * @param[in] height height of the tensor
 * @param[in] width width of the tensor
 * @param[in] use_svm whether to treat pointers as SVM
 */
void rmsnorm_cl(const float *input, const float *gamma, float *result,
                const float epsilon, unsigned int height, unsigned int width,
                const bool use_svm = true);

/**
 * @brief     sscal value element by element immediately
 * @param[in] X float * input
 * @param[in] N unsigned int number of elements
 * @param[in] alpha float multiplier
 * @param[in] context RunLayerContext reference
 */
void sscal_cl(float *X, const unsigned int N, const float alpha);

/**
 * @brief     transpose computation
 * @param[in] input float * for Input Tensor
 * @param[in] res float * for Output Tensor
 * @param[in] input_batch_size  represents the number of samples in the input
 * tensor
 * @param[in] input_channels   represents the channels of the input tensor
 * @param[in] input_height   represents the height of the input tensor
 * @param[in] input_width   represents the width of the input tensor
 * @param[in] axis   transpose about axis, 0-> channels & height, 1-> height &
 * width, 2-> channels & width
 */
void transpose_cl_axis(const float *in, float *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis);
/**
 * @brief  Separate the quantized bits and scale from block_q4_0
 *
 * @param src source pointer to the block_q4_0 data
 * @param dst_q destination pointer for the quantized bits
 * @param dst_d destination pointer for the scale
 * @param num_blocks number of blocks to process
 */
void flatten_block_q4_0_cl(const void *src, void *dst_q, void *dst_d,
                           unsigned int num_blocks);

/**
 * @brief Restore the original block_q4_0 from the quantized bits and scale
 *
 * @param src_q source pointer to the quantized bits
 * @param src_d source pointer to the scale
 * @param dst destination pointer for the restored block_q4_0
 * @param num_blocks number of blocks to process
 */
void restore_block_q4_0_cl(const void *src_q, const void *src_d, void *dst,
                           unsigned int num_blocks);

/**
 * @brief This kernel load & store a 4x4 tile of elements
 *
 * @param data Input FP32 matrix data
 * @param M width (row)
 * @param K height (col)
 *
 * @note This kernel is only used for activations
 * Activation is coverted to FP16 and adds zero padding for non multiple of 8
 * Output is not returned and instead saved to outBufferB
 */
void transpose_32_16(float *data, int M, int K);

/**
 * @brief This kernel transpose fp16 type
 *
 * @param data input fp16 matrix data
 * @param output output fp16 matrix data
 * @param width widh
 * @param height height
 * @param size_bytes data size in bytes
 *
 * @note Temporary disable transpose 16
 */
// void transpose_16(void *data, void *output, int width, int height,
//                   int size_bytes, bool isQuant = false);

#ifdef ENABLE_FP16

/**
 * @brief     fp16 sgemv computation : Y = A*X + Y
 * @param[in] matAdata fp16 * for Matrix A
 * @param[in] vecXdata fp16 * for Vector X
 * @param[in] vecYdata fp16 * for Vector Y
 * @param[in] transA bool transpose
 * @param[in] dim1 number of A's columns
 * @param[in] dim2 number of A's rows
 * @param[in] lda number of X's columns
 * @param[in] context RunLayerContext reference
 */
void sgemv_cl(const _FP16 *matAdata, const _FP16 *vecXdata, _FP16 *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda);

/**
 * @brief     fp16 dot computation : sum of all X * Y
 * @param[in] vecAdata fp16 * for Vector A
 * @param[in] vecXdata fp16 * for Vector X
 * @param[in] dim1 number of elements in both input vectors
 * @param[in] context RunLayerContext reference
 * @return    fp16 dot product result
 */
_FP16 dot_cl(const _FP16 *vecAdata, const _FP16 *vecXdata, unsigned int dim1);

/**
 * @brief     fp16 sgemm computation : Y = op(A)*op(B) + C,
 * where op(X) is one of X or X**T
 * @param[in] transA bool transpose
 * @param[in] transB bool transpose
 * @param[in] A fp16 * for Matrix A
 * @param[in] B fp16 * for Matrix B
 * @param[in] C fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] lda number of A's columns
 * @param[in] ldb number of B's columns
 * @param[in] ldc number of C's columns
 * @param[in] context RunLayerContext reference
 */
void sgemm_cl(bool TransA, bool TransB, const _FP16 *A, const _FP16 *B,
              _FP16 *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc);

/**
 * @brief     fp16 addition : sum of all input vectors
 * @param[in] input fp16 * for input
 * @param[in] res fp16 * for result/output
 * @param[in] size_input number of elements in input vector
 * @param[in] size_res number of elements in result vector
 */
void addition_cl(const _FP16 *input, _FP16 *res, unsigned int size_input,
                 unsigned int size_res);

/**
 * @brief     fp16 sscal value element by element immediately
 * @param[in] X _FP16 * input
 * @param[in] N unsigned int number of elements
 * @param[in] alpha float multiplier
 * @param[in] context RunLayerContext reference
 */
void sscal_cl(_FP16 *X, const unsigned int N, const float alpha);

/**
 * @brief     transpose computation
 * @param[in] input fp16 * for Input Tensor
 * @param[in] res fp16 * for Output Tensor
 * @param[in] input_batch_size  represents the number of samples in the input
 * tensor
 * @param[in] input_channels   represents the channels of the input tensor
 * @param[in] input_height   represents the height of the input tensor
 * @param[in] input_width   represents the width of the input tensor
 * @param[in] axis   transpose about axis, 0-> channels & height, 1-> height &
 * width, 2-> channels and width
 */
void transpose_cl_axis(const _FP16 *in, _FP16 *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis);
#endif

/**
 * @brief v8c int8 act × int4(channel-wise QINT4, offset-encoded) GEMM
 *        (paper-aligned 8/4/4 prefill, validated 87% of Adreno 830 dp4a peak).
 * @param[in] act_image image2d_from_buffer view over int8 act buffer
 *            (CL_RGBA UINT32, width=K/16, height=M)
 * @param[in] weight_image image2d_from_buffer view over int4-offset weight buf
 *            (CL_RGBA UINT32, width=K/32, height=N)
 * @param[in] scale_act per-row fp32 act scale buffer [M]
 * @param[in] scale_wgt per-channel fp32 weight scale buffer [N]
 * @param[in] row_sum_act per-row int32 sum of int8 acts [M] (paper §3.7 quant-kernel output)
 * @param[out] output_fp16 fp16 output buffer [M*N]
 * @param[in] M,N,K shape; K must be multiple of 32
 */
void gemm_int8_v8c_cl(cl_mem act_image, cl_mem weight_image, cl_mem scale_act,
                      cl_mem scale_wgt, cl_mem row_sum_act, cl_mem output_fp16,
                      unsigned int M, unsigned int N, unsigned int K);

/**
 * @brief paper §3.7 activation quant kernel for v8c: fp16/fp32 → int8
 *        + per-row scale + per-row int32 sum.
 * @param[in] act_fp16 or act_fp32 input buffer [M*K]
 * @param[out] out_int8 [M*K] int8 (row-major; later wrapped in image2d view)
 * @param[out] out_scale [M] fp32 per-row scale
 * @param[out] out_row_sum [M] int32 sum_k(int8_value), for v8c bias correction
 * @param[in] M,K shape; K must be multiple of 4 (no other constraint here)
 */
void quantize_act_v8c_fp16_cl(cl_mem act_fp16, cl_mem out_int8, cl_mem out_scale,
                              cl_mem out_row_sum, unsigned int M, unsigned int K);
void quantize_act_v8c_fp32_cl(cl_mem act_fp32, cl_mem out_int8, cl_mem out_scale,
                              cl_mem out_row_sum, unsigned int M, unsigned int K);

} // namespace nntrainer

#include "cl_tensor_view.h"
#include <memory>
namespace nntrainer {
/**
 * @brief Convert a channel-wise QINT4 weight (Int4QTensor osv32_isv2 + fp16
 *        per-group scales) into a v8c-ready backing: row-major + offset-encoded
 *        int4 in a single cl_mem buffer (image2d view created on demand via
 *        TensorBacking::imageView), plus a fp32 per-channel scale cl_mem.
 *        Paper §4.2 alignment: re-quantize from per-group (32) → per-channel
 *        (one scale per output row) during the conversion. ONE-TIME at FC init.
 * @param[in] osv32_packed   pointer to osv32 packed int4 bytes (N*K/2 bytes)
 * @param[in] fp16_scales    pointer to fp16 scales (N*K/group_size values)
 * @param[in] group_size     32 (Int4QTensor default)
 * @param[in] N              output channels
 * @param[in] K              input dim
 * @param[out] out_scale_buf cl_mem (fp32, [N], CL_MEM_READ_ONLY) — caller owns
 * @return TensorBacking holding the v8c row-major+offset weight buffer
 *         (Encoding::INT4_OFFSET, Layout::ROW_MAJOR, bytes = N*K/2)
 */
std::unique_ptr<tv::TensorBacking>
make_v8c_weight_backing(const uint8_t *osv32_packed,
                        const uint16_t *fp16_scales, unsigned int group_size,
                        unsigned int N, unsigned int K,
                        cl_mem *out_scale_buf);

} // namespace nntrainer
#endif /* __BLAS_KERNELS_H__ */

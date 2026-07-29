// SPDX-License-Identifier: Apache-2.0
/**
 * @file	int4_utils.h
 * @date	15 October 2025
 * @brief	This is Int4Utils class for utils for INT4 quantization format.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Grzegorz Kisala <gkisala@gmail.com>
 * @bug		No known bugs
 */

#ifndef __NNTRAINER_INT4_UTILS_H__
#define __NNTRAINER_INT4_UTILS_H__

#include <algorithm>
#include <cstdint>
#include <vector>

namespace nntrainer {

/**
 * @class Int4Utils class
 * @brief Int4Utils class with helpers for 4-bit integers calculation,
 * quantization and dequantization methods for osv32_isv2 layout of data
 */
class Int4Utils {
public:
  /// @brief Block size used in the osv32_isv2 layout
  static constexpr const size_t ROW_BLOCK_SIZE = 32;

  /// @brief Numbers of element in one byte of date in the osv32_isv2 layout
  static constexpr const size_t COLUMN_BLOCK_SIZE = 2;

  /// @brief KAI qsi4cxp packing constants for the {nr=4, kr=16, sr=2}
  ///        qai8dxp/qsi4cxp pack family (matmul_clamp_f32_qai8dxp4x8_
  ///        qsi4cxp4x8_4x4x32_neon_i8mm and the fp16 16x4 variant share this
  ///        family, so one rhs_packed buffer serves both).
  static constexpr const size_t KAI_NR = 4;
  static constexpr const size_t KAI_KR = 16;
  static constexpr const size_t KAI_SR = 2;
  static constexpr const size_t KAI_K_INTERLEAVE = 16;
  static constexpr const size_t KAI_K_PAD_MULTIPLE = 32;

  /**
   * @brief     Compute scale for input weights
   * @param[in] group_weights float * inout vector of weights
   * @param[in] group_size group size (32 or 64 or 128)
   * @return computed scale
   */
  static float computeScaleForGroup(const float *group_weights,
                                    const size_t group_size);

  /**
   * @brief     Compute scales for float* matrix weghts
   * @param[in] weights float * input matrix
   * @param[in] rows_count number of rows of input matrix
   * @param[in] columns_count number of columns of input matrix
   * @param[in] group_size group size (32 or 64 or 128)
   * @param[out] scales float vector output scales
   */
  static void computeScales(const float *weights, const size_t rows_count,
                            const size_t columns_count, const size_t group_size,
                            std::vector<float> &scales);

  /**
   * @brief     Pack one weight from position (row_id, column_id) into 4-bits
   * value
   * @param[in] weights float * input matrix
   * @param[in] scales float * input vector os scales
   * @param[in] row_id number of row
   * @param[in] column_id number of column
   * @param[in] groups_per_row number of groups pre row
   * @param[in] group_size group size (32 or 64 or 128)
   * @param[in] rows_count number of rows of input matrix
   * @param[in] columns_count number of columns of input matrix
   * @return
   */
  static uint8_t pack(const float *weights, const float *scales,
                      const size_t row_id, const size_t column_id,
                      const size_t groups_per_row, const size_t group_size,
                      const size_t rows_count, const size_t columns_count);

  /**
   * @brief Quantize weights float* matrix to OpenVINO layout:
   * OS_IS_YX_OSV32_ISV2, osv32_isv2 layout for int4 packed weight:
   *
   * y0_x0x1 | y1_x0x1 | ....  | y15_x0x1|| y16_x0x1 | y17_x0x1 | ... | y31_x0x1
   * y0_x2x3 | y1_x2x3 | ....  | y15_x2x3|| y16_x2x3 | y17_x2x3 | ... | y31_x2x3
   * ...
   * @param weights float * input matrix
   * @param rows_count number of rows of input matrix
   * @param columns_count number of columns of input matrix
   * @param group_size group size (32 or 64 or 128)
   * @param out_weights output quantized weights in layout osv**_isv2
   * @param out_scales output scales
   */
  static void quantizeAndRepack(const float *weights, const size_t rows_count,
                                const size_t columns_count,
                                const size_t group_size,
                                std::vector<uint8_t> &out_weights,
                                std::vector<uint16_t> &out_scales);

  /**
   * @brief Quantize float* matrix into the plain per-channel int4 nibble
   *        form: N x ceil(K/2) bytes, row-major, even k in the low nibble,
   *        each stored uint4 = int4 + 8. Scale is per-channel absmax/7
   *        (the same formula quantizeToInt4 expects). This is stage 1 of
   *        quantizeAndPackKai.
   * @param out_plain_nibbles N x ceil(K/2) bytes, row-major
   * @param out_scales per-channel fp16 scales, size = rows_count
   */
  static void quantizePlain(const float *weights, const size_t rows_count,
                            const size_t columns_count,
                            std::vector<uint8_t> &out_plain_nibbles,
                            std::vector<uint16_t> &out_scales);

  /**
   * @brief Permute plain per-channel int4 nibbles (quantizePlain's output)
   *        into the KAI qsi4cxp Section A super-row payload: mirrors the
   *        rhs_zero_point=8 path of kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0
   *        (nntrainer/tensor/cpu_backend/arm/kai_interface/kai/pack/) with
   *        the trailer (sums, scales, bias) elided — assembleKaiRhsPacked
   *        reconstructs that separately. This is stage 2 of
   *        quantizeAndPackKai.
   * @param out_section_a caller-provided buffer of
   *        kaiNibblePayloadBytes(rows_count, columns_count) bytes
   */
  static void packPlainToSectionA(const uint8_t *plain_nibbles,
                                  size_t rows_count, size_t columns_count,
                                  uint8_t *out_section_a);

  /**
   * @brief Quantize float* matrix into KAI qsi4cxp Section A nibble payload
   *        (quantizePlain + packPlainToSectionA combined). No sums/scales/
   *        bias trailer is written — assembleKaiRhsPacked reassembles that
   *        at load time from the nibbles + per-channel scales this emits.
   * @param weights float * input matrix (rows_count x columns_count)
   * @param rows_count N (output channels)
   * @param columns_count K (input channels)
   * @param out_weights output nibble payload, size
   *        = ceil(N/KAI_NR) * KAI_NR * (roundup(K, KAI_K_PAD_MULTIPLE) / 2)
   * @param out_scales output per-channel fp16 scales, size = rows_count
   */
  static void quantizeAndPackKai(const float *weights, const size_t rows_count,
                                 const size_t columns_count,
                                 std::vector<uint8_t> &out_weights,
                                 std::vector<uint16_t> &out_scales);

  /**
   * @brief Convenience: byte size of the KAI Section A nibble payload for
   *        the given (N, K) shape.
   */
  static size_t kaiNibblePayloadBytes(size_t rows_count, size_t columns_count);

  /**
   * @brief Reassemble Section A nibbles (quantizeAndPackKai's output) +
   *        per-channel fp16 scales into a full KAI rhs_packed buffer, ready
   *        for the qai8dxp_qsi4cxp matmul micro-kernels. Per super-row of
   *        nr=4 output channels the layout is
   *          [nibbles : 4*(k_internal/2) bytes (copied as-is)]
   *          [sums    : 4 * int32, each = sum_k int4[n][k] * 16]
   *          [scales  : 4 * fp32,  each = (fp16->fp32 scale) * 0.0625]
   *          [bias    : 4 * fp32 = 0]
   *
   * @param section_a       nibble payload; size must be
   *                        kaiNibblePayloadBytes(rows_count, columns_count).
   * @param fp16_scales     per-output-channel fp16 scales, size N.
   * @param rows_count      N (output channels). Must be a multiple of 4.
   * @param columns_count   K (input channels). Must be a multiple of 32.
   * @param out_kai_packed  output buffer, sized by this function.
   */
  static void assembleKaiRhsPacked(const uint8_t *section_a,
                                   const uint16_t *fp16_scales,
                                   size_t rows_count, size_t columns_count,
                                   std::vector<uint8_t> &out_kai_packed);

  /**
   * @brief     Quantize one float value to 4-bits integer
   * @param[in] weight input weight
   * @param[in] scale input scale
   * @return 4-bit integer
   */
  static uint8_t quantizeToInt4(const float weight, const float scale);

  /**
   * @brief     Convert 4-bit integer value to 32-bit integer
   * @param[in] int4_value input 4-bit signed integer value
   * @return output int value
   */
  static int convertInt4ToInt(const uint8_t int4_value);

  /**
   * @brief     Dequantize weights in osv32_isv2 layout and scales to float
   * weights
   * @param[in] weights input matrix with quantized weights in osv32_isv2 layout
   * @param[in] scales fp16 vector input scales
   * @param[in] rows_count number of rows of data
   * @param[in] columns_count number of columns of data
   * @param[in] group_size group size (32 or 64 or 128)
   * @param[out] dequantized_weights float vector of dequantized_weights
   */
  static void dequantizePacked(const std::vector<uint8_t> &weights,
                               const std::vector<uint16_t> &scales,
                               const size_t rows_count,
                               const size_t columns_count,
                               const size_t group_size,
                               std::vector<float> &dequantized_weights);

  /**
   * @brief Dequantize weights in osv32_isv2 layout by row
   *
   * @param weights quantized weights in osv32_isv2 layout
   * @param scales fp16 scales
   * @param rows_count number of rows of data
   * @param columns_count number of columns of data
   * @param group_size group size (32 or 64 or 128)
   * @param row_index row index to dequantize
   * @param dequantized_row dequantized_weights
   */
  static void dequantizePackedRow(uint8_t *weights, uint16_t *scales,
                                  const size_t rows_count,
                                  const size_t columns_count,
                                  const size_t group_size,
                                  const size_t row_index,
                                  float *dequantized_row);

  /**
   * @brief Dequantize weights in osv32_isv2 layout by row
   *
   * @param weights quantized weights in osv32_isv2 layout
   * @param scales fp16 scales
   * @param rows_count number of rows of data
   * @param columns_count number of columns of data
   * @param group_size group size (32 or 64 or 128)
   * @param row_index row index to dequantize
   * @param column_index column start index
   * @param weight_int4_row32 output 32xint4 (16 bytes)
   * @param scale output scale
   */
  static void dequantizePackedRow32ToInt4Scale(
    const uint8_t *weights, const uint16_t *scales, const size_t rows_count,
    const size_t columns_count, const size_t group_size, const size_t row_index,
    const size_t column_index, uint8_t *weight_int4_row32, uint16_t *scale);
};

} // namespace nntrainer

#endif // __NNTRAINER_INT4_UTILS_H__

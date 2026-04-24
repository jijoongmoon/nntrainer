// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   safetensors_util.h
 * @date   24 April 2026
 * @brief  Minimal helpers for the safetensors on-disk format: dtype
 *         mapping, JSON header builder, JSON header parser.
 * @see    https://github.com/huggingface/safetensors
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __SAFETENSORS_UTIL_H__
#define __SAFETENSORS_UTIL_H__
#ifdef __cplusplus

#include <string>
#include <tensor_dim.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nntrainer::safetensors {

/**
 * @brief One tensor entry as it appears in the safetensors JSON header.
 */
struct TensorEntry {
  std::string name;          /**< weight name, becomes the JSON key */
  std::string dtype;         /**< safetensors dtype code, e.g. "F32" */
  std::vector<size_t> shape; /**< tensor shape, most-significant first */
  size_t offset_start;       /**< byte offset from data section start */
  size_t offset_end;         /**< byte offset (exclusive) */
};

/**
 * @brief Map an nntrainer data type to its safetensors dtype code.
 *        Unknown / NONE types fall back to "F32".
 */
const char *dtypeToString(ml::train::TensorDim::DataType dtype);

/**
 * @brief Build the JSON header for a set of tensor entries. The returned
 *        string is padded with spaces so its length is a multiple of 8,
 *        matching the safetensors alignment convention.
 */
std::string buildHeader(const std::vector<TensorEntry> &entries);

/**
 * @brief Parse a safetensors JSON header and return a name -> (offset, size)
 *        map, where offset is the byte offset from the start of the data
 *        section and size is the number of bytes of that tensor's payload.
 *        The "__metadata__" key (if present) is skipped.
 */
std::unordered_map<std::string, std::pair<size_t, size_t>>
parseHeader(const std::string &json);

} // namespace nntrainer::safetensors

#endif /* __cplusplus */
#endif /* __SAFETENSORS_UTIL_H__ */

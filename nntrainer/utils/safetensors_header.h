// SPDX-License-Identifier: Apache-2.0
/**
 * @file   safetensors_header.h
 * @date   14 May 2026
 * @brief  Minimal header parser for HuggingFace .safetensors files.
 *
 *         The safetensors v0.4 binary layout is:
 *
 *             <u64 LE header_size>
 *             <UTF-8 JSON header of header_size bytes>
 *             <concatenated raw tensor bytes>
 *
 *         The JSON header is a flat dictionary:
 *
 *             {
 *               "tensor.name": {
 *                 "dtype": "F32" | "F16" | "BF16" | "I8" | "U8" | ...,
 *                 "shape": [d0, d1, ...],
 *                 "data_offsets": [start_byte, end_byte]
 *               },
 *               "__metadata__": { ... }     // optional
 *             }
 *
 *         This file provides a stand-alone parser used by both the model
 *         loader (for per-tensor dtype + offset lookup) and by unit tests.
 *         No external JSON dependency: the parser is hand-rolled and only
 *         handles the safetensors header shape (no nested objects beyond
 *         the top-level entries, no escape-heavy strings).
 *
 * @see    https://github.com/huggingface/safetensors
 */

#ifndef __NNTRAINER_SAFETENSORS_HEADER_H__
#define __NNTRAINER_SAFETENSORS_HEADER_H__
#ifdef __cplusplus

#include <cstdint>
#include <map>
#include <string>
#include <tensor_dim.h>
#include <vector>

namespace nntrainer {

/** Bring the ccapi TensorDim into the nntrainer namespace (mirrors the
 *  using-declaration in tensor_base.h / base_properties.h). */
using TensorDim = ml::train::TensorDim;

/**
 * @brief One tensor's metadata as recorded in a safetensors header.
 */
struct SafetensorsTensorInfo {
  /** Parsed nntrainer dtype. May be NONE if the safetensors dtype is one we
   *  don't translate (e.g. I32 / BOOL) — callers must check before consuming. */
  TensorDim::DataType dtype = TensorDim::DataType::NONE;
  /** Raw dtype string straight out of the JSON header (e.g. "F32"). Useful
   *  for diagnostics when @c dtype is NONE. */
  std::string dtype_raw;
  /** Tensor shape, outermost first (numpy convention). */
  std::vector<uint64_t> shape;
  /** Byte offsets into the data block that follows the header. */
  uint64_t data_offset_start = 0;
  uint64_t data_offset_end = 0;
};

/**
 * @brief Parsed top-level header view of a .safetensors file.
 */
struct SafetensorsHeader {
  /** Length (in bytes) of the JSON header, taken from the leading u64 LE. */
  uint64_t header_size = 0;
  /** Byte offset where the data block begins inside the file
   *  (= 8 + header_size). All @c data_offset_start/end values are relative
   *  to this. */
  uint64_t data_block_offset = 0;
  /** name -> tensor info map. Preserves insertion order via the JSON parse
   *  walk; std::map gives ordered iteration on top of that. */
  std::map<std::string, SafetensorsTensorInfo> tensors;
  /** Free-form key/value metadata under "__metadata__", if present. */
  std::map<std::string, std::string> metadata;
};

/**
 * @brief Parse a .safetensors file's header from a path on disk.
 *
 * Only the leading <u64 header_size><JSON> bytes are read. The bulk
 * data block is left untouched — callers that need a tensor's bytes
 * should seek to @c data_block_offset + tensor.data_offset_start.
 *
 * @throws std::runtime_error on I/O failure, malformed JSON or sizes that
 *         exceed the file length.
 */
SafetensorsHeader parseSafetensorsHeaderFromFile(const std::string &path);

/**
 * @brief Same as above but consumes an in-memory blob. Useful for tests
 *        and for callers that have already memory-mapped the file.
 *
 * @param data pointer to the start of the file
 * @param size total size in bytes of the file (>= 8 + header_size)
 */
SafetensorsHeader parseSafetensorsHeader(const void *data, size_t size);

/**
 * @brief Translate a safetensors dtype token to the closest
 *        nntrainer TensorDim::DataType. Returns DataType::NONE on
 *        anything unrecognised (caller decides whether to error).
 *
 *        Mapping (the only ones nntrainer can currently consume):
 *            "F32"    -> FP32
 *            "F16"    -> FP16
 *            "I8"     -> QINT8
 *            "U8"     -> UINT8
 *            "U16"    -> UINT16
 *            "U32"    -> UINT32
 *
 *        Anything else (BF16, F64, I32, BOOL, custom Q*_0 prefixed) is
 *        returned as NONE; the converter pipeline should dequantise or
 *        re-quantise such tensors before handing them to nntrainer.
 */
TensorDim::DataType safetensorsDtypeToNntrainer(const std::string &raw);

/**
 * @brief Same mapping as @c safetensorsDtypeToNntrainer but returns the
 *        nntrainer property-format string ("FP32", "FP16", "QINT8", ...)
 *        ready to be plugged into the `weight_dtype_map` property string
 *        ("weight:FP32,bias:FP16").
 *
 *        Returns the empty string for unrecognised safetensors tokens.
 */
std::string safetensorsDtypeToNntrainerString(const std::string &raw);

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NNTRAINER_SAFETENSORS_HEADER_H__ */

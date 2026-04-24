// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   safetensors_util.cpp
 * @date   24 April 2026
 * @brief  safetensors dtype mapping + JSON header build/parse helpers.
 * @see    https://github.com/huggingface/safetensors
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "safetensors_util.h"

#include <cctype>
#include <sstream>
#include <stdexcept>

namespace nntrainer::safetensors {

const char *dtypeToString(ml::train::TensorDim::DataType dtype) {
  using DT = ml::train::TensorDim::DataType;
  switch (dtype) {
  case DT::FP32:
    return "F32";
  case DT::FP16:
    return "F16";
  case DT::QINT4:
    return "I4";
  case DT::QINT8:
    return "I8";
  case DT::QINT16:
    return "I16";
  case DT::UINT4:
    return "U4";
  case DT::UINT8:
    return "U8";
  case DT::UINT16:
    return "U16";
  case DT::UINT32:
    return "U32";
  default:
    return "F32";
  }
}

std::string buildHeader(const std::vector<TensorEntry> &entries) {
  std::ostringstream out;
  out << "{\"__metadata__\":{\"format\":\"nntrainer\"}";
  for (const auto &e : entries) {
    out << ",\"" << e.name << "\":{\"dtype\":\"" << e.dtype << "\",\"shape\":[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i > 0)
        out << ",";
      out << e.shape[i];
    }
    out << "],\"data_offsets\":[" << e.offset_start << "," << e.offset_end
        << "]}";
  }
  out << "}";

  std::string s = out.str();
  // Pad to an 8-byte boundary so the data section is aligned.
  const size_t pad = (8 - (s.size() % 8)) % 8;
  s.append(pad, ' ');
  return s;
}

namespace {

/**
 * @brief Minimal forward-only JSON scanner tailored to safetensors headers:
 *        only handles the shapes the safetensors spec actually uses
 *        (objects, arrays of integers, strings, integers).
 */
class Scanner {
public:
  explicit Scanner(const std::string &s) : src(s), pos(0) {}

  void skipWs() {
    while (pos < src.size() &&
           std::isspace(static_cast<unsigned char>(src[pos])))
      ++pos;
  }

  bool peek(char c) {
    skipWs();
    return pos < src.size() && src[pos] == c;
  }

  void expect(char c) {
    if (!peek(c))
      throw std::runtime_error(std::string("safetensors header: expected '") +
                               c + "'");
    ++pos;
  }

  std::string readString() {
    expect('"');
    std::string out;
    while (pos < src.size() && src[pos] != '"') {
      if (src[pos] == '\\' && pos + 1 < src.size())
        ++pos;
      out += src[pos++];
    }
    expect('"');
    return out;
  }

  size_t readNumber() {
    skipWs();
    size_t v = 0;
    while (pos < src.size() && src[pos] >= '0' && src[pos] <= '9')
      v = v * 10 + (src[pos++] - '0');
    return v;
  }

  /// Skip any JSON value starting at the current position.
  void skipValue() {
    skipWs();
    if (pos >= src.size())
      return;
    char c = src[pos];
    if (c == '"') {
      readString();
      return;
    }
    if (c == '{' || c == '[') {
      const char close = (c == '{') ? '}' : ']';
      int depth = 0;
      while (pos < src.size()) {
        if (src[pos] == '"') {
          readString();
          continue;
        }
        if (src[pos] == c)
          ++depth;
        else if (src[pos] == close && --depth == 0) {
          ++pos;
          return;
        }
        ++pos;
      }
      return;
    }
    // number / literal
    while (pos < src.size() && src[pos] != ',' && src[pos] != '}' &&
           src[pos] != ']')
      ++pos;
  }

private:
  const std::string &src;
  size_t pos;
};

} // namespace

std::unordered_map<std::string, std::pair<size_t, size_t>>
parseHeader(const std::string &json) {
  std::unordered_map<std::string, std::pair<size_t, size_t>> out;
  Scanner s(json);

  s.expect('{');
  for (bool first = true; !s.peek('}'); first = false) {
    if (!first)
      s.expect(',');

    const std::string key = s.readString();
    s.expect(':');

    if (key == "__metadata__") {
      s.skipValue();
      continue;
    }

    // Tensor descriptor: { "dtype": "...", "shape": [...], "data_offsets": [s,
    // e] }
    s.expect('{');
    size_t off_start = 0, off_end = 0;
    bool have_offsets = false;
    for (bool inner_first = true; !s.peek('}'); inner_first = false) {
      if (!inner_first)
        s.expect(',');
      const std::string field = s.readString();
      s.expect(':');
      if (field == "data_offsets") {
        s.expect('[');
        off_start = s.readNumber();
        s.expect(',');
        off_end = s.readNumber();
        s.expect(']');
        have_offsets = true;
      } else {
        s.skipValue();
      }
    }
    s.expect('}');

    if (have_offsets)
      out.emplace(key, std::make_pair(off_start, off_end - off_start));
  }
  s.expect('}');

  return out;
}

} // namespace nntrainer::safetensors

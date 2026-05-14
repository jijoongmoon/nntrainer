// SPDX-License-Identifier: Apache-2.0
/**
 * @file   safetensors_header.cpp
 * @date   14 May 2026
 * @brief  Minimal header parser for HuggingFace .safetensors files.
 *
 *         The safetensors JSON header is small (entry per tensor, no nested
 *         objects beyond top-level), so this file hand-rolls a focused
 *         parser instead of taking a JSON dependency. Edge cases that would
 *         normally bite a tiny JSON parser are absent in the safetensors
 *         spec: keys are always quoted plain strings (no embedded quotes
 *         the spec doesn't require), values are either strings, integers
 *         or arrays of those, and there is no whitespace-sensitive content.
 */

#include "safetensors_header.h"

#include <cctype>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace nntrainer {

// ---------------------------------------------------------------------------
// Public dtype mapping
// ---------------------------------------------------------------------------
TensorDim::DataType safetensorsDtypeToNntrainer(const std::string &raw) {
  if (raw == "F32")
    return TensorDim::DataType::FP32;
  if (raw == "F16")
    return TensorDim::DataType::FP16;
  if (raw == "I8")
    return TensorDim::DataType::QINT8;
  if (raw == "U8")
    return TensorDim::DataType::UINT8;
  if (raw == "U16")
    return TensorDim::DataType::UINT16;
  if (raw == "U32")
    return TensorDim::DataType::UINT32;
  return TensorDim::DataType::NONE;
}

std::string safetensorsDtypeToNntrainerString(const std::string &raw) {
  if (raw == "F32")
    return "FP32";
  if (raw == "F16")
    return "FP16";
  if (raw == "I8")
    return "QINT8";
  if (raw == "U8")
    return "UINT8";
  if (raw == "U16")
    return "UINT16";
  if (raw == "U32")
    return "UINT32";
  return "";
}

namespace {

// Single-pass JSON walker over the safetensors header. State is just an
// index into the buffer; the public entry point recursively pulls keys,
// strings, ints and arrays until the closing brace.

struct Cursor {
  const char *buf;
  size_t pos;
  size_t end;
};

static void skipWs(Cursor &c) {
  while (c.pos < c.end) {
    unsigned char ch = static_cast<unsigned char>(c.buf[c.pos]);
    if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')
      ++c.pos;
    else
      break;
  }
}

static void expect(Cursor &c, char ch) {
  skipWs(c);
  if (c.pos >= c.end || c.buf[c.pos] != ch) {
    throw std::runtime_error("safetensors header: expected '" +
                             std::string(1, ch) + "' at offset " +
                             std::to_string(c.pos));
  }
  ++c.pos;
}

static std::string parseString(Cursor &c) {
  skipWs(c);
  if (c.pos >= c.end || c.buf[c.pos] != '"') {
    throw std::runtime_error("safetensors header: expected string at offset " +
                             std::to_string(c.pos));
  }
  ++c.pos;
  std::string out;
  while (c.pos < c.end && c.buf[c.pos] != '"') {
    if (c.buf[c.pos] == '\\' && c.pos + 1 < c.end) {
      // Minimal escape handling: \", \\, \/, \n, \t, \r — enough for the
      // safetensors spec which only uses tensor names and dtype tokens.
      char e = c.buf[c.pos + 1];
      switch (e) {
      case '"':  out.push_back('"');  break;
      case '\\': out.push_back('\\'); break;
      case '/':  out.push_back('/');  break;
      case 'n':  out.push_back('\n'); break;
      case 't':  out.push_back('\t'); break;
      case 'r':  out.push_back('\r'); break;
      default:   out.push_back(e);    break;
      }
      c.pos += 2;
    } else {
      out.push_back(c.buf[c.pos++]);
    }
  }
  if (c.pos >= c.end) {
    throw std::runtime_error(
      "safetensors header: unterminated string");
  }
  ++c.pos; // closing quote
  return out;
}

static uint64_t parseUInt(Cursor &c) {
  skipWs(c);
  uint64_t v = 0;
  if (c.pos >= c.end || !std::isdigit(static_cast<unsigned char>(c.buf[c.pos]))) {
    throw std::runtime_error("safetensors header: expected integer at offset " +
                             std::to_string(c.pos));
  }
  while (c.pos < c.end && std::isdigit(static_cast<unsigned char>(c.buf[c.pos]))) {
    v = v * 10 + static_cast<uint64_t>(c.buf[c.pos] - '0');
    ++c.pos;
  }
  return v;
}

static std::vector<uint64_t> parseUIntArray(Cursor &c) {
  std::vector<uint64_t> out;
  expect(c, '[');
  skipWs(c);
  if (c.pos < c.end && c.buf[c.pos] == ']') {
    ++c.pos;
    return out;
  }
  while (true) {
    out.push_back(parseUInt(c));
    skipWs(c);
    if (c.pos < c.end && c.buf[c.pos] == ',') {
      ++c.pos;
      continue;
    }
    break;
  }
  expect(c, ']');
  return out;
}

static SafetensorsTensorInfo parseTensorInfo(Cursor &c) {
  SafetensorsTensorInfo info;
  expect(c, '{');
  while (true) {
    skipWs(c);
    if (c.pos < c.end && c.buf[c.pos] == '}') {
      ++c.pos;
      break;
    }
    std::string key = parseString(c);
    expect(c, ':');
    if (key == "dtype") {
      info.dtype_raw = parseString(c);
      info.dtype = safetensorsDtypeToNntrainer(info.dtype_raw);
    } else if (key == "shape") {
      info.shape = parseUIntArray(c);
    } else if (key == "data_offsets") {
      auto offs = parseUIntArray(c);
      if (offs.size() != 2) {
        throw std::runtime_error(
          "safetensors header: data_offsets must have 2 entries (got " +
          std::to_string(offs.size()) + ")");
      }
      info.data_offset_start = offs[0];
      info.data_offset_end = offs[1];
    } else {
      // Unknown key — skip its value defensively. The spec doesn't
      // require this but it keeps us forward-compatible.
      skipWs(c);
      if (c.pos < c.end && c.buf[c.pos] == '"') {
        (void)parseString(c);
      } else if (c.pos < c.end && c.buf[c.pos] == '[') {
        (void)parseUIntArray(c);
      } else {
        (void)parseUInt(c);
      }
    }
    skipWs(c);
    if (c.pos < c.end && c.buf[c.pos] == ',') {
      ++c.pos;
      continue;
    }
    if (c.pos >= c.end || c.buf[c.pos] != '}') {
      throw std::runtime_error(
        "safetensors header: expected ',' or '}' at offset " +
        std::to_string(c.pos));
    }
  }
  return info;
}

static std::map<std::string, std::string> parseMetadata(Cursor &c) {
  std::map<std::string, std::string> out;
  expect(c, '{');
  while (true) {
    skipWs(c);
    if (c.pos < c.end && c.buf[c.pos] == '}') {
      ++c.pos;
      break;
    }
    std::string k = parseString(c);
    expect(c, ':');
    std::string v = parseString(c);
    out[k] = v;
    skipWs(c);
    if (c.pos < c.end && c.buf[c.pos] == ',') {
      ++c.pos;
      continue;
    }
  }
  return out;
}

} // namespace

SafetensorsHeader parseSafetensorsHeader(const void *data, size_t size) {
  if (size < 8) {
    throw std::runtime_error(
      "safetensors: file too small to contain header_size (got " +
      std::to_string(size) + " bytes)");
  }
  const uint8_t *p = static_cast<const uint8_t *>(data);

  // Leading uint64 LE.
  uint64_t header_size = 0;
  for (int i = 0; i < 8; ++i) {
    header_size |= static_cast<uint64_t>(p[i]) << (8 * i);
  }
  if (8 + header_size > size) {
    throw std::runtime_error(
      "safetensors: header_size (" + std::to_string(header_size) +
      ") exceeds file size (" + std::to_string(size) + ")");
  }

  SafetensorsHeader hdr;
  hdr.header_size = header_size;
  hdr.data_block_offset = 8 + header_size;

  Cursor c{reinterpret_cast<const char *>(p + 8), 0, header_size};
  expect(c, '{');
  while (true) {
    skipWs(c);
    if (c.pos < c.end && c.buf[c.pos] == '}') {
      ++c.pos;
      break;
    }
    std::string key = parseString(c);
    expect(c, ':');
    if (key == "__metadata__") {
      hdr.metadata = parseMetadata(c);
    } else {
      hdr.tensors[key] = parseTensorInfo(c);
    }
    skipWs(c);
    if (c.pos < c.end && c.buf[c.pos] == ',') {
      ++c.pos;
      continue;
    }
  }
  return hdr;
}

SafetensorsHeader parseSafetensorsHeaderFromFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    throw std::runtime_error("safetensors: cannot open '" + path + "'");
  }
  std::streamsize sz = f.tellg();
  f.seekg(0, std::ios::beg);
  if (sz < 8) {
    throw std::runtime_error("safetensors: '" + path +
                             "' is too small to be a safetensors file");
  }
  // We only need the header bytes (8 + header_size). Read in two stages so
  // we don't slurp huge model files into RAM just to peek at the metadata.
  uint8_t size_bytes[8];
  if (!f.read(reinterpret_cast<char *>(size_bytes), 8)) {
    throw std::runtime_error("safetensors: short read on header size");
  }
  uint64_t header_size = 0;
  for (int i = 0; i < 8; ++i) {
    header_size |= static_cast<uint64_t>(size_bytes[i]) << (8 * i);
  }
  if (8 + header_size > static_cast<uint64_t>(sz)) {
    throw std::runtime_error("safetensors: header_size (" +
                             std::to_string(header_size) + ") exceeds file " +
                             std::to_string(sz));
  }
  std::vector<uint8_t> buf(8 + header_size);
  std::memcpy(buf.data(), size_bytes, 8);
  if (!f.read(reinterpret_cast<char *>(buf.data() + 8), header_size)) {
    throw std::runtime_error("safetensors: short read on header body");
  }
  return parseSafetensorsHeader(buf.data(), buf.size());
}

} // namespace nntrainer

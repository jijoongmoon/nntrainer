// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   tokenizer_cache.cpp
 * @date   30 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Persistent post-parse tokenizer cache -- see tokenizer_cache.h.
 */

#include "tokenizer_cache.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <stdexcept>

#include <nntrainer_log.h>

namespace causallm {

namespace {

namespace fs = std::filesystem;

/**
 * Cache file layout (little-endian, after which the Rust-side snapshot
 * payload follows verbatim; the payload carries its own magic + version):
 *
 *   offset 0  : magic "NTKC"
 *          4  : u32 header version (bump on ANY layout/semantic change)
 *          8  : u64 source tokenizer.json size          } the (size, mtime,
 *          16 : u64 source tokenizer.json mtime (raw)   } version) cache key
 *          24 : u64 payload length
 *          32 : u64 FNV-1a of the payload (bit-flip / truncation guard)
 *          40 : payload
 */
constexpr char kMagic[4] = {'N', 'T', 'K', 'C'};
constexpr uint32_t kHeaderVersion = 1;
constexpr size_t kHeaderSize = 40;

struct SourceKey {
  uint64_t size;
  uint64_t mtime;
};

uint64_t fnv1a(const void *data, size_t len) {
  const auto *p = static_cast<const unsigned char *>(data);
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < len; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

double nowMs() {
  return std::chrono::duration<double, std::milli>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

bool cacheEnabled() {
  const char *e = std::getenv("NNTR_TOKENIZER_CACHE");
  return e == nullptr || e[0] != '0';
}

bool sourceKey(const std::string &path, SourceKey &out) {
  std::error_code ec;
  const auto size = fs::file_size(path, ec);
  if (ec)
    return false;
  const auto mtime = fs::last_write_time(path, ec);
  if (ec)
    return false;
  out.size = static_cast<uint64_t>(size);
  out.mtime = static_cast<uint64_t>(mtime.time_since_epoch().count());
  return true;
}

/** Primary cache location: next to the tokenizer file. */
std::string primaryCachePath(const std::string &tok_path) {
  return tok_path + ".ntkc";
}

/** Fallback: $XDG_CACHE_HOME/nntrainer/tokenizer (else ~/.cache/...). */
std::string fallbackCachePath(const std::string &tok_path) {
  std::string base;
  if (const char *xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0])
    base = xdg;
  else if (const char *home = std::getenv("HOME"); home && home[0])
    base = std::string(home) + "/.cache";
  else
    return {}; // no stable per-user location on this platform/environment
  std::error_code ec;
  fs::path abs = fs::absolute(tok_path, ec);
  const std::string keysrc = ec ? tok_path : abs.string();
  char name[32];
  std::snprintf(
    name, sizeof(name), "%016llx",
    static_cast<unsigned long long>(fnv1a(keysrc.data(), keysrc.size())));
  return base + "/nntrainer/tokenizer/" + name + ".ntkc";
}

/**
 * Read + fully validate one candidate cache file against the current source
 * key. Returns the payload, or empty on ANY mismatch/short-read/corruption.
 */
std::string readValidatedPayload(const std::string &cache_path,
                                 const SourceKey &key) {
  std::ifstream f(cache_path, std::ios::binary | std::ios::in);
  if (!f.good())
    return {};
  char header[kHeaderSize];
  if (!f.read(header, kHeaderSize))
    return {};
  if (std::memcmp(header, kMagic, 4) != 0)
    return {};
  auto rd_u32 = [&header](size_t off) {
    uint32_t v;
    std::memcpy(&v, header + off, sizeof(v));
    return v;
  };
  auto rd_u64 = [&header](size_t off) {
    uint64_t v;
    std::memcpy(&v, header + off, sizeof(v));
    return v;
  };
  if (rd_u32(4) != kHeaderVersion)
    return {};
  if (rd_u64(8) != key.size || rd_u64(16) != key.mtime)
    return {}; // stale key -> miss -> re-parse + rewrite
  const uint64_t payload_len = rd_u64(24);
  const uint64_t payload_fnv = rd_u64(32);
  // Sanity bound: a payload larger than 1GB is not something we ever wrote.
  if (payload_len == 0 || payload_len > (1ULL << 30))
    return {};
  std::string payload(static_cast<size_t>(payload_len), '\0');
  if (!f.read(payload.data(), static_cast<std::streamsize>(payload_len)))
    return {}; // truncated
  // Trailing bytes after the payload => not a file this version wrote.
  f.peek();
  if (!f.eof())
    return {};
  if (fnv1a(payload.data(), payload.size()) != payload_fnv)
    return {}; // bit-flip
  return payload;
}

/** Atomic-ish write: temp file + rename. Returns true on success. */
bool writeCacheFile(const std::string &cache_path, const SourceKey &key,
                    const std::string &payload) {
  std::error_code ec;
  const fs::path dir = fs::path(cache_path).parent_path();
  if (!dir.empty())
    fs::create_directories(dir, ec); // best-effort; open below decides
  const std::string tmp = cache_path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!f.good())
      return false;
    char header[kHeaderSize];
    std::memcpy(header, kMagic, 4);
    auto wr_u32 = [&header](size_t off, uint32_t v) {
      std::memcpy(header + off, &v, sizeof(v));
    };
    auto wr_u64 = [&header](size_t off, uint64_t v) {
      std::memcpy(header + off, &v, sizeof(v));
    };
    wr_u32(4, kHeaderVersion);
    wr_u64(8, key.size);
    wr_u64(16, key.mtime);
    wr_u64(24, static_cast<uint64_t>(payload.size()));
    wr_u64(32, fnv1a(payload.data(), payload.size()));
    f.write(header, kHeaderSize);
    f.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!f.good()) {
      f.close();
      fs::remove(tmp, ec);
      return false;
    }
  }
  fs::rename(tmp, cache_path, ec);
  if (ec) {
    fs::remove(tmp, ec);
    return false;
  }
  return true;
}

std::string loadFileBytes(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    // Same failure semantics as the parse path (LoadBytesFromFile).
    throw std::runtime_error("Failed to open file: " + path);
  }
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::string buffer(static_cast<size_t>(size), ' ');
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

/**
 * Background cache (re)write. Runs on a side thread with the json bytes
 * moved in, AFTER the parsed tokenizer has been handed to the caller -- a
 * cache miss never adds to the critical path. A partial write is never
 * visible (temp + rename), and even a corrupted rename result is caught by
 * the FNV check on the next read.
 *
 * The worker is held in a function-static std::async future rather than
 * detached: its destructor joins at process exit, so a short-lived process
 * (CLI one-shot) still persists the cache instead of killing the writer
 * mid-cycle (verified: a detached writer's rename never landed when main()
 * returned first). Cost: at most one ~0.5-0.7s teardown wait, only on runs
 * that actually re-wrote the cache.
 */
void scheduleCacheWrite(std::string json, const std::string &tok_path,
                        const SourceKey &key) {
  static std::mutex writer_mtx;
  static std::future<void> writer; // joined on reassignment and at exit
  auto task = [json = std::move(json), tok_path, key]() {
    const double t0 = nowMs();
    const std::string payload = tokenizers::Tokenizer::SnapshotFromJSON(json);
    if (payload.empty()) {
      ml_logd("tokenizer snapshot not cacheable (non-BPE or unknown shape); "
              "parse path stays (%.1f ms spent probing)",
              nowMs() - t0);
      return;
    }
    std::string where = primaryCachePath(tok_path);
    if (!writeCacheFile(where, key, payload)) {
      where = fallbackCachePath(tok_path);
      if (where.empty() || !writeCacheFile(where, key, payload))
        where.clear();
    }
    if (!where.empty())
      ml_logd("tokenizer snapshot written (background, %.1f ms)", nowMs() - t0);
    else
      ml_logd("tokenizer snapshot write failed everywhere (read-only?); "
              "will re-parse next run (%.1f ms)",
              nowMs() - t0);
  };
  std::lock_guard<std::mutex> lk(writer_mtx);
  if (writer.valid())
    writer.wait(); // at most one writer in flight (multi-model processes)
  writer = std::async(std::launch::async, std::move(task));
}

} // namespace

std::unique_ptr<tokenizers::Tokenizer>
LoadTokenizerCached(const std::string &tok_path) {
  SourceKey key{};
  const bool use_cache = cacheEnabled() && sourceKey(tok_path, key);

  if (use_cache) {
    const double t0 = nowMs();
    std::string payload = readValidatedPayload(primaryCachePath(tok_path), key);
    if (payload.empty()) {
      const std::string fb = fallbackCachePath(tok_path);
      if (!fb.empty())
        payload = readValidatedPayload(fb, key);
    }
    if (!payload.empty()) {
      auto tok = tokenizers::Tokenizer::FromSnapshot(payload);
      if (tok != nullptr) {
        ml_logd("tokenizer snapshot HIT, load took %.1f ms (parse avoided)",
                nowMs() - t0);
        return tok;
      }
      ml_logd("tokenizer snapshot rejected by loader after %.1f ms; falling "
              "back to parse",
              nowMs() - t0);
    }
  }

  // Parse path -- identical to the pre-cache behavior.
  const double t0 = nowMs();
  std::string json = loadFileBytes(tok_path);
  auto tok = tokenizers::Tokenizer::FromBlobJSON(json);
  ml_logd("tokenizer parse done, parse took %.1f ms", nowMs() - t0);
  if (use_cache)
    scheduleCacheWrite(std::move(json), tok_path, key);
  return tok;
}

} // namespace causallm

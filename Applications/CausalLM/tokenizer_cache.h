// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   tokenizer_cache.h
 * @date   30 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Persistent post-parse tokenizer cache.
 *
 * Tokenizer::FromBlobJSON on a ~32MB BPE tokenizer.json costs ~600ms on an
 * idle desktop host, and in a synchronous construction path every millisecond
 * of it lands on time-to-first-token. This module caches the expensive part --
 * the parsed vocab/merges tables plus the small component sections -- in a
 * snapshot file so subsequent processes rebuild the tokenizer in ~1/3 the time
 * (measured 597 -> 234 ms).
 *
 * Contract:
 *  - Byte-identical tokenization: the snapshot reconstruction mirrors the
 *    tokenizers crate's own deserialize order (model first, then components,
 *    then added tokens); encode/decode of any prompt equals the parse path.
 *  - Fail-safe: ANY key mismatch (size/mtime), short read, version bump,
 *    checksum or deserialize failure falls back silently to the parse path
 *    (debug log only) and rewrites the cache.
 *  - NNTR_TOKENIZER_CACHE=0 opts out entirely (no read, no write).
 *
 * Cache location: "<tokenizer_file>.ntkc" next to the model; when that
 * directory is not writable, "$XDG_CACHE_HOME/nntrainer/tokenizer/<hash>.ntkc"
 * (fallback "~/.cache/..."). The write happens on a detached thread AFTER the
 * parsed tokenizer has been handed to the caller, so a cache MISS never adds
 * to the critical path.
 */

#ifndef __TOKENIZER_CACHE_H__
#define __TOKENIZER_CACHE_H__

#include <memory>
#include <string>

#include <tokenizers_cpp.h>

namespace causallm {

/**
 * @brief Load a tokenizer through the persistent snapshot cache.
 *
 * On cache hit rebuilds from the snapshot; on miss/stale/corrupt parses the
 * JSON (exactly like Tokenizer::FromBlobJSON) and schedules a background
 * cache (re)write.
 *
 * @param tok_path path to tokenizer.json
 * @return the tokenizer (same success/failure semantics as FromBlobJSON;
 *         file-open failures throw just like the parse path did)
 */
std::unique_ptr<tokenizers::Tokenizer>
LoadTokenizerCached(const std::string &tok_path);

} // namespace causallm

#endif // __TOKENIZER_CACHE_H__

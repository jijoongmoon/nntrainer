// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cuda_pack_cache.h
 * @date   30 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Derive-once disk cache for the CUDA QS4CX derived weight packs.
 *
 * The load-time prewarm (cuda_fc_qs4cx_prewarm) derives two deterministic
 * pure functions of the plain QS4CX nibbles and uploads them to the device:
 *
 *   kind "dp4a": packed signed int4 [N][(K+1)/2] (byte ^ 0x88) + int4 row sums
 *   kind "i8"  : int8-unpacked [K][N] for the cuBLAS IMMA prefill path
 *                (2x the payload of the int4 weight) + per-channel row sums
 *
 * Both are re-derived from scratch on every launch -- the same bytes, every
 * time. This module persists them next to the model bin (<bin>.cudapack; falls
 * back to $XDG_CACHE_HOME/nntrainer/cudapack when the model dir is not
 * writable), so a later launch mmaps the pack and uploads straight from the
 * mapped pages, skipping the host permute/row-sum entirely.
 *
 * This is the CUDA lane's analog of the OpenCL lane's v8c pack cache and
 * deliberately keeps the same on-disk shape, the same identity rules and the
 * same failure semantics, so the two lanes read the same way.
 *
 * Identity/keying (HARD RULE -- never pointer-keyed; the pointer-keyed derived
 * weight caches in this tree have mis-hit under SVM/UVM before): the pack file
 * is bound to the source weight file by (size, mtime-ns, format version); each
 * record is keyed by (record-name FNV-1a, N, K, row_bytes, payload length) and
 * guarded by a sampled payload FNV (both 64 KB ends + 16 interior pages,
 * bounded ~192 KB/record so validation never costs what the cache saves) plus
 * a full FNV over the row-sum block. Any mismatch is a silent per-record miss
 * (derive exactly as before); a stale/absent/corrupt header invalidates the
 * whole file and one launch rewrites it (temp file + fsync + atomic rename,
 * finalized on a background thread that is exit-joined).
 *
 * Off by default: NNTR_CUDA_PACK_CACHE=1 opts in. That default is a measured
 * verdict, not caution -- the "dp4a" derive is bandwidth-bound and 16-way
 * parallel, so caching it wins ~100 ms on a warm page cache and LOSES ~700 ms
 * when the pack has to be read from disk, while costing pack-sized disk; the
 * "i8" derive is a column-strided transpose-unpack and caching that wins a full
 * second on a prefill-heavy turn. See the commit message for the numbers.
 * NNTR_CUDA_PACK_CACHE_MIN_MB (default 0 = every weight) bounds which weights
 * are cached. POSIX-only (no-op stubs elsewhere).
 */

#ifndef __CUDA_PACK_CACHE_H__
#define __CUDA_PACK_CACHE_H__

#include <cstddef>
#include <cstdint>

namespace nntrainer::cuda_pack {

/**
 * @brief A validated cache hit: pointers into the pack file mmap. Valid until
 *        the next set_source().
 */
struct Hit {
  const uint8_t *payload = nullptr; /**< packed bytes, payload_len long */
  const int32_t *rowsum = nullptr;  /**< per-channel row sums, N entries */
  size_t payload_len = 0;
};

/** @brief Opaque per-record writer handle (miss path tee). */
struct RecordWriter;

/**
 * @brief Is the cache armed for this process at all? (env opt-in)
 */
bool enabled();

/**
 * @brief Bind the cache to a source weight file, validating/mapping an
 *        existing pack for it; a stale or corrupt pack arms rewrite mode.
 *        Safe to call again (model switch): joins any in-flight finalize.
 */
void set_source(const char *model_bin_path);

/**
 * @brief All load-time derives are done: finalize a pending rewrite (index +
 *        header + fsync + rename) on a background thread, exit-joined.
 */
void load_complete();

/**
 * @brief Look up a record. Returns true and fills @p out only when every key
 *        field and both checksums match.
 * @param name record name, "<weight name>#<kind>" (caller-composed)
 */
bool lookup(const char *name, unsigned int N, unsigned int K, size_t row_bytes,
            size_t payload_len, Hit &out);

/** @brief Drop the (clean, file-backed) payload pages of a consumed range. */
void payload_consumed(const uint8_t *payload, size_t len);

/**
 * @brief Start teeing a record derive to the pack temp file. Returns nullptr
 *        when the cache is off, the payload is below the size floor, or a
 *        valid pack already exists (a partial rewrite never clobbers a good
 *        pack). Must be paired with commit_record or abort_record.
 */
RecordWriter *begin_record(const char *name, unsigned int N, unsigned int K,
                           size_t row_bytes, size_t payload_len);

/** @brief Tee one derived chunk at @p payload_off within this record. */
void record_write(RecordWriter *rw, size_t payload_off, const void *data,
                  size_t len);

/** @brief Payload fully written: append row sums, checksum, index the record.
 */
void commit_record(RecordWriter *rw, const int32_t *rowsum, size_t count);

/** @brief Derive failed mid-way: forget the record (region left unreferenced).
 */
void abort_record(RecordWriter *rw);

} // namespace nntrainer::cuda_pack

#endif /* __CUDA_PACK_CACHE_H__ */

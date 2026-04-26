// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   compute_ops.cpp
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Global ComputeOps pointer + thread-safe lazy init.
 *
 * `ensureComputeOps()` is the single canonical entry point that
 * guarantees `g_compute_ops` is initialized exactly once across
 * threads. It uses std::call_once, which gives us:
 *
 *   1. Mutual exclusion during init_backend() — non-idempotent setup
 *      (__ggml_init, __openblas_set_num_threads) can no longer race
 *      when getComputeOps() is hit concurrently from a cold cache
 *      (e.g. the first tensor op on each worker thread).
 *
 *   2. Acquire/release synchronization — any thread returning from
 *      call_once observes the writes init_backend() made to
 *      g_compute_ops, even on the "already initialized" path.
 *      The inline getComputeOps() fast-path stays correct: the racy
 *      nullptr check only ever upgrades into ensureComputeOps(), and
 *      the read after returning from it is synchronized through the
 *      once_flag.
 *
 * Callers that previously invoked init_backend() directly (AppContext,
 * Engine, ClContext, QNNContext) now route through ensureComputeOps()
 * so the call_once guard cannot be bypassed.
 */

#include <compute_ops.h>

#include <mutex>

namespace nntrainer {

ComputeOps *g_compute_ops = nullptr;

namespace {
std::once_flag g_compute_ops_init_flag;
} // namespace

void ensureComputeOps() {
  std::call_once(g_compute_ops_init_flag, []() { init_backend(); });
}

} // namespace nntrainer

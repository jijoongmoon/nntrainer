// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_stream_manager.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA wrapper for stream/dispatch management. Peer of
 *          nntrainer::opencl::CommandQueueManager: owns one cudaStream_t and
 *          launches kernels (cuLaunchKernel) / copies (cudaMemcpyAsync) on it.
 */

#ifndef __CUDA_STREAM_MANAGER_H__
#define __CUDA_STREAM_MANAGER_H__

#include <cstddef>

#include <cuda_runtime.h>

#include "singleton.h"

namespace nntrainer::cuda {

class Kernel;

/**
 * @class StreamManager
 * @brief Singleton owning the backend's primary CUDA stream + dispatch helpers.
 */
class StreamManager : public Singleton<StreamManager> {
public:
  /**
   * @brief Get the backend stream
   */
  cudaStream_t GetStream() const { return stream_; }

  /**
   * @brief host->device copy on the backend stream (sync unless async)
   */
  bool EnqueueWriteBuffer(void *dst_dev, size_t size, const void *src_host,
                          bool async = false);

  /**
   * @brief device->host copy on the backend stream (sync unless async)
   */
  bool EnqueueReadBuffer(const void *src_dev, size_t size, void *dst_host,
                         bool async = false);

  /**
   * @brief Launch @p kernel with the given 3D grid (blocks) and block (threads)
   *        dims and optional dynamic shared memory.
   */
  bool DispatchCommand(Kernel &kernel, const int (&grid)[3],
                       const int (&block)[3], unsigned int shared_bytes = 0);

  /**
   * @brief Block until the stream drains (cudaStreamSynchronize).
   */
  void finish();

  /**
   * @brief Conditional drain: finish() unless NNTR_CUDA_ASYNC=1. Per-op
   *        cudaStreamSynchronize is ~90% of decode wall time (it serializes
   *        CPU/GPU); once every decode op is on-GPU (no host op reads UVM
   *        mid-chain), NNTR_CUDA_ASYNC=1 turns these into no-ops so the GPU
   *        pipeline fills and only the final host read (sampling) drains once
   *        per token. Until then, default (sync) keeps coherence.
   */
  void maybeFinish();

  /**
   * @brief Inverse of maybeFinish: drain ONLY when NNTR_CUDA_ASYNC=1. Call this
   *        right before a HOST op reads GPU output on a path that stays on the
   *        host (e.g. the prefill RoPE fallback): in async mode the GPU ops did
   *        not drain, so the host read must sync first; in default mode the
   *        stream is already drained so it is a cheap no-op.
   */
  void finishIfAsync();

  /**
   * @brief Suppress the per-op drain over a bounded region, RAII-style via
   *        pushDeferDrain()/popDeferDrain().
   *
   * On an integrated GPU maybeFinish() is a full cudaStreamSynchronize, so a
   * region that issues many small device ops back to back pays one pipeline
   * flush per op. The MoE expert loop is the extreme case: 8 experts x 3
   * projections per layer per token at decode, and 256 x 3 per layer per chunk
   * at prefill -- 61,440 drains for a single 1,341-token prefill, which
   * measured as ~92% of that layer's time against ~8% of actual GEMM.
   *
   * THE CONTRACT, and it is not a soft one: while deferred, NOTHING on the
   * host may read a buffer any of those ops writes. The caller is responsible
   * for having moved every such read onto the device first, and for calling
   * finish() explicitly at the end of the region -- finish() itself is NOT
   * suppressed, only maybeFinish(). Getting this wrong reproduces exactly the
   * class of bug the missing drain in cuda_fc_dense::gemm_ex caused: work that
   * IS on the device, read too early, wrong and different on every run, and
   * invisible to both host-op detectors.
   */
  void pushDeferDrain() { ++defer_drain_; }
  /** @brief End one deferred-drain region. */
  void popDeferDrain() {
    if (defer_drain_ > 0)
      --defer_drain_;
  }
  /** @brief True while inside a deferred-drain region. */
  bool drainDeferred() const { return defer_drain_ > 0; }

  /**
   * @brief Bounded-staleness drain point inside a deferred-drain region.
   *
   * A no-op outside a region (and under capture). The whole-forward defer
   * region calls this at node boundaries when NNTR_CUDA_PREFILL_DEFER=2, both
   * as a bisect scaffold for defer-exposed races and as a shallower-queue
   * fallback if full-region defer proves too deep for this driver.
   */
  void deferCheckpoint() {
    if (defer_drain_ > 0 && !capturing_)
      finish();
  }

  /**
   * @brief Tag the following dispatches with the graph node they belong to
   *        (diagnostics: an async fault names its victim launch, and the tag
   *        names the layer that issued it). Copied, so any lifetime is fine.
   */
  void setDispatchTag(const char *tag);
  /** @brief Current dispatch tag ("" when unset). */
  const char *dispatchTag() const { return dispatch_tag_; }

  /**
   * @brief Ordering drain whose caller then runs a DEVICE kernel — the same
   *        sync as finish(), minus the [CAP-AUDIT] bookkeeping.
   *
   * finish() and finishIfAsync() log a capture-time skip because their callers
   * are host-fallback preambles: a hit there means a host op ran inside the
   * graph, which on this hardware is a wrong-answer bug (the host op reads
   * buffers whose producing kernels have not run, and is never recorded into
   * the graph). This entry point is for the opposite case — an ordering barrier
   * before a device kernel, where skipping under capture is not merely allowed
   * but correct.
   *
   * Keeping it out of the audit is what lets "zero [CAP-AUDIT] lines" stand as
   * a literal pass condition for the host-op-free requirement, instead of a
   * count with one hand-explained false positive in it.
   */
  void drainPipeline();

  /**
   * @brief Begin CUDA-graph stream capture on the backend stream (Relaxed mode,
   *        which allows the driver-API cuLaunchKernel + cuBLAS sub-launches).
   *        Drains the stream first (start from idle), then enters capture.
   * While capturing, kernel/cuBLAS calls are RECORDED into a graph rather than
   *        executed, and finish()/maybeFinish()/finishIfAsync() become no-ops
   *        (an in-capture cudaStreamSynchronize is illegal -- drains are
   * deferred to after the graph replay). Returns false if the stream is missing
   * / begin fails.
   * @note  Decode CUDA-graph foundation. Capturing the whole
   *        per-token forward additionally needs the embedding host-staging
   *        buffers (embedding_layer.cpp / tie_word_embedding.cpp `emb_stage`)
   * to be PERSISTENT + PINNED (a local std::vector is freed before the graph
   *        replays, and a pageable cudaMemcpyAsync is not capturable). TODO.
   */
  bool beginCapture();

  /**
   * @brief End stream capture; returns the captured graph in @p graph. After
   *        this, isCapturing() is false again.
   */
  bool endCapture(cudaGraph_t *graph);

  /**
   * @brief True while a capture is in progress (drains are suppressed).
   */
  bool isCapturing() const { return capturing_; }

  /**
   * @brief Report that the capture in progress can no longer produce a FAITHFUL
   *        graph, so endCapture() must refuse it.
   *
   * The driver only invalidates a capture for things IT can see (a synchronous
   * API call on the captured stream). The dangerous case is the opposite one:
   * an op that notices it may not run inside a capture (a scratch buffer that
   * would have to grow, a side allocation that would need a cudaMalloc) and
   * quietly DECLINES -- the capture stays "valid" and the graph is instantiated
   * with that op simply MISSING. It then replays to wrong numbers with no error
   * anywhere. Every such decline calls this instead, and endCapture() turns it
   * into "no graph", which the graph callers already handle by re-running the
   * forward eagerly. Correctness first; the lost graph costs one slow forward.
   *
   * @param why short reason, logged once per capture
   */
  void markCaptureDoomed(const char *why);

  /**
   * @brief True if markCaptureDoomed() was called since beginCapture().
   */
  bool captureDoomed() const { return capture_doomed_; }

  /**
   * @brief Monotonic count of kernels dispatched on the backend stream.
   *
   * Producer/consumer ops that want to hand a derived buffer straight to the
   * next op (rather than recomputing it) can only do so while NOTHING ELSE
   * touched the source in between. A raw pointer equality test is not enough
   * for that -- the activation pool recycles buffers, so a later, unrelated
   * tensor can land on the very same address. Stamping the handoff with this
   * counter turns "same pointer" into "same pointer AND not one kernel ran
   * since", which the pool cannot forge.
   */
  unsigned long long dispatchSeq() const { return dispatch_seq_; }

  /**
   * @brief Destroy the stream
   */
  ~StreamManager() override;

protected:
  /**
   * @brief Singleton hook: ensure device/context then create the stream.
   */
  void initialize() noexcept override;

private:
  cudaStream_t stream_{nullptr};
  bool capturing_{false};
  unsigned int defer_drain_{0};
  bool capture_doomed_{false};
  char dispatch_tag_[96]{};
  unsigned long long dispatch_seq_{0};
};

/**
 * @brief NNTR_KERN_PROF=1: per-launch cudaEvent GPU-time histogram.
 *
 * Every DispatchCommand launch (and the two cuBLAS GEMM entry points) is
 * bracketed with a cudaEvent pair on the backend stream and accumulated into a
 * table keyed by "kernel|role", where role is the dispatch tag with its
 * leading "layer<NN>_" stripped so the 40 repeats of a node aggregate into one
 * row. Pairs are drained lazily (a bounded pending ring; the oldest entry is
 * synced only when the ring fills, by which time the GPU has long passed it),
 * so the instrument adds host-side record cost but no stream drains. The table
 * prints to stderr at process exit, sorted by total GPU ms. This measures
 * KERNEL time under the production defer schedule -- the complement (wall
 * minus table total) is memcpy + host gaps. Off (nullptr/zero-cost) unless the
 * env is set; disabled during graph capture (events would record into the
 * graph).
 */
bool kprof_enabled();
/** @brief Record a start event; nullptr when off/capturing. */
void *kprof_begin();
/** @brief Record the stop event and file the pair under kern|role(tag). */
void kprof_end(void *start_ev, const char *kern, const char *tag);
/** @brief Drain pending pairs and print the table (idempotent; also atexit). */
void kprof_dump();
/** @brief Drain, print the table under a window label, then RESET it.
 *  Bracketing a region (e.g. the prefill) scopes the histogram -- and its
 *  host-gap column -- to that region alone. The gap column attributes the
 *  GPU-idle time BEFORE each kernel to that kernel's row, pointing at the
 *  host code path that ran between the previous launch and this one. */
void kprof_window(const char *label);

/**
 * @brief Process-lifetime device int[2] holding the per-token DECODE position:
 *        [0] = pos (== cache_index / RoPE `from`), [1] = N_kv (== pos+1). The
 *        M2-B single-capture decode graph bakes this FIXED device pointer into
 *        its RoPE / attention / KV-write kernel nodes, so cross-token replay
 * only rewrites these 8 bytes (cuda_set_pos) instead of re-recording the graph.
 *        Allocated once on first use.
 */
int *cuda_pos_buffer();

/**
 * @brief Update the per-token decode position (8-byte H2D on the backend
 * stream, issued OUTSIDE graph capture, ordered before the cudaGraphLaunch that
 *        reads it). pos == cache_index for the token; n_kv == pos+1.
 */
void cuda_set_pos(int pos, int n_kv);

} // namespace nntrainer::cuda

#endif // __CUDA_STREAM_MANAGER_H__

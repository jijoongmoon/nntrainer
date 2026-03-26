// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   thread_manager.h
 * @date   20 March 2026
 * @brief  Unified thread manager for compute and I/O operations
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_THREAD_MANAGER_H__
#define __NNTRAINER_THREAD_MANAGER_H__

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <completion_token.h>
#include <singleton.h>

namespace nntrainer {

struct ThreadManagerConfig {
  unsigned int compute_threads =
    std::thread::hardware_concurrency() > 2
      ? std::thread::hardware_concurrency() - 2
      : 1;
  unsigned int io_threads = 1;
  bool enable_affinity = false;
};

/**
 * @class ThreadManager
 * @brief Unified thread pool with GGML-style spin-wait barrier.
 *
 * Compute workers spin-wait for work and synchronize via atomic barrier.
 * No condition variables on the hot path — minimal dispatch latency.
 * I/O workers use condition variable (blocking I/O is fine).
 */
class ThreadManager : public Singleton<ThreadManager> {
  friend class Singleton<ThreadManager>;

public:
  ~ThreadManager();

  /**
   * @brief Parallel for using all compute workers + caller.
   */
  template <typename F> void parallel_for(size_t begin, size_t end, F &&fn) {
    if (begin >= end)
      return;
    if (end - begin == 1 || compute_workers_.empty()) {
      for (size_t i = begin; i < end; ++i)
        fn(i);
      return;
    }
    dispatchAndJoin(begin, end, std::forward<F>(fn));
  }

  /**
   * @brief Parallel for using at most n_workers compute workers + caller.
   */
  template <typename F>
  void parallel_for(size_t begin, size_t end, unsigned int n_workers, F &&fn) {
    if (begin >= end)
      return;
    unsigned int total = static_cast<unsigned int>(compute_workers_.size());
    if (n_workers > total)
      n_workers = total;
    if (end - begin == 1 || n_workers == 0 || compute_workers_.empty()) {
      for (size_t i = begin; i < end; ++i)
        fn(i);
      return;
    }
    dispatchAndJoin(begin, end, std::forward<F>(fn), n_workers);
  }

  /**
   * @brief Parallel for with thread-index based chunking.
   */
  template <typename F> void parallel_for_chunked(size_t n_threads, F &&fn) {
    if (n_threads <= 1) {
      fn(0);
      return;
    }
    parallel_for(0, n_threads, std::forward<F>(fn));
  }

  CompletionToken submit(std::function<void()> task);

  unsigned int getComputeThreadCount() const {
    return static_cast<unsigned int>(compute_workers_.size());
  }

  unsigned int getIOThreadCount() const {
    return static_cast<unsigned int>(io_workers_.size());
  }

  static void setConfig(const ThreadManagerConfig &config) {
    pending_config_ = config;
  }

protected:
  ThreadManager();
  void initialize() noexcept override;

private:
  static inline void cpuRelax() {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
  }

  /**
   * @brief GGML-style spin-wait barrier.
   * Last thread to arrive resets counter and bumps pass count.
   * Other threads spin on pass count with cpu_relax.
   */
  /**
   * @brief Sense-reversing barrier.
   *
   * Each round uses the OPPOSITE sense (true/false). This prevents the
   * race where a fast thread exits barrier, loops back, and re-enters
   * the next barrier before slow threads have left the current one.
   * With alternating sense, a thread spinning on sense=true won't be
   * confused by a leftover sense=true from the previous round.
   */
  void barrier(bool sense) {
    int n_threads =
      active_threads_.load(std::memory_order_acquire);
    int n = n_barrier_.fetch_add(1, std::memory_order_acq_rel);
    if (n == n_threads - 1) {
      n_barrier_.store(0, std::memory_order_relaxed);
      // flip the sense flag to signal completion
      barrier_sense_.store(sense, std::memory_order_release);
      return;
    }
    // spin until the last thread flips the sense
    while (barrier_sense_.load(std::memory_order_acquire) != sense) {
      cpuRelax();
    }
  }

  /**
   * @brief Dispatch work to workers and join via barrier.
   */
  template <typename F>
  void dispatchAndJoin(size_t begin, size_t end, F &&fn,
                       unsigned int n_workers = 0) {
    unsigned int total = static_cast<unsigned int>(compute_workers_.size());
    if (n_workers == 0 || n_workers > total)
      n_workers = total;

    // only active workers + caller participate in barrier
    active_threads_.store(static_cast<int>(n_workers + 1),
                          std::memory_order_release);

    // compute the sense for this round (alternates each dispatch)
    bool sense = !barrier_sense_.load(std::memory_order_acquire);

    current_task_ = [&fn](size_t i) { fn(i); };
    task_end_ = end;
    current_chunk_.store(begin, std::memory_order_relaxed);
    active_workers_.store(n_workers, std::memory_order_release);
    current_sense_.store(sense, std::memory_order_release);

    // wake workers
    generation_.fetch_add(1, std::memory_order_seq_cst);

    // caller participates (work-stealing via atomic counter)
    while (true) {
      size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
      if (idx >= end)
        break;
      fn(idx);
    }

    // barrier: wait for all threads
    barrier(sense);
    current_task_ = nullptr;
  }

  void computeWorkerLoop(unsigned int worker_id);
  void ioWorkerLoop();

  // ─── Compute (all cache-line isolated) ───────────────
  std::vector<std::thread> compute_workers_;
  std::function<void(size_t)> current_task_;
  size_t task_end_{0};

  alignas(64) std::atomic<unsigned int> generation_{0};
  alignas(64) std::atomic<int> n_barrier_{0};
  alignas(64) std::atomic<bool> barrier_sense_{false};
  alignas(64) std::atomic<bool> current_sense_{false};
  alignas(64) std::atomic<size_t> current_chunk_{0};
  alignas(64) std::atomic<unsigned int> active_workers_{0};
  alignas(64) std::atomic<int> active_threads_{1};
  alignas(64) std::atomic<bool> stop_{false};

  // ─── I/O ─────────────────────────────────────────────
  std::vector<std::thread> io_workers_;
  std::queue<std::pair<std::function<void()>,
                       std::shared_ptr<CompletionToken::SharedState>>>
    io_queue_;
  std::mutex io_mutex_;
  std::condition_variable io_cv_;

  // ─── Config ──────────────────────────────────────────
  ThreadManagerConfig config_;
  static ThreadManagerConfig pending_config_;
};

} // namespace nntrainer

#endif // __NNTRAINER_THREAD_MANAGER_H__

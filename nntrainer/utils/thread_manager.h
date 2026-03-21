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
 *
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

/**
 * @enum WaitPolicy
 * @brief How compute workers wait for work between parallel_for calls.
 */
enum class WaitPolicy {
  Sleep,    /**< condition_variable only (lowest CPU, highest latency) */
  Adaptive, /**< spin briefly, then yield, then sleep (balanced) */
  Spin      /**< busy-wait only (lowest latency, highest CPU usage) */
};

struct ThreadManagerConfig {
  unsigned int compute_threads =
    std::thread::hardware_concurrency(); /**< compute worker count */
  unsigned int io_threads = 3;           /**< I/O worker count */
  bool enable_affinity = false;          /**< pin workers to cores 1:1 */
  WaitPolicy wait_policy = WaitPolicy::Adaptive; /**< compute worker wait */
  unsigned int spin_count = 1000; /**< spin iterations before yield/sleep */
};

/**
 * @struct WorkerTask
 * @brief Per-worker task assignment. Each worker gets its own range.
 *        Padded to cache line to prevent false sharing.
 */
struct alignas(64) WorkerTask {
  size_t begin{0};
  size_t end{0};
};

/**
 * @class ThreadManager
 * @brief Unified thread pool managing both compute and I/O workers.
 *
 * Compute workers: Each worker has its own task range assigned by the
 * caller. No atomic contention on the hot path. Workers spin-wait
 * (configurable) for new generations.
 *
 * I/O workers: Condition variable based queue. Independent from compute.
 */
class ThreadManager : public Singleton<ThreadManager> {
  friend class Singleton<ThreadManager>;

public:
  ~ThreadManager();

  /**
   * @brief Parallel for using all compute workers + caller.
   */
  template <typename F> void parallel_for(size_t begin, size_t end, F &&fn) {
    parallel_for_with(begin, end,
                      static_cast<unsigned int>(compute_workers_.size()),
                      std::forward<F>(fn));
  }

  /**
   * @brief Parallel for using at most n_workers compute workers + caller.
   */
  template <typename F>
  void parallel_for(size_t begin, size_t end, unsigned int n_workers, F &&fn) {
    parallel_for_with(begin, end, n_workers, std::forward<F>(fn));
  }

  /**
   * @brief Parallel for with thread-index based chunking.
   * Each index [0, n_threads) is passed to fn.
   */
  template <typename F> void parallel_for_chunked(size_t n_threads, F &&fn) {
    if (n_threads <= 1) {
      fn(0);
      return;
    }
    unsigned int workers_needed = static_cast<unsigned int>(n_threads) - 1;
    parallel_for_with(0, n_threads, workers_needed, std::forward<F>(fn));
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
  /**
   * @brief Core parallel_for with per-worker task assignment.
   */
  template <typename F>
  void parallel_for_with(size_t begin, size_t end, unsigned int n_workers,
                         F &&fn) {
    if (begin >= end)
      return;

    size_t range = end - begin;
    unsigned int total = static_cast<unsigned int>(compute_workers_.size());

    if (n_workers > total)
      n_workers = total;

    // serial fallback
    if (range == 1 || n_workers == 0 || compute_workers_.empty()) {
      for (size_t i = begin; i < end; ++i)
        fn(i);
      return;
    }

    // total threads = n_workers + 1 (caller)
    unsigned int n_threads = n_workers + 1;

    // wait for all workers to be ready
    {
      std::unique_lock<std::mutex> lock(done_mutex_);
      done_cv_.wait(lock, [this, total] {
        return workers_ready_.load(std::memory_order_acquire) >= total;
      });
    }

    // assign per-worker ranges (no atomic contention)
    current_task_ = [&fn](size_t i) { fn(i); };
    for (unsigned int t = 0; t < n_workers; ++t) {
      size_t w_begin = begin + ((t + 1) * range) / n_threads;
      size_t w_end = begin + ((t + 2) * range) / n_threads;
      worker_tasks_[t].begin = w_begin;
      worker_tasks_[t].end = w_end;
    }

    workers_done_.store(0, std::memory_order_release);
    workers_ready_.store(0, std::memory_order_release);
    active_workers_.store(n_workers, std::memory_order_release);

    // wake workers
    {
      std::lock_guard<std::mutex> lock(compute_mutex_);
      compute_generation_.fetch_add(1, std::memory_order_release);
    }
    compute_cv_.notify_all();

    // caller handles chunk 0: [begin, begin + range/n_threads)
    size_t caller_end = begin + range / n_threads;
    for (size_t i = begin; i < caller_end; ++i)
      fn(i);

    // wait for all compute workers to finish
    waitComputeDone();
    current_task_ = nullptr;
  }

  void computeWorkerLoop(unsigned int worker_id);
  void ioWorkerLoop();
  void waitComputeDone();

  // ─── Compute workers ───────────────────────────────────
  // Each frequently-written atomic on its own cache line to prevent
  // false sharing (same fix as llama.cpp #9598: +21% on 80-core ARM).

  std::vector<std::thread> compute_workers_;
  std::vector<WorkerTask> worker_tasks_; /**< per-worker range, cache aligned */
  std::function<void(size_t)> current_task_;

  // generation: written by caller, read by all workers (wake signal)
  alignas(64) std::atomic<unsigned int> compute_generation_{0};

  // workers_done: written by each worker, read by caller (completion signal)
  alignas(64) std::atomic<unsigned int> workers_done_{0};

  // workers_ready: written by each worker, read by caller (readiness check)
  alignas(64) std::atomic<unsigned int> workers_ready_{0};

  // active_workers: written by caller, read by workers (activation check)
  alignas(64) std::atomic<unsigned int> active_workers_{0};

  // stop: written by destructor, read by all workers
  alignas(64) std::atomic<bool> stop_{false};

  // mutexes and CVs (not contended atomically, grouping is fine)
  std::mutex compute_mutex_;
  std::condition_variable compute_cv_;
  std::mutex done_mutex_;
  std::condition_variable done_cv_;

  // ─── I/O workers ─────────────────────────────────────
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

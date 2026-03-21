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
 * @struct ThreadManagerConfig
 * @brief  Configuration for ThreadManager initialization
 */
struct ThreadManagerConfig {
  unsigned int compute_threads =
    std::thread::hardware_concurrency(); /**< compute worker count */
  unsigned int io_threads = 3;           /**< I/O worker count */
  bool enable_affinity = false;          /**< pin workers to cores 1:1 */
};

/**
 * @class ThreadManager
 * @brief Unified thread pool managing both compute and I/O workers.
 *
 * Compute workers: Used by parallel_for(). Supports specifying the number
 * of threads per call — only the requested workers are activated.
 * The calling thread also participates (N+1 way parallelism).
 *
 * I/O workers: Used by submit(). Returns CompletionToken for sync.
 * Physically separate from compute workers.
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
   *
   * If n_workers > total compute workers, uses all available.
   * If n_workers == 0, runs serially on calling thread.
   *
   * @param begin start index (inclusive)
   * @param end end index (exclusive)
   * @param n_workers number of compute workers to activate (not counting caller)
   * @param fn callable with signature void(size_t)
   */
  template <typename F>
  void parallel_for(size_t begin, size_t end, unsigned int n_workers, F &&fn) {
    parallel_for_with(begin, end, n_workers, std::forward<F>(fn));
  }

  /**
   * @brief Parallel for with thread-index based chunking.
   *
   * Each index [0, n_threads) is passed to fn. Uses n_threads-1 workers
   * + the calling thread.
   */
  template <typename F> void parallel_for_chunked(size_t n_threads, F &&fn) {
    if (n_threads <= 1) {
      fn(0);
      return;
    }
    unsigned int workers_needed =
      static_cast<unsigned int>(n_threads) - 1; // caller is one thread
    parallel_for_with(0, n_threads, workers_needed, std::forward<F>(fn));
  }

  /**
   * @brief Submit an async task to I/O workers.
   */
  CompletionToken submit(std::function<void()> task);

  /**
   * @brief Get the number of compute worker threads (not counting caller)
   */
  unsigned int getComputeThreadCount() const {
    return static_cast<unsigned int>(compute_workers_.size());
  }

  /**
   * @brief Get the number of I/O worker threads
   */
  unsigned int getIOThreadCount() const {
    return static_cast<unsigned int>(io_workers_.size());
  }

  /**
   * @brief Configure thread counts before initialization.
   */
  static void setConfig(const ThreadManagerConfig &config) {
    pending_config_ = config;
  }

protected:
  ThreadManager();
  void initialize() noexcept override;

private:
  /**
   * @brief Core parallel_for implementation with worker count control.
   */
  template <typename F>
  void parallel_for_with(size_t begin, size_t end, unsigned int n_workers,
                         F &&fn) {
    if (begin >= end)
      return;

    size_t range = end - begin;
    unsigned int total = static_cast<unsigned int>(compute_workers_.size());

    // clamp n_workers to available
    if (n_workers > total)
      n_workers = total;

    // serial fallback
    if (range == 1 || n_workers == 0 || compute_workers_.empty()) {
      for (size_t i = begin; i < end; ++i)
        fn(i);
      return;
    }

    // wait for needed workers to be ready
    {
      std::unique_lock<std::mutex> lock(done_mutex_);
      done_cv_.wait(lock, [this, total] {
        return workers_ready_.load(std::memory_order_acquire) >= total;
      });
    }

    // setup task
    current_task_ = [&fn](size_t i) { fn(i); };
    task_end_.store(end, std::memory_order_relaxed);
    chunk_counter_.store(begin, std::memory_order_release);
    workers_done_.store(0, std::memory_order_release);
    workers_ready_.store(0, std::memory_order_release);
    active_workers_.store(n_workers, std::memory_order_release);

    // wake all workers (inactive ones will immediately signal done)
    {
      std::lock_guard<std::mutex> lock(compute_mutex_);
      compute_generation_.fetch_add(1, std::memory_order_release);
    }
    compute_cv_.notify_all();

    // calling thread participates
    while (true) {
      size_t idx = chunk_counter_.fetch_add(1, std::memory_order_relaxed);
      if (idx >= end)
        break;
      fn(idx);
    }

    // wait for all compute workers to finish
    waitComputeDone();
    current_task_ = nullptr;
  }

  void computeWorkerLoop(unsigned int worker_id);
  void ioWorkerLoop();
  void waitComputeDone();

  // Compute workers
  std::vector<std::thread> compute_workers_;
  std::mutex compute_mutex_;
  std::condition_variable compute_cv_;
  std::function<void(size_t)> current_task_;
  std::atomic<size_t> chunk_counter_{0};
  std::atomic<size_t> task_end_{0};
  std::atomic<unsigned int> compute_generation_{0};
  std::atomic<unsigned int> workers_done_{0};
  std::atomic<unsigned int> workers_ready_{0};
  std::atomic<unsigned int> active_workers_{0};
  std::mutex done_mutex_;
  std::condition_variable done_cv_;

  // I/O workers
  std::vector<std::thread> io_workers_;
  std::queue<std::pair<std::function<void()>,
                       std::shared_ptr<CompletionToken::SharedState>>>
    io_queue_;
  std::mutex io_mutex_;
  std::condition_variable io_cv_;

  // Shared
  std::atomic<bool> stop_{false};

  static ThreadManagerConfig pending_config_;
};

} // namespace nntrainer

#endif // __NNTRAINER_THREAD_MANAGER_H__

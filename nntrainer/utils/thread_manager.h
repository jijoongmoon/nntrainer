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
  /**
   * @brief Number of compute worker threads.
   * Default uses NNTR_NUM_THREADS if set > 0, otherwise OMP_NUM_THREADS - 1.
   * This avoids creating too many threads on machines with many cores.
   */
  unsigned int compute_threads = defaultComputeThreads();
  unsigned int io_threads = 1;
  bool enable_affinity = false;

private:
  static unsigned int defaultComputeThreads() {
#if defined(NNTR_NUM_THREADS) && NNTR_NUM_THREADS > 0
    return NNTR_NUM_THREADS;
#elif defined(OMP_NUM_THREADS) && OMP_NUM_THREADS > 1
    return OMP_NUM_THREADS - 1; // -1 because caller also participates
#else
    unsigned int hw = std::thread::hardware_concurrency();
    return hw > 2 ? std::min(hw - 2, 6u) : 1;
#endif
  }
};

/**
 * @class ThreadManager
 * @brief Unified thread pool for compute and I/O operations.
 *
 * Compute: condvar-based dispatch + condvar barrier (reliable in release mode).
 * I/O: condvar-based task queue for blocking I/O.
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
  /**
   * @brief Dispatch work to workers and join via condvar barrier.
   */
  template <typename F>
  void dispatchAndJoin(size_t begin, size_t end, F &&fn,
                       unsigned int n_workers = 0) {
    unsigned int total = static_cast<unsigned int>(compute_workers_.size());
    if (n_workers == 0 || n_workers > total)
      n_workers = total;

    // setup work
    {
      std::lock_guard<std::mutex> lock(dispatch_mutex_);
      current_task_ = [&fn](size_t i) { fn(i); };
      task_end_ = end;
      current_chunk_.store(begin, std::memory_order_relaxed);
      active_workers_ = n_workers;
      barrier_count_ = 0;
      barrier_target_ = static_cast<int>(n_workers + 1); // workers + caller
      ++dispatch_gen_;
    }
    dispatch_cv_.notify_all();

    // caller participates (work-stealing via atomic counter)
    while (true) {
      size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
      if (idx >= end)
        break;
      fn(idx);
    }

    // caller arrives at barrier
    {
      std::unique_lock<std::mutex> lock(barrier_mutex_);
      ++barrier_count_;
      barrier_cv_.wait(lock,
                       [this] { return barrier_count_ >= barrier_target_; });
    }
    barrier_cv_.notify_all();
    current_task_ = nullptr;
  }

  void computeWorkerLoop(unsigned int worker_id);
  void ioWorkerLoop();

  // ─── Compute ────────────────────────────────────────
  std::vector<std::thread> compute_workers_;
  std::function<void(size_t)> current_task_;
  size_t task_end_{0};
  unsigned int active_workers_{0};

  // dispatch signaling (condvar-based, reliable in release mode)
  std::mutex dispatch_mutex_;
  std::condition_variable dispatch_cv_;
  unsigned int dispatch_gen_{0};

  // barrier (condvar-based)
  std::mutex barrier_mutex_;
  std::condition_variable barrier_cv_;
  int barrier_count_{0};
  int barrier_target_{0};

  // work-stealing counter
  alignas(64) std::atomic<size_t> current_chunk_{0};
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

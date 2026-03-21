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
};

/**
 * @class ThreadManager
 * @brief Unified thread pool managing both compute and I/O workers.
 *
 * Replaces TaskExecutor, BS::thread_pool, and ParallelBatch with a single
 * lightweight thread manager. Compute and I/O workers are physically
 * separated to prevent GEMM performance degradation from I/O tasks.
 *
 * Compute workers: Used exclusively by parallel_for(). All workers
 * participate in every parallel_for call via atomic chunk distribution.
 * The calling thread also participates (N+1 way parallelism).
 *
 * I/O workers: Used exclusively by submit(). Returns CompletionToken
 * for synchronization. Suitable for blocking operations like FSU
 * load/unload (disk I/O).
 *
 * Usage:
 *   auto &tm = ThreadManager::Global();
 *   tm.parallel_for(0, N, [](size_t i) { compute(i); });
 *   auto token = tm.submit([&]{ load_from_disk(); });
 *   token.wait();
 */
class ThreadManager : public Singleton<ThreadManager> {
  friend class Singleton<ThreadManager>;

public:
  /**
   * @brief Destructor. Stops all workers and joins threads.
   */
  ~ThreadManager();

  /**
   * @brief Parallel for loop over [begin, end) using all compute workers.
   *
   * Distributes work dynamically via atomic counter. The calling thread
   * also participates in computation. Blocks until all iterations complete.
   * Zero heap allocation on the hot path.
   *
   * @tparam F callable with signature void(size_t)
   * @param begin start index (inclusive)
   * @param end end index (exclusive)
   * @param fn function to execute for each index
   * @note Not reentrant. Only one parallel_for can run at a time.
   */
  template <typename F> void parallel_for(size_t begin, size_t end, F &&fn) {
    if (begin >= end)
      return;

    size_t range = end - begin;
    if (range == 1 || compute_workers_.empty()) {
      for (size_t i = begin; i < end; ++i)
        fn(i);
      return;
    }

    unsigned int n = static_cast<unsigned int>(compute_workers_.size());

    // wait for all workers to be ready (back in wait loop)
    {
      std::unique_lock<std::mutex> lock(done_mutex_);
      done_cv_.wait(lock, [this, n] {
        return workers_ready_.load(std::memory_order_acquire) >= n;
      });
    }

    // setup task for workers
    current_task_ = [&fn](size_t i) { fn(i); };
    task_end_.store(end, std::memory_order_relaxed);
    chunk_counter_.store(begin, std::memory_order_release);
    workers_done_.store(0, std::memory_order_release);
    workers_ready_.store(0, std::memory_order_release);

    // wake workers
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

    // wait for all compute workers to finish this round
    waitComputeDone();

    current_task_ = nullptr;
  }

  /**
   * @brief Parallel for with thread-index based chunking.
   *
   * Each index [0, n_threads) is passed to fn. Useful when work is
   * pre-chunked by thread index (e.g., GEMM row blocks).
   *
   * @tparam F callable with signature void(size_t thread_idx)
   * @param n_threads number of logical threads
   * @param fn function to execute for each thread index
   */
  template <typename F> void parallel_for_chunked(size_t n_threads, F &&fn) {
    parallel_for(0, n_threads, std::forward<F>(fn));
  }

  /**
   * @brief Submit an async task to I/O workers.
   *
   * The task runs on a dedicated I/O worker thread, separate from
   * compute workers. Returns a CompletionToken for synchronization.
   *
   * @param task function to execute asynchronously
   * @return CompletionToken for waiting on completion
   */
  CompletionToken submit(std::function<void()> task);

  /**
   * @brief Get the number of compute worker threads
   * @return number of compute workers
   */
  unsigned int getComputeThreadCount() const {
    return static_cast<unsigned int>(compute_workers_.size());
  }

  /**
   * @brief Get the number of I/O worker threads
   * @return number of I/O workers
   */
  unsigned int getIOThreadCount() const {
    return static_cast<unsigned int>(io_workers_.size());
  }

  /**
   * @brief Configure thread counts before initialization.
   *
   * Must be called before the first call to Global(). Has no effect
   * if ThreadManager is already initialized.
   *
   * @param config thread manager configuration
   */
  static void setConfig(const ThreadManagerConfig &config) {
    pending_config_ = config;
  }

protected:
  /**
   * @brief Default constructor
   */
  ThreadManager();

  /**
   * @brief Initialize compute and I/O worker threads
   */
  void initialize() noexcept override;

private:
  /**
   * @brief Main loop for compute worker threads
   * @param worker_id index of this worker [0, N)
   */
  void computeWorkerLoop(unsigned int worker_id);

  /**
   * @brief Main loop for I/O worker threads
   */
  void ioWorkerLoop();

  /**
   * @brief Wait for all compute workers to finish current parallel_for
   */
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

  // Pending configuration (set before initialization)
  static ThreadManagerConfig pending_config_;
};

} // namespace nntrainer

#endif // __NNTRAINER_THREAD_MANAGER_H__

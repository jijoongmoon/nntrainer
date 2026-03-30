// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   thread_manager.h
 * @date   20 March 2026
 * @brief  Unified thread manager with graph-level execution support
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_THREAD_MANAGER_H__
#define __NNTRAINER_THREAD_MANAGER_H__

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <completion_token.h>
#include <singleton.h>

namespace nntrainer {

struct ThreadManagerConfig {
  unsigned int compute_threads = defaultComputeThreads();
  unsigned int io_threads = 1;
  bool enable_affinity = false;

private:
  static unsigned int defaultComputeThreads() {
#if defined(NNTR_NUM_THREADS) && NNTR_NUM_THREADS > 0
    return NNTR_NUM_THREADS;
#elif defined(OMP_NUM_THREADS) && OMP_NUM_THREADS > 1
    return OMP_NUM_THREADS - 1;
#else
    unsigned int hw = std::thread::hardware_concurrency();
    return hw > 2 ? std::min(hw - 2, 6u) : 1;
#endif
  }
};

/**
 * @class ThreadManager
 * @brief Hybrid thread pool with graph-level execution support.
 *
 * Three execution modes:
 *
 * 1. Per-dispatch (default): each parallel_for = dispatch + barrier.
 *    Condvar-based (safe) or spin-wait (fast, needs affinity).
 *
 * 2. Graph execution (beginGraphExec/endGraphExec):
 *    Workers stay alive in a tight loop. Each parallel_for only does
 *    a barrier — no generation spin, no condvar wake. Like llama.cpp's
 *    ggml_graph_compute: ~0.2us per dispatch vs ~1-4us per-dispatch.
 *
 * Usage:
 *   tm.beginGraphExec();          // wake workers once
 *   for (each layer) {
 *     tm.parallel_for(0, N, fn);  // barrier-only, no dispatch overhead
 *   }
 *   tm.endGraphExec();            // release workers
 */
class ThreadManager : public Singleton<ThreadManager> {
  friend class Singleton<ThreadManager>;

public:
  ~ThreadManager();

  // ─── Graph execution API ───────────────────────────────
  /**
   * @brief Begin graph-level execution. Workers enter a tight spin loop
   * and stay alive until endGraphExec(). All parallel_for calls within
   * a graph exec session use barrier-only synchronization (no dispatch
   * overhead).
   */
  void beginGraphExec();

  /**
   * @brief End graph-level execution. Workers return to idle state.
   */
  void endGraphExec();

  bool inGraphExec() const {
    return in_graph_exec_.load(std::memory_order_acquire);
  }

  // ─── parallel_for API ──────────────────────────────────
  template <typename F> void parallel_for(size_t begin, size_t end, F &&fn) {
    if (begin >= end)
      return;
    if (end - begin == 1 || compute_workers_.empty()) {
      for (size_t i = begin; i < end; ++i)
        fn(i);
      return;
    }
    if (in_graph_exec_.load(std::memory_order_acquire)) {
      graphDispatch(begin, end, std::forward<F>(fn));
    } else {
      standaloneDispatch(begin, end, std::forward<F>(fn));
    }
  }

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
    if (in_graph_exec_.load(std::memory_order_acquire)) {
      graphDispatch(begin, end, std::forward<F>(fn));
    } else {
      standaloneDispatch(begin, end, std::forward<F>(fn), n_workers);
    }
  }

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
  bool isSpinMode() const { return spin_mode_; }

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
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#endif
  }

  // ─── Spin barrier (used in graph exec and spin standalone) ──
  void spinBarrier(bool sense) {
    int n_threads = graph_n_threads_.load(std::memory_order_acquire);
    int n = spin_n_barrier_.fetch_add(1, std::memory_order_acq_rel);
    if (n == n_threads - 1) {
      spin_n_barrier_.store(0, std::memory_order_release);
      spin_barrier_sense_.store(sense, std::memory_order_release);
      return;
    }
    while (spin_barrier_sense_.load(std::memory_order_acquire) != sense) {
      cpuRelax();
    }
  }

  // ─── Graph execution dispatch (barrier-only, no generation spin) ──
  template <typename F>
  void graphDispatch(size_t begin, size_t end, F &&fn) {
    // setup work for workers (they're already spinning on barrier)
    current_task_ = [&fn](size_t i) { fn(i); };
    task_end_ = end;
    current_chunk_.store(begin, std::memory_order_relaxed);
    bool sense = !spin_barrier_sense_.load(std::memory_order_relaxed);

    // signal workers: "work is ready" via barrier
    // Workers are spinning on this barrier from previous round
    spinBarrier(sense);

    // caller does work
    while (true) {
      size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
      if (idx >= end)
        break;
      fn(idx);
    }

    // completion barrier: wait for all workers to finish
    bool sense2 = !spin_barrier_sense_.load(std::memory_order_relaxed);
    spinBarrier(sense2);
    current_task_ = nullptr;
  }

  // ─── Standalone dispatch (per-call, condvar or spin) ──────
  template <typename F>
  void standaloneDispatch(size_t begin, size_t end, F &&fn,
                          unsigned int n_workers = 0) {
    unsigned int total = static_cast<unsigned int>(compute_workers_.size());
    if (n_workers == 0 || n_workers > total)
      n_workers = total;

    if (spin_mode_) {
      // spin-wait standalone (same as before)
      graph_n_threads_.store(static_cast<int>(n_workers + 1),
                             std::memory_order_release);
      bool sense = !spin_barrier_sense_.load(std::memory_order_acquire);
      spin_current_sense_.store(sense, std::memory_order_release);

      current_task_ = [&fn](size_t i) { fn(i); };
      task_end_ = end;
      current_chunk_.store(begin, std::memory_order_relaxed);
      spin_active_workers_.store(n_workers, std::memory_order_release);
      spin_generation_.fetch_add(1, std::memory_order_seq_cst);

      while (true) {
        size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= end)
          break;
        fn(idx);
      }
      spinBarrier(sense);
      current_task_ = nullptr;

    } else {
      // condvar standalone (same as before)
      unsigned int my_barrier_gen;
      {
        std::lock_guard<std::mutex> lock(dispatch_mutex_);
        current_task_ = [&fn](size_t i) { fn(i); };
        task_end_ = end;
        current_chunk_.store(begin, std::memory_order_relaxed);
        cv_active_workers_ = n_workers;
        barrier_target_ = static_cast<int>(n_workers + 1);
        ++dispatch_gen_;
        ++barrier_gen_;
        my_barrier_gen = barrier_gen_;
        barrier_arrived_ = 0;
      }
      dispatch_cv_.notify_all();

      while (true) {
        size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= end)
          break;
        fn(idx);
      }

      {
        std::unique_lock<std::mutex> lock(barrier_mutex_);
        ++barrier_arrived_;
        if (barrier_arrived_ >= barrier_target_) {
          barrier_done_gen_ = my_barrier_gen;
          barrier_cv_.notify_all();
        } else {
          // wait with timeout to detect deadlocks
          if (!barrier_cv_.wait_for(lock, std::chrono::seconds(5),
                                    [this, my_barrier_gen] {
                                      return barrier_done_gen_ >= my_barrier_gen;
                                    })) {
            fprintf(stderr,
                    "[ThreadManager] CONDVAR BARRIER TIMEOUT: "
                    "arrived=%d target=%d barrier_gen=%u done_gen=%u "
                    "my_gen=%u\n",
                    barrier_arrived_, barrier_target_, barrier_gen_,
                    barrier_done_gen_, my_barrier_gen);
            // retry wait (not a hard failure, just diagnostic)
            barrier_cv_.wait(lock, [this, my_barrier_gen] {
              return barrier_done_gen_ >= my_barrier_gen;
            });
          }
        }
      }
      current_task_ = nullptr;
    }
  }

  void computeWorkerLoopSpin(unsigned int worker_id);
  void computeWorkerLoopCondvar(unsigned int worker_id);
  void graphWorkerLoop(unsigned int worker_id);
  void ioWorkerLoop();

  // ─── Mode ───────────────────────────────────────────
  bool spin_mode_{false};

  // ─── Shared ─────────────────────────────────────────
  std::vector<std::thread> compute_workers_;
  std::function<void(size_t)> current_task_;
  size_t task_end_{0};
  alignas(64) std::atomic<size_t> current_chunk_{0};
  alignas(64) std::atomic<bool> stop_{false};

  // ─── Graph execution state ──────────────────────────
  alignas(64) std::atomic<bool> in_graph_exec_{false};
  alignas(64) std::atomic<bool> graph_exec_done_{false};
  alignas(64) std::atomic<int> graph_n_threads_{1};
  alignas(64) std::atomic<unsigned int> graph_sleeping_{0};
  std::mutex graph_mutex_;
  std::condition_variable graph_cv_;
  unsigned int graph_gen_{0};
  std::vector<std::thread> graph_workers_;

  // ─── Spin-wait standalone state ─────────────────────
  alignas(64) std::atomic<unsigned int> spin_generation_{0};
  alignas(64) std::atomic<int> spin_n_barrier_{0};
  alignas(64) std::atomic<bool> spin_barrier_sense_{false};
  alignas(64) std::atomic<bool> spin_current_sense_{false};
  alignas(64) std::atomic<unsigned int> spin_active_workers_{0};

  // ─── Condvar standalone state ───────────────────────
  std::mutex dispatch_mutex_;
  std::condition_variable dispatch_cv_;
  unsigned int dispatch_gen_{0};
  unsigned int cv_active_workers_{0};

  std::mutex barrier_mutex_;
  std::condition_variable barrier_cv_;
  int barrier_arrived_{0};
  int barrier_target_{0};
  unsigned int barrier_gen_{0};
  unsigned int barrier_done_gen_{0};

  // ─── I/O ────────────────────────────────────────────
  std::vector<std::thread> io_workers_;
  std::queue<std::pair<std::function<void()>,
                       std::shared_ptr<CompletionToken::SharedState>>>
    io_queue_;
  std::mutex io_mutex_;
  std::condition_variable io_cv_;

  // ─── Config ─────────────────────────────────────────
  ThreadManagerConfig config_;
  static ThreadManagerConfig pending_config_;
};

} // namespace nntrainer

#endif // __NNTRAINER_THREAD_MANAGER_H__

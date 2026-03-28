// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   thread_manager.cpp
 * @date   20 March 2026
 * @brief  Unified thread manager with graph-level execution
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <thread_manager.h>

#if defined(__linux__) || defined(__ANDROID__)
#include <cstring>
#include <fstream>
#include <sched.h>
#if !defined(__ANDROID__)
#include <pthread.h>
#endif
// Fallback for old Android NDK versions
#if defined(__ANDROID__) && !defined(CPU_SET)
#define CPU_SETSIZE 1024
typedef struct {
  unsigned long __bits[CPU_SETSIZE / (8 * sizeof(long))];
} cpu_set_t;
#define CPU_ZERO(set) memset((set), 0, sizeof(cpu_set_t))
#define CPU_SET(cpu, set)                                                      \
  ((set)->__bits[(cpu) / (8 * sizeof(long))] |=                                \
   (1UL << ((cpu) % (8 * sizeof(long)))))
#endif
#endif

namespace nntrainer {

static bool pinSelfToCore(unsigned int core_id) {
#if defined(__linux__) || defined(__ANDROID__)
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  return sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == 0;
#else
  (void)core_id;
  return true;
#endif
}

static std::vector<unsigned int>
getCoresByPerformance(unsigned int hw_threads) {
  std::vector<std::pair<unsigned long, unsigned int>> freq_core;
  freq_core.reserve(hw_threads);

#if defined(__linux__) || defined(__ANDROID__)
  for (unsigned int i = 0; i < hw_threads; ++i) {
    std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) +
                       "/cpufreq/cpuinfo_max_freq";
    std::ifstream f(path);
    unsigned long freq = 0;
    if (f.is_open())
      f >> freq;
    freq_core.push_back({freq, i});
  }
#endif

  bool has_freq = false;
  for (auto &p : freq_core)
    if (p.first > 0) {
      has_freq = true;
      break;
    }

  if (!has_freq) {
    std::vector<unsigned int> cores(hw_threads);
    for (unsigned int i = 0; i < hw_threads; ++i)
      cores[i] = i;
    return cores;
  }

  std::sort(freq_core.begin(), freq_core.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  std::vector<unsigned int> cores;
  cores.reserve(hw_threads);
  for (auto &p : freq_core)
    cores.push_back(p.second);
  return cores;
}

ThreadManagerConfig ThreadManager::pending_config_ = {};

ThreadManager::ThreadManager() {}

ThreadManager::~ThreadManager() {
  // end graph exec if active
  if (in_graph_exec_.load(std::memory_order_acquire))
    endGraphExec();

  stop_.store(true, std::memory_order_release);

  if (spin_mode_) {
    spin_generation_.fetch_add(1, std::memory_order_seq_cst);
  } else {
    dispatch_cv_.notify_all();
  }

  for (auto &t : compute_workers_)
    if (t.joinable())
      t.join();

  io_cv_.notify_all();
  for (auto &t : io_workers_)
    if (t.joinable())
      t.join();
}

void ThreadManager::initialize() noexcept {
  config_ = pending_config_;
  auto config = config_;

  unsigned int hw_threads = std::thread::hardware_concurrency();
  if (hw_threads == 0)
    hw_threads = 1;

  unsigned int available = hw_threads > 1 ? hw_threads - 1 : 1;
  if (config.compute_threads > available)
    config.compute_threads = available;

  unsigned int remaining = available > config.compute_threads
                             ? available - config.compute_threads
                             : 0;
  if (config.io_threads > remaining)
    config.io_threads = remaining > 0 ? remaining : 1;

  spin_mode_ = config.enable_affinity;

  // compute core map
  std::vector<unsigned int> core_map;
  if (config.enable_affinity)
    core_map = getCoresByPerformance(hw_threads);

  // start compute workers
  compute_workers_.reserve(config.compute_threads);
  for (unsigned int i = 0; i < config.compute_threads; ++i) {
    int core_id = config.enable_affinity
                    ? static_cast<int>(core_map[(i + 1) % core_map.size()])
                    : -1;
    if (spin_mode_) {
      compute_workers_.emplace_back([this, i, core_id] {
        if (core_id >= 0)
          pinSelfToCore(static_cast<unsigned int>(core_id));
        computeWorkerLoopSpin(i);
      });
    } else {
      compute_workers_.emplace_back([this, i, core_id] {
        if (core_id >= 0)
          pinSelfToCore(static_cast<unsigned int>(core_id));
        computeWorkerLoopCondvar(i);
      });
    }
  }

  // start I/O workers
  unsigned int io_core_start = config.compute_threads + 1;
  io_workers_.reserve(config.io_threads);
  for (unsigned int i = 0; i < config.io_threads; ++i) {
    int core_id = config.enable_affinity
                    ? static_cast<int>(
                        core_map[(io_core_start + i) % core_map.size()])
                    : -1;
    io_workers_.emplace_back([this, core_id] {
      if (core_id >= 0)
        pinSelfToCore(static_cast<unsigned int>(core_id));
      ioWorkerLoop();
    });
  }
}

// ─── GRAPH EXECUTION ─────────────────────────────────────────────
//
// beginGraphExec(): spawn temporary graph workers that spin-wait on
// barriers. Between barriers they do work. The caller drives the loop:
//
//   beginGraphExec()
//   for (each layer):
//     tm.parallel_for(...)  → graphDispatch: barrier(ready) → work → barrier(done)
//   endGraphExec()
//
// This eliminates generation spin between dispatches (~0.2us vs ~1-4us).

void ThreadManager::beginGraphExec() {
  if (in_graph_exec_.load(std::memory_order_acquire))
    return;

  unsigned int n = static_cast<unsigned int>(compute_workers_.size());
  graph_n_threads_.store(static_cast<int>(n + 1), std::memory_order_release);
  spin_n_barrier_.store(0, std::memory_order_release);
  spin_barrier_sense_.store(false, std::memory_order_release);

  graph_exec_done_.store(false, std::memory_order_release);
  in_graph_exec_.store(true, std::memory_order_release);

  // spawn dedicated graph worker threads
  graph_workers_.reserve(n);
  for (unsigned int i = 0; i < n; ++i) {
    graph_workers_.emplace_back([this, i] { graphWorkerLoop(i); });
  }
}

void ThreadManager::endGraphExec() {
  if (!in_graph_exec_.load(std::memory_order_acquire))
    return;

  // signal graph workers to exit
  graph_exec_done_.store(true, std::memory_order_release);

  // do a final barrier to release workers from their spin
  bool sense = !spin_barrier_sense_.load(std::memory_order_relaxed);
  spinBarrier(sense);

  for (auto &t : graph_workers_)
    if (t.joinable())
      t.join();
  graph_workers_.clear();

  in_graph_exec_.store(false, std::memory_order_release);
}

void ThreadManager::graphWorkerLoop(unsigned int /*worker_id*/) {
  while (true) {
    // ── Barrier 1: wait for work to be ready ──
    bool sense1 = !spin_barrier_sense_.load(std::memory_order_acquire);
    spinBarrier(sense1);

    // check exit
    if (graph_exec_done_.load(std::memory_order_acquire))
      return;

    // ── Do work (grab chunks via atomic counter) ──
    size_t end = task_end_;
    while (true) {
      size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
      if (idx >= end)
        break;
      current_task_(idx);
    }

    // ── Barrier 2: signal work completion ──
    bool sense2 = !spin_barrier_sense_.load(std::memory_order_acquire);
    spinBarrier(sense2);
  }
}

// ─── SPIN-WAIT STANDALONE WORKER ─────────────────────────────────

void ThreadManager::computeWorkerLoopSpin(unsigned int worker_id) {
  unsigned int my_gen = spin_generation_.load(std::memory_order_acquire);

  while (true) {
    while (spin_generation_.load(std::memory_order_acquire) == my_gen) {
      if (stop_.load(std::memory_order_acquire))
        return;
      cpuRelax();
    }
    my_gen = spin_generation_.load(std::memory_order_acquire);

    if (stop_.load(std::memory_order_acquire))
      return;

    if (worker_id < spin_active_workers_.load(std::memory_order_acquire)) {
      bool sense = spin_current_sense_.load(std::memory_order_acquire);
      size_t end = task_end_;
      while (true) {
        size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= end)
          break;
        current_task_(idx);
      }
      spinBarrier(sense);
    }
  }
}

// ─── CONDVAR STANDALONE WORKER ───────────────────────────────────

void ThreadManager::computeWorkerLoopCondvar(unsigned int worker_id) {
  unsigned int my_gen = 0;

  while (true) {
    unsigned int my_barrier_gen;
    {
      std::unique_lock<std::mutex> lock(dispatch_mutex_);
      dispatch_cv_.wait(lock, [this, &my_gen] {
        return dispatch_gen_ != my_gen ||
               stop_.load(std::memory_order_acquire);
      });
      if (stop_.load(std::memory_order_acquire))
        return;
      my_gen = dispatch_gen_;
      my_barrier_gen = barrier_gen_;
    }

    if (worker_id < cv_active_workers_) {
      size_t end = task_end_;
      while (true) {
        size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= end)
          break;
        current_task_(idx);
      }

      {
        std::unique_lock<std::mutex> lock(barrier_mutex_);
        ++barrier_arrived_;
        if (barrier_arrived_ >= barrier_target_) {
          barrier_done_gen_ = my_barrier_gen;
          barrier_cv_.notify_all();
        } else {
          barrier_cv_.wait(lock, [this, my_barrier_gen] {
            return barrier_done_gen_ >= my_barrier_gen;
          });
        }
      }
    }
  }
}

// ─── I/O WORKER ──────────────────────────────────────────────────

void ThreadManager::ioWorkerLoop() {
  while (true) {
    std::pair<std::function<void()>,
              std::shared_ptr<CompletionToken::SharedState>>
      item;
    {
      std::unique_lock<std::mutex> lock(io_mutex_);
      io_cv_.wait(lock, [this] {
        return !io_queue_.empty() || stop_.load(std::memory_order_acquire);
      });
      if (stop_.load(std::memory_order_acquire) && io_queue_.empty())
        return;
      item = std::move(io_queue_.front());
      io_queue_.pop();
    }

    auto &task = item.first;
    auto &state = item.second;

    try {
      task();
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->done.store(true, std::memory_order_release);
      }
      state->cv.notify_all();
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->exception = std::current_exception();
        state->done.store(true, std::memory_order_release);
      }
      state->cv.notify_all();
    }
  }
}

CompletionToken ThreadManager::submit(std::function<void()> task) {
  CompletionToken token = CompletionToken::create();
  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    io_queue_.push({std::move(task), token.getState()});
  }
  io_cv_.notify_one();
  return token;
}

} // namespace nntrainer

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   thread_manager.cpp
 * @date   20 March 2026
 * @brief  Unified thread manager implementation
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <thread_manager.h>

#if defined(__linux__)
#include <fstream>
#include <pthread.h>
#include <sched.h>
#endif

namespace nntrainer {

/**
 * @brief Pin a thread to a specific CPU core.
 */
static bool pinThreadToCore(std::thread &thread, unsigned int core_id) {
#if defined(__linux__)
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  int rc =
    pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);
  return rc == 0;
#else
  (void)thread;
  (void)core_id;
  return true;
#endif
}

/**
 * @brief Get sorted core indices by max frequency (descending).
 *
 * On big.LITTLE (e.g. Android), returns big cores first, then LITTLE.
 * On homogeneous systems, returns cores in order (0, 1, 2, ...).
 *
 * @param hw_threads total number of hardware threads
 * @return vector of core indices sorted by performance (fastest first)
 */
static std::vector<unsigned int>
getCoresByPerformance(unsigned int hw_threads) {
  std::vector<std::pair<unsigned long, unsigned int>> freq_core;
  freq_core.reserve(hw_threads);

#if defined(__linux__)
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

  // check if we got any frequency info
  bool has_freq = false;
  for (auto &p : freq_core) {
    if (p.first > 0) {
      has_freq = true;
      break;
    }
  }

  if (!has_freq) {
    // homogeneous: just return 0, 1, 2, ...
    std::vector<unsigned int> cores(hw_threads);
    for (unsigned int i = 0; i < hw_threads; ++i)
      cores[i] = i;
    return cores;
  }

  // sort by frequency descending (big cores first)
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
  stop_.store(true, std::memory_order_release);

  // wake compute workers so they can exit
  {
    std::lock_guard<std::mutex> lock(compute_mutex_);
    compute_generation_.fetch_add(1, std::memory_order_release);
  }
  compute_cv_.notify_all();

  // wake I/O workers so they can exit
  io_cv_.notify_all();

  for (auto &t : compute_workers_) {
    if (t.joinable())
      t.join();
  }
  for (auto &t : io_workers_) {
    if (t.joinable())
      t.join();
  }
}

void ThreadManager::initialize() noexcept {
  config_ = pending_config_;
  auto config = config_;

  unsigned int hw_threads = std::thread::hardware_concurrency();
  if (hw_threads == 0)
    hw_threads = 1;

  // Reserve cores: 1 for caller + io_threads for I/O
  unsigned int reserved = 1 + config.io_threads;
  unsigned int max_compute =
    hw_threads > reserved ? hw_threads - reserved : 1;
  if (config.compute_threads > max_compute)
    config.compute_threads = max_compute;

  // start compute workers
  compute_workers_.reserve(config.compute_threads);
  for (unsigned int i = 0; i < config.compute_threads; ++i) {
    compute_workers_.emplace_back([this, i] { computeWorkerLoop(i); });
  }

  // start I/O workers
  io_workers_.reserve(config.io_threads);
  for (unsigned int i = 0; i < config.io_threads; ++i) {
    io_workers_.emplace_back([this] { ioWorkerLoop(); });
  }

  // Pin threads to cores sorted by performance (big cores first).
  //
  // On big.LITTLE (Android):
  //   sorted = [big_0, big_1, big_2, big_3, LITTLE_0, LITTLE_1, ...]
  //   caller  → sorted[0] (fastest big core)
  //   compute → sorted[1..N] (remaining big cores)
  //   I/O     → sorted[N+1..N+M] (big if available, else LITTLE)
  //
  // On homogeneous:
  //   sorted = [0, 1, 2, 3, ...]
  //   Same layout, just sequential cores.
  if (config.enable_affinity) {
    auto sorted_cores = getCoresByPerformance(hw_threads);

    // sorted[0] = fastest core → caller (not pinned here)
    // sorted[1..N] → compute workers
    for (unsigned int i = 0; i < compute_workers_.size(); ++i) {
      unsigned int idx = (i + 1) % sorted_cores.size();
      pinThreadToCore(compute_workers_[i], sorted_cores[idx]);
    }

    // sorted[N+1..N+M] → I/O workers (overflow to slower cores)
    unsigned int io_start =
      static_cast<unsigned int>(compute_workers_.size()) + 1;
    for (unsigned int i = 0; i < io_workers_.size(); ++i) {
      unsigned int idx = (io_start + i) % sorted_cores.size();
      pinThreadToCore(io_workers_[i], sorted_cores[idx]);
    }
  }
}

void ThreadManager::computeWorkerLoop(unsigned int worker_id) {
  unsigned int my_gen = compute_generation_.load(std::memory_order_acquire);
  const auto policy = config_.wait_policy;
  const unsigned int spin_count = config_.spin_count;

  // signal initial readiness
  workers_ready_.fetch_add(1, std::memory_order_release);
  done_cv_.notify_one();

  while (true) {
    // ─── Wait for new work: Spin → Yield → Sleep ───
    bool got_work = false;

    // Phase 1: Spin (lowest latency, for inference hot path)
    if (policy == WaitPolicy::Spin || policy == WaitPolicy::Adaptive) {
      for (unsigned int s = 0; s < spin_count; ++s) {
        if (compute_generation_.load(std::memory_order_acquire) != my_gen ||
            stop_.load(std::memory_order_acquire)) {
          got_work = true;
          break;
        }
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield");
#endif
      }
    }

    // Phase 2: Yield (give up timeslice but stay runnable)
    if (!got_work &&
        (policy == WaitPolicy::Spin || policy == WaitPolicy::Adaptive)) {
      for (unsigned int y = 0; y < 32; ++y) {
        if (compute_generation_.load(std::memory_order_acquire) != my_gen ||
            stop_.load(std::memory_order_acquire)) {
          got_work = true;
          break;
        }
        std::this_thread::yield();
      }
    }

    // Phase 3: Sleep on condition variable (saves CPU)
    if (!got_work) {
      std::unique_lock<std::mutex> lock(compute_mutex_);
      compute_cv_.wait(lock, [this, &my_gen] {
        return compute_generation_.load(std::memory_order_acquire) != my_gen ||
               stop_.load(std::memory_order_acquire);
      });
    }

    my_gen = compute_generation_.load(std::memory_order_acquire);

    if (stop_.load(std::memory_order_acquire)) {
      workers_done_.fetch_add(1, std::memory_order_release);
      done_cv_.notify_one();
      return;
    }

    // only active workers do real work
    if (worker_id < active_workers_.load(std::memory_order_acquire)) {
      size_t end = task_end_.load(std::memory_order_relaxed);
      while (true) {
        size_t idx = chunk_counter_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= end)
          break;
        current_task_(idx);
      }
    }

    // all workers signal done + readiness
    workers_done_.fetch_add(1, std::memory_order_release);
    done_cv_.notify_one();
    workers_ready_.fetch_add(1, std::memory_order_release);
    done_cv_.notify_one();
  }
}

void ThreadManager::waitComputeDone() {
  unsigned int n = static_cast<unsigned int>(compute_workers_.size());
  std::unique_lock<std::mutex> lock(done_mutex_);
  done_cv_.wait(lock, [this, n] {
    return workers_done_.load(std::memory_order_acquire) >= n;
  });
}

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

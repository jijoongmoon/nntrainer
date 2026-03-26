// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   thread_manager.cpp
 * @date   20 March 2026
 * @brief  Unified thread manager implementation (GGML-style barrier)
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <thread_manager.h>

#if defined(__linux__)
#include <fstream>
#include <pthread.h>
#include <sched.h>
#endif

namespace nntrainer {

static bool pinThreadToCore(std::thread &thread, unsigned int core_id) {
#if defined(__linux__)
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  return pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t),
                                &cpuset) == 0;
#else
  (void)thread;
  (void)core_id;
  return true;
#endif
}

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
  stop_.store(true, std::memory_order_release);
  generation_.fetch_add(1, std::memory_order_seq_cst);

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

  // start compute workers
  compute_workers_.reserve(config.compute_threads);
  for (unsigned int i = 0; i < config.compute_threads; ++i)
    compute_workers_.emplace_back([this, i] { computeWorkerLoop(i); });

  // start I/O workers
  io_workers_.reserve(config.io_threads);
  for (unsigned int i = 0; i < config.io_threads; ++i)
    io_workers_.emplace_back([this] { ioWorkerLoop(); });

  // CPU affinity
  if (config.enable_affinity) {
    auto sorted = getCoresByPerformance(hw_threads);
    for (unsigned int i = 0; i < compute_workers_.size(); ++i)
      pinThreadToCore(compute_workers_[i], sorted[(i + 1) % sorted.size()]);
    unsigned int io_start =
      static_cast<unsigned int>(compute_workers_.size()) + 1;
    for (unsigned int i = 0; i < io_workers_.size(); ++i)
      pinThreadToCore(io_workers_[i], sorted[(io_start + i) % sorted.size()]);
  }
}

void ThreadManager::computeWorkerLoop(unsigned int worker_id) {
  unsigned int my_gen = generation_.load(std::memory_order_acquire);

  while (true) {
    // spin-wait for new generation with periodic yield for OS scheduling
    unsigned int spin_count = 0;
    while (generation_.load(std::memory_order_acquire) == my_gen) {
      if (stop_.load(std::memory_order_acquire))
        return;
      cpuRelax();
      if (++spin_count > 1024) {
        std::this_thread::yield();
        spin_count = 0;
      }
    }
    my_gen = generation_.load(std::memory_order_acquire);

    if (stop_.load(std::memory_order_acquire))
      return;

    // only active workers do work + barrier; inactive workers skip both
    if (worker_id < active_workers_.load(std::memory_order_acquire)) {
      size_t end = task_end_;
      while (true) {
        size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= end)
          break;
        current_task_(idx);
      }

      bool sense = current_sense_.load(std::memory_order_acquire);
      barrier(sense);
    }
    // inactive workers loop back to generation spin immediately
  }
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

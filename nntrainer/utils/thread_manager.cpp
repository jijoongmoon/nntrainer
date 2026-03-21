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
#include <pthread.h>
#include <sched.h>
#endif

namespace nntrainer {

/**
 * @brief Pin a thread to a specific CPU core.
 * @param thread the thread to pin
 * @param core_id the core index to pin to
 * @return true if successful or not supported on this platform
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
  return true; // no-op on non-Linux
#endif
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
  auto config = pending_config_;

  unsigned int hw_threads = std::thread::hardware_concurrency();
  if (hw_threads == 0)
    hw_threads = 1;

  // cap compute threads to physical cores minus 1 (caller thread uses a core)
  unsigned int max_workers = hw_threads > 1 ? hw_threads - 1 : 1;
  if (config.compute_threads > max_workers)
    config.compute_threads = max_workers;

  // start compute workers
  compute_workers_.reserve(config.compute_threads);
  for (unsigned int i = 0; i < config.compute_threads; ++i) {
    compute_workers_.emplace_back([this, i] { computeWorkerLoop(i); });
  }

  // pin compute workers to cores (worker i → core i+1, core 0 for caller)
  if (config.enable_affinity) {
    for (unsigned int i = 0; i < compute_workers_.size(); ++i) {
      unsigned int core = (i + 1) % hw_threads;
      pinThreadToCore(compute_workers_[i], core);
    }
  }

  // start I/O workers (not pinned — they do blocking I/O)
  io_workers_.reserve(config.io_threads);
  for (unsigned int i = 0; i < config.io_threads; ++i) {
    io_workers_.emplace_back([this] { ioWorkerLoop(); });
  }
}

void ThreadManager::computeWorkerLoop(unsigned int worker_id) {
  unsigned int my_gen = compute_generation_.load(std::memory_order_acquire);

  // signal initial readiness
  workers_ready_.fetch_add(1, std::memory_order_release);
  done_cv_.notify_one();

  while (true) {
    // wait for new work (generation change) or stop
    {
      std::unique_lock<std::mutex> lock(compute_mutex_);
      compute_cv_.wait(lock, [this, &my_gen] {
        return compute_generation_.load(std::memory_order_acquire) != my_gen ||
               stop_.load(std::memory_order_acquire);
      });
      my_gen = compute_generation_.load(std::memory_order_acquire);
    }

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

    // all workers (active or not) signal done
    workers_done_.fetch_add(1, std::memory_order_release);
    done_cv_.notify_one();

    // signal readiness for next round
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

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

namespace nntrainer {

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
}

void ThreadManager::computeWorkerLoop(unsigned int worker_id) {
  (void)worker_id;
  unsigned int my_gen = compute_generation_.load(std::memory_order_acquire);

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
      // signal done so destructor's waitComputeDone unblocks
      workers_done_.fetch_add(1, std::memory_order_release);
      done_cv_.notify_one();
      return;
    }

    // grab chunks via atomic counter
    size_t end = task_end_.load(std::memory_order_relaxed);
    while (true) {
      size_t idx = chunk_counter_.fetch_add(1, std::memory_order_relaxed);
      if (idx >= end)
        break;
      current_task_(idx);
    }

    // signal that this worker is done
    workers_done_.fetch_add(1, std::memory_order_release);
    done_cv_.notify_one();
  }
}

void ThreadManager::waitComputeDone() {
  unsigned int n = static_cast<unsigned int>(compute_workers_.size());
  std::unique_lock<std::mutex> lock(done_mutex_);
  done_cv_.wait(lock, [this, n] {
    return workers_done_.load(std::memory_order_acquire) >= n;
  });
  workers_done_.store(0, std::memory_order_release);
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

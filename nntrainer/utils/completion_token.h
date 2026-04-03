// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   completion_token.h
 * @date   20 March 2026
 * @brief  Completion token for asynchronous task synchronization
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __NNTRAINER_COMPLETION_TOKEN_H__
#define __NNTRAINER_COMPLETION_TOKEN_H__

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>

namespace nntrainer {

/**
 * @class CompletionToken
 * @brief One-shot synchronization token for async tasks.
 *
 * Unlike task IDs that can be reused and cause race conditions,
 * each CompletionToken is a unique handle tied to exactly one task.
 * Calling wait() is guaranteed to either block until the task completes
 * or throw if the task failed.
 */
class CompletionToken {
public:
  /**
   * @brief Default constructor creates an empty (already-done) token
   */
  CompletionToken() : state_(nullptr) {}

  /**
   * @brief Wait for task completion. Blocks until done.
   * @throws Re-throws any exception from the task
   */
  void wait() {
    if (!state_)
      return;
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->cv.wait(lock,
                    [this] { return state_->done.load(std::memory_order_acquire); });
    if (state_->exception)
      std::rethrow_exception(state_->exception);
  }

  /**
   * @brief Non-blocking check if task is done
   * @return true if task has completed
   */
  bool isDone() const {
    if (!state_)
      return true;
    return state_->done.load(std::memory_order_acquire);
  }

  /**
   * @brief Wait with timeout
   * @param timeout duration to wait
   * @return true if task completed within timeout
   */
  template <typename Rep, typename Period>
  bool waitFor(const std::chrono::duration<Rep, Period> &timeout) {
    if (!state_)
      return true;
    std::unique_lock<std::mutex> lock(state_->mutex);
    return state_->cv.wait_for(
      lock, timeout,
      [this] { return state_->done.load(std::memory_order_acquire); });
  }

  /**
   * @brief Check if token is valid (associated with a task)
   * @return true if token is associated with a task
   */
  bool valid() const { return state_ != nullptr; }

private:
  friend class ThreadManager;

  struct SharedState {
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> done{false};
    std::exception_ptr exception{nullptr};
  };

  /**
   * @brief Create a token with a new shared state (called by ThreadManager)
   */
  static CompletionToken create() {
    CompletionToken token;
    token.state_ = std::make_shared<SharedState>();
    return token;
  }

  /**
   * @brief Mark the task as completed successfully
   */
  void complete() {
    if (!state_)
      return;
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->done.store(true, std::memory_order_release);
    }
    state_->cv.notify_all();
  }

  /**
   * @brief Mark the task as failed with an exception
   * @param e the exception pointer
   */
  void fail(std::exception_ptr e) {
    if (!state_)
      return;
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->exception = e;
      state_->done.store(true, std::memory_order_release);
    }
    state_->cv.notify_all();
  }

  /**
   * @brief Get the shared state (for ThreadManager internal use)
   */
  std::shared_ptr<SharedState> getState() const { return state_; }

  std::shared_ptr<SharedState> state_;
};

} // namespace nntrainer

#endif // __NNTRAINER_COMPLETION_TOKEN_H__

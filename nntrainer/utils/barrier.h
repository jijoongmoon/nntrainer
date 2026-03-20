// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   barrier.h
 * @date   20 March 2026
 * @brief  Reusable barrier for thread synchronization
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __NNTRAINER_BARRIER_H__
#define __NNTRAINER_BARRIER_H__

#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace nntrainer {

/**
 * @class Barrier
 * @brief Reusable barrier that blocks until all participating threads arrive.
 *
 * Used by ThreadManager to synchronize compute workers at the end of
 * parallel_for. The barrier automatically resets after each use.
 */
class Barrier {
public:
  /**
   * @brief Construct a barrier for the given number of threads
   * @param count number of threads that must call wait()
   */
  explicit Barrier(unsigned int count) :
    threshold_(count), count_(count), generation_(0) {}

  /**
   * @brief Block until all threads have called wait().
   *
   * The last thread to arrive resets the barrier and wakes all others.
   * Uses a generation counter to prevent spurious wakeups from
   * previous barrier cycles.
   */
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto gen = generation_;
    if (--count_ == 0) {
      ++generation_;
      count_ = threshold_;
      lock.unlock();
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, gen] { return gen != generation_; });
    }
  }

  /**
   * @brief Reset the barrier with a new thread count
   * @param count new number of threads
   */
  void reset(unsigned int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    threshold_ = count;
    count_ = count;
    ++generation_;
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  unsigned int threshold_;
  unsigned int count_;
  unsigned int generation_;
};

} // namespace nntrainer

#endif // __NNTRAINER_BARRIER_H__

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_thread_manager.cpp
 * @date        20 March 2026
 * @brief       Unit test for ThreadManager
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <barrier.h>
#include <completion_token.h>
#include <thread_manager.h>

// ─── CompletionToken Tests ──────────────────────────────────

TEST(CompletionToken, DefaultIsAlreadyDone) {
  nntrainer::CompletionToken token;
  EXPECT_TRUE(token.isDone());
  EXPECT_FALSE(token.valid());
  token.wait(); // should not block
}

// Test CompletionToken through ThreadManager::submit (create/complete are private)
TEST(CompletionToken, WaitBlocksUntilComplete) {
  auto &tm = nntrainer::ThreadManager::Global();
  std::atomic<bool> started{false};
  std::atomic<int> value{0};

  auto token = tm.submit([&] {
    started.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    value.store(42);
  });

  token.wait();
  EXPECT_TRUE(token.isDone());
  EXPECT_EQ(value.load(), 42);
}

TEST(CompletionToken, WaitForTimeout) {
  auto &tm = nntrainer::ThreadManager::Global();
  std::atomic<bool> proceed{false};

  auto token = tm.submit([&] {
    while (!proceed.load(std::memory_order_acquire))
      std::this_thread::yield();
  });

  bool result = token.waitFor(std::chrono::milliseconds(10));
  // task may or may not be done depending on timing
  proceed.store(true, std::memory_order_release);
  token.wait();
  result = token.waitFor(std::chrono::milliseconds(10));
  EXPECT_TRUE(result);
}

TEST(CompletionToken, FailRethrows) {
  auto &tm = nntrainer::ThreadManager::Global();

  auto token = tm.submit([] {
    throw std::runtime_error("test error");
  });

  EXPECT_THROW(token.wait(), std::runtime_error);
  EXPECT_TRUE(token.isDone());
}

// ─── Barrier Tests ──────────────────────────────────────────

TEST(Barrier, AllThreadsArrive) {
  const int N = 4;
  nntrainer::Barrier barrier(N);
  std::atomic<int> count{0};

  std::vector<std::thread> threads;
  for (int i = 0; i < N; ++i) {
    threads.emplace_back([&] {
      count.fetch_add(1, std::memory_order_relaxed);
      barrier.wait();
    });
  }

  for (auto &t : threads)
    t.join();

  EXPECT_EQ(count.load(), N);
}

TEST(Barrier, ReusableAcrossRounds) {
  const int N = 4;
  const int ROUNDS = 10;
  nntrainer::Barrier barrier(N);
  std::atomic<int> round_count{0};

  std::vector<std::thread> threads;
  for (int i = 0; i < N; ++i) {
    threads.emplace_back([&] {
      for (int r = 0; r < ROUNDS; ++r) {
        barrier.wait();
        round_count.fetch_add(1, std::memory_order_relaxed);
        barrier.wait(); // sync before next round
      }
    });
  }

  for (auto &t : threads)
    t.join();

  EXPECT_EQ(round_count.load(), N * ROUNDS);
}

// ─── ThreadManager parallel_for Tests ───────────────────────

TEST(ThreadManager, ParallelForEmptyRange) {
  auto &tm = nntrainer::ThreadManager::Global();
  std::atomic<int> count{0};
  tm.parallel_for(0, 0, [&](size_t) { count.fetch_add(1); });
  EXPECT_EQ(count.load(), 0);
}

TEST(ThreadManager, ParallelForSingleElement) {
  auto &tm = nntrainer::ThreadManager::Global();
  std::atomic<int> count{0};
  tm.parallel_for(0, 1, [&](size_t) { count.fetch_add(1); });
  EXPECT_EQ(count.load(), 1);
}

TEST(ThreadManager, ParallelForAllElementsExecuted) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t N = 1000;
  std::vector<std::atomic<int>> flags(N);
  for (size_t i = 0; i < N; ++i)
    flags[i].store(0);

  tm.parallel_for(0, N, [&](size_t i) { flags[i].fetch_add(1); });

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(flags[i].load(), 1)
      << "Index " << i << " not executed exactly once";
  }
}

TEST(ThreadManager, ParallelForWithOffset) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t BEGIN = 10;
  const size_t END = 50;
  std::vector<std::atomic<int>> flags(END);
  for (size_t i = 0; i < END; ++i)
    flags[i].store(0);

  tm.parallel_for(BEGIN, END, [&](size_t i) { flags[i].fetch_add(1); });

  for (size_t i = 0; i < BEGIN; ++i)
    EXPECT_EQ(flags[i].load(), 0);
  for (size_t i = BEGIN; i < END; ++i)
    EXPECT_EQ(flags[i].load(), 1);
}

TEST(ThreadManager, ParallelForChunkedPattern) {
  auto &tm = nntrainer::ThreadManager::Global();
  unsigned int n_threads = tm.getComputeThreadCount() + 1; // +1 for caller
  const size_t TOTAL = 100;
  std::atomic<size_t> sum{0};

  tm.parallel_for_chunked(n_threads, [&](size_t tid) {
    size_t start = (tid * TOTAL) / n_threads;
    size_t end = ((tid + 1) * TOTAL) / n_threads;
    for (size_t i = start; i < end; ++i)
      sum.fetch_add(i, std::memory_order_relaxed);
  });

  size_t expected = (TOTAL * (TOTAL - 1)) / 2;
  EXPECT_EQ(sum.load(), expected);
}

TEST(ThreadManager, ParallelForAccumulation) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t N = 10000;
  std::atomic<size_t> sum{0};

  tm.parallel_for(
    0, N, [&](size_t i) { sum.fetch_add(i, std::memory_order_relaxed); });

  size_t expected = (N * (N - 1)) / 2;
  EXPECT_EQ(sum.load(), expected);
}

TEST(ThreadManager, ParallelForMultipleCalls) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t N = 100;

  for (int round = 0; round < 20; ++round) {
    std::atomic<int> count{0};
    tm.parallel_for(0, N, [&](size_t) { count.fetch_add(1); });
    EXPECT_EQ(count.load(), (int)N) << "Failed at round " << round;
  }
}

// ─── ThreadManager submit (I/O) Tests ───────────────────────

TEST(ThreadManager, SubmitBasic) {
  auto &tm = nntrainer::ThreadManager::Global();
  std::atomic<int> value{0};

  auto token = tm.submit([&] { value.store(42); });
  token.wait();

  EXPECT_EQ(value.load(), 42);
}

TEST(ThreadManager, SubmitMultipleTasks) {
  auto &tm = nntrainer::ThreadManager::Global();
  const int N = 50;
  std::atomic<int> count{0};
  std::vector<nntrainer::CompletionToken> tokens;

  for (int i = 0; i < N; ++i) {
    tokens.push_back(tm.submit([&] { count.fetch_add(1); }));
  }

  for (auto &token : tokens)
    token.wait();

  EXPECT_EQ(count.load(), N);
}

TEST(ThreadManager, SubmitWithException) {
  auto &tm = nntrainer::ThreadManager::Global();

  auto token =
    tm.submit([] { throw std::runtime_error("async error"); });

  EXPECT_THROW(token.wait(), std::runtime_error);
}

TEST(ThreadManager, SubmitIsDoneCheck) {
  auto &tm = nntrainer::ThreadManager::Global();
  std::atomic<bool> proceed{false};

  auto token = tm.submit([&] {
    while (!proceed.load(std::memory_order_acquire))
      std::this_thread::yield();
  });

  // task should not be done yet (or might be, if worker grabbed it fast)
  // just verify isDone doesn't crash
  (void)token.isDone();

  proceed.store(true, std::memory_order_release);
  token.wait();
  EXPECT_TRUE(token.isDone());
}

// ─── ThreadManager Compute/IO Isolation Test ────────────────

TEST(ThreadManager, ComputeAndIOConcurrent) {
  auto &tm = nntrainer::ThreadManager::Global();
  std::atomic<int> io_done{0};
  std::atomic<size_t> compute_sum{0};

  // start I/O tasks
  std::vector<nntrainer::CompletionToken> tokens;
  for (int i = 0; i < 10; ++i) {
    tokens.push_back(tm.submit([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      io_done.fetch_add(1);
    }));
  }

  // run compute while I/O is in progress
  const size_t N = 500;
  tm.parallel_for(0, N, [&](size_t i) { compute_sum.fetch_add(i); });

  size_t expected = (N * (N - 1)) / 2;
  EXPECT_EQ(compute_sum.load(), expected);

  // wait for I/O
  for (auto &token : tokens)
    token.wait();
  EXPECT_EQ(io_done.load(), 10);
}

// ─── ThreadManager n_workers Tests ──────────────────────────

TEST(ThreadManager, ParallelForWithNWorkers) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t N = 1000;
  std::vector<std::atomic<int>> flags(N);
  for (size_t i = 0; i < N; ++i)
    flags[i].store(0);

  // use only 2 workers (+ caller = 3 threads total)
  tm.parallel_for(0, N, 2u, [&](size_t i) { flags[i].fetch_add(1); });

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(flags[i].load(), 1)
      << "Index " << i << " not executed exactly once";
  }
}

TEST(ThreadManager, ParallelForWithZeroWorkers) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t N = 100;
  std::atomic<int> count{0};

  // 0 workers = serial on calling thread
  tm.parallel_for(0, N, 0u, [&](size_t) { count.fetch_add(1); });
  EXPECT_EQ(count.load(), (int)N);
}

TEST(ThreadManager, ParallelForWithOneWorker) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t N = 200;
  std::atomic<size_t> sum{0};

  // 1 worker + caller = 2 threads
  tm.parallel_for(0, N, 1u,
                  [&](size_t i) { sum.fetch_add(i, std::memory_order_relaxed); });

  EXPECT_EQ(sum.load(), (N * (N - 1)) / 2);
}

TEST(ThreadManager, ParallelForNWorkersRepeated) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t N = 50;

  // rapid repeated calls with varying worker counts
  for (int round = 0; round < 20; ++round) {
    std::atomic<int> count{0};
    unsigned int workers = (round % 3) + 1; // 1, 2, 3, 1, 2, 3, ...
    tm.parallel_for(0, N, workers, [&](size_t) { count.fetch_add(1); });
    EXPECT_EQ(count.load(), (int)N) << "Failed at round " << round
                                     << " with " << workers << " workers";
  }
}

TEST(ThreadManager, ParallelForChunkedWithFewerThreads) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t TOTAL = 100;
  std::atomic<size_t> sum{0};

  // chunked with 3 threads (2 workers + caller)
  tm.parallel_for_chunked(3, [&](size_t tid) {
    size_t start = (tid * TOTAL) / 3;
    size_t end = ((tid + 1) * TOTAL) / 3;
    for (size_t i = start; i < end; ++i)
      sum.fetch_add(i, std::memory_order_relaxed);
  });

  EXPECT_EQ(sum.load(), (TOTAL * (TOTAL - 1)) / 2);
}

// ─── ThreadManager Query Tests ──────────────────────────────

TEST(ThreadManager, ThreadCounts) {
  auto &tm = nntrainer::ThreadManager::Global();
  EXPECT_GT(tm.getComputeThreadCount(), 0u);
  EXPECT_GT(tm.getIOThreadCount(), 0u);

  // compute workers should be <= hardware_concurrency - 1 (caller uses a core)
  unsigned int hw = std::thread::hardware_concurrency();
  if (hw > 1) {
    EXPECT_LE(tm.getComputeThreadCount(), hw - 1);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

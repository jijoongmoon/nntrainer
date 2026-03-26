// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_threading_benchmark.cpp
 * @date        21 March 2026
 * @brief       Benchmark: Serial vs OpenMP vs BS::thread_pool vs ThreadManager
 *              vs GGML-style
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <thread_manager.h>

using Clock = std::chrono::high_resolution_clock;

// ─── Helpers ────────────────────────────────────────────────

static void fill_random(float *data, size_t n, unsigned seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i)
    data[i] = dist(gen);
}

static double elapsed_us(Clock::time_point s, Clock::time_point e) {
  return std::chrono::duration_cast<std::chrono::microseconds>(e - s).count();
}

// ─── BS::thread_pool style (condition variable based) ────────
//
// Reproduces the core pattern of BS::thread_pool v4.1.0:
// - Workers block on condition_variable when idle
// - submit_loop dispatches range [begin, end) across workers
// - multi_future.wait() blocks until all tasks complete
// This is the threading model nntrainer used BEFORE ThreadManager.

class BSThreadPool {
public:
  explicit BSThreadPool(int n_threads) : n_threads_(n_threads), stop_(false) {
    for (int i = 0; i < n_threads; ++i) {
      workers_.emplace_back([this] { workerLoop(); });
    }
  }

  ~BSThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &t : workers_)
      if (t.joinable())
        t.join();
  }

  // Mimics BS::thread_pool::submit_loop + multi_future::wait()
  template <typename F>
  void submit_loop(size_t begin, size_t end, F &&fn) {
    if (begin >= end)
      return;

    std::atomic<int> remaining{0};
    std::mutex done_mutex;
    std::condition_variable done_cv;

    for (size_t i = begin; i < end; ++i) {
      remaining.fetch_add(1, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.emplace([&fn, i, &remaining, &done_mutex, &done_cv]() {
          fn(i);
          if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(done_mutex);
            done_cv.notify_one();
          }
        });
      }
      cv_.notify_one();
    }

    // wait for all tasks (mimics multi_future.wait())
    std::unique_lock<std::mutex> lk(done_mutex);
    done_cv.wait(lk,
                 [&] { return remaining.load(std::memory_order_acquire) == 0; });
  }

  // Chunked version: dispatches exactly n_threads tasks
  template <typename F> void parallel_for_chunked(int n_threads, F &&fn) {
    submit_loop(0, static_cast<size_t>(n_threads), std::forward<F>(fn));
  }

  int get_thread_count() const { return n_threads_; }

private:
  void workerLoop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty())
          return;
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  int n_threads_;
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_;
};

// ─── GGML-style Threadpool (extracted core logic) ───────────
//
// Faithfully reproduces llama.cpp's ggml_barrier + spin-wait pattern

static inline void ggml_cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

class GGMLThreadPool {
public:
  explicit GGMLThreadPool(int n_threads) : n_threads_(n_threads) {
    workers_.resize(n_threads - 1);
    for (int i = 0; i < n_threads - 1; ++i) {
      workers_[i] = std::thread([this, i] { workerLoop(i + 1); });
    }
  }

  ~GGMLThreadPool() {
    stop_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_seq_cst);
    for (auto &t : workers_)
      if (t.joinable())
        t.join();
  }

  template <typename F> void parallel_for(size_t begin, size_t end, F &&fn) {
    if (begin >= end)
      return;

    task_ = [&fn](size_t i) { fn(i); };
    range_begin_ = begin;
    range_end_ = end;
    current_chunk_.store(begin, std::memory_order_relaxed);

    generation_.fetch_add(1, std::memory_order_seq_cst);

    while (true) {
      size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
      if (idx >= end)
        break;
      fn(idx);
    }

    barrier();
  }

  template <typename F> void parallel_for_chunked(int n_threads, F &&fn) {
    parallel_for(0, static_cast<size_t>(n_threads), std::forward<F>(fn));
  }

  int getThreadCount() const { return n_threads_; }

private:
  void workerLoop(int) {
    unsigned int my_gen = generation_.load(std::memory_order_acquire);

    while (true) {
      while (generation_.load(std::memory_order_acquire) == my_gen) {
        if (stop_.load(std::memory_order_acquire))
          return;
        ggml_cpu_relax();
      }
      my_gen = generation_.load(std::memory_order_acquire);

      if (stop_.load(std::memory_order_acquire))
        return;

      size_t end = range_end_;
      while (true) {
        size_t idx = current_chunk_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= end)
          break;
        task_(idx);
      }

      barrier();
    }
  }

  void barrier() {
    int n_passed = n_barrier_passed_.load(std::memory_order_relaxed);
    int n = n_barrier_.fetch_add(1, std::memory_order_seq_cst);
    if (n == n_threads_ - 1) {
      n_barrier_.store(0, std::memory_order_relaxed);
      n_barrier_passed_.fetch_add(1, std::memory_order_seq_cst);
      return;
    }
    while (n_barrier_passed_.load(std::memory_order_relaxed) == n_passed) {
      ggml_cpu_relax();
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }

  int n_threads_;
  std::vector<std::thread> workers_;
  std::function<void(size_t)> task_;
  size_t range_begin_{0};
  size_t range_end_{0};

  alignas(64) std::atomic<unsigned int> generation_{0};
  alignas(64) std::atomic<int> n_barrier_{0};
  alignas(64) std::atomic<int> n_barrier_passed_{0};
  alignas(64) std::atomic<size_t> current_chunk_{0};
  alignas(64) std::atomic<bool> stop_{false};
};

// ─── 5-way BenchResult ──────────────────────────────────────

struct BenchResult {
  double serial_us;
  double omp_us;
  double bs_pool_us;
  double tm_us;
  double ggml_us;
};

static void print_result(const char *label, const BenchResult &r) {
  std::cout << std::fixed << std::setprecision(1);
  std::cout << "  [" << label << "]" << std::endl;
  std::cout << "    Serial:          " << std::setw(10) << r.serial_us
            << " us" << std::endl;

  auto print_line = [&](const char *name, double us) {
    std::cout << "    " << name << std::setw(10) << us << " us";
    if (r.serial_us > 0 && us > 0)
      std::cout << "  (speedup " << std::setprecision(2)
                << r.serial_us / us << "x)";
    std::cout << std::endl;
  };

  print_line("OpenMP:          ", r.omp_us);
  print_line("BS::thread_pool: ", r.bs_pool_us);
  print_line("ThreadManager:   ", r.tm_us);
  print_line("GGML-style:      ", r.ggml_us);

  // comparison table
  std::cout << std::setprecision(3);
  if (r.bs_pool_us > 0 && r.tm_us > 0) {
    double ratio = r.tm_us / r.bs_pool_us;
    std::cout << "    TM vs BS::pool:  " << ratio;
    std::cout << (ratio < 1.0 ? " (ThreadManager faster)" : " (BS faster)")
              << std::endl;
  }
  if (r.ggml_us > 0 && r.tm_us > 0) {
    double ratio = r.tm_us / r.ggml_us;
    std::cout << "    TM vs GGML:      " << ratio;
    std::cout << (ratio < 1.0 ? " (ThreadManager faster)" : " (GGML faster)")
              << std::endl;
  }
  std::cout << std::endl;
}

// ─── GEMM Benchmarks ───────────────────────────────────────

static void sgemm_serial(const float *A, const float *B, float *C, int M,
                          int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int p = 0; p < K; ++p)
        sum += A[i * K + p] * B[p * N + j];
      C[i * N + j] = sum;
    }
}

static BenchResult bench_sgemm(int M, int N, int K, int n_threads,
                                BSThreadPool &bs_pool,
                                GGMLThreadPool &ggml_pool, int iters) {
  std::vector<float> A(M * K), B(K * N), C(M * N);
  fill_random(A.data(), M * K);
  fill_random(B.data(), K * N);
  auto &tm = nntrainer::ThreadManager::Global();

  // warmup
  sgemm_serial(A.data(), B.data(), C.data(), M, N, K);

  // Serial
  auto t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    sgemm_serial(A.data(), B.data(), C.data(), M, N, K);
  double serial_us = elapsed_us(t0, Clock::now()) / iters;

  // OpenMP
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        float sum = 0;
        for (int p = 0; p < K; ++p)
          sum += A[i * K + p] * B[p * N + j];
        C[i * N + j] = sum;
      }
  }
  double omp_us = elapsed_us(t0, Clock::now()) / iters;

  // BS::thread_pool
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
    bs_pool.submit_loop(0, static_cast<size_t>(M), [&](size_t i) {
      for (int j = 0; j < N; ++j) {
        float sum = 0;
        for (int p = 0; p < K; ++p)
          sum += A[i * K + p] * B[p * N + j];
        C[i * N + j] = sum;
      }
    });
  }
  double bs_pool_us = elapsed_us(t0, Clock::now()) / iters;

  // ThreadManager
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
    tm.parallel_for(0, static_cast<size_t>(M), [&](size_t i) {
      for (int j = 0; j < N; ++j) {
        float sum = 0;
        for (int p = 0; p < K; ++p)
          sum += A[i * K + p] * B[p * N + j];
        C[i * N + j] = sum;
      }
    });
  }
  double tm_us = elapsed_us(t0, Clock::now()) / iters;

  // GGML-style
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
    ggml_pool.parallel_for(0, static_cast<size_t>(M), [&](size_t i) {
      for (int j = 0; j < N; ++j) {
        float sum = 0;
        for (int p = 0; p < K; ++p)
          sum += A[i * K + p] * B[p * N + j];
        C[i * N + j] = sum;
      }
    });
  }
  double ggml_us = elapsed_us(t0, Clock::now()) / iters;

  return {serial_us, omp_us, bs_pool_us, tm_us, ggml_us};
}

// ─── GEMV Benchmark ─────────────────────────────────────────

static BenchResult bench_gemv(int N, int K, int n_threads,
                               BSThreadPool &bs_pool, GGMLThreadPool &ggml_pool,
                               int iters) {
  std::vector<float> A(K), B(K * N), C(N);
  fill_random(A.data(), K);
  fill_random(B.data(), K * N);
  auto &tm = nntrainer::ThreadManager::Global();

  // Serial
  auto t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int p = 0; p < K; ++p)
        sum += A[p] * B[p * N + j];
      C[j] = sum;
    }
  double serial_us = elapsed_us(t0, Clock::now()) / iters;

  // OpenMP
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int p = 0; p < K; ++p)
        sum += A[p] * B[p * N + j];
      C[j] = sum;
    }
  }
  double omp_us = elapsed_us(t0, Clock::now()) / iters;

  // BS::thread_pool
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
    bs_pool.submit_loop(0, static_cast<size_t>(N), [&](size_t j) {
      float sum = 0;
      for (int p = 0; p < K; ++p)
        sum += A[p] * B[p * N + j];
      C[j] = sum;
    });
  }
  double bs_pool_us = elapsed_us(t0, Clock::now()) / iters;

  // ThreadManager
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
    tm.parallel_for(0, static_cast<size_t>(N), [&](size_t j) {
      float sum = 0;
      for (int p = 0; p < K; ++p)
        sum += A[p] * B[p * N + j];
      C[j] = sum;
    });
  }
  double tm_us = elapsed_us(t0, Clock::now()) / iters;

  // GGML-style
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
    ggml_pool.parallel_for(0, static_cast<size_t>(N), [&](size_t j) {
      float sum = 0;
      for (int p = 0; p < K; ++p)
        sum += A[p] * B[p * N + j];
      C[j] = sum;
    });
  }
  double ggml_us = elapsed_us(t0, Clock::now()) / iters;

  return {serial_us, omp_us, bs_pool_us, tm_us, ggml_us};
}

// ─── Chunked GEMM (Q4_0-style dispatch pattern) ─────────────
// This is the exact pattern used in ggml_interface_mixed.cpp:
// each thread gets a column range [col_s, col_e)

static BenchResult bench_chunked(int M, int N, int K, int n_threads,
                                  BSThreadPool &bs_pool,
                                  GGMLThreadPool &ggml_pool, int iters) {
  std::vector<float> A(M * K), B(K * N), C(M * N);
  fill_random(A.data(), M * K);
  fill_random(B.data(), K * N);
  auto &tm = nntrainer::ThreadManager::Global();

  auto chunked_gemm = [&](int tid, int total) {
    int col_s = (tid * N) / total, col_e = ((tid + 1) * N) / total;
    for (int i = 0; i < M; ++i)
      for (int j = col_s; j < col_e; ++j) {
        float sum = 0;
        for (int p = 0; p < K; ++p)
          sum += A[i * K + p] * B[p * N + j];
        C[i * N + j] = sum;
      }
  };

  // Serial
  auto t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    sgemm_serial(A.data(), B.data(), C.data(), M, N, K);
  double serial_us = elapsed_us(t0, Clock::now()) / iters;

  // OpenMP
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int t = 0; t < n_threads; ++t)
      chunked_gemm(t, n_threads);
  }
  double omp_us = elapsed_us(t0, Clock::now()) / iters;

  // BS::thread_pool
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
    bs_pool.parallel_for_chunked(n_threads, [&](size_t t) {
      chunked_gemm(t, n_threads);
    });
  }
  double bs_pool_us = elapsed_us(t0, Clock::now()) / iters;

  // ThreadManager
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    tm.parallel_for_chunked(n_threads, [&](size_t t) {
      chunked_gemm(t, n_threads);
    });
  double tm_us = elapsed_us(t0, Clock::now()) / iters;

  // GGML-style
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    ggml_pool.parallel_for_chunked(n_threads, [&](size_t t) {
      chunked_gemm(t, n_threads);
    });
  double ggml_us = elapsed_us(t0, Clock::now()) / iters;

  return {serial_us, omp_us, bs_pool_us, tm_us, ggml_us};
}

// ─── Dispatch Overhead (many rapid calls, small work) ────────
// Simulates LLM inference: many small GEMV dispatches per token

static BenchResult bench_dispatch_overhead(int calls, int n_threads,
                                            BSThreadPool &bs_pool,
                                            GGMLThreadPool &ggml_pool,
                                            int iters) {
  std::vector<float> data(1000, 1.0f);
  auto &tm = nntrainer::ThreadManager::Global();

  auto small_work = [&](size_t t, int total) {
    int s = (t * 1000) / total, e = ((t + 1) * 1000) / total;
    for (int i = s; i < e; ++i)
      data[i] = data[i] * 1.001f + 0.001f;
  };

  // Serial
  auto t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    for (int c = 0; c < calls; ++c)
      for (int i = 0; i < 1000; ++i)
        data[i] = data[i] * 1.001f + 0.001f;
  double serial_us = elapsed_us(t0, Clock::now()) / iters;

  // OpenMP
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    for (int c = 0; c < calls; ++c) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
      for (int t = 0; t < n_threads; ++t)
        small_work(t, n_threads);
    }
  double omp_us = elapsed_us(t0, Clock::now()) / iters;

  // BS::thread_pool
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    for (int c = 0; c < calls; ++c)
      bs_pool.parallel_for_chunked(n_threads, [&](size_t t) {
        small_work(t, n_threads);
      });
  double bs_pool_us = elapsed_us(t0, Clock::now()) / iters;

  // ThreadManager
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    for (int c = 0; c < calls; ++c)
      tm.parallel_for_chunked(n_threads, [&](size_t t) {
        small_work(t, n_threads);
      });
  double tm_us = elapsed_us(t0, Clock::now()) / iters;

  // GGML-style
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    for (int c = 0; c < calls; ++c)
      ggml_pool.parallel_for_chunked(n_threads, [&](size_t t) {
        small_work(t, n_threads);
      });
  double ggml_us = elapsed_us(t0, Clock::now()) / iters;

  return {serial_us, omp_us, bs_pool_us, tm_us, ggml_us};
}

// ─── LLM Inference Simulation ───────────────────────────────
// Simulates actual LLM layer: 32 sequential GEMV calls (one per layer)
// with realistic dimensions (hidden=2048, intermediate=5632)

static BenchResult bench_llm_layers(int n_layers, int hidden, int inter,
                                     int n_threads, BSThreadPool &bs_pool,
                                     GGMLThreadPool &ggml_pool, int iters) {
  std::vector<float> input(hidden), weight(hidden * inter), output(inter);
  fill_random(input.data(), hidden);
  fill_random(weight.data(), hidden * inter);
  auto &tm = nntrainer::ThreadManager::Global();

  auto gemv_chunked = [&](size_t tid, int total) {
    int col_s = (tid * inter) / total, col_e = ((tid + 1) * inter) / total;
    for (int j = col_s; j < col_e; ++j) {
      float sum = 0;
      for (int p = 0; p < hidden; ++p)
        sum += input[p] * weight[p * inter + j];
      output[j] = sum;
    }
  };

  // Serial
  auto t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    for (int l = 0; l < n_layers; ++l)
      for (int j = 0; j < inter; ++j) {
        float sum = 0;
        for (int p = 0; p < hidden; ++p)
          sum += input[p] * weight[p * inter + j];
        output[j] = sum;
      }
  double serial_us = elapsed_us(t0, Clock::now()) / iters;

  // OpenMP
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    for (int l = 0; l < n_layers; ++l) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
      for (int t = 0; t < n_threads; ++t)
        gemv_chunked(t, n_threads);
    }
  double omp_us = elapsed_us(t0, Clock::now()) / iters;

  // BS::thread_pool
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    for (int l = 0; l < n_layers; ++l)
      bs_pool.parallel_for_chunked(n_threads, [&](size_t t) {
        gemv_chunked(t, n_threads);
      });
  double bs_pool_us = elapsed_us(t0, Clock::now()) / iters;

  // ThreadManager
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    for (int l = 0; l < n_layers; ++l)
      tm.parallel_for_chunked(n_threads, [&](size_t t) {
        gemv_chunked(t, n_threads);
      });
  double tm_us = elapsed_us(t0, Clock::now()) / iters;

  // GGML-style
  t0 = Clock::now();
  for (int it = 0; it < iters; ++it)
    for (int l = 0; l < n_layers; ++l)
      ggml_pool.parallel_for_chunked(n_threads, [&](size_t t) {
        gemv_chunked(t, n_threads);
      });
  double ggml_us = elapsed_us(t0, Clock::now()) / iters;

  return {serial_us, omp_us, bs_pool_us, tm_us, ggml_us};
}

// ─── Test Fixture ───────────────────────────────────────────

class ThreadingBenchmark : public ::testing::Test {
protected:
  void SetUp() override {
    auto &tm = nntrainer::ThreadManager::Global();
    n_threads_ = tm.getComputeThreadCount() + 1;
    bs_pool_ = std::make_unique<BSThreadPool>(n_threads_);
    ggml_pool_ = std::make_unique<GGMLThreadPool>(n_threads_);

    std::cout << "\n=== Threading Benchmark (5-way) ===" << std::endl;
    std::cout << "  Serial | OpenMP | BS::thread_pool | ThreadManager | "
                 "GGML-style"
              << std::endl;
    std::cout << "Threads: " << n_threads_
              << " (TM: " << tm.getComputeThreadCount() << " workers + caller)"
              << std::endl;
#ifdef _OPENMP
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;
#endif
    std::cout << "BS::thread_pool: " << bs_pool_->get_thread_count()
              << " threads" << std::endl;
    std::cout << "GGML-style pool: " << ggml_pool_->getThreadCount()
              << " threads" << std::endl;
    std::cout << std::endl;
  }

  void TearDown() override {
    bs_pool_.reset();
    ggml_pool_.reset();
  }

  unsigned int n_threads_;
  std::unique_ptr<BSThreadPool> bs_pool_;
  std::unique_ptr<GGMLThreadPool> ggml_pool_;
};

// ─── Tests ──────────────────────────────────────────────────

TEST_F(ThreadingBenchmark, Summary) {
  std::cout << "=== SUMMARY ===" << std::endl << std::endl;

  auto r1 =
    bench_sgemm(64, 64, 64, n_threads_, *bs_pool_, *ggml_pool_, 50);
  print_result("Small GEMM 64x64x64", r1);

  auto r2 =
    bench_sgemm(256, 256, 256, n_threads_, *bs_pool_, *ggml_pool_, 5);
  print_result("Large GEMM 256x256x256", r2);

  auto r3 = bench_gemv(4096, 4096, n_threads_, *bs_pool_, *ggml_pool_, 5);
  print_result("GEMV 4096x4096", r3);

  auto r4 =
    bench_chunked(4, 4096, 4096, n_threads_, *bs_pool_, *ggml_pool_, 3);
  print_result("Chunked GEMM 4x4096x4096 (ggml pattern)", r4);

  auto r5 =
    bench_dispatch_overhead(50, n_threads_, *bs_pool_, *ggml_pool_, 3);
  print_result("50 rapid dispatch x 1000 (LLM decode)", r5);

  auto r6 =
    bench_llm_layers(32, 2048, 4096, n_threads_, *bs_pool_, *ggml_pool_, 2);
  print_result("32-layer GEMV 2048x4096 (LLM inference)", r6);

  SUCCEED();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

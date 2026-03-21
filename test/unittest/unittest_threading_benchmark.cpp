// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_threading_benchmark.cpp
 * @date        21 March 2026
 * @brief       Benchmark comparing OpenMP vs ThreadManager for matrix ops
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
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

static double elapsed_us(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
    .count();
}

struct BenchResult {
  double omp_us;
  double tm_us;
  double serial_us;
};

static void print_result(const char *label, const BenchResult &r,
                          int warmup = 0) {
  std::cout << std::fixed << std::setprecision(1);
  std::cout << "  [" << label << "]" << std::endl;
  std::cout << "    Serial:        " << std::setw(10) << r.serial_us
            << " us" << std::endl;
  std::cout << "    OpenMP:        " << std::setw(10) << r.omp_us
            << " us";
  if (r.serial_us > 0)
    std::cout << "  (speedup " << std::setprecision(2) << r.serial_us / r.omp_us
              << "x)";
  std::cout << std::endl;
  std::cout << "    ThreadManager: " << std::setw(10) << r.tm_us
            << " us";
  if (r.serial_us > 0)
    std::cout << "  (speedup " << std::setprecision(2)
              << r.serial_us / r.tm_us << "x)";
  std::cout << std::endl;

  if (r.omp_us > 0) {
    double ratio = r.tm_us / r.omp_us;
    std::cout << "    TM/OMP ratio:  " << std::setprecision(3) << ratio;
    if (ratio < 1.0)
      std::cout << " (ThreadManager faster)";
    else if (ratio > 1.0)
      std::cout << " (OpenMP faster)";
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// ─── FP32 SGEMM Benchmark ──────────────────────────────────
//
// Naive row-parallel GEMM: C[M,N] = A[M,K] * B[K,N]
// Parallelization: each thread computes a subset of output rows

static void sgemm_serial(const float *A, const float *B, float *C,
                          int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      const float *a_row = A + i * K;
      for (int p = 0; p < K; ++p) {
        sum += a_row[p] * B[p * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

static void sgemm_omp(const float *A, const float *B, float *C,
                       int M, int N, int K, int n_threads) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      const float *a_row = A + i * K;
      for (int p = 0; p < K; ++p) {
        sum += a_row[p] * B[p * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

static void sgemm_tm(const float *A, const float *B, float *C,
                      int M, int N, int K) {
  auto &tm = nntrainer::ThreadManager::Global();
  tm.parallel_for(0, static_cast<size_t>(M), [&](size_t i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      const float *a_row = A + i * K;
      for (int p = 0; p < K; ++p) {
        sum += a_row[p] * B[p * N + j];
      }
      C[i * N + j] = sum;
    }
  });
}

static BenchResult bench_sgemm(int M, int N, int K, int n_threads,
                                int iterations) {
  std::vector<float> A(M * K), B(K * N), C(M * N);
  fill_random(A.data(), M * K, 42);
  fill_random(B.data(), K * N, 123);

  // Warmup
  sgemm_serial(A.data(), B.data(), C.data(), M, N, K);
  sgemm_omp(A.data(), B.data(), C.data(), M, N, K, n_threads);
  sgemm_tm(A.data(), B.data(), C.data(), M, N, K);

  // Serial
  auto t0 = Clock::now();
  for (int it = 0; it < iterations; ++it)
    sgemm_serial(A.data(), B.data(), C.data(), M, N, K);
  auto t1 = Clock::now();
  double serial_us = elapsed_us(t0, t1) / iterations;

  // OpenMP
  t0 = Clock::now();
  for (int it = 0; it < iterations; ++it)
    sgemm_omp(A.data(), B.data(), C.data(), M, N, K, n_threads);
  t1 = Clock::now();
  double omp_us = elapsed_us(t0, t1) / iterations;

  // ThreadManager
  t0 = Clock::now();
  for (int it = 0; it < iterations; ++it)
    sgemm_tm(A.data(), B.data(), C.data(), M, N, K);
  t1 = Clock::now();
  double tm_us = elapsed_us(t0, t1) / iterations;

  return {omp_us, tm_us, serial_us};
}

// ─── Chunked GEMM Benchmark (simulating Q4_0-style parallel) ──
//
// This simulates the Q4_0 GEMM pattern where threads are assigned
// fixed chunks of output columns (like the GGML interface pattern):
//   thread i handles columns [i*N/n_threads, (i+1)*N/n_threads)

static void chunked_gemm_omp(const float *A, const float *B, float *C,
                              int M, int N, int K, int n_threads) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (int t = 0; t < n_threads; ++t) {
    int col_start = (t * N) / n_threads;
    int col_end = ((t + 1) * N) / n_threads;
    for (int i = 0; i < M; ++i) {
      const float *a_row = A + i * K;
      for (int j = col_start; j < col_end; ++j) {
        float sum = 0.0f;
        for (int p = 0; p < K; ++p) {
          sum += a_row[p] * B[p * N + j];
        }
        C[i * N + j] = sum;
      }
    }
  }
}

static void chunked_gemm_tm(const float *A, const float *B, float *C,
                             int M, int N, int K) {
  auto &tm = nntrainer::ThreadManager::Global();
  unsigned int n_threads = tm.getComputeThreadCount() + 1;
  tm.parallel_for_chunked(n_threads, [=](size_t t) {
    int col_start = (t * N) / n_threads;
    int col_end = ((t + 1) * N) / n_threads;
    for (int i = 0; i < M; ++i) {
      const float *a_row = A + i * K;
      for (int j = col_start; j < col_end; ++j) {
        float sum = 0.0f;
        for (int p = 0; p < K; ++p) {
          sum += a_row[p] * B[p * N + j];
        }
        C[i * N + j] = sum;
      }
    }
  });
}

static BenchResult bench_chunked_gemm(int M, int N, int K, int n_threads,
                                       int iterations) {
  std::vector<float> A(M * K), B(K * N), C(M * N);
  fill_random(A.data(), M * K, 42);
  fill_random(B.data(), K * N, 123);

  // Warmup
  sgemm_serial(A.data(), B.data(), C.data(), M, N, K);
  chunked_gemm_omp(A.data(), B.data(), C.data(), M, N, K, n_threads);
  chunked_gemm_tm(A.data(), B.data(), C.data(), M, N, K);

  // Serial
  auto t0 = Clock::now();
  for (int it = 0; it < iterations; ++it)
    sgemm_serial(A.data(), B.data(), C.data(), M, N, K);
  auto t1 = Clock::now();
  double serial_us = elapsed_us(t0, t1) / iterations;

  // OpenMP (chunked columns)
  t0 = Clock::now();
  for (int it = 0; it < iterations; ++it)
    chunked_gemm_omp(A.data(), B.data(), C.data(), M, N, K, n_threads);
  t1 = Clock::now();
  double omp_us = elapsed_us(t0, t1) / iterations;

  // ThreadManager (chunked columns)
  t0 = Clock::now();
  for (int it = 0; it < iterations; ++it)
    chunked_gemm_tm(A.data(), B.data(), C.data(), M, N, K);
  t1 = Clock::now();
  double tm_us = elapsed_us(t0, t1) / iterations;

  return {omp_us, tm_us, serial_us};
}

// ─── GEMV Benchmark (M=1 case, important for LLM inference) ─

static void gemv_serial(const float *A, const float *B, float *C,
                         int N, int K) {
  for (int j = 0; j < N; ++j) {
    float sum = 0.0f;
    for (int p = 0; p < K; ++p) {
      sum += A[p] * B[p * N + j];
    }
    C[j] = sum;
  }
}

static void gemv_omp(const float *A, const float *B, float *C,
                      int N, int K, int n_threads) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (int j = 0; j < N; ++j) {
    float sum = 0.0f;
    for (int p = 0; p < K; ++p) {
      sum += A[p] * B[p * N + j];
    }
    C[j] = sum;
  }
}

static void gemv_tm(const float *A, const float *B, float *C,
                     int N, int K) {
  auto &tm = nntrainer::ThreadManager::Global();
  tm.parallel_for(0, static_cast<size_t>(N), [&](size_t j) {
    float sum = 0.0f;
    for (int p = 0; p < K; ++p) {
      sum += A[p] * B[p * N + j];
    }
    C[j] = sum;
  });
}

static BenchResult bench_gemv(int N, int K, int n_threads, int iterations) {
  std::vector<float> A(K), B(K * N), C(N);
  fill_random(A.data(), K, 42);
  fill_random(B.data(), K * N, 123);

  // Warmup
  gemv_serial(A.data(), B.data(), C.data(), N, K);
  gemv_omp(A.data(), B.data(), C.data(), N, K, n_threads);
  gemv_tm(A.data(), B.data(), C.data(), N, K);

  // Serial
  auto t0 = Clock::now();
  for (int it = 0; it < iterations; ++it)
    gemv_serial(A.data(), B.data(), C.data(), N, K);
  auto t1 = Clock::now();
  double serial_us = elapsed_us(t0, t1) / iterations;

  // OpenMP
  t0 = Clock::now();
  for (int it = 0; it < iterations; ++it)
    gemv_omp(A.data(), B.data(), C.data(), N, K, n_threads);
  t1 = Clock::now();
  double omp_us = elapsed_us(t0, t1) / iterations;

  // ThreadManager
  t0 = Clock::now();
  for (int it = 0; it < iterations; ++it)
    gemv_tm(A.data(), B.data(), C.data(), N, K);
  t1 = Clock::now();
  double tm_us = elapsed_us(t0, t1) / iterations;

  return {omp_us, tm_us, serial_us};
}

// ─── parallel_for_chunked Overhead Benchmark ────────────────
//
// Measures pure threading overhead using chunked pattern
// (the correct usage for compute workloads)

static BenchResult bench_overhead(int n_tasks, int iterations) {
  std::vector<float> data(n_tasks, 1.0f);

  auto &tm = nntrainer::ThreadManager::Global();
  unsigned int n_threads = tm.getComputeThreadCount() + 1;

  // Serial
  auto t0 = Clock::now();
  for (int it = 0; it < iterations; ++it) {
    for (int i = 0; i < n_tasks; ++i)
      data[i] = std::sin(data[i]) + 1.0f;
  }
  auto t1 = Clock::now();
  double serial_us = elapsed_us(t0, t1) / iterations;

  // OpenMP
  t0 = Clock::now();
  for (int it = 0; it < iterations; ++it) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int i = 0; i < n_tasks; ++i)
      data[i] = std::sin(data[i]) + 1.0f;
  }
  t1 = Clock::now();
  double omp_us = elapsed_us(t0, t1) / iterations;

  // ThreadManager (chunked - correct pattern)
  t0 = Clock::now();
  for (int it = 0; it < iterations; ++it) {
    tm.parallel_for_chunked(n_threads, [&](size_t t) {
      int start = (t * n_tasks) / n_threads;
      int end = ((t + 1) * n_tasks) / n_threads;
      for (int i = start; i < end; ++i)
        data[i] = std::sin(data[i]) + 1.0f;
    });
  }
  t1 = Clock::now();
  double tm_us = elapsed_us(t0, t1) / iterations;

  return {omp_us, tm_us, serial_us};
}

// ─── Repeated parallel_for Call Overhead ─────────────────────
//
// Measures overhead of many rapid parallel_for_chunked calls
// (like GEMM loops where each call distributes N chunks to threads)

static BenchResult bench_rapid_calls(int call_count, int work_per_call,
                                      int iterations) {
  std::vector<float> data(work_per_call, 1.0f);

  auto &tm = nntrainer::ThreadManager::Global();
  unsigned int n_threads = tm.getComputeThreadCount() + 1;

  // Serial
  auto t0 = Clock::now();
  for (int it = 0; it < iterations; ++it) {
    for (int c = 0; c < call_count; ++c) {
      for (int i = 0; i < work_per_call; ++i)
        data[i] = data[i] * 1.001f + 0.001f;
    }
  }
  auto t1 = Clock::now();
  double serial_us = elapsed_us(t0, t1) / iterations;

  // OpenMP (chunked - each call distributes to n_threads)
  t0 = Clock::now();
  for (int it = 0; it < iterations; ++it) {
    for (int c = 0; c < call_count; ++c) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
      for (int t = 0; t < (int)n_threads; ++t) {
        int start = (t * work_per_call) / n_threads;
        int end = ((t + 1) * work_per_call) / n_threads;
        for (int i = start; i < end; ++i)
          data[i] = data[i] * 1.001f + 0.001f;
      }
    }
  }
  t1 = Clock::now();
  double omp_us = elapsed_us(t0, t1) / iterations;

  // ThreadManager (chunked - same pattern)
  t0 = Clock::now();
  for (int it = 0; it < iterations; ++it) {
    for (int c = 0; c < call_count; ++c) {
      tm.parallel_for_chunked(n_threads, [&](size_t t) {
        int start = (t * work_per_call) / n_threads;
        int end = ((t + 1) * work_per_call) / n_threads;
        for (int i = start; i < end; ++i)
          data[i] = data[i] * 1.001f + 0.001f;
      });
    }
  }
  t1 = Clock::now();
  double tm_us = elapsed_us(t0, t1) / iterations;

  return {omp_us, tm_us, serial_us};
}

// ─── Tests ──────────────────────────────────────────────────

class ThreadingBenchmark : public ::testing::Test {
protected:
  void SetUp() override {
    auto &tm = nntrainer::ThreadManager::Global();
    n_threads_ = tm.getComputeThreadCount() + 1;
    std::cout << "\n=== Threading Benchmark ===" << std::endl;
    std::cout << "Compute threads: " << n_threads_
              << " (ThreadManager: " << tm.getComputeThreadCount()
              << " workers + 1 caller)" << std::endl;
#ifdef _OPENMP
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;
#else
    std::cout << "OpenMP: NOT available (serial fallback)" << std::endl;
#endif
    std::cout << std::endl;
  }
  unsigned int n_threads_;
};

// ─── FP32 SGEMM (row-parallel) ─────────────────────────────

TEST_F(ThreadingBenchmark, SGEMM_Small_64x64x64) {
  std::cout << "--- FP32 SGEMM Row-Parallel ---" << std::endl;
  auto r = bench_sgemm(64, 64, 64, n_threads_, 100);
  print_result("M=64 N=64 K=64", r);
  SUCCEED();
}

TEST_F(ThreadingBenchmark, SGEMM_Medium_256x256x256) {
  auto r = bench_sgemm(256, 256, 256, n_threads_, 10);
  print_result("M=256 N=256 K=256", r);
  SUCCEED();
}

TEST_F(ThreadingBenchmark, SGEMM_LLM_1x4096x4096) {
  std::cout << "--- FP32 SGEMM (LLM-like sizes) ---" << std::endl;
  auto r = bench_sgemm(1, 4096, 4096, n_threads_, 5);
  print_result("M=1 N=4096 K=4096 (GEMV)", r);
  SUCCEED();
}

TEST_F(ThreadingBenchmark, SGEMM_LLM_4x4096x4096) {
  auto r = bench_sgemm(4, 4096, 4096, n_threads_, 5);
  print_result("M=4 N=4096 K=4096", r);
  SUCCEED();
}

// ─── Chunked GEMM (Q4_0-style column chunking) ─────────────

TEST_F(ThreadingBenchmark, ChunkedGEMM_256x256x256) {
  std::cout << "--- Chunked GEMM (Q4_0-style column partition) ---"
            << std::endl;
  auto r = bench_chunked_gemm(256, 256, 256, n_threads_, 10);
  print_result("M=256 N=256 K=256", r);
  SUCCEED();
}

TEST_F(ThreadingBenchmark, ChunkedGEMM_4x4096x4096) {
  auto r = bench_chunked_gemm(4, 4096, 4096, n_threads_, 5);
  print_result("M=4 N=4096 K=4096", r);
  SUCCEED();
}

// ─── GEMV (M=1, critical for LLM token generation) ─────────

TEST_F(ThreadingBenchmark, GEMV_N2048_K2048) {
  std::cout << "--- FP32 GEMV (M=1) ---" << std::endl;
  auto r = bench_gemv(2048, 2048, n_threads_, 20);
  print_result("N=2048 K=2048", r);
  SUCCEED();
}

TEST_F(ThreadingBenchmark, GEMV_N4096_K4096) {
  auto r = bench_gemv(4096, 4096, n_threads_, 10);
  print_result("N=4096 K=4096", r);
  SUCCEED();
}

TEST_F(ThreadingBenchmark, GEMV_N11008_K4096) {
  auto r = bench_gemv(11008, 4096, n_threads_, 5);
  print_result("N=11008 K=4096 (Llama-7B FFN)", r);
  SUCCEED();
}

// ─── Threading Overhead ─────────────────────────────────────

TEST_F(ThreadingBenchmark, Overhead_100_tasks) {
  std::cout << "--- parallel_for Overhead (light work) ---" << std::endl;
  auto r = bench_overhead(100, 1000);
  print_result("100 tasks", r);
  SUCCEED();
}

TEST_F(ThreadingBenchmark, Overhead_10000_tasks) {
  auto r = bench_overhead(10000, 100);
  print_result("10000 tasks", r);
  SUCCEED();
}

// ─── Rapid Repeated Calls ───────────────────────────────────

TEST_F(ThreadingBenchmark, RapidCalls_50x1000) {
  std::cout << "--- Rapid Repeated parallel_for_chunked Calls ---" << std::endl;
  auto r = bench_rapid_calls(50, 1000, 3);
  print_result("50 calls x 1000 elements", r);
  SUCCEED();
}

TEST_F(ThreadingBenchmark, RapidCalls_20x10000) {
  auto r = bench_rapid_calls(20, 10000, 3);
  print_result("20 calls x 10000 elements", r);
  SUCCEED();
}

// ─── Summary ────────────────────────────────────────────────

TEST_F(ThreadingBenchmark, Summary) {
  std::cout << "=== SUMMARY: Key Comparisons ===" << std::endl;
  std::cout << std::endl;

  // Small GEMM (overhead-dominated)
  auto r1 = bench_sgemm(64, 64, 64, n_threads_, 50);
  print_result("Small GEMM 64x64x64", r1);

  // Large GEMM (compute-dominated)
  auto r2 = bench_sgemm(256, 256, 256, n_threads_, 5);
  print_result("Large GEMM 256x256x256", r2);

  // GEMV (LLM inference pattern)
  auto r3 = bench_gemv(4096, 4096, n_threads_, 10);
  print_result("GEMV 4096x4096", r3);

  // Chunked (Q4_0 pattern)
  auto r4 = bench_chunked_gemm(4, 4096, 4096, n_threads_, 3);
  print_result("Chunked GEMM 4x4096x4096", r4);

  // Rapid calls (context switch overhead)
  auto r5 = bench_rapid_calls(50, 1000, 3);
  print_result("50 rapid calls x 1000", r5);

  SUCCEED();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

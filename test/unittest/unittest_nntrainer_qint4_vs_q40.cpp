// SPDX-License-Identifier: Apache-2.0
/**
 * @file        unittest_nntrainer_qint4_vs_q40.cpp
 * @date        20 April 2026
 * @brief       Unit test for Int4QTensor vs Q4_0 using Tensor-level dot operation
 * @see         https://github.com/nntrainer/nntrainer
 * @author      HyeongGwon Hong (h0g1) <h0g1.hong@samsung.com>
 * @bug         No known bugs except for NYI items
 */

#include "nntrainer_test_util.h"
#include "int4_utils.h"
#include "q4_0_utils.h"
#include <cpu_backend.h>
#include <int4_tensor.h>
#include <q4_0_tensor.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>
#include <cstring>

#include <chrono>
#include <iostream>
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;

#define QK4_0 32
#define INT4_KERNEL_IDX_SME 8
#define INT4_KERNEL_IDX_NEON 3

template <typename T>
static inline double find_max_diff(T *src, T *src2, int M, int N) {
  float max_diff = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      max_diff = std::max(max_diff, std::abs(src[i * N + j] - src2[i * N + j]));
    }
  }
  return max_diff;
}

template <typename T = float>
static float compute_mse(const uint32_t M, const uint32_t N,
                         std::vector<T> &ref_dst, std::vector<T> &dst,
                         bool print = false) {
  auto mean_squared_error = mse<T, T>(ref_dst.data(), dst.data(), M * N);
  auto cos_sim = cosine_similarity<T, T>(ref_dst.data(), dst.data(), M * N);
  auto max_differ = find_max_diff<T>(ref_dst.data(), dst.data(), M, N);

  auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
  auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);

  if (print) {
    std::cout << "[INFO]            MSE: " << mean_squared_error
              << ", COS_SIM: " << cos_sim << ", MAX_DIFFER: " << max_differ
              << ", SUM: " << sum << ", SUM_GT: " << sum_gt << std::endl;
  }
  return mean_squared_error;
}

/**
 * @brief Test KleidiAI kernel index selection based on compile-time flags
 */
TEST(nntrainer_int4_tensor, kleidiai_kernel_idx) {
  int32_t kernel_idx = nntrainer::Int4QTensor::get_kleidiai_kernel_idx();

#if defined(ENABLE_SME)
  EXPECT_EQ(kernel_idx, INT4_KERNEL_IDX_SME)
    << "SME enabled: kernel_idx should be " << INT4_KERNEL_IDX_SME;
#elif defined(ENABLE_SVE2) || defined(__ARM_NEON) || defined(__ARM_NEON__)
  EXPECT_EQ(kernel_idx, INT4_KERNEL_IDX_NEON)
    << "NEON/SVE2 enabled: kernel_idx should be " << INT4_KERNEL_IDX_NEON;
#else
  EXPECT_EQ(kernel_idx, -1)
    << "No ARM extension: kernel_idx should be -1";
#endif
}

/**
 * @brief Test Int4QTensor memory allocation matches KleidiAI packed size
 */
TEST(nntrainer_int4_tensor, allocation_packed_size) {
  nntrainer::init_backend();

  const uint32_t M = 1, K = 256, N = 128;
  nntrainer::TensorDim dim(1, 1, K, N, nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::QINT4);

  nntrainer::Tensor tensor(dim);
  tensor.allocate();

  int32_t kernel_idx = nntrainer::Int4QTensor::get_kleidiai_kernel_idx();

  if (kernel_idx != -1) {
    size_t packed_size = nntr_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(
      N, K, kernel_idx, true);
    EXPECT_GE(tensor.getMemoryBytes(), packed_size);
  }
}

/**
 * @brief Pack weights for KleidiAI and copy to tensor
 * 
 * This function handles both Android (packed format) and non-Android (unpacked format) cases.
 */
static void pack_and_set_int4_weights(nntrainer::Tensor &weight_tensor,
                                       const float *weights_fp32,
                                       uint32_t K, uint32_t N) {

  int32_t kernel_idx = nntrainer::Int4QTensor::get_kleidiai_kernel_idx();
  
  // Quantize weights to INT4
  size_t quantized_size = N * ((K + 1)/2);
  std::vector<uint8_t> quantized_weights(quantized_size);
  std::vector<float> scales(N);
  nntr_quant_qs4cx_f32(N, K, (void *)weights_fp32, 
                       (void *)quantized_weights.data(), (void *)scales.data(), true);
  
  if (kernel_idx >= 0) {
    // Android: Pack weights for KleidiAI and copy to tensor
    size_t packed_size = nntrainer::nntr_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(
      N, K, kernel_idx, true);
    std::vector<uint8_t> packed_weights(packed_size);
    
    nntrainer::nntr_qsi4cxp_qs4cxs1s0_rhs_pack(N, K, packed_weights.data(),
                                    quantized_weights.data(), scales.data(),
                                    kernel_idx, true);

    auto qw = quantized_weights.data();
    auto sc = scales.data();
    
    auto w = weight_tensor.getData();
    auto p = (float *)(packed_weights.data());
    // Copy packed data directly to tensor
    memcpy(weight_tensor.getData<uint8_t>(), packed_weights.data(), packed_size);

    } else {
    // Non-Android: Copy unpacked data and scales separately
    memcpy(weight_tensor.getData<uint8_t>(), quantized_weights.data(), quantized_size);
    memcpy(weight_tensor.getScale<float>(), scales.data(), N * sizeof(float));
  }
}

/**
 * @brief Test INT4 dot operation using Tensor-level API
 * 
 * This test creates FP32 activation tensor and QINT4 weight tensor,
 * then uses Tensor::dot() to compute the matrix multiplication.
 */
static float test_dot_int4_kleidiai(const uint32_t M, const uint32_t K,
                                    const uint32_t N,
                                    const float *weights_fp32,
                                    const float *activations,
                                    std::vector<float> &ref_dst,
                                    bool print = false) {
  nntrainer::init_backend();

  // Step 1: Create FP32 activation tensor (M x K)
  nntrainer::TensorDim act_dim(1, 1, M, K, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::FP32);
  nntrainer::Tensor act_tensor(act_dim);
  act_tensor.allocate();
  memcpy(act_tensor.getData<float>(), activations, M * K * sizeof(float));

  // Step 2: Create QINT4 weight tensor (K x N) with PER_CHANNEL_AFFINE scheme
  nntrainer::TensorDim weight_dim(1, 1, K, N, nntrainer::Tformat::NCHW,
                                   nntrainer::Tdatatype::QINT4);
  nntrainer::Tensor weight_tensor(weight_dim);
  weight_tensor.allocate();
  
  // Pack and set weights (handles both Android and non-Android)
  pack_and_set_int4_weights(weight_tensor, weights_fp32, K, N);
  
  // Step 3: Create output tensor (M x N)
  nntrainer::TensorDim out_dim(1, 1, M, N, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::FP32);
  nntrainer::Tensor out_tensor(out_dim);
  out_tensor.allocate();

  // Step 4: Run dot operation using Tensor-level API
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD: Tensor::dot() ####
  act_tensor.dot(weight_tensor, out_tensor, false, false, 0.0f);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);

  if (print) {
    std::cout << "[INFO] dot_int4_kleidiai: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step 5: Copy result and compute MSE
  std::vector<float> dst(M * N);
  memcpy(dst.data(), out_tensor.getData<float>(), M * N * sizeof(float));

  std::cout << "ref vs kai" << std::endl;
  std::cout << ref_dst[0] << ref_dst[1] << ref_dst[2] << std::endl;
  std::cout << dst[0] << dst[1] << dst[2] << std::endl;

  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
  return mean_squared_error;
}

/**
 * @brief Test Q4_0 dot operation using Tensor-level API
 * 
 * This test creates FP32 activation tensor and Q4_0 weight tensor,
 * then uses Tensor::dot() to compute the matrix multiplication.
 */
static float test_dot_q4_0(const uint32_t M, const uint32_t K, const uint32_t N,
                           const float *weights, const float *activations,
                           std::vector<float> &ref_dst, bool print = false) {
  nntrainer::init_backend();

  // Step 1: Create FP32 activation tensor (M x K)
  nntrainer::TensorDim act_dim(1, 1, M, K, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::FP32);
  nntrainer::Tensor act_tensor(act_dim);
  act_tensor.allocate();
  memcpy(act_tensor.getData<float>(), activations, M * K * sizeof(float));

  // Step 2: Create Q4_0 weight tensor (K x N)
  // Q4_0 uses block-based quantization with block size 32
  int64_t q4_0_type_size = sizeof(nntrainer::block_q4_0);
  int64_t q4_0_block_size = QK4_0;
  size_t q4_0_data_size =
    (static_cast<size_t>(K) * N / q4_0_block_size) * q4_0_type_size;

  nntrainer::TensorDim weight_dim(1, 1, K, N, nntrainer::Tformat::NCHW,
                                   nntrainer::Tdatatype::Q4_0);
  nntrainer::Tensor weight_tensor(weight_dim);
  weight_tensor.allocate();

  // Quantize weights to Q4_0 format
  std::vector<char> q4_0_offline_qWeight(q4_0_data_size);
  nntrainer::quantize_q4_0(weights, q4_0_offline_qWeight.data(), N, K, nullptr);

  // Repack to q4_0x4/q4_0x8 layout
  std::vector<char> q4_0_repacked_qWeight(q4_0_data_size);
  nntrainer::repack_q4_0(q4_0_repacked_qWeight.data(), q4_0_offline_qWeight.data(),
                         q4_0_data_size, N, K);

  // Copy quantized data to weight tensor
  memcpy(weight_tensor.getData<char>(), q4_0_repacked_qWeight.data(), q4_0_data_size);

  // Step 3: Create output tensor (M x N)
  nntrainer::TensorDim out_dim(1, 1, M, N, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::FP32);
  nntrainer::Tensor out_tensor(out_dim);
  out_tensor.allocate();

  // Step 4: Run dot operation using Tensor-level API
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD: Tensor::dot() ####
  act_tensor.dot(weight_tensor, out_tensor, false, false, 0.0f);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);

  if (print) {
    std::cout << "[INFO] dot_q4_0: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step 5: Copy result and compute MSE
  std::vector<float> dst(M * N);
  memcpy(dst.data(), out_tensor.getData<float>(), M * N * sizeof(float));

  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
  return mean_squared_error;
}

static void run_int4_kleidiai_test(const uint32_t M, const uint32_t K,
                                   const uint32_t N, float &int4_mse,
                                   float &q4_0_mse, float &speedup,
                                   bool print = false) {
  nntrainer::init_backend();

  if (print) {
    std::cout << "[INFO] INT4 KleidiAI Test (M:" << M << ", K:" << K
              << ", N:" << N << ")" << std::endl;
  }

  // Generate random data
  std::vector<float> activation = generate_random_vector<float>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> ref_dst(M * N);

  // GROUND TRUTH: transB SGEMM
  auto t1 = high_resolution_clock::now();
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt_sgemm = duration_cast<nanoseconds>(t2 - t1);

  if (print) {
    std::cout << "[INFO] sgemm:    " << dt_sgemm.count() << " ns "
              << dt_sgemm.count() / 1'000 << " us "
              << dt_sgemm.count() / 1'000'000 << " ms " << std::endl;
  }
  
  // Test INT4 KleidiAI accuracy using Tensor::dot()
  int4_mse = test_dot_int4_kleidiai(M, K, N, weight.data(), activation.data(),
                                    ref_dst, print);

  // Test Q4_0 for comparison using Tensor::dot()
  q4_0_mse =
    test_dot_q4_0(M, K, N, weight.data(), activation.data(), ref_dst, print);

  // Calculate relative MSE (INT4 should be comparable or better)
  speedup = q4_0_mse / int4_mse;

  if (print) {
    std::cout << "[INFO] INT4/Q4_0 MSE ratio: " << speedup << std::endl;
  }
}

/**
 * @brief Test INT4 KleidiAI accuracy vs SGEMM for various dimensions
 */
TEST(nntrainer_int4_tensor, quant_GEMV_1x512x512) {
  const uint32_t M = 1, K = 512, N = 512;
  float int4_mse, q4_0_mse, speedup;
  constexpr float eps = 1e-5;

  run_int4_kleidiai_test(M, K, N, int4_mse, q4_0_mse, speedup, false);

  ASSERT_LE(int4_mse, eps * M * K * N);
}

TEST(nntrainer_int4_tensor, quant_GEMV_1x1024x1024) {
  const uint32_t M = 1, K = 1024, N = 1024;
  float int4_mse, q4_0_mse, speedup;
  constexpr float eps = 1e-5;

  run_int4_kleidiai_test(M, K, N, int4_mse, q4_0_mse, speedup, false);

  ASSERT_LE(int4_mse, eps * M * K * N);
}

TEST(nntrainer_int4_tensor, quant_GEMM_4x512x512) {
  const uint32_t M = 4, K = 512, N = 512;
  float int4_mse, q4_0_mse, speedup;
  constexpr float eps = 1e-5;

  run_int4_kleidiai_test(M, K, N, int4_mse, q4_0_mse, speedup, false);

  ASSERT_LE(int4_mse, eps * M * K * N);
}

TEST(nntrainer_int4_tensor, quant_GEMM_8x1024x1024) {
  const uint32_t M = 8, K = 1024, N = 1024;
  float int4_mse, q4_0_mse, speedup;
  constexpr float eps = 1e-5;

  run_int4_kleidiai_test(M, K, N, int4_mse, q4_0_mse, speedup, false);

  ASSERT_LE(int4_mse, eps * M * K * N);
}

/**
 * @brief Performance comparison: INT4 KleidiAI vs Q4_0 using Tensor::dot()
 */
static void run_performance_test(const uint32_t M, const uint32_t K,
                                 const uint32_t N, bool print = true) {
  nntrainer::init_backend();
  const int TEST_CNT = 20;
  nanoseconds int4_time = nanoseconds(0);
  nanoseconds q4_0_time = nanoseconds(0);

  std::vector<float> activation = generate_random_vector<float>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);

  // Create FP32 activation tensor
  nntrainer::TensorDim act_dim(1, 1, M, K, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::FP32);
  nntrainer::Tensor act_tensor(act_dim);
  act_tensor.allocate();
  memcpy(act_tensor.getData<float>(), activation.data(), M * K * sizeof(float));

  // Create QINT4 weight tensor
  nntrainer::TensorDim int4_weight_dim(1, 1, K, N, nntrainer::Tformat::NCHW,
                                        nntrainer::Tdatatype::QINT4);
  nntrainer::Tensor int4_weight_tensor(int4_weight_dim);
  int4_weight_tensor.allocate();

  // Pack and set weights (handles both Android and non-Android)
  pack_and_set_int4_weights(int4_weight_tensor, weight.data(), K, N);

  // Create Q4_0 weight tensor
  int64_t q4_0_type_size = sizeof(nntrainer::block_q4_0);
  int64_t q4_0_block_size = QK4_0;
  size_t q4_0_data_size =
    (static_cast<size_t>(K) * N / q4_0_block_size) * q4_0_type_size;

  nntrainer::TensorDim q40_weight_dim(1, 1, K, N, nntrainer::Tformat::NCHW,
                                       nntrainer::Tdatatype::Q4_0);
  nntrainer::Tensor q40_weight_tensor(q40_weight_dim);
  q40_weight_tensor.allocate();

  std::vector<char> q4_0_weight(q4_0_data_size);
  nntrainer::quantize_q4_0(weight.data(), q4_0_weight.data(), N, K, nullptr);
  std::vector<char> q4_0_repacked(q4_0_data_size);
  nntrainer::repack_q4_0(q4_0_repacked.data(), q4_0_weight.data(), q4_0_data_size,
                         N, K);
  memcpy(q40_weight_tensor.getData<char>(), q4_0_repacked.data(), q4_0_data_size);

  // Create output tensors
  nntrainer::TensorDim out_dim(1, 1, M, N, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::FP32);
  nntrainer::Tensor int4_out(out_dim);
  nntrainer::Tensor q4_0_out(out_dim);
  int4_out.allocate();
  q4_0_out.allocate();

  for (int i = -1; i < TEST_CNT; i++) {
    // INT4 KleidiAI using Tensor::dot()
    auto t1 = high_resolution_clock::now();
    act_tensor.dot(int4_weight_tensor, int4_out, false, false, 0.0f);
    auto t2 = high_resolution_clock::now();

    if (i >= 0) {
      int4_time += duration_cast<nanoseconds>(t2 - t1);
    }

    // Q4_0 using Tensor::dot()
    auto t3 = high_resolution_clock::now();
    act_tensor.dot(q40_weight_tensor, q4_0_out, false, false, 0.0f);
    auto t4 = high_resolution_clock::now();

    if (i >= 0) {
      q4_0_time += duration_cast<nanoseconds>(t4 - t3);
    }
  }

  double avg_int4_us = int4_time.count() / TEST_CNT / 1000.0;
  double avg_q4_0_us = q4_0_time.count() / TEST_CNT / 1000.0;
  double speedup = avg_q4_0_us / avg_int4_us;

  if (print) {
    std::cout << "[INFO] Performance Test (M=" << M << ", K=" << K
              << ", N=" << N << ")" << std::endl;
    std::cout << "[INFO] INT4 KleidiAI avg: " << avg_int4_us << " us"
              << std::endl;
    std::cout << "[INFO] Q4_0 avg:         " << avg_q4_0_us << " us"
              << std::endl;
    std::cout << "[INFO] Speedup:          " << speedup << "x" << std::endl;
  }

#if defined(ENABLE_SME)
  EXPECT_GT(speedup, 1.0) << "SME kernel should be faster than Q4_0";
#endif
}

TEST(nntrainer_int4_tensor, performance_1x512x512) {
  run_performance_test(1, 512, 512);
}

TEST(nntrainer_int4_tensor, performance_1x1024x1024) {
  run_performance_test(1, 1024, 1024);
}

TEST(nntrainer_int4_tensor, performance_4x512x512) {
  run_performance_test(4, 512, 512);
}

/**
 * @brief SME-specific kernel test using Tensor::dot()
 */
#ifdef ENABLE_SME
TEST(nntrainer_int4_tensor, sme_kernel_enabled) {
  int32_t kernel_idx = nntrainer::Int4QTensor::get_kleidiai_kernel_idx();
  ASSERT_EQ(kernel_idx, INT4_KERNEL_IDX_SME);

  nntrainer::init_backend();
  const uint32_t M = 1, K = 256, N = 128;

  std::vector<float> activation = generate_random_vector<float>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);

  // Create FP32 activation tensor
  nntrainer::TensorDim act_dim(1, 1, M, K, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::FP32);
  nntrainer::Tensor act_tensor(act_dim);
  act_tensor.allocate();
  memcpy(act_tensor.getData<float>(), activation.data(), M * K * sizeof(float));

  // Create QINT4 weight tensor
  nntrainer::TensorDim weight_dim(1, 1, K, N, nntrainer::Tformat::NCHW,
                                   nntrainer::Tdatatype::QINT4);
  nntrainer::Tensor weight_tensor(weight_dim);
  weight_tensor.allocate();

  // Pack and set weights (handles both Android and non-Android)
  pack_and_set_int4_weights(weight_tensor, weight.data(), K, N);

  // Create output tensor
  nntrainer::TensorDim out_dim(1, 1, M, N, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::FP32);
  nntrainer::Tensor out_tensor(out_dim);
  out_tensor.allocate();

  // Run Tensor::dot()
  EXPECT_NO_THROW(act_tensor.dot(weight_tensor, out_tensor, false, false, 0.0f));
}
#endif

/**
 * @brief NEON-specific kernel test using Tensor::dot()
 */
#if !defined(ENABLE_SME) && (defined(ENABLE_SVE2) || defined(__ARM_NEON))
TEST(nntrainer_int4_tensor, neon_kernel_enabled) {
  int32_t kernel_idx = nntrainer::Int4QTensor::get_kleidiai_kernel_idx();
  ASSERT_EQ(kernel_idx, INT4_KERNEL_IDX_NEON);

  nntrainer::init_backend();
  const uint32_t M = 1, K = 256, N = 128;

  std::vector<float> activation = generate_random_vector<float>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);

  // Create FP32 activation tensor
  nntrainer::TensorDim act_dim(1, 1, M, K, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::FP32);
  nntrainer::Tensor act_tensor(act_dim);
  act_tensor.allocate();
  memcpy(act_tensor.getData<float>(), activation.data(), M * K * sizeof(float));

  // Create QINT4 weight tensor
  nntrainer::TensorDim weight_dim(1, 1, K, N, nntrainer::Tformat::NCHW,
                                   nntrainer::Tdatatype::QINT4);
  nntrainer::Tensor weight_tensor(weight_dim);
  weight_tensor.allocate();

  // Pack and set weights (handles both Android and non-Android)
  pack_and_set_int4_weights(weight_tensor, weight.data(), K, N);

  // Create output tensor
  nntrainer::TensorDim out_dim(1, 1, M, N, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::FP32);
  nntrainer::Tensor out_tensor(out_dim);
  out_tensor.allocate();

  // Run Tensor::dot()
  EXPECT_NO_THROW(act_tensor.dot(weight_tensor, out_tensor, false, false, 0.0f));
}
#endif

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
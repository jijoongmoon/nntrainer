#include <gtest/gtest.h>
#include <layer.h>
#include <cl_context.h>
#include "q4_0_utils.h"
#include "nntrainer_test_util.h"
#include "int4_utils.h"
#include <fp16.h>
#include <blas_kernels.h>

using namespace nntrainer;

#define Q4_0 32

static void run_int4_gemm_adreno_test_(const uint32_t M, const uint32_t K, 
                                const uint32_t N, int scale_group_size, bool debug = false) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  const int INT4_BLOCK_N_SIZE = 32;
  uint32_t alignN = align(N, INT4_BLOCK_N_SIZE);
  uint32_t alignK = align(K, scale_group_size);

  uint32_t input_size = M * alignK;

  std::vector<float> input_orig;
  std::vector<float> weight_fp32;



  if (debug){
    float ones_ratio = 0.1f;
    input_orig = generate_01_vector(M * K, ones_ratio);
    weight_fp32 = generate_01_vector(N * K, ones_ratio);
  }
  else {
    input_orig = generate_random_vector<float, false>(M * K, -1.0, 1.0);
    weight_fp32 = generate_random_vector<float, false>(N * K, -1.0, 1.0);
  }

  std::vector<float> ref_dst(M * N, 0.0f);
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, input_orig.data(), K,
                   weight_fp32.data(), K, 0.F, ref_dst.data(), N);


  if (debug) {
    for (int x = 0; x < M; x++) {
      for (int y = 0; y < K; y++) {
        printf("%4.0f ", input_orig[x * K + y]);
      }
      printf("\n");
    }
    printf("---------------------------\n");
    for (int x = 0; x < K; x++) {
      for (int y = 0; y < N; y++) {
        printf("%4.0f ", weight_fp32[y * K + x]);
      }
      printf("\n");
    }
    printf("---------------------------\n");
  }

  std::vector<float> input(M *alignK, 0.0f);
  for (int x = 0; x < M; x++) {
    for (int y = 0; y < K; y++) {
      input[x * alignK + y] = input_orig[x * K + y];
    }
  }

  unsigned int run_count = 100;

  float mse_q4 = 0.0f;

  std::vector<float> q4_output_fp32(M * N);

  if (K % Q4_0 == 0 && N % 8 == 0) {
    size_t q4_data_size = K * N / Q4_0 * sizeof(block_q4_0);
    std::vector<uint8_t> q4_weight(q4_data_size);
    std::vector<uint8_t> q4_weight_repack(q4_data_size);
    nntrainer::quantize_q4_0(weight_fp32.data(), q4_weight.data(), N, K,
                             nullptr);
    nntrainer::repack_q4_0(q4_weight_repack.data(), q4_weight.data(),
                           q4_data_size, N, K);

    for (unsigned int i = 0; i < 10; ++i) {
      nntrainer::gemm_q4_0(M, N, K, input.data(), K, q4_weight_repack.data(), N,
                         q4_output_fp32.data(), N);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < run_count; ++i) {
      nntrainer::gemm_q4_0(M, N, K, input.data(), K, q4_weight_repack.data(), N,
                         q4_output_fp32.data(), N);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto cpu_dt =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << " - time : CPU = " << cpu_dt / (run_count * 1.0f) << " ms"
            << std::endl;

    mse_q4 = mse<float>(ref_dst.data(), q4_output_fp32.data(), M * N);
  }

  std::cout << "MSE int4: " << mse_q4 << std::endl;

  if (debug) {
    for (int x = 0; x < M; x++) {
      for (int y = 0; y < N; y++) {
        printf("%3.0f ",q4_output_fp32[x * N + y]);
      }
      printf("\n");
    }
    printf("-------------------------------------------------\n");
  }

  uint16_t *input_ptr = (uint16_t *)allocateSVM(input_size * sizeof(uint16_t));
  int8_t *weight_ptr = (int8_t *)allocateSVM(alignK * alignN / 2);
  uint16_t *scale_ptr = (uint16_t *)allocateSVM(ceilDiv(K, scale_group_size) *
                                                alignN * sizeof(uint16_t));
  uint16_t *output_ptr = (uint16_t *)allocateSVM(M * N * sizeof(uint16_t));

  blas_cc->command_queue_inst_.enqueueSVMMap(
    input_ptr, input_size * sizeof(uint16_t), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(weight_ptr, alignK * alignN / 2,
                                             false);
  blas_cc->command_queue_inst_.enqueueSVMMap(
    scale_ptr, ceilDiv(K, scale_group_size) * alignN * sizeof(uint16_t), false);

  std::vector<uint8_t> quantized_weights;
  std::vector<uint16_t> quantized_scales;
  Int4Utils::quantizeAndRepack(weight_fp32.data(), N, K, scale_group_size,
                               quantized_weights, quantized_scales);

  for (unsigned int i = 0; i < input_size; ++i) {
    input_ptr[i] = compute_fp32_to_fp16((input.data())[i]);
  }

  for (unsigned int i = 0; i < ceilDiv(K, scale_group_size) * alignN; ++i) {
    scale_ptr[i] = quantized_scales[i];
  }

  for (unsigned int i = 0; i < alignN * align(K, scale_group_size) / 2; ++i) {
    weight_ptr[i] = quantized_weights[i];
  }
  
  blas_cc->command_queue_inst_.enqueueSVMUnmap(input_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(weight_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(scale_ptr);


  for (unsigned int i = 0; i < 10; ++i) {
    nntrainer::gemm_int4_cl_adreno(input_ptr, weight_ptr, scale_ptr, output_ptr, M,
                                N, K, scale_group_size);
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_int4_cl_adreno(input_ptr, weight_ptr, scale_ptr, output_ptr, M,
                                N, K, scale_group_size);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt =
    std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

  std::vector<float> output_fp32(M*N, 0.0f);
  for (unsigned int i = 0; i < M*N; ++i) {
    output_fp32[i] = compute_fp16_to_fp32(output_ptr[i]);
  }


  std::cout << " - time : GPU = " << gpu_dt / (run_count * 1.0f) << " ms"
            << std::endl <<"\n";

  if (debug) {
    for (int x = 0; x < M; x++) {
      for (int y = 0; y < N; y++) {
        printf("%3.0f ",output_fp32[x * N + y]);
      }
      printf("\n");
    }
  }

  float mse_int4_err =
    mse<float>(ref_dst.data(), output_fp32.data(), M*N);

  std::cout << "MSE int4: " << mse_int4_err << std::endl;

  freeSVM(weight_ptr);
  freeSVM(scale_ptr);
  freeSVM(input_ptr);
  freeSVM(output_ptr);

}

#define DECLARE_int4_gemm_adreno_test_M_K_N(M, K, N, G)                               \
  TEST(nntrainer_opencl_adreno_kernels_int4,                                          \
       int4_gemm_adreno_test_##M##_##K##_##N##_Group##G) {                            \
    run_int4_gemm_adreno_test_(M, K, N, G);                                           \
  }

DECLARE_int4_gemm_adreno_test_M_K_N(32, 32, 32, 32);
DECLARE_int4_gemm_adreno_test_M_K_N(1024, 1024, 1024, 32);
DECLARE_int4_gemm_adreno_test_M_K_N(1028, 1024, 1024, 32);
DECLARE_int4_gemm_adreno_test_M_K_N(1024, 1028, 1024, 32);

DECLARE_int4_gemm_adreno_test_M_K_N(1024, 1024, 1028, 128);
DECLARE_int4_gemm_adreno_test_M_K_N(4096, 1024, 1024, 128);
DECLARE_int4_gemm_adreno_test_M_K_N(1024, 4096, 4096, 128);
DECLARE_int4_gemm_adreno_test_M_K_N(4096, 4096, 4096, 128);



// int main(int argc, char **argv) {
//   run_int4_gemm_test_(32, 32, 32, 32, true);
//   run_int4_gemm_test_(1024, 1024, 1024, 32);
//   run_int4_gemm_test_(1028, 1024, 1024, 32);
//   run_int4_gemm_test_(1024, 1024, 1028, 32);
//   run_int4_gemm_test_(1024, 1028, 1024, 32);
//   run_int4_gemm_test_(4096, 1024, 1024, 32);
//   run_int4_gemm_test_(1024, 4096, 4096, 32);
//   run_int4_gemm_test_(4096, 4096, 4096, 32);
// }

GTEST_API_ int main(int argc, char **argv) {
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
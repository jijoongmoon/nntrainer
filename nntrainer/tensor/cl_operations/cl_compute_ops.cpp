// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cl_compute_ops.cpp
 * @date   25 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  OpenCL ComputeOps subclass — provides accelerated quantized
 *         GEMM/GEMV variants on top of the existing nntrainer
 *         OpenCL kernels in cl_operations/blas_kernels.cpp.
 *
 * Only the accelerator-specific ops (Q4_0 batch / accel,
 * INT4 batch / accel) are overridden, with their supports_*()
 * predicates returning true. All other ops fall through to the
 * base ComputeOps default (which throws), so callers rely on
 * supports_*() to decide whether to use this path or fall back
 * to a CPU ops table — exactly the contract float_tensor.cpp's
 * dispatch sites already follow.
 *
 * This file is what unblocks GPU dispatch end-to-end:
 *   ClContext (Engine-registered) -> ContextData -> ClComputeOps
 *   -> nntrainer::gemm_q4_0_async_cl(...) -> OpenCL kernel queue.
 */

#include <blas_kernels.h>
#include <compute_ops.h>
#include <cpu_backend.h> // gemm_q4_0 (host route for CL-ineligible shapes)

namespace nntrainer {

namespace {
/**
 * @brief Whether one (N, K) Q4_0 weight can enter the CL accel route.
 *
 * Both CL entry points (gemm_q4_0_cl / gemm_q4_0_async_cl) prepack the
 * Q4_0x8 weight on the host with unpack_q4_0x8_transpose16, whose x86
 * implementation is an AVX2 256-wide K unroll asserting (K % 256) == 0 and
 * (N % 8) == 0 (avx2_impl.cpp); the kernel's NDRange (N/4 columns) needs
 * N % 4 == 0, subsumed by N % 8. Ineligible shapes (e.g. the hidden=64
 * tiny test fixtures) take the host gemm_q4_0 below instead.
 */
bool cl_q4_0_shape_eligible(unsigned int N, unsigned int K) {
  return (K % 256) == 0 && (N % 8) == 0;
}
} // namespace

class ClComputeOps : public ComputeOps {
public:
  // ── Accelerator-only Q4_0 / INT4 GEMM/GEMV ────────────────
  // The Q4_0 supports_*() predicates are shape-blind, so the shape gate lives
  // here in the CL wrappers: eligible shapes take the CL kernels unchanged,
  // ineligible ones bounce to the host gemm_q4_0 — the same generic route the
  // cpu engine's gemm_q4_0_fp32 uses (cpu_ops_table.h), on the same host/SVM
  // pointers the caller handed us. That host bounce is deliberate, not hidden:
  // there is no CL fallback kernel for these shapes.
  bool supports_gemm_q4_0_batch_fp32() const override { return true; }
  void gemm_q4_0_batch_fp32(std::vector<void *> matAdata, float *matBdata,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> N,
                            unsigned int K) override {
    bool eligible = true;
    for (unsigned int n : N)
      eligible = eligible && cl_q4_0_shape_eligible(n, K);
    if (!eligible) {
      // Whole batch on host (mirrors float_tensor.cpp's non-accel loop).
      for (size_t i = 0; i < matAdata.size(); ++i)
        nntrainer::gemm_q4_0(M, N[i], K, matBdata, K, matAdata[i], N[i],
                             matCdata[i], N[i]);
      return;
    }
    nntrainer::gemm_q4_0_async_cl(matAdata, matBdata, matCdata, M, N, K);
  }

  bool supports_gemm_q4_0_accel_fp32() const override { return true; }
  void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata, float *matCdata,
                            unsigned int M, unsigned int N,
                            unsigned int K) override {
    if (!cl_q4_0_shape_eligible(N, K)) {
      nntrainer::gemm_q4_0(M, N, K, matBdata, K, matAdata, N, matCdata, N);
      return;
    }
    nntrainer::gemm_q4_0_cl(matAdata, matBdata, matCdata, M, N, K);
  }

  // Generic (non-accel) Q4_0 branch of float_tensor.cpp's dispatch — the
  // M == 1 decode step and any caller that skips the accel predicates land
  // here. Host GEMM, identical to CpuComputeOps::gemm_q4_0_fp32
  // (cpu_ops_table.h); without this override the base class throws NI, which
  // the Q4_0 GPU decode path would hit on its first token.
  void gemm_q4_0_fp32(const unsigned int M, const unsigned int N,
                      const unsigned int K, const float *A,
                      const unsigned int lda, const void *B,
                      const unsigned int ldb, float *C,
                      const unsigned int ldc) override {
    nntrainer::gemm_q4_0(M, N, K, A, lda, B, ldb, C, ldc);
  }

  bool supports_gemv_int4_batch_fp32() const override { return true; }
  void gemv_int4_batch_fp32(std::vector<void *> weights,
                            std::vector<uint16_t *> scales, float *input,
                            std::vector<float *> outputs, unsigned int K,
                            std::vector<unsigned int> Ns,
                            unsigned int group_size) override {
    nntrainer::gemv_int4_async_cl(weights, scales, input, outputs, K, Ns,
                                  group_size);
  }

  bool supports_gemm_int4_batch_fp32() const override { return true; }
  void gemm_int4_batch_fp32(float *input, std::vector<void *> weights,
                            std::vector<uint16_t *> scales,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> Ns, unsigned int K,
                            unsigned int group_size) override {
    nntrainer::gemm_int4_async_cl(input, weights, scales, matCdata, M, Ns, K,
                                  group_size);
  }

  bool supports_gemv_int4_accel_fp32() const override { return true; }
  void gemv_int4_accel_fp32(char *weight, uint16_t *scale, float *input,
                            float *output, unsigned int K, unsigned int N,
                            unsigned int group_size) override {
    nntrainer::gemv_int4_cl(weight, scale, input, output, K, N, group_size);
  }

  bool supports_sgemm_int4_accel_fp32() const override { return true; }
  void sgemm_int4_accel_fp32(float *input, char *weight, uint16_t *scale,
                             float *output, unsigned int M, unsigned int N,
                             unsigned int K, unsigned int group_size) override {
    nntrainer::sgemm_int4_cl(input, weight, scale, output, M, N, K, group_size);
  }
};

ComputeOps *get_cl_ops() {
  static ClComputeOps instance;
  return &instance;
}

} // namespace nntrainer

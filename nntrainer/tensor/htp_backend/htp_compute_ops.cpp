// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_compute_ops.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP (Hexagon/HMX) ComputeOps subclass — NPU dispatch entry point.
 *
 * Parallels ClComputeOps: overrides only the accelerated ops and their
 * supports_*() predicates; everything else falls through to the base
 * ComputeOps defaults. Actual HexKL calls live in hmx_ops/hexkl_mm.
 *
 * Compiled only when ENABLE_HEXKL is defined.
 *
 * Phase 0: skeleton. supports_shgemm() reports whether the NPU came up;
 * shgemm() forwards to the (stub) HMX op. Wiring this table into the
 * active dispatch (g_compute_ops / a Context) is a later phase.
 */

#ifdef ENABLE_HEXKL

#include <compute_ops.h>
#include <cpu_ops_table.h>
#include <hexkl_mm.h>
#include <htp_backend.h>

namespace nntrainer {

class HtpComputeOps : public CpuComputeOps {
public:
#ifdef ENABLE_FP16
  // F32 activations x F16 weights -> F32. First targeted kernel (§5.2).
  bool supports_shgemm() const override {
    return HtpBackend::global().enabled();
  }

  void shgemm(unsigned int order, bool tA, bool tB, unsigned int M,
              unsigned int N, unsigned int K, float alpha, const float *A,
              unsigned int lda, const _FP16 *B, unsigned int ldb, float beta,
              float *C, unsigned int ldc) override {
    hmx::shgemm_f32f16_f32(order, tA, tB, M, N, K, alpha, A, lda, B, ldb, beta,
                           C, ldc);
  }
#endif // ENABLE_FP16

  // U8 activations x I8 weights (WH-layout) -> I32 accumulator -> FP32 output.
  // No ENABLE_FP16 dependency.
  bool supports_shgemm_u8i8() const override {
    return HtpBackend::global().enabled();
  }

  void shgemm_u8i8(unsigned int order, unsigned int M, unsigned int N,
                   unsigned int K, const float *A, unsigned int /*lda*/,
                   const int8_t *B, const float *wt_scale,
                   const int32_t *zp_corr, float *C,
                   unsigned int /*ldc*/) override {
    hmx::shgemm_u8i8_i32(M, N, K, A, B, wt_scale, zp_corr, C);
  }
};

ComputeOps *get_htp_ops() {
  static HtpComputeOps instance;
  return &instance;
}

} // namespace nntrainer

#endif // ENABLE_HEXKL

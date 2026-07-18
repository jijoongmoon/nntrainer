// SPDX-License-Identifier: Apache-2.0
// ARM device sanity test for the KAI qsi4cxp matmul path end-to-end.
//
// For both the upstream fp32 matmul (variant 2 = 4x4x32 i8mm) and the
// forked fp16-dst matmul, this:
//   1. Generates a random fp32 weight tensor [N x K].
//   2. Packs it via Int4Utils::quantizeAndPackKai (Section A nibbles +
//      per-channel fp16 scales).
//   3. Reassembles the full KAI rhs_packed buffer via
//      assembleKaiRhsPacked (sums / scales*0.0625 / bias=0 trailer).
//   4. Runs the KAI matmul over a random LHS activation.
//   5. Compares against a reference matmul performed in fp32 with
//      dequantized int4 weights — same quantization noise on both
//      sides, so the residual measures only the matmul path itself.
//
// Standalone (no gtest, no libnntrainer). Cross-compiles directly with
// the NDK clang; pushes to the device and prints diagnostic numbers.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "fp16.h"
#include "int4_utils.h"

extern "C" {
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h"
#include "kai/matmul_clamp_f16_qai8dxp_qsi4cxp/kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h"
#include "kai/pack/kai_lhs_quant_pack_qai8dxp_f16.h"
#include "kai/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
}

using namespace nntrainer;

// Mirror of nntr_kai_gemm_qai8dxp_qsi4cxp_olp_single_thread for variant 2,
// inlined here so this test does not pull in libnntrainer.
static void run_kai_fp32(size_t m, size_t n, size_t k, const float *lhs_f32,
                         const uint8_t *rhs_packed, float *dst_f32) {
  constexpr size_t mr = 4, kr = 16, sr = 2;
  const size_t lhs_packed_size =
    kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr);
  std::vector<uint8_t> lhs_packed(lhs_packed_size);
  kai_run_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr, 0, lhs_f32,
                                     k * sizeof(float), lhs_packed.data());

  const size_t dst_stride = n * sizeof(float);
  kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
    m, n, k, lhs_packed.data(), rhs_packed, dst_f32, dst_stride,
    sizeof(float), -1e30f, 1e30f);
}

static void run_kai_fp16(size_t m, size_t n, size_t k, const __fp16 *lhs_f16,
                         const uint8_t *rhs_packed, __fp16 *dst_f16) {
  constexpr size_t mr = 4, kr = 16, sr = 2;
  const size_t lhs_packed_size =
    kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16(m, k, mr, kr, sr);
  std::vector<uint8_t> lhs_packed(lhs_packed_size);
  kai_run_lhs_quant_pack_qai8dxp_f16(m, k, mr, kr, sr, 0, lhs_f16,
                                     k * sizeof(__fp16), lhs_packed.data());

  const size_t dst_stride = n * sizeof(__fp16);
  kai_run_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
    m, n, k, lhs_packed.data(), rhs_packed, dst_f16, dst_stride,
    sizeof(__fp16), -65504.0f, 65504.0f);
}

// Dequantize Section A bytes back to fp32 weight matrix [N x K] using the
// same reverse-permutation as test/q4_roundtrip_kai.cpp's helper.
static void dequant_for_reference(const std::vector<uint8_t> &section_a,
                                  const std::vector<uint16_t> &fp16_scales,
                                  size_t N, size_t K,
                                  std::vector<float> &W_dq) {
  const size_t k_internal =
    ((K + Int4Utils::KAI_K_PAD_MULTIPLE - 1) / Int4Utils::KAI_K_PAD_MULTIPLE) *
    Int4Utils::KAI_K_PAD_MULTIPLE;
  const size_t super_row_count = (N + Int4Utils::KAI_NR - 1) / Int4Utils::KAI_NR;
  const size_t nibble_bytes_per_super_row =
    Int4Utils::KAI_NR * (k_internal / 2);
  const size_t block_length_in_bytes = Int4Utils::KAI_KR / Int4Utils::KAI_SR;

  W_dq.assign(N * K, 0.0f);

  // fp16 -> fp32 once
  std::vector<float> sc(N);
  for (size_t n = 0; n < N; ++n) {
    sc[n] = compute_fp16_to_fp32(fp16_scales[n]);
  }

  for (size_t sr_i = 0; sr_i < super_row_count; ++sr_i) {
    const uint8_t *src_row =
      section_a.data() + sr_i * nibble_bytes_per_super_row;
    for (size_t b = 0; b < nibble_bytes_per_super_row; ++b) {
      const size_t block_idx = b / block_length_in_bytes;
      const size_t block_byte_idx = b % block_length_in_bytes;
      const size_t super_block_idx = block_idx / Int4Utils::KAI_NR;
      const size_t nr_idx = block_idx % Int4Utils::KAI_NR;

      const size_t k_adjustment =
        ((block_byte_idx + super_block_idx * block_length_in_bytes) /
         Int4Utils::KAI_K_INTERLEAVE) *
        Int4Utils::KAI_K_INTERLEAVE;
      const size_t k0 =
        block_byte_idx + super_block_idx * block_length_in_bytes + k_adjustment;
      const size_t k1 = k0 + Int4Utils::KAI_K_INTERLEAVE;
      const size_t n0 = sr_i * Int4Utils::KAI_NR + nr_idx;
      if (n0 >= N)
        continue;

      const uint8_t stored = src_row[b] ^ 0x88;
      const int q0 = (stored & 0x0F) - 8;
      const int q1 = ((stored >> 4) & 0x0F) - 8;
      if (k0 < K)
        W_dq[n0 * K + k0] = (float)q0 * sc[n0];
      if (k1 < K)
        W_dq[n0 * K + k1] = (float)q1 * sc[n0];
    }
  }
}

static void run_shape(unsigned M, unsigned N, unsigned K, const char *tag) {
  std::mt19937 rng(987);
  std::normal_distribution<float> w_nd(0.0f, 0.05f);
  std::normal_distribution<float> a_nd(0.0f, 1.0f);

  // RHS weight [N x K] in row-major (this is what quantizeAndPackKai expects).
  std::vector<float> W((size_t)N * K);
  for (auto &v : W)
    v = w_nd(rng);

  // LHS activation [M x K].
  std::vector<float> A((size_t)M * K);
  for (auto &v : A)
    v = a_nd(rng);

  // Pack + assemble.
  std::vector<uint8_t> qw;
  std::vector<uint16_t> qs;
  Int4Utils::quantizeAndPackKai(W.data(), N, K, qw, qs);
  std::vector<uint8_t> rhs_packed;
  Int4Utils::assembleKaiRhsPacked(qw.data(), qs.data(), N, K, rhs_packed);

  // Reference: dequant int4 -> fp32 weights, then plain matmul. This shares
  // the same quantization noise as the KAI path, so the comparison measures
  // matmul fidelity only.
  std::vector<float> W_dq;
  dequant_for_reference(qw, qs, N, K, W_dq);
  std::vector<float> ref((size_t)M * N, 0.0f);
  for (unsigned i = 0; i < M; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (unsigned k = 0; k < K; ++k) {
        acc += A[(size_t)i * K + k] * W_dq[(size_t)j * K + k];
      }
      ref[(size_t)i * N + j] = acc;
    }
  }

  auto report = [&](const char *path, const std::vector<float> &got) {
    double se = 0.0, sigpow = 0.0, maxabs = 0.0;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
      double e = (double)got[i] - (double)ref[i];
      se += e * e;
      sigpow += (double)ref[i] * (double)ref[i];
      if (std::fabs(e) > maxabs)
        maxabs = std::fabs(e);
    }
    double mse = se / ((double)M * N);
    double rel = std::sqrt(se / (sigpow + 1e-12));
    printf("  [%s/%s] M=%u N=%u K=%u  MSE=%.3e relL2=%.3e maxAbs=%.3e\n",
           tag, path, M, N, K, mse, rel, maxabs);
  };

  // fp32 KAI matmul path.
  std::vector<float> out_f32((size_t)M * N);
  run_kai_fp32(M, N, K, A.data(), rhs_packed.data(), out_f32.data());
  report("fp32", out_f32);

  // fp16 KAI matmul path: cast input + cast output to compare on the same
  // scale as ref.
  std::vector<__fp16> A_f16((size_t)M * K);
  for (size_t i = 0; i < (size_t)M * K; ++i) A_f16[i] = (__fp16)A[i];
  std::vector<__fp16> out_f16_raw((size_t)M * N);
  run_kai_fp16(M, N, K, A_f16.data(), rhs_packed.data(), out_f16_raw.data());
  std::vector<float> out_f16((size_t)M * N);
  for (size_t i = 0; i < (size_t)M * N; ++i) out_f16[i] = (float)out_f16_raw[i];
  report("fp16", out_f16);
}

int main() {
  printf("[KAI matmul device sanity]\n");
  run_shape(1, 1536, 1536, "square_M=1");
  run_shape(4, 1536, 1536, "square_M=4");
  run_shape(16, 1536, 1536, "square_M=16");
  run_shape(64, 1536, 1536, "square_M=64");
  run_shape(1, 6144, 1536, "ffn_up_M=1");
  run_shape(4, 6144, 1536, "ffn_up_M=4");
  printf("[done]\n");
  return 0;
}

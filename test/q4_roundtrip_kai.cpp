// SPDX-License-Identifier: Apache-2.0
// QINT4 channel-wise (KAI qsi4cxp Section A) round-trip check:
//   pack:    Int4Utils::quantizeAndPackKai   (matches kai_run_rhs_pack_nxk_
//                                             qsi4cxp_qs4cxs1s0 nibble bytes,
//                                             trailer stripped)
//   dequant: reverse-permute Section A → uint4 → int4 → * scale
//
// Standalone (no gtest). Mirrors the structure of test/q4_roundtrip.cpp so
// the two paths can be compared side-by-side from CLI output.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <fp16.h>
#include <int4_utils.h>

using namespace nntrainer;

// Reverse-permute KAI Section A back into [N][K] dequantized floats. This
// mirrors Int4Utils::quantizeAndPackKai byte-for-byte, then undoes the
// XOR-0x88 and the uint4 = int4 + 8 encoding to recover the original signed
// 4-bit value, finally scales by the per-channel fp16 scale (expanded to
// fp32 at use time).
static void dequant_kai_section_a(const std::vector<uint8_t> &packed,
                                  const std::vector<uint16_t> &scales,
                                  unsigned N, unsigned K,
                                  std::vector<float> &out) {
  const size_t k_internal =
    ((size_t)K + Int4Utils::KAI_K_PAD_MULTIPLE - 1) /
    Int4Utils::KAI_K_PAD_MULTIPLE * Int4Utils::KAI_K_PAD_MULTIPLE;
  const size_t super_row_count =
    ((size_t)N + Int4Utils::KAI_NR - 1) / Int4Utils::KAI_NR;
  const size_t nibble_bytes_per_super_row =
    Int4Utils::KAI_NR * (k_internal / 2);
  const size_t block_length_in_bytes = Int4Utils::KAI_KR / Int4Utils::KAI_SR;
  const size_t kK = (size_t)K;

  out.assign((size_t)N * K, 0.0f);

  // Expand fp16 scales to fp32 once, since K iterations re-use them.
  std::vector<float> sc(N);
  for (unsigned n = 0; n < N; ++n) {
    sc[n] = compute_fp16_to_fp32(scales[n]);
  }

  for (size_t dst_row_idx = 0; dst_row_idx < super_row_count; ++dst_row_idx) {
    const uint8_t *dst_row =
      packed.data() + dst_row_idx * nibble_bytes_per_super_row;

    for (size_t dst_byte_idx = 0; dst_byte_idx < nibble_bytes_per_super_row;
         ++dst_byte_idx) {
      const size_t block_idx = dst_byte_idx / block_length_in_bytes;
      const size_t block_byte_idx = dst_byte_idx % block_length_in_bytes;
      const size_t super_block_idx = block_idx / Int4Utils::KAI_NR;
      const size_t nr_idx = block_idx % Int4Utils::KAI_NR;

      const size_t k_adjustment =
        ((block_byte_idx + super_block_idx * block_length_in_bytes) /
         Int4Utils::KAI_K_INTERLEAVE) *
        Int4Utils::KAI_K_INTERLEAVE;
      const size_t k0_idx =
        block_byte_idx + super_block_idx * block_length_in_bytes + k_adjustment;
      const size_t k1_idx = k0_idx + Int4Utils::KAI_K_INTERLEAVE;
      const size_t n0_idx = dst_row_idx * Int4Utils::KAI_NR + nr_idx;
      if (n0_idx >= N)
        continue;

      const uint8_t stored = dst_row[dst_byte_idx] ^ 0x88;
      const int u_lo = stored & 0x0F;
      const int u_hi = (stored >> 4) & 0x0F;
      const int q0 = u_lo - 8; // signed int4 in [-8, 7]
      const int q1 = u_hi - 8;
      const float s = sc[n0_idx];

      if (k0_idx < kK)
        out[n0_idx * kK + k0_idx] = (float)q0 * s;
      if (k1_idx < kK)
        out[n0_idx * kK + k1_idx] = (float)q1 * s;
    }
  }
}

static void run(unsigned K, unsigned N, const char *tag) {
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.0f, 0.05f);

  // Original FC weight as nntrainer stores it: [K, N] row-major.
  std::vector<float> W((size_t)K * N);
  for (auto &v : W)
    v = nd(rng);

  // ---- save path: transpose [K,N] -> [N,K], pack KAI Section A ----
  std::vector<float> Wt((size_t)N * K);
  for (unsigned k = 0; k < K; ++k)
    for (unsigned n = 0; n < N; ++n)
      Wt[(size_t)n * K + k] = W[(size_t)k * N + n];

  std::vector<uint8_t> qw;
  std::vector<uint16_t> qs;
  Int4Utils::quantizeAndPackKai(Wt.data(), N, K, qw, qs);

  // ---- dequant path: reverse-permute Section A -> [N, K] ----
  std::vector<float> deq;
  dequant_kai_section_a(qw, qs, N, K, deq);

  // rebuild [K, N]
  std::vector<float> R((size_t)K * N);
  for (unsigned n = 0; n < N; ++n)
    for (unsigned k = 0; k < K; ++k)
      R[(size_t)k * N + n] = deq[(size_t)n * K + k];

  // Expected packed size for sanity:
  const size_t expected = Int4Utils::kaiNibblePayloadBytes(N, K);

  double se = 0.0, maxabs = 0.0, sigpow = 0.0;
  for (size_t i = 0; i < (size_t)K * N; ++i) {
    double e = (double)R[i] - (double)W[i];
    se += e * e;
    sigpow += (double)W[i] * (double)W[i];
    if (std::fabs(e) > maxabs)
      maxabs = std::fabs(e);
  }
  double mse = se / ((double)K * N);
  double rel = se / (sigpow + 1e-12);
  printf("[%s/KAI] K=%u N=%u  qw=%zuB (exp %zuB) qs=%zu  MSE=%.3e relErr=%.3e "
         "maxAbs=%.3e\n  W[0..4]  =",
         tag, K, N, qw.size(), expected, qs.size(), mse, rel, maxabs);
  for (int i = 0; i < 5; ++i)
    printf(" % .5f", W[i]);
  printf("\n  R[0..4]  =");
  for (int i = 0; i < 5; ++i)
    printf(" % .5f", R[i]);
  printf("\n");
}

int main() {
  run(1536, 1536, "square");
  run(1536, 6144, "ffn_up");
  run(6144, 1536, "ffn_down");
  run(256, 1536, "ple_proj");
  return 0;
}

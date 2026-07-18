// SPDX-License-Identifier: Apache-2.0
// Byte-equality check between Int4Utils::quantizeAndPackKai (host C++
// emit) and KAI's own kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0 (the
// reference packer from nntrainer/tensor/cpu_backend/arm/kai/pack/).
//
// The KAI source is pure C with a __ARM_NEON guard, so it builds on x86
// for verification. We feed both packers from the same FP32 weights and
// scales, then strip KAI's per-super-row trailer (sums + scales + bias =
// 48 B for nr=4) and compare the remaining nibble payload byte-for-byte
// to our packer's output.
//
// Standalone (no gtest).

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <fp16.h>
#include <int4_utils.h>

extern "C" {
#include "kai/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
}

using namespace nntrainer;

static void quantize_raw_nibbles(const std::vector<float> &W, unsigned N,
                                 unsigned K, std::vector<uint8_t> &raw_nibbles,
                                 std::vector<float> &scales_fp32) {
  scales_fp32.assign(N, 1.0f);
  const size_t row_bytes = (K + 1) / 2;
  raw_nibbles.assign((size_t)N * row_bytes, 0u);
  for (unsigned n = 0; n < N; ++n) {
    float max_abs = 0.0f;
    for (unsigned k = 0; k < K; ++k) {
      float a = std::fabs(W[(size_t)n * K + k]);
      if (a > max_abs)
        max_abs = a;
    }
    scales_fp32[n] = (max_abs == 0.0f) ? 1.0f : (max_abs / 7.0f);
    const float s = scales_fp32[n];
    for (unsigned k = 0; k < K; k += 2) {
      const uint8_t u0 = Int4Utils::quantizeToInt4(W[(size_t)n * K + k], s);
      uint8_t byte = (uint8_t)((u0 + 8) & 0x0F);
      if (k + 1 < K) {
        const uint8_t u1 =
          Int4Utils::quantizeToInt4(W[(size_t)n * K + k + 1], s);
        byte |= (uint8_t)(((u1 + 8) & 0x0F) << 4);
      } else {
        byte |= (uint8_t)(8u << 4);
      }
      raw_nibbles[(size_t)n * row_bytes + (k / 2)] = byte;
    }
  }
}

static int run(unsigned K, unsigned N, const char *tag) {
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.0f, 0.05f);
  std::vector<float> W((size_t)N * K);
  for (auto &v : W)
    v = nd(rng);

  // 1) Our packer (FP32 in, Section A out).
  std::vector<uint8_t> qw_ours;
  std::vector<uint16_t> qs_ours;
  Int4Utils::quantizeAndPackKai(W.data(), N, K, qw_ours, qs_ours);

  // 2) KAI reference packer: feed pre-quantized nibbles + fp32 scales.
  //    Round-trip the scales through fp16 so the trailer bytes match
  //    Int4Utils::assembleKaiRhsPacked (which reads fp16 from disk).
  std::vector<uint8_t> raw_nibbles;
  std::vector<float> scales_fp32;
  quantize_raw_nibbles(W, N, K, raw_nibbles, scales_fp32);
  for (auto &s : scales_fp32) {
    s = compute_fp16_to_fp32(compute_fp32_to_fp16(s));
  }

  const size_t nr = Int4Utils::KAI_NR;
  const size_t kr = Int4Utils::KAI_KR;
  const size_t sr = Int4Utils::KAI_SR;
  const size_t kai_packed_size =
    kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(N, K, nr, kr, sr);
  std::vector<uint8_t> kai_packed(kai_packed_size, 0u);

  kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params;
  params.lhs_zero_point = 1;
  params.rhs_zero_point = 8;
  kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(1, N, K, nr, kr, sr,
                                         raw_nibbles.data(),
                                         /*bias=*/nullptr, scales_fp32.data(),
                                         kai_packed.data(),
                                         /*extra_bytes=*/0, &params);

  // 3) Strip KAI trailer per super-row, compare Section A.
  const size_t k_internal =
    ((size_t)K + Int4Utils::KAI_K_PAD_MULTIPLE - 1) /
    Int4Utils::KAI_K_PAD_MULTIPLE * Int4Utils::KAI_K_PAD_MULTIPLE;
  const size_t nibble_bytes_per_super_row = nr * (k_internal / 2);
  const size_t kai_stride =
    kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(K, nr, kr, sr);
  const size_t super_row_count = (N + nr - 1) / nr;

  if (qw_ours.size() != super_row_count * nibble_bytes_per_super_row) {
    printf("[%s/bytecompat] FAIL: ours size %zu != expected %zu\n", tag,
           qw_ours.size(), super_row_count * nibble_bytes_per_super_row);
    return 1;
  }

  size_t mismatches = 0;
  size_t first_mismatch = (size_t)-1;
  uint8_t first_a = 0, first_b = 0;
  for (size_t sr_i = 0; sr_i < super_row_count; ++sr_i) {
    const uint8_t *kai_row = kai_packed.data() + sr_i * kai_stride;
    const uint8_t *ours_row =
      qw_ours.data() + sr_i * nibble_bytes_per_super_row;
    for (size_t b = 0; b < nibble_bytes_per_super_row; ++b) {
      if (kai_row[b] != ours_row[b]) {
        ++mismatches;
        if (first_mismatch == (size_t)-1) {
          first_mismatch = sr_i * nibble_bytes_per_super_row + b;
          first_a = ours_row[b];
          first_b = kai_row[b];
        }
      }
    }
  }

  if (mismatches != 0) {
    printf("[%s/bytecompat] FAIL nibble: %zu mismatched bytes, first at "
           "offset %zu (ours=0x%02x vs kai=0x%02x)\n",
           tag, mismatches, first_mismatch, first_a, first_b);
    return 1;
  }

  // Now exercise Int4Utils::assembleKaiRhsPacked: reassemble Section A +
  // fp16 scales into a full KAI rhs_packed buffer (with trailer) and
  // compare byte-for-byte to the KAI packer's output. This locks the
  // load-time trailer reassembly (sums, scales*0.0625, bias) to the
  // matmul kernel's expectations.
  std::vector<uint8_t> ours_full;
  Int4Utils::assembleKaiRhsPacked(qw_ours.data(), qs_ours.data(), N, K,
                                  ours_full);
  if (ours_full.size() != kai_packed.size()) {
    printf("[%s/bytecompat] FAIL trailer: assembled size %zu vs KAI %zu\n",
           tag, ours_full.size(), kai_packed.size());
    return 1;
  }
  size_t t_mismatches = 0;
  size_t t_first = (size_t)-1;
  uint8_t t_a = 0, t_b = 0;
  for (size_t i = 0; i < ours_full.size(); ++i) {
    if (ours_full[i] != kai_packed[i]) {
      ++t_mismatches;
      if (t_first == (size_t)-1) {
        t_first = i;
        t_a = ours_full[i];
        t_b = kai_packed[i];
      }
    }
  }
  if (t_mismatches != 0) {
    printf("[%s/bytecompat] FAIL trailer: %zu mismatched bytes, first at "
           "offset %zu (ours=0x%02x vs kai=0x%02x)\n",
           tag, t_mismatches, t_first, t_a, t_b);
    return 1;
  }

  printf("[%s/bytecompat] OK: %zu super-rows, nibble %zu B + trailer 48 B = "
         "%zu B byte-identical with KAI rhs_pack\n",
         tag, super_row_count, nibble_bytes_per_super_row, kai_packed.size());
  return 0;
}

int main() {
  int rc = 0;
  rc |= run(32, 4, "tiny");
  rc |= run(256, 32, "small");
  rc |= run(1536, 1536, "square");
  rc |= run(1536, 6144, "ffn_up");
  rc |= run(6144, 1536, "ffn_down");
  return rc;
}

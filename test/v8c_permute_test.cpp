// SPDX-License-Identifier: Apache-2.0
//
// Self-contained byte-level test for KAI Section A → v8c nibble permuter.
// Build: g++ -std=c++17 -O0 -g test/v8c_permute_test.cpp -o /tmp/v8c_permute_test
//
// Reproduces the relevant slices of Int4Utils::quantizeToInt4,
// Int4Utils::quantizeAndPackKai, and
// make_v8c_weight_backing_from_kai_section_a inline so the test has no
// nntrainer dependencies. Then for every (n, k) it checks that the v8c
// byte at the kernel-expected offset decodes back to the same
// offset-encoded uint4 as a direct per-channel quantization.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr unsigned int N = 8;
static constexpr unsigned int K = 128;

// ---- quantizeToInt4 (verbatim from Int4Utils::quantizeToInt4) ----
static uint8_t quantizeToInt4(float weight, float scale) {
  float div = std::nearbyintf(weight / scale);
  if (std::isnan(div))
    div = 0.0f;
  if (div < -8.0f)
    div = -8.0f;
  if (div > 7.0f)
    div = 7.0f;
  int q = (int)div;
  return (uint8_t)(q & 0xF); // two's complement int4 in low 4 bits
}

// ---- Mini quantizeAndPackKai (Section A nibble payload only; no trailer) ----
// Mirrors nntrainer/tensor/int4_utils.cpp:quantizeAndPackKai.
static void quantizeAndPackKaiSectionA(const float *weights, size_t n_rows,
                                       size_t n_cols,
                                       std::vector<uint8_t> &out_section_a,
                                       std::vector<float> &out_scales) {
  constexpr size_t KAI_NR = 4;
  constexpr size_t KAI_KR = 16;
  constexpr size_t KAI_SR = 2;
  constexpr size_t KAI_K_PAD_MULTIPLE = 32;
  constexpr size_t KAI_K_INTERLEAVE = 16;
  out_scales.resize(n_rows);
  // 1) per-channel scales (amax / 7)
  for (size_t n = 0; n < n_rows; ++n) {
    float amax = 0.0f;
    for (size_t k = 0; k < n_cols; ++k) {
      float a = std::abs(weights[n * n_cols + k]);
      if (a > amax)
        amax = a;
    }
    out_scales[n] = (amax == 0.0f) ? 1.0f : (amax / 7.0f);
  }
  // 2) raw_nibbles in N × ceil(K/2) bytes, offset-encoded
  const size_t rhs_row_bytes = (n_cols + 1) / 2;
  std::vector<uint8_t> raw_nibbles(n_rows * rhs_row_bytes, 0);
  for (size_t n = 0; n < n_rows; ++n) {
    float s = out_scales[n];
    for (size_t k = 0; k < n_cols; k += 2) {
      uint8_t u0 = quantizeToInt4(weights[n * n_cols + k], s);
      uint8_t byte = (uint8_t)((u0 + 8) & 0x0F);
      if (k + 1 < n_cols) {
        uint8_t u1 = quantizeToInt4(weights[n * n_cols + k + 1], s);
        byte |= (uint8_t)(((u1 + 8) & 0x0F) << 4);
      } else {
        byte |= (uint8_t)(8u << 4); // pad nibble
      }
      raw_nibbles[n * rhs_row_bytes + (k / 2)] = byte;
    }
  }
  // 3) Permute into KAI Section A super-row form (no trailer)
  const size_t k_internal =
    ((n_cols + KAI_K_PAD_MULTIPLE - 1) / KAI_K_PAD_MULTIPLE) *
    KAI_K_PAD_MULTIPLE;
  const size_t super_row_count = (n_rows + KAI_NR - 1) / KAI_NR;
  const size_t nibble_bytes_per_super_row = KAI_NR * (k_internal / 2);
  const size_t dst_num_bytes_per_row = nibble_bytes_per_super_row;
  const size_t block_length_in_bytes = KAI_KR / KAI_SR; // 8
  const uint8_t pad_byte = (uint8_t)(8u | (8u << 4));

  out_section_a.assign(super_row_count * nibble_bytes_per_super_row, 0);
  for (size_t dst_row_idx = 0; dst_row_idx < super_row_count; ++dst_row_idx) {
    uint8_t *dst_row =
      out_section_a.data() + dst_row_idx * nibble_bytes_per_super_row;
    for (size_t dst_byte_idx = 0; dst_byte_idx < dst_num_bytes_per_row;
         ++dst_byte_idx) {
      const size_t block_idx = dst_byte_idx / block_length_in_bytes;
      const size_t block_byte_idx = dst_byte_idx % block_length_in_bytes;
      const size_t super_block_idx = block_idx / KAI_NR;
      const size_t nr_idx = block_idx % KAI_NR;
      const size_t k_adjustment =
        ((block_byte_idx + super_block_idx * block_length_in_bytes) /
         KAI_K_INTERLEAVE) *
        KAI_K_INTERLEAVE;
      const size_t k0_idx =
        block_byte_idx + super_block_idx * block_length_in_bytes + k_adjustment;
      const size_t k1_idx = k0_idx + KAI_K_INTERLEAVE;
      const size_t n0_idx = dst_row_idx * KAI_NR + nr_idx;
      const size_t n0_valid_idx =
        (n0_idx < n_rows) ? n0_idx : (n_rows - 1);
      uint8_t byte0 = pad_byte, byte1 = pad_byte;
      if (k0_idx < n_cols)
        byte0 = raw_nibbles[n0_valid_idx * rhs_row_bytes + (k0_idx / 2)];
      if (k1_idx < n_cols)
        byte1 = raw_nibbles[n0_valid_idx * rhs_row_bytes + (k1_idx / 2)];
      const size_t shift_right_x0 = (k0_idx % 2) * 4;
      const size_t shift_right_x1 = (k1_idx % 2) * 4;
      const uint8_t lo = (uint8_t)((byte0 >> shift_right_x0) & 0x0F);
      const uint8_t hi = (uint8_t)((byte1 >> shift_right_x1) & 0x0F);
      uint8_t out_byte = (uint8_t)(lo | (hi << 4));
      *dst_row = (uint8_t)(out_byte ^ 0x88);
      ++dst_row;
    }
  }
}

// ---- Permuter under test (verbatim from
//      make_v8c_weight_backing_from_kai_section_a, just no GPU upload) ----
static void kaiSectionAToV8cPacked(const uint8_t *section_a, unsigned int n,
                                   unsigned int k,
                                   std::vector<uint8_t> &out_packed) {
  constexpr size_t KAI_NR = 4;
  constexpr size_t KAI_KR_BY_SR = 8;
  constexpr size_t KAI_K_INTERLEAVE = 16;
  constexpr size_t KAI_BYTES_PER_KBLK = 2 * KAI_NR * KAI_KR_BY_SR;

  const size_t k_blocks = k / 32;
  const size_t super_row_count = n / KAI_NR;
  const size_t nibble_bytes_per_super_row = KAI_NR * (k / 2);
  const size_t v8c_row_bytes = (size_t)k / 2;
  out_packed.assign((size_t)n * v8c_row_bytes, 0);

  for (size_t sr = 0; sr < super_row_count; ++sr) {
    const uint8_t *sr_base = section_a + sr * nibble_bytes_per_super_row;
    for (size_t nr = 0; nr < KAI_NR; ++nr) {
      const size_t n_idx = sr * KAI_NR + nr;
      uint8_t *v8c_row = out_packed.data() + n_idx * v8c_row_bytes;
      for (size_t kblk = 0; kblk < k_blocks; ++kblk) {
        const uint8_t *blk_a =
          sr_base + kblk * KAI_BYTES_PER_KBLK + nr * KAI_KR_BY_SR;
        const uint8_t *blk_b = blk_a + KAI_NR * KAI_KR_BY_SR;
        uint8_t k_nibbles[32];
        for (size_t i = 0; i < KAI_KR_BY_SR; ++i) {
          const uint8_t b_a = blk_a[i] ^ 0x88;
          k_nibbles[i + 0] = (uint8_t)(b_a & 0x0F);
          k_nibbles[i + KAI_K_INTERLEAVE] = (uint8_t)((b_a >> 4) & 0x0F);
          const uint8_t b_b = blk_b[i] ^ 0x88;
          k_nibbles[i + KAI_KR_BY_SR] = (uint8_t)(b_b & 0x0F);
          k_nibbles[i + KAI_KR_BY_SR + KAI_K_INTERLEAVE] =
            (uint8_t)((b_b >> 4) & 0x0F);
        }
        uint8_t *v8c_kblk = v8c_row + kblk * 16;
        for (size_t c = 0; c < 4; ++c) {
          for (size_t b = 0; b < 4; ++b) {
            const uint8_t qL = k_nibbles[c * 8 + b];
            const uint8_t qH = k_nibbles[c * 8 + b + 4];
            v8c_kblk[c * 4 + b] = (uint8_t)((qH << 4) | qL);
          }
        }
      }
    }
  }
}

int main() {
  std::vector<float> w(N * K);
  for (unsigned int n = 0; n < N; ++n)
    for (unsigned int k = 0; k < K; ++k)
      w[n * K + k] = n * 100.0f + k * 0.1f - 1.5f * ((k + n) % 2 == 0 ? 1 : -1);

  std::vector<uint8_t> section_a;
  std::vector<float> scales;
  quantizeAndPackKaiSectionA(w.data(), N, K, section_a, scales);
  std::printf("KAI Section A: %zu bytes, %zu scales\n", section_a.size(),
              scales.size());

  // Reference: directly quantize each (n,k) to offset-encoded uint4
  std::vector<uint8_t> ref(N * K);
  for (unsigned int n = 0; n < N; ++n)
    for (unsigned int k = 0; k < K; ++k)
      ref[n * K + k] =
        (uint8_t)((quantizeToInt4(w[n * K + k], scales[n]) + 8) & 0x0F);

  std::vector<uint8_t> packed;
  kaiSectionAToV8cPacked(section_a.data(), N, K, packed);

  // Decode packed bytes the way the kernel walks them
  const size_t v8c_row_bytes = K / 2;
  int mismatches = 0;
  for (unsigned int n = 0; n < N; ++n) {
    for (unsigned int k = 0; k < K; ++k) {
      const size_t kbl = k / 32;
      const size_t kp = k % 32;
      const size_t c = kp / 8;
      const size_t bp = kp - c * 8;
      const size_t b = bp & 0x3u;
      const bool is_hi = (bp >= 4);
      const size_t byte_off = kbl * 16 + c * 4 + b;
      const uint8_t byte = packed[n * v8c_row_bytes + byte_off];
      const uint8_t got = is_hi ? (uint8_t)((byte >> 4) & 0x0F)
                                : (uint8_t)(byte & 0x0F);
      const uint8_t exp = ref[n * K + k];
      if (got != exp) {
        if (mismatches < 32) {
          std::printf("MISMATCH n=%u k=%u kbl=%zu kp=%zu c=%zu bp=%zu b=%zu "
                      "is_hi=%d byte_off=%zu byte=0x%02x got=%u exp=%u "
                      "(w=%.3f scale=%.3f)\n",
                      n, k, kbl, kp, c, bp, b, (int)is_hi, byte_off, byte,
                      got, exp, w[n * K + k], scales[n]);
        }
        ++mismatches;
      }
    }
  }
  if (mismatches == 0) {
    std::printf("OK: all %u*%u = %u nibbles match.\n", N, K, N * K);
    return 0;
  }
  std::printf("FAIL: %d mismatches out of %u nibbles.\n", mismatches, N * K);
  return 1;
}

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_sampling.cpp
 * @date   01 October 2026
 * @brief  Host tests for the portable token sampler (token_sampler.h)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details Pins the properties sampled decoding relies on to be reproducible:
 * the RNG stream is the standard one, the same logits and seed give the same
 * token, top-k ordering and its tie-break are deterministic, and the top-p
 * boundary is the documented (Hugging Face) one. No GPU needed.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <token_sampler.h>

using causallm::kDefaultSamplingSeed;
using causallm::sampleIndex;
using causallm::sampleToken;
using causallm::samplingRanksBefore;
using causallm::SamplingRng;
using causallm::samplingUniform;

namespace {

/** @brief A fixed, non-trivial logits row */
std::vector<float> makeLogits(int n) {
  std::vector<float> l(n);
  for (int i = 0; i < n; ++i)
    l[i] = std::sin(0.37f * i) * 4.0f + 0.001f * (i % 7);
  return l;
}

/**
 * @brief The sampler as it was before the top-k selection became a one-pass
 *        heap: a V-sized candidate array and std::partial_sort. Kept as the
 *        reference the selection must reproduce exactly.
 */
unsigned int referenceSampleToken(const float *logits, int len,
                                  float temperature, unsigned int top_k,
                                  float top_p, SamplingRng &rng) {
  if (temperature <= 1e-5f)
    return static_cast<unsigned int>(
      std::distance(logits, std::max_element(logits, logits + len)));
  std::vector<std::pair<unsigned int, float>> cand(len);
  for (int i = 0; i < len; ++i)
    cand[i] = {static_cast<unsigned int>(i), logits[i] / temperature};
  std::size_t k = static_cast<std::size_t>(len);
  if (top_k > 0 && top_k < static_cast<unsigned int>(len))
    k = top_k;
  std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                    samplingRanksBefore);
  const double max_score = cand[0].second;
  if (!std::isfinite(max_score))
    return cand[0].first;
  std::vector<double> probs(k);
  double sum = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    probs[i] = std::exp(static_cast<double>(cand[i].second) - max_score);
    sum += probs[i];
  }
  for (std::size_t i = 0; i < k; ++i)
    probs[i] /= sum;
  std::size_t keep = k;
  if (top_p > 0.0f && top_p < 1.0f) {
    const double p = top_p;
    double above = 0.0;
    keep = 1;
    for (std::size_t i = 1; i < k; ++i) {
      above += probs[i - 1];
      if (!(above < p))
        break;
      keep = i + 1;
    }
  }
  return cand[sampleIndex(probs.data(), keep, rng)].first;
}

/**
 * @brief Random logits drawn from a small set of levels, so exact ties are
 *        common everywhere in the row, including at the top-k boundary.
 */
std::vector<float> makeTiedLogits(int n, int levels, SamplingRng &gen) {
  std::vector<float> l(n);
  for (int i = 0; i < n; ++i)
    l[i] = static_cast<float>(gen() % levels) * 0.25f - 4.0f;
  return l;
}

/**
 * @brief Random continuous logits.
 */
std::vector<float> makeRandomLogits(int n, SamplingRng &gen) {
  std::vector<float> l(n);
  for (int i = 0; i < n; ++i)
    l[i] = static_cast<float>(causallm::samplingUniform(gen) * 30.0 - 15.0);
  return l;
}

} // namespace

/**
 * @brief The engine's output sequence is fixed by the standard: the 10000th
 *        value of a default-seeded mt19937_64 is 9981545732273789042.
 */
TEST(causallm_sampling, rng_stream_is_standard_p) {
  SamplingRng rng(5489u); // std::mt19937_64::default_seed
  rng.discard(9999);
  EXPECT_EQ(rng(), 9981545732273789042ULL);
}

/**
 * @brief The uniform draw is 53 bits in [0, 1).
 */
TEST(causallm_sampling, uniform_range_p) {
  SamplingRng rng(1);
  for (int i = 0; i < 10000; ++i) {
    double u = samplingUniform(rng);
    EXPECT_GE(u, 0.0);
    EXPECT_LT(u, 1.0);
  }
}

/**
 * @brief Same logits + same seed => same token stream.
 */
TEST(causallm_sampling, same_seed_same_tokens_p) {
  const auto logits = makeLogits(4096);
  SamplingRng a(kDefaultSamplingSeed), b(kDefaultSamplingSeed);
  for (int step = 0; step < 200; ++step) {
    EXPECT_EQ(sampleToken(logits.data(), 4096, 0.7f, 40, 0.95f, a),
              sampleToken(logits.data(), 4096, 0.7f, 40, 0.95f, b));
  }
}

/**
 * @brief A different seed gives a different stream (sanity for the above).
 */
TEST(causallm_sampling, different_seed_differs_p) {
  const auto logits = makeLogits(4096);
  SamplingRng a(1), b(2);
  int differ = 0;
  for (int step = 0; step < 200; ++step)
    differ += sampleToken(logits.data(), 4096, 1.0f, 64, 0.95f, a) !=
              sampleToken(logits.data(), 4096, 1.0f, 64, 0.95f, b);
  EXPECT_GT(differ, 0);
}

/**
 * @brief top-k with ties keeps the lowest ids; top_p just above one
 *        candidate's mass then keeps only the best-ranked one.
 */
TEST(causallm_sampling, topk_tie_break_lower_id_p) {
  std::vector<float> logits(100, 0.0f);
  SamplingRng rng(kDefaultSamplingSeed);
  std::vector<int> hits(100, 0);
  for (int i = 0; i < 3000; ++i)
    ++hits[sampleToken(logits.data(), 100, 1.0f, 3, 1.0f, rng)];
  EXPECT_GT(hits[0], 0);
  EXPECT_GT(hits[1], 0);
  EXPECT_GT(hits[2], 0);
  for (int i = 3; i < 100; ++i)
    EXPECT_EQ(hits[i], 0) << "token " << i;
}

/**
 * @brief Ranking is by score first: the best token wins whenever only one
 *        candidate survives, regardless of position or ties further down.
 */
TEST(causallm_sampling, topk_order_deterministic_p) {
  std::vector<float> logits = {1.0f, 5.0f, 5.0f, 3.0f, 5.0f, -2.0f};
  SamplingRng rng(7);
  for (int i = 0; i < 50; ++i) // top_k 1: always the lowest id among the best
    EXPECT_EQ(sampleToken(logits.data(), 6, 1.0f, 1, 1.0f, rng), 1u);
  // top_k 3 keeps exactly ids 1, 2, 4 (all scored 5)
  std::vector<int> hits(6, 0);
  for (int i = 0; i < 3000; ++i)
    ++hits[sampleToken(logits.data(), 6, 1.0f, 3, 1.0f, rng)];
  EXPECT_EQ(hits[0] + hits[3] + hits[5], 0);
  EXPECT_GT(hits[1], 0);
  EXPECT_GT(hits[2], 0);
  EXPECT_GT(hits[4], 0);
}

/**
 * @brief top-p boundary: a candidate whose predecessors already hold exactly
 *        top_p of the mass is dropped (Hugging Face rule); just above it is
 *        kept.
 */
TEST(causallm_sampling, topp_boundary_exclusive_p) {
  std::vector<float> logits = {0.0f, 0.0f}; // exactly 0.5 / 0.5
  SamplingRng rng(kDefaultSamplingSeed);
  for (int i = 0; i < 200; ++i)
    EXPECT_EQ(sampleToken(logits.data(), 2, 1.0f, 0, 0.5f, rng), 0u);
  int second = 0;
  for (int i = 0; i < 2000; ++i)
    second += sampleToken(logits.data(), 2, 1.0f, 0, 0.51f, rng) == 1u;
  EXPECT_GT(second, 0);
}

/**
 * @brief Near-zero temperature is argmax, lowest id on a tie, and does not
 *        consume the RNG.
 */
TEST(causallm_sampling, zero_temperature_is_argmax_p) {
  std::vector<float> logits = {0.5f, 2.0f, 2.0f, 1.0f};
  SamplingRng a(3), b(3);
  EXPECT_EQ(sampleToken(logits.data(), 4, 0.0f, 40, 0.95f, a), 1u);
  EXPECT_EQ(a(), b());
}

/**
 * @brief Inverse CDF follows the weights.
 */
TEST(causallm_sampling, sample_index_frequencies_p) {
  const double w[2] = {1.0, 3.0};
  SamplingRng rng(kDefaultSamplingSeed);
  int ones = 0;
  const int n = 40000;
  for (int i = 0; i < n; ++i)
    ones += sampleIndex(w, 2, rng) == 1u;
  EXPECT_NEAR(static_cast<double>(ones) / n, 0.75, 0.01);
}

/**
 * @brief Zero weights are never picked.
 */
TEST(causallm_sampling, sample_index_skips_zero_weight_p) {
  const double w[4] = {0.0, 1.0, 0.0, 0.0};
  SamplingRng rng(kDefaultSamplingSeed);
  for (int i = 0; i < 1000; ++i)
    EXPECT_EQ(sampleIndex(w, 4, rng), 1u);
}

/**
 * @brief The one-pass top-k selection draws exactly the token (and leaves
 *        exactly the RNG state) of the full-sort reference, on random rows
 *        with and without ties, for every top_k / temperature / top_p shape.
 */
TEST(causallm_sampling, topk_selection_matches_full_sort_p) {
  SamplingRng gen(2024);
  const unsigned int ks[] = {0, 1, 2, 5, 40, 64, 100, 999, 1000, 5000};
  const float temps[] = {0.3f, 0.7f, 1.0f, 1.7f};
  const float tps[] = {0.0f, 0.5f, 0.9f, 0.95f, 1.0f};
  int draws = 0;
  for (int row = 0; row < 24; ++row) {
    const int len = (row % 3 == 0) ? 1000 : 4099;
    const auto logits = (row % 2 == 0) ? makeTiedLogits(len, 7 + row, gen)
                                       : makeRandomLogits(len, gen);
    for (unsigned int k : ks) {
      for (float t : temps) {
        for (float tp : tps) {
          SamplingRng a(row * 131u + k), b(row * 131u + k);
          for (int step = 0; step < 3; ++step) {
            ASSERT_EQ(sampleToken(logits.data(), len, t, k, tp, a),
                      referenceSampleToken(logits.data(), len, t, k, tp, b))
              << "row " << row << " k " << k << " T " << t << " p " << tp;
            ++draws;
          }
          EXPECT_EQ(a(), b());
        }
      }
    }
  }
  EXPECT_GT(draws, 10000);
}

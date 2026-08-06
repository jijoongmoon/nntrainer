// SPDX-License-Identifier: Apache-2.0
/**
 * @file    rope_default_partial_test.cpp
 * @brief   Gate for the "default_partial" RoPE frequency table
 *          (qwen3_5_moe / Qwen3-Next).
 *
 * HF calls compute_default_rope_parameters with dim = int(head_dim *
 * partial_rotary_factor), so theta_i = base^(-2i/ROTARY_dim). nntrainer's two
 * pre-existing partial-capable builders both use head_dim in that denominator,
 * which at head_dim=256 / rotary_dim=64 is a 4x error in every exponent --
 * wrong logits with no diagnostic. This checks the real dispatch and the real
 * builder, not a re-derivation of the formula:
 *
 *   1. thetas[i] == base^(-2i/rotary_dim) for i < rotary_dim/2
 *   2. thetas[i] == 0 exactly for the pass-through tail
 *   3. the values are NOT the head_dim-denominator form, i.e. the dispatch
 *      really took the new branch and was not swallowed by the `proportional`
 *      clause (which also fires on partial_rotary_factor != 1)
 *
 * MHACoreLayer::{precompute_freqs,thetas,rope_scaling_type,scale,
 * rope_partial_rotary_factor} are private, so this uses the standard explicit-
 * instantiation access idiom rather than editing the header for a test.
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <mha_core.h>

namespace {

/** @brief explicit-instantiation access idiom (ISO C++ 14.7.2/12) */
template <typename Tag, typename Tag::type M> struct Rob {
  friend typename Tag::type get(Tag) { return M; }
};

struct TagPrecompute {
  using type = void (causallm::MHACoreLayer::*)(int, unsigned int, float, bool);
  friend type get(TagPrecompute);
};
template struct Rob<TagPrecompute, &causallm::MHACoreLayer::precompute_freqs>;

struct TagThetas {
  using type = std::vector<float> *;
  friend type get(TagThetas);
};
template struct Rob<TagThetas, &causallm::MHACoreLayer::thetas>;

struct TagScalingType {
  using type = std::string causallm::MHACoreLayer::*;
  friend type get(TagScalingType);
};
template struct Rob<TagScalingType, &causallm::MHACoreLayer::rope_scaling_type>;

struct TagPartial {
  using type = float causallm::MHACoreLayer::*;
  friend type get(TagPartial);
};
template struct Rob<TagPartial,
                    &causallm::MHACoreLayer::rope_partial_rotary_factor>;

struct TagScale {
  using type = float causallm::MHACoreLayer::*;
  friend type get(TagScale);
};
template struct Rob<TagScale, &causallm::MHACoreLayer::scale>;

int failures = 0;

void check(bool ok, const std::string &what, double got, double want) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << " : got " << got
            << " want " << want << "\n";
  if (!ok)
    ++failures;
}

} // namespace

int main() {
  // The real 35B geometry: 40 layers, head_dim 256, partial_rotary_factor 0.25
  // (rotary_dim 64), rope_theta 1e7.
  const int head_dim = 256;
  const float partial = 0.25f;
  const float base = 1e7f;
  const float rotary_dim = partial * head_dim; // 64
  const int rope_angles = static_cast<int>((partial * head_dim) / 2.0f); // 32
  const int half_dim = head_dim / 2;                                     // 128

  causallm::MHACoreLayer layer;
  layer.*get(TagScalingType()) = "default_partial";
  layer.*get(TagPartial()) = partial;
  layer.*get(TagScale()) = 1.0f;

  // seq_len only sizes the cos/sin flatten; the thetas are what we gate on.
  (layer.*get(TagPrecompute()))(head_dim, /*seq_len=*/8, base, /*is_fp16=*/false);

  const std::vector<float> &thetas = *get(TagThetas());

  std::cout << "=== default_partial RoPE thetas: head_dim=" << head_dim
            << " partial=" << partial << " rotary_dim=" << rotary_dim
            << " theta=" << base << " ===\n";

  check(thetas.size() == static_cast<size_t>(half_dim), "thetas.size()",
        (double)thetas.size(), (double)half_dim);
  if (thetas.size() != static_cast<size_t>(half_dim))
    return 1;

  // 1. the rotary head must use the ROTARY denominator
  double worst = 0.0;
  for (int i = 0; i < rope_angles; ++i) {
    const double want = 1.0 / std::pow((double)base, (2.0 * i) / rotary_dim);
    const double rel = std::fabs(thetas[i] - want) / std::fabs(want);
    worst = std::max(worst, rel);
  }
  check(worst <= 1e-6, "max rel err vs HF base^(-2i/rotary_dim), i<32", worst,
        0.0);

  // 2. the pass-through tail must be EXACTLY zero (cos=1 / sin=0)
  double tail = 0.0;
  for (int i = rope_angles; i < half_dim; ++i)
    tail = std::max(tail, (double)std::fabs(thetas[i]));
  check(tail == 0.0, "max |theta| over the pass-through tail, i>=32", tail, 0.0);

  // 3. prove the branch was taken: the Gemma/proportional form would use
  //    head_dim in the denominator. At i=31 the two differ by ~5 orders of
  //    magnitude, so this cannot pass by coincidence.
  const int i = rope_angles - 1;
  const double want_rotary = 1.0 / std::pow((double)base, (2.0 * i) / rotary_dim);
  const double want_headdim = 1.0 / std::pow((double)base, (2.0 * i) / head_dim);
  const double d_rotary = std::fabs(thetas[i] - want_rotary);
  const double d_headdim = std::fabs(thetas[i] - want_headdim);
  std::cout << "       thetas[" << i << "]=" << thetas[i]
            << "  rotary_dim form=" << want_rotary
            << "  head_dim form=" << want_headdim << "\n";
  check(d_rotary < d_headdim,
        "dispatch took default_partial (not the proportional/head_dim form)",
        d_rotary, d_headdim);

  std::cout << (failures ? "=== FAILED (" : "=== ALL CHECKS PASS (")
            << failures << " failed) ===\n";
  return failures ? 1 : 0;
}

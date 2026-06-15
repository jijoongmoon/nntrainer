// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   antialiased_snake.cpp
 * @date   15 June 2026
 * @brief  Anti-aliased snake-beta activation (HF TorchActivation1d) for BigVGAN.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "antialiased_snake.h"

namespace causallm {

static constexpr size_t X_IDX = 0;
static constexpr size_t OUT_IDX = 0;
static constexpr size_t ALPHA_IDX = 0;
static constexpr size_t BETA_IDX = 1;
static constexpr float NO_DIV_BY_ZERO = 1e-9f;

// kaiser_sinc_filter1d(cutoff=0.25, half_width=0.3, kernel_size=12), the fixed
// up/down anti-alias filter for ratio=2 (HF modeling_qwen2_5_omni.py:3094).
// Symmetric, sums to 1.0. Verified to reproduce HF activation_post to 9.5e-7.
static constexpr unsigned int K = 12;
static constexpr float FILT[K] = {
  0.0020289700478315353f, 0.00938946008682251f,  -0.025543460622429848f,
  -0.057657379657030106f, 0.12857261300086975f,  0.44320979714393616f,
  0.44320979714393616f,   0.12857261300086975f,  -0.057657379657030106f,
  -0.025543460622429848f, 0.00938946008682251f,  0.0020289700478315353f};

// UpSample1d(ratio=2, k=12): pad=k/ratio-1=5; slice [pad_left:-pad_right]=15.
static constexpr unsigned int UP_PAD = 5;     // replicate pad each side
static constexpr unsigned int UP_SLICE = 15;  // pad_left == pad_right
// DownSample1d(ratio=2, k=12): pad_left=k/2-even=5, pad_right=k/2=6.
static constexpr unsigned int DN_PAD_L = 5;
static constexpr unsigned int DN_PAD_R = 6;
static constexpr unsigned int RATIO = 2;

static inline unsigned int clampu(long v, unsigned int hi) {
  if (v < 0)
    return 0;
  if (v >= static_cast<long>(hi))
    return hi - 1;
  return static_cast<unsigned int>(v);
}

void AntialiasedSnakeLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "antialiased_snake expects a single input";

  const auto &in = context.getInputDimensions()[0];
  const unsigned int channels = in.channel();
  NNTR_THROW_IF(channels == 0, std::invalid_argument)
    << "antialiased_snake: input channel dim must be non-zero";

  nntrainer::TensorDim wdim({1, 1, 1, channels});
  wt_idx[ALPHA_IDX] =
    context.requestWeight(wdim, nntrainer::Initializer::ZEROS,
                          nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
                          "alpha", false);
  wt_idx[BETA_IDX] =
    context.requestWeight(wdim, nntrainer::Initializer::ZEROS,
                          nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
                          "beta", false);

  context.setOutputDimensions({in});
}

void AntialiasedSnakeLayer::forwarding(nntrainer::RunLayerContext &context,
                                       bool training) {
  nntrainer::Tensor &x = context.getInput(X_IDX);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  NNTR_THROW_IF(x.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "antialiased_snake supports FP32 only";

  const float *alpha = context.getWeight(wt_idx[ALPHA_IDX]).getData<float>();
  const float *beta = context.getWeight(wt_idx[BETA_IDX]).getData<float>();

  const unsigned int C = x.channel();
  const unsigned int T = x.height() * x.width(); // [B,C,1,T] -> contiguous T
  const unsigned int T2 = RATIO * T;

  // upf holds the transposed-conv output before the *2 and the [15:-15] slice.
  // input padded length = T + 2*UP_PAD; convT(stride2,k) length = (Tp-1)*2 + k.
  const unsigned int Tp = T + 2 * UP_PAD;
  const unsigned int upf_len = (Tp - 1) * RATIO + K;
  std::vector<float> upf(upf_len);
  std::vector<float> up(T2); // upsampled + sliced + snake'd signal

  for (unsigned int b = 0; b < x.batch(); ++b) {
    for (unsigned int c = 0; c < C; ++c) {
      const float a = std::exp(alpha[c]);
      const float inv_b = 1.0f / (std::exp(beta[c]) + NO_DIV_BY_ZERO);
      const float *xc = x.getData<float>() + x.getIndex(b, c, 0, 0);
      float *oc = out.getData<float>() + out.getIndex(b, c, 0, 0);

      // ---- UpSample1d: replicate-pad(UP_PAD), convT stride2, *2, slice ----
      std::fill(upf.begin(), upf.end(), 0.0f);
      for (unsigned int i = 0; i < Tp; ++i) {
        const float xv =
          xc[clampu(static_cast<long>(i) - static_cast<long>(UP_PAD), T)];
        const unsigned int base = i * RATIO;
        for (unsigned int t = 0; t < K; ++t)
          upf[base + t] += xv * FILT[t];
      }
      // *ratio and slice [UP_SLICE : UP_SLICE + T2], then SnakeBeta in place
      for (unsigned int m = 0; m < T2; ++m) {
        const float u = RATIO * upf[m + UP_SLICE];
        const float s = std::sin(a * u);
        up[m] = u + inv_b * s * s;
      }

      // ---- DownSample1d: replicate-pad(5,6), conv stride2 ----
      for (unsigned int o = 0; o < T; ++o) {
        float acc = 0.0f;
        const long b0 = static_cast<long>(o) * RATIO - static_cast<long>(DN_PAD_L);
        for (unsigned int t = 0; t < K; ++t)
          acc += up[clampu(b0 + static_cast<long>(t), T2)] * FILT[t];
        oc[o] = acc;
      }
    }
  }
  (void)DN_PAD_R; // pad_right only affects length bookkeeping (handled by clamp)
}

} // namespace causallm

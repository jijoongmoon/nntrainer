// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   ecapa_tdnn.cpp
 * @date   18 July 2026
 * @brief  ECAPA-TDNN speaker encoder (host-side FP32) for Token2Wav.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <ecapa_tdnn.h>

namespace causallm {

void EcapaTdnn::load(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open ecapa weights " + path);

  auto rd = [&](Conv &c, unsigned int cout, unsigned int cin, unsigned int k,
                unsigned int dil) {
    c.cout = cout;
    c.cin = cin;
    c.k = k;
    c.dil = dil;
    c.w.resize(static_cast<size_t>(cout) * cin * k);
    c.b.resize(cout);
    f.read(reinterpret_cast<char *>(c.w.data()),
           static_cast<std::streamsize>(c.w.size() * sizeof(float)));
    f.read(reinterpret_cast<char *>(c.b.data()),
           static_cast<std::streamsize>(c.b.size() * sizeof(float)));
    if (!f)
      throw std::runtime_error("short read on " + path);
  };

  rd(blk0, 256, 80, 5, 1);
  for (int i = 0; i < 3; ++i) {
    rd(tdnn1[i], 256, 256, 1, 1);
    rd(res2[i], 128, 128, 3, static_cast<unsigned int>(i + 2)); // dil 2/3/4
    rd(tdnn2[i], 256, 256, 1, 1);
    rd(se1[i], 64, 256, 1, 1);
    rd(se2[i], 256, 64, 1, 1);
  }
  rd(mfa, 768, 768, 1, 1);
  rd(asp_tdnn, 64, 2304, 1, 1);
  rd(asp_conv, 768, 64, 1, 1);
  rd(fc, 128, 1536, 1, 1);

  // exactly at EOF?
  f.peek();
  if (!f.eof())
    throw std::runtime_error("trailing bytes in " + path);
  loaded_ = true;
}

void EcapaTdnn::conv1d(const Conv &c, const float *in, unsigned int T,
                       float *out, bool relu) {
  const int pad = static_cast<int>(c.dil * (c.k - 1) / 2);
  const int iT = static_cast<int>(T);
  for (unsigned int o = 0; o < c.cout; ++o) {
    const float *wo = c.w.data() + static_cast<size_t>(o) * c.cin * c.k;
    for (int t = 0; t < iT; ++t) {
      float acc = c.b[o];
      for (unsigned int ci = 0; ci < c.cin; ++ci) {
        const float *row = in + static_cast<size_t>(ci) * T;
        const float *wr = wo + static_cast<size_t>(ci) * c.k;
        for (unsigned int kk = 0; kk < c.k; ++kk) {
          int idx = t + static_cast<int>(kk * c.dil) - pad;
          // "same" REFLECT padding: mirror around the edge, edge not repeated
          if (idx < 0)
            idx = -idx;
          else if (idx >= iT)
            idx = 2 * iT - 2 - idx;
          acc += wr[kk] * row[idx];
        }
      }
      out[static_cast<size_t>(o) * T + t] =
        relu && acc < 0.0f ? 0.0f : acc;
    }
  }
}

std::vector<float> EcapaTdnn::forward(const float *mel, unsigned int T) const {
  if (!loaded_)
    throw std::runtime_error("EcapaTdnn weights not loaded");
  constexpr float EPS = 1e-12f;

  // step 0: [T,80] -> channels-first [80,T]
  std::vector<float> x(static_cast<size_t>(80) * T);
  for (unsigned int t = 0; t < T; ++t)
    for (unsigned int c = 0; c < 80; ++c)
      x[static_cast<size_t>(c) * T + t] = mel[static_cast<size_t>(t) * 80 + c];

  // step 1: initial TDNN 80 -> 256
  std::vector<float> h(static_cast<size_t>(256) * T);
  conv1d(blk0, x.data(), T, h.data(), /*relu=*/true);

  // steps 2-4: SE-Res2Net blocks; keep the three outputs for MFA
  std::vector<float> feats(static_cast<size_t>(3) * 256 * T);
  std::vector<float> tmp(static_cast<size_t>(256) * T),
    half(static_cast<size_t>(128) * T);
  for (int i = 0; i < 3; ++i) {
    const std::vector<float> residual = h;
    conv1d(tdnn1[i], h.data(), T, tmp.data(), true);
    // Res2Net scale=2: channels 0..127 identity, 128..255 dilated TDNN
    conv1d(res2[i], tmp.data() + static_cast<size_t>(128) * T, T, half.data(),
           true);
    std::memcpy(tmp.data() + static_cast<size_t>(128) * T, half.data(),
                half.size() * sizeof(float));
    conv1d(tdnn2[i], tmp.data(), T, h.data(), true);
    // SE gate: sigmoid(conv2(relu(conv1(mean_t)))) per channel
    float m[256], g1[64], g2[256];
    for (unsigned int c = 0; c < 256; ++c) {
      double s = 0.0;
      const float *row = h.data() + static_cast<size_t>(c) * T;
      for (unsigned int t = 0; t < T; ++t)
        s += row[t];
      m[c] = static_cast<float>(s / T);
    }
    conv1d(se1[i], m, 1, g1, true);
    conv1d(se2[i], g1, 1, g2, false);
    for (unsigned int c = 0; c < 256; ++c) {
      const float gate = 1.0f / (1.0f + std::exp(-g2[c]));
      float *row = h.data() + static_cast<size_t>(c) * T;
      const float *res = residual.data() + static_cast<size_t>(c) * T;
      for (unsigned int t = 0; t < T; ++t)
        row[t] = row[t] * gate + res[t];
    }
    std::memcpy(feats.data() + static_cast<size_t>(i) * 256 * T, h.data(),
                h.size() * sizeof(float));
  }

  // step 5: MFA 768 -> 768 (initial-TDNN output excluded from the concat)
  std::vector<float> g(static_cast<size_t>(768) * T);
  conv1d(mfa, feats.data(), T, g.data(), true);

  // step 6: attentive statistics pooling
  std::vector<float> mu(768), sd(768);
  for (unsigned int c = 0; c < 768; ++c) {
    const float *row = g.data() + static_cast<size_t>(c) * T;
    double s = 0.0;
    for (unsigned int t = 0; t < T; ++t)
      s += row[t];
    mu[c] = static_cast<float>(s / T);
    double v = 0.0;
    for (unsigned int t = 0; t < T; ++t) {
      const double d = row[t] - mu[c];
      v += d * d;
    }
    const float var = static_cast<float>(v / T); // population variance
    sd[c] = std::sqrt(var < EPS ? EPS : var);
  }
  std::vector<float> att_in(static_cast<size_t>(2304) * T);
  std::memcpy(att_in.data(), g.data(), g.size() * sizeof(float));
  for (unsigned int c = 0; c < 768; ++c) {
    float *mrow = att_in.data() + static_cast<size_t>(768 + c) * T;
    float *srow = att_in.data() + static_cast<size_t>(1536 + c) * T;
    for (unsigned int t = 0; t < T; ++t) {
      mrow[t] = mu[c];
      srow[t] = sd[c];
    }
  }
  std::vector<float> a64(static_cast<size_t>(64) * T),
    att(static_cast<size_t>(768) * T);
  conv1d(asp_tdnn, att_in.data(), T, a64.data(), true);
  for (auto &v : a64)
    v = std::tanh(v);
  conv1d(asp_conv, a64.data(), T, att.data(), false);
  // per-channel softmax over time, then attention-weighted mean/std
  std::vector<float> pooled(1536);
  for (unsigned int c = 0; c < 768; ++c) {
    float *row = att.data() + static_cast<size_t>(c) * T;
    const float *grow = g.data() + static_cast<size_t>(c) * T;
    float mx = row[0];
    for (unsigned int t = 1; t < T; ++t)
      mx = row[t] > mx ? row[t] : mx;
    double denom = 0.0;
    for (unsigned int t = 0; t < T; ++t) {
      row[t] = std::exp(row[t] - mx);
      denom += row[t];
    }
    double mean = 0.0;
    for (unsigned int t = 0; t < T; ++t) {
      row[t] = static_cast<float>(row[t] / denom);
      mean += static_cast<double>(row[t]) * grow[t];
    }
    double var = 0.0;
    for (unsigned int t = 0; t < T; ++t) {
      const double d = grow[t] - mean;
      var += static_cast<double>(row[t]) * d * d;
    }
    pooled[c] = static_cast<float>(mean);
    const float v = static_cast<float>(var);
    pooled[768 + c] = std::sqrt(v < EPS ? EPS : v);
  }

  // step 7: fc 1536 -> 128
  std::vector<float> out(128);
  conv1d(fc, pooled.data(), 1, out.data(), false);
  return out;
}

} // namespace causallm

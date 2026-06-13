// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   whisper_mel.cpp
 * @date   13 June 2026
 * @brief  Whisper-compatible log-mel feature extraction.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "whisper_mel.h"

namespace causallm {
namespace whisper_mel {

namespace {

constexpr unsigned int kNBins = kNFft / 2 + 1; // 201

/**
 * @brief transformers.audio_utils hertz_to_mel(..., mel_scale="slaney")
 */
double hertzToMelSlaney(double freq) {
  constexpr double min_log_hertz = 1000.0;
  constexpr double min_log_mel = 15.0;
  const double logstep = 27.0 / std::log(6.4);
  if (freq >= min_log_hertz)
    return min_log_mel + std::log(freq / min_log_hertz) * logstep;
  return 3.0 * freq / 200.0;
}

/**
 * @brief transformers.audio_utils mel_to_hertz(..., mel_scale="slaney")
 */
double melToHertzSlaney(double mel) {
  constexpr double min_log_mel = 15.0;
  const double logstep = std::log(6.4) / 27.0;
  if (mel >= min_log_mel)
    return 1000.0 * std::exp(logstep * (mel - min_log_mel));
  return 200.0 * mel / 3.0;
}

/**
 * @brief transformers.audio_utils mel_filter_bank(201, 128, 0, 8000, 16000,
 *        norm="slaney", mel_scale="slaney"), returned bin-major
 *        [kNBins][kNMels].
 */
std::vector<double> buildMelFilterBank() {
  const double mel_min = hertzToMelSlaney(0.0);
  const double mel_max = hertzToMelSlaney(kFMax);

  std::vector<double> filter_freqs(kNMels + 2);
  for (unsigned int i = 0; i < kNMels + 2; ++i) {
    const double mel = mel_min + (mel_max - mel_min) * i / (kNMels + 1);
    filter_freqs[i] = melToHertzSlaney(mel);
  }

  std::vector<double> fb(static_cast<size_t>(kNBins) * kNMels, 0.0);
  for (unsigned int b = 0; b < kNBins; ++b) {
    const double f = (kSampleRate / 2.0) * b / (kNBins - 1);
    for (unsigned int m = 0; m < kNMels; ++m) {
      const double down =
        (f - filter_freqs[m]) / (filter_freqs[m + 1] - filter_freqs[m]);
      const double up =
        (filter_freqs[m + 2] - f) / (filter_freqs[m + 2] - filter_freqs[m + 1]);
      double v = std::max(0.0, std::min(down, up));
      // slaney area normalization
      v *= 2.0 / (filter_freqs[m + 2] - filter_freqs[m]);
      fb[static_cast<size_t>(b) * kNMels + m] = v;
    }
  }
  return fb;
}

} // namespace

std::vector<float> loadWav16kMono(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Failed to open wav file: " + path);

  char riff[4], wave[4];
  uint32_t riff_size = 0;
  f.read(riff, 4);
  f.read(reinterpret_cast<char *>(&riff_size), 4);
  f.read(wave, 4);
  if (!f || std::memcmp(riff, "RIFF", 4) != 0 ||
      std::memcmp(wave, "WAVE", 4) != 0)
    throw std::runtime_error("Not a RIFF/WAVE file: " + path);

  uint16_t format = 0, channels = 0, bits = 0;
  uint32_t rate = 0;
  std::vector<int16_t> pcm;

  while (f) {
    char id[4];
    uint32_t size = 0;
    f.read(id, 4);
    f.read(reinterpret_cast<char *>(&size), 4);
    if (!f)
      break;
    if (std::memcmp(id, "fmt ", 4) == 0) {
      if (size < 16)
        throw std::runtime_error("Malformed fmt chunk in wav: " + path);
      std::vector<char> fmt(size + (size & 1));
      f.read(fmt.data(), fmt.size());
      std::memcpy(&format, fmt.data(), 2);
      std::memcpy(&channels, fmt.data() + 2, 2);
      std::memcpy(&rate, fmt.data() + 4, 4);
      std::memcpy(&bits, fmt.data() + 14, 2);
    } else if (std::memcmp(id, "data", 4) == 0) {
      pcm.resize(size / sizeof(int16_t));
      f.read(reinterpret_cast<char *>(pcm.data()),
             pcm.size() * sizeof(int16_t));
      if (!f)
        throw std::runtime_error("Truncated wav data chunk: " + path);
      break;
    } else {
      f.seekg(size + (size & 1), std::ios::cur); // chunks are word-aligned
    }
  }

  if (format != 1 || bits != 16)
    throw std::runtime_error("Expected 16-bit PCM wav: " + path);
  if (rate != kSampleRate)
    throw std::runtime_error("Expected 16 kHz wav, got " +
                             std::to_string(rate) + " Hz: " + path);
  if (channels == 0 || pcm.empty())
    throw std::runtime_error("Empty or malformed wav: " + path);

  const size_t n = pcm.size() / channels;
  std::vector<float> samples(n);
  for (size_t i = 0; i < n; ++i) {
    float acc = 0.0f;
    for (unsigned int c = 0; c < channels; ++c)
      acc += pcm[i * channels + c];
    samples[i] = acc / (channels * 32768.0f);
  }
  return samples;
}

std::vector<float> melSpectrogram(const std::vector<float> &samples_in,
                                  unsigned int &n_frames) {
  if (samples_in.size() < kHop)
    throw std::invalid_argument("audio too short for one mel frame");

  // HF truncates to the 300 s canvas before feature extraction
  constexpr size_t kMaxSamples = 300UL * kSampleRate;
  std::vector<float> truncated;
  const std::vector<float> &samples =
    samples_in.size() <= kMaxSamples
      ? samples_in
      : (truncated.assign(samples_in.begin(), samples_in.begin() + kMaxSamples),
         truncated);

  // valid frames on HF's zero-padded canvas, forced even
  unsigned int frames = (samples.size() + kHop - 1) / kHop;
  frames -= frames % 2;
  if (frames < 2)
    throw std::invalid_argument("audio too short (needs >= 20 ms)");
  n_frames = frames;

  // centered STFT: 200-sample reflect pad in front, zeros behind. Both pads
  // mirror/read HF's zero-extended 300 s canvas, so reflect positions past
  // the signal are zeros, not clamped samples.
  const unsigned int pad = kNFft / 2;
  std::vector<float> x(pad + samples.size() + kNFft, 0.0f);
  for (unsigned int i = 0; i < pad; ++i)
    if (pad - i < samples.size())
      x[i] = samples[pad - i];
  std::memcpy(x.data() + pad, samples.data(), samples.size() * sizeof(float));

  // periodic hann and DFT basis tables
  static const std::vector<double> fb = buildMelFilterBank();
  std::vector<float> window(kNFft);
  for (unsigned int i = 0; i < kNFft; ++i)
    window[i] = 0.5f - 0.5f * std::cos(2.0 * M_PI * i / kNFft);

  // magic-static init keeps the lazy twiddle tables thread-safe
  static const auto twiddles = [] {
    std::pair<std::vector<float>, std::vector<float>> t;
    t.first.resize(static_cast<size_t>(kNBins) * kNFft);
    t.second.resize(static_cast<size_t>(kNBins) * kNFft);
    for (unsigned int k = 0; k < kNBins; ++k)
      for (unsigned int i = 0; i < kNFft; ++i) {
        const double a = 2.0 * M_PI * k * i / kNFft;
        t.first[static_cast<size_t>(k) * kNFft + i] = std::cos(a);
        t.second[static_cast<size_t>(k) * kNFft + i] = std::sin(a);
      }
    return t;
  }();
  const std::vector<float> &cos_t = twiddles.first;
  const std::vector<float> &sin_t = twiddles.second;

  std::vector<float> mel(static_cast<size_t>(kNMels) * frames);
  std::vector<float> frame(kNFft);
  std::vector<double> power(kNBins);

  float log_max = -1e30f;
  for (unsigned int t = 0; t < frames; ++t) {
    const float *src = x.data() + static_cast<size_t>(t) * kHop;
    for (unsigned int i = 0; i < kNFft; ++i)
      frame[i] = src[i] * window[i];

    for (unsigned int k = 0; k < kNBins; ++k) {
      const float *ct = &cos_t[static_cast<size_t>(k) * kNFft];
      const float *st = &sin_t[static_cast<size_t>(k) * kNFft];
      double re = 0.0, im = 0.0;
      for (unsigned int i = 0; i < kNFft; ++i) {
        re += frame[i] * ct[i];
        im -= frame[i] * st[i];
      }
      power[k] = re * re + im * im;
    }

    for (unsigned int m = 0; m < kNMels; ++m) {
      double acc = 0.0;
      for (unsigned int k = 0; k < kNBins; ++k)
        acc += fb[static_cast<size_t>(k) * kNMels + m] * power[k];
      const float v =
        std::log10(std::max(1e-10, acc)); // mel_floor then log10
      mel[static_cast<size_t>(m) * frames + t] = v;
      log_max = std::max(log_max, v);
    }
  }

  // clamp to per-audio max - 8, then (x + 4) / 4
  const float floor_v = log_max - 8.0f;
  for (auto &v : mel)
    v = (std::max(v, floor_v) + 4.0f) / 4.0f;

  return mel;
}

} // namespace whisper_mel
} // namespace causallm

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   whisper_mel.h
 * @date   13 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Whisper-compatible log-mel feature extraction (C++ port of
 *         transformers' WhisperFeatureExtractor as configured for
 *         Qwen2.5-Omni: 16 kHz, n_fft 400, hop 160, 128 slaney mels
 *         0-8000 Hz, log10 floor 1e-10, clamp to max-8, (x+4)/4).
 *
 * Frame semantics match the HF pipeline exactly for the valid frames:
 * STFT is centered (200-sample reflect pad at the start; the tail reads the
 * zero padding HF's 300 s canvas provides), audio is truncated to 300 s,
 * valid frame count is ceil(n_samples / 160), and the per-audio max for the
 * -8 clamp is taken over the valid frames (padding frames sit at the -10
 * floor and can never win).
 *
 * The returned frame count is forced even (the audio encoder requires it to
 * keep the conv2 window off HF's post-conv1 zero mask; see the encoder
 * header). Consequence vs HF for ODD valid counts: the dropped 10 ms frame
 * is also excluded from the -8 clamp max, and about half of odd counts
 * yield one fewer audio token than HF's formula — sub-frame differences
 * that do not affect even-count audio at all.
 */

#ifndef __WHISPER_MEL_H__
#define __WHISPER_MEL_H__

#include <string>
#include <vector>

namespace causallm {
namespace whisper_mel {

constexpr unsigned int kSampleRate = 16000;
constexpr unsigned int kNFft = 400;
constexpr unsigned int kHop = 160;
constexpr unsigned int kNMels = 128;
constexpr float kFMax = 8000.0f;

/**
 * @brief Load a 16 kHz 16-bit PCM wav (mono or stereo, mixed down) as floats
 *        scaled to [-1, 1).
 */
std::vector<float> loadWav16kMono(const std::string &path);

/**
 * @brief Whisper log-mel features over the valid frames of @p samples.
 * @param[out] n_frames number of mel frames (forced even)
 * @return mel-bin-major buffer [kNMels][n_frames]
 */
std::vector<float> melSpectrogram(const std::vector<float> &samples,
                                  unsigned int &n_frames);

} // namespace whisper_mel
} // namespace causallm

#endif /* __WHISPER_MEL_H__ */

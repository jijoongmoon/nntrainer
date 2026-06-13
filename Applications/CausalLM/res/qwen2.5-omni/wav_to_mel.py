# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>

# @file wav_to_mel.py
# @brief Convert a 16 kHz mono wav into the mel feature file consumed by
#        Qwen25OmniAudioEncoder / Qwen25OmniAudioCausalLM
#        ([int32 n_mels][int32 n_frames][fp32 mel], n_frames even).
#
# @usage python wav_to_mel.py input.wav [-o input.mel]
#
# @author Jijoong Moon <jijoong.moon@samsung.com>

import argparse
import struct
import wave

import numpy as np


def load_wav_16k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, \
            f"expected 16 kHz wav, got {w.getframerate()}"
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return (data / 32768.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()
    out_path = args.output or args.wav.rsplit(".", 1)[0] + ".mel"

    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000,
                                 hop_length=160, n_fft=400, chunk_length=300,
                                 dither=0.0)
    wave_data = load_wav_16k_mono(args.wav)
    out = fe(wave_data, sampling_rate=16000, padding="max_length",
             return_attention_mask=True, return_tensors="np")
    valid = int(out["attention_mask"][0].sum())
    valid -= valid % 2  # even frame count (see audio encoder docs)
    mel = out["input_features"][0][:, :valid].astype(np.float32)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<ii", mel.shape[0], mel.shape[1]))
        np.ascontiguousarray(mel).tofile(f)
    print(f"{args.wav}: {len(wave_data)/16000:.2f}s -> {out_path} "
          f"(mel {mel.shape}, {valid // 8} audio tokens)")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>

# @file test_audio_encoder.py
# @brief Verify the nntrainer Qwen2.5-Omni audio encoder against the HF
#        reference implementation (transformers >= 4.52).
#
#        1. Builds a deterministic synthetic waveform, extracts the Whisper
#           mel features the Omni processor would produce, truncated to the
#           valid (even) frame count, and writes the mel file the
#           Qwen25OmniAudioEncoder binary consumes.
#        2. Runs HF Qwen2_5OmniAudioEncoder (audio tower only, fp32) on the
#           same mel exactly as Qwen2_5OmniThinker.get_audio_features does.
#        3. Runs `nntr_causallm <audio_model_dir> <mel_file>` and compares
#           the resulting embeddings (per-token cosine / max abs diff).
#
# @usage
#   python test_audio_encoder.py --model_path Qwen/Qwen2.5-Omni-3B \
#       --audio_model_dir ./qwen2.5-omni-3b-audio \
#       --binary <build>/Applications/CausalLM/nntr_causallm \
#       --seconds 3.52
#
# @author Jijoong Moon <jijoong.moon@samsung.com>

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np
import torch

from weight_converter import ShardedSafetensors, resolve_model_dir


def make_waveform(seconds: float, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(1234)
    t = np.arange(int(seconds * sr)) / sr
    chirp = np.sin(2 * np.pi * (200 + 800 * t) * t)
    return (0.6 * chirp + 0.1 * rng.standard_normal(t.size)).astype(np.float32)


def extract_mel(wave: np.ndarray) -> np.ndarray:
    """128-mel Whisper features over the valid frames only (T even)."""
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000,
                                 hop_length=160, n_fft=400, chunk_length=300,
                                 dither=0.0)
    out = fe(wave, sampling_rate=16000, padding="max_length",
             return_attention_mask=True, return_tensors="np")
    mel = out["input_features"][0]              # (128, 30000), zero padded
    valid = int(out["attention_mask"][0].sum()) # valid mel frames
    valid -= valid % 2                          # even-length policy
    return mel[:, :valid].astype(np.float32)


def hf_reference(model_dir: str, mel: np.ndarray) -> np.ndarray:
    """Run the fp32 HF audio tower exactly like thinker.get_audio_features."""
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoderConfig)
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoder)

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    audio_cfg = cfg["thinker_config"]["audio_config"]
    config = Qwen2_5OmniAudioEncoderConfig(**{
        k: v for k, v in audio_cfg.items()
        if k in Qwen2_5OmniAudioEncoderConfig().to_dict()})
    encoder = Qwen2_5OmniAudioEncoder(config).eval()

    weights = ShardedSafetensors(model_dir)
    prefix = "thinker.audio_tower."
    state = {k[len(prefix):]: weights.get(k).to(torch.float32)
             for k in weights.weight_map if k.startswith(prefix)}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert all("positional_embedding" in m for m in missing), \
        f"missing non-buffer keys: {missing}"

    feature_lens = torch.tensor([mel.shape[1]], dtype=torch.long)
    aftercnn_lens = (feature_lens - 1) // 2 + 1
    with torch.no_grad():
        out = encoder(torch.from_numpy(mel),  # (128, T) as get_audio_features
                      feature_lens=feature_lens,
                      aftercnn_lens=aftercnn_lens)
    return out.last_hidden_state.numpy()  # (n_tokens, output_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--audio_model_dir", type=str, required=True)
    parser.add_argument("--binary", type=str, required=True,
                        help="path to nntr_causallm")
    parser.add_argument("--seconds", type=float, default=3.52)
    parser.add_argument("--workdir", type=str, default="/tmp/omni_audio_test")
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    model_dir = resolve_model_dir(args.model_path)

    wave = make_waveform(args.seconds)
    mel = extract_mel(wave)
    print(f"waveform {args.seconds}s -> mel {mel.shape}")

    mel_path = os.path.join(args.workdir, "mel_input.bin")
    with open(mel_path, "wb") as f:
        f.write(struct.pack("<ii", mel.shape[0], mel.shape[1]))
        np.ascontiguousarray(mel, dtype=np.float32).tofile(f)

    ref = hf_reference(model_dir, mel)
    print(f"HF reference: {ref.shape}")

    env = dict(os.environ)
    res = subprocess.run([args.binary, args.audio_model_dir, mel_path],
                         capture_output=True, text=True, env=env)
    sys.stdout.write(res.stdout[-600:])
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-2000:])
        sys.exit(f"nntr_causallm failed (exit {res.returncode})")

    with open(mel_path + ".embd", "rb") as f:
        n_tokens, dim = struct.unpack("<ii", f.read(8))
        got = np.fromfile(f, dtype=np.float32).reshape(n_tokens, dim)

    assert got.shape == ref.shape, f"shape {got.shape} != ref {ref.shape}"
    diff = np.abs(got - ref)
    cos = np.sum(got * ref, axis=1) / (
        np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1) + 1e-12)
    print(f"tokens={n_tokens} max|diff|={diff.max():.5f} "
          f"mean|diff|={diff.mean():.6f} cos[min/mean]="
          f"{cos.min():.6f}/{cos.mean():.6f}")
    print(f"ref norm mean={np.linalg.norm(ref, axis=1).mean():.3f}")

    ok = cos.min() > 0.999
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

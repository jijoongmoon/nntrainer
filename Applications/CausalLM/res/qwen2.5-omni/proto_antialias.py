# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
# @file proto_antialias.py
# @brief Numpy prototype of HF TorchActivation1d (UpSample1d -> SnakeBeta ->
#        DownSample1d), validated against the BigVGAN activation_post dump
#        (stage5 -> activation_post). De-risks the antialiased_snake C++ layer.
#
#   python proto_antialias.py --model_path <local snapshot> --dump /tmp/omni_t2w_dump

import argparse
import math
import os

import numpy as np


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """Exact port of HF kaiser_sinc_filter1d (modeling.py:3094)."""
    half_size = kernel_size // 2
    is_even = kernel_size % 2 == 0
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    win = np.kaiser(kernel_size, beta).astype(np.float64)  # periodic=False
    if is_even:
        t = np.arange(-half_size, half_size) + 0.5
    else:
        t = np.arange(kernel_size) - half_size
    sinc = np.sinc(2 * cutoff * t)
    f = 2 * cutoff * win * sinc
    f = f / f.sum()
    return f.astype(np.float32)


def replicate_pad(x, left, right):
    # x: [C, T] -> [C, T+left+right] edge replication
    return np.pad(x, ((0, 0), (left, right)), mode="edge")


def up_sample(x, filt, ratio=2):
    """x [C,T] -> [C,2T]. HF UpSample1d (k=12)."""
    C, T = x.shape
    k = filt.shape[0]
    pad = k // ratio - 1                                # 5
    pad_left = pad * ratio + (k - ratio) // 2           # 15
    pad_right = pad * ratio + (k - ratio + 1) // 2      # 15
    xp = replicate_pad(x, pad, pad)                     # [C, T+2*pad]
    Tp = xp.shape[1]
    # transposed conv1d, stride=ratio, no internal padding -> length (Tp-1)*ratio + k
    Lout = (Tp - 1) * ratio + k
    y = np.zeros((C, Lout), dtype=np.float64)
    for i in range(Tp):
        base = i * ratio
        # scatter-add input[i] * filter across k taps
        y[:, base:base + k] += xp[:, i][:, None] * filt[None, :]
    y *= ratio
    y = y[:, pad_left:Lout - pad_right]
    return y.astype(np.float32)


def down_sample(x, filt, ratio=2):
    """x [C,2T] -> [C,T]. HF DownSample1d (k=12)."""
    C, T2 = x.shape
    k = filt.shape[0]
    even = (k % 2 == 0)
    pad_left = k // 2 - int(even)                       # 5
    pad_right = k // 2                                  # 6
    xp = replicate_pad(x, pad_left, pad_right)          # [C, 2T+11]
    Lp = xp.shape[1]
    Lout = (Lp - k) // ratio + 1
    y = np.zeros((C, Lout), dtype=np.float64)
    for o in range(Lout):
        seg = xp[:, o * ratio:o * ratio + k]            # [C,k]
        y[:, o] = (seg * filt[None, :]).sum(axis=1)
    return y.astype(np.float32)


def snake_beta(x, alpha_raw, beta_raw):
    a = np.exp(alpha_raw)[:, None]
    b = np.exp(beta_raw)[:, None]
    return x + (1.0 / (b + 1e-9)) * np.sin(x * a) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dump", default="/tmp/omni_t2w_dump")
    args = ap.parse_args()

    filt = kaiser_sinc_filter1d(0.5 / 2, 0.6 / 2, 12)
    print("filter taps:", np.round(filt, 8).tolist())
    print("filter sum:", float(filt.sum()))

    from safetensors import safe_open
    shard = os.path.join(args.model_path, "model-00003-of-00003.safetensors")
    pre = "token2wav.code2wav_bigvgan_model.activation_post.act."
    with safe_open(shard, framework="np") as f:
        alpha = f.get_tensor(pre + "alpha").astype(np.float32)  # [24]
        beta = f.get_tensor(pre + "beta").astype(np.float32)

    stage5 = np.load(os.path.join(args.dump, "stage5.npy"))[0]          # [24,30720]
    expect = np.load(os.path.join(args.dump, "activation_post.npy"))[0]  # [24,30720]
    print("stage5", stage5.shape, "expect", expect.shape)

    up = up_sample(stage5, filt)
    print("after up:", up.shape, "(expect 2x T)")
    act = snake_beta(up, alpha, beta)
    got = down_sample(act, filt)
    print("after down:", got.shape)

    d = np.abs(got - expect)
    print(f"antialias proto: max_abs={d.max():.3e}  rmse={np.sqrt((d**2).mean()):.3e}")
    print("first5 got: ", np.round(got[0, :5], 6).tolist())
    print("first5 exp: ", np.round(expect[0, :5], 6).tolist())


if __name__ == "__main__":
    main()

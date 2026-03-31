#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Verify RMS Norm backward pass using PyTorch autograd.

Compares:
1. PyTorch autograd gradients (ground truth)
2. NumPy analytical implementation (used in golden file)
3. C++ layer output (from golden test)

This provides independent verification that the calcDerivative
implementation is mathematically correct.
"""

import numpy as np
import struct
import os
import torch
import torch.nn as nn

SEED = 1234
EPSILON = 1e-7


class RMSNorm(nn.Module):
    """RMS Normalization using PyTorch (for autograd verification)."""

    def __init__(self, dim, eps=EPSILON):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        inv_rms = torch.rsqrt(variance)
        return x * inv_rms * self.gamma


def numpy_rms_norm_backward(dL_dy, x, gamma, epsilon):
    """NumPy analytical backward (same as gen_rms_norm_golden.py)."""
    variance = np.mean(x ** 2, axis=-1, keepdims=True) + epsilon
    inv_rms = 1.0 / np.sqrt(variance)

    dL_dgamma = np.sum(dL_dy * x * inv_rms, axis=(0, 1, 2))

    mean_dy_x = np.mean(dL_dy * x, axis=-1, keepdims=True)
    dL_dx = inv_rms * gamma * (dL_dy - x * mean_dy_x / variance)

    return dL_dx, dL_dgamma


def read_golden_file(filepath):
    """Read tensors from .nnlayergolden file."""
    tensors = []
    with open(filepath, "rb") as f:
        while True:
            size_bytes = f.read(4)
            if len(size_bytes) < 4:
                break
            size = struct.unpack("i", size_bytes)[0]
            data = np.fromfile(f, dtype=np.float32, count=size)
            if len(data) < size:
                break
            tensors.append(data)
    return tensors


def main():
    # Fix seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    input_shape = (2, 3, 3, 3)
    batch, channel, height, width = input_shape

    # Generate same random input as golden file generator
    x_np = np.random.randint(0, 10, input_shape).astype(np.float32)
    gamma_np = np.ones(width, dtype=np.float32)

    # ==========================================
    # 1. PyTorch autograd (ground truth)
    # ==========================================
    x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    model = RMSNorm(width, eps=EPSILON)
    # gamma is already ones by default

    output_torch = model(x_torch)

    # Incoming derivative = 2.0 (matching golden test framework)
    incoming_deriv = torch.full_like(output_torch, 2.0)

    # Compute gradients
    output_torch.backward(incoming_deriv)

    dx_torch = x_torch.grad.detach().numpy()
    dgamma_torch = model.gamma.grad.detach().numpy()
    output_torch_np = output_torch.detach().numpy()

    # ==========================================
    # 2. NumPy analytical implementation
    # ==========================================
    variance = np.mean(x_np ** 2, axis=-1, keepdims=True) + EPSILON
    inv_rms = 1.0 / np.sqrt(variance)
    output_numpy = x_np * inv_rms * gamma_np

    dL_dy = np.full_like(output_numpy, 2.0)
    dx_numpy, dgamma_numpy = numpy_rms_norm_backward(dL_dy, x_np, gamma_np, EPSILON)

    # ==========================================
    # 3. Read C++ golden test results
    # ==========================================
    golden_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "rms_norm_training.nnlayergolden",
    )
    tensors = read_golden_file(golden_path)
    # Order: initial_weights, inputs, outputs, gradients, weights, derivatives
    golden_output = tensors[2].reshape(input_shape)
    golden_dgamma = tensors[3].reshape(width)
    golden_dx = tensors[5].reshape(input_shape)

    # ==========================================
    # Compare results
    # ==========================================
    print("=" * 60)
    print("RMS Norm Backward Verification: PyTorch vs NumPy vs C++")
    print("=" * 60)

    def compare(name, a, b, label_a, label_b):
        max_abs = np.max(np.abs(a - b))
        max_rel = np.max(np.abs(a - b) / (np.abs(b) + 1e-10))
        cos_sim = np.dot(a.flatten(), b.flatten()) / (
            np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
        )
        status = "PASS" if max_abs < 1e-5 else "FAIL"
        print(f"\n  [{status}] {name} ({label_a} vs {label_b})")
        print(f"    Max absolute error: {max_abs:.2e}")
        print(f"    Max relative error: {max_rel:.2e}")
        print(f"    Cosine similarity:  {cos_sim:.10f}")
        return max_abs < 1e-5

    all_pass = True

    # Forward output comparisons
    print("\n--- Forward Output ---")
    all_pass &= compare("output", output_torch_np, output_numpy, "PyTorch", "NumPy")
    all_pass &= compare("output", output_torch_np, golden_output, "PyTorch", "Golden")

    # dL/dx comparisons
    print("\n--- Input Derivatives (dL/dx) ---")
    all_pass &= compare("dx", dx_torch, dx_numpy, "PyTorch", "NumPy")
    all_pass &= compare("dx", dx_torch, golden_dx, "PyTorch", "Golden")

    # dL/dgamma comparisons
    print("\n--- Gamma Gradients (dL/dgamma) ---")
    all_pass &= compare("dgamma", dgamma_torch, dgamma_numpy, "PyTorch", "NumPy")
    all_pass &= compare("dgamma", dgamma_torch, golden_dgamma, "PyTorch", "Golden")

    # Show sample values for manual inspection
    print("\n--- Sample Values (first 6 elements) ---")
    print(f"  dx PyTorch: {dx_torch.flatten()[:6]}")
    print(f"  dx NumPy:   {dx_numpy.flatten()[:6]}")
    print(f"  dx Golden:  {golden_dx.flatten()[:6]}")
    print(f"  dgamma PyTorch: {dgamma_torch}")
    print(f"  dgamma NumPy:   {dgamma_numpy}")
    print(f"  dgamma Golden:  {golden_dgamma}")

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED - PyTorch, NumPy, and Golden file agree!")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()

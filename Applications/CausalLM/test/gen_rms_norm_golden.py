#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   gen_rms_norm_golden.py
# @date   31 March 2026
# @brief  Generate golden test data for RMS normalization layer
# @author Generated for CausalLM RMS Norm backward pass testing
#
# Usage: python3 gen_rms_norm_golden.py
# Output: rms_norm_training.nnlayergolden
#
# Golden file format (record_single compatible):
#   For each tensor: [int32 size] [float32[] data]
#   Order: initial_weights, inputs, outputs, gradients, weights, derivatives

import numpy as np
import struct
import os

SEED = 1234
np.random.seed(SEED)

EPSILON = 1e-7


def write_tensor(f, tensor):
    """Write a tensor in nnlayergolden format: int32 size + float32 data."""
    data = tensor.astype(np.float32).flatten()
    f.write(struct.pack("i", len(data)))
    data.tofile(f)


def rms_norm_forward(x, gamma, epsilon):
    """RMS Normalization forward pass.

    Args:
        x: input tensor [batch, channel, height, width]
        gamma: scale parameter [width]
        epsilon: numerical stability constant

    Returns:
        output, variance, inv_rms
    """
    # variance = mean(x^2) + epsilon, averaged along last axis (width)
    variance = np.mean(x ** 2, axis=-1, keepdims=True) + epsilon
    # inv_rms = 1 / sqrt(variance)
    inv_rms = 1.0 / np.sqrt(variance)
    # output = x * inv_rms * gamma
    output = x * inv_rms * gamma
    return output, variance, inv_rms


def rms_norm_backward(dL_dy, x, gamma, variance, inv_rms):
    """RMS Normalization backward pass.

    Computes dL/dx and dL/dgamma given incoming derivative dL/dy.

    Math:
        y = gamma * x * inv_rms
        dL/dx = inv_rms * gamma * (dL/dy - x * mean(dL/dy * x) / variance)
        dL/dgamma = sum over (batch,channel,height) of (dL/dy * x * inv_rms)

    Args:
        dL_dy: incoming derivative [batch, channel, height, width]
        x: input from forward pass [batch, channel, height, width]
        gamma: scale parameter [width]
        variance: mean(x^2) + eps [batch, channel, height, 1]
        inv_rms: 1/sqrt(variance) [batch, channel, height, 1]

    Returns:
        dL_dx: input derivative [batch, channel, height, width]
        dL_dgamma: gamma gradient [width]
    """
    n = x.shape[-1]

    # dL/dgamma = sum of (dL/dy * x * inv_rms) along (batch, channel, height)
    dL_dgamma = np.sum(dL_dy * x * inv_rms, axis=(0, 1, 2))

    # mean(dL/dy * x) along width axis
    mean_dy_x = np.mean(dL_dy * x, axis=-1, keepdims=True)

    # dL/dx = inv_rms * gamma * (dL/dy - x * mean(dL/dy * x) / variance)
    dL_dx = inv_rms * gamma * (dL_dy - x * mean_dy_x / variance)

    return dL_dx, dL_dgamma


def generate_golden(filename, input_shape, epsilon=EPSILON):
    """Generate golden test data for RMS norm layer.

    Follows the same pattern as recorder.py record_single:
    - Random integer input scaled to [0, 10)
    - Incoming derivative is 2.0 (dy_constant = output * 2)
    - No optimizer update (weights unchanged)
    """
    batch, channel, height, width = input_shape

    # Generate random input (matches _rand_like with 'int')
    x = np.random.randint(0, 10, input_shape).astype(np.float32)

    # Initialize gamma to ones
    gamma = np.ones(width, dtype=np.float32)
    initial_gamma = gamma.copy()

    # Forward pass (warm up 4 times then 1 final, same result for deterministic)
    output, variance, inv_rms = rms_norm_forward(x, gamma, epsilon)

    # Incoming derivative = 2.0 (matching golden test framework)
    dL_dy = np.full_like(output, 2.0, dtype=np.float32)

    # Backward pass
    dL_dx, dL_dgamma = rms_norm_backward(dL_dy, x, gamma, variance, inv_rms)

    # Write golden file
    with open(filename, "wb") as f:
        # 1. Initial weights (gamma)
        write_tensor(f, initial_gamma)
        # 2. Inputs
        write_tensor(f, x)
        # 3. Outputs (forward result)
        write_tensor(f, output)
        # 4. Gradients (d_gamma) - only for trainable weights
        write_tensor(f, dL_dgamma)
        # 5. Weights after update (same as initial, no optimizer)
        write_tensor(f, gamma)
        # 6. Derivatives (d_input)
        write_tensor(f, dL_dx)

    print(f"Generated: {filename}")
    print(f"  Input shape: {input_shape}")
    print(f"  Input sample:\n    {x.flatten()[:12]}...")
    print(f"  Output sample:\n    {output.flatten()[:12]}...")
    print(f"  dL/dgamma sample:\n    {dL_dgamma[:12]}...")
    print(f"  dL/dx sample:\n    {dL_dx.flatten()[:12]}...")


def verify_with_numerical_gradient(input_shape, epsilon=EPSILON, delta=1e-4):
    """Verify analytical gradients against numerical gradients."""
    batch, channel, height, width = input_shape

    x = np.random.rand(*input_shape).astype(np.float64) + 0.1
    gamma = np.ones(width, dtype=np.float64) * 1.5

    output, variance, inv_rms = rms_norm_forward(x, gamma, epsilon)
    dL_dy = np.full_like(output, 2.0)
    dL_dx_analytical, dL_dgamma_analytical = rms_norm_backward(
        dL_dy, x, gamma, variance, inv_rms
    )

    # Numerical gradient for dx
    dL_dx_numerical = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += delta
        x_minus[idx] -= delta
        out_plus, _, _ = rms_norm_forward(x_plus, gamma, epsilon)
        out_minus, _, _ = rms_norm_forward(x_minus, gamma, epsilon)
        # loss = sum(2 * output), so dL/dx = 2 * d(sum(output))/dx
        dL_dx_numerical[idx] = 2.0 * np.sum(out_plus - out_minus) / (2 * delta)

    dx_max_err = np.max(np.abs(dL_dx_analytical - dL_dx_numerical))
    dx_rel_err = np.max(
        np.abs(dL_dx_analytical - dL_dx_numerical)
        / (np.abs(dL_dx_numerical) + 1e-8)
    )

    # Numerical gradient for dgamma
    dL_dgamma_numerical = np.zeros_like(gamma)
    for i in range(width):
        gamma_plus = gamma.copy()
        gamma_minus = gamma.copy()
        gamma_plus[i] += delta
        gamma_minus[i] -= delta
        out_plus, _, _ = rms_norm_forward(x, gamma_plus, epsilon)
        out_minus, _, _ = rms_norm_forward(x, gamma_minus, epsilon)
        dL_dgamma_numerical[i] = 2.0 * np.sum(out_plus - out_minus) / (2 * delta)

    dgamma_max_err = np.max(np.abs(dL_dgamma_analytical - dL_dgamma_numerical))
    dgamma_rel_err = np.max(
        np.abs(dL_dgamma_analytical - dL_dgamma_numerical)
        / (np.abs(dL_dgamma_numerical) + 1e-8)
    )

    print("\n=== Numerical Gradient Verification ===")
    print(f"  dx max absolute error: {dx_max_err:.2e}")
    print(f"  dx max relative error: {dx_rel_err:.2e}")
    print(f"  dgamma max absolute error: {dgamma_max_err:.2e}")
    print(f"  dgamma max relative error: {dgamma_rel_err:.2e}")

    assert dx_max_err < 1e-4, f"dx gradient check failed: {dx_max_err}"
    assert dgamma_max_err < 1e-4, f"dgamma gradient check failed: {dgamma_max_err}"
    print("  All gradient checks PASSED!")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate golden file for shape (2, 3, 3, 3)
    generate_golden(
        os.path.join(script_dir, "rms_norm_training.nnlayergolden"),
        (2, 3, 3, 3),
    )

    # Verify gradients numerically with a small tensor
    verify_with_numerical_gradient((2, 1, 2, 4))

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate golden test data for depthwise conv1d layer.

Depthwise conv1d: each input channel is convolved independently with its own
kernel. Equivalent to nn.Conv1d(C, C, K, groups=C) in PyTorch.

Weight layout (nntrainer): (channels, kernel_size) stored as 2D
Input layout  (nntrainer): (batch, channels, 1, width) - NCHW with H=1
Output layout (nntrainer): (batch, channels, 1, out_width)

Golden file format:
  For each tensor: [uint32 num_elements] [float32 data...]
  Order: initial_weights, inputs, outputs, gradients, weights, derivatives
"""

import numpy as np
import struct
import os

SEED = 1234
np.random.seed(SEED)


def write_tensor(f, data):
    """Write a tensor to file: uint32 size followed by float32 data."""
    flat = data.flatten().astype(np.float32)
    f.write(struct.pack('I', len(flat)))
    flat.tofile(f)


def depthwise_conv1d_forward(input_data, weight, bias, stride, padding,
                             dilation):
    """
    Compute depthwise conv1d forward pass.

    Args:
        input_data: (batch, channels, width)
        weight: (channels, kernel_size)
        bias: (channels,) or None
        stride: int
        padding: (pad_left, pad_right)
        dilation: int

    Returns:
        output: (batch, channels, out_width)
    """
    batch, channels, in_width = input_data.shape
    kernel_size = weight.shape[1]
    pad_left, pad_right = padding

    eff_k = (kernel_size - 1) * dilation + 1
    padded_width = in_width + pad_left + pad_right
    out_width = (padded_width - eff_k) // stride + 1

    output = np.zeros((batch, channels, out_width), dtype=np.float32)

    for b in range(batch):
        for c in range(channels):
            for ow in range(out_width):
                val = 0.0
                base_w = ow * stride - pad_left
                for k in range(kernel_size):
                    iw = base_w + k * dilation
                    if 0 <= iw < in_width:
                        val += input_data[b, c, iw] * weight[c, k]
                output[b, c, ow] = val

    if bias is not None:
        output += bias[np.newaxis, :, np.newaxis]

    return output


def depthwise_conv1d_calc_derivative(incoming_deriv, weight, stride, padding,
                                     dilation, in_width):
    """Compute input derivative (backprop through depthwise conv1d)."""
    batch, channels, out_width = incoming_deriv.shape
    kernel_size = weight.shape[1]
    pad_left, _ = padding

    input_deriv = np.zeros((batch, channels, in_width), dtype=np.float32)

    for b in range(batch):
        for c in range(channels):
            for ow in range(out_width):
                grad_out = incoming_deriv[b, c, ow]
                base_w = ow * stride - pad_left
                for k in range(kernel_size):
                    iw = base_w + k * dilation
                    if 0 <= iw < in_width:
                        input_deriv[b, c, iw] += grad_out * weight[c, k]

    return input_deriv


def depthwise_conv1d_calc_gradient(incoming_deriv, input_data, channels,
                                   kernel_size, stride, padding, dilation):
    """Compute weight and bias gradients."""
    batch, _, out_width = incoming_deriv.shape
    in_width = input_data.shape[2]
    pad_left, _ = padding

    weight_grad = np.zeros((channels, kernel_size), dtype=np.float32)
    bias_grad = np.zeros(channels, dtype=np.float32)

    for b in range(batch):
        for c in range(channels):
            for ow in range(out_width):
                grad_out = incoming_deriv[b, c, ow]
                base_w = ow * stride - pad_left
                for k in range(kernel_size):
                    iw = base_w + k * dilation
                    if 0 <= iw < in_width:
                        weight_grad[c, k] += grad_out * input_data[b, c, iw]

    # bias gradient: sum over batch and width
    bias_grad = incoming_deriv.sum(axis=(0, 2))

    return weight_grad, bias_grad


def compute_padding_same(in_width, kernel_size, stride, dilation):
    """Compute 'same' padding (output_width = ceil(in_width / stride))."""
    eff_k = (kernel_size - 1) * dilation + 1
    out_width = (in_width + stride - 1) // stride
    total_pad = max(0, (out_width - 1) * stride + eff_k - in_width)
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    return (pad_left, pad_right)


def compute_padding_causal(kernel_size, dilation):
    """Compute causal padding (only left padding)."""
    eff_k = (kernel_size - 1) * dilation + 1
    return (eff_k - 1, 0)


def gen_rand_input(shape):
    """Generate random input similar to TF's approach."""
    return np.random.uniform(-1, 1, shape).astype(np.float32)


def generate_golden(name, batch, channels, width, kernel_size, stride=1,
                    padding_mode="valid", dilation=1, disable_bias=False,
                    output_dir="."):
    """Generate a single golden test file."""

    # Compute padding
    if padding_mode == "valid":
        padding = (0, 0)
    elif padding_mode == "same":
        padding = compute_padding_same(width, kernel_size, stride, dilation)
    elif padding_mode == "causal":
        padding = compute_padding_causal(kernel_size, dilation)
    elif isinstance(padding_mode, tuple):
        padding = padding_mode
    else:
        raise ValueError(f"Unknown padding mode: {padding_mode}")

    # Generate random input: nntrainer uses (batch, channels, 1, width)
    # but we compute as (batch, channels, width) for simplicity
    input_3d = gen_rand_input((batch, channels, width))

    # Generate random weights: (channels, kernel_size)
    weight = gen_rand_input((channels, kernel_size))

    # Generate random bias: (channels,)
    bias = gen_rand_input((channels,)) if not disable_bias else None

    # Store initial weights
    initial_weight = weight.copy()
    initial_bias = bias.copy() if bias is not None else None

    # Forward pass
    output_3d = depthwise_conv1d_forward(input_3d, weight, bias, stride,
                                         padding, dilation)

    # Incoming derivative is fixed to 2.0 (matching C++ test framework convention)
    incoming_deriv = np.full_like(output_3d, 2.0)

    # Compute weight gradient
    weight_grad, bias_grad = depthwise_conv1d_calc_gradient(
        incoming_deriv, input_3d, channels, kernel_size, stride, padding,
        dilation)

    # Compute input derivative
    input_deriv = depthwise_conv1d_calc_derivative(
        incoming_deriv, weight, stride, padding, dilation, width)

    # Reshape to nntrainer 4D format (batch, channels, 1, width)
    input_4d = input_3d.reshape(batch, channels, 1, width)
    out_width = output_3d.shape[2]
    output_4d = output_3d.reshape(batch, channels, 1, out_width)
    input_deriv_4d = input_deriv.reshape(batch, channels, 1, width)

    # Bias stored as (1, channels, 1, 1) in nntrainer (matching conv2d pattern)
    bias_4d = bias.reshape(1, channels, 1, 1) if bias is not None else None
    initial_bias_4d = initial_bias.reshape(1, channels, 1, 1) if initial_bias is not None else None
    bias_grad_4d = bias_grad.reshape(1, channels, 1, 1) if bias is not None else None

    filepath = os.path.join(output_dir, name + ".nnlayergolden")
    with open(filepath, "wb") as f:
        # 1. initial_weights (weight, then bias if present)
        write_tensor(f, initial_weight)
        if initial_bias_4d is not None:
            write_tensor(f, initial_bias_4d)

        # 2. inputs
        write_tensor(f, input_4d)

        # 3. outputs
        write_tensor(f, output_4d)

        # 4. gradients (weight grad, then bias grad if present)
        write_tensor(f, weight_grad)
        if bias_grad_4d is not None:
            write_tensor(f, bias_grad_4d)

        # 5. weights (same as initial for single forward)
        write_tensor(f, weight)
        if bias_4d is not None:
            write_tensor(f, bias_4d)

        # 6. derivatives (input derivatives)
        write_tensor(f, input_deriv_4d)

    print(f"Generated: {filepath}")
    print(f"  input: ({batch}, {channels}, 1, {width}), "
          f"weight: ({channels}, {kernel_size}), "
          f"output: ({batch}, {channels}, 1, {out_width}), "
          f"padding: {padding}, stride: {stride}, dilation: {dilation}")


def main():
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(output_dir, exist_ok=True)

    # Basic tests - single batch
    generate_golden("depthwise_conv1d_sb_minimum", batch=1, channels=3,
                    width=8, kernel_size=3, output_dir=output_dir)

    # Multi-batch
    generate_golden("depthwise_conv1d_mb_minimum", batch=3, channels=3,
                    width=8, kernel_size=3, output_dir=output_dir)

    # Same padding
    generate_golden("depthwise_conv1d_sb_same", batch=1, channels=4,
                    width=8, kernel_size=3, padding_mode="same",
                    output_dir=output_dir)

    generate_golden("depthwise_conv1d_mb_same", batch=3, channels=4,
                    width=8, kernel_size=3, padding_mode="same",
                    output_dir=output_dir)

    # Stride
    generate_golden("depthwise_conv1d_sb_stride", batch=1, channels=3,
                    width=8, kernel_size=3, stride=2, output_dir=output_dir)

    generate_golden("depthwise_conv1d_mb_stride", batch=3, channels=3,
                    width=8, kernel_size=3, stride=2, output_dir=output_dir)

    # Dilation
    generate_golden("depthwise_conv1d_sb_dilation", batch=1, channels=3,
                    width=11, kernel_size=3, dilation=2, output_dir=output_dir)

    generate_golden("depthwise_conv1d_mb_dilation", batch=3, channels=3,
                    width=11, kernel_size=3, dilation=2, output_dir=output_dir)

    # Causal padding (important for SSM/Mamba)
    generate_golden("depthwise_conv1d_sb_causal", batch=1, channels=4,
                    width=8, kernel_size=4, padding_mode="causal",
                    output_dir=output_dir)

    generate_golden("depthwise_conv1d_mb_causal", batch=3, channels=4,
                    width=8, kernel_size=4, padding_mode="causal",
                    output_dir=output_dir)

    # No bias
    generate_golden("depthwise_conv1d_sb_no_bias", batch=1, channels=3,
                    width=8, kernel_size=3, disable_bias=True,
                    output_dir=output_dir)

    generate_golden("depthwise_conv1d_mb_no_bias", batch=3, channels=3,
                    width=8, kernel_size=3, disable_bias=True,
                    output_dir=output_dir)


if __name__ == "__main__":
    main()

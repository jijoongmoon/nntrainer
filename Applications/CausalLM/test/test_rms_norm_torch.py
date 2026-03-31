#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for RMS Norm backward pass verification using PyTorch autograd.

Compares PyTorch autograd gradients (ground truth) against:
- NumPy analytical implementation (used in golden file generation)
- C++ layer output (from .nnlayergolden golden file)

Usage:
    python3 -m pytest test_rms_norm_torch.py -v
    python3 test_rms_norm_torch.py
"""

import os
import struct
import unittest

import numpy as np
import torch
import torch.nn as nn

SEED = 1234
EPSILON = 1e-7
ABS_TOL = 1e-5


class TorchRMSNorm(nn.Module):
    """RMS Normalization in PyTorch for autograd ground truth."""

    def __init__(self, dim, eps=EPSILON):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        return x * torch.rsqrt(variance) * self.gamma


def numpy_rms_norm_forward(x, gamma, eps):
    """NumPy RMS Norm forward pass."""
    variance = np.mean(x ** 2, axis=-1, keepdims=True) + eps
    inv_rms = 1.0 / np.sqrt(variance)
    return x * inv_rms * gamma, variance, inv_rms


def numpy_rms_norm_backward(dL_dy, x, gamma, variance, inv_rms):
    """NumPy RMS Norm backward pass (analytical)."""
    dL_dgamma = np.sum(dL_dy * x * inv_rms, axis=(0, 1, 2))
    mean_dy_x = np.mean(dL_dy * x, axis=-1, keepdims=True)
    dL_dx = inv_rms * gamma * (dL_dy - x * mean_dy_x / variance)
    return dL_dx, dL_dgamma


def read_golden_file(filepath):
    """Read tensors from .nnlayergolden binary file."""
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


class TestRMSNormBackward(unittest.TestCase):
    """Test RMS Norm backward pass against PyTorch autograd."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test data."""
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        cls.input_shape = (2, 3, 3, 3)
        cls.width = cls.input_shape[-1]

        # Generate random input (matches golden file generator)
        cls.x_np = np.random.randint(0, 10, cls.input_shape).astype(np.float32)
        cls.gamma_np = np.ones(cls.width, dtype=np.float32)

        # --- PyTorch autograd (ground truth) ---
        x_torch = torch.tensor(cls.x_np, dtype=torch.float32, requires_grad=True)
        model = TorchRMSNorm(cls.width, eps=EPSILON)
        output = model(x_torch)
        output.backward(torch.full_like(output, 2.0))

        cls.torch_output = output.detach().numpy()
        cls.torch_dx = x_torch.grad.detach().numpy()
        cls.torch_dgamma = model.gamma.grad.detach().numpy()

        # --- NumPy analytical ---
        cls.np_output, variance, inv_rms = numpy_rms_norm_forward(
            cls.x_np, cls.gamma_np, EPSILON
        )
        dL_dy = np.full_like(cls.np_output, 2.0)
        cls.np_dx, cls.np_dgamma = numpy_rms_norm_backward(
            dL_dy, cls.x_np, cls.gamma_np, variance, inv_rms
        )

        # --- Golden file (C++ layer output) ---
        golden_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "rms_norm_training.nnlayergolden",
        )
        if os.path.exists(golden_path):
            tensors = read_golden_file(golden_path)
            cls.golden_output = tensors[2].reshape(cls.input_shape)
            cls.golden_dgamma = tensors[3].reshape(cls.width)
            cls.golden_dx = tensors[5].reshape(cls.input_shape)
            cls.has_golden = True
        else:
            cls.has_golden = False

    # ---- Forward output tests ----

    def test_forward_torch_vs_numpy(self):
        np.testing.assert_allclose(
            self.torch_output, self.np_output, atol=ABS_TOL,
            err_msg="Forward output: PyTorch vs NumPy mismatch",
        )

    def test_forward_torch_vs_golden(self):
        if not self.has_golden:
            self.skipTest("Golden file not found")
        np.testing.assert_allclose(
            self.torch_output, self.golden_output, atol=ABS_TOL,
            err_msg="Forward output: PyTorch vs Golden mismatch",
        )

    # ---- Input derivative (dL/dx) tests ----

    def test_dx_torch_vs_numpy(self):
        np.testing.assert_allclose(
            self.torch_dx, self.np_dx, atol=ABS_TOL,
            err_msg="dL/dx: PyTorch vs NumPy mismatch",
        )

    def test_dx_torch_vs_golden(self):
        if not self.has_golden:
            self.skipTest("Golden file not found")
        np.testing.assert_allclose(
            self.torch_dx, self.golden_dx, atol=ABS_TOL,
            err_msg="dL/dx: PyTorch vs Golden mismatch",
        )

    # ---- Gamma gradient (dL/dgamma) tests ----

    def test_dgamma_torch_vs_numpy(self):
        np.testing.assert_allclose(
            self.torch_dgamma, self.np_dgamma, atol=ABS_TOL,
            err_msg="dL/dgamma: PyTorch vs NumPy mismatch",
        )

    def test_dgamma_torch_vs_golden(self):
        if not self.has_golden:
            self.skipTest("Golden file not found")
        np.testing.assert_allclose(
            self.torch_dgamma, self.golden_dgamma, atol=ABS_TOL,
            err_msg="dL/dgamma: PyTorch vs Golden mismatch",
        )

    # ---- Cosine similarity tests ----

    def _cosine_sim(self, a, b):
        a_flat, b_flat = a.flatten(), b.flatten()
        return np.dot(a_flat, b_flat) / (
            np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-10
        )

    def test_dx_cosine_similarity(self):
        cos_sim = self._cosine_sim(self.torch_dx, self.np_dx)
        self.assertAlmostEqual(cos_sim, 1.0, places=5,
                               msg=f"dL/dx cosine similarity too low: {cos_sim}")

    def test_dgamma_cosine_similarity(self):
        cos_sim = self._cosine_sim(self.torch_dgamma, self.np_dgamma)
        self.assertAlmostEqual(cos_sim, 1.0, places=5,
                               msg=f"dL/dgamma cosine similarity too low: {cos_sim}")

    # ---- Numerical gradient check ----

    def test_numerical_gradient_dx(self):
        """Verify dL/dx with finite differences (small tensor)."""
        np.random.seed(42)
        shape = (1, 1, 2, 4)
        x = np.random.rand(*shape).astype(np.float64) + 0.1
        gamma = np.ones(shape[-1], dtype=np.float64)
        delta = 1e-5

        _, var, irms = numpy_rms_norm_forward(x, gamma, EPSILON)
        dL_dy = np.full(shape, 2.0, dtype=np.float64)
        dx_analytical, _ = numpy_rms_norm_backward(dL_dy, x, gamma, var, irms)

        dx_numerical = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            xp, xm = x.copy(), x.copy()
            xp[idx] += delta
            xm[idx] -= delta
            op, _, _ = numpy_rms_norm_forward(xp, gamma, EPSILON)
            om, _, _ = numpy_rms_norm_forward(xm, gamma, EPSILON)
            dx_numerical[idx] = 2.0 * np.sum(op - om) / (2 * delta)

        np.testing.assert_allclose(
            dx_analytical, dx_numerical, atol=1e-4,
            err_msg="Numerical gradient check for dL/dx failed",
        )

    def test_numerical_gradient_dgamma(self):
        """Verify dL/dgamma with finite differences (small tensor)."""
        np.random.seed(42)
        shape = (1, 1, 2, 4)
        x = np.random.rand(*shape).astype(np.float64) + 0.1
        gamma = np.ones(shape[-1], dtype=np.float64) * 1.5
        delta = 1e-5

        _, var, irms = numpy_rms_norm_forward(x, gamma, EPSILON)
        dL_dy = np.full(shape, 2.0, dtype=np.float64)
        _, dgamma_analytical = numpy_rms_norm_backward(dL_dy, x, gamma, var, irms)

        dgamma_numerical = np.zeros_like(gamma)
        for i in range(len(gamma)):
            gp, gm = gamma.copy(), gamma.copy()
            gp[i] += delta
            gm[i] -= delta
            op, _, _ = numpy_rms_norm_forward(x, gp, EPSILON)
            om, _, _ = numpy_rms_norm_forward(x, gm, EPSILON)
            dgamma_numerical[i] = 2.0 * np.sum(op - om) / (2 * delta)

        np.testing.assert_allclose(
            dgamma_analytical, dgamma_numerical, atol=1e-4,
            err_msg="Numerical gradient check for dL/dgamma failed",
        )


if __name__ == "__main__":
    unittest.main()

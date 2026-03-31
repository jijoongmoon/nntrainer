#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for CausalLM layer backward passes using PyTorch autograd.

Tests: QKV Layer, SwiGLU, LM Head, Embedding, Reshaped RMS Norm

Usage:
    python3 test_causallm_layers_torch.py -v
"""

import math
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ABS_TOL = 1e-4


class TestQKVLayerBackward(unittest.TestCase):
    """Test QKV Layer: Q=X@Wq, K=X@Wk, V=X@Wv."""

    def test_numerical_gradient_input(self):
        """Verify dL/dInput via finite differences."""
        torch.manual_seed(42)
        B, S, D_in, D_q, D_k, D_v = 1, 3, 8, 6, 6, 6
        delta = 1e-4

        x = torch.randn(B, 1, S, D_in, dtype=torch.float64, requires_grad=True)
        Wq = torch.randn(D_in, D_q, dtype=torch.float64)
        Wk = torch.randn(D_in, D_k, dtype=torch.float64)
        Wv = torch.randn(D_in, D_v, dtype=torch.float64)

        q = x @ Wq
        k = x @ Wk
        v = x @ Wv
        loss = 2.0 * (q.sum() + k.sum() + v.sum())
        loss.backward()
        dx_analytical = x.grad.clone()

        dx_numerical = torch.zeros_like(x)
        x_flat = x.detach().flatten()
        for i in range(x_flat.numel()):
            xp, xm = x_flat.clone(), x_flat.clone()
            xp[i] += delta
            xm[i] -= delta
            xp_t, xm_t = xp.view_as(x), xm.view_as(x)
            lp = 2.0 * ((xp_t @ Wq).sum() + (xp_t @ Wk).sum() + (xp_t @ Wv).sum())
            lm = 2.0 * ((xm_t @ Wq).sum() + (xm_t @ Wk).sum() + (xm_t @ Wv).sum())
            dx_numerical.flatten()[i] = (lp - lm) / (2 * delta)

        max_err = (dx_analytical - dx_numerical).abs().max().item()
        self.assertLess(max_err, ABS_TOL, f"QKV dX error: {max_err:.2e}")

    def test_weight_gradient(self):
        """Verify dL/dW via finite differences."""
        torch.manual_seed(42)
        B, S, D_in, D_out = 1, 3, 8, 6
        delta = 1e-4

        x = torch.randn(B, 1, S, D_in, dtype=torch.float64)
        W = torch.randn(D_in, D_out, dtype=torch.float64, requires_grad=True)

        out = x @ W
        loss = 2.0 * out.sum()
        loss.backward()
        dW_analytical = W.grad.clone()

        dW_numerical = torch.zeros_like(W)
        W_flat = W.detach().flatten()
        for i in range(W_flat.numel()):
            wp, wm = W_flat.clone(), W_flat.clone()
            wp[i] += delta
            wm[i] -= delta
            lp = 2.0 * (x @ wp.view_as(W)).sum()
            lm = 2.0 * (x @ wm.view_as(W)).sum()
            dW_numerical.flatten()[i] = (lp - lm) / (2 * delta)

        max_err = (dW_analytical - dW_numerical).abs().max().item()
        self.assertLess(max_err, ABS_TOL, f"QKV dW error: {max_err:.2e}")


class TestSwiGLUBackward(unittest.TestCase):
    """Test SwiGLU: out = in1 * silu(in2), silu(x) = x * sigmoid(x)."""

    def test_numerical_gradient_in1(self):
        """Verify dL/din1 via finite differences."""
        torch.manual_seed(42)
        shape = (1, 1, 3, 8)
        delta = 1e-5

        in1 = torch.randn(shape, dtype=torch.float64, requires_grad=True)
        in2 = torch.randn(shape, dtype=torch.float64)

        out = in1 * F.silu(in2)
        out.backward(torch.full_like(out, 2.0))
        d_analytical = in1.grad.clone()

        d_numerical = torch.zeros_like(in1)
        flat = in1.detach().flatten()
        for i in range(flat.numel()):
            fp, fm = flat.clone(), flat.clone()
            fp[i] += delta
            fm[i] -= delta
            op = fp.view_as(in1) * F.silu(in2)
            om = fm.view_as(in1) * F.silu(in2)
            d_numerical.flatten()[i] = 2.0 * (op.sum() - om.sum()) / (2 * delta)

        max_err = (d_analytical - d_numerical).abs().max().item()
        self.assertLess(max_err, ABS_TOL, f"SwiGLU din1 error: {max_err:.2e}")

    def test_numerical_gradient_in2(self):
        """Verify dL/din2 via finite differences."""
        torch.manual_seed(42)
        shape = (1, 1, 3, 8)
        delta = 1e-5

        in1 = torch.randn(shape, dtype=torch.float64)
        in2 = torch.randn(shape, dtype=torch.float64, requires_grad=True)

        out = in1 * F.silu(in2)
        out.backward(torch.full_like(out, 2.0))
        d_analytical = in2.grad.clone()

        d_numerical = torch.zeros_like(in2)
        flat = in2.detach().flatten()
        for i in range(flat.numel()):
            fp, fm = flat.clone(), flat.clone()
            fp[i] += delta
            fm[i] -= delta
            op = in1 * F.silu(fp.view_as(in2))
            om = in1 * F.silu(fm.view_as(in2))
            d_numerical.flatten()[i] = 2.0 * (op.sum() - om.sum()) / (2 * delta)

        max_err = (d_analytical - d_numerical).abs().max().item()
        self.assertLess(max_err, ABS_TOL, f"SwiGLU din2 error: {max_err:.2e}")


class TestLMHeadBackward(unittest.TestCase):
    """Test LM Head: output = input @ weight + bias."""

    def test_numerical_gradient_input(self):
        """Verify dL/dInput."""
        torch.manual_seed(42)
        B, S, D_in, D_out = 1, 4, 16, 32
        delta = 1e-4

        x = torch.randn(B, 1, S, D_in, dtype=torch.float64, requires_grad=True)
        W = torch.randn(D_in, D_out, dtype=torch.float64)
        b = torch.randn(D_out, dtype=torch.float64)

        out = x @ W + b
        out.backward(torch.full_like(out, 2.0))
        dx_a = x.grad.clone()

        dx_n = torch.zeros_like(x)
        xf = x.detach().flatten()
        for i in range(xf.numel()):
            xp, xm = xf.clone(), xf.clone()
            xp[i] += delta
            xm[i] -= delta
            op = xp.view_as(x) @ W + b
            om = xm.view_as(x) @ W + b
            dx_n.flatten()[i] = 2.0 * (op.sum() - om.sum()) / (2 * delta)

        max_err = (dx_a - dx_n).abs().max().item()
        self.assertLess(max_err, ABS_TOL, f"LMHead dX error: {max_err:.2e}")

    def test_weight_and_bias_gradient(self):
        """Verify dL/dW and dL/db."""
        torch.manual_seed(42)
        B, S, D_in, D_out = 2, 3, 8, 4

        x = torch.randn(B, 1, S, D_in, dtype=torch.float64)
        W = torch.randn(D_in, D_out, dtype=torch.float64, requires_grad=True)
        b = torch.randn(D_out, dtype=torch.float64, requires_grad=True)

        out = x @ W + b
        out.backward(torch.full_like(out, 2.0))

        # dL/dW = X^T @ dL/dOut, where dL/dOut = 2
        expected_dW = 2.0 * x.reshape(-1, D_in).T @ torch.ones(B * S, D_out, dtype=torch.float64)
        expected_db = 2.0 * torch.ones(D_out, dtype=torch.float64) * B * S

        max_err_w = (W.grad - expected_dW).abs().max().item()
        max_err_b = (b.grad - expected_db).abs().max().item()
        self.assertLess(max_err_w, ABS_TOL, f"LMHead dW error: {max_err_w:.2e}")
        self.assertLess(max_err_b, ABS_TOL, f"LMHead db error: {max_err_b:.2e}")


class TestEmbeddingBackward(unittest.TestCase):
    """Test Embedding: output[i] = weight[idx[i]] * scale."""

    def test_weight_gradient(self):
        """Verify scatter-add weight gradient."""
        torch.manual_seed(42)
        vocab_size, embed_dim = 10, 8
        B, S = 2, 4
        scale = 1.0

        emb = nn.Embedding(vocab_size, embed_dim, scale_grad_by_freq=False)
        indices = torch.randint(0, vocab_size, (B, S))

        out = emb(indices) * scale
        out.backward(torch.full_like(out, 2.0))

        # For each index, gradient should be 2.0 * scale * count
        dW = emb.weight.grad
        self.assertIsNotNone(dW)

        # Manual check: count occurrences and verify
        for idx in range(vocab_size):
            count = (indices == idx).sum().item()
            expected_grad = 2.0 * scale * count
            actual_grad_sum = dW[idx].sum().item()
            expected_sum = expected_grad * embed_dim
            self.assertAlmostEqual(actual_grad_sum, expected_sum, places=3,
                                   msg=f"Embedding dW[{idx}] error")

    def test_no_input_gradient(self):
        """Embedding input is discrete indices - no gradient."""
        emb = nn.Embedding(10, 8)
        indices = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = emb(indices)
        out.backward(torch.full_like(out, 2.0))
        # indices don't have gradients (not float tensor)
        self.assertIsNone(indices.grad)


class TestReshapedRMSNormBackward(unittest.TestCase):
    """Test Reshaped RMS Norm: reshape by feature_size, then RMS normalize."""

    def _rms_norm(self, x, gamma, eps=1e-7):
        """RMS norm along last dimension."""
        variance = x.pow(2).mean(dim=-1, keepdim=True) + eps
        return x * torch.rsqrt(variance) * gamma

    def test_numerical_gradient_input(self):
        """Verify dL/dx for reshaped RMS norm."""
        torch.manual_seed(42)
        B, S, W = 1, 3, 12
        feature_size = 4  # W must be divisible by feature_size
        eps = 1e-7
        delta = 1e-5

        x = torch.randn(B, 1, S, W, dtype=torch.float64, requires_grad=True)
        gamma = torch.ones(feature_size, dtype=torch.float64)

        # Reshape, normalize, reshape back
        x_reshaped = x.view(B, 1, S * (W // feature_size), feature_size)
        out_reshaped = self._rms_norm(x_reshaped, gamma, eps)
        out = out_reshaped.view(B, 1, S, W)
        out.backward(torch.full_like(out, 2.0))
        dx_a = x.grad.clone()

        dx_n = torch.zeros_like(x)
        xf = x.detach().flatten()
        for i in range(xf.numel()):
            xp, xm = xf.clone(), xf.clone()
            xp[i] += delta
            xm[i] -= delta
            xpr = xp.view(B, 1, S * (W // feature_size), feature_size)
            xmr = xm.view(B, 1, S * (W // feature_size), feature_size)
            op = self._rms_norm(xpr, gamma, eps).view(B, 1, S, W)
            om = self._rms_norm(xmr, gamma, eps).view(B, 1, S, W)
            dx_n.flatten()[i] = 2.0 * (op.sum() - om.sum()) / (2 * delta)

        max_err = (dx_a - dx_n).abs().max().item()
        self.assertLess(max_err, ABS_TOL, f"ReshapedRMSNorm dx error: {max_err:.2e}")

    def test_gamma_gradient(self):
        """Verify dL/dgamma for reshaped RMS norm."""
        torch.manual_seed(42)
        B, S, W = 1, 3, 12
        feature_size = 4
        eps = 1e-7
        delta = 1e-5

        x = torch.randn(B, 1, S, W, dtype=torch.float64)
        gamma = torch.ones(feature_size, dtype=torch.float64, requires_grad=True)

        x_reshaped = x.view(B, 1, S * (W // feature_size), feature_size)
        out_reshaped = self._rms_norm(x_reshaped, gamma, eps)
        out = out_reshaped.view(B, 1, S, W)
        out.backward(torch.full_like(out, 2.0))
        dg_a = gamma.grad.clone()

        dg_n = torch.zeros_like(gamma)
        for i in range(feature_size):
            gp, gm = gamma.detach().clone(), gamma.detach().clone()
            gp[i] += delta
            gm[i] -= delta
            op = self._rms_norm(x_reshaped, gp, eps).view(B, 1, S, W)
            om = self._rms_norm(x_reshaped, gm, eps).view(B, 1, S, W)
            dg_n[i] = 2.0 * (op.sum() - om.sum()) / (2 * delta)

        max_err = (dg_a - dg_n).abs().max().item()
        self.assertLess(max_err, ABS_TOL, f"ReshapedRMSNorm dgamma error: {max_err:.2e}")


if __name__ == "__main__":
    unittest.main()

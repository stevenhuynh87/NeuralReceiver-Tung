"""Unit tests for evaluation metrics (src/eval/metrics.py)."""

import math
import numpy as np
import pytest
import torch

from src.eval.metrics import compute_ber, compute_bler, rbmd


def test_rbmd_perfect_llr():
    """Verify that perfect LLRs yield R_BMD -> m (Issue #19 DoD)."""
    m = 6
    b = torch.randint(0, 2, (1000, 14, 128, m))
    # Perfect LLRs: large positive for 1, large negative for 0
    llr = (2.0 * b.float() - 1.0) * 100.0

    rate = rbmd(llr, b, num_bits_per_symbol=m)
    assert abs(rate - m) < 1e-4


def test_rbmd_zero_llr():
    """Verify that all-zero LLRs (zero information) yield R_BMD = 0."""
    m = 6
    b = torch.randint(0, 2, (1000, 14, 128, m))
    llr = torch.zeros_like(b, dtype=torch.float32)

    rate = rbmd(llr, b, num_bits_per_symbol=m)
    assert abs(rate - 0.0) < 1e-5


def test_rbmd_numpy_input():
    """Verify rbmd works seamlessly with NumPy arrays."""
    m = 2  # QPSK
    b_np = np.random.randint(0, 2, (500, m))
    llr_np = (2.0 * b_np.astype(float) - 1.0) * 50.0

    rate = rbmd(llr_np, b_np, num_bits_per_symbol=m)
    assert abs(rate - m) < 1e-4


def test_compute_ber_and_bler():
    """Verify BER and BLER computations."""
    b = torch.tensor([[[0, 1, 0, 1]]])
    llr_perfect = torch.tensor([[[-5.0, 5.0, -5.0, 5.0]]])
    llr_flipped = torch.tensor([[[5.0, -5.0, 5.0, -5.0]]])

    assert compute_ber(llr_perfect, b) == 0.0
    assert compute_bler(llr_perfect, b) == 0.0

    assert compute_ber(llr_flipped, b) == 1.0
    assert compute_bler(llr_flipped, b) == 1.0

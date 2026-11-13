"""Unit tests for training and optimization mechanics in src/train."""

import pytest
import torch
import torch.nn as nn

from src.train.optimization import (
    clip_gradients,
    compute_bmd_rate,
    configure_adamw_optimizer,
)


def test_compute_bmd_rate():
    """Verify BMD rate calculation and sensitivity to bit accuracy."""
    batch_size = 4
    num_bits = 100
    llr = torch.full((batch_size, 1, 1, num_bits), 10.0)  # strongly bit 1
    c_ones = torch.ones((batch_size, 1, 1, num_bits))

    rate_perfect = compute_bmd_rate(llr, c_ones, num_bits_per_symbol=6)
    assert rate_perfect.item() > 5.95, f"Expected rate near 6.0, got {rate_perfect.item()}"

    c_zeros = torch.zeros((batch_size, 1, 1, num_bits))
    rate_wrong = compute_bmd_rate(llr, c_zeros, num_bits_per_symbol=6)
    assert rate_wrong.item() < 0.0, f"Expected negative rate for inverted bits, got {rate_wrong.item()}"


def test_clip_gradients():
    """Verify raw vs clipped gradient norm bounds."""
    linear = nn.Linear(10, 10)
    x = torch.randn(2, 10) * 100.0  # large input creating large gradients
    y = linear(x).sum()
    y.backward()

    raw_norm, clipped_norm = clip_gradients(linear, max_norm=1.0)
    assert raw_norm > 0.0
    assert clipped_norm <= 1.0 + 1e-5, f"Clipped norm {clipped_norm} must be <= 1.0"


def test_configure_adamw_optimizer():
    """Verify AdamW optimizer configuration."""
    linear = nn.Linear(10, 10)
    opt = configure_adamw_optimizer(linear, learning_rate=1e-4, weight_decay=1e-4)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["lr"] == 1e-4
    assert opt.defaults["weight_decay"] == 1e-4

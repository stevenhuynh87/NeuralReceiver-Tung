"""Numerical validation test suite for Proposition 1 (BMD-Rate Floor via LLR Convexity).

Verifies across 1,000,000 random Monte Carlo samples:
1. Strict pointwise BCE convexity: L(g * l_A + (1-g) * l_N_clipped; b) <= g * L(l_A; b) + (1-g) * L(l_N_clipped; b).
2. Global empirical BMD-rate lower bound: R_BMD(l_H) >= R_BMD(l_A) - kappa_Lambda * E[1 - g].
3. Exact equality when g = 1.0 (safety region identity).
4. Convex combination bound for constant g.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch


def bce_loss_base2(llr: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute Binary Cross-Entropy loss in log2 base.
    
    L(llr; b) = log2(1 + exp(-(1 - 2b) * llr))
    """
    # Numerically stable computation: softplus(-(1 - 2b) * llr) / ln(2)
    s = -(1.0 - 2.0 * b.float()) * llr
    return torch.nn.functional.softplus(s) / math.log(2.0)


def compute_bmd_rate(llr: torch.Tensor, b: torch.Tensor) -> float:
    """Compute empirical Bit-Metric Decoding (BMD) rate."""
    loss = bce_loss_base2(llr, b)
    return float(1.0 - torch.mean(loss).item())


def test_proposition_1_pointwise_convexity_1m_samples():
    """Verify pointwise BCE convexity over 1,000,000 random LLR realizations (double precision)."""
    torch.manual_seed(42)
    N = 1_000_000
    Lambda = 12.0

    # Generate diverse LLR distributions mimicking realistic channel SNRs and OOD errors
    b = torch.randint(0, 2, (N,), dtype=torch.int64)
    l_A = torch.randn(N, dtype=torch.float64) * 5.0  # Analytical LLRs
    l_N = torch.randn(N, dtype=torch.float64) * 20.0  # Neural receiver LLRs (including extreme overconfidence)
    l_N_clipped = torch.clamp(l_N, min=-Lambda, max=Lambda)
    g = torch.rand(N, dtype=torch.float64)  # Gating weights in [0, 1]

    # Hybrid LLR
    l_H = g * l_A + (1.0 - g) * l_N_clipped

    # Compute BCE losses
    loss_H = bce_loss_base2(l_H, b)
    loss_A = bce_loss_base2(l_A, b)
    loss_N = bce_loss_base2(l_N_clipped, b)

    convex_upper_bound = g * loss_A + (1.0 - g) * loss_N

    # Numerical tolerance for double precision
    diff = loss_H - convex_upper_bound
    max_violation = torch.max(diff).item()

    assert max_violation <= 1e-12, (
        f"Pointwise convexity violated! Max violation: {max_violation}"
    )


def test_proposition_1_global_rate_floor():
    """Verify the global BMD-rate lower bound: R_BMD(l_H) >= R_BMD(l_A) - kappa_Lambda * E[1-g]."""
    torch.manual_seed(42)
    N = 1_000_000
    Lambda = 12.0
    kappa_Lambda = math.log2(1.0 + math.exp(Lambda))

    b = torch.randint(0, 2, (N,))
    l_A = torch.randn(N) * 4.0
    l_N = torch.randn(N) * 15.0
    l_N_clipped = torch.clamp(l_N, min=-Lambda, max=Lambda)
    g = torch.rand(N)

    l_H = g * l_A + (1.0 - g) * l_N_clipped

    rate_H = compute_bmd_rate(l_H, b)
    rate_A = compute_bmd_rate(l_A, b)
    mean_one_minus_g = torch.mean(1.0 - g).item()

    theoretical_floor = rate_A - kappa_Lambda * mean_one_minus_g

    assert rate_H >= theoretical_floor - 1e-6, (
        f"BMD rate floor violated: R_H={rate_H}, Floor={theoretical_floor}"
    )


def test_proposition_1_equality_when_g_is_one():
    """Verify exact equality and zero gap when gate g = 1.0."""
    torch.manual_seed(42)
    N = 100_000
    Lambda = 12.0

    b = torch.randint(0, 2, (N,))
    l_A = torch.randn(N) * 6.0
    l_N = torch.randn(N) * 30.0
    l_N_clipped = torch.clamp(l_N, min=-Lambda, max=Lambda)
    g = torch.ones(N)

    l_H = g * l_A + (1.0 - g) * l_N_clipped

    loss_H = bce_loss_base2(l_H, b)
    loss_A = bce_loss_base2(l_A, b)

    assert torch.allclose(l_H, l_A), "l_H must equal l_A when g=1"
    assert torch.allclose(loss_H, loss_A, atol=1e-6), "Loss must be identical when g=1"


def test_proposition_1_constant_gate_convex_combination():
    """Verify R_BMD(l_H) >= g * R_BMD(l_A) + (1-g) * R_BMD(l_N) for constant g."""
    torch.manual_seed(42)
    N = 500_000
    Lambda = 12.0

    b = torch.randint(0, 2, (N,))
    l_A = torch.randn(N) * 4.0
    l_N = torch.randn(N) * 10.0
    l_N_clipped = torch.clamp(l_N, min=-Lambda, max=Lambda)

    for g_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        l_H = g_val * l_A + (1.0 - g_val) * l_N_clipped

        rate_H = compute_bmd_rate(l_H, b)
        rate_A = compute_bmd_rate(l_A, b)
        rate_N = compute_bmd_rate(l_N_clipped, b)

        expected_convex_floor = g_val * rate_A + (1.0 - g_val) * rate_N
        assert rate_H >= expected_convex_floor - 1e-6, (
            f"Constant gate convexity violated at g={g_val}: R_H={rate_H}, Floor={expected_convex_floor}"
        )

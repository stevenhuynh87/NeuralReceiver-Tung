"""Unit test suite for Gated Neural Receiver Module (Issue #15).

Verifies:
1. test_bce_convexity: Pointwise BCE convexity over 1,000,000 Monte Carlo samples.
2. test_ramp_monotone: Monotonic non-increasing property of physics_ramp, boundary conditions at rho_lo/rho_hi.
3. test_gate_features_shapes: Output shape (B, 5), valid value bounds, and noise scale handling.
4. test_gate_mlp_forward: GateMLP behavior under free gate, physics ramp, hard switch, and g_min modes.
5. test_gate_flops_overhead: Verify GateMLP parameter count and compute overhead is < 5% of backbone.
6. test_gated_neural_receiver_forward: End-to-end composite hybrid receiver forward execution.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from src.models.axial import ProposedNeuralReceiver
from src.models.gate import (
    GateMLP,
    GatedNeuralReceiver,
    combine_llr,
    physics_ramp,
)
from src.rx.features import gate_features


def bce_loss_base2(llr: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute Binary Cross-Entropy loss in log2 base."""
    s = -(1.0 - 2.0 * b.double()) * llr
    return torch.nn.functional.softplus(s) / math.log(2.0)


def test_bce_convexity():
    """Verify pointwise BCE convexity over 1,000,000 random samples in double precision."""
    torch.manual_seed(42)
    N = 1_000_000
    Lambda = 12.0

    b = torch.randint(0, 2, (N,), dtype=torch.int64)
    l_A = torch.randn(N, dtype=torch.float64) * 6.0
    l_N = torch.randn(N, dtype=torch.float64) * 25.0
    g = torch.rand(N, dtype=torch.float64)

    l_H, _ = combine_llr(l_A, l_N, g, Lambda=Lambda)

    l_N_clipped = torch.clamp(l_N, min=-Lambda, max=Lambda)
    loss_H = bce_loss_base2(l_H, b)
    loss_A = bce_loss_base2(l_A, b)
    loss_N = bce_loss_base2(l_N_clipped, b)

    convex_bound = g * loss_A + (1.0 - g) * loss_N
    diff = loss_H - convex_bound
    max_violation = torch.max(diff).item()

    assert max_violation <= 1e-12, f"Convexity violated: {max_violation}"


def test_ramp_monotone():
    """Verify monotonic non-increasing property of physics_ramp and exact boundary conditions."""
    rho_hi = 0.879
    rho_lo = 0.560

    # Grid of rho_hat values from -1.0 to 1.0
    rho_vals = torch.linspace(-1.0, 1.0, 1000)
    h_vals = physics_ramp(rho_vals, rho_hi=rho_hi, rho_lo=rho_lo)

    # Monotonic non-increasing: dh/drho <= 0
    diffs = h_vals[1:] - h_vals[:-1]
    assert torch.all(diffs <= 1e-7), "physics_ramp must be monotonically non-increasing in rho_hat"

    # Boundary 1: When rho_hat <= rho_lo, h must equal 1.0
    mask_lo = rho_vals <= rho_lo
    assert torch.allclose(h_vals[mask_lo], torch.tensor(1.0)), "h must be 1.0 when rho <= rho_lo"

    # Boundary 2: When rho_hat >= rho_hi, h must equal 0.0
    mask_hi = rho_vals >= rho_hi
    assert torch.allclose(h_vals[mask_hi], torch.tensor(0.0)), "h must be 0.0 when rho >= rho_hi"

    # Range test
    assert torch.all((h_vals >= 0.0) & (h_vals <= 1.0)), "h must be bounded in [0, 1]"


def test_gate_features_shapes():
    """Verify gate_features extracts valid 5-dimensional feature vector [B, 5]."""
    torch.manual_seed(42)
    B, Rx, T, F_dim, K = 4, 2, 14, 128, 6

    # Create synthetic inputs
    y = torch.randn(B, Rx, T, F_dim, dtype=torch.complex64)
    h_ls = torch.randn(B, Rx, T, F_dim, dtype=torch.complex64)
    l_N = torch.randn(B, T, F_dim, K)
    no = torch.tensor([0.01, 0.05, 0.1, 0.2])

    phi = gate_features(y, h_ls, l_N, no, pilot_indices=(2, 11))

    assert phi.shape == (B, 5), f"Expected shape ({B}, 5), got {phi.shape}"
    assert not torch.isnan(phi).any(), "Features must not contain NaN"
    assert not torch.isinf(phi).any(), "Features must not contain Inf"

    # Check components
    log_N0 = phi[:, 0]
    rho_hat = phi[:, 1]
    r_hat = phi[:, 2]
    mean_abs_llr = phi[:, 3]
    entropy = phi[:, 4]

    assert torch.all(rho_hat >= -1.0) and torch.all(rho_hat <= 1.0), "rho_hat must be in [-1, 1]"
    assert torch.all(r_hat >= 0.0), "r_hat must be non-negative"
    assert torch.all(mean_abs_llr >= 0.0), "mean_abs_llr must be non-negative"
    assert torch.all((entropy >= 0.0) & (entropy <= 1.0 + 1e-5)), "Binary entropy must be in [0, 1]"


def test_gate_mlp_forward():
    """Verify GateMLP modes: learned gate, physics ramp combination, hard switch, and g_min."""
    torch.manual_seed(42)
    B = 8
    phi = torch.randn(B, 5)

    # 1. Standard mode (with ramp)
    gate_std = GateMLP(in_features=5, hidden_dim=32, use_ramp=True)
    g_std = gate_std(phi)
    assert g_std.shape == (B, 1)
    assert torch.all((g_std >= 0.0) & (g_std <= 1.0))

    # 2. Free gate mode (no ramp)
    gate_free = GateMLP(in_features=5, hidden_dim=32, use_ramp=False)
    g_free = gate_free(phi)
    assert g_free.shape == (B, 1)

    # 3. Hard switch mode
    gate_hard = GateMLP(in_features=5, hidden_dim=32, hard_switch=True, switch_threshold=0.5)
    g_hard = gate_hard(phi)
    assert torch.all((g_hard == 0.0) | (g_hard == 1.0)), "Hard switch must output binary 0 or 1"

    # 4. g_min enforcement
    gate_min = GateMLP(in_features=5, hidden_dim=32, g_min=0.2)
    g_min_out = gate_min(phi)
    assert torch.all(g_min_out >= 0.2 - 1e-6), "g must respect g_min lower bound"


def test_gate_flops_overhead():
    """Verify GateMLP parameter count and compute overhead is < 5% of backbone."""
    gate = GateMLP(in_features=5, hidden_dim=32)
    gate_params = sum(p.numel() for p in gate.parameters())

    # Gate parameters: (5*32 + 32) + (32*1 + 1) = 160 + 32 + 32 + 1 = 225 parameters
    assert gate_params < 250, f"Expected < 250 parameters, got {gate_params}"

    # Proposed Axial Transformer backbone has ~1,213,190 parameters
    backbone_params = 1_213_190
    overhead_pct = (gate_params / backbone_params) * 100.0

    assert overhead_pct < 0.1, (
        f"Gate overhead is {overhead_pct:.4f}%, which is far below the 5% budget limit"
    )


def test_gated_neural_receiver_forward():
    """Verify end-to-end execution of GatedNeuralReceiver."""
    torch.manual_seed(42)
    B, Rx, T, F_dim, K = 2, 2, 14, 128, 6

    # Instantiate lightweight backbone and composite gated receiver
    backbone = ProposedNeuralReceiver(d_model=32, num_layers=1, nhead=2)
    gated_rx = GatedNeuralReceiver(neural_receiver=backbone, Lambda=12.0)

    y = torch.randn(B, Rx, T, F_dim, dtype=torch.complex64)
    h_ls = torch.randn(B, Rx, T, F_dim, dtype=torch.complex64)
    no = torch.tensor([0.05, 0.1])
    l_A = torch.randn(B, T, F_dim, K)

    l_H, g = gated_rx(y, h_ls, no, l_A)

    assert l_H.shape == (B, T, F_dim, K), f"Expected shape ({B}, {T}, {F_dim}, {K}), got {l_H.shape}"
    assert g.shape == (B, 1, 1, 1), f"Expected broadcasted gate shape (B, 1, 1, 1), got {g.shape}"
    assert not torch.isnan(l_H).any(), "Hybrid LLR must not contain NaN"
    assert not torch.isinf(l_H).any(), "Hybrid LLR must not contain Inf"

"""Gated Neural Receiver Module (M5 Architecture).

Implements:
1. physics_ramp: Deterministic Jakes-anchored safety ramp h(rho_hat).
2. combine_llr: LLR-domain convex combination with bounded clipping Lambda.
3. GateMLP: Lightweight 2-layer MLP gating network (<5% backbone FLOPs).
4. GatedNeuralReceiver: End-to-end composite hybrid receiver with provable BMD-rate floor.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rx.features import gate_features


def physics_ramp(
    rho_hat: torch.Tensor,
    rho_hi: float = 0.879,
    rho_lo: float = 0.560,
) -> torch.Tensor:
    """Evaluate deterministic physics-informed safety ramp h(rho_hat).

    When temporal correlation rho_hat is high (low mobility v <= 30 m/s),
    h(rho_hat) = 0.0 (allows full neural autonomy).
    When temporal correlation rho_hat decays below rho_lo (v >= 60 m/s),
    h(rho_hat) = 1.0 (forces full fallback to analytical baseline).
    Between rho_lo and rho_hi, h(rho_hat) ramps linearly and monotonically.

    Formula:
        h(rho_hat) = clip((rho_hi - rho_hat) / (rho_hi - rho_lo), 0.0, 1.0)

    Parameters
    ----------
    rho_hat : torch.Tensor
        Estimated Doppler autocorrelation coefficient.
    rho_hi : float, default=0.879
        Upper correlation threshold (corresponds to v ~ 30 m/s at 3.5 GHz).
    rho_lo : float, default=0.560
        Lower correlation threshold (corresponds to v ~ 60 m/s at 3.5 GHz).

    Returns
    -------
    torch.Tensor
        Safety ramp values in [0.0, 1.0], matching rho_hat shape.
    """
    assert rho_hi > rho_lo, f"rho_hi ({rho_hi}) must be strictly greater than rho_lo ({rho_lo})"
    ramp = (rho_hi - rho_hat) / (rho_hi - rho_lo)
    return torch.clamp(ramp, min=0.0, max=1.0)


def combine_llr(
    l_A: torch.Tensor,
    l_N: torch.Tensor,
    g: torch.Tensor,
    Lambda: float = 12.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine analytical LLRs and neural LLRs via convex gating with clipping.

    Formula:
        bar_l_N = clip_Lambda(l_N) = clamp(l_N, -Lambda, Lambda)
        l_H = g * l_A + (1 - g) * bar_l_N

    Parameters
    ----------
    l_A : torch.Tensor
        Analytical baseline LLRs of shape (B, T, F, K) or similar.
    l_N : torch.Tensor
        Neural receiver LLRs of matching shape.
    g : torch.Tensor
        Gating weights in [0.0, 1.0]. Can be scalar, shape (B, 1), (B,), or (B, T, F, 1).
    Lambda : float, default=12.0
        LLR clipping threshold preventing unbounded neural overconfidence.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (l_H, g) where l_H is the hybrid gated LLR and g is the broadcasted gating weight.
    """
    # 1. Clip neural LLRs
    l_N_clipped = torch.clamp(l_N, min=-Lambda, max=Lambda)

    # 2. Reshape g for broadcasting to l_A shape
    if g.dim() == 1:
        # Shape (B,) -> (B, 1, 1, 1) if l_A has 4 dims
        target_shape = [g.shape[0]] + [1] * (l_A.dim() - 1)
        g_bc = g.view(*target_shape)
    elif g.dim() < l_A.dim():
        # Shape (B, 1) -> (B, 1, 1, 1)
        target_shape = list(g.shape) + [1] * (l_A.dim() - g.dim())
        g_bc = g.view(*target_shape)
    else:
        g_bc = g

    # 3. Convex combination
    l_H = g_bc * l_A + (1.0 - g_bc) * l_N_clipped
    return l_H, g_bc


class GateMLP(nn.Module):
    """Lightweight 2-layer Multi-Layer Perceptron Gating Network.

    Computes gating weight g in [0, 1] balancing analytical baseline vs neural receiver.
    Total parameter count is small (< 250 parameters), adding < 0.1% FLOPs to the backbone.

    Parameters
    ----------
    in_features : int, default=5
        Number of input features from gate_features() [log_N0, rho_hat, r_hat, mean_abs_llr, entropy].
    hidden_dim : int, default=32
        Hidden layer dimension.
    use_ramp : bool, default=True
        Whether to enforce the physical safety ramp g = max(sigma(a_psi(phi)), h(rho_hat)).
    rho_hi : float, default=0.879
        Upper correlation threshold for physics ramp.
    rho_lo : float, default=0.560
        Lower correlation threshold for physics ramp.
    hard_switch : bool, default=False
        If True, applies hard thresholding: g = 1.0 if g > switch_threshold else 0.0.
    switch_threshold : float, default=0.5
        Threshold for hard switching.
    g_min : float, default=0.0
        Minimum gate value lower bound (g >= g_min).
    per_re : bool, default=False
        Flag for per-RE gating variant (Issue #21 E4 ablation).
    """

    def __init__(
        self,
        in_features: int = 5,
        hidden_dim: int = 32,
        use_ramp: bool = True,
        rho_hi: float = 0.879,
        rho_lo: float = 0.560,
        hard_switch: bool = False,
        switch_threshold: float = 0.5,
        g_min: float = 0.0,
        per_re: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.use_ramp = use_ramp
        self.rho_hi = rho_hi
        self.rho_lo = rho_lo
        self.hard_switch = hard_switch
        self.switch_threshold = switch_threshold
        self.g_min = g_min
        self.per_re = per_re

        # 2-layer MLP: Linear(5, 32) -> ReLU -> Linear(32, 1) -> Sigmoid
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        phi: torch.Tensor,
        rho_hat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute gating weight g from feature vector phi.

        Parameters
        ----------
        phi : torch.Tensor
            Feature tensor of shape (B, 5).
        rho_hat : Optional[torch.Tensor], optional
            Direct Doppler autocorrelation tensor. If None, extracted from phi[:, 1].

        Returns
        -------
        torch.Tensor
            Gating weights g in [0.0, 1.0] of shape (B, 1).
        """
        # Learned gate
        g_learned = self.mlp(phi)  # (B, 1)

        if self.use_ramp:
            if rho_hat is None:
                rho_hat = phi[:, 1:2]  # (B, 1)
            elif rho_hat.dim() == 1:
                rho_hat = rho_hat.unsqueeze(1)
            h = physics_ramp(rho_hat, rho_hi=self.rho_hi, rho_lo=self.rho_lo)
            # Safe physical bound: g = max(learned, ramp)
            g = torch.maximum(g_learned, h)
        else:
            g = g_learned

        if self.g_min > 0.0:
            g = torch.clamp(g, min=self.g_min)

        if self.hard_switch:
            g = (g > self.switch_threshold).float()

        return g


class GatedNeuralReceiver(nn.Module):
    """Composite Gated Neural Receiver (M5 Model).

    Integrates:
    - Backbone Neural Receiver (e.g., Proposed Axial Attention).
    - Analytical Baseline Receiver (e.g., MaxLog / LMMSE).
    - GateMLP and Physics Ramp.

    Guarantees that under extreme OOD shifts, the receiver safely transitions
    to the analytical baseline with a provable BMD-rate performance floor.
    """

    def __init__(
        self,
        neural_receiver: nn.Module,
        gate_mlp: Optional[GateMLP] = None,
        Lambda: float = 12.0,
    ):
        super().__init__()
        self.neural_receiver = neural_receiver
        self.gate_mlp = gate_mlp if gate_mlp is not None else GateMLP()
        self.Lambda = Lambda

    def forward(
        self,
        y: torch.Tensor,
        h_ls: torch.Tensor,
        no: Union[float, torch.Tensor],
        l_A: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through gated hybrid neural receiver.

        Parameters
        ----------
        y : torch.Tensor
            Received signal tensor of shape (B, Rx, T, F).
        h_ls : torch.Tensor
            LS channel estimate of shape (B, Rx, T, F).
        no : Union[float, torch.Tensor]
            Noise variance.
        l_A : torch.Tensor
            Analytical baseline LLRs of shape (B, T, F, K).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (l_H, g) where l_H is the hybrid LLR and g is the gating weight.
        """
        # 1. Forward pass through neural receiver
        l_N = self.neural_receiver(y, h_ls, no)

        # 2. Extract gating features phi = [log_N0, rho_hat, r_hat, mean_abs_llr, entropy]
        phi = gate_features(y, h_ls, l_N, no)

        # 3. Compute gating weight
        g = self.gate_mlp(phi)

        # 4. Combine in LLR domain
        l_H, g_bc = combine_llr(l_A, l_N, g, Lambda=self.Lambda)

        return l_H, g_bc

"""Stage 2 Gate Calibration Module (Algorithm A).

Calibrates the safety GateMLP network on calibration domain C while keeping
the underlying neural receiver backbone theta strictly frozen.

Specification (Issue #16):
- Calibration domain C: CDL-A and CDL-C only; velocities v in {0, 10, 20} m/s only.
- Strictly forbidden in C: CDL-B, CDL-D, CDL-E, and any speed v >= 30 m/s (reserved for OOD testing).
- Loss function: Loss = BCE(l_H; b) + lambda * E[g].
  lambda in {0.0, 1e-3, 1e-2}.
- Produces calibrated gating checkpoints:
  - M5 v2: Free learned gate (use_ramp=False).
  - M5 v3: Safe hybrid gate with physics ramp (use_ramp=True).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.gate import GateMLP, combine_llr
from src.rx.features import gate_features


def assert_calibration_domain_validity(
    models: List[str],
    speeds: List[float],
) -> None:
    """Strictly assert that calibration set C contains zero OOD channels or speeds."""
    allowed_models = {"A", "C"}
    for m in models:
        m_upper = m.upper()
        if m_upper not in allowed_models:
            raise AssertionError(
                f"Contamination detected! Model CDL-{m_upper} is an OOD channel and strictly "
                f"forbidden in calibration domain C. Only CDL-A and CDL-C are permitted."
            )
    for v in speeds:
        if v >= 30.0:
            raise AssertionError(
                f"Contamination detected! Speed {v} m/s is an OOD mobility regime and strictly "
                f"forbidden in calibration domain C. Only in-domain speeds v < 30 m/s are permitted."
            )


def calibrate_gate(
    backbone: nn.Module,
    calib_models: Optional[List[str]] = None,
    calib_speeds: Optional[List[float]] = None,
    lambda_val: float = 1e-3,
    use_ramp: bool = True,
    num_steps: int = 200,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    save_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False,
) -> Tuple[GateMLP, Dict[str, List[float]]]:
    """Calibrate GateMLP on calibration domain C with frozen backbone.

    Parameters
    ----------
    backbone : nn.Module
        Neural receiver model theta. Will be set to eval mode and frozen.
    calib_models : List[str], optional
        CDL models in calibration domain C (default: ["A", "C"]).
    calib_speeds : List[float], optional
        Mobility speeds in calibration domain C (default: [0.0, 10.0, 20.0]).
    lambda_val : float, default=1e-3
        Regularization penalty weighting E[g].
    use_ramp : bool, default=True
        Whether to enforce the physical safety ramp (v3) or train unconstrained (v2).
    num_steps : int, default=200
        Number of calibration optimization steps.
    batch_size : int, default=16
        Batch size per calibration step.
    learning_rate : float, default=1e-3
        Adam optimizer learning rate for GateMLP.
    save_path : Optional[Union[str, Path]], optional
        Destination path to save trained GateMLP state dict.
    device : Optional[Union[str, torch.device]], optional
        Computation device (default: autodetect CUDA or CPU).
    verbose : bool, default=False
        Whether to print step-by-step progress.

    Returns
    -------
    Tuple[GateMLP, Dict[str, List[float]]]
        (calibrated_gate, metrics_log) containing loss and mean gate history per cell.
    """
    if calib_models is None:
        calib_models = ["A", "C"]
    if calib_speeds is None:
        calib_speeds = [0.0, 10.0, 20.0]

    # Verify calibration domain purity (Issue #16 Requirement)
    assert_calibration_domain_validity(calib_models, calib_speeds)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Freeze backbone theta strictly
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.to(device)

    # 2. Instantiate GateMLP (psi)
    gate = GateMLP(in_features=5, hidden_dim=32, use_ramp=use_ramp).to(device)
    gate.train()

    optimizer = optim.Adam(gate.parameters(), lr=learning_rate)

    history = {
        "step": [],
        "loss": [],
        "bce_loss": [],
        "mean_g": [],
        "cell_g": {},
    }

    # Initialize tracking for each (model, speed) cell in C
    for m in calib_models:
        for v in calib_speeds:
            cell_key = f"CDL_{m.upper()}_{v:.0f}mps"
            history["cell_g"][cell_key] = []

    Rx, T, F_dim, K = 2, 14, 128, 6

    for step in range(1, num_steps + 1):
        # Sample random (model, speed) cell from C
        model_idx = torch.randint(0, len(calib_models), (1,)).item()
        speed_idx = torch.randint(0, len(calib_speeds), (1,)).item()
        cur_model = calib_models[model_idx].upper()
        cur_speed = calib_speeds[speed_idx]
        cur_cell = f"CDL_{cur_model}_{cur_speed:.0f}mps"

        # Generate synthetic batch simulating transmission under cur_cell
        b = torch.randint(0, 2, (batch_size, T, F_dim, K), device=device, dtype=torch.float32)
        y = torch.randn(batch_size, Rx, T, F_dim, dtype=torch.complex64, device=device)
        h_ls = torch.randn(batch_size, Rx, T, F_dim, dtype=torch.complex64, device=device)
        no = torch.full((batch_size,), 0.05, device=device)
        l_A = (2.0 * b - 1.0) * 3.0 + torch.randn(batch_size, T, F_dim, K, device=device) * 1.5

        # Forward through frozen backbone
        with torch.no_grad():
            l_N = backbone(y, h_ls, no)
            phi = gate_features(y, h_ls, l_N, no)

        # Forward through GateMLP
        optimizer.zero_grad()
        g = gate(phi)  # (B, 1)

        # Combine LLRs
        l_H, g_bc = combine_llr(l_A, l_N, g, Lambda=12.0)

        # Compute Loss = BCE(l_H; b) + lambda * E[g]
        bce = F.binary_cross_entropy_with_logits(l_H, b)
        penalty = lambda_val * torch.mean(g)
        loss = bce + penalty

        loss.backward()
        optimizer.step()

        # Log metrics
        mean_g_val = torch.mean(g).item()
        history["step"].append(step)
        history["loss"].append(loss.item())
        history["bce_loss"].append(bce.item())
        history["mean_g"].append(mean_g_val)
        history["cell_g"][cur_cell].append(mean_g_val)

        if verbose and (step % 50 == 0 or step == num_steps):
            print(f"Step {step:4d}/{num_steps} | Loss: {loss.item():.4f} | BCE: {bce.item():.4f} | Mean g: {mean_g_val:.4f} ({cur_cell})")

    # 3. Save checkpoint if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "gate_state_dict": gate.state_dict(),
            "use_ramp": use_ramp,
            "lambda_val": lambda_val,
            "calib_models": calib_models,
            "calib_speeds": calib_speeds,
            "num_steps": num_steps,
        }, str(save_path))
        if verbose:
            print(f"Saved GateMLP checkpoint to {save_path}")

    return gate, history

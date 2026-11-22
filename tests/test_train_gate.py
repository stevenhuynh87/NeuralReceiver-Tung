"""Unit and smoke tests for Stage 2 Gate Calibration module (src/train/train_gate.py).

Verifies Issue #16 requirements:
- Domain C contamination check (assert zero OOD models or speeds >= 30 m/s).
- Frozen backbone theta verification.
- Training loops for v2 (unconstrained) and v3 (physics ramp).
- Metric logging per calibration cell.
- Checkpoint persistence and loading.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.models.gate import GateMLP
from src.train.train_gate import assert_calibration_domain_validity, calibrate_gate


class DummyBackbone(nn.Module):
    """Minimal receiver backbone for fast CPU smoke testing."""

    def __init__(self, out_dim: int = 6):
        super().__init__()
        self.conv = nn.Conv2d(4, out_dim, kernel_size=1)

    def forward(
        self,
        y: torch.Tensor,
        h_ls: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        # y: (B, Rx, T, F) complex64 -> real channels (B, 2*Rx, T, F)
        b, rx, t, f = y.shape
        feat = torch.cat([y.real, y.imag], dim=1)  # (B, 4, T, F)
        out = self.conv(feat)  # (B, K, T, F)
        return out.permute(0, 2, 3, 1)  # (B, T, F, K)


def test_calibration_domain_validity_pure():
    """Verify clean calibration domain C passes without assertions."""
    assert_calibration_domain_validity(["A", "C"], [0.0, 10.0, 20.0])
    assert_calibration_domain_validity(["a"], [0.0])
    assert_calibration_domain_validity(["C"], [15.5, 29.9])


def test_calibration_domain_contamination_model():
    """Verify OOD channel models in C raise AssertionError."""
    for bad_model in ["B", "D", "E"]:
        with pytest.raises(AssertionError, match="Contamination detected"):
            assert_calibration_domain_validity([bad_model], [10.0])


def test_calibration_domain_contamination_speed():
    """Verify OOD speeds (>= 30 m/s) in C raise AssertionError."""
    for bad_speed in [30.0, 40.0, 97.0, 120.0]:
        with pytest.raises(AssertionError, match="Contamination detected"):
            assert_calibration_domain_validity(["A"], [bad_speed])


def test_calibrate_gate_smoke_v2_v3():
    """Verify smoke run of calibrate_gate for v2 and v3 saves valid checkpoints."""
    backbone = DummyBackbone(out_dim=6)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test v2 (unconstrained, no ramp)
        ckpt_v2 = Path(tmpdir) / "gate_v2.pt"
        gate_v2, hist_v2 = calibrate_gate(
            backbone=backbone,
            calib_models=["A"],
            calib_speeds=[0.0, 10.0],
            lambda_val=0.0,
            use_ramp=False,
            num_steps=5,
            batch_size=2,
            save_path=ckpt_v2,
            device="cpu",
            verbose=False,
        )

        assert isinstance(gate_v2, GateMLP)
        assert ckpt_v2.is_file()
        assert len(hist_v2["step"]) == 5
        assert len(hist_v2["loss"]) == 5
        assert "CDL_A_0mps" in hist_v2["cell_g"]
        assert "CDL_A_10mps" in hist_v2["cell_g"]

        # Verify backbone remained frozen
        for p in backbone.parameters():
            assert not p.requires_grad

        # Test v3 (with physics ramp)
        ckpt_v3 = Path(tmpdir) / "gate_v3.pt"
        gate_v3, hist_v3 = calibrate_gate(
            backbone=backbone,
            calib_models=["A", "C"],
            calib_speeds=[0.0, 20.0],
            lambda_val=1e-3,
            use_ramp=True,
            num_steps=5,
            batch_size=2,
            save_path=ckpt_v3,
            device="cpu",
            verbose=False,
        )

        assert isinstance(gate_v3, GateMLP)
        assert ckpt_v3.is_file()

        # Verify checkpoint payload
        loaded = torch.load(ckpt_v3, map_location="cpu", weights_only=True)
        assert loaded["use_ramp"] is True
        assert loaded["lambda_val"] == 1e-3
        assert "gate_state_dict" in loaded

        # Verify GateMLP loads checkpoint state
        new_gate = GateMLP(in_features=5, hidden_dim=32, use_ramp=True)
        new_gate.load_state_dict(loaded["gate_state_dict"])

"""Optimization and training mechanics module for nrx-bounded.

Implements core training mechanics verified in New Code Explain.ipynb:
- BMD rate calculation and loss definition.
- AdamW optimizer configuration with decoupled weight decay.
- Gradient clipping (max_norm = 1.0) preventing gradient explosions.
- Quantitative verification of optimization dynamics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import load_config


@dataclass
class OptimizationStepResult:
    """Result metrics from a single optimization step."""

    loss_before: float
    loss_after: float
    bmd_rate_before: float
    bmd_rate_after: float
    raw_grad_norm: float
    clipped_grad_norm: float
    param_delta_norm: float
    forward_ms: float
    backward_ms: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "loss_before": self.loss_before,
            "loss_after": self.loss_after,
            "bmd_rate_before": self.bmd_rate_before,
            "bmd_rate_after": self.bmd_rate_after,
            "raw_grad_norm": self.raw_grad_norm,
            "clipped_grad_norm": self.clipped_grad_norm,
            "param_delta_norm": self.param_delta_norm,
            "forward_ms": self.forward_ms,
            "backward_ms": self.backward_ms,
        }


def compute_bmd_rate(
    llr: torch.Tensor,
    c: torch.Tensor,
    num_bits_per_symbol: int = 6,
) -> torch.Tensor:
    """Compute Bit-Metric Decoding (BMD) rate in bits per channel use.

    Rate = (1.0 - BCE(llr, c) / ln(2)) * num_bits_per_symbol

    Parameters
    ----------
    llr : torch.Tensor
        Log-likelihood ratios on coded bits.
    c : torch.Tensor
        Ground truth transmitted bits {0, 1}.
    num_bits_per_symbol : int, default=6
        Modulation order bits per symbol.

    Returns
    -------
    torch.Tensor
        Scalar BMD rate tensor.
    """
    bce = F.binary_cross_entropy_with_logits(llr, c.float(), reduction="none")
    rate = (1.0 - bce.mean() / math.log(2.0)) * num_bits_per_symbol
    return rate


def configure_adamw_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.AdamW:
    """Configure AdamW optimizer with standard hyper-parameters."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
) -> Tuple[float, float]:
    """Calculate raw gradient norm, clip gradients, and return both norms.

    Parameters
    ----------
    model : nn.Module
        Neural network model with gradients computed.
    max_norm : float, default=1.0
        Maximum L2 norm threshold for gradient clipping.

    Returns
    -------
    Tuple[float, float]
        (raw_grad_norm, post_clipped_norm)
    """
    # Raw gradient norm before clipping
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0, 0.0

    raw_norm = torch.sqrt(sum(p.grad.data.norm(2) ** 2 for p in parameters)).item()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)

    # Post-clip norm
    post_norm = torch.sqrt(sum(p.grad.data.norm(2) ** 2 for p in parameters)).item()

    return raw_norm, post_norm

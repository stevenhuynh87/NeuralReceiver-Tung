"""Evaluation metrics for neural and baseline receivers.

Includes:
- Bit-Metric Decoding (BMD) rate: rbmd()
- Bit Error Rate (BER) and Block Error Rate (BLER)
- Calibration and gating analysis metrics
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


def rbmd(
    llr: Union[torch.Tensor, np.ndarray],
    b: Union[torch.Tensor, np.ndarray],
    num_bits_per_symbol: int = 6,
) -> float:
    """Compute Bit-Metric Decoding (BMD) rate in bits per channel use (bpcu).

    Formula:
        R_BMD = m * (1.0 - BCE(llr, b) / ln(2))
    where m = num_bits_per_symbol.

    Parameters
    ----------
    llr : torch.Tensor or np.ndarray
        Log-likelihood ratio predictions.
    b : torch.Tensor or np.ndarray
        Ground-truth binary transmitted bits in {0, 1}.
    num_bits_per_symbol : int, default=6
        Number of bits per constellation symbol (e.g., 6 for 64-QAM, 2 for QPSK).

    Returns
    -------
    float
        Achievable BMD rate. For perfect LLRs, R_BMD -> m. For zero LLRs, R_BMD -> 0.
    """
    if isinstance(llr, np.ndarray):
        llr_t = torch.from_numpy(llr).float()
    else:
        llr_t = llr.float()

    if isinstance(b, np.ndarray):
        b_t = torch.from_numpy(b).float()
    else:
        b_t = b.float()

    bce = F.binary_cross_entropy_with_logits(llr_t, b_t)
    rate = num_bits_per_symbol * (1.0 - bce.item() / math.log(2.0))
    return float(rate)


def compute_ber(llr: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Bit Error Rate (BER) from LLRs and ground-truth bits."""
    pred_bits = (llr > 0.0).long()
    gt_bits = b.long()
    return float((pred_bits != gt_bits).float().mean().item())


def compute_bler(llr: torch.Tensor, b: torch.Tensor, block_dim: int = -1) -> float:
    """Compute Block Error Rate (BLER) from LLRs and ground-truth bits."""
    pred_bits = (llr > 0.0).long()
    gt_bits = b.long()
    bit_errors = (pred_bits != gt_bits)
    # Block has error if any bit inside the block is erroneous
    dims_to_reduce = [i for i in range(1, bit_errors.ndim)]
    if not dims_to_reduce:
        block_errors = bit_errors
    else:
        block_errors = bit_errors.any(dim=dims_to_reduce[0])
        for d in dims_to_reduce[1:]:
            block_errors = block_errors.any(dim=1)
    return float(block_errors.float().mean().item())

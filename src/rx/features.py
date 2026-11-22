"""Doppler autocorrelation and correlation feature extraction module.

Provides:
- jakes_autocorrelation: Theoretical Jakes/Bessel autocorrelation J_0(2*pi*f_d*Delta_t)
  for a given user mobility velocity v.
- rho_hat: Doppler correlation coefficient estimation from DMRS pilot symbols
  with noise bias subtraction.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    import scipy.special as sp

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _bessel_j0_approx(x: float, max_terms: int = 30) -> float:
    """Evaluate 0th-order Bessel function of the first kind J_0(x) using Taylor series."""
    val = 0.0
    x_half_sq = (x / 2.0) ** 2
    term = 1.0
    val += term
    for m in range(1, max_terms):
        term *= -x_half_sq / (m * m)
        val += term
        if abs(term) < 1e-15:
            break
    return val


def jakes_autocorrelation(
    v: float,
    fc: float = 3.5e9,
    scs: float = 30000.0,
    delta_sym: int = 9,
    num_symbols_per_slot: int = 14,
    c: float = 3.0e8,
) -> float:
    """Calculate theoretical Jakes autocorrelation J_0(2*pi*f_d*Delta_t).

    Parameters
    ----------
    v : float
        User mobility speed in m/s.
    fc : float, optional
        Carrier frequency in Hz (default: 3.5e9).
    scs : float, optional
        Subcarrier spacing in Hz (default: 30000.0).
    delta_sym : int, optional
        Symbol spacing between two DMRS pilot symbols (default: 9, e.g. symbols 2 and 11).
    num_symbols_per_slot : int, optional
        Number of OFDM symbols per slot (default: 14).
    c : float, optional
        Speed of light in m/s (default: 3.0e8).

    Returns
    -------
    float
        Theoretical Doppler autocorrelation coefficient in [-1.0, 1.0].
    """
    if v < 0:
        raise ValueError(f"Velocity v must be non-negative, got {v}")

    # Maximum Doppler shift: f_d = (v * f_c) / c
    fd = float(v) * float(fc) / float(c)

    # Slot duration: T_slot = 15 / scs (in seconds, e.g. 0.5 ms for 30 kHz SCS)
    t_slot = 15.0 / float(scs)
    t_sym = t_slot / float(num_symbols_per_slot)
    delta_t = float(delta_sym) * t_sym

    arg = 2.0 * math.pi * fd * delta_t

    if _SCIPY_AVAILABLE:
        rho = float(sp.j0(arg))
    else:
        rho = _bessel_j0_approx(arg)

    return float(max(-1.0, min(1.0, rho)))


def _extract_pilot_symbols(
    H_ls: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Sequence[torch.Tensor]],
    pilot_indices: Tuple[int, int] = (2, 11),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract pilot channel responses H_0 and H_1 from various input tensor formats.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        H0, H1 standardized to 2D shape (B, N_elements).
    """
    t0, t1 = pilot_indices

    if isinstance(H_ls, (tuple, list)):
        if len(H_ls) != 2:
            raise ValueError(f"Expected sequence of 2 pilot tensors, got length {len(H_ls)}")
        H0_raw, H1_raw = H_ls[0], H_ls[1]
    elif not isinstance(H_ls, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor or tuple/list of tensors, got {type(H_ls)}")
    elif H_ls.ndim == 7:
        # Sionna format: [B, 1, rx_ant, 1, 1, T, F]
        H0_raw = H_ls[:, 0, :, 0, 0, t0, :]
        H1_raw = H_ls[:, 0, :, 0, 0, t1, :]
    elif H_ls.ndim == 6:
        # [B, 1, rx_ant, 1, T, F]
        H0_raw = H_ls[:, 0, :, 0, t0, :]
        H1_raw = H_ls[:, 0, :, 0, t1, :]
    elif H_ls.ndim == 5:
        # [B, 1, rx_ant, T, F] or [B, rx_ant, 1, T, F]
        if H_ls.shape[1] == 1:
            H0_raw = H_ls[:, 0, :, t0, :]
            H1_raw = H_ls[:, 0, :, t1, :]
        elif H_ls.shape[2] == 1:
            H0_raw = H_ls[:, :, 0, t0, :]
            H1_raw = H_ls[:, :, 0, t1, :]
        else:
            # Multi-tx: take stream 0
            H0_raw = H_ls[:, :, 0, t0, :]
            H1_raw = H_ls[:, :, 0, t1, :]
    elif H_ls.ndim == 4:
        # [2, B, rx_ant, F] or [B, rx_ant, 2, F] or [B, rx_ant, T, F]
        if H_ls.shape[0] == 2:
            H0_raw = H_ls[0]
            H1_raw = H_ls[1]
        elif H_ls.shape[2] == 2:
            H0_raw = H_ls[:, :, 0, :]
            H1_raw = H_ls[:, :, 1, :]
        else:
            H0_raw = H_ls[:, :, t0, :]
            H1_raw = H_ls[:, :, t1, :]
    elif H_ls.ndim == 3:
        # [2, B, F] or [B, 2, F]
        if H_ls.shape[0] == 2:
            H0_raw = H_ls[0]
            H1_raw = H_ls[1]
        elif H_ls.shape[1] == 2:
            H0_raw = H_ls[:, 0, :]
            H1_raw = H_ls[:, 1, :]
        else:
            raise ValueError(f"Cannot identify 2 pilot symbols from 3D tensor of shape {H_ls.shape}")
    else:
        raise ValueError(f"Unsupported H_ls tensor rank {H_ls.ndim} with shape {H_ls.shape}")

    batch_size = H0_raw.shape[0]
    H0 = H0_raw.reshape(batch_size, -1)
    H1 = H1_raw.reshape(batch_size, -1)

    return H0, H1


def rho_hat(
    H_ls: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Sequence[torch.Tensor]],
    no: Union[float, torch.Tensor] = 0.0,
    pilot_indices: Tuple[int, int] = (2, 11),
    debias: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Estimate Doppler autocorrelation coefficient rho_hat from DMRS pilot symbols.

    Extracts channel responses at two pilot symbols (H_0 and H_1) and computes:
        cov_{01} = (1 / N) * sum Re{ H_0^* * H_1 }
        P_0 = (1 / N) * sum |H_0|^2
        P_1 = (1 / N) * sum |H_1|^2

    With noise bias subtraction (debias=True):
        P_{0, debiased} = max(P_0 - N_0, eps)
        P_{1, debiased} = max(P_1 - N_0, eps)
        \\hat{\\rho} = \\frac{cov_{01}}{\\sqrt{P_{0, debiased} * P_{1, debiased}}}

    Parameters
    ----------
    H_ls : torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        Channel estimate at pilot symbols or full grid. Supports 4D [B, rx, T, F],
        5D [B, 1, rx, T, F], 7D [B, 1, rx, 1, 1, T, F], 2-pilot tensor [2, B, rx, F],
        or tuple (H_0, H_1).
    no : float or torch.Tensor, optional
        Noise power spectral density N0 (scalar or tensor of shape (B,)). Default 0.0.
    pilot_indices : Tuple[int, int], optional
        Symbol indices of the two DMRS pilot symbols (default: (2, 11)).
    debias : bool, optional
        Whether to subtract noise variance N0 from estimated power (default: True).
    eps : float, optional
        Small positive constant for numerical stability (default: 1e-12).

    Returns
    -------
    torch.Tensor
        1D tensor of shape (B,) containing Doppler correlation estimates in [-1.0, 1.0].
    """
    H0, H1 = _extract_pilot_symbols(H_ls, pilot_indices=pilot_indices)
    batch_size = H0.shape[0]

    # Cross-correlation: cov_01 = (1 / N) * sum Re(H0^* * H1)
    cov_01 = torch.mean((torch.conj(H0) * H1).real, dim=-1)  # [B]

    # Average powers
    p0 = torch.mean(torch.abs(H0) ** 2, dim=-1)  # [B]
    p1 = torch.mean(torch.abs(H1) ** 2, dim=-1)  # [B]

    # Convert noise variance to tensor [B]
    if isinstance(no, torch.Tensor):
        no_tensor = no.to(device=H0.device, dtype=p0.dtype).reshape(-1)
        if no_tensor.numel() == 1:
            no_tensor = no_tensor.expand(batch_size)
        elif no_tensor.shape[0] != batch_size:
            no_tensor = no_tensor.view(-1).mean().expand(batch_size)
    else:
        no_tensor = torch.full((batch_size,), float(no), device=H0.device, dtype=p0.dtype)

    if debias:
        p0_deb = torch.clamp(p0 - no_tensor, min=eps)
        p1_deb = torch.clamp(p1 - no_tensor, min=eps)
    else:
        p0_deb = torch.clamp(p0, min=eps)
        p1_deb = torch.clamp(p1, min=eps)

    denom = torch.sqrt(p0_deb * p1_deb)
    rho = cov_01 / (denom + eps)

    # Clamp in [-1.0, 1.0]
    rho_clamped = torch.clamp(rho, min=-1.0, max=1.0)
    return rho_clamped


def gate_features(
    y: torch.Tensor,
    h_ls: torch.Tensor,
    l_N: torch.Tensor,
    no: Union[float, torch.Tensor],
    pilot_indices: Tuple[int, int] = (2, 11),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Extract 5-dimensional gating feature vector phi for GateMLP.

    Feature vector components:
    1. log10(N0): Noise variance in log10 scale.
    2. rho_hat: Doppler temporal autocorrelation in [-1.0, 1.0].
    3. r_hat: Normalized pilot residual error.
    4. mean_abs_llr: Mean absolute LLR magnitude E[|l_N|].
    5. llr_entropy: Mean binary Shannon entropy H(l_N).

    Parameters
    ----------
    y : torch.Tensor
        Received signal grid of shape (B, Rx, T, F) or (B, T, F).
    h_ls : torch.Tensor
        LS channel estimate of shape (B, Rx, T, F) or (B, T, F).
    l_N : torch.Tensor
        Neural receiver LLRs of shape (B, T, F, K) or (B, ...).
    no : Union[float, torch.Tensor]
        Noise variance.
    pilot_indices : Tuple[int, int], default=(2, 11)
        OFDM symbol indices carrying pilots.
    eps : float, default=1e-8
        Numerical stability epsilon.

    Returns
    -------
    torch.Tensor
        Tensor of shape (B, 5) containing features [log_N0, rho_hat, r_hat, mean_abs_llr, llr_entropy].
    """
    batch_size = l_N.shape[0]

    # 1. log10(N0)
    if isinstance(no, torch.Tensor):
        no_tensor = no.to(device=l_N.device, dtype=torch.float32).reshape(-1)
        if no_tensor.numel() == 1:
            no_tensor = no_tensor.expand(batch_size)
    else:
        no_tensor = torch.full((batch_size,), float(no), device=l_N.device, dtype=torch.float32)
    feat_log_n0 = torch.log10(torch.clamp(no_tensor, min=1e-12))

    # 2. rho_hat
    feat_rho_hat = rho_hat(h_ls, no_tensor, pilot_indices=pilot_indices)

    # 3. Normalized pilot residual r_hat
    # Extract pilot symbols from y and h_ls
    if y.ndim == 4:
        p0_idx, p1_idx = pilot_indices
        y_p = y[:, :, [p0_idx, p1_idx], :]
        h_p = h_ls[:, :, [p0_idx, p1_idx], :]
        diff_p = y_p - h_p
        # Residual power normalized by total received pilot power
        res_power = torch.mean(torch.abs(diff_p) ** 2, dim=(1, 2, 3))
        sig_power = torch.mean(torch.abs(y_p) ** 2, dim=(1, 2, 3)) + eps
        feat_r_hat = torch.clamp(res_power / sig_power, min=0.0, max=10.0)
    else:
        feat_r_hat = torch.zeros(batch_size, device=l_N.device, dtype=torch.float32)

    # 4. mean_abs_llr: E[|l_N|]
    l_N_flat = l_N.reshape(batch_size, -1)
    feat_mean_abs_llr = torch.mean(torch.abs(l_N_flat), dim=-1)

    # 5. llr_entropy: H(l_N) = -p*log2(p) - (1-p)*log2(1-p)
    # p = sigmoid(l_N)
    p = torch.sigmoid(l_N_flat)
    p_clamped = torch.clamp(p, min=eps, max=1.0 - eps)
    binary_entropy = -(p_clamped * torch.log2(p_clamped) + (1.0 - p_clamped) * torch.log2(1.0 - p_clamped))
    feat_entropy = torch.mean(binary_entropy, dim=-1)

    # Stack into [B, 5]
    phi = torch.stack([
        feat_log_n0,
        feat_rho_hat,
        feat_r_hat,
        feat_mean_abs_llr,
        feat_entropy,
    ], dim=1)

    return phi

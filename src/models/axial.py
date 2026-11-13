"""Proposed Hybrid LMMSE + Factorized Axial Attention Neural Receiver.

Implements the proposed 6G neural receiver architecture:
- LMMSEFeatureExtractor: Computes pilot-aided single-tap soft equalized feature x_lmmse.
- PositionalEncoding2D: 2D learnable temporal and spectral positional embeddings.
- AxialAttentionBlock / FactorizedAxialAttention: Decoupled Time and Frequency
  Multi-Head Attention blocks with LayerNorm and Feed-Forward Networks.
- ProposedNeuralReceiver / NeuralReceiver_proposed: Full neural receiver taking
  received symbols y, LS channel estimates h_ls, and noise power N0.
"""

from __future__ import annotations

import math
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import load_config


class LMMSEFeatureExtractor(nn.Module):
    """Computes initial soft LMMSE equalized estimates as hybrid domain features."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        y: torch.Tensor,
        h_ls: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        """Equalize y using LS channel estimate h_ls and noise variance N0.

        Inputs:
            y:    [B, rx_ant, T, F]
            h_ls: [B, rx_ant, T, F]
            no:   [B, 1, 1, 1] or scalar
        Returns:
            x_lmmse: [B, 1, T, F] complex-valued equalized tensor.
        """
        # num = sum_ant (h* * y)
        num = torch.sum(torch.conj(h_ls) * y, dim=1, keepdim=True)
        # denom = sum_ant |h|^2 + N0
        denom = torch.sum(torch.abs(h_ls) ** 2, dim=1, keepdim=True) + no
        return num / (denom + 1e-8)


class PositionalEncoding2D(nn.Module):
    """2D Learnable Positional Encoding for Time (OFDM symbols) and Frequency (subcarriers)."""

    def __init__(
        self,
        d_model: int,
        max_time: int = 14,
        max_freq: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_time = max_time
        self.max_freq = max_freq
        self.time_embed = nn.Parameter(torch.randn(1, max_time, 1, d_model) * 0.02)
        self.freq_embed = nn.Parameter(torch.randn(1, 1, max_freq, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add time and frequency embeddings to input feature grid.

        Input:  [B, T, F, d_model]
        Output: [B, T, F, d_model]
        """
        return x + self.time_embed + self.freq_embed


class FactorizedAxialAttention(nn.Module):
    """Factorized Axial Attention layer decoupling Time-MHA and Frequency-MHA."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 2

        # Frequency attention (applied along subcarrier dimension)
        self.freq_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_freq = nn.LayerNorm(d_model)

        # Time attention (applied along OFDM symbol dimension)
        self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_time = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through factorized axial attention.

        Input:  [B, T, F, d_model]
        Output: [B, T, F, d_model]
        """
        B, T, F_dim, C = x.shape

        # 1. Frequency Attention (batch * time, freq, channels)
        x_freq = x.reshape(B * T, F_dim, C)
        attn_freq, _ = self.freq_attn(x_freq, x_freq, x_freq)
        x_freq = self.norm_freq(x_freq + attn_freq)
        x = x_freq.view(B, T, F_dim, C)

        # 2. Time Attention (batch * freq, time, channels)
        x_time = x.permute(0, 2, 1, 3).reshape(B * F_dim, T, C)
        attn_time, _ = self.time_attn(x_time, x_time, x_time)
        x_time = self.norm_time(x_time + attn_time)
        x = x_time.view(B, F_dim, T, C).permute(0, 2, 1, 3)

        # 3. FFN
        x = self.norm_ffn(x + self.ffn(x))
        return x


# Alias for compatibility with notebook naming
AxialAttentionBlock = FactorizedAxialAttention


class ProposedNeuralReceiver(nn.Module):
    """Proposed Hybrid LMMSE + Factorized Axial Attention Neural Receiver.

    Input features (11 channels standard, updated in New Code):
    - y_real [0, 1] (2 channels)
    - y_imag [0, 1] (2 channels)
    - h_ls_real [0, 1] (2 channels)
    - h_ls_imag [0, 1] (2 channels)
    - x_lmmse_real, x_lmmse_imag (2 channels)
    - log10(N0) (1 channel)
    Total = 11 channels (pilot_mask eliminated in New Code).
    """

    def __init__(
        self,
        d_model: int = 128,
        num_conv_channels: Optional[int] = None,
        num_bits_per_symbol: int = 6,
        num_layers: int = 6,
        nhead: int = 4,
        dim_feedforward: Optional[int] = None,
        in_channels: int = 11,
        max_time: int = 14,
        max_freq: int = 128,
        include_pilot_mask: bool = False,
    ):
        super().__init__()
        if num_conv_channels is not None:
            d_model = num_conv_channels

        self.d_model = d_model
        self.num_bits_per_symbol = num_bits_per_symbol
        self.num_layers = num_layers
        self.nhead = nhead
        self.max_time = max_time
        self.max_freq = max_freq
        self.include_pilot_mask = include_pilot_mask

        # If pilot mask is included (legacy 12-channel mode), adjust in_channels
        self.num_input_channels = 12 if include_pilot_mask else in_channels

        self.lmmse_extractor = LMMSEFeatureExtractor()
        self.input_proj = nn.Linear(self.num_input_channels, d_model)
        self.pos_encoder_2d = PositionalEncoding2D(d_model, max_time=max_time, max_freq=max_freq)

        ffn_dim = dim_feedforward if dim_feedforward is not None else d_model * 2
        self.axial_layers = nn.ModuleList([
            FactorizedAxialAttention(d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(d_model, num_bits_per_symbol)

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None) -> "ProposedNeuralReceiver":
        """Instantiate proposed neural receiver from single-source config."""
        if cfg is None:
            cfg = load_config()
        phy = cfg.get("phy", {})
        models = cfg.get("models", {}).get("proposed_axial", {})
        return cls(
            d_model=models.get("d_model", 128),
            num_bits_per_symbol=phy.get("modulation", {}).get("num_bits_per_symbol", 6),
            num_layers=models.get("num_layers", 6),
            nhead=models.get("num_heads", 4),
            dim_feedforward=models.get("dim_feedforward", 256),
            in_channels=models.get("in_channels", 11),
            max_time=models.get("max_time", 14),
            max_freq=models.get("max_freq", 128),
        )

    def forward(
        self,
        y: torch.Tensor,
        h_ls: torch.Tensor,
        no: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through proposed neural receiver.

        Parameters
        ----------
        y : torch.Tensor
            Received symbols [B, rx_ant, T, F].
        h_ls : torch.Tensor
            LS channel estimates mapped to full FFT grid [B, rx_ant, T, F].
        no : float or torch.Tensor
            Noise variance N0.

        Returns
        -------
        torch.Tensor
            LLR tensor of shape [B, T, F, num_bits_per_symbol].
        """
        if y.ndim == 5:
            y = y.squeeze(1)
        if h_ls.ndim == 5:
            h_ls = h_ls.squeeze(1)

        batch_size, rx_ant, time_dim, freq_dim = y.shape

        if not isinstance(no, torch.Tensor):
            no = torch.tensor(float(no), dtype=y.real.dtype, device=y.device)
        no = no.to(device=y.device, dtype=y.real.dtype)
        no_fmt = no.view(-1, 1, 1, 1).expand(batch_size, 1, 1, 1)

        # 1. Compute Hybrid LMMSE equalized feature
        x_lmmse = self.lmmse_extractor(y, h_ls, no_fmt)

        # 2. Decompose real/imaginary parts
        y_real = y.real
        y_imag = y.imag
        h_ls_real = h_ls.real
        h_ls_imag = h_ls.imag
        x_lmmse_real = x_lmmse.real
        x_lmmse_imag = x_lmmse.imag

        # 3. Expand log10 noise
        no_log = torch.log10(no_fmt).expand(batch_size, 1, time_dim, freq_dim)

        # 4. Feature concatenation
        features = [
            y_real[:, 0:1], y_real[:, 1:2],
            y_imag[:, 0:1], y_imag[:, 1:2],
            h_ls_real[:, 0:1], h_ls_real[:, 1:2],
            h_ls_imag[:, 0:1], h_ls_imag[:, 1:2],
            x_lmmse_real, x_lmmse_imag,
            no_log,
        ]

        # Optional legacy pilot mask (12th channel)
        if self.num_input_channels == 12:
            pilot_mask = torch.zeros(batch_size, 1, time_dim, freq_dim, device=y.device)
            pilot_mask[:, :, [2, 11], :] = 1.0
            features.append(pilot_mask)

        # Concatenate along channel dim: [B, in_channels, T, F]
        z = torch.cat(features, dim=1)

        # Permute to [B, T, F, in_channels] for Linear projection
        z = z.permute(0, 2, 3, 1)

        # Linear projection to d_model
        z = self.input_proj(z)

        # 2D Positional Encoding
        z = self.pos_encoder_2d(z)

        # Factorized Axial Attention layers
        for layer in self.axial_layers:
            z = layer(z)

        # Output projection to LLRs: [B, T, F, num_bits_per_symbol]
        return self.output_proj(z)


# Alias for backward compatibility with Colab notebook naming
NeuralReceiver_proposed = ProposedNeuralReceiver

"""Proposed Hybrid LMMSE + Factorized Axial Attention Neural Receiver.

Implements the proposed 6G neural receiver architecture from Thesis Folder - Core Data:
- LMMSEFeatureExtractor: Computes analytical LMMSE equalized soft symbol features.
- PositionalEncoding2D: 2D learned positional embeddings for Time-Frequency grid.
- AxialAttentionBlock / FactorizedAxialAttention: Frequency-axis and Time-axis
  Multi-Head Attention blocks with LayerNorm and Feed-Forward Networks.
- ProposedNeuralReceiver / NeuralReceiver_proposed: Full neural receiver taking
  received symbols y, LS channel estimates h_ls, and noise variance N0.
"""

from __future__ import annotations

import math
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import load_config


class LMMSEFeatureExtractor(nn.Module):
    """Computes analytical LMMSE soft-equalized QAM symbols.

    from SIMO received grid y, LS channel estimate h_ls, and noise variance N0.
    """

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
            y:    (B, Rx, T, F) complex
            h_ls: (B, Rx, T, F) complex
            no:   (B, 1, 1, 1) real noise variance
        Returns:
            x_lmmse: (B, 1, T, F) complex equalized tensor.
        """
        num = torch.sum(torch.conj(h_ls) * y, dim=1, keepdim=True)
        denom = torch.sum(torch.abs(h_ls) ** 2, dim=1, keepdim=True) + no
        return num / (denom + 1e-8)


class PositionalEncoding2D(nn.Module):
    """Learned 2D positional embedding for Time-Frequency OFDM Grid (B, T, F, C)."""

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

        Input:  (B, T, F, C)
        Output: (B, T, F, C)
        """
        time_dim = x.size(1)
        freq_dim = x.size(2)
        return x + self.time_embed[:, :time_dim, :, :] + self.freq_embed[:, :, :freq_dim, :]


class AxialAttentionBlock(nn.Module):
    """Factorized 2D Axial Self-Attention Block matching Thesis Folder - Core Data.

    Performs Frequency-axis attention followed by Time-axis attention and FFN.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 2

        # Frequency-Axis Self-Attention (Seq Len = F)
        self.freq_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_freq = nn.LayerNorm(d_model)

        # Time-Axis Self-Attention (Seq Len = T)
        self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_time = nn.LayerNorm(d_model)

        # Feed-Forward Network (FFN) with Dropout matching checkpoint
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through factorized axial attention.

        Input x shape: (B, T, F, C)
        """
        B, T, F_dim, C = x.shape

        # 1. Frequency-Domain Attention (across F for each T)
        x_freq = x.view(B * T, F_dim, C)
        attn_freq, _ = self.freq_attn(x_freq, x_freq, x_freq)
        x_freq = self.norm_freq(x_freq + attn_freq).view(B, T, F_dim, C)

        # 2. Time-Domain Attention (across T for each F)
        x_time = x_freq.permute(0, 2, 1, 3).reshape(B * F_dim, T, C)
        attn_time, _ = self.time_attn(x_time, x_time, x_time)
        x_time = attn_time.view(B, F_dim, T, C).permute(0, 2, 1, 3)
        x_time = self.norm_time(x_freq + x_time)

        # 3. Feed-Forward Network
        out = self.norm_ffn(x_time + self.ffn(x_time))
        return out


# Alias for backward compatibility
FactorizedAxialAttention = AxialAttentionBlock


class ProposedNeuralReceiver(nn.Module):
    """Hybrid LMMSE + Axial Self-Attention Neural Receiver (11 input channels)."""

    def __init__(
        self,
        num_conv_channels: int = 128,
        num_bits_per_symbol: int = 6,
        nhead: int = 4,
        num_layers: int = 6,
        max_time: int = 14,
        max_freq: int = 128,
        d_model: Optional[int] = None,
        dim_feedforward: Optional[int] = None,
        in_channels: int = 11,
        include_pilot_mask: bool = False,
    ):
        super().__init__()
        if d_model is not None:
            num_conv_channels = d_model
        d_model = num_conv_channels

        self._num_bits_per_symbol = num_bits_per_symbol
        self.d_model = d_model
        self.num_layers = num_layers
        self.nhead = nhead
        self.max_time = max_time
        self.max_freq = max_freq
        self.include_pilot_mask = include_pilot_mask

        # LMMSE Analytical Feature Generator
        self.lmmse_extractor = LMMSEFeatureExtractor()

        num_input_channels = 12 if include_pilot_mask else in_channels
        self.input_proj = nn.Linear(num_input_channels, d_model)

        self.pos_encoder_2d = PositionalEncoding2D(d_model, max_time=max_time, max_freq=max_freq)

        ffn_dim = dim_feedforward if dim_feedforward is not None else d_model * 2
        self.axial_layers = nn.ModuleList([
            AxialAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim)
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
            num_conv_channels=models.get("d_model", 128),
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

        Inputs:
            y, h_ls: (B, Rx, T, F) complex
            no: (B, 1, 1, 1) or scalar real noise variance
        Returns:
            z: (B, T, F, num_bits_per_symbol)
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

        # 1. Compute LMMSE Equalized Soft Symbol Feature
        x_lmmse = self.lmmse_extractor(y, h_ls, no_fmt)

        # 2. Extract Real & Imaginary Features
        y_real, y_imag = y.real, y.imag
        h_ls_real, h_ls_imag = h_ls.real, h_ls.imag
        x_lmmse_real, x_lmmse_imag = x_lmmse.real, x_lmmse.imag

        # Expand Noise Variance to 4D Grid
        no_log = torch.log10(no_fmt).expand(batch_size, 1, time_dim, freq_dim)

        # 3. Concatenate Features along Channel Dim (Dim 1)
        features = [
            y_real[:, 0:1], y_real[:, 1:2],
            y_imag[:, 0:1], y_imag[:, 1:2],
            h_ls_real[:, 0:1], h_ls_real[:, 1:2],
            h_ls_imag[:, 0:1], h_ls_imag[:, 1:2],
            x_lmmse_real, x_lmmse_imag,
            no_log,
        ]

        if self.include_pilot_mask or features[0].shape[1] == 12:
            pilot_mask = torch.zeros(batch_size, 1, time_dim, freq_dim, device=y.device)
            pilot_mask[:, :, [2, 11], :] = 1.0
            features.append(pilot_mask)

        z = torch.cat(features, dim=1)

        # 4. Project to Feature Dimension d_model
        z = z.permute(0, 2, 3, 1)
        z = self.input_proj(z)
        z = self.pos_encoder_2d(z)

        # 5. Process through Factorized Axial Attention Layers
        for layer in self.axial_layers:
            z = layer(z)

        # 6. Output Projection to LLRs
        z = self.output_proj(z)

        return z


# Alias for backward compatibility with notebook naming
NeuralReceiver_proposed = ProposedNeuralReceiver

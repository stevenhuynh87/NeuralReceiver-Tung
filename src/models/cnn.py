"""CNN-ResNet baseline neural receiver module for OFDM SIMO systems.

Implements DeepRx architecture with residual blocks and 2D LayerNorm:
- ResidualBlock: 2-layer convolutional residual block with LayerNorm and ReLU.
- CNNNeuralReceiver / NeuralReceiver_CNN: Full receiver network taking received
  symbols y and noise variance N0, outputting log-likelihood ratios (LLRs).
"""

from __future__ import annotations

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import load_config


class ResidualBlock(nn.Module):
    """Residual convolutional block with LayerNorm and ReLU activations.

    Input shape: [batch_size, num_conv_channels, num_ofdm_symbols, num_subcarriers]
    Output shape: [batch_size, num_conv_channels, num_ofdm_symbols, num_subcarriers]
    """

    def __init__(
        self,
        num_conv_channels: int,
        num_ofdm_symbols: int = 14,
        fft_size: int = 128,
    ):
        super().__init__()
        self.num_conv_channels = num_conv_channels
        self.num_ofdm_symbols = num_ofdm_symbols
        self.fft_size = fft_size

        self._layer_norm_1 = nn.LayerNorm([num_conv_channels, num_ofdm_symbols, fft_size])
        self._conv_1 = nn.Conv2d(
            in_channels=num_conv_channels,
            out_channels=num_conv_channels,
            kernel_size=(3, 3),
            padding="same",
        )
        self._layer_norm_2 = nn.LayerNorm([num_conv_channels, num_ofdm_symbols, fft_size])
        self._conv_2 = nn.Conv2d(
            in_channels=num_conv_channels,
            out_channels=num_conv_channels,
            kernel_size=(3, 3),
            padding="same",
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        x = self._layer_norm_1(inputs)
        x = F.relu(x)
        x = self._conv_1(x)
        x = self._layer_norm_2(x)
        x = F.relu(x)
        x = self._conv_2(x)
        return inputs + x


class CNNNeuralReceiver(nn.Module):
    """Baseline CNN-ResNet Neural Receiver (DeepRx).

    Takes received OFDM signal y and noise power N0, concatenates 5 input channels:
    [y_real (rx0, rx1), y_imag (rx0, rx1), log10(N0)], and outputs LLRs.

    Supports both 4-residual-block architecture (standardized in New Code)
    and configurable num_blocks (e.g. 11 blocks).
    """

    def __init__(
        self,
        num_conv_channels: int = 128,
        num_bits_per_symbol: int = 6,
        num_blocks: int = 4,
        num_rx_ant: int = 2,
        num_ofdm_symbols: int = 14,
        fft_size: int = 128,
    ):
        super().__init__()
        self.num_conv_channels = num_conv_channels
        self.num_bits_per_symbol = num_bits_per_symbol
        self.num_blocks = num_blocks
        self.num_rx_ant = num_rx_ant
        self.num_ofdm_symbols = num_ofdm_symbols
        self.fft_size = fft_size

        # Input channels: 2 * num_rx_ant (real + imag) + 1 (noise log10) = 5
        self.num_input_channels = 2 * num_rx_ant + 1

        self._input_conv = nn.Conv2d(
            in_channels=self.num_input_channels,
            out_channels=num_conv_channels,
            kernel_size=(3, 3),
            padding="same",
        )

        # Residual blocks: if num_blocks == 4, expose explicit attributes _res_block_1..4
        # to match exact state_dict keys of trained checkpoints and New Code notebooks
        if num_blocks == 4:
            self._res_block_1 = ResidualBlock(num_conv_channels, num_ofdm_symbols, fft_size)
            self._res_block_2 = ResidualBlock(num_conv_channels, num_ofdm_symbols, fft_size)
            self._res_block_3 = ResidualBlock(num_conv_channels, num_ofdm_symbols, fft_size)
            self._res_block_4 = ResidualBlock(num_conv_channels, num_ofdm_symbols, fft_size)
            self._blocks = None
        else:
            self._blocks = nn.ModuleList([
                ResidualBlock(num_conv_channels, num_ofdm_symbols, fft_size)
                for _ in range(num_blocks)
            ])

        self._output_conv = nn.Conv2d(
            in_channels=num_conv_channels,
            out_channels=num_bits_per_symbol,
            kernel_size=(3, 3),
            padding="same",
        )

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None) -> "CNNNeuralReceiver":
        """Instantiate CNN receiver from single-source config."""
        if cfg is None:
            cfg = load_config()
        phy = cfg.get("phy", {})
        models = cfg.get("models", {}).get("baseline_cnn", {})
        return cls(
            num_conv_channels=models.get("num_conv_channels", 128),
            num_bits_per_symbol=phy.get("modulation", {}).get("num_bits_per_symbol", 6),
            num_blocks=models.get("num_blocks", 4),
            num_rx_ant=phy.get("num_rx_ant", 2),
            num_ofdm_symbols=phy.get("num_ofdm_symbols", 14),
            fft_size=phy.get("fft_size", 128),
        )

    def forward(self, y: torch.Tensor, no: Union[float, torch.Tensor]) -> torch.Tensor:
        """Forward pass through CNN receiver.

        Parameters
        ----------
        y : torch.Tensor
            Received symbols tensor of shape [batch_size, num_rx_ant, num_ofdm_symbols, fft_size]
            or [batch_size, 1, num_rx_ant, num_ofdm_symbols, fft_size].
        no : float or torch.Tensor
            Noise variance N0.

        Returns
        -------
        torch.Tensor
            LLR tensor of shape [batch_size, num_ofdm_symbols, fft_size, num_bits_per_symbol].
        """
        if y.ndim == 5:
            y = y.squeeze(1)

        batch_size = y.shape[0]
        y_real = y.real
        y_imag = y.imag

        # Format noise N0 into 4D tensor
        if not isinstance(no, torch.Tensor):
            no = torch.tensor(float(no), dtype=y.real.dtype, device=y.device)
        no = no.to(device=y.device, dtype=y.real.dtype)
        no_4d = no.view(-1, 1, 1, 1).expand(batch_size, 1, self.num_ofdm_symbols, self.fft_size)
        no_log = torch.log10(no_4d)

        # Concatenate: [B, 5, T, F]
        z = torch.cat([y_real, y_imag, no_log], dim=1)

        # Feature processing
        z = self._input_conv(z)
        if self._blocks is not None:
            for block in self._blocks:
                z = block(z)
        else:
            z = self._res_block_1(z)
            z = self._res_block_2(z)
            z = self._res_block_3(z)
            z = self._res_block_4(z)

        z = self._output_conv(z)

        # Permute to [batch_size, num_ofdm_symbols, fft_size, num_bits_per_symbol]
        return z.permute(0, 2, 3, 1)


# Alias for backward compatibility with Colab notebook naming
NeuralReceiver_CNN = CNNNeuralReceiver
NeuralReceiver = CNNNeuralReceiver

"""Unit tests for neural receiver architectures in src/models."""

import pytest
import torch

from src.models.cnn import CNNNeuralReceiver, ResidualBlock
from src.models.axial import (
    FactorizedAxialAttention,
    LMMSEFeatureExtractor,
    PositionalEncoding2D,
    ProposedNeuralReceiver,
)


def test_residual_block():
    """Verify ResidualBlock tensor dimensions and residual connection."""
    block = ResidualBlock(num_conv_channels=64, num_ofdm_symbols=14, fft_size=128)
    x = torch.randn(2, 64, 14, 128)
    out = block(x)
    assert out.shape == (2, 64, 14, 128)
    assert torch.all(torch.isfinite(out))


def test_cnn_neural_receiver():
    """Verify CNNNeuralReceiver (4 blocks standard from New Code and 11 blocks)."""
    # 4-block receiver (New Code architecture)
    model_4 = CNNNeuralReceiver(
        num_conv_channels=64,
        num_bits_per_symbol=6,
        num_blocks=4,
        num_rx_ant=2,
        num_ofdm_symbols=14,
        fft_size=128,
    )
    y = torch.randn(2, 2, 14, 128, dtype=torch.complex64)
    no = torch.tensor(0.01)
    llr = model_4(y, no)
    assert llr.shape == (2, 14, 128, 6)
    assert torch.all(torch.isfinite(llr))

    # Test from_config
    model_cfg = CNNNeuralReceiver.from_config()
    assert model_cfg.num_bits_per_symbol == 6
    assert model_cfg.num_blocks == 4


def test_lmmse_feature_extractor():
    """Verify LMMSEFeatureExtractor equalization formula."""
    extractor = LMMSEFeatureExtractor()
    y = torch.randn(2, 2, 14, 128, dtype=torch.complex64)
    h_ls = torch.randn(2, 2, 14, 128, dtype=torch.complex64)
    no = torch.tensor([0.01, 0.01]).view(2, 1, 1, 1)

    x_lmmse = extractor(y, h_ls, no)
    assert x_lmmse.shape == (2, 1, 14, 128)
    assert torch.all(torch.isfinite(x_lmmse.real))
    assert torch.all(torch.isfinite(x_lmmse.imag))


def test_positional_encoding_2d():
    """Verify PositionalEncoding2D additive embeddings."""
    pos_enc = PositionalEncoding2D(d_model=64, max_time=14, max_freq=128)
    x = torch.zeros(2, 14, 128, 64)
    out = pos_enc(x)
    assert out.shape == (2, 14, 128, 64)
    assert not torch.allclose(out, torch.zeros_like(out))


def test_factorized_axial_attention():
    """Verify decoupled Time and Frequency self-attention."""
    layer = FactorizedAxialAttention(d_model=64, nhead=4, dim_feedforward=128)
    x = torch.randn(2, 14, 128, 64)
    out = layer(x)
    assert out.shape == (2, 14, 128, 64)
    assert torch.all(torch.isfinite(out))


def test_proposed_neural_receiver():
    """Verify ProposedNeuralReceiver with 11 channels (New Code) and 12 channels (legacy)."""
    # 11-channel standard model
    model_11 = ProposedNeuralReceiver(
        d_model=64,
        num_bits_per_symbol=6,
        num_layers=2,
        nhead=4,
        dim_feedforward=128,
        in_channels=11,
        include_pilot_mask=False,
    )
    assert model_11.input_proj.in_features == 11

    y = torch.randn(2, 2, 14, 128, dtype=torch.complex64)
    h_ls = torch.randn(2, 2, 14, 128, dtype=torch.complex64)
    no = torch.tensor(0.01)

    llr = model_11(y, h_ls, no)
    assert llr.shape == (2, 14, 128, 6)
    assert torch.all(torch.isfinite(llr))

    # Gradient flow test
    loss = llr.sum()
    loss.backward()
    assert model_11.input_proj.weight.grad is not None
    assert torch.all(torch.isfinite(model_11.input_proj.weight.grad))

    # 12-channel legacy model
    model_12 = ProposedNeuralReceiver(
        d_model=64,
        num_bits_per_symbol=6,
        num_layers=1,
        nhead=4,
        include_pilot_mask=True,
    )
    assert model_12.input_proj.in_features == 12
    llr_12 = model_12(y, h_ls, no)
    assert llr_12.shape == (2, 14, 128, 6)

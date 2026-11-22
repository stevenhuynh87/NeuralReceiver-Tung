"""Smoke test suite for nrx-bounded infrastructure."""

import numpy as np
import torch
import yaml

from src import load_config, set_all_seeds
import src.config
import src.utils


def test_imports():
    """Verify that all core project modules and libraries import cleanly."""
    assert src is not None
    assert src.config is not None
    assert src.utils is not None
    assert torch is not None
    assert np is not None
    assert yaml is not None


def test_config_validity():
    """Verify single-source config/config.yaml integrity and key PHY/model parameters."""
    cfg = load_config()
    assert isinstance(cfg, dict), "Config should load as a dictionary"
    assert cfg.get("seed") == 42, f"Expected seed=42, got {cfg.get('seed')}"

    phy = cfg.get("phy", {})
    assert phy.get("fft_size") == 128, f"Expected fft_size=128, got {phy.get('fft_size')}"
    assert phy.get("num_ofdm_symbols") == 14, f"Expected num_ofdm_symbols=14, got {phy.get('num_ofdm_symbols')}"

    models = cfg.get("models", {})
    axial = models.get("proposed_axial", {})
    assert axial.get("d_model") == 128, f"Expected d_model=128, got {axial.get('d_model')}"
    assert axial.get("num_layers") == 6, f"Expected num_layers=6, got {axial.get('num_layers')}"
    assert axial.get("in_channels") in [11, 12], f"Expected in_channels 11 or 12, got {axial.get('in_channels')}"

    channel = cfg.get("channel", {})
    assert channel.get("cdl_model") == "C", f"Expected cdl_model='C', got {channel.get('cdl_model')}"
    assert abs(channel.get("delay_spread") - 100e-9) < 1e-12, f"Expected delay_spread=100e-9, got {channel.get('delay_spread')}"

    eval_cfg = cfg.get("evaluation", {})
    assert eval_cfg.get("ebno_db_step") in [0.5, 1.0], f"Expected ebno_db_step in [0.5, 1.0], got {eval_cfg.get('ebno_db_step')}"


def test_seed_reproducibility():
    """Verify deterministic RNG behavior across NumPy and PyTorch."""
    # Run 1
    set_all_seeds(1234)
    np_sample_1 = np.random.randn(10, 10)
    torch_sample_1 = torch.randn(10, 10)

    # Run 2
    set_all_seeds(1234)
    np_sample_2 = np.random.randn(10, 10)
    torch_sample_2 = torch.randn(10, 10)

    assert np.allclose(np_sample_1, np_sample_2), "NumPy RNG must be deterministic"
    assert torch.allclose(torch_sample_1, torch_sample_2), "PyTorch RNG must be deterministic"


def test_plots_artifacts():
    """Verify presence and completeness of official thesis result plots across categorized folders."""
    from pathlib import Path
    plots_dir = Path("plots") if Path("plots").exists() else Path(__file__).resolve().parent.parent / "plots"
    assert plots_dir.exists() and plots_dir.is_dir(), "plots/ directory must exist"
    assert (plots_dir / "benchmark").exists(), "plots/benchmark directory must exist"
    assert (plots_dir / "evaluation").exists(), "plots/evaluation directory must exist"
    png_files = list(plots_dir.rglob("*.png"))
    assert len(png_files) >= 19, f"Expected at least 19 result plots, found {len(png_files)}"
    complexity_pdf = plots_dir / "benchmark" / "complexity" / "Data" / "Complexity.pdf"
    assert complexity_pdf.exists(), "plots/benchmark/complexity/Data/Complexity.pdf must exist"
    assert (plots_dir / "benchmark" / "complexity" / "Figures").exists(), "benchmark/complexity/Figures must exist"
    assert (plots_dir / "evaluation" / "40mps" / "Figures").exists(), "evaluation/40mps/Figures must exist"
    assert (plots_dir / "evaluation" / "40mps" / "Data").exists(), "evaluation/40mps/Data must exist"
    readme = plots_dir / "README.md"
    assert readme.exists(), "plots/README.md catalog must exist"


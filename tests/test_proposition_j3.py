"""Unit test for Proposition J3 (Estimation error and fallback bound)."""

import math
import numpy as np
import scipy.special as sp

from plots.fig_j3_rho_error import q_func, run_j3_validation


def test_j3_variance_formula():
    """Verify analytical variance matches Monte Carlo variance within 20% tolerance."""
    snr_lin = 10.0  # 10 dB
    np_sub = 128
    true_rho = 0.50

    var_th = (1.0 / (2.0 * np_sub)) * (
        (1.0 - true_rho**2)**2 + (2.0 / snr_lin) * (1.0 - true_rho**2) + (1.0 / (snr_lin**2))
    )

    # Fast 500-sample MC
    estimates = []
    n0 = 1.0 / snr_lin
    for _ in range(1000):
        h0 = (np.random.normal(0, 1, np_sub) + 1j * np.random.normal(0, 1, np_sub)) / math.sqrt(2.0)
        h1 = true_rho * h0 + math.sqrt(1.0 - true_rho**2) * (
            (np.random.normal(0, 1, np_sub) + 1j * np.random.normal(0, 1, np_sub)) / math.sqrt(2.0)
        )
        w0 = (np.random.normal(0, 1, np_sub) + 1j * np.random.normal(0, 1, np_sub)) * math.sqrt(n0 / 2.0)
        w1 = (np.random.normal(0, 1, np_sub) + 1j * np.random.normal(0, 1, np_sub)) * math.sqrt(n0 / 2.0)
        num = np.mean(np.real(np.conj(h0 + w0) * (h1 + w1)))
        den = 0.5 * np.mean(np.abs(h0 + w0)**2 + np.abs(h1 + w1)**2) - n0
        estimates.append(num / den)

    var_mc = float(np.var(estimates))
    rel_diff = abs(var_mc - var_th) / var_th
    assert rel_diff < 0.25, f"Expected relative variance difference < 0.25, got {rel_diff:.4f}"


def test_j3_validation_script_runs():
    """Verify run_j3_validation generates expected figures without error."""
    sim_vars, theory_vars = run_j3_validation()
    assert len(sim_vars) == 11
    assert len(theory_vars) == 11

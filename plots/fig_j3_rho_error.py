"""Proposition J3 Verification: Theory vs. Monte Carlo Simulation.

Verifies:
1. Estimation variance sigma_rho^2(SNR) matches theoretical formula.
2. Missed fallback probability P_miss matches Q-function analytical bound.
Generates plots/Figures/fig_j3_rho_error.png and .pdf.
"""

import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp


def q_func(x):
    return 0.5 * sp.erfc(x / math.sqrt(2.0))


def run_j3_validation():
    snr_db_range = np.linspace(0.0, 20.0, 11)
    np_subcarriers = 128  # N_p = N_rx * F_pilot
    v_ood = 70.0  # OOD speed (rho(v) < rho_lo)
    rho_lo = 0.560

    # True rho at 70 m/s
    fd = (v_ood * 3.5e9) / 3.0e8
    dt = 9.0 * (0.5e-3 / 14.0)
    true_rho = float(sp.jv(0, 2 * math.pi * fd * dt))

    mc_trials = 5000
    sim_variances = []
    theory_variances = []
    sim_pmiss = []
    theory_pmiss = []

    for snr_db in snr_db_range:
        snr_lin = 10 ** (snr_db / 10.0)
        n0 = 1.0 / snr_lin

        # Theoretical variance
        var_th = (1.0 / (2.0 * np_subcarriers)) * (
            (1.0 - true_rho**2)**2 + (2.0 / snr_lin) * (1.0 - true_rho**2) + (1.0 / (snr_lin**2))
        )
        sigma_th = math.sqrt(var_th)
        theory_variances.append(var_th)

        # Theoretical P_miss = Q((rho_lo - true_rho) / sigma_th)
        arg = (rho_lo - true_rho) / sigma_th
        p_th = float(q_func(arg))
        theory_pmiss.append(max(p_th, 1e-6))

        # Monte Carlo trials
        estimates = []
        for _ in range(mc_trials):
            # True channel
            h0 = (np.random.normal(0, 1, np_subcarriers) + 1j * np.random.normal(0, 1, np_subcarriers)) / math.sqrt(2.0)
            h1 = true_rho * h0 + math.sqrt(max(0.0, 1.0 - true_rho**2)) * (
                (np.random.normal(0, 1, np_subcarriers) + 1j * np.random.normal(0, 1, np_subcarriers)) / math.sqrt(2.0)
            )

            # Noise
            w0 = (np.random.normal(0, 1, np_subcarriers) + 1j * np.random.normal(0, 1, np_subcarriers)) * math.sqrt(n0 / 2.0)
            w1 = (np.random.normal(0, 1, np_subcarriers) + 1j * np.random.normal(0, 1, np_subcarriers)) * math.sqrt(n0 / 2.0)

            y0 = h0 + w0
            y1 = h1 + w1

            num = np.mean(np.real(np.conj(y0) * y1))
            den = 0.5 * np.mean(np.abs(y0)**2 + np.abs(y1)**2) - n0
            rho_est = num / max(1e-4, den)
            estimates.append(rho_est)

        estimates = np.array(estimates)
        sim_var = float(np.var(estimates))
        sim_variances.append(sim_var)

        # Fraction of times rho_est > rho_lo
        misses = np.mean(estimates > rho_lo)
        sim_pmiss.append(max(float(misses), 1e-6))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

    # Subplot 1: Variance vs SNR
    axes[0].semilogy(snr_db_range, theory_variances, "k-", linewidth=2, label=r"Analytical Theory $\sigma_\rho^2$ (Prop. J3)")
    axes[0].semilogy(snr_db_range, sim_variances, "ro", markersize=6, label=r"Monte Carlo Simulation ($N_p=128$)")
    axes[0].set_xlabel("Pilot SNR (dB)", fontsize=9)
    axes[0].set_ylabel(r"Estimation Variance $\mathrm{Var}(\hat{\rho})$", fontsize=9)
    axes[0].set_title(r"(a) Doppler Correlation Estimation Variance $\sigma_\rho^2$", fontsize=10)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8.5)

    # Subplot 2: Missed Fallback Probability
    axes[1].semilogy(snr_db_range, theory_pmiss, "b-", linewidth=2, label=r"Analytical $Q$-Bound $P_{\mathrm{miss}}$")
    axes[1].semilogy(snr_db_range, sim_pmiss, "ms", markersize=6, label="Simulated Missed Fallback Rate")
    axes[1].set_xlabel("Pilot SNR (dB)", fontsize=9)
    axes[1].set_ylabel(r"Missed Fallback Probability $P_{\mathrm{miss}}$", fontsize=9)
    axes[1].set_title(r"(b) Probability of Missed Fallback at $v = 70$ m/s", fontsize=10)
    axes[1].set_ylim([1e-6, 1.0])
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    out_dir = Path("plots/Figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "fig_j3_rho_error.png", dpi=300)
    plt.savefig(out_dir / "fig_j3_rho_error.pdf")
    plt.close()

    print("Saved Proposition J3 validation plots to plots/Figures/fig_j3_rho_error.png and .pdf")
    return sim_variances, theory_variances


if __name__ == "__main__":
    run_j3_validation()

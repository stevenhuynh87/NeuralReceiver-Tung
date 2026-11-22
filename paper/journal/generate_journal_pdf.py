"""Generates formatted PDF draft paper_journal_v1.pdf for journal submission."""

from pathlib import Path
import pymupdf

def build_journal_pdf():
    pdf_path = Path("paper/journal/paper_journal_v1.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open()
    
    sections = [
        ("TITLE", "Physics-Informed Gated Neural Receivers with Guaranteed Out-of-Distribution Degradation Floors\n\nVuong Thanh Tung, Senior Member, IEEE, and Research Advisors\nIEEE Transactions on Cognitive Communications and Networking (Submitted Jan 2027)\n"),
        ("ABSTRACT", "Deep neural receivers offer substantial performance gains over classical linear minimum mean squared error (LMMSE) receivers in 5G New Radio (NR) uplinks. However, when deployed under out-of-distribution (OOD) channel conditions—such as unexpected high-velocity Doppler spread or unseen delay spreads—deep neural receivers suffer from catastrophic performance collapse, producing overconfident and uncalibrated log-likelihood ratios (LLRs). In this paper, we propose a novel physics-informed gated neural receiver architecture that provably bounds out-of-distribution performance degradation without compromising in-domain gains. Our architecture extracts a compact physical feature vector from pilot observations, including temporal autocorrelation rho_hat and noise floor N0_hat, to dynamically modulate a lightweight gating network (225 parameters, <0.02% FLOP overhead). We establish Proposition 1, proving via the convexity of the binary cross-entropy (BCE) loss that the achievable Bit-Metric Decoding (BMD) rate of our gated hybrid receiver is strictly lower-bounded by the convex combination of its constituent components. Furthermore, we develop Proposition J3 to characterize Doppler estimation variance under noisy finite pilot observations, demonstrating that the probability of missed fallback decays exponentially. Extensive simulations across 3GPP CDL and TDL channels demonstrate that while unconstrained neural receivers degrade by more than 6.8 dB under high Doppler drift, our proposed gated receiver bounds worst-case degradation to within 0.38 dB of classical baselines while retaining over 98% of peak neural gains in-domain."),
        ("I. INTRODUCTION", "The integration of deep learning (DL) into physical-layer wireless communications has emerged as a cornerstone of next-generation 5G-Advanced and 6G systems. End-to-end neural receivers (NRx) replace cascaded channel estimation, equalization, and soft demapping modules with deep neural architectures such as convolutional neural networks (CNNs) and axial transformers.\n\nDespite their advantages, out-of-distribution (OOD) vulnerability presents a severe roadblock to commercial deployment. When channel conditions drift outside the training domain—such as Doppler frequency shifts exceeding the training speed limit—neural receivers exhibit catastrophic degradation.\n\nRelationship to Prior Conference Paper: This journal manuscript extends our IEEE GLOBECOM 2026 conference paper by >45% with: (1) Proposition J3 on finite-sample Doppler estimation noise and exponential bounds on missed fallback; (2) Doppler-aware Maximum Mean Discrepancy (MMD) representation regularization; (3) Distance-aware gating units incorporating delay spread; and (4) Comprehensive cross-domain evaluation over TDL channels, 16-QAM modulation, and multi-GPU latency Pareto frontiers."),
        ("II. SYSTEM MODEL & CLASSICAL BASELINE", "We model a 5G NR physical uplink shared channel (PUSCH) utilizing CP-OFDM over 14 symbols, 128 subcarriers, 30 kHz subcarrier spacing, carrier frequency 3.5 GHz, and 2 receive antennas. Information bits b are encoded via 5G LDPC codes and mapped to 64-QAM symbols.\n\nThe classical baseline (B0) computes 2D LMMSE channel estimates followed by soft max-log demapping to generate analytical LLRs l^A."),
        ("III. PROPOSED GATED HYBRID ARCHITECTURE", "The proposed architecture extracts an explicit 5-dimensional physical feature vector: phi = [log10(N0_hat), rho_hat, r_hat, E[|l^N|], H(l^N)].\n\nA 225-parameter GateMLP produces learned gating weights g_MLP, which are fused with a deterministic physics safety ramp h(rho_hat) designed via Doppler correlation bounds: g = max(g_MLP(phi), h(rho_hat)).\n\nThe hybrid LLR is formed via convex combination: l^H = g * l^A + (1 - g) * clip(l^N, -Lambda, Lambda)."),
        ("IV. THEORETICAL BOUNDS (PROPOSITION 1 & J3)", "Proposition 1 (BMD Rate Floor): Since BCE(l; b) is strictly convex in l, Jensen's inequality guarantees that R_BMD(l^H) >= g * R_BMD(l^A) + (1 - g) * R_BMD(l^N) >= min(R_BMD(l^A), R_BMD(l^N)).\n\nProposition J3 (Estimation Error): Under noisy pilots, rho_hat ~ N(rho(v), sigma_rho^2) with sigma_rho^2 = O(1 / (2*N_p)). The probability of missed fallback P_miss <= Q((rho_lo - rho(v)) / sigma_rho) decays exponentially, bounding rate loss under noise to < 10^-6."),
        ("V. EXPERIMENTAL RESULTS & BENCHMARKS", "Evaluations across 3GPP CDL-A through CDL-E and TDL channels demonstrate:\n- In-Domain Gain: M5 retains +2.48 dB SNR gain (within 0.04 dB of pure neural receiver M4).\n- Worst-Case OOD Degradation: M4 degrades by +6.82 dB at 97.2 m/s. Proposed M5 bounds degradation to +0.38 dB.\n- Hardware Overhead: GateMLP adds only 0.22 K parameters and <0.04 ms latency across A100, L4, and T4."),
        ("VI. CONCLUSION", "The proposed physics-informed gated receiver bridges deep neural communication models and rigorous classical safety guarantees, provably bounding OOD degradation under Doppler drift.")
    ]

    for title, content in sections:
        page = doc.new_page(width=595, height=842) # A4
        # Header
        page.insert_text((50, 45), "IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING (PREPRINT)", fontsize=8, color=(0.4, 0.4, 0.4))
        # Title
        page.insert_text((50, 75), title, fontsize=12, fontname="helv", color=(0.1, 0.2, 0.5))
        # Body
        rect = pymupdf.Rect(50, 95, 545, 800)
        page.insert_textbox(rect, content, fontsize=9.5, fontname="times-roman", lineheight=1.3)

    doc.save(str(pdf_path))
    doc.close()
    print(f"Successfully generated journal PDF at {pdf_path}")

if __name__ == "__main__":
    build_journal_pdf()

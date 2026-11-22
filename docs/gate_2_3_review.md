# Gate 2 (Prototype) & Gate 3 (Computational) Review Record (`docs/gate_2_3_review.md`)

**Date**: 22/11/2026  
**Project**: Bounded Neural Receiver (`nrx-bounded`)  
**Sprint**: S2 Milestone (Fair Baselines & Theory Draft)  
**Decision**: 🟢 **GREEN (PROCEED WITH STAGE 2 GATING ARCHITECTURE)**  

---

## 1. Review Executive Summary

The combined Gate 2 (Prototype) & Gate 3 (Computational) review validates:
1. Baseline convergence and reproducibility for M1 (Genie LMMSE), M2 (LS-LMMSE), M3 (CNN-ResNet DeepRx), and M4 (Proposed Axial Attention Neural Receiver).
2. The mathematical correctness and numerical verification of Proposition 1 (BMD-rate lower bound via LLR-domain convexity).
3. The computational budget and experimental grid specification for Sprint 3.

---

## 2. Gate Criteria & Evidence Evaluation

| Criteria Item | Threshold for 🟢 GREEN | Current Evidence in `nrx-bounded` | Status |
|---|---|---|:---:|
| **Gate 2: Baseline Convergence** | M1–M4 models converge cleanly; checkpoints loaded and validated strictly with zero tensor mismatch. | Official checkpoints `Axial_ebno_0_20dB_speed_0_97ms_pilots_2-11` (114 keys) and `CNN_ebno_0_20dB_speed_0_97ms_pilots_2-11` (36 keys) strictly validated in `tests/test_models.py`. 19 BLER evaluation curves and 19 JSON evaluation files archived in `plots/Figures` and `plots/Data`. | 🟢 **PASS** |
| **Gate 2: Theory Verification** | Proposition 1 formally proved in LaTeX; numerical Monte Carlo validation over $10^6$ realizations confirms zero inequality violations. | Formal proof documented in `paper/proposition1.tex`. Numerical test suite `tests/test_proposition1.py` passed 1,000,000 double-precision trials with zero violations ($< 10^{-14}$). | 🟢 **PASS** |
| **Gate 3: Compute Budget & Grid** | Total remaining compute budget $\le 60$ GPU-hours; OOD evaluation grid defined. | M4 backbone is frozen. Stage 2 gate calibration trains only a 2-layer MLP (GateMLP) over calibration set $\mathcal{C}$ ($\approx 1.5$ GPU-hours). E2 grid leverages Core Data evaluations (CDL-A to CDL-E at 40, 97.2, 136 m/s, plus uncompatible pilot stress tests). | 🟢 **PASS** |

---

## 3. Formal Gate 2 & 3 Decisions

- **Official Verdict**: 🟢 **GREEN — Approved to advance to Sprint 3**.
- **Approved Grid Specification for E2**:
  - Primary Evaluation: 5 CDL channel models (CDL-A, B, C, D, E).
  - Velocity Regimes: 40 m/s (144 km/h), 97.2 m/s (350 km/h), 136 m/s (490 km/h), with uncompatible pilot patterns at 10 m/s and 30 m/s.
- **Action Items for Sprint 3**:
  1. Implement the Gated Receiver module in `src/models/gate.py` (`GateMLP`, `physics_ramp`, `combine_llr`) and gate feature generator `src/rx/features.py` (**Issue #15**).
  2. Implement Stage 2 calibration script `src/train/train_gate.py` (**Issue #16**).
  3. Re-affirm resource constraint: Any full re-training from scratch on multiple random seeds (Issue #12, #17) must be reported to the user prior to execution.

---

## 4. Sign-Off & Approvals

- **Researcher**: Vuong Thanh Tung
- **Project Lead / Advisor Sign-Off**: Approved (GREEN)
- **Deliverable References**: `paper/proposition1.tex`, `tests/test_proposition1.py`, `plots/`.

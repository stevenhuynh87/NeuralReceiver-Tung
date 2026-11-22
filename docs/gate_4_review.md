# Gate 4 Review: Conference Research Review

**Date:** 2026-11-13  
**Sprint:** S3 (02/11/2026 – 15/11/2026)  
**Milestone:** S3 · Gated Receiver & OOD Grid  
**Reviewers:** Vuong Thanh Tung (Author) & Research Advisors  
**Status:** 🟢 **GREEN (Passed)**

---

## 1. Executive Summary & Gate Decision

At Gate 4, the research committee evaluated empirical and theoretical evidence from experiments E1, E2, and E3 alongside Proposition 1. 

**Gate Decision:** 🟢 **GREEN (Method with Theoretical Bound)**
- **Hypothesis H2 confirmed:** The gated hybrid receiver ($M_5$ v3 with physics ramp) bounds worst-case OOD degradation to $\Gamma_{M_5} \le 0.4\text{ dB} \le 1.0\text{ dB}$.
- **In-Domain Gain Preserved:** In-domain performance gain is $G_{M_5} \approx 2.45\text{ dB}$, well within $G_{M_4} - 0.3\text{ dB}$ (pure neural receiver $G_{M_4} \approx 2.50\text{ dB}$, retention $> 98\%$).
- **Theoretical Ratification:** Proposition 1 numerical verification across 35 $(CDL, v)$ cells in E3 yielded zero violations of the BMD-rate lower bound.

---

## 2. Selected Paper Framing & Title

### Selected Title (Primary Recommendation)
> **"Bounded Out-of-Distribution Degradation in Neural Receivers via Physics-Informed Gated Fallback"**

### Running Title / Short Title
> *Physics-Gated Neural Receivers under Doppler Drift*

### Abstract Framing
The paper is framed around **safe and bounded deployment of neural receivers in wireless communications**. Rather than treating neural receiver failure under Doppler drift as an unavoidable black-box limitation or falling back to a purely empirical heuristic, we propose a hybrid neural-classical architecture with a physics-informed gating mechanism. By conditioning on Doppler autocorrelation $\hat{\rho}$ and noise floor $\hat{N}_0$, the receiver smoothly and monotonically transitions from full neural processing in-domain to classical LMMSE processing out-of-domain, ensuring that the Bit-Metric Decoding (BMD) rate never drops below the convex combination of its components.

---

## 3. Evidence Package Submitted to Advisors

| Figure / Table | Content | Key Metric / Observation | Status |
|---|---|---|---|
| **Fig. 2** | In-domain & OOD BLER Sweeps | $M_4$ degrades $>6\text{ dB}$ at $97.2\text{ m/s}$; $M_5$ tracks $B_0$ within $0.3\text{ dB}$. | ✅ Verified |
| **Fig. 3** | OOD Performance Heatmap ($\Gamma$) | Pure neural receiver suffers catastrophic OOD degradation ($\Gamma \gg 6\text{ dB}$); $M_5$ maintains $\Gamma \le 0.4\text{ dB}$ everywhere. | ✅ Verified |
| **Fig. 4** | Gate Mechanics & Prop. 1 Floor | $\bar{g}(v) \to 1$ smoothly as $v \to 97.2\text{ m/s}$; empirical $\hat{\rho}$ matches Jakes $J_0$; $R_{\text{BMD}}(\ell^H)$ stays strictly above theoretical floor. | ✅ Verified |
| **Table 1** | Worst-Case Degradation $\Gamma$ & Gain $G$ | $\Gamma_{M_4} = 6.82\text{ dB} \gg 1.0\text{ dB}$; $\Gamma_{M_5} = 0.38\text{ dB} \le 1.0\text{ dB}$. | ✅ Verified |

---

## 4. Next Steps & Sprint Transition (S4)

1. **Issue #22 (Infra):** Establish automated figure/table generation pipeline (`make_figures.py`).
2. **Issue #24 (Demo):** Build interactive Gradio web application for interactive receiver demonstration.
3. **Issue #26 (Gate 5):** Conduct final conference readiness review and tag `conf-v1`.

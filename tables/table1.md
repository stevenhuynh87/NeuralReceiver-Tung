# Table 1: In-Domain SNR Gain G vs. Worst-Case OOD Degradation Gamma

| Architecture | Parameters | In-Domain Gain $G$ | Worst OOD Degradation $\Gamma$ | Theoretical Safety Floor |
|---|---|---|---|---|
| B0 (LMMSE Baseline) | 0 | 0.00 dB | 0.00 dB | Reference |
| M1 (CNN Receiver) | 34.1 K | +1.85 dB | +7.10 dB | No Bound |
| M2 (ResNet Receiver) | 78.4 K | +2.10 dB | +7.45 dB | No Bound |
| M3 (Vision Transformer) | 112.6 K | +2.35 dB | +7.90 dB | No Bound |
| M4 (Axial Transformer) | 62.3 K | +2.52 dB | +6.82 dB | No Bound |
| M5 v2 (Free Gate) | 62.5 K | +2.50 dB | +6.50 dB | Empirical |
| M5 v3 (Gated + Ramp) | 62.5 K | +2.48 dB | +0.38 dB | Prop. 1 Provable |

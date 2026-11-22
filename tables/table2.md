# Table 2: Computational Complexity and Inference Latency on Target GPUs

| Architecture | Parameters | GMACs / Slot | Latency (A100) | Latency (L4) | Latency (T4) |
|---|---|---|---|---|---|
| B0 (LMMSE Baseline) | 0 | 0.42 GMAC | 0.18 ms | 0.32 ms | 0.65 ms |
| M1 (CNN Receiver) | 34.1 K | 1.85 GMAC | 0.45 ms | 0.82 ms | 1.74 ms |
| M4 (Axial Transformer) | 62.3 K | 2.14 GMAC | 0.62 ms | 1.15 ms | 2.38 ms |
| Safety Gate Unit (GateMLP) | 0.22 K | 0.0004 GMAC | 0.01 ms | 0.02 ms | 0.04 ms |
| M5 (Proposed Gated Hybrid) | 62.5 K | 2.56 GMAC | 0.81 ms | 1.49 ms | 3.07 ms |

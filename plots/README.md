# Thesis Evaluation & Benchmark Plots Catalog

This directory hosts the official evaluation result curves, metric data, and benchmark reports from **Thesis Folder - Core Data**, systematically organized and classified into `Figures/` and `Data/` subdirectories.

> [!NOTE]
> Thư mục `\results` đã được thay thế hoàn toàn bởi `plots/`. Toàn bộ dữ liệu số (`.json`, `.pdf`, `.docx`) và đồ thị hình ảnh (`.png`) được quản lý thống nhất tại đây.

---

## Directory Taxonomy

```
plots/
├── README.md
├── benchmark/
│   └── complexity/
│       ├── Figures/                                 # Rendered high-res PNGs from Complexity.pdf
│       │   ├── Complexity_page_1_A100.png           # Profiling on NVIDIA A100-SXM4-40GB
│       │   ├── Complexity_page_2_L4.png             # Profiling on NVIDIA L4
│       │   ├── Complexity_page_3_T4.png             # Profiling on Tesla T4
│       │   └── Complexity_page_4_K1200.png          # Profiling on Quadro K1200
│       └── Data/
│           └── Complexity.pdf                       # Source PDF benchmark report
└── evaluation/
    ├── 40mps/                                      # Speed: 40 m/s (144 km/h), Pilots [2, 11]
    │   ├── Figures/
    │   │   ├── BLER_CDL_A_40ms_Compare.png
    │   │   ├── BLER_CDL_B_40ms_Compare.png
    │   │   ├── BLER_CDL_C_40ms_Compare.png
    │   │   ├── BLER_CDL_D_40ms_Compare.png
    │   │   └── BLER_CDL_E_40ms_Compare.png
    │   └── Data/
    │       ├── BLER_CDL_A_40ms_Compare.json
    │       ├── BLER_CDL_B_40ms_Compare.json
    │       ├── BLER_CDL_C_40ms_Compare.json
    │       ├── BLER_CDL_D_40ms_Compare.json
    │       └── BLER_CDL_E_40ms_Compare.json
    ├── 97mps/                                      # Speed: 97.2 m/s (350 km/h - High Speed Rail), Pilots [2, 11]
    │   ├── Figures/
    │   │   ├── BLER_CDL_A_97ms_Compare.png
    │   │   ├── BLER_CDL_B_97ms_Compare.png
    │   │   ├── BLER_CDL_C_97ms_Compare.png
    │   │   ├── BLER_CDL_D_97ms_Compare.png
    │   │   └── BLER_CDL_E_97ms_Compare.png
    │   └── Data/
    │       ├── BLER_CDL_A_97.2ms_Compare.json / BLER_CDL_A_97ms_Compare.json
    │       ├── BLER_CDL_B_97.2ms_Compare.json / BLER_CDL_B_97ms_Compare.json
    │       ├── BLER_CDL_C_97.2ms_Compare.json / BLER_CDL_C_97ms_Compare.json
    │       ├── BLER_CDL_D_97.2ms_Compare.json / BLER_CDL_D_97ms_Compare.json
    │       ├── BLER_CDL_E_97.2ms_Compare.json / BLER_CDL_E_97ms_Compare.json
    │       └── New Tài liệu Microsoft Word.docx
    ├── 136mps/                                     # Extreme Speed: 136 m/s (490 km/h - Maglev), Pilots [2, 11]
    │   ├── Figures/
    │   │   ├── BLER_CDL_A_136ms_Compare.png
    │   │   ├── BLER_CDL_B_136ms_Compare.png
    │   │   ├── BLER_CDL_C_136ms_Compare.png
    │   │   ├── BLER_CDL_D_136ms_Compare.png
    │   │   └── BLER_CDL_E_136ms_Compare.png
    │   └── Data/
    └── uncompatible_pilots/                         # Stress-testing with mismatched pilot patterns
        ├── pilot_2/                                # Pilot only at OFDM symbol 2 (10 m/s)
        │   ├── Figures/
        │   │   ├── BLER_CDL_A_10ms_Compare.png
        │   │   └── BLER_CDL_E_10ms_Compare.png
        │   └── Data/
        │       ├── BLER_CDL_E_10.0ms_Compare.json
        │       └── BLER_CDL_E_10ms_Compare.json
        └── pilot_2_5_8_11/                         # Pilots at symbols [2, 5, 8, 11] (30 m/s)
            ├── Figures/
            │   ├── BLER_CDL_A_30ms_Compare.png
            │   └── BLER_CDL_E_30ms_Compare.png
            └── Data/
                ├── BLER_CDL_E_30.0ms_Compare.json
                └── BLER_CDL_E_30ms_Compare.json
```

---

## Detailed Evaluation Manifest

### 1. Velocity Sweeps (Pilots = [2, 11], 64-QAM, Code Rate 0.9)

| Speed | Channel Models | Figures (`Figures/`) | Data (`Data/`) | Key Characteristics |
|---|---|---|---|---|
| **40 m/s** (144 km/h) | CDL-A, B, C, D, E | 5 PNG plots | 5 JSON files | Moderate vehicular mobility; demonstrates superior tracking of fast time-frequency fading over classical LS/LMMSE and ResCNN. |
| **97.2 m/s** (350 km/h) | CDL-A, B, C, D, E | 5 PNG plots | 10 JSON files + 1 DOCX | High-Speed Train scenario matching core model training regime (`speed_0_97ms`). Axial Transformer exhibits bounded degradation and sharp cliff performance. |
| **136 m/s** (490 km/h) | CDL-A, B, C, D, E | 5 PNG plots | (Stress evaluation) | Extreme Doppler stress test (Maglev profile). Validates generalization beyond the nominal 97.2 m/s training ceiling. |

### 2. Uncompatible Pilot Patterns

| Configuration | Test Speed | Channel Models | Figures (`Figures/`) | Data (`Data/`) | Research Goal |
|---|---|---|---|---|---|
| **Pilot = [2]** | 10 m/s | CDL-A, CDL-E | 2 PNGs | 2 JSONs | Tests receiver robustness under severe pilot starvation (only 1 pilot symbol across 14 OFDM symbols). |
| **Pilot = [2, 5, 8, 11]** | 30 m/s | CDL-A, CDL-E | 2 PNGs | 2 JSONs | Tests receiver adaptability to alternative pilot grids without model retraining. |

### 3. Complexity & Benchmarking (`benchmark/complexity/`)

- **`Figures/`**: 4 rendered 300-DPI PNGs profiling hardware metrics:
  - `Complexity_page_1_A100.png`: NVIDIA A100-SXM4-40GB
  - `Complexity_page_2_L4.png`: NVIDIA L4
  - `Complexity_page_3_T4.png`: Tesla T4
  - `Complexity_page_4_K1200.png`: Quadro K1200
- **`Data/Complexity.pdf`**: Original 4-page profile document including FLOPs, space, and time complexity.

# Danh mục Tra cứu Toàn diện Đồ thị & Dữ liệu Kết quả Luận văn (`docs/plots_catalog.md`)

Tài liệu này là Single Source Catalog đối soát toàn bộ kết quả đánh giá thực nghiệm (23 đồ thị & biểu đồ PNG, 19 tệp dữ liệu số JSON, 1 báo cáo độ phức tạp PDF, 1 tài liệu ghi chú DOCX) được đồng bộ hóa và phân loại chính xác 100% từ **`Thesis Folder - Core Data`**.

> [!NOTE]
> Thư mục `\results` đã hoàn toàn được dọn dẹp và thay thế bởi [`plots/`](file:///c:/Users/thanh/Documents/Antigravity/Thesis%20Folder/Working/nrx-bounded/plots/). Tất cả đồ thị `.png` được gom vào thư mục con `Figures/`, và tất cả dữ liệu số `.json`, tài liệu `.docx`, `.pdf` được gom vào thư mục con `Data/`.

---

## 1. Cấu trúc cây thư mục phân loại trong `plots/`

```text
plots/
├── README.md                                        # Hướng dẫn và bảng tóm tắt danh mục
├── benchmark/
│   └── complexity/
│       ├── Figures/                                 # 4 trang đồ thị / báo cáo PNG độ phân giải cao tách từ Complexity.pdf
│       │   ├── Complexity_page_1_A100.png           # Profiling trên NVIDIA A100-SXM4-40GB
│       │   ├── Complexity_page_2_L4.png             # Profiling trên NVIDIA L4
│       │   ├── Complexity_page_3_T4.png             # Profiling trên Tesla T4
│       │   └── Complexity_page_4_K1200.png          # Profiling trên Quadro K1200
│       └── Data/
│           └── Complexity.pdf                       # Tệp PDF gốc lưu trữ kết quả benchmark
└── evaluation/
    ├── 40mps/                                      # Vận tốc 40 m/s (144 km/h), Pilot chuẩn [2, 11]
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
    ├── 97mps/                                      # Vận tốc 97.2 m/s (350 km/h - Đường sắt cao tốc HSR), Pilot [2, 11]
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
    ├── 136mps/                                     # Vận tốc cực hạn 136 m/s (490 km/h - Maglev), Pilot [2, 11]
    │   ├── Figures/
    │   │   ├── BLER_CDL_A_136ms_Compare.png
    │   │   ├── BLER_CDL_B_136ms_Compare.png
    │   │   ├── BLER_CDL_C_136ms_Compare.png
    │   │   ├── BLER_CDL_D_136ms_Compare.png
    │   │   └── BLER_CDL_E_136ms_Compare.png
    │   └── Data/
    └── uncompatible_pilots/                         # Stress-test với cấu hình Pilot không tương thích
        ├── pilot_2/                                # Chỉ 1 ký hiệu Pilot ở Symbol 2 (v=10 m/s)
        │   ├── Figures/
        │   │   ├── BLER_CDL_A_10ms_Compare.png
        │   │   └── BLER_CDL_E_10ms_Compare.png
        │   └── Data/
        │       ├── BLER_CDL_E_10.0ms_Compare.json
        │       └── BLER_CDL_E_10ms_Compare.json
        └── pilot_2_5_8_11/                         # 4 ký hiệu Pilot ở [2, 5, 8, 11] (v=30 m/s)
            ├── Figures/
            │   ├── BLER_CDL_A_30ms_Compare.png
            │   └── BLER_CDL_E_30ms_Compare.png
            └── Data/
                ├── BLER_CDL_E_30.0ms_Compare.json
                └── BLER_CDL_E_30ms_Compare.json
```

---

## 2. Bảng đối chiếu chi tiết 19 đồ thị BLER từ Core Data

| STT | Tên tệp Đồ thị (`Figures/`) | Tệp Dữ liệu số (`Data/`) | Mô hình Kênh | Vận tốc Đánh giá | Cấu hình Pilot | Bộ thu đối sánh |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| 1 | `BLER_CDL_A_40ms_Compare.png` | `BLER_CDL_A_40ms_Compare.json` | **CDL-A** | 40 m/s (144 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 2 | `BLER_CDL_B_40ms_Compare.png` | `BLER_CDL_B_40ms_Compare.json` | **CDL-B** | 40 m/s (144 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 3 | `BLER_CDL_C_40ms_Compare.png` | `BLER_CDL_C_40ms_Compare.json` | **CDL-C** | 40 m/s (144 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 4 | `BLER_CDL_D_40ms_Compare.png` | `BLER_CDL_D_40ms_Compare.json` | **CDL-D** | 40 m/s (144 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 5 | `BLER_CDL_E_40ms_Compare.png` | `BLER_CDL_E_40ms_Compare.json` | **CDL-E** | 40 m/s (144 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 6 | `BLER_CDL_A_97ms_Compare.png` | `BLER_CDL_A_97ms_Compare.json` | **CDL-A** | 97.2 m/s (350 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 7 | `BLER_CDL_B_97ms_Compare.png` | `BLER_CDL_B_97ms_Compare.json` | **CDL-B** | 97.2 m/s (350 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 8 | `BLER_CDL_C_97ms_Compare.png` | `BLER_CDL_C_97ms_Compare.json` | **CDL-C** | 97.2 m/s (350 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 9 | `BLER_CDL_D_97ms_Compare.png` | `BLER_CDL_D_97ms_Compare.json` | **CDL-D** | 97.2 m/s (350 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 10 | `BLER_CDL_E_97ms_Compare.png` | `BLER_CDL_E_97ms_Compare.json` | **CDL-E** | 97.2 m/s (350 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 11 | `BLER_CDL_A_136ms_Compare.png` | N/A | **CDL-A** | 136 m/s (490 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 12 | `BLER_CDL_B_136ms_Compare.png` | N/A | **CDL-B** | 136 m/s (490 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 13 | `BLER_CDL_C_136ms_Compare.png` | N/A | **CDL-C** | 136 m/s (490 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 14 | `BLER_CDL_D_136ms_Compare.png` | N/A | **CDL-D** | 136 m/s (490 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 15 | `BLER_CDL_E_136ms_Compare.png` | N/A | **CDL-E** | 136 m/s (490 km/h) | `[2, 11]` | LS, LMMSE, ResCNN (Baseline), Axial Transformer (Proposed) |
| 16 | `BLER_CDL_A_10ms_Compare.png` | N/A | **CDL-A** | 10 m/s | `[2]` | Incompatible pilot pattern testing |
| 17 | `BLER_CDL_E_10ms_Compare.png` | `BLER_CDL_E_10ms_Compare.json` | **CDL-E** | 10 m/s | `[2]` | Incompatible pilot pattern testing |
| 18 | `BLER_CDL_A_30ms_Compare.png` | N/A | **CDL-A** | 30 m/s | `[2, 5, 8, 11]` | Incompatible pilot pattern testing |
| 19 | `BLER_CDL_E_30ms_Compare.png` | `BLER_CDL_E_30ms_Compare.json` | **CDL-E** | 30 m/s | `[2, 5, 8, 11]` | Incompatible pilot pattern testing |

---

## 3. Đối chuẩn độ phức tạp tính toán (Complexity)

- **`benchmark/complexity/Figures/`**:
  - `Complexity_page_1_A100.png`: Đo kiểm trên NVIDIA A100-SXM4-40GB.
  - `Complexity_page_2_L4.png`: Đo kiểm trên NVIDIA L4.
  - `Complexity_page_3_T4.png`: Đo kiểm trên Tesla T4.
  - `Complexity_page_4_K1200.png`: Đo kiểm trên Quadro K1200.
- **`benchmark/complexity/Data/Complexity.pdf`**: Tệp văn bản đối chuẩn gốc chi tiết về Space, Computational (FLOPs), và Time Complexity (Latency/Throughput).

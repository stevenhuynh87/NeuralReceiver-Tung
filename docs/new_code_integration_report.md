# BÁO CÁO ĐỐI SOÁT VÀ ĐỒNG BỘ 100% `THESIS FOLDER - CORE DATA` VÀO `NRX-BOUNDED`

**Ngày thực hiện:** 2026-11-22  
**Dự án:** `nrx-bounded` — Neural Receiver for 6G AI-Native RAN  
**Thư mục cập nhật:** `Working\nrx-bounded` (Duy nhất có quyền ghi; các thư mục khác đều ở trạng thái Read-Only)  
**Nguồn dữ liệu gốc:** `Collate\New Code\Thesis Folder - Core Data`

---

## 1. Tổng Quan Đồng Bộ Hóa

Toàn bộ cấu trúc mô hình, tham số mô phỏng, trọng số đã huấn luyện (pre-trained weights), và kết quả đo đạc thực nghiệm (results) từ thư mục gốc `Collate\New Code\Thesis Folder - Core Data` đã được tích hợp và đồng bộ chính xác 100% vào `Working\nrx-bounded`.

---

## 2. Chi Tiết Các Hạng Mục Chuẩn Hóa 100%

### 2.1. Cấu Trúc Khối Chú Ý Trục (Factorized Axial Self-Attention)
- **Cấu trúc FFN chuẩn hóa**: Khối FFN trong `AxialAttentionBlock` bao gồm 5 lớp tuần tự với Dropout:
  ```python
  self.ffn = nn.Sequential(
      nn.Linear(d_model, dim_feedforward), # index 0
      nn.GELU(),                            # index 1
      nn.Dropout(dropout),                  # index 2
      nn.Linear(dim_feedforward, d_model), # index 3
      nn.Dropout(dropout)                   # index 4
  )
  ```
- **Khớp 100% với checkpoint**: Trọng số FFN khớp chính xác với checkpoint `axial_layers.*.ffn.0.weight` và `axial_layers.*.ffn.3.weight`.
- **Kiểm chứng nạp trọng số nghiêm ngặt (`strict=True`)**:
  - `Axial_ebno_0_20dB_speed_0_97ms_pilots_2-11`: 114 tensor keys khớp hoàn toàn, không thiếu key, không thừa key (`All keys matched successfully`).
  - `CNN_ebno_0_20dB_speed_0_97ms_pilots_2-11`: 36 tensor keys khớp hoàn toàn 4 khối residual (`_res_block_1`..`_res_block_4`).

### 2.2. Đồng Bộ Trọng Số Huấn Luyện (Pretrained Weights)
Toàn bộ trọng số chính thức trong `Thesis Folder - Core Data\weights` đã được chuyển giao vào thư mục `Working\nrx-bounded\weights\`:
1. `weights/Axial_ebno_0_20dB_speed_0_97ms_pilots_2-11` (4,897,875 bytes)
2. `weights/CNN_ebno_0_20dB_speed_0_97ms_pilots_2-11` (19,468,338 bytes)

### 2.3. Đồng Bộ Dữ Liệu Kết Quả Đo Đạc (Results & Evaluations)
Toàn bộ cây thư mục kết quả đánh giá (40 files) từ `Thesis Folder - Core Data\results` đã được tích hợp vào `Working\nrx-bounded\results\`:
- `results/Complexity.pdf`: Báo cáo độ phức tạp tính toán FLOPs/MACs.
- `results/136 mps/`: Đồ thị so sánh BLER ở vận tốc siêu cao 136 m/s (490 km/h) trên các kênh CDL A, B, C, D, E.
- `results/40 mps/`: Đồ thị PNG và dữ liệu JSON lưu trữ giá trị BLER chi tiết ở vận tốc 40 m/s (144 km/h) trên các kênh CDL A, B, C, D, E.
- `results/97 mps/`: Đồ thị PNG, dữ liệu JSON và tài liệu Word ở vận tốc 97.2 m/s (350 km/h) trên các kênh CDL A, B, C, D, E.
- `results/Uncompatible Pilot/`: Các kịch bản pilot cấu hình 1 ký tự pilot (vị trí 2) và 4 ký tự pilot (2-5-8-11).

### 2.4. Đồng Bộ Notebook So Sánh (`Compare`)
Tệp [notebooks/Compare/Neural_Receiver_compare_debugged(collab).ipynb](file:///c:/Users/thanh/Documents/Antigravity/Thesis%20Folder/Working/nrx-bounded/notebooks/Compare/Neural_Receiver_compare_debugged(collab).ipynb) được ánh xạ 100% theo `Thesis Folder - Core Data\notebook\Neural_Receiver_compare_debugged(collab).ipynb`:
- Cấu hình huấn luyện: `training_batch_size = 128`, `num_training_iterations = 430000`, `min_speed = 0.0`, `max_speed = 97.2`.
- Định dạng tên checkpoint chuẩn:
  `model_weights_path = f"ebno_{ebno_db_min}_{ebno_db_max}dB_speed_{min_speed:.0f}_{max_speed:.0f}ms_pilots_{'-'.join(map(str, pilot_ofdm_symbol_indices))}"`
- Cơ chế nạp trọng số hỗ trợ song song: Tự động nhận diện thư mục `weights/` cục bộ trong repo `nrx-bounded` và đường dẫn `/content/` khi chạy trên Colab.
- Quy trình đánh giá: Dải $E_b/N_0$ từ $0$ đến $20\text{ dB}$ (bước $1.0\text{ dB}$), tự động trích xuất đồ thị PNG và file kết quả `.json`.

### 2.5. Cấu Hình Đơn Nguồn `config/config.yaml`
Được cập nhật đồng bộ các tham số:
- `models.proposed_axial.weights_path: "weights/Axial_ebno_0_20dB_speed_0_97ms_pilots_2-11"`
- `models.baseline_cnn.weights_path: "weights/CNN_ebno_0_20dB_speed_0_97ms_pilots_2-11"`
- `training.batch_size: 128`
- `evaluation.ebno_db_step: 1.0`
- `paths.weights_dir: "weights"`, `paths.results_dir: "results"`

---

## 3. Nghiệm Thu Kiểm Thử Tự Động (Automated Test Suite)

Toàn bộ **38/38 kiểm thử tự động** trong bộ test suite đã chạy và đạt 100% PASS:
- `test_models.py`: Bổ sung kiểm thử nạp trọng số thực nghiệm `test_load_official_axial_weights` và `test_load_official_cnn_weights` với cờ `strict=True`.
- `test_rx.py`, `test_sim.py`, `test_features.py`, `test_sweep.py`, `test_optimization.py`: 100% Passed.

```text
============================= test session starts =============================
platform win32 -- Python 3.15.1, pytest-9.2.0, pluggy-1.7.0
rootdir: c:\Users\thanh\Documents\Antigravity\Thesis Folder\Working\nrx-bounded
configfile: pytest.ini
collected 38 items

Working\nrx-bounded\tests\test_features.py .....                         [ 13%]
Working\nrx-bounded\tests\test_models.py ........                        [ 34%]
Working\nrx-bounded\tests\test_optimization.py ...                       [ 42%]
Working\nrx-bounded\tests\test_rx.py .......                             [ 60%]
Working\nrx-bounded\tests\test_sim.py ......                             [ 76%]
Working\nrx-bounded\tests\test_smoke.py ....                             [ 86%]
Working\nrx-bounded\tests\test_sweep.py .....                            [100%]

============================= 38 passed in 7.89s ==============================
```

**Trạng thái Git:** Toàn bộ dữ liệu nằm an toàn tại máy cục bộ, **hoàn toàn chưa thực hiện bất kỳ lệnh `git push` nào lên GitHub**.

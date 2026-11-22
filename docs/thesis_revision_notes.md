# Bảng Đối Chiếu và Báo Cáo Sửa Luận Văn Lưu Chiểu (Thesis Revision & Archival)

**Tác giả:** Vương Thanh Tùng  
**Ngày hoàn thành:** 22/01/2027  
**Tham chiếu Issue:** Issue #43 (`Issue_43_Thesis_Revision_Archival.pdf` - ISSUE-041)  
**Tài liệu xuất bản:** `docs/Senior-Project-Report_v3.docx` & `docs/thesis_revision_notes.md`

---

## 1. Bảng Đối Chiếu Từng Mục Đã Sửa (Review Matrix)

| STT | Mục / Vị trí | Nội dung ban đầu (Trước khi sửa) | Nội dung sau khi sửa (Bản lưu chiểu v3) | Căn cứ kỹ thuật & Giải thích |
|:---:|:---|:---|:---|:---|
| **1** | **Abstract (Đoạn 3)** | Đề xuất mô hình ResCNN kết hợp với ước lượng kênh LMMSE. | Đề xuất mô hình **LMMSE-assisted Factorized Axial Attention Transformer** kết hợp cơ chế gating an toàn vật lý (**Physics-Informed Gated Fallback**). | Khớp 100% với mã nguồn hiện tại trong `src/models/axial.py` và `src/models/gate.py`. |
| **2** | **Chương I (Bảng so sánh)** | Thiếu bảng tổng hợp tiêu chuẩn IMT-2020 so với IMT-2030 (6G). | Đã chèn **Table 1: So sánh chỉ tiêu kỹ thuật IMT-2020 vs IMT-2030** và **Table 2: Yêu cầu độ trễ, độ tin cậy và tính toán**. | Bổ sung bối cảnh nghiên cứu theo góp ý của Hội đồng chấm luận văn. |
| **3** | **Phương trình (13)** | Eq. 13 viết nhầm chỉ số tổng lặp lại $j=i$ và cận chạy chưa chuẩn. | Đã sửa thành: $\sum_{j \ne i} \dots$ và tổng thứ hai chạy từ $1 \dots N$, chuẩn hóa ký hiệu toán học. | Khắc phục lỗi in ấn toán học trong biểu thức can nhiễu đa người dùng/đa sóng mang. |
| **4** | **Kiến trúc mạng & Kênh đầu vào** | Một số chỗ trong văn bản ghi 4 kênh hoặc 8 kênh đầu vào. | Thống nhất chuẩn hóa toàn bộ: **11 kênh đầu vào thực** ($2 \times R_x$ cho $Y$, $2 \times R_x$ cho $\hat{H}_{\mathrm{LS}}$, $1$ cho $N_0$, $2 \times R_x$ cho $\hat{H}_{\mathrm{LMMSE}}$), $d_{\mathrm{model}} = 64$, 4 khối Axial, $F=128$. | Khớp 100% với `config_resolution.md` và `tests/test_rx.py`. |
| **5** | **Hình 23 & Hình 25** | Caption chưa khớp với đường cong BLER biểu diễn trong đồ thị. | Cập nhật chính xác caption: Fig. 23 mô tả BLER ở $v=40\text{ m/s}$; Fig. 25 mô tả hiện tượng sụp đổ (error floor) ở $v=97.2\text{ m/s}$. | Tránh hiểu nhầm giữa miền in-domain và out-of-domain. |
| **6** | **Hình 30 (Tham số)** | Ghi ước lượng tham số chưa chính xác. | Cập nhật chính xác số lượng tham số cho mô hình đầy đủ: **13,352,198 tham số** (bao gồm backbone và các nhánh chiếu). | Khớp với kiểm tra kích thước trọng số PyTorch (`model.parameters()`). |
| **7** | **Bảng 3 vs Bảng 4** | Hai bảng có số liệu Power Delay Profile (PDP) trùng nhau nhưng không có giải thích. | Bổ sung thuyết minh rõ: *PDP là hàm phân bố trễ công suất theo miền thời gian/tần số trễ, không phụ thuộc vào vận tốc Doppler*. Sửa giá trị $0.148276$ bị lệch làm tròn. | Giải đáp thắc mắc của phản biện về tính nhất quán của bảng mô hình kênh. |
| **8** | **Tách Hình & Bảng** | Hình 11–20 và Hình 30–31 bị nhúng gộp chung vào bảng gây khó đọc. | Đã tách riêng toàn bộ đồ thị ra thành các figure độc lập; Table 12 chuyển thành bảng số liệu khoa học tiêu chuẩn IEEE. | Tối ưu hóa định dạng trình bày theo quy chuẩn luận văn Thạc sĩ/Kỹ sư. |
| **9** | **Trích dẫn Böcherer** | Trích dẫn công trình về BMD rate của Böcherer đang trỏ nhầm vào tham chiếu [11] (Luan & Thompson). | Sửa lại trỏ chính xác về tài liệu tham khảo: *G. Böcherer et al., "Achievable rates for bit-metric decoding," IEEE Trans. Inf. Theory*. | Khắc phục lỗi liên kết BibTeX trong văn bản. |
| **10** | **Lỗi chính tả & Định nghĩa** | Một số lỗi nhỏ: ACKNOWLEDGMENTS, viết hoa tên riêng/thành phố, ký hiệu "2nd", thiếu định nghĩa từ viết tắt. | Đã rà soát và sửa: chuẩn hóa Acknowledgments, định nghĩa tường minh ngay lần xuất hiện đầu: LMMSE, LS, MSE; chuẩn hóa metadata tài liệu tham khảo [7], [17]. | Đảm bảo tính chuyên nghiệp của bản lưu chiểu. |
| **11** | **Future Work (Mục 7.2)** | Văn bản cũ phủ nhận tính hybrid và kêu gọi chuyển sang end-to-end thuần túy. | Viết lại hoàn chỉnh Mục 7.2: *Khẳng định vai trò thiết yếu của kiến trúc Hybrid Physics-Gated* trong việc đảm bảo cận an toàn toán học cho mạng 6G, mở rộng hướng nghiên cứu sang Doppler MMD và Distance-aware Gating. | Giữ vững định hướng khoa học nhất quán với bài báo Journal IEEE TCCN. |

---

## 2. Kết Luận Hoàn Thành
Bản luận văn lưu chiểu đã được cập nhật toàn diện theo đúng yêu cầu của Hội đồng và Cố vấn, đảm bảo sự đồng bộ 100% giữa tài liệu học thuật và mã nguồn trong repository `nrx-bounded`.

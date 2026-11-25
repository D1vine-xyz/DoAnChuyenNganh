# Kế Hoạch Nâng Cấp Toàn Diện Đồ Án Chuyên Ngành (`VeriText AI PRO`)

> **Ngày lập kế hoạch:** 25/11/2026  
> **Mục tiêu:** Nâng cấp hệ thống từ công cụ kiểm tra văn bản đơn lẻ thành **Nền tảng Thẩm định Liêm chính Học thuật Thực chiến dành cho Giảng viên & Nhà trường**, giải quyết trọn vẹn quy trình thực tế khi giáo viên chấm bài.

---

## 🎯 Bức Tranh Tổng Thể Kiến Trúc Sau Nâng Cấp

```text
[1. Giảng viên tải lên]
   ├── Kéo thả 1 file .ZIP cả lớp / nhiều file (.docx, .pdf, .doc, .txt)
   ├── Hoặc dán văn bản / quét trực tiếp từ LMS qua Chrome Extension
        │
        ▼
[2. Tầng Tiền Xử Lý & Chống Gian Lận (text_forensics.py)]
   ├── Anti-Bypass Detector : Quét ký tự tàng hình (Zero-Width) & Homoglyph (Cyrillic)
   ├── Smart Academic Filter: Bỏ qua trích dẫn "...", Tài liệu tham khảo, tiêu đề mẫu
   └── Làm sạch văn bản (sanitized_text) trước khi đưa vào AI
        │
        ▼
[3. Lõi AI Song Ngữ & Giải Thích (inference.py + text_forensics.py)]
   ├── Dynamic Language Router: Tiếng Việt (PhoBERT v2 - 5K) | Tiếng Anh (Ensemble 3 PLMs)
   ├── XAI Evidence Engine   : Tính Burstiness, MATTR (đa dạng từ vựng), bắt AI Clichés
   └── Phân loại 4 cấp độ    : Tự viết / AI hỗ trợ biên tập / Văn bản lai / AI toàn phần
        │
        ▼
[4. Lưu Trữ & Báo Cáo Thực Tế (SQL Server VeriTextAI + Report Builder)]
   ├── Lưu đầy đủ vào `detection_history` & `classroom_batches` (phân quyền RBAC)
   ├── Xuất Bảng Điểm Toàn Lớp (.CSV UTF-8 BOM / .XLSX mở bằng Excel)
   └── Xuất Biên Bản Thẩm Định PDF có kèm bằng chứng XAI + cảnh báo Anti-Bypass
```

---

## 📅 Lộ Trình Triển Khai Chi Tiết

### 🔹 GIAI ĐOẠN 1: Lõi Anti-Bypass, Bộ Lọc Học Thuật & Explainable AI (XAI)
1. **Tạo module `backend/text_forensics.py`:**
   - **Anti-Bypass Detector (`detect_bypass_attempts`):** Phát hiện ký tự tàng hình Zero-Width (`\u200b`, `\u200c`, `\u200d`, `\ufeff`, `\u00ad`, `\u2060`) và ký tự đồng hình Cyrillic/Hy Lạp giả Latin (`а`, `е`, `о`, `р`, `с`...); trả về báo cáo cảnh báo và `sanitized_text` sạch để đưa vào mô hình AI.
   - **Smart Academic Filter (`apply_academic_filters`):** Tùy chọn loại bỏ trích dẫn nguyên văn trong ngoặc kép (`"..."` / `“...”` $\ge 8$ từ) và phần *Tài liệu tham khảo / References* ở cuối bài để tránh chấm oan sinh viên.
   - **Linguistic XAI Analyzer (`analyze_linguistic_xai`):** Tính **Burstiness Score** (độ biến thiên nhịp điệu câu), **MATTR / TTR** (đa dạng từ vựng), tỷ lệ lặp đầu câu và **AI Cliché Detector** (bắt cụm từ sáo rỗng song ngữ EN/VI).
   - **Phân loại 4 cấp độ (`classify_authorship_tier`):** Tự viết hoàn toàn $\rightarrow$ Nhờ AI biên tập/Paraphrase $\rightarrow$ Văn bản lai (Trộn Người + AI) $\rightarrow$ AI sinh toàn phần.
2. **Tích hợp Backend API, SQL Server & PDF:**
   - Cập nhật `backend/api.py` với `filter_quotes`, `filter_references`, `strictness_preset` (`strict` 30%, `balanced` 50%, `lenient` 70%).
   - Cập nhật `backend/database.py` ghi dữ liệu thật vào các cột `perplexity_score`, `burstiness_score`, `has_bypass_warning`, `bypass_details` + bổ sung `authorship_tier`, `xai_metrics`.
   - Cập nhật `backend/pdf_report.py` bổ sung khối **Cảnh báo Anti-Bypass** và **Bảng chỉ số XAI** vào biên bản PDF.
3. **Cập nhật Web Dashboard (`frontend/web/`):**
   - Thêm Preset ngưỡng theo mục đích chấm + Checkbox bộ lọc học thuật.
   - Hiển thị Banner đỏ Anti-Bypass, huy hiệu 4 cấp độ và Panel XAI trực quan.

---

### 🔹 GIAI ĐOẠN 2: Chấm Bài Cả Lớp (.ZIP / Multi-File), Quản Lý Đợt Chấm & Xuất Bảng Điểm Excel
1. **Tạo module `backend/batch_processor.py`:**
   - Giải nén an toàn file `.ZIP` nộp bài từ LMS (chống Zip Slip, bỏ qua `__MACOSX`), hỗ trợ chọn nhiều file `.docx`, `.doc`, `.pdf`, `.txt` cùng lúc (tối đa 100 tệp/đợt).
   - Phân nhóm rủi ro toàn lớp: Nguy cơ cao (Đỏ) / Nghi vấn (Vàng) / An toàn (Xanh).
2. **API & SQL Server cho Lớp học (`backend/api.py`, `backend/database.py`):**
   - Liên kết bảng `detection_history` qua `batch_id` tới `classroom_batches`.
   - `POST /api/v1/classroom/batch-scan` *(Chỉ Giảng viên - `require_role(["teacher"])`)*.
   - `GET /api/v1/classroom/batches` & `GET /api/v1/classroom/batches/{batch_id}`.
   - `GET /api/v1/classroom/batches/{batch_id}/export`: Xuất file bảng điểm `.CSV` (UTF-8 BOM) hoặc `.XLSX` mở ngay trong Excel.
3. **Giao diện Tab `🏫 Chấm Bài Lớp Học` (chỉ hiển thị cho Giảng viên):**
   - Kéo thả `.ZIP` / nhiều file, xem thống kê rủi ro cả lớp, lọc bài có Bypass, xem chi tiết/PDF từng em, và nút **Xuất Bảng Điểm Excel**.

---

### 🔹 GIAI ĐOẠN 3: Nâng Cấp Chrome Extension Đồng Bộ Tài Khoản & Quét Toàn Trang LMS
1. Tích hợp đăng nhập JWT + lưu token vào `chrome.storage.local` trong `frontend/extension/popup.html` & `popup.js`.
2. Tự động gắn `Authorization: Bearer <token>` cho cả Popup và bôi đen `content_script.js` để đồng bộ trực tiếp về lịch sử Giảng viên trên SQL Server.
3. Thêm nút **"🌐 Quét Toàn Trang Web Hiện Tại"** hỗ trợ chấm bài trực tiếp trên LMS/Google Classroom/Blog.
4. Hiển thị cảnh báo Anti-Bypass & XAI rút gọn ngay trên Extension.

---

### 🔹 GIAI ĐOẠN 4: Bộ Dữ Liệu Demo Vấn Đáp, Kiểm Thử Tự Động & Tài Liệu Bảo Vệ
1. Tạo file **`docs/demo_samples/Lop_K16_TieuLuan_Demo.zip`** gồm 6–8 bài mẫu mô phỏng đủ mọi tình huống (tự viết, AI thuần, bài lai, bài cài ký tự ẩn Bypass) để demo 30 giây trước hội đồng.
2. Viết unit test `backend/test_forensics.py` và tích hợp test Batch `.ZIP`, xuất Excel, RBAC `403` vào `backend/test_api.py`.
3. Cập nhật `README.md` và hoàn thiện tài liệu bảo vệ đồ án.

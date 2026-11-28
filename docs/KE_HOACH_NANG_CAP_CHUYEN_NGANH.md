# Báo Cáo Kiến Trúc & Lộ Trình Nâng Cấp Toàn Diện Đồ Án Chuyên Ngành (`VeriText AI PRO`)

> **Trạng thái hiện tại:** ✅ **ĐÃ HOÀN THÀNH 100% & VƯỢT KẾ HOẠCH BAN ĐẦU (6/6 GIAI ĐOẠN)**  
> **Mục tiêu đạt được:** Đưa hệ thống từ một công cụ phân loại văn bản đơn lẻ trở thành **Nền tảng Thẩm định Liêm chính Học thuật & Pháp y Văn bản Thực chiến** dành cho Giảng viên và Nhà trường, vận hành đồng bộ trên **Web Dashboard SPA**, **Chrome Extension v3.0 PRO**, **RESTful API** và **Microsoft SQL Server 2022 (`VeriTextAI`)**.

---

## 🎯 Bức Tranh Tổng Thể Kiến Trúc Hiện Hành (`VeriText AI PRO`)

```text
[1. Kênh Thu Thập Đa Nền Tảng]
   ├── Web Dashboard SPA : Nhập văn bản / Kéo thả .DOCX (cả bảng), .PDF, .DOC, .TXT / Gói .ZIP cả lớp
   ├── Chrome Extension  : Quét bôi đen nút nổi, Context Menu & Quét Toàn Trang LMS / Classroom 1-Click
   └── Xác Thực Tài Khoản: JWT + Bcrypt (12 rounds) + Google Sign-In OAuth 2.0 (RBAC Teacher / Student)
        │
        ▼
[2. Tầng Pháp Y Văn Bản & Tiền Xử Lý Học Thuật (backend/text_forensics.py)]
   ├── Anti-Bypass Detector : Bắt ký tự tàng hình (Zero-Width/Bidi), đồng hình Cyrillic/Hy Lạp & Weird Spaces
   ├── Smart Academic Filter: Tự động loại trừ trích dẫn "..." (>= 8 từ) & mục Tài liệu tham khảo / References
   └── Văn bản sạch chuẩn hóa NFC (sanitized_text) đưa vào lõi Học sâu
        │
        ▼
[3. Lõi Học Sâu Song Ngữ + Cửa Sổ Trượt Đa Đoạn (backend/inference.py)]
   ├── Dynamic Language Router (< 0.05ms): Tự động nhận diện Tiếng Việt ('vi') và Tiếng Anh ('en')
   ├── Tuyến Tiếng Việt ('vi'): PhoBERT base v2 + PyVi + Temp Scaling trên 15.000 mẫu Pre-2022 v3.1 (22 nhóm)
   │     └── Test Accuracy: 98.58% | F1: 98.59% | ROC-AUC: 0.9989 | Sliding-Window (>256 tokens, 65% Mean + 35% Peak)
   ├── Tuyến Tiếng Anh ('en') : Ensemble 3 PLMs (BERT + RoBERTa + Longformer 1024) + Temp Scaling + Stacking
   │     └── Huấn luyện đa miền english_v2 (12.000 mẫu): Acc 95.08%, F1 95.14% | Zero-Shot TOEFL: 83.33% (Recall 100%)
   ├── XAI Evidence Engine    : Tính Burstiness, Perplexity Proxy, MATTR, phát hiện cụm từ khuôn mẫu LLM (EN/VI)
   └── Phân loại 4 cấp độ     : Tự viết hoàn toàn / Nghi vấn AI biên tập / Văn bản lai (Hybrid) / AI toàn phần
        │
        ▼
[4. Lưu Trữ Tập Trung & Chứng Thực Chống Giả Mạo (SQL Server VeriTextAI + Report Builder)]
   ├── Lưu trữ bất biến vào `users`, `detection_history`, `classroom_batches` (cô lập nghiêm ngặt theo RBAC)
   ├── Xuất Bảng Điểm Lớp Học (.XLSX 2 sheet tô màu trực quan / .CSV chuẩn UTF-8 BOM)
   └── Xuất Biên Bản PDF A4 kèm QR Code + Chữ ký số mật mã HMAC-SHA256 (`VT-XXXXXX-XXXXXXXX`) & Cổng `/verify`
```

---

## 📅 Chi Tiết 6 Giai Đoạn Đã Triển Khai Hoàn Tất

### ✅ GIAI ĐOẠN 1: Lõi Anti-Bypass, Bộ Lọc Học Thuật & Explainable AI (XAI)
1. **Module [`backend/text_forensics.py`](file:///c:/DoAn/backend/text_forensics.py):**
   - **Anti-Bypass Detector (`detect_bypass_attempts`):** Phát hiện 15 loại ký tự tàng hình Zero-Width/Bidi, 46 ký tự đồng hình Cyrillic/Hy Lạp giả Latin và khoảng trắng Unicode bất thường; tự động làm sạch văn bản trước khi chấm và xuất danh sách từ bị gắn cờ.
   - **Smart Academic Filter (`apply_academic_filters`):** Tùy chọn loại bỏ trích dẫn nguyên văn dài ($\ge 8$ từ) và mục *Tài liệu tham khảo / References* cuối bài (có cơ chế fallback an toàn nếu bài quá ngắn).
   - **Linguistic XAI Analyzer (`analyze_linguistic_xai`):** Đo Burstiness Score, Perplexity Proxy, MATTR, tỷ lệ lặp từ đầu câu và bộ từ điển cụm sáo rỗng LLM song ngữ Anh - Việt.
   - **Phân loại 4 cấp độ (`classify_authorship_tier`):** `HUMAN_PURE`, `AI_ASSISTED`, `HYBRID_MIXED`, `AI_PURE`.
2. **Preset học thuật linh hoạt:** Hỗ trợ 3 chế độ chấm (`Khóa luận 30%`, `Tiểu luận 50%`, `Tham khảo 70%`) đồng bộ từ Web UI xuống Backend API.

---

### ✅ GIAI ĐOẠN 2: Chấm Bài Cả Lớp (`.ZIP` / Multi-File) & Xuất Bảng Điểm Excel
1. **Module [`backend/batch_processor.py`](file:///c:/DoAn/backend/batch_processor.py) & [`backend/file_parser.py`](file:///c:/DoAn/backend/file_parser.py):**
   - Giải nén an toàn gói `.ZIP` tải từ Moodle / Google Classroom (chặn **Zip Slip**, giới hạn chống **Zip Bomb**, bỏ rác `__MACOSX`), hỗ trợ đọc cả văn bản nằm trong bảng biểu `.DOCX`, `.PDF`, `.DOC` và `.TXT` (tự bóc UTF-8 BOM).
   - Xếp hạng rủi ro tự động toàn lớp (ưu tiên đưa bài dính **Anti-Bypass** lên đầu danh sách).
2. **Quản trị đợt chấm & Xuất báo cáo:**
   - Bảo vệ nghiêm ngặt bằng `require_role(["teacher"])`; lưu vào bảng `classroom_batches` liên kết khóa ngoại với `detection_history`.
   - Xuất bảng điểm **`.XLSX`** (2 sheet: `Bang_Diem_Kiem_Dinh` tô màu theo mức rủi ro + `Tong_Quan_Lop_Hoc`) và **`.CSV` (UTF-8 BOM)** đính kèm mã chứng thực `VT-...` cho từng sinh viên.

---

### ✅ GIAI ĐOẠN 3: Chrome Extension Manifest v3 (`v3.0.0 PRO`) Đồng Bộ SQL Server
1. Tích hợp đăng nhập JWT trực tiếp trong Popup (`chrome.storage.local`), đồng bộ mọi lượt quét từ trình duyệt về lịch sử tài khoản trên SQL Server `VeriTextAI` (có cơ chế tự fallback khách an toàn nếu token hết hạn).
2. Nút **"🌐 Quét Toàn Trang Web Hiện Tại"** bóc tách nội dung bài nộp trên LMS / Google Classroom / Blog kèm Bộ lọc học thuật, cảnh báo Anti-Bypass và dải chỉ số XAI rút gọn.

---

### ✅ GIAI ĐOẠN 4: Bộ Mẫu Demo Đa Định Dạng & Hệ Thống Kiểm Thử Tự Động 100%
1. Gói mẫu [`docs/demo_samples/Lop_K16_TieuLuan_Demo.zip`](file:///c:/DoAn/docs/demo_samples/Lop_K16_TieuLuan_Demo.zip) tích hợp đa định dạng thực tế (`.DOCX` có bảng thông tin sinh viên, `.PDF` văn bản số, `.TXT`), song ngữ Anh - Việt, có bài trích dẫn + *Tài liệu tham khảo*, bài lai, bài biên tập nhẹ và bài cài bẫy Anti-Bypass.
2. Bộ 3 suite kiểm thử tự động ([`test_forensics.py`](file:///c:/DoAn/backend/test_forensics.py), [`test_batch.py`](file:///c:/DoAn/backend/test_batch.py), [`test_api.py`](file:///c:/DoAn/backend/test_api.py)) bao phủ toàn diện luồng nghiệp vụ và bảo mật.

---

### 🚀 GIAI ĐOẠN 5 *(Vượt Kế Hoạch)*: Nâng Cấp Lõi Học Sâu Đa Miền & Cửa Sổ Trượt Bài Dài
1. **Tuyến Tiếng Việt `PhoBERT v3.1` (15.000 mẫu Pre-2022 — 22 phân nhóm):** Mở rộng gấp ~3 lần so với kế hoạch 5K ban đầu, bổ sung Blog tự thuật ngôi thứ nhất `C4-vi <= 2020`, Kỹ thuật CNTT thực tế, Học thuật `.edu.vn` và các biến thể AI Frontier 2025 / Humanized AI — đạt **Accuracy `98.58%`**, **F1 `98.59%`**, **AUC `0.9989`**, hiệu chuẩn xác suất xuất sắc (`ECE = 0.0167`).
2. **Tuyến Tiếng Anh Đa Miền `english_v2` (12.000 mẫu) + Temperature Scaling + Stacking:** Giải quyết triệt để bài toán lệch miền giữa tin tức và tiểu luận sinh viên — đạt **Accuracy `95.08%`**, **F1 `95.14%`** đa miền và nâng kết quả **Zero-Shot TOEFL** từ `64.68%` lên **`83.33%`** (Recall `100%`).
3. **Sliding-Window Multi-Chunk Aggregation & Bảo vệ Tiêu đề ngắn:** Quét trọn vẹn 100% bài khóa luận dài vượt ngưỡng 256/1024 tokens với trọng số kết hợp `65% Mean + 35% Peak Hotspot`, đồng thời chống chấm oan các dòng đề mục ngắn ($\le 4$ từ).

---

### 🔐 GIAI ĐOẠN 6 *(Vượt Kế Hoạch)*: Chữ Ký Số `HMAC-SHA256` + QR Code, Google OAuth 2.0 & Trải Nghiệm SaaS
1. **Chống giả mạo Biên bản Kiểm định:** Ký số `HMAC-SHA256` (`VT-XXXXXX-XXXXXXXX`) và in mã QR lên từng Biên bản PDF, tích hợp Cổng xác thực trực tuyến `GET /api/v1/verify/{audit_code}` ngay trên thanh Navbar.
2. **Tài khoản & Quản trị Lịch sử nâng cao:** Hỗ trợ đăng nhập **Google OAuth 2.0 (1-Click)**, quản lý Hồ sơ & Đổi mật khẩu, cùng thanh lọc nhật ký 4 tiêu chí (`Từ khóa`, `Ngôn ngữ`, `Kết luận`, `Chỉ bài dính Bypass`).
3. **Giao diện hiện đại:** Chuyển đổi Sáng/Tối mượt mà bằng **View Transitions API**, chế độ xem **Mạch văn gốc tô màu kiểu Turnitin** kèm Tooltip xác suất từng câu và sao chép nhanh nhận xét phản hồi cho sinh viên.

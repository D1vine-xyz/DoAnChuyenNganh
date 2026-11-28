# BÁO CÁO TIẾN ĐỘ TOÀN DỰ ÁN — VERITEXT AI PRO (CHỦ ĐỀ 6)

- **Đề tài:** Nghiên cứu phát hiện văn bản do LLM sinh sử dụng mô hình kết hợp Song ngữ (Ensemble PLMs & PhoBERT)
- **Thời gian thực hiện:** `25/10/2026` – `28/11/2026` (Toàn bộ 6 giai đoạn từ khởi tạo đến hoàn thiện nâng cao)
- **Kho lưu trữ Git:** `https://github.com/D1vine-xyz/DoAnChuyenNganh`
- **Trạng thái:** ✅ Hoàn thành vượt kế hoạch đề cương (`28/28` bài kiểm thử tự động Passed)

---

## 1. DÒNG THỜI GIAN PHÁT TRIỂN TOÀN DỰ ÁN (TÓM TẮT THEO MỐC NGÀY)

### Giai đoạn 1: Dữ liệu gốc Tiếng Anh & Pipeline PyTorch (`25/10/2026 – 01/11/2026`)
- **25/10 – 27/10/2026:** Khởi tạo môi trường ảo Python, Git repo; tải và làm sạch dữ liệu tin ngắn **LIAR** (`10,198` train sạch) và tin dài **GossipCop++** (`19,088` bài viết sạch sau tiền xử lý).
- **28/10 – 29/10/2026:** Chuẩn hóa tập kiểm thử ngoại miền **TOEFL** (`252` bài luận: 126 Người / 126 ChatGPT), viết script Gemini API sinh mẫu đối sánh song song và hợp nhất thành tập `final_train_dataset.csv` (`19,148` mẫu).
- **30/10 – 31/10/2026:** Chia dữ liệu phân tầng chống rò rỉ 70/15/15 (`13,403` Train / `2,872` Val / `2,873` Test); khảo sát Tokenizer chứng minh ~25–27% văn bản vượt 512 tokens; đóng gói `dataset.py` với `DynamicPaddingCollator` giảm 40–60% VRAM.
- **01/11/2026:** Thiết lập hạ tầng huấn luyện từ xa trên **Kaggle GPU Tesla T4** (`kaggle_jobs/`).
- **Hạ tầng MCP NotebookLM (`28/10 – 09/11/2026`):** Kết nối sổ tay nghiên cứu lý thuyết lên NotebookLM thông qua `backend/scripts/` (`dang_nhap_notebooklm.bat`, `auth_setup.mjs`, `query_notebooklm.mjs`) phục vụ tra cứu học thuật và soạn thảo báo cáo.

---

### Giai đoạn 2: Huấn luyện 3 Mô hình Nền tảng, Ensemble & Đánh giá Khoa học (`06/11/2026 – 10/11/2026`)
- **06/11 – 07/11/2026:** Xây dựng kiến trúc và fine-tune thành công **BERT** (`bert-base-uncased`) và **RoBERTa** (`roberta-base`) trên Kaggle GPU T4 với Mixed Precision FP16 (`torch.amp`).
- **09/11/2026:** Huấn luyện **Longformer** (`allenai/longformer-base-4096`, ngữ cảnh 1024 tokens với Global Attention + Gradient Accumulation); tối ưu đa luồng CPU và đánh giá chiến lược kết hợp **Ensemble Learning** (`ensemble.py`: Simple Soft Voting, Weighted Soft Voting, Stacking Meta-Classifier).
- **10/11/2026:** Hoàn tất đánh giá Cross-Domain trên 252 bài luận TOEFL, benchmark độ trễ CPU (`1.30s – 1.97s`), đóng gói bộ suy luận độc lập `inference.py` (kèm bóc tách từng câu) và xuất bộ đồ thị khoa học 300 DPI tại [`docs/reports/figures/`](file:///c:/DoAn/docs/reports/figures).

---

### Giai đoạn 3: FastAPI Backend, Chrome Extension & Web Dashboard VeriText AI (`13/11/2026 – 18/11/2026`)
- **13/11/2026:** Triển khai máy chủ RESTful API **FastAPI + Uvicorn** (Lifespan Singleton nạp sẵn mô hình, CORS toàn diện, Swagger UI `/docs`).
- **14/11/2026:** Xây dựng **Chrome Extension Manifest v3** (`frontend/extension/`): Popup giao tiếp FastAPI, Context Menu chuột phải và nút nổi bôi đen `content_script.js`.
- **16/11 – 17/11/2026:** Ra mắt **Web Application Dashboard** dưới thương hiệu **VeriText AI** (`frontend/web/`), hỗ trợ tải tệp `.PDF`, `.DOCX`, `.DOC`, `.TXT`, xuất biên bản PDF (`pdf_report.py`), giao diện Minimalist Paper và chuyển đổi Sáng/Tối mượt mà bằng **View Transitions API**.
- **18/11/2026:** Tái cấu trúc mã nguồn sạch sẽ thành 3 phân hệ chuẩn kỹ thuật phần mềm: `frontend/`, `backend/`, `docs/` và tạo script `start_server.bat`.

---

### Giai đoạn 4: Nâng cấp Song Ngữ (PhoBERT), SQL Server 2022 & Phân quyền RBAC (`19/11/2026 – 23/11/2026`)
- **19/11/2026:** Triển khai tuyến Tiếng Việt (**PhoBERT base v2 + PyVi**) với bộ dữ liệu khởi đầu `1,000` mẫu; tích hợp **Dynamic Language Router** định tuyến tự động Anh/Việt (`< 0.05ms`) cho API, Web và Extension.
- **21/11 – 23/11/2026:**
  - Chuyển đổi lưu trữ từ SQLite sang **Microsoft SQL Server 2022 (`VeriTextAI`)** (`database.py`, `view_db.py`).
  - Tích hợp xác thực **JWT + Bcrypt (12 rounds)** và cô lập dữ liệu theo vai trò **RBAC** (`Giảng viên` / `Sinh viên`).
  - Mở rộng tập Tiếng Việt lên **5,000 mẫu** cân bằng 50/50 và làm mới giao diện Chrome Extension đồng bộ thẩm mỹ với Web Dashboard.

---

### Giai đoạn 5: Nâng tầm "VeriText AI PRO" Thực chiến cho Giảng viên & Nhà trường (`25/11/2026 – 27/11/2026`)
- **25/11 – 26/11/2026:**
  - Xây dựng lõi pháp y văn bản [`backend/text_forensics.py`](file:///c:/DoAn/backend/text_forensics.py): **Anti-Bypass Detector** (bắt ký tự tàng hình Zero-Width & đồng hình Cyrillic/Hy Lạp), **Smart Academic Filter** (bỏ trích dẫn `"..."` $\ge 8$ từ & *Tài liệu tham khảo*), **Explainable AI (XAI)** (Burstiness, Perplexity Proxy, MATTR, AI Clichés EN/VI) và **Phân loại 4 cấp độ tác quyền** (`HUMAN_PURE`, `AI_ASSISTED`, `HYBRID_MIXED`, `AI_PURE`).
  - Triển khai phân hệ **Chấm Bài Lớp Học (`.ZIP` / Multi-file)** (`batch_processor.py`, `RBAC Teacher Only`): giải nén chống Zip Slip/Bomb, đọc cả bảng biểu trong `.DOCX`, tự bóc UTF-8 BOM, xuất bảng điểm **Excel `.XLSX` (2 sheet tô màu)** & **`.CSV`**.
  - Nâng cấp **Đăng nhập/Đăng ký**, hồ sơ cá nhân, đổi mật khẩu và **1-Click Google OAuth 2.0** (`auth.py`).
  - Nâng cấp **Chrome Extension v3.0 PRO**: đồng bộ tài khoản JWT về SQL Server và quét toàn trang LMS/Classroom chỉ với 1 click.
  - Mở rộng dữ liệu Tiếng Việt lên **8,000 mẫu Pre-2022 (12 thể loại)**.
- **27/11/2026:**
  - Thêm cơ chế **Sliding-Window Multi-Chunk Aggregation** (`65%` Mean + `35%` Peak Hotspot) quét trọn vẹn 100% bài dài vượt 256/1024 tokens và chống chấm oan tiêu đề ngắn ($\le 4$ từ).
  - Thêm **Chữ ký số `HMAC-SHA256` (`VT-XXXXXX-XXXXXXXX`) + Mã QR** trên Biên bản PDF và Cổng xác thực chống giả mạo `GET /api/v1/verify/{audit_code}`.
  - Bổ sung chế độ xem **Mạch văn gốc tô màu kiểu Turnitin**, thanh lọc nhật ký 4 tiêu chí, bộ 3 biểu đồ 300 DPI mới (`generate_vietnamese_figures.py`) và nhận diện logo thương hiệu mới.

---

### Giai đoạn 6: Chuẩn hóa Dữ liệu Đa miền v2 / v3.1, Hiệu chuẩn Xác suất & Tự động hóa Antigravity (`28/11/2026`)
- **Rạng sáng 28/11/2026:**
  - Nâng cấp dữ liệu Tiếng Anh đa miền **`english_v2` (12,000 mẫu)** (News + DAIGT Essays + HC3 + Humanized AI), áp dụng **Temperature Scaling** (`T_BERT=1.1454`, `T_RoBERTa=1.3377`, `T_Longformer=1.3900`) và **Stacking Meta-Classifier** (trọng số `BERT 46.61%`, `RoBERTa 37.50%`, `Longformer 15.89%`).
  - Nâng cấp dữ liệu Tiếng Việt **`vietnamese_v3.1` (15,000 mẫu Pre-2022, 22 phân nhóm)** bổ sung Blog tự thuật `C4-vi <= 2020`, Kỹ thuật CNTT thực tế, Học thuật `.edu.vn` kết hợp Label Smoothing và khóa hiệu chuẩn `T >= 1.0`.
- **Chiều 28/11/2026:**
  - Thiết lập quy tắc dự án [`AGENTS.md`](file:///c:/DoAn/AGENTS.md) và Skill tự động đẩy/tải job Kaggle GPU [`kaggle-pipeline`](file:///c:/DoAn/.agents/skills/kaggle-pipeline/SKILL.md).
  - Nâng cấp gói mẫu demo [`docs/demo_samples/Lop_K16_TieuLuan_Demo.zip`](file:///c:/DoAn/docs/demo_samples/Lop_K16_TieuLuan_Demo.zip) lên **7 bài đa định dạng** (4 `.docx` kèm bảng, 2 `.pdf`, 1 `.txt` song ngữ EN/VI, bao phủ đủ Anti-Bypass, Bài lai, Biên tập nhẹ, Tự viết & AI toàn phần).
  - Đồng bộ toàn bộ giao diện Web (`index.html`, `app.js`, `style.css`), Biên bản PDF (`pdf_report.py`), `README.md` và `KE_HOACH_NANG_CAP_CHUYEN_NGANH.md`; chạy qua **`28/28` bài kiểm thử tự động (`100% Passed`)** và kiểm thử E2E trên trình duyệt (`0` lỗi Console/Network).

---

## 2. BẢNG TỔNG HỢP SỐ LIỆU THỰC NGHIỆM HIỆN HÀNH (`backend/output/`)

| Tuyến / Mô hình | Tập dữ liệu kiểm thử | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Ghi chú |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PhoBERT v2 + PyVi** *(VI v3.1)* | `vietnamese_v3_test.csv` (`2,250` mẫu, 22 nhóm) | **98.58%** | **97.89%** | **99.29%** | **98.59%** | **0.9989** | `ECE = 0.0167`, `T = 1.0` |
| **Stacking + Temp Scaling** *(EN v2)* | `english_v2_test.csv` (`1,200` mẫu đa miền) | **95.08%** | **93.98%** | **96.33%** | **95.14%** | **98.60%** | Chiến lược dẫn đầu tuyến Tiếng Anh |
| **Weighted Soft Voting** *(EN v2)* | `english_v2_test.csv` (`1,200` mẫu đa miền) | 94.25% | 92.34% | 96.50% | 94.38% | 98.56% | Trọng số: BERT `46.61%`, RoBERTa `37.50%`, Longformer `15.89%` |
| **Zero-Shot TOEFL** *(Stacking EN v2)* | `toefl_dataset.csv` (`252` bài luận, `th = 0.5`) | **83.33%** | 75.00% | **100.00%** | **85.71%** | **96.82%** | Bắt trúng `126/126` bài ChatGPT không cần chỉnh tay ngưỡng |

---

## 3. KIỂM CHỨNG CHẤT LƯỢNG & VẬN HÀNH HỆ THỐNG

- **Kiểm thử Tự động (`backend/`):** `28/28` test cases Passed (`test_forensics.py`: 8 tests, `test_batch.py`: 3 tests, `test_api.py`: 17 tests).
- **Kiểm thử E2E Trình duyệt (`http://localhost:8000`):** Xác nhận đăng nhập Giảng viên mẫu, quét Anti-Bypass + XAI + Mạch văn tô màu, Chấm lớp `.ZIP` hoạt động ổn định với `0` lỗi JavaScript và `0` lỗi HTTP `4xx/5xx`.
- **Tài khoản Demo mặc định (SQL Server `VeriTextAI`):**
  - Giảng viên: `giangvien@veritext.ai` / `123456`
  - Sinh viên: `sinhvien@veritext.ai` / `123456`

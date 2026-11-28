# VeriText AI PRO — Hệ Thống Thẩm Định Văn Bản AI Song Ngữ & Liêm Chính Học Thuật (Ensemble PLMs & PhoBERT)

Đồ án Chuyên ngành Kỹ thuật Phần mềm (**Chủ đề 6**) nghiên cứu và xây dựng nền tảng **VeriText AI PRO** phát hiện văn bản do các Mô hình Ngôn ngữ Lớn (LLMs như ChatGPT, Gemini, Claude, Gemma) tạo ra, tích hợp **Định tuyến Ngôn ngữ Thông minh (Dynamic Language Router)**, **Pháp y Văn bản chống qua mặt (Anti-Bypass)**, **Giải thích Ngôn ngữ học (Explainable AI - XAI)**, **Chấm bài lớp học hàng loạt (`.ZIP` $\rightarrow$ Excel)**, **Chứng thực Biên bản PDF bằng chữ ký số HMAC-SHA256 & QR Code**, cùng CSDL **Microsoft SQL Server 2022 (`VeriTextAI`)** và **Phân quyền RBAC (JWT + Bcrypt + Google OAuth 2.0)**:

- **🇬🇧 Tuyến Tiếng Anh (`en` — Đa miền v2, 12.000 mẫu):** Học kết hợp (**Ensemble Learning**) từ 3 mô hình tiền huấn luyện **BERT**, **RoBERTa** và **Longformer (1024 tokens)** trên tập dữ liệu đa miền `english_v2_*.csv` (`12.000` mẫu cân bằng 50/50 gồm Tin tức + Bài luận sinh viên DAIGT + Học thuật HC3 + Humanized AI), kết hợp hiệu chuẩn xác suất **Post-hoc Temperature Scaling** và **Stacking Meta-Classifier** — đạt **Accuracy `95.08%`**, **Precision `93.98%`**, **Recall `96.33%`**, **F1-Score `95.14%`**, **AUC-ROC `98.60%`** trên tập Test đa miền; nâng khả năng tổng quát hóa **Zero-Shot ngoại miền TOEFL** (`252` bài luận, `th = 0.5`) từ `64.68%` lên **`83.33%` Accuracy**, **`100.00%` Recall**, **`85.71%` F1-Score**, **`96.82%` AUC-ROC**.
- **🇻🇳 Tuyến Tiếng Việt (`vi` — Tiền ChatGPT v3.1, 15.000 mẫu):** Mô hình chuyên biệt **PhoBERT base v2** (`vinai/phobert-base-v2`) kết hợp tách từ **PyVi**, Label Smoothing và Temperature Scaling huấn luyện trên tập dữ liệu **`vietnamese_v3_*.csv` (15.000 mẫu chuẩn Tiền ChatGPT $\le 2022$, 22 phân nhóm**: Báo chí, Học thuật/Bách khoa, Blog tự thuật ngôi thứ nhất `C4-vi <= 2020`, Kỹ thuật CNTT thực tế, Học thuật `.edu.vn`, Giáo dục, Pháp luật $\le 2022$, Sách tri thức, Văn học kinh điển, Truyện ngắn, Thơ ca đối trọng Multi-LLM AI Frontier 2025, AI Paraphrase & Humanized AI) — đạt **Accuracy `98.58%`**, **Precision `97.89%`**, **Recall `99.29%`**, **F1-Score `98.59%`**, **ROC-AUC `0.9989`** (`ECE = 0.0167`).

---

## 📌 Cấu Trúc Dự Án (Fullstack 3-Tier Architecture)

```text
DoAn/
├── AGENTS.md                          # Quy tắc & bộ nhớ thường trực của dự án cho AI Agent
├── .agents/skills/kaggle-pipeline/    # Skill tự động hóa đẩy & thu hoạch mô hình trên Kaggle GPU T4
│
├── frontend/                          # Tầng Giao diện Người dùng (Client-Side)
│   ├── web/                           # Web Application Dashboard SPA (HTML5, Modern CSS, Vanilla JS)
│   │   ├── index.html                 # 4 Tab nghiệp vụ: Thẩm định, Chấm lớp (.ZIP), Lịch sử, Kiến trúc + Xác thực PDF
│   │   ├── style.css                  # Minimalist Paper & Graphite Dark/Light Mode (View Transitions API)
│   │   └── app.js                     # Logic suy luận, Anti-Bypass, XAI, Turnitin Highlight, Batch Excel, Google GIS
│   ├── extension/                     # Tiện ích trình duyệt Chrome Extension Manifest v3 (v3.0.0 PRO)
│   │   ├── manifest.json              # Khai báo quyền, Service Worker, Context Menus (v3.0.0)
│   │   ├── background.js              # Menu chuột phải: Quét đoạn bôi đen & Quét toàn trang LMS
│   │   ├── popup.html / popup.js      # Đăng nhập JWT đồng bộ SQL Server, Quét toàn trang LMS, Anti-Bypass & XAI
│   │   ├── content_script.js          # Nút nổi bôi đen, Quét toàn trang & Modal nổi trực tiếp trên trang web
│   │   └── icons/                     # Bộ biểu tượng ứng dụng đa kích thước
│   └── README.md                      # Hướng dẫn cài đặt và sử dụng Frontend
│
├── backend/                           # Tầng Dịch vụ Máy chủ, Pháp y Văn bản & Lõi AI (Server & AI Pipeline)
│   ├── api.py                         # RESTful API FastAPI v3.0 (Non-blocking Threadpool, RBAC, HMAC Verify)
│   ├── run_server.py                  # Script khởi động server & tự động mở Web Dashboard (http://localhost:8000)
│   ├── inference.py                   # Lõi suy luận Song ngữ + Sliding-Window Multi-Chunk + Temperature Scaling
│   ├── text_forensics.py              # Anti-Bypass (Zero-Width/Homoglyph), Bộ lọc học thuật, XAI & 4 cấp độ
│   ├── batch_processor.py             # Giải nén .ZIP an toàn (chống Zip Slip/Bomb) & Xuất Excel (.XLSX) / .CSV
│   ├── auth.py                        # Xác thực JWT, Bcrypt (12 rounds), RBAC (Teacher/Student) & Google OAuth 2.0
│   ├── database.py                    # Quản trị CSDL Microsoft SQL Server 2022 (VeriTextAI) & chữ ký HMAC-SHA256
│   ├── file_parser.py                 # Trích xuất văn bản đa định dạng (.txt UTF-8 BOM, .docx + bảng, .doc, .pdf)
│   ├── pdf_report.py                  # Sinh Biên bản Kiểm định PDF A4 (ReportLab Unicode + QR Code + XAI + Anti-Bypass)
│   ├── models.py                      # Kiến trúc PyTorch cho BERT, RoBERTa, Longformer (1024), PhoBERT v2
│   ├── test_forensics.py              # Unit tests cho Anti-Bypass, Bộ lọc học thuật, XAI & phân loại 4 cấp độ
│   ├── test_batch.py                  # Unit tests cho giải nén .ZIP an toàn, đọc bảng .DOCX & xuất Excel/CSV
│   ├── test_api.py                    # Suite 16 integration tests E2E (Song ngữ, Sliding-Window, QR Verify, RBAC)
│   ├── data/                          # Dữ liệu chuẩn hóa: english_v2_*.csv (12K), vietnamese_v3_*.csv (15K), TOEFL
│   ├── output/                        # Checkpoints (*_best.pt), Stacking Meta-Learner, Temperature JSON, kết quả Test
│   ├── kaggle_jobs/                   # 5 Kernel cấu hình huấn luyện từ xa trên Kaggle GPU Tesla T4
│   └── scripts/                       # Kịch bản tiện ích & kết nối NotebookLM MCP
│
├── docs/                              # Tài liệu đồ án, lộ trình nâng cấp, biểu đồ khoa học & gói Demo
│   ├── CHỦ ĐỀ 6.docx                  # Phiếu giao đề tài chính thức
│   ├── KE_HOACH_NANG_CAP_CHUYEN_NGANH.md # Báo cáo kiến trúc & các giai đoạn nâng cấp VeriText AI PRO
│   ├── demo_samples/                  # Gói mẫu đa định dạng `Lop_K16_TieuLuan_Demo.zip` (.docx, .pdf, .txt)
│   └── reports/figures/               # Bộ 9 biểu đồ khoa học chuẩn 300 DPI phục vụ Luận văn & Slide bảo vệ
│
├── start_server.bat                   # 🚀 Script 1-click khởi chạy toàn bộ hệ thống trên Windows
└── README.md                          # Tài liệu tổng quan dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Cài đặt môi trường & CSDL
- **Python:** Khuyến nghị Python 3.10+ (môi trường ảo tại `c:\DoAn\venv`).
- **CSDL:** Microsoft SQL Server 2022 + `ODBC Driver 17 for SQL Server` (hệ thống tự khởi tạo database `VeriTextAI`, 3 bảng `users`, `detection_history`, `classroom_batches` và 2 tài khoản mẫu ngay lần chạy đầu tiên).

```powershell
# Tạo và kích hoạt môi trường ảo (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Cài đặt thư viện Backend
pip install -r backend/requirements.txt
```

### 2. Khởi chạy Hệ thống (Backend API + Web Dashboard)
```powershell
# Cách 1: Nhấp đúp hoặc chạy file batch ở thư mục gốc
.\start_server.bat

# Cách 2: Khởi chạy qua môi trường ảo Python
.\venv\Scripts\python.exe backend\run_server.py
```
- **Web Dashboard SPA:** `http://localhost:8000`
- **Tài liệu Swagger UI:** `http://localhost:8000/docs`
- **Tài khoản Demo tích hợp sẵn:**
  - 👨‍🏫 **Giảng viên:** `giangvien@veritext.ai` / `123456` *(Toàn quyền chấm lớp `.ZIP`, xuất Excel, xem toàn bộ lịch sử)*
  - 🎓 **Sinh viên:** `sinhvien@veritext.ai` / `123456` *(Thẩm định cá nhân, cô lập lịch sử theo RBAC)*

### 3. Chạy Bộ Kiểm Thử Tự Động (Unit & Integration Tests)
```powershell
cd backend
..\venv\Scripts\python.exe -m unittest test_forensics.py test_batch.py test_api.py -v
```

### 4. Huấn luyện từ xa trên Kaggle GPU (Tesla T4)
```powershell
# Đẩy kernel lên Kaggle GPU T4
kaggle kernels push -p c:\DoAn\backend\kaggle_jobs\train_phobert

# Kiểm tra trạng thái thực thi
kaggle kernels status quannguyenxx/do-an-train-phobert

# Tải kết quả đánh giá về máy (sao lưu *_best.pt trước khi ghi đè)
kaggle kernels output quannguyenxx/do-an-train-phobert -p c:\DoAn\backend\output
```

---

## 📊 Thống Kê Tập Dữ Liệu Huấn Luyện & Kiểm Thử (Bản Nâng Cấp Mới Nhất)

| Bộ dữ liệu | Tổng số mẫu | Phân chia Train / Val / Test | Tỷ lệ Nhãn | Đặc tả & Mục đích sử dụng |
| :--- | :---: | :---: | :---: | :--- |
| **`english_v2_*.csv`** *(Đa miền v2)* | **12,000** | `9,600` / `1,200` / `1,200` | 50% Người / 50% AI | Đa miền Tiếng Anh: Tin tức + Bài luận sinh viên DAIGT + Học thuật HC3 + Humanized AI |
| **`vietnamese_v3_*.csv`** *(Pre-2022 v3.1)* | **15,000** | `10,500` / `2,250` / `2,250` | 50% Người / 50% AI | **22 phân nhóm** Tiếng Việt: Văn bản người xuất bản $\le 2022$ đối trọng Multi-LLM 2025 & Paraphrase |
| **`toefl_dataset.csv`** *(Zero-Shot OOD)* | **252** | Kiểm thử độc lập 100% | 126 Người / 126 AI | Đánh giá khả năng tổng quát hóa ngoại miền (**Cross-Domain Zero-Shot**) trên bài luận học thuật |
| *(Tập cơ sở Giai đoạn v1)* `train/val/test.csv` | *19,148* | *`13,403` / `2,872` / `2,873`* | *57.1% Người / 42.9% AI* | *Tập đối chứng đơn miền tin tức ban đầu (GossipCop + LIAR) dùng để chứng minh hiệu quả nâng cấp đa miền* |

---

## 🏆 Kết Quả Thực Nghiệm Hệ Thống Mô Hình Song Ngữ

### 1. Tuyến Tiếng Anh Đa Miền (`english_v2` — 12.000 mẫu & Ngoại miền `TOEFL` — 252 bài luận)

Để khắc phục hiện tượng **Dịch chuyển miền (Domain Shift)** khi mô hình chỉ học trên tin tức gặp bài luận học thuật của sinh viên, hệ thống nâng cấp lên tập đa miền `english_v2` (12.000 mẫu), hiệu chuẩn xác suất **Temperature Scaling** (`T_BERT = 1.1454`, `T_RoBERTa = 1.3377`, `T_Longformer = 1.3900`) và dung hợp qua **Stacking Meta-Classifier** (trọng số tối ưu `ensemble_weights.json`: **BERT `46.61%`**, **RoBERTa `37.50%`**, **Longformer `15.89%`**):

| Mô hình / Chiến lược (`threshold = 0.5`) | Multi-Domain Acc | Precision | Recall | F1-Score | AUC-ROC | Zero-Shot TOEFL Acc | TOEFL Recall | TOEFL F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BERT** (`bert-base-uncased`) | 93.92% | 91.76% | 96.50% | 94.07% | 98.50% | 81.75% | **100.00%** | 84.56% |
| **RoBERTa** (`roberta-base`) | 92.33% | 88.72% | **97.00%** | 92.68% | 96.70% | 75.40% | **100.00%** | 80.25% |
| **Longformer** (`longformer-base-4096`, 1024) | 90.83% | 86.57% | 96.67% | 91.34% | 98.30% | 75.79% | **100.00%** | 80.51% |
| **Ensemble (Simple Soft Voting)** | 93.17% | 90.09% | **97.00%** | 93.42% | **98.69%** | 76.59% | **100.00%** | 81.03% |
| **Ensemble (Weighted Soft Voting)** | 94.25% | 92.34% | 96.50% | 94.38% | 98.56% | 80.16% | **100.00%** | 83.44% |
| **🏆 Ensemble (Stacking + Temp Scaling)** | **95.08%** | **93.98%** | 96.33% | **95.14%** | 98.60% | **83.33%** | **100.00%** | **85.71%** |

> 💡 **Giá trị khoa học nổi bật:** So với mô hình Giai đoạn v1 đơn miền (vốn đạt `96.52%` trên miền tin tức nhưng giảm xuống `64.68%` ở ngưỡng mặc định `0.5` khi gặp bài luận TOEFL), kiến trúc **Ensemble Stacking + Temperature Scaling v2** vừa giữ vững độ chính xác đa miền **`95.08%`**, vừa đưa độ chính xác **Zero-Shot trên TOEFL tăng vọt `+18.65%` (lên `83.33%`)** mà không cần chỉnh tay ngưỡng, đồng thời bắt trúng **100% (126/126)** bài luận ChatGPT!

---

### 2. Tuyến Tiếng Việt Bản Địa Hóa (`PhoBERT base v2` + `PyVi` — v3.1 15.000 mẫu, 22 phân nhóm)

Nhằm triệt tiêu nguy cơ "nhiễm bẩn dữ liệu hậu ChatGPT" ở các nguồn web sau năm 2023 và chống chấm oan văn phong tự sự/kỹ thuật của người thật, tuyến Tiếng Việt được huấn luyện trên bộ dữ liệu **`vietnamese_v3_*.csv` (15.000 mẫu, 22 phân nhóm)**:

| Phiên bản Tuyến Tiếng Việt | Quy mô & Đa dạng Thể loại | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Ghi chú cải tiến |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Giai đoạn 2 (`PhoBERT 8K Pre-2022`)* | 8,000 mẫu (12 thể loại) | 99.08% | 98.52% | 99.67% | 99.09% | 0.9991 | Tăng `+22.9%` đa thể loại so với bản 5K đời đầu |
| **🏆 Hiện hành (`PhoBERT v3.1 Pre-2022`)** | **15,000 mẫu (22 phân nhóm)** | **98.58%** | **97.89%** | **99.29%** | **98.59%** | **0.9989** | Bổ sung Blog tự thuật `C4-vi <=2020` (`96.67%`), CNTT thực tế (`96.00%`), Pháp luật/Giáo dục/Báo chí (`100%`), AI Humanized (`100%`), AI Paraphrase (`97.56%`); sai số hiệu chuẩn cực thấp `ECE = 0.0167` |

---

## 🛡️ Các Phân Hệ Kỹ Thuật Vượt Trội Của `VeriText AI PRO`

### 1. Cơ Chế Cửa Sổ Trượt Đa Đoạn (`Sliding-Window Multi-Chunk Aggregation`)
- Tự động chia các bài tiểu luận / khóa luận dài vượt ngưỡng ngữ cảnh (`> 256` tokens với PhoBERT hoặc `> 1024` tokens với Longformer) thành các đoạn chồng lấp thông minh.
- Hợp nhất theo công thức **65% Trung bình trọng số độ dài + 35% Đỉnh cục bộ (Peak Hotspot)** để phát hiện chính xác thủ thuật trộn *"Mở bài tự viết + Thân bài dán AI"*.
- Tự động bảo vệ các tiêu đề mục ngắn ($\le 4$ từ như `1. Mở đầu`, `Kết luận:`) ở trạng thái trung tính, tránh báo động giả.

### 2. Pháp Y Văn Bản (`Anti-Bypass`), Bộ Lọc Học Thuật & Giải Thích Ngôn Ngữ Học (`XAI`)
- **Anti-Bypass Detector (`text_forensics.py`):** Phát hiện và khôi phục tức thì các kỹ thuật qua mặt máy dò: chèn **ký tự tàng hình Zero-Width** (`ZWSP`, `ZWNJ`, `ZWJ`, `Soft Hyphen`, `Bidi`), tráo **ký tự đồng hình Cyrillic/Hy Lạp giả Latin** (`а`, `е`, `о`, `р`, `с`...) và khoảng trắng Unicode bất thường.
- **Smart Academic Filter:** Tùy chọn tự động loại trừ trích dẫn nguyên văn trong ngoặc kép (`"..."` $\ge 8$ từ) và phần *Tài liệu tham khảo / References* cuối bài để không chấm oan sinh viên.
- **Explainable AI (XAI) & Phân loại 4 Cấp độ:** Cung cấp bằng chứng minh bạch cho giảng viên gồm **Burstiness Score** (độ biến thiên nhịp câu), **Perplexity Proxy**, **MATTR** (đa dạng từ vựng), danh sách **cụm từ khuôn mẫu LLM** (EN/VI) và phân loại 4 mức: `👤 Tự viết hoàn toàn` $\rightarrow$ `✍️ Nghi vấn AI biên tập` $\rightarrow$ `🧩 Văn bản lai (Hybrid)` $\rightarrow$ `🤖 AI sinh toàn phần`.

### 3. Chấm Bài Lớp Học Hàng Loạt (`.ZIP` $\rightarrow$ Bảng Điểm Excel `.XLSX` / `.CSV`)
- Dành riêng cho vai trò **Giảng viên (`RBAC Teacher Only`)**: kéo thả 1 gói `.ZIP` tải từ Moodle / Google Classroom hoặc chọn tới 100 tệp (`.docx` có bảng biểu, `.pdf`, `.doc`, `.txt` UTF-8 BOM).
- Giải nén an toàn chống lỗ hổng **Zip Slip / Zip Bomb**, tự động bóc tách MSSV & Họ tên, xếp hạng rủi ro toàn lớp (ưu tiên đưa bài dính **Anti-Bypass** lên đầu bảng) và xuất bảng điểm **Excel `.XLSX` (2 sheet định dạng màu)** hoặc **`.CSV` chuẩn UTF-8 BOM**.

### 4. Biên Bản Kiểm Định PDF + Chữ Ký Số `HMAC-SHA256` & Mã QR Chống Giả Mạo
- Xuất chứng nhận PDF khổ A4 chuyên nghiệp bằng `ReportLab` Unicode, tích hợp đầy đủ khối cảnh báo Anti-Bypass, bảng XAI, đóng góp từng mô hình PLM và danh sách từng câu tô màu.
- Mỗi biên bản được gắn **Mã chứng thực chữ ký số mật mã `HMAC-SHA256`** (`VT-XXXXXX-XXXXXXXX`) kèm **QR Code** đối chiếu trực tiếp với bản ghi gốc bất biến trong SQL Server qua cổng `GET /api/v1/verify/{audit_code}`.

### 5. Chrome Extension Manifest v3 (`v3.0.0 PRO`)
- Tích hợp đăng nhập JWT ngay trong Popup để đồng bộ mọi lượt quét trên trình duyệt thẳng về tài khoản trên SQL Server `VeriTextAI`.
- Hỗ trợ **Quét Toàn Trang LMS / Google Classroom / Blog 1-Click**, quét đoạn bôi đen bằng nút nổi `🤖 Quét AI` và menu chuột phải kèm cảnh báo Anti-Bypass & XAI rút gọn.

---

## 📡 Danh Mục RESTful API Endpoints Chính (`backend/api.py`)

| Phương thức | Đường dẫn Endpoint | Mô tả Chức năng |
| :---: | :--- | :--- |
| **`GET`** | `/` | Phục vụ giao diện **Web Application Dashboard (SPA)** |
| **`GET`** | `/health` | Trạng thái máy chủ, thiết bị thực thi (`CPU`/`CUDA`) và tình trạng 4 mô hình |
| **`POST`** | `/api/v1/detect` | Dự đoán nhanh văn bản đơn lẻ (Song ngữ EN/VI + phân rã mô hình) |
| **`POST`** | `/api/v1/detect/sentences` | Phân tích kèm bóc tách cấp câu (phục vụ Chrome Extension & đồng bộ JWT) |
| **`POST`** | `/api/v1/detect/web` | Phân tích toàn diện (Song ngữ + Sliding-Window + Anti-Bypass + Lọc học thuật + XAI + 4 mức) |
| **`POST`** | `/api/v1/detect/upload` | Tải lên và thẩm định tệp `.PDF`, `.DOCX` (kèm bảng), `.DOC`, `.TXT` |
| **`POST`** | `/api/v1/detect/batch` | Phân tích hàng loạt tối đa 50 văn bản JSON trong một request |
| **`POST`** | `/api/v1/classroom/batch-scan` | **Chấm bài cả lớp (`.ZIP` / Multi-file)** — *Teacher RBAC Only* |
| **`GET`** | `/api/v1/classroom/batches` | Danh sách các đợt chấm lớp học đã lưu trên SQL Server — *Teacher RBAC* |
| **`GET`** | `/api/v1/classroom/batches/{id}` | Chi tiết bảng xếp hạng rủi ro sinh viên của đợt chấm — *Teacher RBAC* |
| **`GET`** | `/api/v1/classroom/batches/{id}/export` | **Xuất Bảng Điểm `.XLSX` (2 sheet tô màu) hoặc `.CSV` (UTF-8 BOM)** — *Teacher RBAC* |
| **`DELETE`** | `/api/v1/classroom/batches/{id}` | Xóa đợt chấm lớp học — *Teacher RBAC* |
| **`POST`** | `/api/v1/auth/register` / `login` / `google` | Đăng ký, đăng nhập JWT + Bcrypt và **1-Click Google OAuth 2.0** |
| **`GET/PUT`** | `/api/v1/auth/me` / `profile` / `change-password` | Hồ sơ người dùng, chuyển đổi vai trò RBAC và đổi mật khẩu an toàn |
| **`GET/DEL`** | `/api/v1/history` • `/api/v1/history/{id}` | Nhật ký thẩm định SQL Server kèm bộ lọc (`q`, `language`, `verdict`, `bypass_only`) |
| **`GET`** | `/api/v1/stats` | Thống kê KPI tổng quan theo phân quyền RBAC |
| **`GET`** | `/api/v1/report/pdf/{id}` | **Xuất Biên bản Kiểm định PDF A4** kèm QR Code & chữ ký số `HMAC-SHA256` |
| **`GET`** | `/api/v1/verify/{audit_code}` | **Cổng Xác thực Chống giả mạo Biên bản** đối chiếu `HMAC-SHA256` với SQL Server |

---

## 🎨 Bộ 9 Biểu Đồ Khoa Học 300 DPI (`docs/reports/figures/`)

1. `roc_curves_comparison.png` — Đường cong ROC & AUC đối sánh các mô hình và chiến lược Ensemble.
2. `confusion_matrix_weighted_ensemble.png` — Ma trận nhầm lẫn chi tiết của mô hình kết hợp.
3. `confusion_matrices_all_models.png` — Lưới đối sánh 6 ma trận nhầm lẫn cạnh nhau.
4. `metrics_comparison_bar.png` — Biểu đồ cột so sánh 4 thang đo (Accuracy, Precision, Recall, F1).
5. `ensemble_weights_distribution.png` — Trực quan hóa đóng góp trọng số các mô hình thành phần.
6. `toefl_domain_shift_comparison.png` — Phân tích hiện tượng Dịch chuyển miền (Domain Shift) trên tập TOEFL.
7. `vietnamese_5k_vs_8k_genre_comparison.png` — Đối sánh bước nhảy vọt đa thể loại của tuyến Tiếng Việt Pre-2022.
8. `vietnamese_8k_confusion_and_roc.png` — Ma trận nhầm lẫn & đường cong ROC-AUC tuyến Tiếng Việt Pre-2022.
9. `length_sensitivity_analysis.png` — Phân tích độ nhạy theo độ dài văn bản (Short / Medium / Long) song ngữ Việt - Anh.

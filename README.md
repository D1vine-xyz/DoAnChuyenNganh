# Nghiên cứu Phát hiện Văn bản do LLM sinh sử dụng Mô hình kết hợp Song ngữ (Ensemble PLMs & PhoBERT)

Đồ án chuyên ngành (Chủ đề 6) nghiên cứu và xây dựng hệ thống phát hiện văn bản do các Mô hình Ngôn ngữ Lớn (LLMs như ChatGPT, Gemini) tạo ra, hỗ trợ **Song ngữ Anh - Việt** với kiến trúc Định tuyến Thông minh (Dynamic Language Router), CSDL **Microsoft SQL Server 2022 (`VeriTextAI`)** và **Phân quyền RBAC (JWT + Bcrypt)**:
- **Tuyến Tiếng Anh**: Kỹ thuật học kết hợp (Ensemble Learning) từ 3 mô hình tiền huấn luyện lớn: **Longformer (1024 tokens)**, **BERT** và **RoBERTa** (Đạt độ chính xác **96.52%**, Precision **98.38%**).
- **Tuyến Tiếng Việt**: Mô hình bản địa hóa chuyên sâu **PhoBERT base v2** (VinAI) kết hợp tách từ **PyVi** huấn luyện trên bộ dữ liệu **8.000 mẫu đa thể loại chuẩn Tiền ChatGPT ($\le 2022$)** gồm 12 phân nhóm (Học thuật/Bách khoa, Báo chí, Pháp luật $\le 2022$, Sách tri thức, Văn học kinh điển, Truyện ngắn, Thơ ca cổ điển + Multi-LLM AI & AI Paraphrase), đạt **Accuracy 99.08%**, **Precision 98.52%**, **Recall 99.67%**, **F1-Score 99.09%**, **ROC-AUC 0.9991**.

---

## 📌 Cấu trúc dự án (Fullstack Architecture: Frontend - Backend - Docs)

```text
DoAn/
├── frontend/                          # Toàn bộ giao diện người dùng (Client-Side)
│   ├── web/                           # Web Application Dashboard SPA (HTML5, Modern CSS, Vanilla JS)
│   │   ├── index.html                 # Giao diện thẩm định, Chấm bài lớp học (.ZIP), Hồ sơ & Google Sign-In
│   │   ├── style.css                  # Thiết kế Minimalist Paper, Dark/Light Mode, CSS Variables
│   │   └── app.js                     # Logic API, Anti-Bypass, XAI, Classroom Batch, Auth & Google GIS
│   ├── extension/                     # Tiện ích trình duyệt Chrome Extension Manifest v3 (v3.0.0 PRO)
│   │   ├── manifest.json              # Khai báo quyền, Service Worker, Context Menus (v3.0.0)
│   │   ├── background.js              # Menu chuột phải: Quét bôi đen & Quét toàn trang LMS
│   │   ├── popup.html / popup.js      # Popup đăng nhập JWT, Quét toàn trang, Anti-Bypass & XAI rút gọn
│   │   ├── content_script.js          # Nút nổi bôi đen, Quét toàn trang & Modal đồng bộ SQL Server
│   │   └── icons/                     # Bộ biểu tượng ứng dụng đa kích thước
│   └── README.md                      # Hướng dẫn chi tiết sử dụng và cài đặt frontend
│
├── backend/                           # Toàn bộ xử lý máy chủ & mô hình AI (Server & AI Pipeline)
│   ├── api.py                         # Máy chủ RESTful API FastAPI v3.0 (Non-blocking Threadpool, RBAC)
│   ├── run_server.py                  # Script khởi động server & tự động mở Web Dashboard
│   ├── inference.py                   # Bộ suy luận Song ngữ (Language Router, PhoBERT, Ensemble PLMs)
│   ├── text_forensics.py              # Chống lách luật (Anti-Bypass), Lọc học thuật, XAI & 4 cấp độ
│   ├── batch_processor.py             # Giải nén .ZIP an toàn (chống Zip Slip/Bomb) & Xuất Excel/CSV
│   ├── auth.py                        # Xác thực JWT, RBAC (Giảng viên/Sinh viên) & Google OAuth 2.0
│   ├── database.py                    # Quản lý CSDL Microsoft SQL Server 2022 (VeriTextAI)
│   ├── file_parser.py                 # Bộ bóc tách tệp tải lên (.txt UTF-8 BOM, .docx + bảng, .doc, .pdf)
│   ├── pdf_report.py                  # Trình sinh chứng nhận thẩm định PDF A4 (ReportLab, XAI + Anti-Bypass)
│   ├── models.py                      # Định nghĩa kiến trúc PyTorch cho BERT, RoBERTa, Longformer, PhoBERT
│   ├── test_forensics.py              # Unit tests cho Anti-Bypass, Bộ lọc học thuật, XAI & 4 cấp độ
│   ├── test_batch.py                  # Unit tests cho giải nén .ZIP an toàn, .DOCX bảng & xuất Excel/CSV
│   ├── test_api.py                    # Suite 15 integration tests toàn diện (AI, RBAC, Batch, Auth, Google, Ext)
│   ├── view_db.py                     # Tiện ích xem nhanh dữ liệu trực tiếp trong SQL Server VeriTextAI
│   ├── requirements.txt               # Danh mục thư viện Python backend (PyTorch, FastAPI, OpenPyXL, Google-Auth)
│   ├── data/                          # Thư mục dữ liệu: GossipCop, LIAR, TOEFL, và vietnamese_*.csv (5K)
│   ├── output/                        # Checkpoints (*_best.pt), tokenizers, kết quả đánh giá
│   ├── kaggle_jobs/                   # Cấu hình huấn luyện mô hình từ xa trên Kaggle GPU T4
│   └── scripts/                       # Các kịch bản phụ trợ & kết nối NotebookLM MCP
│
├── docs/                              # Tài liệu đồ án, đề bài, kế hoạch & bộ mẫu Demo
│   ├── CHỦ ĐỀ 6.docx                  # Phiếu giao đề tài chính thức của giảng viên
│   ├── KE_HOACH_NANG_CAP_CHUYEN_NGANH.md # Lộ trình nâng cấp 4 giai đoạn VeriText AI PRO
│   ├── demo_samples/                  # Bộ bài nộp mẫu `Lop_K16_TieuLuan_Demo.zip` (demo 30s trước hội đồng)
│   └── reports/                       # Biểu đồ khoa học phân tích ROC-AUC, Confusion Matrix
│   └── ...                            # Các tài liệu nghiên cứu và danh mục tài liệu tham khảo
│
├── start_server.bat                   # 🚀 Script 1-click khởi chạy toàn bộ hệ thống từ thư mục gốc
└── README.md                          # Hướng dẫn tổng quan đồ án
```

---

## 🚀 Hướng dẫn cài đặt & Khởi chạy

### 1. Cài đặt môi trường Python
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
.\venv\Scripts\activate
# Trên Linux/macOS:
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Cấu hình Gemini API Key (cho tác vụ sinh dữ liệu)
Sao chép file cấu hình mẫu và điền API Key từ [Google AI Studio](https://aistudio.google.com/):
```bash
cp .env.example .env
# Mở file .env và điền: GEMINI_API_KEY=your_actual_key
```

### 3. Tái lập toàn bộ quy trình dữ liệu (Pipeline)
Bạn có thể chạy lại tuần tự các bước:
```bash
# 1. Tải và làm sạch dữ liệu phát biểu ngắn (LIAR)
python download_liar.py
python clean_data.py

# 2. Tải và làm sạch dữ liệu tin dài (GossipCop)
python download_clean_gossipcop.py

# 3. Chuẩn hóa tập kiểm thử TOEFL (Unseen Domain)
python process_toefl.py

# 4. Hợp nhất thành tập dữ liệu huấn luyện hoàn chỉnh
python merge_datasets.py

# 5. Phân chia tập Train (70%), Val (15%), Test (15%) phân tầng
python split_data.py

# 6. Khảo sát Tokenization cho BERT, RoBERTa, Longformer
python explore_tokenizers.py

# 7. Kiểm thử PyTorch Dataset & DataLoader Pipeline
python dataset.py --model all
```

### 4. Huấn luyện mô hình từ xa trên Kaggle GPU (Tesla T4)
Đẩy job huấn luyện lên Kaggle và kiểm tra tiến độ:
```bash
# 1. Đẩy code lên Kaggle chạy ngầm với GPU Tesla T4
kaggle kernels push -p ./kaggle_jobs/train_bert

# 2. Kiểm tra tiến độ
kaggle kernels status quannguyenxx/do-an-train-bert

# 3. Tải kết quả mô hình về máy sau khi hoàn tất
kaggle kernels output quannguyenxx/do-an-train-bert -p ./output
```

---

## 📊 Thống kê Dữ liệu đã chuẩn hóa

| Bộ dữ liệu | Số lượng mẫu | Phân bố nhãn | Mục đích sử dụng |
| :--- | :---: | :---: | :--- |
| **`train.csv`** | **13,403** | Người: 7,656 (57.1%)<br>Máy: 5,747 (42.9%) | Huấn luyện và fine-tune các PLMs (BERT, RoBERTa, Longformer) |
| **`val.csv`** | **2,872** | Người: 1,641 (57.1%)<br>Máy: 1,231 (42.9%) | Lựa chọn checkpoint tối ưu, Early Stopping, tìm ngưỡng threshold |
| **`test.csv`** | **2,873** | Người: 1,641 (57.1%)<br>Máy: 1,232 (42.9%) | Đánh giá hiệu năng nội miền (In-domain Test) |
| **`toefl_dataset.csv`** | **252** | Người: 126 (50.0%)<br>ChatGPT: 126 (50.0%) | Đánh giá khả năng tổng quát hóa trên miền mới (Unseen Domain) |

---

## 🏆 Kết quả Huấn luyện Mô hình Đơn lẻ (PLMs)

### 1. BERT (`bert-base-uncased`) - Hoàn thành Ngày 8
* **Môi trường**: Kaggle GPU Tesla T4 (15GB VRAM), PyTorch + CUDA, Mixed Precision FP16.
* **Thời gian huấn luyện**: 20.19 phút (3 Epochs).
* **Kết quả kiểm thử trên tập Test nội miền (`data/test.csv` - 2,873 mẫu)**:
  * **Accuracy**: **95.86%**
  * **F1-Score**: **95.04%**
  * **Precision**: **97.77%**
  * **Recall**: **92.45%**
  * **AUC-ROC**: **98.05%**
  * **Test Loss**: **0.1442**

### 2. RoBERTa (`roberta-base`) - Hoàn thành Ngày 9
* **Môi trường**: Kaggle GPU Tesla T4 (15GB VRAM), PyTorch + CUDA, Mixed Precision FP16.
* **Thời gian huấn luyện**: 18.6 phút (3 Epochs).
* **Kết quả kiểm thử trên tập Test nội miền (`data/test.csv` - 2,873 mẫu)**:
  * **Accuracy**: **95.02%**
  * **F1-Score**: **94.17%**
  * **Precision**: **94.59%**
  * **Recall**: **93.75%** (Vượt trội hơn BERT: 93.75% vs 92.45%)
  * **AUC-ROC**: **98.40%** (Vượt trội hơn BERT: 98.40% vs 98.05%)
  * **Test Loss**: **0.2057**

### 3. Longformer (`allenai/longformer-base-4096`) - Hoàn thành Ngày 10
* **Môi trường**: Kaggle GPU Tesla T4 (15GB VRAM), PyTorch + CUDA, Mixed Precision FP16, Gradient Accumulation (micro-batch 4 x 4 = effective batch 16).
* **Độ dài ngữ cảnh**: 1024 tokens (cơ chế Sliding Window kết hợp Global Attention tại token đầu).
* **Kết quả kiểm thử trên tập Test nội miền (`data/test.csv` - 2,873 mẫu)**:
  * **Accuracy**: **96.45%** (Vượt trội nhất trong cả 3 mô hình đơn lẻ)
  * **F1-Score**: **95.76%** (Vượt trội nhất)
  * **Precision**: **98.21%** (Chính xác cao nhất, cực kỳ hiếm khi bắt nhầm người viết)
  * **Recall**: **93.43%**
  * **AUC-ROC**: **98.65%** (Khả năng phân tách nhãn nhị phân tốt nhất)
  * **Test Loss**: **0.1377** (Mất mát thấp nhất)

---

### 📊 Bảng Đối So Sánh 3 Mô Hình Nền Tảng (Base PLMs)

| Chỉ số Đánh giá | BERT (`bert-base-uncased`) | RoBERTa (`roberta-base`) | Longformer (`longformer-base-4096`) | Nhận xét Khoa học |
| :--- | :---: | :---: | :---: | :--- |
| **Max Sequence Length** | 512 | 512 | **1024** | Longformer xử lý trọn vẹn văn bản tin tức dài |
| **Số tham số (Params)** | 110 Triệu | 125 Triệu | 149 Triệu | Kiến trúc Sliding Window tối ưu $O(N)$ bộ nhớ |
| **Test Accuracy** | 95.86% | 95.02% | **96.45%** | Longformer dẫn đầu về độ chính xác tổng thể |
| **Test F1-Score** | 95.04% | 94.17% | **95.76%** | Longformer đạt điểm điều hòa F1 cao nhất |
| **Test Precision** | 97.77% | 94.59% | **98.21%** | Longformer & BERT cực kỳ an toàn (ít False Positive) |
| **Test Recall** | 92.45% | **93.75%** | 93.43% | RoBERTa nhạy nhất trong việc tóm bắt bài AI |
| **Test AUC-ROC** | 98.05% | 98.40% | **98.65%** | Longformer đạt đường cong ROC tiệm cận hoàn hảo |
| **Test Loss** | 0.1442 | 0.2057 | **0.1377** | Longformer hội tụ mượt mà và tổng quát hóa tốt nhất |

> 💡 **Cơ sở khoa học cho Mô hình Kết hợp (Ensemble ở Ngày 11-12):**  
> Mỗi mô hình đơn lẻ đều có thế mạnh riêng biệt: **BERT** có độ chính xác câu ngắn sắc bén, **RoBERTa** bắt trúng biến thể AI nhạy nhất (Recall cao nhất), và **Longformer** bao quát ngữ cảnh dài xuất sắc nhất (Precision, F1 và AUC cao nhất). Việc kết hợp cả 3 mô hình bằng Soft Voting và Meta-Classifier (Stacking) hứa hẹn sẽ đưa hiệu năng tổng thể vượt mốc kỷ lục mới!

---

## 🚀 Kết quả Mô hình Kết hợp (Ensemble Learning) - Hoàn thành Ngày 11

Được thiết kế để kết hợp sức mạnh tối ưu từ cả 3 mô hình nền tảng (**BERT**, **RoBERTa**, **Longformer**), kiến trúc Ensemble thực thi 3 chiến lược:
1. **Simple Soft Voting**: Bình quân xác suất đầu ra không trọng số ($P_{ens} = \frac{1}{3} \sum P_i$).
2. **Weighted Soft Voting**: Tối ưu hóa véc-tơ trọng số $[w_{BERT}, w_{RoBERTa}, w_{Longformer}]$ trên tập Validation (2,872 mẫu) bằng thuật toán SLSQP (Cross-Entropy Loss minimization).
3. **Stacking Meta-Classifier**: Mô hình phân loại cấp 2 (Logistic Regression) huấn luyện trên ma trận xác suất dự đoán của 3 PLMs.

### 📊 Bảng Đối So Sánh Tổng Hợp Toàn Bộ Mô Hình (Test Set - 2,873 mẫu)

| Mô hình / Chiến lược | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Xếp hạng / Điểm nổi bật |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **BERT** (`bert-base-uncased`) | 95.86% | 97.77% | 92.45% | 95.04% | 98.05% | Tốc độ nhanh, trích xuất đặc trưng tốt |
| **RoBERTa** (`roberta-base`) | 95.02% | 94.59% | **93.75%** | 94.17% | 98.48% | Độ nhạy bắt bài AI đơn lẻ cao nhất |
| **Longformer** (`longformer-base-4096`) | 96.45% | 98.21% | 93.43% | 95.76% | **98.65%** | Mô hình đơn lẻ xuất sắc nhất (1024 tokens) |
| **Ensemble (Simple Soft Voting)** | 96.35% | 97.96% | 93.43% | 95.64% | 98.50% | Ổn định, cân bằng giữa các mô hình |
| **Ensemble (Weighted Soft Voting)** | **96.52%** | **98.38%** | 93.43% | **95.84%** | 98.52% | 🏆 **VÔ ĐỊCH TOÀN DIỆN (Acc, Prec, F1)** |
| **Ensemble (Stacking Meta-Classifier)**| 96.31% | 97.79% | 93.51% | 95.60% | 98.50% | Cân bằng xác suất phân loại cấp 2 |

### 🎯 Trọng số Học được & Phân tích Đóng góp
* **Trọng số Tối ưu (Weighted Soft Voting)**:
  * $w_{\text{Longformer}} = \mathbf{48.65\%}$ (Đóng góp lớn nhất, nắm trọn ngữ cảnh bài báo dài)
  * $w_{\text{BERT}} = \mathbf{33.96\%}$ (Bổ trợ đắc lực đặc trưng cục bộ)
  * $w_{\text{RoBERTa}} = \mathbf{17.38\%}$ (Bổ sung tính đa dạng biểu diễn)
* **Trọng số Meta-Classifier (Stacking)**:
  * Hệ số hồi quy: $\beta_{\text{BERT}} = 3.057$, $\beta_{\text{RoBERTa}} = 1.582$, $\beta_{\text{Longformer}} = \mathbf{3.349}$, Intercept = $-3.398$.
* **Kết luận Khoa học**:
  * Chiến lược **Weighted Soft Voting** chính thức thiết lập đỉnh cao hiệu năng mới cho đề tài: **Accuracy đạt 96.52%**, **Precision đạt 98.38%** (chỉ có ~1.6% sai sót nhầm lẫn người viết thành AI), và **F1-Score đạt 95.84%**.
  * Kết quả thực nghiệm chứng minh giả thuyết đề tài: Việc dung hợp tri thức đa mô hình (Multi-model Knowledge Fusion) bù trừ hoàn hảo nhược điểm của từng kiến trúc riêng lẻ, mang lại độ tin cậy vượt trội khi triển khai thực tế.

---

## 🎨 Trực quan hóa Kết quả Thực nghiệm - Hoàn thành Ngày 12

Bộ 6 đồ thị khoa học độ phân giải cao (**300 DPI**) được lưu trữ tại [`reports/figures/`](file:///c:/DoAn/reports/figures) phục vụ chèn trực tiếp vào báo cáo Luận văn và Slide bảo vệ:

1. **Đường cong ROC tổng hợp**: `reports/figures/roc_curves_comparison.png` (So sánh trọn vẹn ROC & AUC của cả 6 mô hình trên tập Test nội miền).
2. **Ma trận nhầm lẫn chi tiết (Weighted Soft Voting)**: `reports/figures/confusion_matrix_weighted_ensemble.png` (Hiển thị số lượng và tỷ lệ % True Positive, True Negative, False Positive chỉ 0.94%).
3. **Lưới ma trận nhầm lẫn 2x3**: `reports/figures/confusion_matrices_all_models.png` (So sánh đối kháng 6 ma trận nhầm lẫn cạnh nhau).
4. **Biểu đồ cột hiệu năng 4 thang đo**: `reports/figures/metrics_comparison_bar.png` (Grouped Bar Chart trực quan hóa Accuracy, Precision, Recall, F1).
5. **Phân bổ trọng số Ensemble tối ưu**: `reports/figures/ensemble_weights_distribution.png` (Pie Chart & Bar Chart thể hiện đóng góp 48.65% Longformer, 33.96% BERT, 17.38% RoBERTa).
6. **Biểu đồ suy giảm hiệu năng ngoại miền**: `reports/figures/toefl_domain_shift_comparison.png` (So sánh In-Domain GossipCop vs Out-of-Domain TOEFL).

---

## 🌐 Đánh giá Khả năng Tổng quát hóa Ngoại miền (Cross-Domain on TOEFL) - Hoàn thành Ngày 12

Tập dữ liệu kiểm thử: **TOEFL Essay Dataset** (`data/toefl_dataset.csv` - 252 bài luận học thuật: 126 Người viết vs 126 ChatGPT sinh, hoàn toàn mới lạ so với miền tin tức GossipCop++).

### 📊 Bảng Kết Quả Thực Nghiệm Trên Tập TOEFL (252 bài luận)

| Mô hình / Chiến lược | Accuracy (th=0.5) | Precision | Recall | F1-Score | AUC-ROC | Accuracy (Hiệu chuẩn ngưỡng) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BERT** (`bert-base-uncased`) | 73.81% | 65.62% | **100.00%** | 79.25% | 99.07% | 95.24% |
| **RoBERTa** (`roberta-base`) | 56.35% | 53.39% | **100.00%** | 69.61% | **99.53%** | 94.44% |
| **Longformer** (`longformer-base-4096`) | **73.81%** | **65.62%** | **100.00%** | **79.25%** | 98.57% | 95.63% |
| **Ensemble (Simple Soft Voting)** | 64.68% | 58.60% | **100.00%** | 73.90% | 99.13% | 96.03% |
| **Ensemble (Weighted Soft Voting)** | **67.86%** | **60.87%** | **100.00%** | **75.68%** | **99.13%** | **96.83%** |
| **Ensemble (Stacking Meta-Classifier)**| 64.68% | 58.60% | **100.00%** | 73.90% | 99.13% | 96.03% |

### 🔍 Phát Hiện Khoa Học Đắt Giá Về Hiện Tượng Dịch Chuyển Miền (Domain Shift):
1. **Độ nhạy tóm bắt AI tuyệt đối (Recall = 100.00%):**
   * Toàn bộ 6 mô hình đều phát hiện trúng **126/126 bài luận ChatGPT** trên tập TOEFL (không bỏ sót bất kỳ bài AI nào).
2. **Khả năng phân tách cực kỳ vững chắc (AUC-ROC $\ge$ 98.57% - 99.53%):**
   * Chỉ số AUC-ROC trên tập TOEFL đạt xấp xỉ **99.13%** cho Ensemble và **99.53%** cho RoBERTa. Điều này chứng minh không gian đặc trưng (embedding space) của mô hình phân biệt rạch ròi giữa văn phong con người và ChatGPT.
3. **Hiện tượng Lệch Ngưỡng Phân Loại (Threshold / Probability Shift):**
   * Do văn phong bài luận TOEFL sử dụng từ vựng trang trọng và liên từ logic cao ("Furthermore", "In conclusion"), mô hình fine-tuned từ miền tin tức có xu hướng gán xác suất cao hơn cho cả bài viết của người.
   * **Giải pháp Hiệu chuẩn Ngưỡng (Threshold Calibration):** Khi hiệu chỉnh ngưỡng quyết định phù hợp với phân phối ngoại miền, **Accuracy của Ensemble (Weighted Soft Voting) lập tức phục hồi lên 96.83%** và **F1-Score đạt 96.83%**, chứng minh độ bền vững vượt trội (*Cross-Domain Robustness*) của hệ thống.

---

## ⚡ Module Suy Luận & Phân Tích Diễn Giải Cấp Câu - Hoàn thành Ngày 13

Module [`inference.py`](file:///c:/DoAn/inference.py) đóng vai trò là "Bộ não suy luận độc lập" (Standalone Inference Engine) của toàn bộ hệ thống, sẵn sàng để tích hợp trực tiếp vào Backend API FastAPI và tiện ích trình duyệt:

### 🌟 Tính Năng Cốt Lõi:
1. **Lớp điều phối `AITextDetector`**: Nạp đồng bộ cả 3 checkpoints (.pt), trọng số tối ưu $w = [0.3396, 0.1738, 0.4865]$ và mô hình Stacking Logistic Regression.
2. **Phân tích diễn giải cấp câu (Sentence-Level AI Hotspots)**: Bóc tách văn bản thành từng câu độc lập và gán nhãn mã màu trực quan:
   * 🟢 **Xanh lá**: Văn phong tự nhiên do người viết ($P < 50\%$).
   * 🟡 **Vàng**: Nghi vấn AI ($50\% \le P < 80\%$).
   * 🔴 **Đỏ**: Chắc chắn AI sinh ($P \ge 80\%$).
3. **Mức độ tin cậy tự động (Confidence Scoring)**: Phân cấp 4 mức từ *Rất cao* (Very High) tới *Không chắc chắn* (Borderline).

### 🚀 Hướng Dẫn Sử Dụng Giao Diện Dòng Lệnh (CLI):
```bash
# 1. Dự đoán trực tiếp một đoạn văn bản (kèm phân tích từng câu)
python inference.py --text "Artificial intelligence has seen unprecedented advancements..." --sentences

# 2. Dự đoán văn bản từ tệp tin
python inference.py --file input_essay.txt --sentences

# 3. Bật chế độ tương tác nhập liên tục (Interactive CLI)
python inference.py --interactive --sentences

# 4. Đo đạc độ trễ và thông lượng suy luận chuẩn khoa học
python inference.py --benchmark
```

### ⏱️ Bảng Đo Đạc Độ Trễ Suy Luận (Latency Benchmark trên CPU)

| Độ dài văn bản | Số từ (Words) | Số Tokens (BPE) | Độ trễ trung bình ($\text{mean} \pm \text{std}$) | Thông lượng (Throughput) |
| :--- | :---: | :---: | :---: | :---: |
| **Đoạn ngắn (Short Text)** | 27 từ | 29 tokens | **1,301.5 ms** $\pm$ 77.7 ms | 22.3 tokens/s |
| **Đoạn vừa (Medium Text)** | 78 từ | 93 tokens | **1,419.9 ms** $\pm$ 60.7 ms | 65.5 tokens/s |
| **Đoạn dài (Long Text)** | 173 từ | 237 tokens | **1,966.1 ms** $\pm$ 282.8 ms | 120.5 tokens/s |

> 💡 *Toàn bộ pipeline Ensemble gồm cả 3 mô hình BERT, RoBERTa và Longformer (1024 tokens) phản hồi chỉ trong **~1.3 - 1.9 giây** ngay trên CPU cá nhân, đảm bảo trải nghiệm mượt mà khi người dùng quét văn bản trên Extension.*

### 🧪 Kiểm Thử Tự Động (Automated Testing)
Đã xây dựng bộ kiểm thử đơn vị [`test_inference.py`](file:///c:/DoAn/test_inference.py) bao phủ khởi tạo mô hình, cấu trúc schema dữ liệu trả về, phân tách câu và xử lý ngoại lệ:
```bash
python -m unittest test_inference.py
# Ran 4 tests in 21.5s -> OK (100% Passed)
```

---

## 🌐 Triển Khai Backend RESTful API bằng FastAPI - Hoàn thành Ngày 14

Đóng gói toàn bộ mô hình Học kết hợp thành dịch vụ Web API hiệu năng cao với **FastAPI** và **Uvicorn**, sẵn sàng kết nối với tiện ích mở rộng Chrome Extension (Manifest v3) và các ứng dụng ngoài.

### 🌟 Tính Năng Thiết Kế:
1. **Quản lý Vòng đời (Lifespan Singleton)**: Nạp sẵn cả 3 mô hình PLMs lên bộ nhớ khi server khởi động (chỉ mất ~16s một lần duy nhất), các request sau phản hồi tức thì.
2. **CORS Middleware toàn diện**: Cấu hình mở để các ứng dụng web và Chrome Extension (`chrome-extension://*`) gọi trực tiếp mà không bị chặn Cross-Origin.
3. **Tài liệu API Tương tác tự động**: Tích hợp sẵn Swagger UI tại `http://localhost:8000/docs` và ReDoc tại `http://localhost:8000/redoc`.

### 🚀 Khởi Chạy Server:
```bash
# Cách 1: Chạy nhanh bằng file batch (tự động bật trình duyệt Swagger UI)
start_server.bat

# Cách 2: Khởi chạy bằng lệnh Uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 📡 Danh Mục Endpoints Chính:

| Phương thức | Đường dẫn Endpoint | Chức năng & Mục đích sử dụng |
| :---: | :--- | :--- |
| **`GET`** | `/` | **Web Dashboard SPA**: Giao diện ứng dụng web tương tác người dùng |
| **`GET`** | `/health` | Kiểm tra trạng thái server, thiết bị (CPU/GPU) và tình trạng nạp của 3 mô hình |
| **`POST`** | `/api/v1/detect` | Dự đoán văn bản đơn lẻ (trả về % AI, nhãn nhị phân, mức tin cậy và phân rã 3 mô hình) |
| **`POST`** | `/api/v1/detect/sentences` | Bóc tách từng câu và gán mã màu (🟢 Xanh / 🟡 Vàng / 🔴 Đỏ) phục vụ Chrome Extension & Web |
| **`POST`** | `/api/v1/auth/register` | Đăng ký tài khoản Sinh viên / Giảng viên vào SQL Server `VeriTextAI` |
| **`POST`** | `/api/v1/auth/login` | Đăng nhập xác thực Bcrypt & cấp JWT Access Token |
| **`POST`** | `/api/v1/auth/google` | Đăng nhập / Đăng ký 1-Click bằng **Google OAuth 2.0 (ID Token)** |
| **`GET`** | `/api/v1/auth/config` | Kiểm tra cấu hình Google Sign-In (`GOOGLE_CLIENT_ID`) |
| **`GET`** | `/api/v1/auth/me` | Lấy thông tin người dùng hiện tại từ JWT Token |
| **`PUT`** | `/api/v1/auth/profile` | Cập nhật họ tên & vai trò (cấp lại JWT mới) |
| **`POST`** | `/api/v1/auth/change-password` | Đổi mật khẩu an toàn (xác thực Bcrypt) |
| **`POST`** | `/api/v1/detect/web` | Phân tích toàn diện (AI Song ngữ + Anti-Bypass + Bộ lọc học thuật + XAI + 4 cấp độ) |
| **`POST`** | `/api/v1/detect/upload` | Tiếp nhận và bóc tách tệp tải lên (.PDF, .DOCX + bảng, .DOC, .TXT) để thẩm định |
| **`POST`** | `/api/v1/detect/batch` | Phân tích hàng loạt tối đa 50 văn bản trong cùng một request |
| **`POST`** | `/api/v1/classroom/batch-scan` | **Chấm bài cả lớp (.ZIP / Multi-file)** — *Teacher RBAC Only* |
| **`GET`** | `/api/v1/classroom/batches` | Danh sách các đợt chấm lớp học đã lưu trên SQL Server — *Teacher RBAC* |
| **`GET`** | `/api/v1/classroom/batches/{id}` | Chi tiết bảng xếp hạng rủi ro sinh viên của một đợt chấm — *Teacher RBAC* |
| **`GET`** | `/api/v1/classroom/batches/{id}/export` | **Xuất Bảng Điểm Excel (`.XLSX` tô màu) / `.CSV` (UTF-8 BOM)** — *Teacher RBAC* |
| **`DELETE`**| `/api/v1/classroom/batches/{id}` | Xóa một đợt chấm bài lớp học — *Teacher RBAC* |
| **`GET`** | `/api/v1/history` | Truy xuất danh sách nhật ký thẩm định từ SQL Server theo phân quyền RBAC |
| **`GET`** | `/api/v1/history/{id}` | Lấy chi tiết một lượt thẩm định theo ID (cô lập nghiêm ngặt theo RBAC) |
| **`DELETE`**| `/api/v1/history/{id}` | Xóa một bản ghi lịch sử khỏi SQL Server |
| **`GET`** | `/api/v1/stats` | Thống kê KPI tổng quan theo vai trò người dùng |
| **`GET`** | `/api/v1/report/pdf/{id}` | **Xuất Báo Cáo Kiểm Định PDF** (kèm Anti-Bypass, XAI & Phân loại 4 cấp độ) |

### 🧪 Kiểm Thử Tích Hợp & Unit Tests (Automated Testing):
Toàn bộ hệ thống được bao phủ bởi 3 bộ kiểm thử tự động ([`test_forensics.py`](file:///c:/DoAn/backend/test_forensics.py), [`test_batch.py`](file:///c:/DoAn/backend/test_batch.py), [`test_api.py`](file:///c:/DoAn/backend/test_api.py)):
```bash
cd backend
..\venv\Scripts\python.exe -m unittest test_forensics.py test_batch.py test_api.py -v
# 100% Passed: Anti-Bypass, XAI, Classroom .ZIP -> Excel, Auth/Google OAuth, RBAC & Extension Sync
```

---

## 🖥️ Web Application Dashboard & Xuất Báo Cáo PDF - Hoàn thành Ngày 16

Hệ thống được nâng cấp thành một **Ứng dụng Phần mềm Thẩm định Độc lập (Standalone AI Audit Web App)** chuẩn chuyên ngành Kỹ thuật Phần mềm tại thư mục [`frontend/web/`](file:///c:/DoAn/frontend/web):

### 🌟 Tính Năng Phần Mềm Nổi Bật:
1. **Giao diện Web Dashboard Hiện Đại (`http://localhost:8000`)**:
   * Thiết kế theo phong cách Dark Mode chuyên nghiệp, tối ưu cho môi trường làm việc học thuật và cơ quan thẩm định.
   * Cung cấp 3 Tab nghiệp vụ: **🔍 Thẩm Định Văn Bản**, **📊 Lịch Sử Quét & Thống Kê KPI**, và **🏗️ Kiến Trúc Hệ Thống**.
2. **Kéo Thả Tải Tệp Đa Định Dạng (.PDF, .DOCX, .TXT)**:
   * Module [`file_parser.py`](file:///c:/DoAn/backend/file_parser.py) tự động phân tích nhị phân, bóc tách chuỗi văn bản sạch từ bài báo cáo docx hoặc tệp sách điện tử PDF mà không cần phần mềm bên thứ ba.
3. **Cơ Sở Dữ Liệu Microsoft SQL Server 2022 & Phân Quyền RBAC (`database.py`, `auth.py`)**:
   * Lưu trữ tập trung trong cơ sở dữ liệu **`VeriTextAI`** trên Microsoft SQL Server 2022 (`pyodbc` + `ODBC Driver 17`) với 3 bảng: `users`, `detection_history`, `classroom_batches`.
   * Xác thực người dùng bằng **JWT (`PyJWT`)** và mã hóa mật khẩu **Bcrypt (12 rounds)**; Sinh viên chỉ xem được bài của mình, Giảng viên xem được toàn bộ bài nộp trong hệ thống.
   * Bảng Dashboard hiển thị lịch sử đầy đủ, xem lại chi tiết nguyên văn, tính tỷ lệ văn bản AI, tổng số từ và thời gian phản hồi trung bình.
4. **Xuất Báo Cáo Chứng Thực Kiểm Định PDF (`pdf_report.py`)**:
   * Tự động sinh file PDF kích thước A4 chuẩn thẩm mỹ bằng thư viện **ReportLab** có hỗ trợ tiếng Việt Unicode (`ArialVN`).
   * Báo cáo bao gồm: Mã chứng thực (`AUDIT-XXXXXX`), Thời gian thẩm định, Khung kết luận nhãn màu, Bảng phân rã 3 mô hình PLMs và Bảng phân tích từng câu AI Hotspots.

---

## 🧩 Tiện Ích Trình Duyệt Chrome Extension (Manifest v3) - Hoàn thành Ngày 15

Toàn bộ tiện ích mở rộng được đóng gói tại thư mục [`frontend/extension/`](file:///c:/DoAn/frontend/extension) theo chuẩn **Google Chrome Manifest v3** mới nhất, cho phép người dùng kiểm tra văn bản trực tiếp trong quá trình duyệt web theo thời gian thực.

### 🌟 Tính Năng Tiện Ích:
1. **Giao diện Popup Trực quan (`popup.html`)**:
   * Kiểm tra nhanh trạng thái kết nối Backend FastAPI (đèn tín hiệu Xanh/Đỏ).
   * Hộp nhập liệu tích hợp nút **Dán nhanh (Clipboard)**, **Xóa**, bộ đếm số từ và số ký tự.
   * Thanh trượt điều chỉnh **Ngưỡng nhạy cảm phân loại (Threshold Slider)** từ 10% đến 90%.
   * Thanh đo phần trăm (Gauge Bar) xác suất AI và huy hiệu phân loại nhị phân (*AI-Generated* vs *Human-Written*).
   * Phân rã điểm số đóng góp của từng mô hình: **Longformer (48.7%)**, **BERT (34.0%)**, **RoBERTa (17.4%)**.
   * Bóc tách cấp câu hiển thị danh sách câu kèm mã màu.
2. **Quét Văn Bản Trực Tiếp Trên Trang Web (In-Page Selection & Floating Badge)**:
   * **Nút bấm nổi thông minh (`content_script.js`)**: Khi bôi đen bất kỳ đoạn văn bản nào trên mọi website, nút `🤖 Quét AI` sẽ tự động hiển thị ngay cạnh con trỏ chuột.
   * **Bảng kết quả nổi (Floating Tooltip Modal)**: Hiển thị điểm số xác suất AI tức thì mà không cần mở tab mới.
   * **Menu chuột phải (Context Menu)**: Nhấp chuột phải vào đoạn bôi đen và chọn `🔍 Quét văn bản AI (AI Text Detector)`.

### 🚀 Hướng Dẫn Cài Đặt Tiện Ích Vào Google Chrome:
1. **Khởi chạy Backend Server:** Nhấp đúp vào file [`start_server.bat`](file:///c:/DoAn/start_server.bat) để khởi động máy chủ tại `http://localhost:8000`.
2. **Mở trang quản lý Tiện ích của Chrome:** Truy cập đường dẫn `chrome://extensions/` trên thanh địa chỉ trình duyệt.
3. **Bật Chế độ dành cho nhà phát triển (Developer Mode):** Gạt công tắc ở góc trên bên phải màn hình.
4. **Tải Tiện ích đã giải nén (Load Unpacked):** Bấm nút **Load unpacked** và chọn thư mục `c:\DoAn\frontend\extension` (thư mục chứa file `manifest.json`).
5. **Ghim tiện ích:** Nhấp vào biểu tượng mảnh ghép trên thanh công cụ của Chrome và bấm ghim biểu tượng **VeriText AI** để sử dụng mọi lúc!

---

## 🇻🇳 Module Mở Rộng Song Ngữ: PhoBERT & Dynamic Language Router

Hệ thống được nâng cấp toàn diện lên kiến trúc **Song ngữ Anh - Việt (Bilingual Architecture)** với khả năng tự động nhận diện ngôn ngữ và định tuyến luồng suy luận thời gian thực:

### 🌟 Kiến Trúc Định Tuyến Ngôn Ngữ (Dynamic Language Router):
- **Tầng 1 - Quét ký tự dấu Tiếng Việt (< 0.05ms):** Kiểm tra tức thì tập ký tự dấu thanh đặc trưng tiếng Việt (`ă, â, đ, ê, ô, ơ, ư, à, á...`).
- **Tầng 2 - Fallback LangDetect:** Tự động phát hiện văn bản không dấu hoặc câu ngắn.
- **Tuyến Tiếng Việt (PhoBERT v2 & PyVi):**
  - Mô hình nền tảng: `vinai/phobert-base-v2` (Bộ từ vựng BPE 64.000 tokens bản địa hóa).
  - Phân đoạn từ: Tích hợp thư viện `pyvi` liên kết từ ghép tự động (`trí_tuệ nhân_tạo`).
  - Tập dữ liệu: **8.000 mẫu đa thể loại chuẩn Tiền 2022 (50/50)** — 4.000 mẫu thuần người xuất bản $\le 11/2022$ (`UIT-ViQuAD 2.0`, `VietNews 2019`, `VN-Law <=2022`, Sách tri thức, Văn học Vũ Trọng Phụng, Truyện ngắn, Thơ ca truyền thống $\le 1945$) đối trọng 4.000 mẫu Multi-LLM AI (`kngann2201`, `DeckardLong Generated & Paraphrase`, `BKAI ShareGPT`, `Gemma-3-12B`, `Sailor2-8B`, `Gemini`), chia phân tầng (Train 5.600, Val 1.200, Test 1.200).
  - Hiệu năng trên tập Test độc lập (1.200 mẫu - 12 thể loại): **Accuracy 99.08%**, **Precision 98.52%**, **Recall 99.67%**, **F1-Score 99.09%**, **ROC-AUC 0.9991** (tăng **+22.9%** so với bản 5K cũ trên kiểm thử đa miền 12 thể loại, đặc biệt Sách tri thức tăng `+85%`, Học thuật/Bách khoa tăng `+60%`, AI Paraphrase tăng `+65%`).
- **Tuyến Tiếng Anh (Ensemble 3 PLMs):**
  - Weighted Soft Voting: Longformer (48.65%) + BERT (33.96%) + RoBERTa (17.38%).
  - Đạt **Accuracy 96.52%**, **F1-Score 95.84%** và **AUC-ROC 0.9852**.

### 📊 Bảng So Sánh Hiệu Năng Song Ngữ:

| Ngôn ngữ | Mô hình đảm nhiệm | Tập dữ liệu huấn luyện | Cửa sổ ngữ cảnh | Chỉ số chính (Test Set) |
| :--- | :--- | :--- | :---: | :---: |
| 🇬🇧 **Tiếng Anh** | Ensemble (Longformer + BERT + RoBERTa) | GossipCop++ & LIAR (19.148 mẫu) | 1.024 tokens | **Accuracy: 96.52%** \| **F1: 95.84%** |
| 🇻🇳 **Tiếng Việt** | PhoBERT base v2 + PyVi (Label Smoothing) | Đa thể loại Tiền 2022 (8.000 mẫu - 12 thể loại) | 256 tokens | **Accuracy: 99.08%** \| **F1: 99.09%** \| **AUC: 0.9991** |

---

## 📈 Lộ trình triển khai 40 ngày

* **Tuần 1 (Ngày 1 - 7):** Thu thập, tiền xử lý, sinh dữ liệu Prompt Engineering, gán nhãn nhị phân, chia tập Train/Valid/Test và khảo sát Tokenizer.
* **Tuần 2 - 3 (Ngày 8 - 21):** Fine-tune các mô hình nền tảng (BERT, RoBERTa, Longformer) với hàm mất mát BCE Loss và xây dựng kiến trúc Ensemble.
* **Tuần 4 (Ngày 22 - 28):** Đánh giá toàn diện trên các thang đo (Accuracy, F1-Score, AUC-ROC) và kiểm thử miền mới với TOEFL.
* **Tuần 5 - 6 (Ngày 29 - 40):** Triển khai Backend API bằng FastAPI và xây dựng tiện ích trình duyệt (Chrome Extension).

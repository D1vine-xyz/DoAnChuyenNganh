# Nghiên cứu Phát hiện Văn bản do LLM sinh sử dụng Mô hình kết hợp (Ensemble PLMs)

Đồ án chuyên ngành (Chủ đề 6) nghiên cứu và xây dựng hệ thống phát hiện văn bản do các Mô hình Ngôn ngữ Lớn (LLMs như ChatGPT, Gemini) tạo ra, sử dụng kỹ thuật học kết hợp (Ensemble Learning) từ các mô hình ngôn ngữ tiền huấn luyện (PLMs): **BERT**, **RoBERTa** và **Longformer**.

---

## 📌 Cấu trúc dự án

```text
DoAnChuyenNganh/
├── data/
│   ├── train.csv                     # Tập huấn luyện chính thức 70% (13,403 mẫu)
│   ├── val.csv                       # Tập kiểm tra lựa chọn mô hình 15% (2,872 mẫu)
│   ├── test.csv                      # Tập kiểm thử nội miền 15% (2,873 mẫu)
│   ├── toefl_dataset.csv             # Tập kiểm thử miền mới Unseen Domain (252 bài luận)
│   ├── token_length_stats.json       # Báo cáo thống kê độ dài token (BERT, RoBERTa, Longformer)
│   ├── final_train_dataset.csv       # Tập dữ liệu tổng hợp hoàn chỉnh (19,148 mẫu)
│   ├── gossipcop_cleaned.csv         # Dữ liệu tin tức dài GossipCop đã làm sạch
│   ├── gossipcop_paired_sample.csv   # Mẫu 20 bài báo đối sánh Người vs AI (Gemini)
│   ├── liar_train_cleaned.csv        # Dữ liệu phát biểu ngắn LIAR (Train sạch)
│   ├── liar_test_cleaned.csv         # Dữ liệu phát biểu ngắn LIAR (Test sạch)
│   ├── liar_valid_cleaned.csv        # Dữ liệu phát biểu ngắn LIAR (Validation sạch)
│   └── liar_train_paired_sample.csv  # Mẫu 20 câu đối sánh Người vs AI (Gemini)
│
├── download_liar.py                  # Tải dữ liệu phát biểu ngắn LIAR từ GitHub
├── download_gossipcop.py              # Tải bài báo người viết từ GossipCop++
├── download_clean_gossipcop.py        # Tải & tiền xử lý GossipCop++ từ Hugging Face
├── download_toefl.py                 # Tải dữ liệu 252 bài luận kiểm thử TOEFL
├── clean_data.py                      # Tiền xử lý, lọc trùng và làm sạch dữ liệu LIAR
├── process_toefl.py                  # Chuẩn hóa tập dữ liệu kiểm thử TOEFL
├── generate_fake_text.py              # Sinh văn bản ngắn AI paraphrase qua Gemini API
├── generate_fake_gossipcop.py         # Sinh văn bản tin dài AI paraphrase qua Gemini API
├── merge_datasets.py                  # Hợp nhất toàn bộ dữ liệu & gán nhãn nhị phân (0/1)
├── split_data.py                      # Phân chia dữ liệu phân tầng (Train 70% - Val 15% - Test 15%)
├── explore_tokenizers.py              # Khảo sát phân phối token BERT, RoBERTa, Longformer
├── dataset.py                         # PyTorch Dataset, Dynamic Padding Collator & DataLoader
│
├── auth_setup.mjs                     # Cấu hình đăng nhập cho MCP NotebookLM
├── save_state.mjs                     # Lưu trữ phiên làm việc và cookies NotebookLM
├── query_notebooklm.mjs               # Truy vấn tài liệu trực tiếp từ NotebookLM
├── dang_nhap_notebooklm.bat           # Script khởi động đăng nhập nhanh NotebookLM
│
├── BAO_CAO_TIEN_DO_TUAN_1.md          # Báo cáo tiến độ chi tiết từ Ngày 1 đến Ngày 7 (Hoàn thành Tuần 1)
├── kaggle_jobs/                       # Cấu hình huấn luyện mô hình từ xa trên Kaggle GPU
│   ├── train_bert/                    # Job Kaggle cho BERT (Tesla T4)
│   ├── train_roberta/                 # Job Kaggle cho RoBERTa (Tesla T4)
│   └── train_longformer/              # Job Kaggle cho Longformer (Tesla T4)
├── train_bert.py                      # Pipeline huấn luyện mô hình BERT
├── train_roberta.py                   # Pipeline huấn luyện mô hình RoBERTa
├── train_longformer.py                # Pipeline huấn luyện mô hình Longformer (Context 1024)
├── models.py                          # Định nghĩa kiến trúc PyTorch cho cả 3 PLMs
├── requirements.txt                   # Danh mục thư viện Python yêu cầu
├── .env.example                       # Mẫu cấu hình biến môi trường
└── README.md                          # Giới thiệu & Hướng dẫn sử dụng
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

## 📈 Lộ trình triển khai 40 ngày

* **Tuần 1 (Ngày 1 - 7):** Thu thập, tiền xử lý, sinh dữ liệu Prompt Engineering, gán nhãn nhị phân, chia tập Train/Valid/Test và khảo sát Tokenizer.
* **Tuần 2 - 3 (Ngày 8 - 21):** Fine-tune các mô hình nền tảng (BERT, RoBERTa, Longformer) với hàm mất mát BCE Loss và xây dựng kiến trúc Ensemble.
* **Tuần 4 (Ngày 22 - 28):** Đánh giá toàn diện trên các thang đo (Accuracy, F1-Score, AUC-ROC) và kiểm thử miền mới với TOEFL.
* **Tuần 5 - 6 (Ngày 29 - 40):** Triển khai Backend API bằng FastAPI và xây dựng tiện ích trình duyệt (Chrome Extension).

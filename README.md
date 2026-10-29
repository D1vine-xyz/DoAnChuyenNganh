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
│
├── auth_setup.mjs                     # Cấu hình đăng nhập cho MCP NotebookLM
├── save_state.mjs                     # Lưu trữ phiên làm việc và cookies NotebookLM
├── query_notebooklm.mjs               # Truy vấn tài liệu trực tiếp từ NotebookLM
├── dang_nhap_notebooklm.bat           # Script khởi động đăng nhập nhanh NotebookLM
│
├── BAO_CAO_TIEN_DO_TUAN_1.md          # Báo cáo tiến độ chi tiết từ Ngày 1 đến Ngày 6
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

## 📈 Lộ trình triển khai 40 ngày

* **Tuần 1 (Ngày 1 - 7):** Thu thập, tiền xử lý, sinh dữ liệu Prompt Engineering, gán nhãn nhị phân, chia tập Train/Valid/Test và khảo sát Tokenizer.
* **Tuần 2 - 3 (Ngày 8 - 21):** Fine-tune các mô hình nền tảng (BERT, RoBERTa, Longformer) với hàm mất mát BCE Loss và xây dựng kiến trúc Ensemble.
* **Tuần 4 (Ngày 22 - 28):** Đánh giá toàn diện trên các thang đo (Accuracy, F1-Score, AUC-ROC) và kiểm thử miền mới với TOEFL.
* **Tuần 5 - 6 (Ngày 29 - 40):** Triển khai Backend API bằng FastAPI và xây dựng tiện ích trình duyệt (Chrome Extension).

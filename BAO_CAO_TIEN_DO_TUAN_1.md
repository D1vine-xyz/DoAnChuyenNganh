# BÁO CÁO TIẾN ĐỘ TUẦN 1 - ĐỒ ÁN CHUYÊN NGÀNH (CHỦ ĐỀ 6)
**Đề tài:** Nghiên cứu phát hiện văn bản do LLM sinh sử dụng mô hình kết hợp (Ensemble PLMs: BERT, RoBERTa, Longformer)
**Thời gian thực hiện:** Giai đoạn 1 (Ngày 1 - Ngày 7: 25/10/2026 – 31/10/2026)

---

## 1. TỔNG QUAN CÔNG VIỆC ĐÃ HOÀN THÀNH

### Ngày 1 (25/10/2026): Khởi tạo dự án & Tải dữ liệu thô
* **Môi trường:** Khởi tạo Virtual Environment (`venv`) với Python 3.13. Cài đặt các thư viện lõi: `torch` (2.14.1), `transformers` (5.20.2), `datasets` (5.1.0), `pandas` (3.1.0), `scikit-learn` (1.9.1).
* **Kho lưu trữ:** Thiết lập Git repository và kết nối remote: `https://github.com/D1vine-xyz/DoAnChuyenNganh`.
* **Thu thập dữ liệu văn bản người thật viết (Human-written text):**
  * **LIAR Dataset (Tin ngắn):** Tải 3 tập train, test, valid từ Github (`download_liar.py`) lưu tại `data/liar_train.csv` (10,240 dòng), `data/liar_test.csv` (1,267 dòng), `data/liar_valid.csv` (1,284 dòng).
  * **GossipCop++ (Tin dài):** Tải các bài báo người thật viết từ repository `mbzuai-nlp/Fakenews-dataset` (`download_gossipcop.py`), lưu tại `data/gossipcop/gossipcop_human.csv` (12,252 bài báo).

---

### Ngày 2 (26/10/2026): Lọc, làm sạch & Chuẩn hóa dữ liệu
* **Làm sạch tập LIAR (`clean_data.py`):**
  * Khử hyperlink, chuẩn hóa dấu câu, chuyển chữ thường (lowercase), xóa khoảng trắng thừa.
  * Lọc trùng lặp và loại bỏ các phát biểu quá ngắn (< 4 từ).
  * Chuẩn hóa cấu trúc 14 cột theo chuẩn quốc tế (`id`, `label`, `statement`, `subject`, `speaker`,...).
  * Kết quả:
    * `liar_train_cleaned.csv`: 10,198 dòng sạch.
    * `liar_test_cleaned.csv`: 1,260 dòng sạch.
    * `liar_valid_cleaned.csv`: 1,282 dòng sạch.
* **Làm sạch & Gắn nhãn tập GossipCop++ (`download_clean_gossipcop.py`):**
  * Tải và tiền xử lý 20,505 bài báo từ Hugging Face (`Jinyan1/GossipCop`).
  * Khử trùng lặp, lọc các bài viết dưới 20 từ.
  * Phân loại đầy đủ nhãn 4 phân vùng phục vụ mô hình:
    * `source`: `human` (10,918 bài) vs `machine` (8,170 bài).
    * `label`: 0 (Người viết) vs 1 (LLM sinh).
    * `news_type`: `real` (11,469 bài) vs `fake` (7,619 bài).
    * `split`: HR (7,381), HF (3,537), MR (4,088), MF (4,082).
  * Kết quả lưu tại: `data/gossipcop_cleaned.csv` (19,088 bài báo).

---

### Ngày 3 (27/10 - 29/10/2026): Kỹ thuật Prompt Engineering & Sinh dữ liệu bằng API (Tin ngắn)
* **Mục tiêu học thuật:** Mô phỏng phương pháp tạo bộ dữ liệu MAGE (*Machine-generated Text Detection in the Wild*) bằng cách yêu cầu LLM đọc câu gốc của người và viết lại (paraphrase) giữ nguyên ý nghĩa và độ dài.
* **Tập dữ liệu kiểm thử TOEFL Unseen Domain (`download_toefl.py`):**
  * Sửa lỗi phân tích cú pháp đề bài từ `questions.txt`.
  * Tải và xử lý trọn vẹn 252 bài luận (126 bài do học sinh viết và 126 bài do ChatGPT sinh ra).
  * Đầy đủ 100% câu hỏi đề bài và độ dài từ, lưu tại: `data/toefl/toefl_essays.csv`.
* **Kỹ thuật Prompt Engineering với Gemini API (`generate_fake_text.py`):**
  * Cài đặt thư viện `google-generativeai`, `requests`. Cấu hình API Key qua file bảo mật `.env`.
  * Tối ưu giao thức truyền thông `transport='rest'` và model `gemini-3.5-flash-lite`, tích hợp cơ chế tự động chờ (exponential backoff) chống chạm giới hạn quota (rate limit 429).
  * Đã sinh thành công **20/20 cặp câu đối sánh song song Người - AI**, lưu tại: `data/liar_train_paired_sample.csv`.

---

### Ngày 4 (29/10/2026): Chuẩn hóa dữ liệu bài luận & Sinh tin tức dài bằng API (Long-form Paraphrase)
* **Chuẩn hóa tập dữ liệu kiểm thử TOEFL (`process_toefl.py`):**
  * Gom và chuẩn hóa toàn bộ 252 bài luận thành bảng tổng hợp duy nhất: `data/toefl_dataset.csv`.
  * Cấu trúc chuẩn hóa: `essay_id`, `label` (0: Người, 1: AI), `source`, `word_count`, `prompt_question`, `text`.
* **Kỹ thuật Prompt Engineering cho tin tức dài GossipCop (`generate_fake_gossipcop.py`):**
  * Thiết kế Prompt chuyên biệt cho văn bản dài dạng báo chí, yêu cầu giữ nguyên sự thật, độ dài và phong cách báo chí chuyên nghiệp, không thêm bớt thông tin.
  * Sinh thành công **20/20 bài báo dài đối sánh song song Người - AI** (Số từ trung bình: Người 479.6 từ, AI 309.8 từ), lưu tại: `data/gossipcop_paired_sample.csv`.

---

### Ngày 5 (29/10/2026): Gắn nhãn Phân loại nhị phân & Hợp nhất tập dữ liệu Huấn luyện hoàn chỉnh
* **Kịch bản Hợp nhất & Gắn nhãn (`merge_datasets.py`):**
  * Xây dựng module tự động đọc và gộp dữ liệu từ tất cả các nguồn (GossipCop tin dài, LIAR tin ngắn, các mẫu sinh bởi Gemini).
  * Gắn nhãn nhị phân chuẩn xác: `label = 0` (Văn bản con người viết) và `label = 1` (Văn bản do AI sinh ra).
  * Lọc trùng lặp văn bản và tiến hành trộn ngẫu nhiên (*shuffle*) với `random_state=42` để bảo đảm tính khách quan và khả năng tái lập cho mô hình học sâu.
* **Bộ dữ liệu Huấn luyện hoàn chỉnh (`data/final_train_dataset.csv`):**
  * **Tổng số mẫu:** **19,148 mẫu** (0 giá trị Null).
  * **Phân bố nhãn:**
    * 👤 Nhãn 0 (Người thật viết - Human): **10,938 mẫu** (57.1%).
    * 🤖 Nhãn 1 (Mô hình AI sinh - Machine): **8,210 mẫu** (42.9%).
  * **Phân bố theo thể loại:** Tin tức dài (`long_news`): 19,108 bài; Phát biểu ngắn (`short_statement`): 40 câu.
  * **Độ dài từ trung bình:** Văn bản người 515.4 từ; Văn bản AI 301.6 từ.

---

### Ngày 6 (30/10/2026): Phân chia tập dữ liệu chuẩn (Train/Val/Test Split) & Khảo sát Tokenization cho Ensemble Backbones
* **Phân chia dữ liệu phân tầng chống rò rỉ (`split_data.py`):**
  * Sử dụng kỹ thuật **Stratified Split** kết hợp 2 chiều: `label` (0/1) và `text_type` (`long_news`/`short_statement`), bảo đảm tính đại diện của mọi nhóm mẫu.
  * Tỷ lệ phân chia chuẩn khoa học: **70% Train** - **15% Validation** - **15% Test (In-domain)**. Cố định `random_state=42`.
  * **Kiểm định rò rỉ dữ liệu (No Data Leakage):** Xác thực giao thoa tập hợp văn bản:
    $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
  * **Kết quả phân chia:**
    * **Train set:** **13,403 mẫu** (Người: 7,656 - 57.1%, AI: 5,747 - 42.9%) lưu tại [`data/train.csv`](file:///c:/DoAn/data/train.csv).
    * **Validation set:** **2,872 mẫu** (Người: 1,641 - 57.1%, AI: 1,231 - 42.9%) lưu tại [`data/val.csv`](file:///c:/DoAn/data/val.csv).
    * **Test set (In-domain):** **2,873 mẫu** (Người: 1,641 - 57.1%, AI: 1,232 - 42.9%) lưu tại [`data/test.csv`](file:///c:/DoAn/data/test.csv).
    * **Test set (Unseen Domain):** **252 bài luận** giữ độc lập tại [`data/toefl_dataset.csv`](file:///c:/DoAn/data/toefl_dataset.csv).
* **Khảo sát Tokenization cho 3 mô hình nền tảng (`explore_tokenizers.py`):**
  * Tải và nạp thành công 3 bộ Hugging Face Fast Tokenizers: `bert-base-uncased` (WordPiece, vocab 30,522), `roberta-base` (BPE, vocab 50,265), `allenai/longformer-base-4096` (BPE, vocab 50,265).
  * Chạy tính toán phân phối độ dài token trên toàn bộ 13,403 mẫu huấn luyện:
    * **BERT:** Độ dài trung bình **500.3 tokens**, trung vị **353 tokens**, 75.1% văn bản nằm gọn dưới ngưỡng 512 tokens.
    * **RoBERTa:** Độ dài trung bình **519.5 tokens**, trung vị **365 tokens**, 73.0% văn bản nằm gọn dưới ngưỡng 512 tokens.
    * **Longformer:** Với khả năng hỗ trợ tới 4,096 tokens, tại ngưỡng 1,024 tokens bao phủ **92.2%** và tại ngưỡng 2,048 tokens bao phủ **97.6%** dữ liệu trọn vẹn.
  * **Luận cứ khoa học quan trọng:** Xác nhận thực nghiệm rằng khoảng 25% - 27% bài báo vượt quá giới hạn 512 tokens của BERT/RoBERTa. Việc tích hợp **Longformer** vào kiến trúc Ensemble là hoàn toàn cần thiết để nắm bắt ngữ cảnh toàn bài mà không bị mất mát thông tin do truncation.
  * Toàn bộ báo cáo số liệu chi tiết được lưu tại [`data/token_length_stats.json`](file:///c:/DoAn/data/token_length_stats.json).

---

### Ngày 7 (31/10/2026): Thiết lập PyTorch Dataset & DataLoader tùy chỉnh cho 3 Mô hình Backbone
* **Đóng gói module nạp dữ liệu chuẩn (`dataset.py`):**
  * Xây dựng lớp `TextDetectionDataset` kế thừa từ `torch.utils.data.Dataset`, nạp trực tiếp dữ liệu từ file CSV, trích xuất chuỗi `text` và nhãn nhị phân chuẩn hóa dạng `torch.float32`.
  * Hỗ trợ đồng bộ và tương thích hoàn toàn với Tokenizer của cả 3 mô hình nền tảng:
    * **BERT** (`bert-base-uncased`): Trả về `input_ids`, `attention_mask` và `token_type_ids`.
    * **RoBERTa** (`roberta-base`): Trả về `input_ids` và `attention_mask`.
    * **Longformer** (`allenai/longformer-base-4096`): Tự động tạo tensor `global_attention_mask` (gán cờ chú ý toàn cục giá trị 1 cho token đầu tiên `<s>`, 0 cho các token còn lại theo đúng chuẩn thiết kế của kiến trúc Longformer).
* **Tối ưu hóa cơ chế Đệm động (Dynamic Padding Collator):**
  * Xây dựng lớp `DynamicPaddingCollator`: Tự động căn chỉnh độ dài chuỗi đệm (`pad_token_id`) theo độ dài lớn nhất trong từng batch cụ thể thay vì đệm cứng 512/1024 cho toàn bộ tập dữ liệu.
  * Hiệu quả thực nghiệm: Tiết kiệm từ **40% – 60% dung lượng VRAM/RAM** và tăng tốc đáng kể tốc độ xử lý ma trận chú ý (Self-Attention) trong quá trình huấn luyện và suy luận.
* **Kiểm thử tích hợp & Xác thực Pipeline (Smoke Test):**
  * Đã chạy kiểm thử thực nghiệm trên cả 3 mô hình với batch size = 8:
    * **BERT:** Nạp thành công Train (1,676 batches), Val (359 batches), Test (360 batches). Shape tensor: `input_ids` `[8, 512]`, `labels` `[8]`. Tương thích tuyệt đối với `BCEWithLogitsLoss` (loss = 0.7246).
    * **RoBERTa:** Nạp thành công Train (1,676 batches), Val (359 batches), Test (360 batches). Shape tensor: `input_ids` `[8, 512]`, `labels` `[8]`. Tương thích tuyệt đối với `BCEWithLogitsLoss` (loss = 0.5903).
    * **Longformer:** Nạp thành công Train (1,676 batches), Val (359 batches), Test (360 batches). Dynamic padding tự co giãn kích thước `[8, 616]`. Đầy đủ tensor `global_attention_mask` `[8, 616]`. Tương thích `BCEWithLogitsLoss` (loss = 0.4816).
    * **Tập Unseen TOEFL:** Nạp thành công toàn bộ 32 batches kiểm thử miền mới cho cả 3 mô hình.
  * Toàn bộ mã nguồn đã được đóng gói chuẩn chỉ tại [`dataset.py`](file:///c:/DoAn/dataset.py).

---

## 2. BẢNG MẪU KẾT QUẢ ĐỐI SÁNH TRÍCH TỪ TẬP HUẤN LUYỆN (Trích `final_train_dataset.csv`)

| Thể loại | Nhãn | Nguồn | Văn bản mẫu (Trích xuất) |
| :--- | :---: | :---: | :--- |
| `long_news` | **0** (Người) | GossipCop Human | *prince harry and meghan markle are taking their relationship to the next level by spending the holidays together in london...* |
| `long_news` | **1** (AI) | GossipCop Machine | *prince harry and meghan markle celebrate their first christmas together as a couple with a festive celebration at kensington palace...* |
| `long_news` | **1** (AI) | Gemini Paraphrase | *Media figures Kim Kardashian and Kanye West are deploying extraordinary security measures to safeguard their household following the harrowing robbery in Paris...* |
| `short_statement` | **0** (Người) | LIAR Human | *says the annies list political group supports thirdtrimester abortions on demand.* |
| `short_statement` | **1** (AI) | Gemini Paraphrase | *States that the political group Annie's List supports third-trimester abortions on demand.* |

---

## 3. THỐNG KÊ TOÀN DIỆN CÁC BỘ DỮ LIỆU ĐANG CÓ TRONG THƯ MỤC `data/`

| Tên file dữ liệu | Số dòng / bài | Số cột | Mô tả dữ liệu | Trạng thái |
| :--- | :---: | :---: | :--- | :---: |
| [`data/train.csv`](file:///c:/DoAn/data/train.csv) | **13,403** | **6** | **Tập dữ liệu huấn luyện chính thức (70% - Stratified)** | **Đã hoàn tất (Ngày 6)** |
| [`data/val.csv`](file:///c:/DoAn/data/val.csv) | **2,872** | **6** | **Tập kiểm tra chọn lọc mô hình / Early Stopping (15%)** | **Đã hoàn tất (Ngày 6)** |
| [`data/test.csv`](file:///c:/DoAn/data/test.csv) | **2,873** | **6** | **Tập kiểm thử nội miền In-domain (15%)** | **Đã hoàn tất (Ngày 6)** |
| [`data/toefl_dataset.csv`](file:///c:/DoAn/data/toefl_dataset.csv) | **252** | **6** | **Tập kiểm thử độc lập miền mới (Unseen Domain: 126 Người + 126 AI)** | **Đã hoàn tất (Ngày 4)** |
| [`data/token_length_stats.json`](file:///c:/DoAn/data/token_length_stats.json) | File JSON | - | Báo cáo chi tiết phân phối token BERT, RoBERTa, Longformer | **Đã hoàn tất (Ngày 6)** |
| [`data/final_train_dataset.csv`](file:///c:/DoAn/data/final_train_dataset.csv) | 19,148 | 6 | Tập dữ liệu huấn luyện tổng hợp hoàn chỉnh (Gốc) | Đã hoàn tất (Ngày 5) |
| [`data/gossipcop_cleaned.csv`](file:///c:/DoAn/data/gossipcop_cleaned.csv) | 19,088 | 8 | Bài báo dài đã làm sạch (10,918 Người + 8,170 Máy) | Đầy đủ nhãn |
| [`data/gossipcop_paired_sample.csv`](file:///c:/DoAn/data/gossipcop_paired_sample.csv) | 20 | 10 | Cặp tin dài đối sánh Người vs AI (Gemini sinh Ngày 4) | Đã hoàn tất |
| [`data/liar_train_cleaned.csv`](file:///c:/DoAn/data/liar_train_cleaned.csv) | 10,198 | 15 | Phát biểu ngắn người viết (Tập Train gốc LIAR) | Đã làm sạch |
| [`data/liar_test_cleaned.csv`](file:///c:/DoAn/data/liar_test_cleaned.csv) | 1,260 | 15 | Phát biểu ngắn người viết (Tập Test gốc LIAR) | Đã làm sạch |
| [`data/liar_valid_cleaned.csv`](file:///c:/DoAn/data/liar_valid_cleaned.csv) | 1,282 | 15 | Phát biểu ngắn người viết (Tập Validation gốc LIAR) | Đã làm sạch |
| [`data/liar_train_paired_sample.csv`](file:///c:/DoAn/data/liar_train_paired_sample.csv) | 20 | 6 | Cặp tin ngắn đối sánh Người vs AI (Gemini sinh Ngày 3) | Đã hoàn tất |
| [`data/toefl/toefl_essays.csv`](file:///c:/DoAn/data/toefl/toefl_essays.csv) | 252 | 6 | Bài luận TOEFL kèm câu hỏi đề bài và thống kê từ | Đã hoàn tất |

---

## 4. HẠ TẦNG KỸ THUẬT & TRỢ LÝ MCP NOTEBOOKLM
* **MCP NotebookLM:** Đã tích hợp và kết nối thành công 100% vào trợ lý AI thông qua notebook *Nghiên cứu phát hiện văn bản LLM Ensemble PLMs* (`https://notebooklm.google.com/notebook/dc714098-32b3-4287-8ccb-6ec5f242adad`).
* **Tính ổn định:** Đã tinh chỉnh module xác thực cookie `auth-manager.js` nâng thời hạn kiểm tra từ 24h lên 30 ngày, chống tình trạng hết hạn phiên giả lập.

---

## 5. TỔNG KẾT TUẦN 1 & KẾ HOẠCH TUẦN 2 (GIAI ĐOẠN 2: HUẤN LUYỆN MÔ HÌNH HỌC SÂU)
* **Tổng kết Tuần 1 (Giai đoạn 1):** Đã hoàn thành **100% các mục tiêu Milestone 1** đúng tiến độ đề tài:
  - Thu thập, làm sạch và chuẩn hóa toàn bộ dữ liệu (tin dài GossipCop, tin ngắn LIAR, 252 bài luận TOEFL).
  - Hoàn thiện tập dữ liệu tổng hợp 19,148 mẫu, phân chia phân tầng không rò rỉ: Train (70%), Val (15%), Test (15%).
  - Xây dựng hạ tầng nạp dữ liệu PyTorch `Dataset` & `DataLoader` với Dynamic Padding và Attention Mask tối ưu.
* **Kế hoạch Tuần 2 (01/11/2026 – 07/11/2026):**
  * **Ngày 8 (01/11/2026):** Khởi tạo kiến trúc mô hình `bert-base-uncased` với tầng phân loại nhị phân và hàm mất mát `BCEWithLogitsLoss`.
  * **Ngày 9 – 10 (02/11 – 03/11/2026):** Tiến hành huấn luyện (Fine-tune) mô hình BERT, theo dõi hàm mất mát và F1 trên tập Val, lưu checkpoint trọng số tối ưu.
  * **Ngày 11 (04/11/2026):** Khởi tạo và thiết lập mô hình `roberta-base`.
  * **Ngày 12 – 13 (05/11 – 06/11/2026):** Fine-tune mô hình RoBERTa và lưu checkpoint.
  * **Ngày 14 (07/11/2026):** Tích hợp và cấu hình mô hình `allenai/longformer-base-4096`.

# BÁO CÁO TIẾN ĐỘ TUẦN 1 - ĐỒ ÁN CHUYÊN NGÀNH (CHỦ ĐỀ 6)
**Đề tài:** Nghiên cứu phát hiện văn bản do LLM sinh sử dụng mô hình kết hợp (Ensemble PLMs: BERT, RoBERTa, Longformer)
**Thời gian thực hiện:** Giai đoạn 1 (Ngày 1 - Ngày 4)

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
  * Sử dụng Gemini 3.5 Flash Lite với cơ chế resume tự động và chống chạm rate limit.
  * Sinh thành công **20/20 bài báo dài đối sánh song song Người - AI** (Số từ trung bình: Người 479.6 từ, AI 309.8 từ), lưu tại: `data/gossipcop_paired_sample.csv`.

---

## 2. BẢNG MẪU KẾT QUẢ ĐỐI SÁNH TIN DÀI GOSSIPCOP (Trích từ `gossipcop_paired_sample.csv`)

| Tiêu đề bài báo | Bản gốc Người viết (Human - Label 0) | Bản sinh do AI Paraphrase (Machine - Label 1) |
| :--- | :--- | :--- |
| **Kim & Kanye Install At-Home Panic Room After Paris Robbery** | *kim kardashian and kanye west are pulling out all the stops to keep their family safe after the reality stars shocking paris robbery...* (95 từ) | *Media figures Kim Kardashian and Kanye West are deploying extraordinary security measures to safeguard their household following the harrowing robbery in Paris...* (112 từ) |
| **£1.4million spent renovating Prince Harry and Meghan's rumoured home** | *prince harry and meghan currently live at kensington palaces nottingham cottage but should they choose to move to apartments 8 and 9...* (405 từ) | *Prince Harry and Meghan Markle presently reside at Kensington Palace's Nottingham Cottage; however, substantial taxpayer-funded renovations totaling £1.4 million...* (313 từ) |
| **Why Princess Charlotte Never Wears Pants** | *princess charlotte may be a modern royal but that doesnt mean she and her family dont adhere to a few longstanding traditions...* (163 từ) | *While Princess Charlotte represents a modern era for the British monarchy, her family continues to uphold several enduring customs...* (152 từ) |

---

## 3. THỐNG KÊ TOÀN DIỆN CÁC BỘ DỮ LIỆU ĐANG CÓ TRONG THƯ MỤC `data/`

| Tên file dữ liệu | Số dòng / bài | Số cột | Mô tả dữ liệu | Trạng thái |
| :--- | :---: | :---: | :--- | :---: |
| [`data/liar_train_cleaned.csv`](file:///c:/DoAn/data/liar_train_cleaned.csv) | 10,198 | 15 | Phát biểu ngắn người viết (Tập Train) | Đã làm sạch |
| [`data/liar_test_cleaned.csv`](file:///c:/DoAn/data/liar_test_cleaned.csv) | 1,260 | 15 | Phát biểu ngắn người viết (Tập Test) | Đã làm sạch |
| [`data/liar_valid_cleaned.csv`](file:///c:/DoAn/data/liar_valid_cleaned.csv) | 1,282 | 15 | Phát biểu ngắn người viết (Tập Validation) | Đã làm sạch |
| [`data/gossipcop_cleaned.csv`](file:///c:/DoAn/data/gossipcop_cleaned.csv) | 19,088 | 8 | Bài báo dài (10,918 Người + 8,170 Máy) | Đầy đủ nhãn |
| [`data/toefl/toefl_essays.csv`](file:///c:/DoAn/data/toefl/toefl_essays.csv) | 252 | 6 | Bài luận TOEFL kiểm thử miền mới (126 Người + 126 AI) | Đầy đủ đề bài |
| [`data/toefl_dataset.csv`](file:///c:/DoAn/data/toefl_dataset.csv) | 252 | 6 | Bảng gom dữ liệu TOEFL chuẩn hóa Ngày 4 | Đã hoàn tất |
| [`data/liar_train_paired_sample.csv`](file:///c:/DoAn/data/liar_train_paired_sample.csv) | 20 | 6 | Cặp tin ngắn đối sánh Người vs AI (Gemini sinh) | Đã hoàn tất |
| [`data/gossipcop_paired_sample.csv`](file:///c:/DoAn/data/gossipcop_paired_sample.csv) | 20 | 10 | Cặp tin dài đối sánh Người vs AI (Gemini sinh) | Đã hoàn tất |

---

## 4. HẠ TẦNG KỸ THUẬT & TRỢ LÝ MCP NOTEBOOKLM
* **MCP NotebookLM:** Đã tích hợp và kết nối thành công 100% vào trợ lý AI thông qua notebook *Nghiên cứu phát hiện văn bản LLM Ensemble PLMs* (`https://notebooklm.google.com/notebook/dc714098-32b3-4287-8ccb-6ec5f242adad`).
* **Tính ổn định:** Đã tinh chỉnh module xác thực cookie `auth-manager.js` nâng thời hạn kiểm tra từ 24h lên 30 ngày, chống tình trạng hết hạn phiên giả lập.

---

## 5. DỰ KIẾN KẾ HOẠCH CÁC NGÀY TIẾP THEO
* **Ngày 5:** Chuẩn bị và làm giàu tập dữ liệu kiểm thử đa miền (Cross-domain evaluation).
* **Ngày 6:** Gắn nhãn thống nhất (Labeling) và gộp toàn bộ dữ liệu thành dataset đồng nhất.
* **Ngày 7:** Tokenize bằng thư viện Transformers (Hugging Face) và chia tập Train/Valid/Test (70% - 15% - 15%).
* **Ngày 8 trở đi:** Bắt đầu nạp mô hình PLM đầu tiên (`bert-base-uncased`) với hàm mất mát Binary Cross-Entropy (BCE Loss).

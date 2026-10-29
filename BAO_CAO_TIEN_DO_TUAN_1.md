# BÁO CÁO TIẾN ĐỘ TUẦN 1 - ĐỒ ÁN CHUYÊN NGÀNH (CHỦ ĐỀ 6)
**Đề tài:** Nghiên cứu phát hiện văn bản do LLM sinh sử dụng mô hình kết hợp (Ensemble PLMs: BERT, RoBERTa, Longformer)
**Thời gian thực hiện:** Giai đoạn 1 (Ngày 1 - Ngày 3)

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

### Ngày 3 (27/10 - 29/10/2026): Kỹ thuật Prompt Engineering & Sinh dữ liệu bằng API
* **Mục tiêu học thuật:** Mô phỏng phương pháp tạo bộ dữ liệu MAGE (*Machine-generated Text Detection in the Wild*) bằng cách yêu cầu LLM đọc câu gốc của người và viết lại (paraphrase) giữ nguyên ý nghĩa và độ dài.
* **Tập dữ liệu kiểm thử TOEFL Unseen Domain (`download_toefl.py`):**
  * Sửa lỗi phân tích cú pháp đề bài từ `questions.txt`.
  * Tải và xử lý trọn vẹn 252 bài luận (126 bài do học sinh viết và 126 bài do ChatGPT sinh ra).
  * Đầy đủ 100% câu hỏi đề bài và độ dài từ, lưu tại: `data/toefl/toefl_essays.csv`.
* **Kỹ thuật Prompt Engineering với Gemini API (`generate_fake_text.py`):**
  * Cài đặt thư viện `google-generativeai`, `requests`. Cấu hình API Key qua file bảo mật `.env`.
  * Thiết kế Prompt chuyên biệt để LLM chỉ trả về câu viết lại, không thêm lời chào, không giải thích:
    > *"Rewrite the following short statement naturally, keeping the exact core meaning, tone, and approximate length. Do NOT include any introductory remarks, no explanations, and no surrounding quotation marks: '{original_text}'"*
  * Tối ưu giao thức truyền thông `transport='rest'` và model `gemini-3.5-flash-lite`, tích hợp cơ chế tự động chờ (exponential backoff) chống chạm giới hạn quota (rate limit 429).
  * Đã sinh thành công **20/20 cặp câu đối sánh song song Người - AI**, lưu tại: `data/liar_train_paired_sample.csv`.

---

## 2. BẢNG MẪU KẾT QUẢ ĐỐI SÁNH NGƯỜI - MÁY (Trích xuất từ `liar_train_paired_sample.csv`)

| ID | Người thật viết (Human - Label 0) | AI Paraphrase (Machine - Label 1) | Nhãn sự thật gốc |
| :--- | :--- | :--- | :---: |
| `2635.json` | says the annies list political group supports thirdtrimester abortions on demand. | States that the political group Annie's List supports third-trimester abortions on demand. | false |
| `10540.json` | when did the decline of coal start it started when natural gas took off that started to begin in president george w. bushs administration. | The decline of coal started when natural gas took off, which began during President George W. Bush's administration. | half-true |
| `324.json` | the economic turnaround started at the end of my term. | The economic recovery began toward the end of my term. | barely-true |
| `1123.json` | the chicago bears have had more starting quarterbacks in the last 10 years than the green bay packers have had in the last 30. | The Chicago Bears have gone through more starting quarterbacks in the past ten years than the Green Bay Packers have in the last thirty. | true |
| `1268.json` | since 2000 nearly 12 million americans have slipped out of the middle class and into poverty. | Almost 12 million Americans have fallen from the middle class into poverty since 2000. | half-true |
| `1128.json` | the economy bled 24 billion due to the government shutdown. | The government shutdown drained $24 billion from the economy. | true |
| `4307.json` | water rates in manila philippines were raised up to 845 percent following the privatization of water services. | water rates in manila philippines surged by as much as 845 percent after water services were privatized | half-true |
| `943.json` | almost 100000 people left puerto rico last year. | Nearly 100,000 people left Puerto Rico last year. | mostly-true |

---

## 3. THỐNG KÊ CÁC BỘ DỮ LIỆU ĐANG CÓ TRONG THƯ MỤC `data/`

| Tên file dữ liệu | Số dòng / bài | Số cột | Mô tả dữ liệu | Trạng thái |
| :--- | :---: | :---: | :--- | :---: |
| [`data/liar_train_cleaned.csv`](file:///c:/DoAn/data/liar_train_cleaned.csv) | 10,198 | 15 | Phát biểu ngắn người viết (Tập Train) | Đã làm sạch |
| [`data/liar_test_cleaned.csv`](file:///c:/DoAn/data/liar_test_cleaned.csv) | 1,260 | 15 | Phát biểu ngắn người viết (Tập Test) | Đã làm sạch |
| [`data/liar_valid_cleaned.csv`](file:///c:/DoAn/data/liar_valid_cleaned.csv) | 1,282 | 15 | Phát biểu ngắn người viết (Tập Validation) | Đã làm sạch |
| [`data/gossipcop_cleaned.csv`](file:///c:/DoAn/data/gossipcop_cleaned.csv) | 19,088 | 8 | Bài báo dài (10,918 Người + 8,170 Máy) | Đầy đủ nhãn |
| [`data/toefl/toefl_essays.csv`](file:///c:/DoAn/data/toefl/toefl_essays.csv) | 252 | 6 | Bài luận kiểm thử miền mới (126 Người + 126 AI) | Đầy đủ đề bài |
| [`data/liar_train_paired_sample.csv`](file:///c:/DoAn/data/liar_train_paired_sample.csv) | 20 | 6 | Cặp câu đối sánh Người vs AI do Gemini sinh | Đã hoàn tất |

---

## 4. HẠ TẦNG KỸ THUẬT & TRỢ LÝ MCP NOTEBOOKLM
* **MCP NotebookLM:** Đã tích hợp và kết nối thành công 100% vào trợ lý AI thông qua notebook *Nghiên cứu phát hiện văn bản LLM Ensemble PLMs* (`https://notebooklm.google.com/notebook/dc714098-32b3-4287-8ccb-6ec5f242adad`).
* **Tính ổn định:** Đã tinh chỉnh module xác thực cookie `auth-manager.js` nâng thời hạn kiểm tra từ 24h lên 30 ngày, chống tình trạng hết hạn phiên giả lập.

---

## 5. DỰ KIẾN KẾ HOẠCH TUẦN TIẾP THEO (Giai đoạn 1 hoàn thiện & Bắt đầu Giai đoạn 2)
* **Ngày 4:** Mở rộng sinh dữ liệu AI paraphrase cho tập LIAR trên quy mô lớn hơn.
* **Ngày 5 - 6:** Hợp nhất toàn bộ dữ liệu, chuẩn hóa phân phối nhãn (0: Người, 1: Máy).
* **Ngày 7:** Chạy Tokenizer của Hugging Face và chia tập Train/Valid/Test chuẩn (70% - 15% - 15%).
* **Ngày 8 trở đi:** Bắt đầu nạp mô hình PLM đầu tiên (`bert-base-uncased`) với hàm mất mát Binary Cross-Entropy (BCE Loss).


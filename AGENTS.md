# Quy tắc dự án VeriText AI (c:\DoAn)

## 1. Môi trường & Lệnh chuẩn
- Hệ điều hành: Windows (PowerShell). Luôn chạy Python bằng môi trường ảo: `c:\DoAn\venv\Scripts\python.exe`.
- Khởi chạy server: chạy `backend/run_server.py` (cổng `http://localhost:8000`) hoặc `start_server.bat`.
- Chạy bộ kiểm thử tự động (từ thư mục `backend/`):
  `..\venv\Scripts\python.exe -m unittest test_forensics.py test_batch.py test_api.py -v`

## 2. Kiến trúc & Số liệu hiện hành (Bản nâng cấp mới nhất)
- **Tuyến Tiếng Anh (`en`):** Ensemble 3 PLMs (BERT + RoBERTa + Longformer 1024) + Temperature Scaling + Stacking Meta-Classifier huấn luyện trên `english_v2_*.csv` (12.000 mẫu đa miền) — Test đa miền Acc `95.08%`, F1 `95.14%`; TOEFL Zero-shot `83.33%` (Recall `100%`).
- **Tuyến Tiếng Việt (`vi`):** PhoBERT base v2 + PyVi huấn luyện trên `vietnamese_v3_*.csv` (v3.1 — 15.000 mẫu Tiền ChatGPT <= 2022, 22 phân nhóm) — Test Acc `98.58%`, F1 `98.59%`, AUC `0.9989`.
- **CSDL:** Microsoft SQL Server 2022 (`VeriTextAI`), tài khoản demo mặc định: `giangvien@veritext.ai` / `sinhvien@veritext.ai` (mật khẩu `123456`).

## 3. Quy tắc an toàn bắt buộc
- KHÔNG xóa hoặc ghi đè các file checkpoint `backend/output/*_best.pt` và thư mục `backend/output/backup_pre_option_b/` khi chưa có xác nhận rõ ràng.
- Khi sửa API hoặc UI liên quan đến RBAC / Anti-Bypass / PDF QR, luôn chạy lại bộ test tương ứng để xác nhận xanh 100%.
- Trả lời ngắn gọn bằng tiếng Việt; mọi số liệu báo cáo phải đọc trực tiếp từ file JSON/CSV trong `backend/output/`, không suy đoán.
---
name: kaggle-pipeline
description: >-
  Quy trình đẩy job huấn luyện BERT, RoBERTa, Longformer, PhoBERT hoặc đánh giá
  Ensemble lên Kaggle GPU Tesla T4, theo dõi trạng thái định kỳ và tải kết quả
  về backend/output/. Kích hoạt khi người dùng yêu cầu train lại mô hình, đẩy
  kernel lên Kaggle, kiểm tra tiến độ Kaggle hoặc tải checkpoint về máy.
---

# Quy Trình Huấn Luyện Từ Xa Trên Kaggle GPU (VeriText AI)

## Danh mục 5 Kernel của dự án
- BERT: `backend/kaggle_jobs/train_bert` -> slug `quannguyenxx/do-an-train-bert`
- RoBERTa: `backend/kaggle_jobs/train_roberta` -> slug `quannguyenxx/do-an-train-roberta`
- Longformer: `backend/kaggle_jobs/train_longformer` -> slug `quannguyenxx/do-an-train-longformer`
- PhoBERT: `backend/kaggle_jobs/train_phobert` -> slug `quannguyenxx/do-an-train-phobert`
- Ensemble Eval: `backend/kaggle_jobs/evaluate_ensemble` -> slug `quannguyenxx/do-an-ensemble-eval`

## Các bước thực thi bắt buộc
1. **Đồng bộ mã nguồn lên GitHub trước khi đẩy:** Các script `run_training.py` trên Kaggle tự `git clone` từ repo GitHub của dự án. Luôn kiểm tra `git status`, nhắc người dùng commit & push mọi thay đổi trong `backend/` trước khi push kernel.
2. **Đẩy job lên Kaggle:**
   `kaggle kernels push -p c:\DoAn\backend\kaggle_jobs\<tên_thư_mục>`
3. **Theo dõi tiến độ (không dùng vòng lặp sleep):**
   Dùng công cụ `schedule` (hoặc hướng dẫn người dùng `/schedule`) kiểm tra định kỳ `kaggle kernels status <slug>` mỗi 10 phút cho tới khi trạng thái là `COMPLETE` hoặc `ERROR`.
4. **Tải kết quả & kiểm chứng:**
   - Khi `COMPLETE`, sao lưu checkpoint cũ (`backend/output/*_best.pt`) và hỏi xác nhận trước khi ghi đè, sau đó tải về:
     `kaggle kernels output <slug> -p c:\DoAn\backend\output`
   - Đọc file `*_test_results.json` mới tải về và báo cáo bảng so sánh Accuracy / F1 / AUC với phiên bản cũ.

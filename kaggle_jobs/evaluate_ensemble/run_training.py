import os
import sys
import subprocess
import shutil

print("=" * 80)
print("  KAGGLE REMOTE JOB: THỰC NGHIỆM ĐÁNH GIÁ MÔ HÌNH HỌC KẾT HỢP (ENSEMBLE PLMs)")
print("=" * 80)

# 1. Kiểm tra GPU CUDA trên máy chủ Kaggle
try:
    import torch
    print(f"[*] Phiên bản PyTorch : {torch.__version__}")
    print(f"[*] CUDA Sẵn sàng     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] Tên GPU           : {torch.cuda.get_device_name(0)}")
        print(f"[*] Bộ nhớ GPU        : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("[!] CẢNH BÁO: Không tìm thấy GPU! Kiểm tra 'enable_gpu: true'.")
except Exception as e:
    print(f"[!] Lỗi kiểm tra GPU: {e}")

# 2. Clone mã nguồn từ GitHub vào thư mục tạm /tmp để giữ sạch Kaggle Output
REPO_URL = "https://github.com/D1vine-xyz/DoAnChuyenNganh.git"
CLONE_DIR = "/tmp/DoAnChuyenNganh"

if os.path.exists(CLONE_DIR):
    print(f"[*] Thư mục mã nguồn {CLONE_DIR} đã tồn tại, cập nhật git pull...")
    subprocess.run(["git", "-C", CLONE_DIR, "pull"], check=False)
else:
    print(f"[*] Đang tải mã nguồn từ GitHub: {REPO_URL} ...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CLONE_DIR], check=True)

# 3. Chuyển vào thư mục dự án
os.chdir(CLONE_DIR)
sys.path.insert(0, CLONE_DIR)

# 4. Tìm kiếm tự động các file checkpoint trong /kaggle/input
def find_checkpoint(filename):
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                found_path = os.path.join(root, filename)
                print(f"  [v] Tìm thấy {filename} tại: {found_path}")
                return found_path
    local_path = os.path.join("output", filename)
    if os.path.exists(local_path):
        print(f"  [v] Tìm thấy {filename} cục bộ tại: {local_path}")
        return local_path
    print(f"  [X] KHÔNG tìm thấy: {filename}")
    return None

bert_ckpt = find_checkpoint("bert_best.pt")
roberta_ckpt = find_checkpoint("roberta_best.pt")
longformer_ckpt = find_checkpoint("longformer_best.pt")

if not (bert_ckpt and roberta_ckpt and longformer_ckpt):
    print("\n[!] LỖI: Thiếu checkpoint của 1 hoặc nhiều mô hình. Kiểm tra lại kernel_sources.")
    sys.exit(1)

# 5. Thiết lập thư mục Output trên Kaggle (/kaggle/working)
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 6. Khởi động thực nghiệm Ensemble
print("\n[*] BẮT ĐẦU CHẠY THỰC NGHIỆM ĐÁNH GIÁ ENSEMBLE TRÊN TẬP VAL VÀ TEST...")
eval_cmd = [
    sys.executable,
    "ensemble.py",
    "--data_dir", "data",
    "--output_dir", OUTPUT_DIR,
    "--bert_path", bert_ckpt,
    "--roberta_path", roberta_ckpt,
    "--longformer_path", longformer_ckpt,
    "--batch_size", "16",
    "--fp16"
]

result = subprocess.run(eval_cmd)
if result.returncode == 0:
    print("\n" + "=" * 80)
    print("  [v] ĐÁNH GIÁ ENSEMBLE THÀNH CÔNG VÀ HOÀN TẤT TRÊN KAGGLE GPU!")
    print(f"  - Comparison Table   : {OUTPUT_DIR}/ensemble_comparison_table.csv")
    print(f"  - Detailed Metrics   : {OUTPUT_DIR}/ensemble_test_results.json")
    print(f"  - Optimal Weights    : {OUTPUT_DIR}/ensemble_weights.json")
    print(f"  - Stacking Model     : {OUTPUT_DIR}/stacking_meta_learner.joblib")
    print(f"  - Test Predictions   : {OUTPUT_DIR}/test_predictions_ensemble.csv")
    print("=" * 80)
else:
    print(f"\n[!] Tiến trình đánh giá gặp lỗi (Return code: {result.returncode})")
    sys.exit(result.returncode)

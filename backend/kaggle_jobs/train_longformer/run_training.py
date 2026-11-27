import os
import sys
import subprocess
import shutil

print("=" * 75)
print("  KAGGLE REMOTE JOB: HUẤN LUYỆN MÔ HÌNH LONGFORMER (GPU ACCELERATED)")
print("=" * 75)

# 1. Kiểm tra GPU CUDA trên máy chủ Kaggle
try:
    import torch
    print(f"[*] Phiên bản PyTorch : {torch.__version__}")
    print(f"[*] CUDA Sẵn sàng     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] Tên GPU           : {torch.cuda.get_device_name(0)}")
        print(f"[*] Bộ nhớ GPU        : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("[!] CẢNH BÁO: Không tìm thấy GPU! Kiểm tra 'enable_gpu: true' trong kernel-metadata.json.")
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

# 3. Chuyển vào thư mục backend của dự án
BACKEND_DIR = os.path.join(CLONE_DIR, "backend")
os.chdir(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

# 4. Ưu tiên nạp bộ dữ liệu Tiếng Anh Đa Miền v2 (News + Student Essays + HC3 + Humanized AI)
if os.path.exists("data/english_v2_train.csv"):
    TRAIN_CSV = "data/english_v2_train.csv"
    VAL_CSV = "data/english_v2_val.csv"
    TEST_CSV = "data/english_v2_test.csv"
else:
    TRAIN_CSV = "data/train.csv"
    VAL_CSV = "data/val.csv"
    TEST_CSV = "data/test.csv"

for f in [TRAIN_CSV, VAL_CSV, TEST_CSV, "dataset.py", "models.py", "train_longformer.py"]:
    if os.path.exists(f):
        print(f"  [v] Sẵn sàng: {f}")
    else:
        print(f"  [X] Thiếu file: {f}")

# 5. Thiết lập thư mục Output trên Kaggle (/kaggle/working)
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 6. Khởi động huấn luyện Longformer với Mixed Precision FP16, Length Bucketing (2.5x nhanh hơn) & Label Smoothing
print("\n[*] BẮT ĐẦU CHẠY HUẤN LUYỆN LONGFORMER ĐA MIỀN (MAX_LENGTH=1024, LENGTH BUCKETING, 2 EPOCHS)...")
train_cmd = [
    sys.executable,
    "train_longformer.py",
    "--data_dir", "data",
    "--train_file", TRAIN_CSV,
    "--val_file", VAL_CSV,
    "--test_file", TEST_CSV,
    "--output_dir", OUTPUT_DIR,
    "--epochs", "2",
    "--batch_size", "4",
    "--gradient_accumulation_steps", "4",
    "--max_length", "1024",
    "--lr", "2.5e-5",
    "--weight_decay", "0.01",
    "--warmup_ratio", "0.1",
    "--label_smoothing", "0.05",
    "--length_bucketing",
    "--seed", "42",
    "--fp16"
]

result = subprocess.run(train_cmd)
if result.returncode == 0:
    print("\n" + "=" * 75)
    print("  [v] HUẤN LUYỆN THÀNH CÔNG VÀ HOÀN TẤT LONGFORMER TRÊN KAGGLE GPU!")
    print(f"  - Model Checkpoint : {OUTPUT_DIR}/longformer_best.pt")
    print(f"  - Training Log     : {OUTPUT_DIR}/training_log_longformer.csv")
    print(f"  - Test Evaluation  : {OUTPUT_DIR}/longformer_test_results.json")
    print("=" * 75)
else:
    print(f"\n[!] Tiến trình huấn luyện gặp lỗi (Return code: {result.returncode})")
    sys.exit(result.returncode)

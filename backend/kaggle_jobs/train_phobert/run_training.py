import os
import sys
import subprocess
import shutil

print("=" * 75)
print("  KAGGLE REMOTE JOB: HUẤN LUYỆN MÔ HÌNH PHOBERT TIẾNG VIỆT (TESLA T4 GPU)")
print("=" * 75)

# 1. Kiểm tra GPU CUDA
try:
    import torch
    print(f"[*] Phiên bản PyTorch : {torch.__version__}")
    print(f"[*] CUDA Sẵn sàng     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] Tên GPU           : {torch.cuda.get_device_name(0)}")
        print(f"[*] Bộ nhớ GPU        : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("[!] CẢNH BÁO: Không tìm thấy GPU!")
except Exception as e:
    print(f"[!] Lỗi kiểm tra GPU: {e}")

# 2. Cài đặt thư viện pyvi cho tiếng Việt
print("[*] Đang cài đặt thư viện pyvi...")
subprocess.run([sys.executable, "-m", "pip", "install", "pyvi"], check=False)

# 3. Clone mã nguồn từ GitHub
REPO_URL = "https://github.com/D1vine-xyz/DoAnChuyenNganh.git"
CLONE_DIR = "/tmp/DoAnChuyenNganh"

if os.path.exists(CLONE_DIR):
    shutil.rmtree(CLONE_DIR, ignore_errors=True)

print(f"[*] Đang tải mã nguồn từ GitHub: {REPO_URL} ...")
subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CLONE_DIR], check=True)

BACKEND_DIR = os.path.join(CLONE_DIR, "backend")
os.chdir(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

# 4. Ưu tiên nạp bộ dữ liệu Tiếng Việt v3 (12K Academic Pre-2022 + Humanized AI)
if os.path.exists("data/vietnamese_v3_train.csv"):
    TRAIN_CSV = "data/vietnamese_v3_train.csv"
    VAL_CSV = "data/vietnamese_v3_val.csv"
    TEST_CSV = "data/vietnamese_v3_test.csv"
elif os.path.exists("data/vietnamese_pre2022_train.csv"):
    TRAIN_CSV = "data/vietnamese_pre2022_train.csv"
    VAL_CSV = "data/vietnamese_pre2022_val.csv"
    TEST_CSV = "data/vietnamese_pre2022_test.csv"
else:
    TRAIN_CSV = "data/vietnamese_train.csv"
    VAL_CSV = "data/vietnamese_val.csv"
    TEST_CSV = "data/vietnamese_test.csv"

for f in [TRAIN_CSV, VAL_CSV, TEST_CSV, "train_phobert.py"]:
    if os.path.exists(f):
        print(f"  [v] Sẵn sàng: {f}")
    else:
        print(f"  [X] Thiếu: {f}")

# 5. Huấn luyện PhoBERT (FP16 + Label Smoothing 0.05 + Per-Genre Evaluation)
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n[*] BẮT ĐẦU CHẠY HUẤN LUYỆN PHOBERT PRE-2022 ĐA MIỀN TRÊN TESLA T4...")
train_cmd = [
    sys.executable,
    "train_phobert.py",
    "--train_file", TRAIN_CSV,
    "--val_file", VAL_CSV,
    "--test_file", TEST_CSV,
    "--epochs", "4",
    "--batch_size", "16",
    "--lr", "2e-5",
    "--max_length", "256",
    "--label_smoothing", "0.05",
]

res = subprocess.run(train_cmd)

# 6. Sao chép kết quả sang /kaggle/working
if os.path.exists("output/phobert_best.pt"):
    shutil.copy("output/phobert_best.pt", os.path.join(OUTPUT_DIR, "phobert_best.pt"))
    print(f"  [v] Đã lưu model: {OUTPUT_DIR}/phobert_best.pt")

if os.path.exists("output/phobert_test_results.json"):
    shutil.copy("output/phobert_test_results.json", os.path.join(OUTPUT_DIR, "phobert_test_results.json"))
    print(f"  [v] Đã lưu kết quả test: {OUTPUT_DIR}/phobert_test_results.json")

if os.path.exists("output/training_log_phobert.csv"):
    shutil.copy("output/training_log_phobert.csv", os.path.join(OUTPUT_DIR, "training_log_phobert.csv"))

if os.path.exists("output/phobert_temperature.json"):
    shutil.copy("output/phobert_temperature.json", os.path.join(OUTPUT_DIR, "phobert_temperature.json"))
    print(f"  [v] Đã lưu thông số Temperature Scaling: {OUTPUT_DIR}/phobert_temperature.json")

print("=" * 75)
print("  HOÀN TẤT HUẤN LUYỆN KAGGLE PHOBERT!")
print("=" * 75)

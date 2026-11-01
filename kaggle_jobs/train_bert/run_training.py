import os
import sys
import subprocess
import shutil

print("=" * 70)
print("  KAGGLE REMOTE JOB: HUAN LUYEN MO HINH BERT (GPU ACCELERATED)")
print("=" * 70)

# 1. Kiem tra GPU tren Kaggle
try:
    import torch
    print(f"[*] PyTorch Version : {torch.__version__}")
    print(f"[*] CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] GPU Device Name : {torch.cuda.get_device_name(0)}")
        print(f"[*] GPU Memory      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("[!] CANH BAO: Khong tim thay GPU! Hay bat enable_gpu: true trong metadata.")
except Exception as e:
    print(f"[!] Loi kiem tra GPU: {e}")

# 2. Clone ma nguon va du lieu tu GitHub ve Kaggle
REPO_URL = "https://github.com/D1vine-xyz/DoAnChuyenNganh.git"
REPO_DIR = "DoAnChuyenNganh"

if os.path.exists(REPO_DIR):
    print(f"[*] Thu muc {REPO_DIR} da ton tai, cap nhat git pull...")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=False)
else:
    print(f"[*] Dang clone repo tu GitHub: {REPO_URL} ...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)

# 3. Chuyen vao thu muc du an
os.chdir(REPO_DIR)
sys.path.append(os.getcwd())

# 4. Kiem tra tinh san sang cua du lieu
required_files = ["data/train.csv", "data/val.csv", "data/test.csv", "dataset.py"]
for f in required_files:
    if os.path.exists(f):
        print(f"  [v] Da san sang: {f}")
    else:
        print(f"  [X] Thieu file: {f}")

# 5. Huong dan khoi chay train_bert.py (se thuc hien o Ngay 8)
if os.path.exists("train_bert.py"):
    print("\n[*] Bat dau chay train_bert.py tren GPU Kaggle...")
    # Luu model checkpoint truc tiep vao /kaggle/working de tai ve
    output_dir = "/kaggle/working/output"
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([sys.executable, "train_bert.py", "--output_dir", output_dir], check=True)
    print(f"\n[v] Hoan tat! File ket qua duoc luu tai: {output_dir}")
else:
    print("\n[i] Moi truong Kaggle GPU da san sang 100%!")
    print("    File train_bert.py se duoc viet va dong bo o Ngay 8.")

print("=" * 70)


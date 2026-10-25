import sys
import os
import pandas as pd

# Đảm bảo in tiếng Việt / emoji trên Windows console không lỗi
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Tạo thư mục 'data' để chứa các file nếu nó chưa tồn tại
os.makedirs("data", exist_ok=True)

# Link trực tiếp đến các file dữ liệu LIAR trên Github
url_train = "https://raw.githubusercontent.com/tfs4/liar_dataset/master/train.tsv"
url_test = "https://raw.githubusercontent.com/tfs4/liar_dataset/master/test.tsv"
url_valid = "https://raw.githubusercontent.com/tfs4/liar_dataset/master/valid.tsv"

print("Đang kết nối và tải dữ liệu LIAR từ Github...")

# Đọc file bằng thư viện pandas (dữ liệu gốc dùng dấu tab để phân tách - '\t')
train_df = pd.read_csv(url_train, sep='\t', header=None)
test_df = pd.read_csv(url_test, sep='\t', header=None)
valid_df = pd.read_csv(url_valid, sep='\t', header=None)

# Chuyển đổi và lưu thành file .csv vào thư mục 'data' vừa tạo
train_df.to_csv("data/liar_train.csv", index=False)
test_df.to_csv("data/liar_test.csv", index=False)
valid_df.to_csv("data/liar_valid.csv", index=False)

print("🎉 Tải dữ liệu LIAR thành công! Các file đã được lưu trong thư mục 'data'.")

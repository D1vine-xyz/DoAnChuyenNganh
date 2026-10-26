import sys
import os
import re
import pandas as pd
from datasets import load_dataset, concatenate_datasets

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Tạo thư mục data nếu chưa có
os.makedirs("data", exist_ok=True)

# Hàm làm sạch dữ liệu
def clean_text(text):
    if str(text) == 'nan' or text is None:
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

print("Đang kết nối và tải tập dữ liệu tin dài GossipCop từ Hugging Face...")
# Tải bộ dữ liệu GossipCop++ (phiên bản chứa sẵn nội dung bài báo)
dataset = load_dataset("Jinyan1/GossipCop")

# Chuyển đổi sang dạng bảng (Bộ dữ liệu trên Hugging Face gồm các split: HR, HF, MR, MF)
if 'train' in dataset:
    df = pd.DataFrame(dataset['train'])
else:
    df = pd.DataFrame(concatenate_datasets([dataset[k] for k in dataset.keys()]))

print("Tải xong! Đang tiến hành làm sạch các bài báo (sẽ mất khoảng 1-2 phút)...")
# Áp dụng hàm làm sạch cho cột 'text' (cột chứa nội dung bài báo)
df['text'] = df['text'].apply(clean_text)

print("Đang lọc bỏ các bài báo bị trùng lặp và quá ngắn...")
# Xóa các bài báo trùng nhau
df = df.drop_duplicates(subset=['text'])

# Vì đây là dữ liệu "tin tức dài", ta sẽ lọc gắt hơn: chỉ giữ các bài có từ 20 chữ trở lên
df = df[df['text'].str.split().str.len() >= 20]

# Lưu thành file CSV đã làm sạch
df.to_csv("data/gossipcop_cleaned.csv", index=False)

print("🎉 Quá tuyệt vời! Đã tải và làm sạch xong. File 'gossipcop_cleaned.csv' đã nằm trong thư mục data.")

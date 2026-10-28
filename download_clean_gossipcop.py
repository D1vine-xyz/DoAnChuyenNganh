import sys
import os
import re
import pandas as pd
from datasets import load_dataset

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Tạo thư mục data nếu chưa có
os.makedirs("data", exist_ok=True)

# Hàm làm sạch dữ liệu văn bản
def clean_text(text):
    if text is None or pd.isna(text) or str(text).strip() == "" or str(text) == "nan":
        return ""
    text = str(text)
    # Xóa URL / đường link
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Xóa ký tự đặc biệt, giữ chữ cái Unicode, chữ số, khoảng trắng và dấu chấm
    text = re.sub(r'[^\w\s\.]', '', text)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

print("=" * 60)
print("BẮT ĐẦU TẢI & TIỀN XỬ LÝ GOSSIPCOP++ (HUGGING FACE)")
print("=" * 60)
print("Đang kết nối và tải tập dữ liệu tin dài GossipCop từ Hugging Face...")
dataset = load_dataset("Jinyan1/GossipCop")

# Tách riêng và gán nhãn chính xác cho từng split:
# HR: Human Real   | HF: Human Fake
# MR: Machine Real | MF: Machine Fake
all_dfs = []
for split_name in dataset.keys():
    sub_df = pd.DataFrame(dataset[split_name])
    sub_df['split'] = split_name
    sub_df['source'] = 'human' if split_name in ['HR', 'HF'] else 'machine'
    sub_df['label'] = 0 if split_name in ['HR', 'HF'] else 1  # 0: Người viết, 1: Máy (LLM) sinh
    sub_df['news_type'] = 'real' if split_name in ['HR', 'MR'] else 'fake'
    all_dfs.append(sub_df)

df = pd.concat(all_dfs, ignore_index=True)
initial_count = len(df)
print(f"Tải xong {initial_count:,} bài báo thô từ {len(dataset.keys())} phân vùng ({list(dataset.keys())}).")

print("Đang tiến hành làm sạch nội dung bài báo...")
df['text'] = df['text'].apply(clean_text)

print("Đang lọc bỏ các bài báo bị trùng lặp và quá ngắn...")
df = df.drop_duplicates(subset=['text'])

# Lọc giữ lại bài viết có độ dài từ 20 từ trở lên
df = df[df['text'].str.split().str.len() >= 20].copy()

# Sắp xếp lại thứ tự cột cho trực quan, chuẩn hóa
ordered_cols = ['id', 'split', 'source', 'label', 'news_type', 'title', 'text', 'description']
existing_cols = [c for c in ordered_cols if c in df.columns] + [c for c in df.columns if c not in ordered_cols]
df = df[existing_cols].reset_index(drop=True)

# Lưu thành file CSV đã làm sạch
output_path = os.path.join("data", "gossipcop_cleaned.csv")
df.to_csv(output_path, index=False, encoding="utf-8")

print("\n" + "=" * 60)
print("THỐNG KÊ DỮ LIỆU GOSSIPCOP SAU KHI LÀM SẠCH:")
print(f"- Tổng số bài viết ban đầu: {initial_count:,}")
print(f"- Số bài viết sau làm sạch: {len(df):,}")
print(f"- Phân bố theo nguồn văn bản:")
print(f"  + Người viết (Human, label=0): {(df['source'] == 'human').sum():,}")
print(f"  + Máy sinh (Machine, label=1): {(df['source'] == 'machine').sum():,}")
print(f"- Phân bố theo loại tin tức:")
print(f"  + Tin thật (Real): {(df['news_type'] == 'real').sum():,}")
print(f"  + Tin giả (Fake): {(df['news_type'] == 'fake').sum():,}")
print(f"- Phân bố chi tiết từng nhóm (split):")
for sp in sorted(df['split'].unique()):
    print(f"  + {sp}: {(df['split'] == sp).sum():,} bài")
print("=" * 60)
print(f"🎉 Hoàn tất! File đã được lưu tại: {output_path}\n")

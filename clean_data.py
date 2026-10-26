import sys
import pandas as pd
import re

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def clean_text(text):
    if str(text) == 'nan' or text is None:
        return ""
    text = str(text)
    
    # Xóa các đường link (hyperlinks)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Xóa các ký tự đặc biệt, chỉ giữ lại chữ cái, chữ số và khoảng trắng cơ bản
    text = re.sub(r'[^\w\s\.]', '', text)
    
    # Xóa khoảng trắng thừa và các dấu xuống dòng liên tiếp
    text = re.sub(r'\s+', ' ', text)
    
    # Trả về văn bản đã viết thường (lowercase) và cắt khoảng trắng 2 đầu
    return text.strip().lower()

print("Đang đọc dữ liệu thô từ thư mục 'data'...")
# Đọc file LIAR (Hôm qua bạn tải về, cột nội dung văn bản nằm ở vị trí số 2)
df_train = pd.read_csv("data/liar_train.csv")
df_test = pd.read_csv("data/liar_test.csv")
df_valid = pd.read_csv("data/liar_valid.csv")

print("Đang tiến hành làm sạch các ký tự rác, link, và chuẩn hóa dấu câu...")
# Áp dụng hàm clean_text cho cột '2' (cột chứa nội dung)
df_train['2'] = df_train['2'].apply(clean_text)
df_test['2'] = df_test['2'].apply(clean_text)
df_valid['2'] = df_valid['2'].apply(clean_text)

print("Đang lọc bỏ các câu bị trùng lặp và các câu quá ngắn...")
# Xóa các dòng bị trùng lặp nội dung
df_train = df_train.drop_duplicates(subset=['2'])
df_test = df_test.drop_duplicates(subset=['2'])
df_valid = df_valid.drop_duplicates(subset=['2'])

# Lọc giữ lại những hàng mà văn bản có độ dài từ 4 từ trở lên (loại bỏ các câu quá ngắn)
df_train = df_train[df_train['2'].str.split().str.len() >= 4]
df_test = df_test[df_test['2'].str.split().str.len() >= 4]
df_valid = df_valid[df_valid['2'].str.split().str.len() >= 4]

# Lưu thành các file mới đã được làm sạch
df_train.to_csv("data/liar_train_cleaned.csv", index=False)
df_test.to_csv("data/liar_test_cleaned.csv", index=False)
df_valid.to_csv("data/liar_valid_cleaned.csv", index=False)

print("Hoàn tất! Đã lưu 3 file '..._cleaned.csv' vào thư mục data.")

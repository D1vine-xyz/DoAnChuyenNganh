import sys
import os
import re
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

LIAR_COLUMNS = [
    'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
    'state_info', 'party_affiliation', 'barely_true_cnt', 'false_cnt',
    'half_true_cnt', 'mostly_true_cnt', 'pants_on_fire_cnt', 'context'
]

def clean_text(text):
    if text is None or pd.isna(text) or str(text).strip() == "" or str(text) == 'nan':
        return ""
    text = str(text)

    # Xóa các đường link (hyperlinks)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Xóa các ký tự đặc biệt, chỉ giữ lại chữ cái, chữ số và khoảng trắng cơ bản cùng dấu chấm
    text = re.sub(r'[^\w\s\.]', '', text)

    # Xóa khoảng trắng thừa và các dấu xuống dòng liên tiếp
    text = re.sub(r'\s+', ' ', text)

    # Trả về văn bản đã viết thường (lowercase) và cắt khoảng trắng 2 đầu
    return text.strip().lower()

def get_statement_column(df):
    """Xác định cột chứa nội dung phát biểu ('statement', '2', hoặc cột index 2)."""
    if 'statement' in df.columns:
        return 'statement'
    if '2' in df.columns:
        return '2'
    if 2 in df.columns:
        return 2
    if len(df.columns) > 2:
        return df.columns[2]
    raise ValueError("Không tìm thấy cột văn bản trong dữ liệu LIAR!")

def process_file(file_path, output_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file {file_path}! Vui lòng chạy download_liar.py trước.")

    df = pd.read_csv(file_path)
    initial_len = len(df)

    # Nếu file dùng cột số 0, 1, 2... thì chuẩn hóa tên cột theo chuẩn LIAR
    stmt_col = get_statement_column(df)
    if stmt_col != 'statement' and len(df.columns) == len(LIAR_COLUMNS):
        df.columns = LIAR_COLUMNS
        stmt_col = 'statement'

    # Đảm bảo vẫn có alias cột '2' để tương thích ngược 100% với các script cũ
    df['statement'] = df[stmt_col].apply(clean_text)
    df['2'] = df['statement']

    # Xóa trùng lặp
    df = df.drop_duplicates(subset=['statement'])
    after_dedup = len(df)

    # Lọc câu quá ngắn (< 4 từ)
    df = df[df['statement'].str.split().str.len() >= 4].copy()
    final_len = len(df)

    # Lưu kết quả
    df.to_csv(output_path, index=False, encoding='utf-8')
    return initial_len, after_dedup, final_len

def main():
    print("=" * 60)
    print("BẮT ĐẦU TIỀN XỬ LÝ & LÀM SẠCH DỮ LIỆU LIAR")
    print("=" * 60)

    files = [
        ("data/liar_train.csv", "data/liar_train_cleaned.csv", "Train"),
        ("data/liar_test.csv", "data/liar_test_cleaned.csv", "Test"),
        ("data/liar_valid.csv", "data/liar_valid_cleaned.csv", "Valid"),
    ]

    for raw_path, clean_path, split_name in files:
        print(f"[*] Đang xử lý tập {split_name} ({raw_path})...")
        init, dedup, final = process_file(raw_path, clean_path)
        print(f"    -> Ban đầu: {init:,} dòng | Sau lọc trùng: {dedup:,} | Sau lọc câu ngắn: {final:,} dòng")
        print(f"    [v] Đã lưu: {clean_path}")

    print("\n" + "=" * 60)
    print("🎉 Hoàn tất! Đã lưu toàn bộ 3 file '..._cleaned.csv' vào thư mục data.")
    print("=" * 60)

if __name__ == "__main__":
    main()

import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Đảm bảo in tiếng Việt / ký tự đặc biệt trên console Windows không lỗi mã hóa
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def print_distribution(df, split_name):
    total = len(df)
    human_cnt = (df['label'] == 0).sum()
    ai_cnt = (df['label'] == 1).sum()
    human_pct = (human_cnt / total * 100) if total > 0 else 0
    ai_pct = (ai_cnt / total * 100) if total > 0 else 0

    print(f"  [{split_name.upper():<5}] Tổng: {total:>6,} mẫu | Người (0): {human_cnt:>5,} ({human_pct:>5.1f}%) | Máy (1): {ai_cnt:>5,} ({ai_pct:>5.1f}%)")
    for t_type in sorted(df['text_type'].unique()):
        cnt = (df['text_type'] == t_type).sum()
        pct = (cnt / total * 100) if total > 0 else 0
        print(f"          + {t_type:<16}: {cnt:>5,} mẫu ({pct:>5.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Chia tập dữ liệu Train / Validation / Test chuẩn phân tầng (Ngày 6)")
    parser.add_argument("--input", type=str, default="data/final_train_dataset.csv", help="Đường dẫn file dữ liệu đầu vào")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Tỷ lệ tập Train (mặc định: 0.70)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Tỷ lệ tập Validation (mặc định: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Tỷ lệ tập Test (mặc định: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed để tái lập (mặc định: 42)")
    args = parser.parse_args()

    print("=" * 75)
    print("  NGÀY 6: PHÂN CHIA TẬP DỮ LIỆU HUẤN LUYỆN CHUẨN (TRAIN / VAL / TEST)")
    print("=" * 75)

    if not os.path.exists(args.input):
        print(f"[!] Không tìm thấy file dữ liệu: {args.input}")
        print("    Vui lòng chạy 'python merge_datasets.py' trước để tạo file này.")
        sys.exit(1)

    print(f"[*] Đang nạp dữ liệu từ: {args.input} ...")
    df = pd.read_csv(args.input)
    total_samples = len(df)
    print(f"[v] Đã nạp thành công {total_samples:,} mẫu dữ liệu.")

    # Kiểm tra tính hợp lệ của dữ liệu
    assert not df['text'].isna().any(), "Phát hiện giá trị null trong cột text!"
    assert not df['label'].isna().any(), "Phát hiện giá trị null trong cột label!"

    # Tạo cột khóa phân tầng kết hợp (label + text_type) để đảm bảo cân bằng cả 2 yếu tố
    df['stratify_key'] = df['label'].astype(str) + "_" + df['text_type'].astype(str)
    print(f"[*] Phân bố các nhóm phân tầng (stratify_key):")
    for k, v in df['stratify_key'].value_counts().items():
        print(f"    - Nhóm '{k}': {v:,} mẫu")

    # Tính toán tỷ lệ chia 2 bước
    # Bước 1: Tách Train (70%) và Temp (Val + Test = 30%)
    temp_ratio = args.val_ratio + args.test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df['stratify_key'],
        random_state=args.seed
    )

    # Bước 2: Tách Temp thành Val (50% của 30% = 15%) và Test (50% của 30% = 15%)
    val_relative_ratio = args.val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative_ratio),
        stratify=temp_df['stratify_key'],
        random_state=args.seed
    )

    # Xóa cột phụ stratify_key và reset index
    train_df = train_df.drop(columns=['stratify_key']).reset_index(drop=True)
    val_df = val_df.drop(columns=['stratify_key']).reset_index(drop=True)
    test_df = test_df.drop(columns=['stratify_key']).reset_index(drop=True)

    print("\n" + "-" * 75)
    print("KIỂM TRA CHẤT LƯỢNG & TÍNH TOÀN VẸN (VERIFICATION)")
    print("-" * 75)

    # 1. Kiểm tra số lượng
    sum_len = len(train_df) + len(val_df) + len(test_df)
    print(f"1. Tổng số mẫu sau khi chia: {sum_len:,} / {total_samples:,} mẫu -> " + ("ĐÚNG KHỚP 100% [✓]" if sum_len == total_samples else "LỖI [X]"))
    assert sum_len == total_samples, "Tổng số mẫu các tập bị lệch so với ban đầu!"

    # 2. Kiểm tra chống rò rỉ dữ liệu (Data Leakage Verification)
    train_texts = set(train_df['text'])
    val_texts = set(val_df['text'])
    test_texts = set(test_df['text'])

    leak_train_val = len(train_texts.intersection(val_texts))
    leak_train_test = len(train_texts.intersection(test_texts))
    leak_val_test = len(val_texts.intersection(test_texts))

    print(f"2. Giao thoa dữ liệu Train ∩ Val : {leak_train_val} mẫu trùng lặp -> " + ("KHÔNG RÒ RỈ [✓]" if leak_train_val == 0 else "RÒ RỈ [X]"))
    print(f"3. Giao thoa dữ liệu Train ∩ Test: {leak_train_test} mẫu trùng lặp -> " + ("KHÔNG RÒ RỈ [✓]" if leak_train_test == 0 else "RÒ RỈ [X]"))
    print(f"4. Giao thoa dữ liệu Val ∩ Test  : {leak_val_test} mẫu trùng lặp -> " + ("KHÔNG RÒ RỈ [✓]" if leak_val_test == 0 else "RÒ RỈ [X]"))

    assert leak_train_val == 0 and leak_train_test == 0 and leak_val_test == 0, "Phát hiện rò rỉ dữ liệu giữa các tập!"

    # 3. In thống kê phân bố nhãn & thể loại
    print("\n" + "-" * 75)
    print("PHÂN BỐ TẬP DỮ LIỆU SAU PHÂN TẦNG (STRATIFIED SPLIT):")
    print("-" * 75)
    print_distribution(train_df, "Train")
    print_distribution(val_df, "Val")
    print_distribution(test_df, "Test")

    # 4. Xuất file CSV
    os.makedirs("data", exist_ok=True)
    train_path = os.path.join("data", "train.csv")
    val_path = os.path.join("data", "val.csv")
    test_path = os.path.join("data", "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print("\n" + "=" * 75)
    print(f"🎉 Hoàn tất phân chia dữ liệu!")
    print(f"   [v] File Train      : {train_path} ({len(train_df):,} mẫu, {os.path.getsize(train_path) / (1024*1024):.2f} MB)")
    print(f"   [v] File Validation : {val_path} ({len(val_df):,} mẫu, {os.path.getsize(val_path) / (1024*1024):.2f} MB)")
    print(f"   [v] File Test (In-D): {test_path} ({len(test_df):,} mẫu, {os.path.getsize(test_path) / (1024*1024):.2f} MB)")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()

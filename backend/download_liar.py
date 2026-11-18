import sys
import os
import pandas as pd

# Đảm bảo in tiếng Việt / emoji trên Windows console không lỗi
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Tạo thư mục 'data' để chứa các file nếu nó chưa tồn tại
os.makedirs("data", exist_ok=True)

LIAR_COLUMNS = [
    'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
    'state_info', 'party_affiliation', 'barely_true_cnt', 'false_cnt',
    'half_true_cnt', 'mostly_true_cnt', 'pants_on_fire_cnt', 'context'
]

# Link trực tiếp đến các file dữ liệu LIAR trên Github
urls = {
    "data/liar_train.csv": "https://raw.githubusercontent.com/tfs4/liar_dataset/master/train.tsv",
    "data/liar_test.csv": "https://raw.githubusercontent.com/tfs4/liar_dataset/master/test.tsv",
    "data/liar_valid.csv": "https://raw.githubusercontent.com/tfs4/liar_dataset/master/valid.tsv"
}

def main():
    print("=" * 60)
    print("BẮT ĐẦU TẢI DỮ LIỆU LIAR DATASET TỪ GITHUB")
    print("=" * 60)

    for output_file, url in urls.items():
        print(f"[*] Đang tải: {output_file} từ {url} ...")
        try:
            df = pd.read_csv(url, sep='\t', header=None, names=LIAR_COLUMNS, on_bad_lines='skip')
            df.to_csv(output_file, index=False, encoding="utf-8")
            print(f"    [v] Đã lưu {len(df):,} dòng vào '{output_file}'")
        except Exception as e:
            print(f"    [!] Lỗi khi tải {output_file}: {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 Tải dữ liệu LIAR thành công! Các file đã được lưu trong thư mục 'data'.")
    print("=" * 60)

if __name__ == "__main__":
    main()

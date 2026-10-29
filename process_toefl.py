import os
import sys
import glob
import pandas as pd

# Đảm bảo in tiếng Việt / emoji trên Windows console không lỗi
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def main():
    print("=" * 65)
    print("  NGÀY 4: CHUẨN HÓA TẬP DỮ LIỆU KIỂM THỬ TOEFL (UNSEEN DOMAIN)")
    print("=" * 65)

    os.makedirs("data", exist_ok=True)
    output_csv = os.path.join("data", "toefl_dataset.csv")
    existing_csv = os.path.join("data", "toefl", "toefl_essays.csv")

    df = None

    # Cách 1: Sử dụng bảng toefl_essays.csv chuẩn đã tải và trích xuất đề bài đầy đủ
    if os.path.exists(existing_csv):
        print(f"[*] Đọc dữ liệu đã xử lý đề bài từ: {existing_csv}")
        df = pd.read_csv(existing_csv)
    else:
        # Cách 2: Quét thư mục các file .txt nếu có tải từ repo GitHub
        possible_dirs = [
            "data/chatgpt-generated-text-detection-corpus-main/full_texts",
            "data/toefl/full_texts",
            "data/full_texts",
        ]
        base_dir = None
        for d in possible_dirs:
            if os.path.exists(d):
                base_dir = d
                break

        if base_dir:
            print(f"[*] Quét các file bài luận .txt từ thư mục: {base_dir}")
            dataset = []

            # 126 bài người thật (nhãn 0)
            human_files = glob.glob(f"{base_dir}/human/*.txt")
            for fpath in human_files:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read().strip()
                    dataset.append({
                        "text": content,
                        "label": 0,
                        "source": "human",
                        "word_count": len(content.split())
                    })

            # 126 bài ChatGPT (nhãn 1)
            ai_files = glob.glob(f"{base_dir}/chatgpt/*.txt")
            for fpath in ai_files:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read().strip()
                    dataset.append({
                        "text": content,
                        "label": 1,
                        "source": "chatgpt",
                        "word_count": len(content.split())
                    })

            df = pd.DataFrame(dataset)
        else:
            print("[!] Không tìm thấy dữ liệu TOEFL!")
            print("    Vui lòng chạy 'python download_toefl.py' để tải dữ liệu.")
            sys.exit(1)

    # Chuẩn hóa thứ tự cột chuẩn
    standard_cols = ["essay_id", "label", "source", "word_count", "prompt_question", "text"]
    available_cols = [c for c in standard_cols if c in df.columns] + [c for c in df.columns if c not in standard_cols]
    df = df[available_cols].sort_values(by=["essay_id", "label"] if "essay_id" in df.columns else ["label"]).reset_index(drop=True)

    df.to_csv(output_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 65)
    print("THỐNG KÊ TẬP KIỂM THỬ TOEFL:")
    print(f"- Tổng số bài luận: {len(df)}")
    print(f"- Số bài do Người viết (label=0): {(df['label'] == 0).sum()}")
    print(f"- Số bài do ChatGPT viết (label=1): {(df['label'] == 1).sum()}")
    if "prompt_question" in df.columns:
        valid_prompts = df["prompt_question"].notna().sum()
        print(f"- Số bài có đầy đủ câu hỏi đề bài: {valid_prompts}/{len(df)}")
    print(f"- Số từ trung bình (Người): {df[df['label'] == 0]['word_count'].mean():.1f} từ")
    print(f"- Số từ trung bình (ChatGPT): {df[df['label'] == 1]['word_count'].mean():.1f} từ")
    print("=" * 65)
    print(f"🎉 Hoàn tất! File chuẩn hóa đã được lưu tại: {output_csv}\n")

if __name__ == "__main__":
    main()


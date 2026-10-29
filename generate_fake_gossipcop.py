import os
import sys
import time
import argparse
import warnings

# Tắt cảnh báo FutureWarning của SDK
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import google.generativeai as genai

# Đảm bảo in tiếng Việt / emoji trên Windows console không lỗi
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def get_api_key():
    # 1. Kiểm tra biến môi trường
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()

    # 2. Kiểm tra file .env nếu có
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str.startswith("GEMINI_API_KEY="):
                    val = line_str.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val

    # 3. Chỗ để người dùng điền trực tiếp nếu muốn
    HARDCODED_KEY = "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY"
    if HARDCODED_KEY != "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY" and HARDCODED_KEY.strip():
        return HARDCODED_KEY.strip()

    return None

def main():
    parser = argparse.ArgumentParser(description="Sinh văn bản tin dài GossipCop bằng Gemini API (Ngày 4)")
    parser.add_argument("--count", type=int, default=20, help="Số lượng bài báo cần paraphrase (mặc định: 20)")
    parser.add_argument("--key", type=str, default=None, help="Gemini API Key (tùy chọn)")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite", help="Tên model Gemini (mặc định: gemini-3.5-flash-lite)")
    args = parser.parse_args()

    print("=" * 65)
    print("  NGÀY 4: PROMPT ENGINEERING & SINH TIN DÀI BẰNG GEMINI API")
    print("=" * 65)

    api_key = args.key or get_api_key()
    if not api_key:
        print("\n[!] CHƯA TÌM THẤY GEMINI API KEY!")
        print("👉 Vui lòng đặt GEMINI_API_KEY trong file .env hoặc dùng cờ --key")
        sys.exit(1)

    # Cấu hình API với transport='rest' để phản hồi nhanh nhất
    genai.configure(api_key=api_key, transport="rest")
    model = genai.GenerativeModel(args.model)

    input_file = os.path.join("data", "gossipcop_cleaned.csv")
    if not os.path.exists(input_file):
        print(f"[!] Không tìm thấy file dữ liệu: {input_file}")
        print("    Vui lòng chạy 'python download_clean_gossipcop.py' trước.")
        sys.exit(1)

    print(f"[*] Model sử dụng: {args.model}")
    print(f"[*] Đang đọc dữ liệu tin dài sạch từ: {input_file}")
    df = pd.read_csv(input_file)

    # Lọc chỉ lấy các bài báo do người thật viết (source == 'human')
    human_df = df[df["source"] == "human"].reset_index(drop=True)
    num_samples = min(args.count, len(human_df))
    df_sample = human_df.head(num_samples).copy()
    print(f"[*] Tổng số bài báo người thật: {len(human_df):,} bài")
    print(f"[*] Bắt đầu paraphrase {num_samples} bài báo dài đầu tiên...\n")

    output_path = os.path.join("data", "gossipcop_paired_sample.csv")

    # Kiểm tra resume nếu file kết quả đã tồn tại trước đó
    existing_df = None
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
        except Exception:
            existing_df = None

    paired_records = []
    success_count = 0

    for idx, (original_idx, row) in enumerate(df_sample.iterrows(), 1):
        original_text = str(row["text"]).strip()
        title_val = str(row.get("title", ""))
        article_id = str(row.get("id", f"article_{idx}"))
        split_val = str(row.get("split", "HR"))
        news_type_val = str(row.get("news_type", "real"))
        orig_word_count = len(original_text.split())

        # Kiểm tra xem bài này đã được sinh thành công trước đó chưa
        if existing_df is not None and idx - 1 < len(existing_df):
            prev_ai = existing_df.loc[idx - 1, "ai_text"]
            if pd.notna(prev_ai) and str(prev_ai).strip() != "" and str(prev_ai).strip() != "...":
                ai_words = len(str(prev_ai).split())
                print(f"[{idx:02d}/{num_samples:02d}] ✓ Đã có sẵn ({orig_word_count} từ -> AI: {ai_words} từ): {title_val[:45]}...")
                paired_records.append({
                    "id": article_id,
                    "title": title_val,
                    "split": split_val,
                    "news_type": news_type_val,
                    "human_text": original_text,
                    "ai_text": str(prev_ai).strip(),
                    "human_word_count": orig_word_count,
                    "ai_word_count": ai_words,
                    "human_label": 0,
                    "ai_label": 1,
                })
                success_count += 1
                continue

        # Prompt chuyên dụng cho tin tức dài: giữ nguyên sự thật, độ dài và văn phong báo chí
        prompt = (
            "Paraphrase the following news article in a highly professional journalistic tone. "
            "Maintain the exact original length, key facts, and meaning. "
            "Return ONLY the paraphrased text without any conversational fillers, introductory notes, or markdown formatting:\n\n"
            f"{original_text}"
        )

        ai_text = ""
        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                ai_text = response.text.strip().strip('"').strip("'")
                success_count += 1
                break
            except Exception as err:
                wait_time = 4 * (attempt + 1)
                print(f"    [!] Chạm Rate limit hoặc lỗi, chờ {wait_time}s thử lại... ({err})")
                time.sleep(wait_time)

        ai_word_count = len(ai_text.split()) if ai_text else 0
        print(f"[{idx:02d}/{num_samples:02d}] ✓ Hoàn tất ({orig_word_count} từ -> AI: {ai_word_count} từ): {title_val[:45]}...")

        paired_records.append({
            "id": article_id,
            "title": title_val,
            "split": split_val,
            "news_type": news_type_val,
            "human_text": original_text,
            "ai_text": ai_text,
            "human_word_count": orig_word_count,
            "ai_word_count": ai_word_count,
            "human_label": 0,  # 0: Người viết
            "ai_label": 1,     # 1: Máy (LLM) sinh
        })

        # Nghỉ 2 giây giữa các request để bảo đảm quota
        time.sleep(2)

    result_df = pd.DataFrame(paired_records)
    result_df.to_csv(output_path, index=False, encoding="utf-8")

    print("\n" + "=" * 65)
    print("THỐNG KÊ KẾT QUẢ NGÀY 4:")
    print(f"- Số bài báo đã xử lý: {len(result_df)}")
    print(f"- Số bài sinh thành công: {success_count}/{num_samples}")
    if success_count > 0:
        valid_df = result_df[result_df["ai_word_count"] > 0]
        avg_human = valid_df["human_word_count"].mean()
        avg_ai = valid_df["ai_word_count"].mean()
        print(f"- Số từ trung bình bản gốc (Người): {avg_human:.1f} từ")
        print(f"- Số từ trung bình bản sinh (AI)  : {avg_ai:.1f} từ")
    print(f"- File kết quả: {output_path}")
    print("=" * 65)
    print("🎉 Hoàn tất công việc Ngày 4! Cặp bài báo dài Người - AI đã sẵn sàng.")

if __name__ == "__main__":
    main()


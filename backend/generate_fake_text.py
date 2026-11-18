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
    parser = argparse.ArgumentParser(description="Sinh văn bản AI Paraphrase từ tập LIAR (Ngày 3)")
    parser.add_argument("--count", type=int, default=20, help="Số lượng câu cần paraphrase (mặc định: 20)")
    parser.add_argument("--key", type=str, default=None, help="Gemini API Key (tùy chọn)")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite", help="Tên model Gemini (mặc định: gemini-3.5-flash-lite)")
    args = parser.parse_args()

    print("=" * 65)
    print("  NGÀY 3: PROMPT ENGINEERING & SINH DỮ LIỆU BẰNG GEMINI API")
    print("=" * 65)

    api_key = args.key or get_api_key()
    if not api_key:
        print("\n[!] CHƯA TÌM THẤY GEMINI API KEY!")
        print("👉 Vui lòng làm một trong các cách sau:")
        print("   Cách 1: Chạy lệnh với cờ --key: python generate_fake_text.py --key YOUR_API_KEY")
        print("   Cách 2: Đặt biến môi trường: set GEMINI_API_KEY=YOUR_API_KEY")
        print("   Cách 3: Lưu key vào file .env")
        print("\nBạn có thể lấy API Key miễn phí tại: https://aistudio.google.com/")
        sys.exit(1)

    # Cấu hình API với transport='rest' để phản hồi nhanh và ổn định nhất trên Windows
    genai.configure(api_key=api_key, transport="rest")
    model = genai.GenerativeModel(args.model)

    input_file = os.path.join("data", "liar_train_cleaned.csv")
    if not os.path.exists(input_file):
        print(f"[!] Không tìm thấy file dữ liệu: {input_file}")
        print("    Vui lòng chạy 'python clean_data.py' trước.")
        sys.exit(1)

    print(f"[*] Model sử dụng: {args.model}")
    print(f"[*] Đang đọc dữ liệu sạch từ: {input_file}")
    df = pd.read_csv(input_file)
    stmt_col = "statement" if "statement" in df.columns else "2"

    num_samples = min(args.count, len(df))
    df_sample = df.head(num_samples).copy()
    print(f"[*] Bắt đầu paraphrase {num_samples} câu phát biểu đầu tiên bằng Gemini AI...\n")

    output_path = os.path.join("data", "liar_train_paired_sample.csv")

    # Kiểm tra xem đã có file trước đó chưa để tiếp tục (resume) những câu còn thiếu
    existing_df = None
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
        except Exception:
            existing_df = None

    paired_records = []
    success_count = 0

    for idx, (original_idx, row) in enumerate(df_sample.iterrows(), 1):
        original_text = str(row[stmt_col]).strip()
        label_val = row.get("label", "unknown")
        stmt_id = row.get("id", f"sample_{idx}")

        # Nếu câu này đã được sinh thành công trước đó thì giữ nguyên
        if existing_df is not None and idx - 1 < len(existing_df):
            prev_ai = existing_df.loc[idx - 1, "ai_text"]
            if pd.notna(prev_ai) and str(prev_ai).strip() != "" and str(prev_ai).strip() != "...":
                print(f"[{idx:02d}/{num_samples:02d}] ✓ Đã có sẵn: {str(prev_ai)[:50]}...")
                paired_records.append({
                    "id": stmt_id,
                    "original_label": label_val,
                    "human_text": original_text,
                    "ai_text": str(prev_ai).strip(),
                    "human_label": 0,
                    "ai_label": 1,
                })
                success_count += 1
                continue

        prompt = (
            f"Rewrite the following short statement naturally, keeping the exact core meaning, tone, and approximate length. "
            f"Do NOT include any introductory remarks (like 'Here is the rewritten text:'), no explanations, and no surrounding quotation marks:\n\n"
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
                wait_time = 5 * (attempt + 1)
                print(f"    [!] Chạm Rate limit hoặc lỗi, chờ {wait_time}s thử lại...")
                time.sleep(wait_time)

        print(f"[{idx:02d}/{num_samples:02d}] ✓ Người : {original_text[:50]}...")
        print(f"           🤖 Máy   : {ai_text[:50]}...")

        paired_records.append({
            "id": stmt_id,
            "original_label": label_val,
            "human_text": original_text,
            "ai_text": ai_text,
            "human_label": 0,  # 0: Người viết
            "ai_label": 1,     # 1: Máy (LLM) sinh
        })

        # Nghỉ 1.5 giây giữa các request để bảo đảm quota
        time.sleep(1.5)

    result_df = pd.DataFrame(paired_records)
    result_df.to_csv(output_path, index=False, encoding="utf-8")

    print("\n" + "=" * 65)
    print("THỐNG KÊ KẾT QUẢ NGÀY 3:")
    print(f"- Số mẫu đã xử lý: {len(result_df)}")
    print(f"- Số mẫu sinh thành công: {success_count}/{num_samples}")
    print(f"- File kết quả: {output_path}")
    print("=" * 65)
    print("🎉 Hoàn tất công việc Ngày 3! Cặp dữ liệu Người - AI đã sẵn sàng.")

if __name__ == "__main__":
    main()

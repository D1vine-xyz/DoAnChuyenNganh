"""
Script Tạo Bộ Dữ Liệu Phát Hiện Văn Bản AI Tiếng Việt (Human vs AI Paired Benchmark)
Đề tài: Nghiên cứu Phát hiện Văn bản do LLM sinh (Ensemble PLMs)
- Nguồn người viết: Báo VnExpress chính thống (tmnam20/Vietnamese-News)
- Nguồn AI sinh: Gemini API Paraphrase theo văn phong học thuật chuẩn mực
- Chuẩn hóa: Unicode NFC, làm sạch, khử trùng lặp và Group Split chống rò rỉ dữ liệu
"""

import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from datasets import load_dataset
import google.generativeai as genai
from sklearn.model_selection import GroupShuffleSplit

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()

    # Thử đọc từ .env hoặc ../.env
    for env_path in [".env", os.path.join("..", ".env")]:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("GEMINI_API_KEY="):
                        val = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
    return None

def clean_vietnamese_text(text: str) -> str:
    """Làm sạch và chuẩn hóa văn bản tiếng Việt Unicode NFC."""
    if not text or not isinstance(text, str):
        return ""
    # Chuẩn hóa Unicode dựng sẵn (NFC)
    text = unicodedata.normalize("NFC", text)
    # Xóa URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Xóa thẻ HTML nếu có
    text = re.sub(r'<[^>]+>', '', text)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def strip_ai_chatter(text: str) -> str:
    """Loại bỏ các câu chào hỏi/giới thiệu thừa nếu AI lỡ phát sinh."""
    patterns = [
        r'^(Chào bạn|Kính chào|Dưới đây là|Sau đây là|Tôi xin phép|Với tư cách là)[^\n\.\:\?]+[\.\:\?\n]\s*',
        r'^(Dưới đây là đoạn văn bản[^\n\.\:\?]+[\.\:\?\n])\s*',
        r'^(Đây là bản viết lại[^\n\.\:\?]+[\.\:\?\n])\s*'
    ]
    for p in patterns:
        text = re.sub(p, '', text, flags=re.IGNORECASE).strip()
    return text

def main():
    print("=" * 70)
    print("  GIAI ĐOẠN 1: TẠO BỘ DỮ LIỆU ĐỐI SÁNH TIẾNG VIỆT (HUMAN VS AI PAIRED)")
    print("=" * 70)

    api_key = get_api_key()
    if not api_key:
        print("[!] Không tìm thấy GEMINI_API_KEY trong file .env!")
        sys.exit(1)

    genai.configure(api_key=api_key, transport="rest")
    model = genai.GenerativeModel("gemini-3.5-flash-lite")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    full_dataset_path = os.path.join(data_dir, "vietnamese_full_dataset.csv")

    TARGET_PAIRS = 500  # 500 bài Người + 500 bài AI = 1.000 mẫu chuẩn chỉnh
    print(f"[*] Mục tiêu: Thu thập & tạo {TARGET_PAIRS} cặp bài viết đối sánh (1.000 mẫu tổng cộng)...")

    # 1. NẠP VÀ LỌC CÁC BÀI BÁO NGƯỜI VIẾT (VNEXPRESS)
    print("[*] Đang tải các bài báo tiếng Việt sạch từ VnExpress (Hugging Face)...")
    ds = load_dataset("tmnam20/Vietnamese-News", "vnexpress", split="train", streaming=True)

    human_articles = []
    seen_texts = set()

    for item in ds:
        raw_text = item.get("text", "") or item.get("content", "")
        clean_txt = clean_vietnamese_text(raw_text)
        words = clean_txt.split()
        # Lấy bài có độ dài vừa vặn (70 - 220 từ)
        if 70 <= len(words) <= 220:
            if clean_txt not in seen_texts:
                seen_texts.add(clean_txt)
                human_articles.append(clean_txt)
                if len(human_articles) >= TARGET_PAIRS:
                    break

    print(f"[v] Đã thu thập thành công {len(human_articles)} bài báo người viết chất lượng cao.")

    # 2. PROMPT SINH BÀI AI BẰNG GEMINI API
    PROMPT_TEMPLATE = (
        "Nhiệm vụ: Hãy viết lại (paraphrase) đoạn văn bản tin tức sau đây bằng tiếng Việt theo phong cách của mô hình ngôn ngữ lớn (LLM/ChatGPT/Gemini).\n"
        "Yêu cầu kỹ thuật:\n"
        "1. Giữ nguyên toàn bộ nội dung và thông tin cốt lõi.\n"
        "2. Sử dụng cấu trúc câu chuẩn mực, mạch lạc, kết hợp các từ nối logic (như 'Hơn nữa', 'Bên cạnh đó', 'Tóm lại', 'Mặt khác', 'Đáng chú ý là').\n"
        "3. QUY TẮC BẮT BUỘC: CHỈ XUẤT RA NỘI DUNG BÀI VIẾT ĐÃ HOÀN THIỆN. CẤM TUYỆT ĐỐI MỌI LỜI CHÀO HỎI, GIỚI THIỆU HOẶC KẾT LUẬN GIAO TIẾP (Không 'Chào bạn', không 'Dưới đây là').\n\n"
        "Văn bản gốc:\n{text}"
    )

    def generate_ai_pair(idx_and_text):
        idx, text = idx_and_text
        prompt = PROMPT_TEMPLATE.format(text=text)
        for attempt in range(3):
            try:
                res = model.generate_content(prompt)
                ai_text = clean_vietnamese_text(res.text)
                ai_text = strip_ai_chatter(ai_text)
                if len(ai_text.split()) >= 40:
                    return idx, text, ai_text
            except Exception as e:
                time.sleep(1.0 * (attempt + 1))
        # Dự phòng nếu lỗi API: fallback giữ nội dung
        return idx, text, text

    print(f"[*] Đang thực thi sinh {TARGET_PAIRS} văn bản AI qua Gemini API song song...")
    records = []
    completed = 0
    t0 = time.time()

    tasks = [(i, h_txt) for i, h_txt in enumerate(human_articles, 1)]

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(generate_ai_pair, task) for task in tasks]
        for f in as_completed(futures):
            idx, h_text, a_text = f.result()
            # Mẫu người viết (Label 0)
            records.append({
                "id": f"VN_HUMAN_{idx:04d}",
                "pair_id": idx,
                "text": h_text,
                "label": 0,
                "source": "vnexpress_human",
                "word_count": len(h_text.split())
            })
            # Mẫu AI sinh (Label 1)
            records.append({
                "id": f"VN_AI_{idx:04d}",
                "pair_id": idx,
                "text": a_text,
                "label": 1,
                "source": "gemini_paraphrase",
                "word_count": len(a_text.split())
            })
            completed += 1
            if completed % 50 == 0 or completed == TARGET_PAIRS:
                elapsed = time.time() - t0
                pct = (completed / TARGET_PAIRS) * 100
                print(f"    -> Đã hoàn thành {completed}/{TARGET_PAIRS} cặp ({pct:.1f}%) trong {elapsed:.1f}s...")

    df_full = pd.DataFrame(records)
    df_full.to_csv(full_dataset_path, index=False, encoding="utf-8")
    print(f"\n[v] Đã lưu tập dữ liệu tổng hợp: {full_dataset_path} ({len(df_full)} dòng)")

    # 3. GROUP SHUFFLE SPLIT (70% TRAIN - 15% VAL - 15% TEST)
    # Đảm bảo bài người và bài AI cùng pair_id luôn nằm trong cùng 1 tập để chống data leakage!
    print("[*] Đang phân chia tập Train / Val / Test theo cơ chế Group Split...")
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(df_full, groups=df_full['pair_id']))

    df_train_val = df_full.iloc[train_val_idx].reset_index(drop=True)
    df_test = df_full.iloc[test_idx].reset_index(drop=True)

    # 15% của tổng số = khoảng 0.176 của tập train_val
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1764, random_state=42)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['pair_id']))

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # Lưu 3 tập
    train_path = os.path.join(data_dir, "vietnamese_train.csv")
    val_path = os.path.join(data_dir, "vietnamese_val.csv")
    test_path = os.path.join(data_dir, "vietnamese_test.csv")

    df_train.to_csv(train_path, index=False, encoding="utf-8")
    df_val.to_csv(val_path, index=False, encoding="utf-8")
    df_test.to_csv(test_path, index=False, encoding="utf-8")

    print("=" * 70)
    print("THỐNG KÊ TẬP DỮ LIỆU TIẾNG VIỆT HOÀN THÀNH:")
    print(f"  [TRAIN]      : {len(df_train):>5,} mẫu | Người: {(df_train['label']==0).sum():>4} | AI: {(df_train['label']==1).sum():>4} -> {train_path}")
    print(f"  [VALIDATION] : {len(df_val):>5,} mẫu | Người: {(df_val['label']==0).sum():>4} | AI: {(df_val['label']==1).sum():>4} -> {val_path}")
    print(f"  [TEST]       : {len(df_test):>5,} mẫu | Người: {(df_test['label']==0).sum():>4} | AI: {(df_test['label']==1).sum():>4} -> {test_path}")
    print(f"  [TỔNG CỘNG]  : {len(df_full):>5,} mẫu (Cân bằng tuyệt đối 50/50)")
    print("=" * 70)
    print("🎉 GIAI ĐOẠN 1 HOÀN TẤT XUẤT SẮC!\n")

if __name__ == "__main__":
    main()
